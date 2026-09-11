import math

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported


CARTRIDGE_VARIANT_DECL = r"""
#include <flashinfer/attention/hopper/variants.cuh>
using CartridgeAttention = flashinfer::StandardAttention;
"""


def _jit_args(live_v_dtype: torch.dtype, decode: bool):
    suffix = "bf16" if live_v_dtype == torch.bfloat16 else "fp8e4m3"
    phase = "decode" if decode else "prefill"
    scalar_names = [
        "cartridge_num_tokens",
        "cartridge_k_page_stride",
        "cartridge_v_page_stride",
        "sm_scale",
    ]
    scalar_dtypes = ["int64_t", "int64_t", "int64_t", "double"]
    if decode:
        scalar_names.append("cartridge_decode_marker")
        scalar_dtypes.append("int64_t")
    args = (
        f"batch_prefill_cartridge_mixed_paged_{phase}_{suffix}",
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        ["cartridge_k_ptr", "cartridge_v_ptr"],
        ["DTypeK", "__nv_fp8_e4m3"],
        scalar_names,
        scalar_dtypes,
        "CartridgeAttention",
        CARTRIDGE_VARIANT_DECL,
    )
    kwargs = {"dtype_k": torch.bfloat16, "dtype_v": live_v_dtype}
    return args, kwargs


def _reference(
    q: torch.Tensor,
    qo_indptr: torch.Tensor,
    live_k: torch.Tensor,
    live_v: torch.Tensor,
    live_indices: list[list[int]],
    live_lens: list[int],
    cartridge_k: torch.Tensor,
    cartridge_v: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    num_qo_heads = q.shape[1]
    num_kv_heads = live_k.shape[2]
    group_size = num_qo_heads // num_kv_heads
    for batch_idx, (indices, live_len) in enumerate(
        zip(live_indices, live_lens, strict=True)
    ):
        q_begin = int(qo_indptr[batch_idx])
        q_end = int(qo_indptr[batch_idx + 1])
        q_i = q[q_begin:q_end].float()
        k_live = torch.cat([live_k[p] for p in indices], dim=0)[:live_len].float()
        v_live = torch.cat([live_v[p] for p in indices], dim=0)[:live_len].float()
        k = torch.cat([cartridge_k.flatten(0, 1), k_live], dim=0).float()
        v = torch.cat([cartridge_v.flatten(0, 1).float(), v_live], dim=0)
        k = k.repeat_interleave(group_size, dim=1)
        v = v.repeat_interleave(group_size, dim=1)
        logits = torch.einsum("qhd,khd->hqk", q_i, k) / math.sqrt(q.shape[-1])
        qo_len = q_end - q_begin
        kv_len = k.shape[0]
        q_pos = torch.arange(kv_len - qo_len, kv_len, device=q.device)
        k_pos = torch.arange(kv_len, device=q.device)
        logits.masked_fill_(k_pos[None, None, :] > q_pos[None, :, None], -torch.inf)
        probs = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hqk,khd->qhd", probs, v))
    return torch.cat(outputs, dim=0).to(torch.bfloat16)


@pytest.mark.parametrize("live_v_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("decode", [False, True])
def test_cartridge_mixed_paged_fa3(live_v_dtype: torch.dtype, decode: bool):
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("SM90A is required")

    torch.manual_seed(42)
    device = torch.device("cuda")
    page_size = 8
    cartridge_num_tokens = 632
    cartridge_num_pages = cartridge_num_tokens // page_size
    num_qo_heads = 32
    num_kv_heads = 8
    head_dim = 128
    qo_lens = [1, 1] if decode else [17, 9]
    live_lens = [33, 25]
    live_pages_per_req = [math.ceil(x / page_size) for x in live_lens]
    num_live_pages = sum(live_pages_per_req)

    q = torch.randn(
        sum(qo_lens), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    cartridge_k = torch.randn(
        cartridge_num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    cartridge_v = torch.randn_like(cartridge_k).to(torch.float8_e4m3fn)
    live_k = torch.randn(
        num_live_pages,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    live_v = torch.randn_like(live_k).to(live_v_dtype)
    if live_v_dtype == torch.bfloat16:
        packed_live = torch.stack((live_k, live_v), dim=1)
        live_k = packed_live[:, 0]
        live_v = packed_live[:, 1]
        live_cache = packed_live
    else:
        live_cache = (live_k, live_v)

    live_indices: list[list[int]] = []
    next_page = 0
    combined_indices = []
    combined_indptr = [0]
    for num_pages in live_pages_per_req:
        request_pages = list(range(next_page, next_page + num_pages))
        live_indices.append(request_pages)
        next_page += num_pages
        combined_indices.extend(range(cartridge_num_pages))
        combined_indices.extend(request_pages)
        combined_indptr.append(len(combined_indices))

    qo_indptr = torch.tensor([0, qo_lens[0], sum(qo_lens)], dtype=torch.int32)
    kv_indptr = torch.tensor(combined_indptr, dtype=torch.int32)
    kv_indices = torch.tensor(combined_indices, dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [
            x - (n - 1) * page_size
            for x, n in zip(live_lens, live_pages_per_req, strict=True)
        ],
        dtype=torch.int32,
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    jit_args, jit_kwargs = _jit_args(live_v_dtype, decode)

    if decode:
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_tensor_cores=True,
            backend="fa3",
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )
        wrapper.plan(
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.bfloat16,
            k_data_type=torch.bfloat16,
            v_data_type=live_v_dtype,
            disable_split_kv=True,
        )
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend="fa3",
            jit_args=jit_args,
            jit_kwargs=jit_kwargs,
        )
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            k_data_type=torch.bfloat16,
            v_data_type=live_v_dtype,
            disable_split_kv=True,
        )

    cartridge_args = [
        cartridge_k,
        cartridge_v,
        cartridge_num_tokens,
        cartridge_k.stride(0),
        cartridge_v.stride(0),
        1.0 / math.sqrt(head_dim),
    ]
    if decode:
        cartridge_args.append(1)
    output = wrapper.run(
        q,
        live_cache,
        *cartridge_args,
    )
    expected = _reference(
        q,
        qo_indptr,
        live_k,
        live_v,
        live_indices,
        live_lens,
        cartridge_k,
        cartridge_v,
    )
    torch.testing.assert_close(output, expected, rtol=3e-2, atol=3e-2)
