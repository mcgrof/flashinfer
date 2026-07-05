"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for asymmetric K/V cache dtypes in batch prefill: the K cache stays
at native precision (fp16/bf16) while the V cache is stored in FP8. Keys
feed the softmax score path, so K precision dominates output quality;
values sit behind the associative reduction, so FP8 V costs only FP8-level
noise. These mirror the batch-decode asymmetric tests but exercise the
FA2 prefill kernel (multi-token queries + causal masking, paged and
ragged) against a float32 torch reference that reads K natively and
dequantizes V.
"""

import itertools

import pytest
import torch

import flashinfer
from flashinfer.jit.attention.modules import (
    gen_batch_prefill_module,
    get_batch_prefill_uri,
)
from flashinfer.utils import get_compute_capability, has_flashinfer_jit_cache

# FP8 e4m3/e5m2 V requires sm89+ (Ada / Hopper). On sm80 (A100) the FP8
# numeric path is unavailable, so the whole asymmetric-V module is skipped.
if torch.cuda.is_available():
    _CAP = get_compute_capability(torch.device("cuda:0"))
    pytestmark = pytest.mark.skipif(
        _CAP < (8, 9),
        reason="asymmetric K16/V8 (FP8 V) prefill requires sm89+ (FP8 e4m3/e5m2)",
    )
    _IS_SM90 = _CAP >= (9, 0)
else:
    _IS_SM90 = False

K_DTYPES = [torch.float16, torch.bfloat16]
V_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
HEAD_DIMS = [128, 256]

# fp8 quantization of V bounds the relative error of the attention output;
# empirically the median sits near 0.02 (e4m3) / 0.04 (e5m2) for random
# inputs. 0.10 gives headroom without letting a K-side bug (whose error
# would be O(1)) pass.
MEDIAN_REL_ERR_BOUND = 0.10


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    specs = []
    for k_dtype, v_dtype, head_dim in itertools.product(K_DTYPES, V_DTYPES, HEAD_DIMS):
        specs.append(
            gen_batch_prefill_module(
                "fa2",
                k_dtype,  # dtype_q (q matches k precision)
                k_dtype,  # dtype_kv anchor (dtype_k defaults to this)
                k_dtype,  # dtype_o
                torch.int32,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                0,  # pos_encoding_mode
                False,  # use_sliding_window
                False,  # use_logits_soft_cap
                False,  # use_fp16_qk_reduction
                dtype_k=k_dtype,
                dtype_v=v_dtype,
            )
        )
    flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


def _build_paged_cache(
    batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, dtype
):
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    data_fp32 = torch.randn(*shape, dtype=torch.float32, device="cuda:0") * 0.5
    data = data_fp32.to(dtype)
    return data, num_pages_per_seq, total_num_pages


def _gather_seq(cache_fp32, indptr, last_page_len, i, kv_layout):
    """Gather sequence i's entries from a paged cache into [L, H, D]."""
    if kv_layout == "HND":
        full = cache_fp32[indptr[i] : indptr[i + 1] - 1].permute(0, 2, 1, 3)
        last = cache_fp32[indptr[i + 1] - 1, :, : last_page_len[i]].permute(1, 0, 2)
    else:
        full = cache_fp32[indptr[i] : indptr[i + 1] - 1]
        last = cache_fp32[indptr[i + 1] - 1, : last_page_len[i]]
    num_kv_heads, head_dim = full.shape[-2], full.shape[-1]
    return torch.cat([full.reshape(-1, num_kv_heads, head_dim), last], dim=0)


def _ref_prefill(q, k, v, sm_scale, causal):
    """float32 reference prefill. q: [Lq, H_qo, D]; k, v: [Lkv, H_kv, D].
    Causal masking uses FlashInfer's bottom-right alignment: query row r
    (0-indexed within the Lq block) attends to kv [0, Lkv - Lq + r]."""
    qo_len, num_qo_heads = q.shape[0], q.shape[1]
    kv_len, num_kv_heads = k.shape[0], k.shape[1]
    group_size = num_qo_heads // num_kv_heads
    k = k.repeat_interleave(group_size, dim=1)  # [Lkv, H_qo, D]
    v = v.repeat_interleave(group_size, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * sm_scale
    if causal:
        row = torch.arange(qo_len, device=q.device).view(qo_len, 1)
        col = torch.arange(kv_len, device=q.device).view(1, kv_len)
        keep = col <= (kv_len - qo_len + row)  # [Lq, Lkv]
        logits = logits.masked_fill(~keep.view(1, qo_len, kv_len), float("-inf"))
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hqk,khd->qhd", p, v.float())  # [Lq, H_qo, D]


@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("k_dtype", K_DTYPES)
@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_batch_prefill_paged_asymmetric_kv(
    num_qo_heads,
    head_dim,
    kv_layout,
    causal,
    k_dtype,
    v_dtype,
):
    # Correctness-critical axes are crossed above; geometry is fixed here
    # (shape robustness lives in test_batch_prefill_paged_asymmetric_shapes).
    batch_size, qo_len, kv_len, page_size, num_kv_heads = 3, 32, 129, 16, 4
    torch.manual_seed(42)
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype
    )
    k_cache, num_pages_per_seq, total_num_pages = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, k_dtype
    )
    v_cache, _, _ = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, v_dtype
    )

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=causal,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    o = wrapper.run(q, (k_cache, v_cache))

    # Reference: K read at its stored precision, V dequantized from fp8.
    sm_scale = 1.0 / (head_dim**0.5)
    k_stored_fp32 = k_cache.float()
    v_dequant_fp32 = v_cache.float()
    rel_errs = []
    for i in range(batch_size):
        ki = _gather_seq(k_stored_fp32, kv_indptr, kv_last_page_len, i, kv_layout)
        vi = _gather_seq(v_dequant_fp32, kv_indptr, kv_last_page_len, i, kv_layout)
        qi = q[i * qo_len : (i + 1) * qo_len]
        o_ref = _ref_prefill(qi, ki, vi, sm_scale, causal)
        oi = o[i * qo_len : (i + 1) * qo_len]
        rel = (oi.float() - o_ref).norm() / o_ref.norm().clamp_min(1e-6)
        rel_errs.append(rel.item())
    median_rel_err = sorted(rel_errs)[len(rel_errs) // 2]
    assert median_rel_err < MEDIAN_REL_ERR_BOUND, (
        f"median rel err {median_rel_err} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(k={k_dtype}, v={v_dtype}, hd={head_dim}, layout={kv_layout}, causal={causal})"
    )


@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("kv_len", [54, 129])
@pytest.mark.parametrize("qo_len", [1, 17, 32])
@pytest.mark.parametrize("batch_size", [1, 7])
def test_batch_prefill_paged_asymmetric_shapes(page_size, kv_len, qo_len, batch_size):
    """Shape robustness (page boundaries, unaligned kv_len, qo_len==1, batch)
    at fixed bf16-K + fp8-e4m3-V, NHD, causal, GQA=8, head_dim=128."""
    if qo_len > kv_len:
        pytest.skip("causal with qo_len > kv_len is degenerate")
    num_kv_heads, num_qo_heads, head_dim = 4, 32, 128
    kv_layout, causal = "NHD", True
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn
    torch.manual_seed(11)
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype
    )
    k_cache, num_pages_per_seq, total_num_pages = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, k_dtype
    )
    v_cache, _, _ = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, v_dtype
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=causal,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    o = wrapper.run(q, (k_cache, v_cache))
    sm_scale = 1.0 / (head_dim**0.5)
    k_stored_fp32, v_dequant_fp32 = k_cache.float(), v_cache.float()
    rel_errs = []
    for i in range(batch_size):
        ki = _gather_seq(k_stored_fp32, kv_indptr, kv_last_page_len, i, kv_layout)
        vi = _gather_seq(v_dequant_fp32, kv_indptr, kv_last_page_len, i, kv_layout)
        qi = q[i * qo_len : (i + 1) * qo_len]
        o_ref = _ref_prefill(qi, ki, vi, sm_scale, causal)
        oi = o[i * qo_len : (i + 1) * qo_len]
        rel_errs.append(
            ((oi.float() - o_ref).norm() / o_ref.norm().clamp_min(1e-6)).item()
        )
    median_rel_err = sorted(rel_errs)[len(rel_errs) // 2]
    assert median_rel_err < MEDIAN_REL_ERR_BOUND, (
        f"shape median rel err {median_rel_err} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(bs={batch_size}, qo={qo_len}, kv={kv_len}, page={page_size})"
    )


@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("k_dtype", K_DTYPES)
@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_batch_prefill_ragged_asymmetric_kv(
    num_qo_heads,
    head_dim,
    causal,
    k_dtype,
    v_dtype,
):
    batch_size, qo_len, kv_len, num_kv_heads = 3, 32, 129, 4
    torch.manual_seed(7)
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype
    )
    k = (
        torch.randn(
            batch_size * kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=k_dtype
        )
        * 0.5
    )
    v_fp32 = (
        torch.randn(
            batch_size * kv_len,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float32,
        )
        * 0.5
    )
    v = v_fp32.to(v_dtype)

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=causal,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    o = wrapper.run(q, k, v)

    sm_scale = 1.0 / (head_dim**0.5)
    rel_errs = []
    for i in range(batch_size):
        qi = q[i * qo_len : (i + 1) * qo_len]
        ki = k[i * kv_len : (i + 1) * kv_len].float()
        vi = v[i * kv_len : (i + 1) * kv_len].float()
        o_ref = _ref_prefill(qi, ki, vi, sm_scale, causal)
        oi = o[i * qo_len : (i + 1) * qo_len]
        rel = (oi.float() - o_ref).norm() / o_ref.norm().clamp_min(1e-6)
        rel_errs.append(rel.item())
    median_rel_err = sorted(rel_errs)[len(rel_errs) // 2]
    assert median_rel_err < MEDIAN_REL_ERR_BOUND, (
        f"ragged median rel err {median_rel_err} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(k={k_dtype}, v={v_dtype}, hd={head_dim}, causal={causal})"
    )


def test_asymmetric_one_hot_v_selection_and_lse():
    """Structured oracle: K concentrates each query's attention on one known
    token and V carries a distinctive fp8-representable constant there, so the
    output must reproduce the selected V row. The LSE and output are also
    cross-checked against the symmetric kernel run on the dequantized V (a
    convention-free oracle: LSE depends only on Q and K, and K is bit-exact)."""
    batch, page_size, pages_per_seq = 2, 16, 4
    num_kv = num_qo = 4
    head_dim = 128
    qo_len = 8
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn

    # Every query token gets an all-ones vector; the "selected" kv token has
    # a large K so it dominates the (non-causal) softmax for every query.
    q = torch.ones(batch * qo_len, num_qo, head_dim, device="cuda:0", dtype=k_dtype)
    k = torch.zeros(
        batch * pages_per_seq,
        page_size,
        num_kv,
        head_dim,
        device="cuda:0",
        dtype=k_dtype,
    )
    v = torch.zeros(
        batch * pages_per_seq,
        page_size,
        num_kv,
        head_dim,
        device="cuda:0",
        dtype=torch.float32,
    )
    expected = torch.zeros(batch * qo_len, num_qo, head_dim, device="cuda:0")
    targets = [3, 40]  # one selected token per sequence
    for i, t in enumerate(targets):
        page, slot = divmod(t, page_size)
        k[i * pages_per_seq + page, slot, :, :] = 30.0
        val = 0.5 + 0.25 * i  # exactly representable in fp8 e4m3
        v[i * pages_per_seq + page, slot, :, :] = val
        expected[i * qo_len : (i + 1) * qo_len, :, :] = val
    v_fp8 = v.to(v_dtype)

    qo_indptr = torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * qo_len
    indptr = (
        torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * pages_per_seq
    )
    indices = torch.arange(0, batch * pages_per_seq, device="cuda:0", dtype=torch.int32)
    last = torch.full((batch,), page_size, dtype=torch.int32, device="cuda:0")

    def _plan(w, **extra):
        w.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last,
            num_qo_heads=num_qo,
            num_kv_heads=num_kv,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=False,
            q_data_type=k_dtype,
            **extra,
        )

    w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0"), "NHD"
    )
    _plan(w, k_data_type=k_dtype, v_data_type=v_dtype)
    out, lse = w.run(q, (k, v_fp8), return_lse=True)
    assert (out.float() - expected).abs().max().item() < 1e-2
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()

    w_sym = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0"), "NHD"
    )
    _plan(w_sym, kv_data_type=k_dtype)
    out_sym, lse_sym = w_sym.run(q, (k, v_fp8.to(k_dtype)), return_lse=True)
    # LSE is a pure function of Q and K (K bit-exact) -> must match closely.
    assert (lse.float() - lse_sym.float()).abs().max().item() < 1e-2
    assert (out.float() - out_sym.float()).abs().max().item() < 1e-2


def test_asymmetric_scale_linearity():
    """Output is linear in V: scaling the stored V (and its dequant) by c
    scales the attention output by c. Exercises the fp8 dequant path with a
    non-trivial magnitude rather than a fixed scale."""
    batch, qo_len, kv_len, page_size = 2, 16, 48, 16
    num_kv = num_qo = 4
    head_dim = 128
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn
    torch.manual_seed(3)
    ppq = (kv_len + page_size - 1) // page_size
    total = ppq * batch
    qo_indptr = torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * qo_len
    q = torch.randn(batch * qo_len, num_qo, head_dim, device="cuda:0", dtype=k_dtype)
    k = (
        torch.randn(
            total, page_size, num_kv, head_dim, device="cuda:0", dtype=torch.float32
        )
        * 0.5
    ).to(k_dtype)
    v_base = (
        torch.randn(
            total, page_size, num_kv, head_dim, device="cuda:0", dtype=torch.float32
        )
        * 0.25
    )
    indptr = torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * ppq
    indices = torch.arange(0, total, device="cuda:0", dtype=torch.int32)
    last = torch.full(
        (batch,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    def run_with_v(vf32):
        w = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0"), "NHD"
        )
        w.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
            paged_kv_last_page_len=last,
            num_qo_heads=num_qo,
            num_kv_heads=num_kv,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=True,
            q_data_type=k_dtype,
            k_data_type=k_dtype,
            v_data_type=v_dtype,
        )
        return w.run(q, (k, vf32.to(v_dtype))).float()

    o1 = run_with_v(v_base)
    o2 = run_with_v(v_base * 3.0)
    # 3*o1 ≈ o2 up to fp8 requantization noise of the scaled V.
    rel = (o2 - 3.0 * o1).norm() / (3.0 * o1).norm().clamp_min(1e-6)
    assert rel.item() < 0.05, f"scale-linearity rel err {rel.item()}"


def test_asymmetric_uri_backward_compat():
    """Symmetric prefill callers must keep their exact JIT cache keys."""
    common = (
        "fa2",
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,
        128,  # head_dim_qk
        128,  # head_dim_vo
        0,
        False,
        False,
        False,
    )
    legacy = get_batch_prefill_uri(*common)
    explicit_symmetric = get_batch_prefill_uri(
        *common, dtype_k=torch.float16, dtype_v=torch.float16
    )
    assert legacy == explicit_symmetric
    assert "dtype_kv_f16" in legacy

    asym = get_batch_prefill_uri(
        *common, dtype_k=torch.float16, dtype_v=torch.float8_e4m3fn
    )
    assert asym != legacy
    assert "dtype_k_f16" in asym and "dtype_v_e4m3" in asym


def _plan_asym_paged(wrapper, head_dim=128, page_size=16):
    batch_size, kv_len, qo_len = 4, 64, 16
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(
        0, num_pages_per_seq * batch_size, device="cuda:0", dtype=torch.int32
    )
    kv_last_page_len = torch.full(
        (batch_size,), page_size, dtype=torch.int32, device="cuda:0"
    )
    wrapper.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=kv_indptr,
        paged_kv_indices=kv_indices,
        paged_kv_last_page_len=kv_last_page_len,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=torch.float16,
        k_data_type=torch.float16,
        v_data_type=torch.float8_e4m3fn,
    )
    return batch_size, kv_len, qo_len


def test_asymmetric_run_v_dtype_mismatch_raises():
    """A V cache whose dtype contradicts the planned v_data_type must raise,
    not silently reinterpret fp16 bytes as fp8."""
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    batch_size, kv_len, qo_len = _plan_asym_paged(wrapper)
    page_size, num_kv_heads, head_dim = 16, 4, 128
    total_pages = ((kv_len + page_size - 1) // page_size) * batch_size
    q = torch.randn(
        batch_size * qo_len, 4, head_dim, device="cuda:0", dtype=torch.float16
    )
    k_cache = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    v_cache_wrong = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    with pytest.raises(ValueError, match="v_data_type"):
        wrapper.run(q, (k_cache, v_cache_wrong))


def test_asymmetric_run_k_dtype_mismatch_raises():
    """A K cache handed in fp8 against a bf16-K plan must raise."""
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    batch_size, kv_len, qo_len = _plan_asym_paged(wrapper)
    page_size, num_kv_heads, head_dim = 16, 4, 128
    total_pages = ((kv_len + page_size - 1) // page_size) * batch_size
    q = torch.randn(
        batch_size * qo_len, 4, head_dim, device="cuda:0", dtype=torch.float16
    )
    k_bad = torch.zeros(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float8_e4m3fn,
    )
    v_cache = torch.zeros(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float8_e4m3fn,
    )
    with pytest.raises(ValueError, match="k_data_type|kv_data_type"):
        wrapper.run(q, (k_bad, v_cache))


def test_asymmetric_head_dim_gt256_fails_closed():
    """Asymmetric K/V is validated for head_dim <= 256. head_dim >= 512 takes
    the VO-split prefill path, which has no FP8-V dequant and would read FP8 V
    as the key dtype -- it must fail closed at plan() with a clear error rather
    than compile a silently-corrupting kernel. (Symmetric head_dim >= 512 is
    unaffected: the guard only trips when k_data_type != v_data_type.)"""
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    qo_indptr = torch.tensor([0, 8], dtype=torch.int32, device="cuda:0")
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda:0")
    kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda:0")
    kv_last = torch.tensor([16], dtype=torch.int32, device="cuda:0")
    with pytest.raises(NotImplementedError, match=r"head_dim <= 256"):
        wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=kv_indptr,
            paged_kv_indices=kv_indices,
            paged_kv_last_page_len=kv_last,
            num_qo_heads=8,
            num_kv_heads=8,
            head_dim_qk=512,
            page_size=16,
            causal=True,
            q_data_type=torch.bfloat16,
            k_data_type=torch.bfloat16,
            v_data_type=torch.float8_e4m3fn,
        )


@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("k_dtype", K_DTYPES)
@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_single_prefill_asymmetric_kv(head_dim, causal, k_dtype, v_dtype):
    """Single-prefill with asymmetric K/V: 16-bit K (fp16/bf16) attended against
    FP8 (e4m3/e5m2) V. The kernel dequantizes FP8 V to the key dtype; the output
    must match an fp32 reference (error dominated by FP8 V quantization noise, as
    on the batch paged/ragged paths). Backend is auto-selected (fa2 on SM80/89,
    fa3 on SM90); both single kernels thread DTypeV."""
    num_qo_heads, num_kv_heads, qo_len, kv_len = 32, 4, 32, 129
    torch.manual_seed(7)
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype)
    k = (
        torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=k_dtype)
        * 0.5
    )
    v_fp32 = (
        torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float32
        )
        * 0.5
    )
    v = v_fp32.to(v_dtype)

    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal)

    sm_scale = 1.0 / (head_dim**0.5)
    o_ref = _ref_prefill(q, k.float(), v.float(), sm_scale, causal)
    rel = (o.float() - o_ref).norm() / o_ref.norm().clamp_min(1e-6)
    assert rel.item() < MEDIAN_REL_ERR_BOUND, (
        f"single asym rel err {rel.item()} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(k={k_dtype}, v={v_dtype}, hd={head_dim}, causal={causal})"
    )


def test_single_prefill_asymmetric_hd512_fails_closed():
    """head_dim > 256 asymmetric single-prefill takes the VO-split path, which
    keys V dequant on the K dtype (would mis-read FP8 V), so it must fail closed
    with NotImplementedError — mirroring the batch-prefill head_dim>256 guard."""
    q = torch.randn(16, 4, 512, device="cuda:0", dtype=torch.bfloat16)
    k = torch.randn(64, 4, 512, device="cuda:0", dtype=torch.bfloat16)
    v = torch.randn(64, 4, 512, device="cuda:0", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    with pytest.raises(NotImplementedError, match=r"head_dim"):
        flashinfer.single_prefill_with_kv_cache(q, k, v)


@pytest.mark.skipif(
    not _IS_SM90, reason="explicit fa3 asymmetric prefill kernel requires sm90 (Hopper)"
)
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_asymmetric_kv_fa3_explicit(head_dim, v_dtype):
    """Explicit backend="fa3" asymmetric K/V across single + ragged + paged.

    Pins backend="fa3" so the Hopper dequant-into-smem V producer (the SM90
    asymmetric path) stays under explicit-backend coverage independent of what
    backend="auto" resolves to -- the auto-selected tests above hit the same
    kernel on SM90, but pinning it guards against routing/detection drift and
    checks that an explicit fa3 request threads DTypeV correctly rather than
    being rejected by the asymmetric backend guard. Skipped off SM90, where fa3
    is unavailable. bf16 K, causal, NHD, GQA 32/8 -- one representative point
    per (head_dim, v_dtype)."""
    k_dtype = torch.bfloat16
    causal = True
    kv_layout = "NHD"
    num_qo_heads, num_kv_heads, qo_len, kv_len = 32, 8, 32, 129
    sm_scale = 1.0 / (head_dim**0.5)
    torch.manual_seed(11)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    def _rel(o, o_ref):
        return ((o.float() - o_ref).norm() / o_ref.norm().clamp_min(1e-6)).item()

    # --- single prefill (fa3) ---
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=k_dtype) * 0.5
    v = (
        torch.randn(kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=torch.float32)
        * 0.5
    ).to(v_dtype)
    o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=causal, backend="fa3")
    rel = _rel(o, _ref_prefill(q, k.float(), v.float(), sm_scale, causal))
    assert rel < MEDIAN_REL_ERR_BOUND, (
        f"single fa3 rel err {rel} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(v={v_dtype}, hd={head_dim})"
    )

    # --- ragged prefill (fa3) ---
    batch_size = 3
    qo_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * kv_len
    )
    qr = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype
    )
    kr = (
        torch.randn(
            batch_size * kv_len, num_kv_heads, head_dim, device="cuda:0", dtype=k_dtype
        )
        * 0.5
    )
    vr = (
        torch.randn(
            batch_size * kv_len,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float32,
        )
        * 0.5
    ).to(v_dtype)
    ragged = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa3"
    )
    ragged.plan(
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        causal=causal,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    o_ragged = ragged.run(qr, kr, vr)
    ragged_rel = []
    for i in range(batch_size):
        qi = qr[i * qo_len : (i + 1) * qo_len]
        ki = kr[i * kv_len : (i + 1) * kv_len].float()
        vi = vr[i * kv_len : (i + 1) * kv_len].float()
        oi = o_ragged[i * qo_len : (i + 1) * qo_len]
        ragged_rel.append(_rel(oi, _ref_prefill(qi, ki, vi, sm_scale, causal)))
    med = sorted(ragged_rel)[len(ragged_rel) // 2]
    assert med < MEDIAN_REL_ERR_BOUND, (
        f"ragged fa3 median rel err {med} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(v={v_dtype}, hd={head_dim})"
    )

    # --- paged prefill (fa3) ---
    page_size = 16
    k_cache, num_pages_per_seq, total_num_pages = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, k_dtype
    )
    v_cache, _, _ = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, v_dtype
    )
    paged_kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    paged_kv_indices = torch.arange(
        0, total_num_pages, device="cuda:0", dtype=torch.int32
    )
    paged_kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )
    paged = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, backend="fa3"
    )
    paged.plan(
        qo_indptr=qo_indptr,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        paged_kv_last_page_len=paged_kv_last_page_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=causal,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    o_paged = paged.run(qr, (k_cache, v_cache))
    paged_rel = []
    for i in range(batch_size):
        ki = _gather_seq(
            k_cache.float(), paged_kv_indptr, paged_kv_last_page_len, i, kv_layout
        )
        vi = _gather_seq(
            v_cache.float(), paged_kv_indptr, paged_kv_last_page_len, i, kv_layout
        )
        qi = qr[i * qo_len : (i + 1) * qo_len]
        oi = o_paged[i * qo_len : (i + 1) * qo_len]
        paged_rel.append(_rel(oi, _ref_prefill(qi, ki, vi, sm_scale, causal)))
    med = sorted(paged_rel)[len(paged_rel) // 2]
    assert med < MEDIAN_REL_ERR_BOUND, (
        f"paged fa3 median rel err {med} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(v={v_dtype}, hd={head_dim})"
    )


def test_asym_prefill_backend_guard_helper():
    """The prefill backend guard rejects asymmetric K/V on any backend other than
    fa2/fa3/auto (which would compile a symmetric kernel and silently reinterpret
    FP8 V as the K dtype), and never rejects symmetric K/V. Pure-Python check of
    the guard the plan()/workspace_size()/single-prefill paths all call."""
    from flashinfer.prefill import _check_asym_prefill_backend

    all_backends = ["fa2", "fa3", "auto", "cutlass", "cudnn", "trtllm-gen", "cute-dsl"]
    # Symmetric K/V: never rejected, regardless of backend.
    for b in all_backends:
        _check_asym_prefill_backend(b, torch.bfloat16, torch.bfloat16)
    # Asymmetric K/V: allowed only on the fa2/fa3 kernels (auto resolves to them).
    for b in ["fa2", "fa3", "auto"]:
        _check_asym_prefill_backend(b, torch.bfloat16, torch.float8_e4m3fn)
    for b in ["cutlass", "cudnn", "trtllm-gen", "cute-dsl"]:
        with pytest.raises(NotImplementedError, match=r"fa2"):
            _check_asym_prefill_backend(b, torch.bfloat16, torch.float8_e4m3fn)


def test_batch_prefill_ragged_asymmetric_rejects_nonfa_backend():
    """End-to-end guard wiring: constructing the ragged prefill wrapper with a
    non-fa2/fa3 backend and planning an asymmetric K/V request must fail closed at
    plan() — before any backend-specific kernel is built — not silently run a
    symmetric kernel over the FP8 V cache. Uses the in-tree 'cutlass' backend so
    the check is exercised without an optional runtime dependency, and raises
    regardless of GPU arch (the guard runs before SM100 dispatch)."""
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace, "NHD", backend="cutlass"
    )
    qo_indptr = torch.tensor([0, 32], device="cuda:0", dtype=torch.int32)
    kv_indptr = torch.tensor([0, 64], device="cuda:0", dtype=torch.int32)
    with pytest.raises(NotImplementedError, match=r"fa2"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=32,
            num_kv_heads=4,
            head_dim_qk=128,
            causal=False,
            q_data_type=torch.bfloat16,
            k_data_type=torch.bfloat16,
            v_data_type=torch.float8_e4m3fn,
        )


def test_workspace_size_accepts_asymmetric():
    """workspace_size() threads k_data_type/v_data_type for an asymmetric plan and
    returns a sane (float, int) byte tuple, and fails closed for head_dim > 256 the
    way plan() does.

    workspace_size() is an FA2-path query: upstream (#3741) wired it through the
    FA2 scheduler only, so the SM90/fa3 batch-prefill binding exports no
    workspace_size and backend="auto" -- which resolves to fa3 on Hopper -- would
    raise NotImplementedError there. Pin backend="fa2" (the CUDA-core kernel runs
    on every supported arch, SM90 included), mirroring upstream's own
    workspace_size test, so this exercises the asymmetric dtype threading
    arch-independently instead of only where "auto" happens to pick fa2."""
    batch, kv_len, qo_len, page_size = 4, 64, 16, 16
    ppq = (kv_len + page_size - 1) // page_size
    qo_indptr = torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * qo_len
    kv_indptr = torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * ppq
    kv_indices = torch.arange(0, ppq * batch, device="cuda:0", dtype=torch.int32)
    kv_last = torch.full((batch,), page_size, dtype=torch.int32, device="cuda:0")
    wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0"),
        "NHD",
        backend="fa2",
    )
    fb, ib = wrapper.workspace_size(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim_qk=128,
        page_size=page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        k_data_type=torch.bfloat16,
        v_data_type=torch.float8_e4m3fn,
    )
    assert isinstance(fb, int) and isinstance(ib, int)
    assert fb >= 0 and ib >= 0

    # head_dim > 256 asymmetric fails closed before any module is built (the same
    # VO-split fail-close plan() enforces; the check runs ahead of backend
    # resolution, so it fires regardless of the pinned backend).
    with pytest.raises(NotImplementedError, match=r"head_dim <= 256"):
        wrapper.workspace_size(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last,
            num_qo_heads=8,
            num_kv_heads=8,
            head_dim_qk=512,
            page_size=page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            k_data_type=torch.bfloat16,
            v_data_type=torch.float8_e4m3fn,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
