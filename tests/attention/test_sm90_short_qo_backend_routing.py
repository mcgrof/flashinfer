"""
SM90 decode-like short-query backend routing (fix A).

On SM90 the batch-prefill ``auto`` backend must send decode-like short-query
shapes (small per-request qo_len against a long KV cache -- speculative-decode
verify / append / chunked decode) to FA2, not the prefill-tuned FA3 kernel.

Rationale (measured on H100 SXM5, Nsight Compute, bf16, GQA n=28/nkv=4, hd=128):
FA3's prefill kernel schedules work per query-head with no GQA K/V reuse and no
split-KV, so at the verify shape (qo_len=4) it rereads the KV cache ~5x from HBM
(21.4 GB vs FA2's 4.3 GB at b=64/kv=32k) and starves at low batch -- 5.9x/12.7x
slower than FA2, which sits at the memory roofline. FA2 is a numeric drop-in
(rel-err ~4e-3). See flashinfer.utils.SM90_DECODE_LIKE_QO_THRESHOLD.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import (
    PosEncodingMode,
    SM90_DECODE_LIKE_QO_THRESHOLD,
    determine_attention_backend,
    is_sm90a_supported,
)

THR = SM90_DECODE_LIKE_QO_THRESHOLD


def _skip_if_not_sm90():
    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda:0")):
        pytest.skip("SM90 short-qo routing is SM90-only")


@pytest.mark.parametrize(
    "max_qo_len,expected",
    [
        (1, "fa2"),  # single-token decode-like
        (4, "fa2"),  # chain speculative verify
        (THR, "fa2"),  # at threshold -> still FA2
        (THR + 1, "fa3"),  # just above -> FA3 (prefill-like)
        (256, "fa3"),  # real prefill
        (None, "fa3"),  # unknown -> backward-compatible (prior behavior)
    ],
)
def test_determine_backend_short_qo_routing(max_qo_len, expected):
    _skip_if_not_sm90()
    dev = torch.device("cuda:0")
    got = determine_attention_backend(
        dev,
        PosEncodingMode.NONE.value,
        False,  # use_fp16_qk_reductions
        False,  # use_custom_mask
        torch.bfloat16,  # dtype_q
        torch.bfloat16,  # dtype_kv
        max_qo_len=max_qo_len,
    )
    assert got == expected, f"max_qo_len={max_qo_len}: expected {expected}, got {got}"


def test_determine_backend_fp8_kv_not_rerouted():
    """fp8 KV keeps its existing FA3 selection: short qo does NOT force FA2
    (no measured evidence there; the dtype gate is float16/bfloat16 only)."""
    _skip_if_not_sm90()
    dev = torch.device("cuda:0")
    got = determine_attention_backend(
        dev,
        PosEncodingMode.NONE.value,
        False,
        False,
        torch.float8_e4m3fn,  # dtype_q (fp8 q required for fa3 fp8 kv)
        torch.float8_e4m3fn,  # dtype_kv
        max_qo_len=4,
    )
    assert got == "fa3"


def _make_verify_shape(kv_len=8192, batch=8, qo_len=4, page_size=16):
    num_qo_heads, num_kv_heads, head_dim = 28, 4, 128
    dev, dtype = "cuda:0", torch.bfloat16
    pages_per_req = kv_len // page_size
    total_pages = batch * pages_per_req
    qo_indptr = torch.arange(
        0, (batch + 1) * qo_len, qo_len, dtype=torch.int32, device=dev
    )
    kv_indptr = torch.arange(
        0, (batch + 1) * pages_per_req, pages_per_req, dtype=torch.int32, device=dev
    )
    kv_indices = torch.arange(0, total_pages, dtype=torch.int32, device=dev)
    last = torch.full((batch,), page_size, dtype=torch.int32, device=dev)
    q = torch.randn(batch * qo_len, num_qo_heads, head_dim, dtype=dtype, device=dev)
    kv = torch.randn(
        total_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=dev
    )
    plan_args = dict(
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim_qk=head_dim,
        page_size=page_size,
        causal=True,
        q_data_type=dtype,
        kv_data_type=dtype,
    )
    return (qo_indptr, kv_indptr, kv_indices, last), (q, kv), plan_args


def _run(backend, idx, tensors, plan_args):
    ws = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        ws, kv_layout="NHD", backend=backend
    )
    w.plan(*idx, **plan_args)
    out = w.run(*tensors)
    return w._backend, out


def test_auto_selects_fa2_for_verify_shape():
    """The paged wrapper with backend='auto' picks FA2 for the qo_len=4 verify
    shape and FA3 for a long-qo prefill shape."""
    _skip_if_not_sm90()
    idx, tensors, plan_args = _make_verify_shape(qo_len=4)
    sel, _ = _run("auto", idx, tensors, plan_args)
    assert sel == "fa2", f"verify shape should route to fa2, got {sel}"

    idx_l, tensors_l, plan_args_l = _make_verify_shape(qo_len=256, kv_len=2048, batch=2)
    sel_l, _ = _run("auto", idx_l, tensors_l, plan_args_l)
    assert sel_l == "fa3", f"long-qo prefill should stay fa3, got {sel_l}"


def test_fa2_is_numeric_dropin_for_fa3():
    """FA2 and FA3 agree at the verify shape (routing introduces no error)."""
    _skip_if_not_sm90()
    idx, tensors, plan_args = _make_verify_shape(qo_len=4)
    _, out_fa2 = _run("fa2", idx, tensors, plan_args)
    _, out_fa3 = _run("fa3", idx, tensors, plan_args)
    torch.testing.assert_close(out_fa2, out_fa3, rtol=2e-2, atol=2e-2)
