"""Check that mixed output-scale fusion preserves the two FP16 roundings."""

import pytest
import torch

import flashinfer


@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("length", [33, 8192])
@pytest.mark.parametrize("kind", ["decode-split", "decode-nosplit", "prefill"])
@pytest.mark.parametrize("scale", [0.03125, 0.037, 1.7])
def test_mixed_rounded_output_scale(batch, length, kind, scale, monkeypatch):
    torch.manual_seed(173)
    pages = (length + 15) // 16
    key = torch.randn(batch * pages, 16, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key).to(torch.float8_e4m3fn)
    ptr = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * pages
    indices = torch.randperm(batch * pages, device="cuda").to(torch.int32)
    tail = torch.full((batch,), (length - 1) % 16 + 1, device="cuda", dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    if kind.startswith("decode"):
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=True, backend="auto"
        )
        wrapper.plan(
            ptr,
            indices,
            tail,
            32,
            8,
            128,
            16,
            q_data_type=torch.float16,
            k_data_type=torch.float16,
            v_data_type=torch.float8_e4m3fn,
            disable_split_kv=kind == "decode-nosplit",
            fixed_split_size=256 if kind == "decode-split" else None,
        )
        query = torch.randn(batch, 32, 128, device="cuda", dtype=torch.float16)
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa3"
        )
        wrapper.plan(
            torch.arange(batch + 1, device="cuda", dtype=torch.int32) * length,
            ptr,
            indices,
            tail,
            32,
            8,
            128,
            16,
            causal=True,
            q_data_type=torch.float16,
            k_data_type=torch.float16,
            v_data_type=torch.float8_e4m3fn,
        )
        query = torch.randn(batch * length, 32, 128, device="cuda", dtype=torch.float16)
    monkeypatch.setenv("FLASHINFER_FUSE_V_SCALE", "0")
    original, original_lse = wrapper.run(
        query, (key, value), v_scale=scale, return_lse=True
    )
    original, original_lse = original.clone(), original_lse.clone()
    monkeypatch.setenv("FLASHINFER_FUSE_V_SCALE", "1")
    fused, fused_lse = wrapper.run(query, (key, value), v_scale=scale, return_lse=True)
    assert torch.equal(original, fused)
    assert torch.equal(original_lse, fused_lse)


@pytest.mark.parametrize("kind", ["decode", "prefill"])
def test_other_output_type_keeps_standalone_scale(kind, monkeypatch):
    torch.manual_seed(391)
    workspace = torch.empty(128 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    key = torch.randn(3, 16, 8, 128, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key).to(torch.float8_e4m3fn)
    ptr = torch.tensor([0, 3], device="cuda", dtype=torch.int32)
    pages = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    tail = torch.tensor([1], device="cuda", dtype=torch.int32)
    if kind == "decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=True, backend="auto"
        )
        wrapper.plan(
            ptr,
            pages,
            tail,
            32,
            8,
            128,
            16,
            q_data_type=torch.float16,
            k_data_type=torch.float16,
            v_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
        query = torch.randn(1, 32, 128, device="cuda", dtype=torch.float16)
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa3"
        )
        wrapper.plan(
            torch.tensor([0, 3], device="cuda", dtype=torch.int32),
            ptr,
            pages,
            tail,
            32,
            8,
            128,
            16,
            causal=True,
            q_data_type=torch.float16,
            k_data_type=torch.float16,
            v_data_type=torch.float8_e4m3fn,
            o_data_type=torch.bfloat16,
        )
        query = torch.randn(3, 32, 128, device="cuda", dtype=torch.float16)
    monkeypatch.setenv("FLASHINFER_FUSE_V_SCALE", "0")
    original, lse = wrapper.run(query, (key, value), v_scale=0.037, return_lse=True)
    original, lse = original.clone(), lse.clone()
    monkeypatch.setenv("FLASHINFER_FUSE_V_SCALE", "1")
    output, output_lse = wrapper.run(
        query, (key, value), v_scale=0.037, return_lse=True
    )
    assert output.dtype == torch.bfloat16
    assert torch.equal(original, output)
    assert torch.equal(lse, output_lse)
