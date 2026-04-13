"""INT4 packed V cache utilities for FlashInfer asymmetric K16/V4.

Provides:
- quantize_to_int4_packed: FP16 -> packed uint8 + per-group scales
- dequantize_int4_packed: packed uint8 + scales -> FP16 (for verification)
"""
import torch


def quantize_to_int4_packed(
    v: torch.Tensor, group_size: int = 128
) -> tuple:
    """Quantize FP16/BF16 values to symmetric INT4 packed as uint8.

    Args:
        v: Value tensor, shape [..., head_dim]. head_dim must be
           divisible by group_size and by 2 (for nibble packing).
        group_size: Elements per scale group. Default 128 means
                    1 scale per head (head_dim=128).

    Returns:
        packed: uint8 tensor, shape [..., head_dim // 2].
                Low nibble = even element, high nibble = odd element.
        scales: float16 tensor, shape [..., head_dim // group_size].
    """
    assert v.shape[-1] % 2 == 0, "head_dim must be even"
    assert v.shape[-1] % group_size == 0, (
        f"head_dim {v.shape[-1]} not divisible by group_size {group_size}"
    )

    orig_shape = v.shape
    head_dim = orig_shape[-1]
    num_groups = head_dim // group_size

    # Reshape for per-group quantization
    v_grouped = v.reshape(*orig_shape[:-1], num_groups, group_size)
    v_float = v_grouped.float()

    # Per-group absmax scale
    absmax = v_float.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scales = (absmax / 7.0).to(torch.float16)  # INT4 range [-8, 7]

    # Quantize
    v_quant = (v_float / scales.float()).round().clamp(-8, 7)
    # Map to unsigned [0, 15] for packing
    v_unsigned = (v_quant + 8).to(torch.uint8)
    v_unsigned = v_unsigned.reshape(*orig_shape)

    # Pack pairs of elements into bytes: low nibble = even, high = odd
    even = v_unsigned[..., 0::2]  # [..., head_dim//2]
    odd = v_unsigned[..., 1::2]   # [..., head_dim//2]
    packed = (odd << 4) | even  # high nibble = odd, low = even

    # Scales: [..., num_groups]
    scales_out = scales.squeeze(-1).reshape(
        *orig_shape[:-1], num_groups
    )

    return packed, scales_out


def dequantize_int4_packed(
    packed: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Dequantize packed INT4 uint8 back to float16.

    Args:
        packed: uint8, shape [..., head_dim // 2]
        scales: float16, shape [..., head_dim // group_size]
        group_size: Elements per scale group.

    Returns:
        v: float16 tensor, shape [..., head_dim]
    """
    # Unpack nibbles
    even = (packed & 0x0F).to(torch.int8) - 8  # signed [-8, 7]
    odd = ((packed >> 4) & 0x0F).to(torch.int8) - 8

    # Interleave back to head_dim
    head_dim = packed.shape[-1] * 2
    v = torch.zeros(
        *packed.shape[:-1], head_dim,
        dtype=torch.float16, device=packed.device,
    )
    v[..., 0::2] = even.float()
    v[..., 1::2] = odd.float()

    # Apply per-group scales
    num_groups = scales.shape[-1]
    v_grouped = v.reshape(*v.shape[:-1], num_groups, group_size)
    scales_expanded = scales.float().unsqueeze(-1)
    v_grouped = v_grouped * scales_expanded
    v = v_grouped.reshape(*v.shape[:-1], head_dim).to(torch.float16)

    return v


if __name__ == "__main__":
    # Quick self-test
    torch.manual_seed(42)
    head_dim = 128
    v = torch.randn(2, 4, 32, head_dim, dtype=torch.float16)

    packed, scales = quantize_to_int4_packed(v, group_size=128)
    print(f"Input:  {v.shape} {v.dtype}")
    print(f"Packed: {packed.shape} {packed.dtype}")
    print(f"Scales: {scales.shape} {scales.dtype}")

    v_deq = dequantize_int4_packed(packed, scales, group_size=128)
    print(f"Dequant: {v_deq.shape} {v_deq.dtype}")

    # Check reconstruction error
    err = (v.float() - v_deq.float()).abs()
    print(f"Max abs error: {err.max().item():.6f}")
    print(f"Mean abs error: {err.mean().item():.6f}")

    # Cosine similarity
    cos = torch.nn.functional.cosine_similarity(
        v.float().reshape(-1, head_dim),
        v_deq.float().reshape(-1, head_dim),
        dim=-1,
    )
    print(f"Cosine similarity: {cos.mean().item():.6f}")
    print(f"Min cosine sim: {cos.min().item():.6f}")

    assert packed.shape[-1] == head_dim // 2
    assert scales.shape[-1] == 1  # group_size == head_dim
    print("\nSelf-test PASSED")
