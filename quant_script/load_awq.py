"""Load a Tier-A portable packed AWQ checkpoint (see quant_script/pack_awq.py)
into an already-built DVD DiT, in place.

No custom CUDA kernel involved: every packed Linear is unpacked and
dequantized back to a plain bf16 weight, so this runs at ordinary bf16 VRAM
(the win is the checkpoint's on-disk/download size, not runtime memory --
see task-7-report.md for the Tier B torchao attempt at an actual VRAM win).

Dequantization reproduces awq.quantize.quantizer.pseudo_quantize_tensor's
output bit-for-bit: pack_awq.quantize_linear_int stores levels satisfying
`levels == clamp(round(w / scales) + zeros, 0, max_int)` (computed in the
weight's own dtype, e.g. bf16 -- the same expression pseudo_quantize_tensor
evaluates internally). Dequantizing here with `(levels - zeros) * scales`,
in that same dtype, is therefore the exact same formula run over the exact
same tensors as pseudo_quantize_tensor's `(clamp(...) - zeros) * scales` --
not merely a close approximation.
"""
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import quant_script.common  # noqa: F401  (installs llm-awq sys.path)
from awq.quantize.qmodule import ScaledActivation

FORMAT = "dvd_awq_packed_v1"


def unpack_int4(qweight, in_features):
    """Inverse of pack_awq.pack_int4: two uint4 levels per uint8 byte ->
    [out_features, in_features] uint8 levels (0-15)."""
    lo = qweight & 0x0F
    hi = (qweight >> 4) & 0x0F
    out = torch.empty(qweight.shape[0], in_features, dtype=torch.uint8, device=qweight.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


@torch.no_grad()
def dequantize_linear(qweight, scales, zeros, w_bit, q_group_size):
    """Unpack + dequantize one packed Linear's qweight/scales/zeros back to
    a float weight tensor of shape [out_features, in_features], in the same
    dtype `scales`/`zeros` were stored in (see module docstring: this dtype
    match, not just the formula, is what makes the round trip exact)."""
    if w_bit == 4:
        levels = unpack_int4(qweight, scales.shape[1] * q_group_size)
    else:
        levels = qweight
    levels = levels.to(scales.dtype)  # exact: levels are ints 0..255, representable in bf16
    scales_full = scales.repeat_interleave(q_group_size, dim=1)
    zeros_full = zeros.repeat_interleave(q_group_size, dim=1)
    return (levels - zeros_full) * scales_full


def is_packed_awq_checkpoint(path):
    """True if `path` is a Tier-A dvd_awq_packed_v1 checkpoint (vs. e.g. a
    plain merged state dict or a Tier-B torchao checkpoint)."""
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    return metadata.get("format") == FORMAT


@torch.no_grad()
def load_awq_dit(dit, path):
    """Load the Tier-A packed checkpoint at `path` into `dit` in place.
    Reinstates a `ScaledActivation` on each block's `ffn[1]` (GELU) wherever
    the checkpoint has one (apply_awq's AWQ-scale fusion, saved as
    `blocks.{i}.ffn.1.scales`) *before* loading, since that's a structural
    change (ffn[1] stops being a bare nn.GELU) -- not just a tensor value.
    Every packed `<prefix>.qweight/.scales/.zeros` triple is unpacked and
    written back as `<prefix>.weight`; everything else in the checkpoint
    (norms, patch_embedding, head, modulation, biases, ffn.*.scales) loads
    unchanged. Caller is responsible for moving `dit` to its target device
    afterward (e.g. `dit.cuda()`) -- see test_script/test_single_video.py.
    """
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    assert metadata.get("format") == FORMAT, (
        f"{path} metadata {metadata!r} is not a Tier-A {FORMAT!r} checkpoint")
    w_bit = int(metadata["w_bit"])
    q_group_size = int(metadata["q_group_size"])

    sd = load_file(path, device="cpu")

    for i, block in enumerate(dit.blocks):
        scale_key = f"blocks.{i}.ffn.1.scales"
        if scale_key in sd:
            block.ffn[1] = ScaledActivation(block.ffn[1], sd[scale_key].clone())

    packed_prefixes = {k[: -len(".qweight")] for k in sd if k.endswith(".qweight")}
    consumed = set()
    for prefix in packed_prefixes:
        consumed.update((f"{prefix}.qweight", f"{prefix}.scales", f"{prefix}.zeros"))

    final_sd = {}
    for prefix in packed_prefixes:
        final_sd[f"{prefix}.weight"] = dequantize_linear(
            sd[f"{prefix}.qweight"], sd[f"{prefix}.scales"], sd[f"{prefix}.zeros"],
            w_bit, q_group_size)
    for k, v in sd.items():
        if k not in consumed:
            final_sd[k] = v

    missing, unexpected = dit.load_state_dict(final_sd, strict=False)
    assert not unexpected, f"unexpected keys loading {path}: {unexpected}"
    assert not missing, f"missing keys loading {path}: {missing}"
    print(f"loaded {path}: w{w_bit}g{q_group_size}, {len(packed_prefixes)} unpacked linears")
    return dit
