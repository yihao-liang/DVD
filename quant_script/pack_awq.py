"""Pack an AWQ-smoothed DVD DiT into a real, smaller low-bit checkpoint.

Tier A of Task 7 ("portable real-low-bit checkpoint"): no custom CUDA kernel
required (llm-awq's WQLinear/awq_inference_engine is INT4-only and would not
build on this box's sm_100 + CUDA-13.0-nvcc-vs-CUDA-12.8-torch toolchain --
see task-7-report.md). Instead we group-quantize every DiT block Linear with
the *exact* routine test_single_video.py's --pseudo_quant path already uses
(awq.quantize.quantizer.pseudo_quantize_tensor, get_scale_zp=True), but keep
the integer levels instead of the dequantized float weight, and pack them:

  - w8: one uint8 byte per weight (levels are 0..255).
  - w4: two uint4 levels (0..15) bit-packed per uint8 byte (true 4-bit on disk).

Everything that isn't a block Linear (patch_embedding, head, modulation,
norms, and the `ScaledActivation` `.scales` that `apply_awq` inserts on each
block's `ffn[1]` GELU) is copied through unchanged. See quant_script/load_awq.py
for the loader and docstring proving the round trip reproduces
pseudo_quantize_tensor bit-for-bit (not just approximately).
"""
import argparse

import torch
import tqdm
from safetensors.torch import save_file

import quant_script.common  # noqa: F401  (installs llm-awq sys.path)
from awq.quantize.pre_quant import apply_awq
from awq.quantize.quantizer import pseudo_quantize_tensor
from quant_script.awq_dit import get_named_linears
from quant_script.common import build_model

FORMAT = "dvd_awq_packed_v1"


@torch.no_grad()
def quantize_linear_int(w, n_bit, q_group_size):
    """Group-quantize `w` ([out_features, in_features]) and return the
    integer levels alongside the per-group scales/zeros, instead of the
    dequantized float weight pseudo_quantize_tensor normally returns.

    Calls pseudo_quantize_tensor(w, n_bit, zero_point=True, q_group_size,
    get_scale_zp=True) -- the same call test_single_video.py's --pseudo_quant
    path makes -- purely to get `scales`/`zeros` (its dequantized return
    value is discarded). The levels are then recomputed as
    `clamp(round(w / scales) + zeros, 0, max_int)`: the exact same tensors
    (`w`, `scales`, `zeros`) run through the exact same elementwise formula
    pseudo_quantize_tensor uses internally just before its final
    `(... - zeros) * scales` dequantization step. Elementwise ops don't
    depend on tensor shape/strides, so this is bit-identical to what
    pseudo_quantize_tensor computed internally -- i.e. dequantizing these
    levels reproduces pseudo_quantize_tensor's output exactly (see
    load_awq.dequantize_linear).
    """
    _, scales, zeros = pseudo_quantize_tensor(
        w, n_bit=n_bit, zero_point=True, q_group_size=q_group_size, get_scale_zp=True)
    max_int = 2 ** n_bit - 1
    scales_full = scales.repeat_interleave(q_group_size, dim=1)
    zeros_full = zeros.repeat_interleave(q_group_size, dim=1)
    levels = torch.clamp(torch.round(w / scales_full) + zeros_full, 0, max_int)
    return levels.to(torch.uint8), scales.contiguous(), zeros.contiguous()


def pack_int4(levels):
    """Bit-pack two uint4 levels (0-15) per byte along dim=1 (in_features)."""
    out_features, in_features = levels.shape
    assert in_features % 2 == 0, f"in_features={in_features} must be even to pack int4"
    lo = levels[:, 0::2]
    hi = levels[:, 1::2]
    return (lo | (hi << 4)).to(torch.uint8).contiguous()


@torch.no_grad()
def real_quantize_dit(dit, w_bit, q_group_size):
    """Pack every DiT block Linear to real low-bit levels. Returns a dict of
    {key: tensor} to merge into the checkpoint: packed linears contribute
    `<prefix>.qweight`/`.scales`/`.zeros` (in place of `<prefix>.weight`);
    biases are left for the caller to copy through from dit.state_dict()
    as-is (they aren't quantized)."""
    packed = {}
    n_packed = 0
    for i, block in enumerate(tqdm.tqdm(dit.blocks, desc=f"packing w{w_bit}g{q_group_size}")):
        block.cuda()
        for name, m in get_named_linears(block).items():
            levels, scales, zeros = quantize_linear_int(m.weight.data, w_bit, q_group_size)
            prefix = f"blocks.{i}.{name}"
            qweight = pack_int4(levels) if w_bit == 4 else levels
            packed[f"{prefix}.qweight"] = qweight.cpu().contiguous()
            packed[f"{prefix}.scales"] = scales.cpu().contiguous()
            packed[f"{prefix}.zeros"] = zeros.cpu().contiguous()
            n_packed += 1
        block.cpu()
    print(f"packed {n_packed} linears to w{w_bit}g{q_group_size}")
    return packed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt")
    p.add_argument("--model_config", default="ckpt/model_config.yaml")
    p.add_argument("--dit_sd", default="ckpt/dit_merged.safetensors")
    p.add_argument("--awq_results", default="ckpt/awq_results_w4g128.pt")
    p.add_argument("--out", default="ckpt/dvd_dit_awq_w4g128.safetensors")
    p.add_argument("--w_bit", type=int, default=4)
    p.add_argument("--q_group_size", type=int, default=128)
    args = p.parse_args()

    model = build_model(args.ckpt, args.model_config, dit_sd=args.dit_sd)
    dit = model.pipe.dit.eval()

    apply_awq(dit, torch.load(args.awq_results, map_location="cpu"))
    print(f"applied AWQ results from {args.awq_results}")

    packed_linears = real_quantize_dit(dit, args.w_bit, args.q_group_size)
    packed_prefixes = {k.rsplit(".", 1)[0] for k in packed_linears if k.endswith(".qweight")}

    sd = {}
    for k, v in dit.state_dict().items():
        prefix = k.rsplit(".", 1)[0]
        if k.endswith(".weight") and prefix in packed_prefixes:
            continue  # superseded by packed qweight/scales/zeros below
        sd[k] = v.contiguous().cpu()
    sd.update(packed_linears)

    metadata = {"format": FORMAT, "w_bit": str(args.w_bit), "q_group_size": str(args.q_group_size)}
    save_file(sd, args.out, metadata=metadata)

    size_gb = sum(v.numel() * v.element_size() for v in sd.values()) / 1e9
    print(f"saved {args.out}: {size_gb:.3f} GB, {len(sd)} tensors, "
          f"{len(packed_prefixes)} packed linears (w{args.w_bit}g{args.q_group_size})")


if __name__ == "__main__":
    main()
