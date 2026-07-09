import argparse

import torch
from safetensors.torch import save_file

from quant_script.common import build_model, unwrap_lora


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="ckpt")
    p.add_argument("--model_config", default="ckpt/model_config.yaml")
    p.add_argument("--out", default="ckpt/dit_merged.safetensors")
    args = p.parse_args()

    model = build_model(args.ckpt, args.model_config)
    n = unwrap_lora(model.pipe.dit)
    print(f"Unwrapped {n} LoRA linears")
    # q/k/v/o target names match BOTH self_attn and cross_attn -> 10 linears/block x 30 blocks
    # (confirmed empirically in Task 1: "Merged 300 LoRA layers into base weights.")
    assert n == 300, f"expected 300 (10 lora-wrapped linears x 30 blocks), got {n}"

    sd = {k: v.to(torch.bfloat16).contiguous() for k, v in model.pipe.dit.state_dict().items()}
    bad = [k for k in sd if "lora" in k or "base_layer" in k]
    assert not bad, f"PEFT keys leaked into merged state dict: {bad[:5]}"
    save_file(sd, args.out)
    total = sum(v.numel() for v in sd.values())
    print(f"Saved {args.out}: {len(sd)} tensors, {total/1e9:.3f}B params")


if __name__ == "__main__":
    main()
