import argparse
import glob

import torch

from quant_script.awq_dit import run_awq_dit
from quant_script.common import build_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dit_sd", default="ckpt/dit_merged.safetensors")
    p.add_argument("--ckpt", default="ckpt")
    p.add_argument("--model_config", default="ckpt/model_config.yaml")
    p.add_argument("--calib_glob", default="calib/robot_navi_*.pt")
    p.add_argument("--w_bit", type=int, default=4)
    p.add_argument("--q_group_size", type=int, default=128)
    p.add_argument("--out", default="ckpt/awq_results_w4g128.pt")
    args = p.parse_args()

    patterns = [pat.strip() for pat in args.calib_glob.split(",") if pat.strip()]
    files = sorted(sum((glob.glob(pat) for pat in patterns), []))
    assert files, f"no calib files match {args.calib_glob}"
    samples = [torch.load(f, map_location="cpu") for f in files]
    print(f"{len(samples)} calibration windows")

    model = build_model(args.ckpt, args.model_config, dit_sd=args.dit_sd)
    dit = model.pipe.dit.eval()

    q_config = dict(zero_point=True, q_group_size=args.q_group_size)
    results = run_awq_dit(dit, samples, w_bit=args.w_bit, q_config=q_config)
    torch.save(results, args.out)
    print(f"saved {args.out}: {len(results['scale'])} scale groups, {len(results['clip'])} clip entries")


if __name__ == "__main__":
    main()
