# DVD DiT AWQ Quantization

Weight-only [AWQ](https://arxiv.org/abs/2306.00978) quantization of the DVD
depth-estimation DiT — a step-by-step tutorial. The pipeline lives in
`quant_script/`; three small vendored edits in `llm-awq/` teach llm-awq's AWQ
implementation about DVD's DiT block.

Two checkpoints come out of it:

| checkpoint | bits | size | vs bf16 (2.84 GB) |
|---|---|---|---|
| `dvd_dit_awq_w8g128.safetensors` | w8g128 | 1.49 GB | 1.9× smaller — near-lossless |
| `dvd_dit_awq_w4g128.safetensors` | w4g128 | 0.79 GB | 3.6× smaller — some quality trade-off |

## How it works

DVD's DiT is a Wan2.1-1.3B video-depth regressor: 30 `DiTBlock`s, 300
`nn.Linear` layers. We quantize those 300 linears weight-only (activations
stay bf16 — W4A16 / W8A16) with AWQ: group size 128, zero-point, per-group
scale search plus weight clipping. AWQ's activation-aware scaling is applied
to 4 mathematically-exact fusion groups per block (`norm3 → cross_attn.q`,
`self_attn.v → .o`, `cross_attn.v → .o`, `ffn GELU → ffn.2`); `q`/`k` are
skipped for clipping. See [Notes](#notes) for the design details worth knowing.

## Setup

```bash
git clone https://github.com/EnVision-Research/DVD.git && cd DVD
conda create -n dvd python=3.10 -y && conda activate dvd
pip install -e .
```

Any single CUDA GPU with ~20 GB free is enough — AWQ processes one DiT block
at a time. Re-run `pip install -e .` if you add top-level packages later.

**Vendored llm-awq.** `llm-awq/` carries a minimal, DVD-modified subset of
MIT-licensed [llm-awq](https://github.com/mit-han-lab/llm-awq) (the DiTBlock
edits are in `llm-awq/awq/quantize/auto_scale.py` and `auto_clip.py`).
`quant_script.common` puts it on `sys.path` automatically — nothing to install.

Download the base checkpoint:

```bash
huggingface-cli login   # or: hf auth login
huggingface-cli download FayeHongfeiZhang/DVD --revision main --local-dir ckpt
```

## Step 1 — Merge LoRA into a stand-alone bf16 DiT

The released checkpoint is a base DiT plus a LoRA adapter. Fold them into one
clean bf16 DiT that every later step starts from:

```bash
python quant_script/save_merged_dit.py \
  --ckpt ckpt --model_config ckpt/model_config.yaml \
  --out ckpt/dit_merged.safetensors
```

Expect `Unwrapped 300 LoRA linears` / `Saved ckpt/dit_merged.safetensors:
825 tensors, 1.419B params`.

## Step 2 — Capture calibration activations

AWQ needs example activations. Because the DVD DiT runs single-pass at one
fixed timestep with an empty text context, the natural calibration set is the
real tensors block 0 sees during depth inference. `capture_calib.py` records
them via a forward hook, one `.pt` per 81-frame window:

```bash
python quant_script/capture_calib.py \
  --videos demo/robot_navi.mp4 --dit_sd ckpt/dit_merged.safetensors \
  --out_dir calib --max_windows_per_video 4
```

Use a few videos representative of your target content (pass several to
`--videos`). Calibration quality matters more for w4 than w8.

## Step 3 — AWQ scale/clip search

```bash
# w8 (near-lossless)
python quant_script/run_awq_search.py \
  --dit_sd ckpt/dit_merged.safetensors --calib_glob 'calib/*.pt' \
  --w_bit 8 --q_group_size 128 --out ckpt/awq_results_w8g128.pt

# w4 (max compression)
python quant_script/run_awq_search.py \
  --dit_sd ckpt/dit_merged.safetensors --calib_glob 'calib/*.pt' \
  --w_bit 4 --q_group_size 128 --out ckpt/awq_results_w4g128.pt
```

`--calib_glob` accepts one or more comma-separated glob patterns. Expect
`saved ckpt/awq_results_*.pt: 120 scale groups, 180 clip entries`. The search
is block-by-block (~47M params resident at a time), so it is fast and not
memory-heavy.

## Step 4 — Pack into a real low-bit checkpoint

```bash
python quant_script/pack_awq.py --dit_sd ckpt/dit_merged.safetensors \
  --awq_results ckpt/awq_results_w8g128.pt --w_bit 8 --q_group_size 128 \
  --out ckpt/dvd_dit_awq_w8g128.safetensors

python quant_script/pack_awq.py --dit_sd ckpt/dit_merged.safetensors \
  --awq_results ckpt/awq_results_w4g128.pt --w_bit 4 --q_group_size 128 \
  --out ckpt/dvd_dit_awq_w4g128.safetensors
```

w4 bit-packs two 4-bit levels per byte (true 4-bit on disk); w8 stores one
byte per weight. Everything outside the 300 block linears is copied through
unchanged. The pack/unpack round-trip is exactly lossless.

## Step 5 — Run inference with the quantized checkpoint

```bash
python test_script/test_single_video.py \
  --ckpt ckpt --model_config ckpt/model_config.yaml \
  --dit_sd ckpt/dit_merged.safetensors \
  --input_video demo/drone.mp4 --output_dir inference_results/awq_w4 \
  --awq_ckpt ckpt/dvd_dit_awq_w4g128.safetensors
```

`--awq_ckpt` auto-detects the packed format, unpacks/dequantizes every linear
back to bf16, reinstates the `ffn[1]` `ScaledActivation` wrappers, and loads.
It is mutually exclusive with `--awq_results`/`--pseudo_quant` (an assertion
blocks combining them — the checkpoint is already quantized).

## Notes

- **`norm3` scales `cross_attn.q` only.** In `CrossAttention.forward`, `q`
  comes from the block's residual stream (`norm3`), but `k`/`v` come from the
  external text `context` — so fusing a `norm3`-derived scale into `k`/`v`
  would be unsound. The other three groups are exact by construction.
- **Config.** `--w_bit` and `--q_group_size` must match between the search
  (Step 3) and the pack (Step 4). `zero_point=True`, `q_group_size=128` are the
  defaults used throughout.
- **Disk vs. VRAM.** This packing saves *disk/download* size. The loader
  dequantizes back to bf16 at load time, so runtime VRAM is roughly the same
  as bf16 — there is no custom low-bit matmul kernel. For genuine VRAM savings,
  a weight-only int8/int4 runtime (e.g. torchao) applied after AWQ is a natural
  extension, not included here.

## License

- **Code** (`quant_script/`, the DiT edits, this document): Apache 2.0, matching
  the DVD repo. Vendored `llm-awq/` is MIT.
- **Quantized weights** are derivative works of the DVD v1.0 weights and inherit
  their [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license
  — non-commercial use only. Please cite the DVD paper:

  ```bibtex
  @article{zhang2026dvd,
    title={DVD: Deterministic Video Depth Estimation with Generative Priors},
    author={Zhang, Hongfei and Chen, Harold Haodong and Liao, Chenfei and He, Jing and Zhang, Zixin and Li, Haodong and Liang, Yihao and Chen, Kanghao and Ren, Bin and Zheng, Xu and others},
    journal={arXiv preprint arXiv:2603.12250},
    year={2026}
  }
  ```
