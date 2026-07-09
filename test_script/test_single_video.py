import argparse
import os
import re
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from diffsynth import save_video
from quant_script.common import (get_window_index, pad_time_mod4,
                                 read_video, resize_for_training_scale)


# =============================
# Helper: Math & Alignment
# =============================
def compute_scale_and_shift(curr_frames, ref_frames, mask=None):
    """Computes scale and shift for overlap alignment."""
    if mask is None:
        mask = np.ones_like(ref_frames)

    a_00 = np.sum(mask * curr_frames * curr_frames)
    a_01 = np.sum(mask * curr_frames)
    a_11 = np.sum(mask)
    b_0 = np.sum(mask * curr_frames * ref_frames)
    b_1 = np.sum(mask * ref_frames)

    det = a_00 * a_11 - a_01 * a_01
    if det != 0:
        scale = (a_11 * b_0 - a_01 * b_1) / det
        shift = (-a_01 * b_0 + a_00 * b_1) / det
    else:
        scale, shift = 1.0, 0.0

    return scale, shift


# =============================
# Helper: Video Processing
# =============================
def resize_depth_back(depth_np, orig_size):
    orig_H, orig_W = orig_size
    depth_tensor = torch.from_numpy(depth_np).permute(0, 3, 1, 2).float()
    depth_tensor = F.interpolate(depth_tensor, size=(
        orig_H, orig_W), mode='bilinear', align_corners=False)
    return depth_tensor.permute(0, 2, 3, 1).cpu().numpy()


# =============================
# Core Inference
# =============================
def generate_depth_sliced(model, input_rgb, window_size=45, overlap=9, scale_only=False):
    B, T, C, H, W = input_rgb.shape
    depth_windows = get_window_index(T, window_size, overlap)
    print(f"depth_windows {depth_windows}")

    depth_res_list = []
    window_times = []

    # 1. Inference per window
    for start, end in tqdm(depth_windows, desc="Inferencing Slices"):
        _window_t0 = time.time()
        _input_rgb_slice = input_rgb[:, start:end]

        # Ensure 4n+1 padding
        _input_rgb_slice, origin_T = pad_time_mod4(_input_rgb_slice)
        _input_frame = _input_rgb_slice.shape[1]
        _input_height, _input_width = _input_rgb_slice.shape[-2:]

        outputs = model.pipe(
            prompt=[""] * B,
            negative_prompt=[""] * B,
            mode=model.args.mode,
            height=_input_height,
            width=_input_width,
            num_frames=_input_frame,
            batch_size=B,
            input_image=_input_rgb_slice[:, 0],
            extra_images=_input_rgb_slice,
            extra_image_frame_index=torch.ones(
                [B, _input_frame]).to(model.pipe.device),
            input_video=_input_rgb_slice,
            cfg_scale=1,
            seed=0,
            tiled=False,
            denoise_step=model.args.denoise_step,
        )
        # Drop the padded frames
        depth_res_list.append(outputs['depth'][:, :origin_T])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        window_times.append(time.time() - _window_t0)

    if window_times:
        print(f"[bench] {len(window_times)} windows, "
              f"mean {sum(window_times) / len(window_times):.3f} s/window, "
              f"total {sum(window_times):.1f} s")

    # 2. Overlap Alignment
    depth_list_aligned = None
    prev_end = None

    for i, (t, (start, end)) in enumerate(zip(depth_res_list, depth_windows)):
        print(f"Handling window {i} start: {start}, end: {end}")

        if i == 0:
            depth_list_aligned = t
            prev_end = end
            continue

        curr_start = start
        real_overlap = prev_end - curr_start

        if real_overlap > 0:
            ref_frames = depth_list_aligned[:, -real_overlap:]
            curr_frames = t[:, :real_overlap]

            if scale_only:
                scale = np.sum(curr_frames * ref_frames) / \
                    (np.sum(curr_frames * curr_frames) + 1e-6)
                shift = 0.0
            else:
                scale, shift = compute_scale_and_shift(curr_frames, ref_frames)

            scale = np.clip(scale, 0.7, 1.5)

            aligned_t = t * scale + shift
            aligned_t[aligned_t < 0] = 0

            # Debugging Output
            curr_overlap_aligned = aligned_t[:, :real_overlap]
            diff = np.abs(curr_overlap_aligned - ref_frames)
            mae_scalar = float(
                diff.mean(axis=tuple(range(1, diff.ndim))).mean())

            print(f"\n[Overlap {i}]")
            print(f"real_overlap = {real_overlap}")
            print(f"scale = {scale:.8f}, shift = {shift:.8f}")
            print(
                f"aligned curr range = {aligned_t.min():.6f} ~ {aligned_t.max():.6f}")
            print(f"overlap MAE(after align) = {mae_scalar:.6f}")

            # Smooth blending
            alpha = np.linspace(0, 1, real_overlap, dtype=np.float32).reshape(
                1, real_overlap, 1, 1, 1)
            smooth_overlap = (1 - alpha) * ref_frames + \
                alpha * aligned_t[:, :real_overlap]

            depth_list_aligned = np.concatenate(
                [depth_list_aligned[:, :-real_overlap], smooth_overlap,
                 aligned_t[:, real_overlap:]], axis=1
            )
        else:
            # Fallback if no overlap exists
            depth_list_aligned = np.concatenate(
                [depth_list_aligned, t], axis=1)

        print(
            f"Total depth range after concat = {depth_list_aligned.min():.6f} ~ {depth_list_aligned.max():.6f}")
        prev_end = end

    # Crop to original length
    return depth_list_aligned[:, :T]


# =============================
# Pipeline Components
# =============================
def load_model(ckpt_dir, model_config_path, dit_sd=None):
    """Initializes and loads the model checkpoint."""
    from quant_script.common import build_model
    model = build_model(ckpt_dir, model_config_path, dit_sd=dit_sd)
    model = model.to("cuda")
    return model


def load_video_data(args):
    """Loads and resizes the input video."""
    input_tensor, origin_fps = read_video(args.input_video)
    print("Original shape:", input_tensor.shape)

    input_tensor, orig_size = resize_for_training_scale(
        input_tensor, args.height, args.width)
    print("Resized shape:", input_tensor.shape)
    print(f"input range {input_tensor.min()} - {input_tensor.max()}")

    return input_tensor, orig_size, origin_fps


def predict_depth(model, input_tensor, orig_size, args):
    """Runs depth prediction and post-processes the output to original size."""
    depth = generate_depth_sliced(
        model, input_tensor, args.window_size, args.overlap)[0]
    print(f"depth range shape {depth.min()} - {depth.max()}, shape {depth.shape}")

    # Post Process: resize back to original
    depth = resize_depth_back(depth, orig_size)
    print(f"after resizing {depth.min()} - {depth.max()}, {depth.shape}")

    return depth


def save_results(depth, origin_fps, args):
    """Normalizes and saves the depth video to disk."""
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.input_video).split('.')[0]
    gray_scale = 'gray' if args.grayscale else 'color'
    out_prefix = os.path.join(
        args.output_dir, f"{base_name}_{gray_scale}")

    if args.save_npy:
        npy_path = f"{out_prefix}_depth.npy"
        np.save(npy_path, depth)
        print(f"Saved raw depth to {npy_path}")

    output_path = f"{out_prefix}_depth_vis.mp4"
    print(f"Saving to {output_path}")
    d_min, d_max = depth.min(), depth.max()
    vis_depth = (depth - d_min) / (d_max - d_min + 1e-8)
    
    save_video(vis_depth, output_path,
               fps=origin_fps, quality=6, grayscale=args.grayscale)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input_video", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./inference_results")
    parser.add_argument('--model_config', default='ckpt/model_config.yaml')
    parser.add_argument("--window_size", type=int, default=81)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument("--overlap", type=int, default=9)
    parser.add_argument('--grayscale', action='store_true')
    parser.add_argument("--dit_sd", type=str, default=None,
                        help="Load a merged LoRA-free DiT state dict instead of ckpt/model.safetensors")
    parser.add_argument("--save_npy", action="store_true",
                        help="Also dump raw depth (T,H,W,3) as .npy next to the mp4")
    parser.add_argument("--pseudo_quant", nargs="?", const="w4g128", default=None,
                        help="Fake-quantize DiT block linears after loading; optional "
                             "spec like w4g128, w4g64, w8g128 (default if flag given "
                             "with no value: w4g128)")
    parser.add_argument("--awq_results", type=str, default=None,
                        help="Apply AWQ scales/clips (from run_awq_search.py) before pseudo-quant")
    parser.add_argument("--awq_ckpt", type=str, default=None,
                        help="Load a real packed low-bit AWQ DiT checkpoint (from "
                             "quant_script/pack_awq.py) after loading --dit_sd; "
                             "auto-detects the packed checkpoint format")
    return parser.parse_args()


# =============================
# Main Script
# =============================
def main():
    args = parse_args()

    # 1. Load Model
    model = load_model(args.ckpt, args.model_config, dit_sd=args.dit_sd)

    # 1a. Optional real packed low-bit checkpoint (built via quant_script/pack_awq.py
    # on top of --dit_sd; replaces the just-loaded bf16 weights in place). Mutually
    # exclusive in practice with --awq_results/--pseudo_quant (the packed checkpoint
    # already has AWQ applied and the weights quantized).
    assert not (args.awq_ckpt and (args.awq_results or args.pseudo_quant)), \
        "--awq_ckpt loads an already-quantized checkpoint; do not combine it with --awq_results/--pseudo_quant (would double-quantize)."
    if args.awq_ckpt:
        from quant_script.load_awq import is_packed_awq_checkpoint, load_awq_dit
        if is_packed_awq_checkpoint(args.awq_ckpt):
            load_awq_dit(model.pipe.dit, args.awq_ckpt)
        else:
            raise ValueError(
                f"{args.awq_ckpt} is not a recognized packed AWQ checkpoint format "
                "(expected Tier-A dvd_awq_packed_v1 metadata)")
        model.pipe.dit.cuda()
        print(f"loaded AWQ checkpoint {args.awq_ckpt}")

    # 1b. Optional quantization (model is still on CUDA at this point)
    if args.awq_results:
        from awq.quantize.pre_quant import apply_awq
        apply_awq(model.pipe.dit, torch.load(args.awq_results, map_location="cpu"))
        print(f"applied AWQ results from {args.awq_results}")
    if args.pseudo_quant:
        from quant_script.awq_dit import pseudo_quantize_dit
        m = re.match(r"^w(\d+)g(\d+)$", args.pseudo_quant)
        assert m, f"--pseudo_quant spec must look like w4g128, got {args.pseudo_quant!r}"
        w_bit, group_size = int(m.group(1)), int(m.group(2))
        pseudo_quantize_dit(model.pipe.dit, w_bit=w_bit,
                            q_config=dict(zero_point=True, q_group_size=group_size))
    if args.awq_results or args.pseudo_quant:
        # apply_awq (apply_scale/apply_clip) and pseudo_quantize_dit each cycle
        # the touched submodules through .cuda()/.cpu() per-layer (a pattern
        # inherited from llm-awq's memory-frugal block-by-block search, where
        # only the block currently being processed lives on GPU). Since our
        # model is already fully resident on CUDA (load_model moved it there),
        # that .cpu() tail-call stranded 510-600 of the ~825-855 DiT params on
        # CPU while the rest of the model stayed on GPU, causing a device
        # mismatch on the next forward pass. Moving the whole DiT back to CUDA
        # in one shot after quantization fixes this without touching the
        # vendored llm-awq quantization code.
        model.pipe.dit.cuda()

    # 2. Load Video
    input_tensor, orig_size, origin_fps = load_video_data(args)

    # 3. Predict Depth (peak-VRAM window starts here, i.e. excludes one-time
    # model/checkpoint loading, so bf16 vs w4/w8 runs are comparable)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    depth = predict_depth(model, input_tensor, orig_size, args)
    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"[bench] peak VRAM during inference: {peak_gb:.2f} GB")

    # 4. Save Results
    save_results(depth, origin_fps, args)

    print("Inference completed successfully!")


if __name__ == "__main__":
    main()