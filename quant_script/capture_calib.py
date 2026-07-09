import argparse
import os

import torch

from quant_script.common import (build_model, get_window_index, pad_time_mod4,
                                 read_video, resize_for_training_scale)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", nargs="+", required=True)
    p.add_argument("--ckpt", default="ckpt")
    p.add_argument("--model_config", default="ckpt/model_config.yaml")
    p.add_argument("--dit_sd", default="ckpt/dit_merged.safetensors")
    p.add_argument("--out_dir", default="calib")
    p.add_argument("--window_size", type=int, default=81)
    p.add_argument("--overlap", type=int, default=21)
    p.add_argument("--max_windows_per_video", type=int, default=4)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=640)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    model = build_model(args.ckpt, args.model_config, dit_sd=args.dit_sd).to("cuda")

    captured = []

    def pre_hook(module, hook_args):
        x, context, t_mod, freqs = hook_args
        captured.append({
            "x": x.detach().cpu(), "context": context.detach().cpu(),
            "t_mod": t_mod.detach().cpu(), "freqs": freqs.detach().cpu(),
        })

    handle = model.pipe.dit.blocks[0].register_forward_pre_hook(pre_hook)

    for vid in args.videos:
        name = os.path.splitext(os.path.basename(vid))[0]
        video, _fps = read_video(vid)
        video, _orig_size = resize_for_training_scale(video, args.height, args.width)
        windows = get_window_index(video.shape[1], args.window_size, args.overlap)[: args.max_windows_per_video]
        for wi, (s, e) in enumerate(windows):
            clip, _origin_T = pad_time_mod4(video[:, s:e])
            captured.clear()
            model.pipe(
                prompt=[""], negative_prompt=[""], mode=model.args.mode,
                height=clip.shape[-2], width=clip.shape[-1], num_frames=clip.shape[1],
                batch_size=1, input_image=clip[:, 0], extra_images=clip,
                extra_image_frame_index=torch.ones([1, clip.shape[1]]).to(model.pipe.device),
                input_video=clip, cfg_scale=1, seed=0, tiled=False,
                denoise_step=model.args.denoise_step,
            )
            assert len(captured) == 1, f"expected 1 DiT pass per window, got {len(captured)}"
            out = os.path.join(args.out_dir, f"{name}_w{wi}.pt")
            torch.save(captured[0], out)
            shapes = {k: tuple(v.shape) for k, v in captured[0].items()}
            print(f"saved {out}: {shapes}")

    handle.remove()


if __name__ == "__main__":
    main()
