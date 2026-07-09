import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from safetensors.torch import load_file

# Pre-cache peft's (lru_cache'd) autoawq-availability check BEFORE the vendored
# llm-awq/ path below is added to sys.path. llm-awq ships its own top-level
# `awq/` package (an unrelated research codebase, not the pip `autoawq`
# package peft actually probes for), which would otherwise satisfy
# importlib.util.find_spec("awq") and make peft.tuners.lora.awq.dispatch_awq
# try `from awq.modules.linear import WQLinear_GEMM` -> ModuleNotFoundError,
# breaking any LoRA injection (add_lora_to_model) inside build_model().
from peft.import_utils import is_auto_awq_available
is_auto_awq_available()

# vendored llm-awq (not pip-installed; see plan Global Constraints)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "llm-awq"))

# Fail-fast guard: ensure peft's lru_cache pre-priming still works. If peft drops
# the lru_cache, is_auto_awq_available() would find the vendored awq/ package and
# incorrectly return True, causing LoRA injection to crash with ModuleNotFoundError.
assert not is_auto_awq_available(), (
    "peft's is_auto_awq_available() flipped to True after adding vendored llm-awq "
    "to sys.path — its lru_cache pre-priming no longer works. peft would crash LoRA "
    "injection with 'No module named awq.modules'. Fix the workaround above."
)

from examples.wanvideo.model_training.WanTrainingModule import WanTrainingModule


def build_model(ckpt_dir, cfg_path, dit_sd=None):
    """Build the DVD model. With dit_sd, load a merged LoRA-free DiT state dict
    (no PEFT adapters are attached); otherwise replicate the stock load path."""
    yaml_args = OmegaConf.load(cfg_path)
    accelerator = Accelerator()
    model = WanTrainingModule(
        accelerator=accelerator,
        model_id_with_origin_paths=yaml_args.model_id_with_origin_paths,
        trainable_models=None,
        use_gradient_checkpointing=False,
        lora_rank=yaml_args.lora_rank,
        lora_base_model=None if dit_sd else yaml_args.lora_base_model,
        args=yaml_args,
    )
    if dit_sd is not None:
        sd = load_file(dit_sd, device="cpu")
        model.pipe.dit.load_state_dict(sd, strict=True)
    else:
        sd = load_file(os.path.join(ckpt_dir, "model.safetensors"), device="cpu")
        dit_state_dict = {k.replace("pipe.dit.", ""): v for k, v in sd.items() if "pipe.dit." in k}
        model.pipe.dit.load_state_dict(dit_state_dict, strict=True)
        model.merge_lora_layer()
    return model


def unwrap_lora(dit):
    """After merge_lora_layer(), replace every peft LoraLinear with its merged base nn.Linear."""
    from peft.tuners.lora.layer import Linear as LoraLinear
    replaced = 0
    for module in list(dit.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoraLinear):
                setattr(module, child_name, child.base_layer)
                replaced += 1
    return replaced


# ---- video preprocessing helpers (single source; moved verbatim from
# ---- test_script/test_single_video.py, which now imports them from here) ----

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    video_np = np.stack(frames)
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2).float() / 255.0
    return video_tensor.unsqueeze(0), fps  # [1, T, C, H, W], fps


def resize_for_training_scale(video_tensor, target_h=480, target_w=640):
    B, T, C, H, W = video_tensor.shape
    ratio = max(target_h / H, target_w / W)
    new_H = (int(np.ceil(H * ratio)) + 15) // 16 * 16
    new_W = (int(np.ceil(W * ratio)) + 15) // 16 * 16
    if new_H == H and new_W == W:
        return video_tensor, (H, W)
    video_reshape = video_tensor.view(B * T, C, H, W)
    resized = F.interpolate(video_reshape, size=(new_H, new_W), mode="bilinear", align_corners=False)
    return resized.view(B, T, C, new_H, new_W), (H, W)


def pad_time_mod4(video_tensor):
    """Pads the temporal dimension to satisfy 4n+1 requirement."""
    B, T, C, H, W = video_tensor.shape
    remainder = T % 4
    if remainder != 1:
        pad_len = (4 - remainder + 1) % 4
        pad_frames = video_tensor[:, -1:, :, :, :].repeat(1, pad_len, 1, 1, 1)
        video_tensor = torch.cat([video_tensor, pad_frames], dim=1)
    return video_tensor, T


def get_window_index(T, window_size, overlap):
    if T <= window_size:
        return [(0, T)]
    res = [(0, window_size)]
    start = window_size - overlap
    while start < T:
        end = start + window_size
        if end < T:
            res.append((start, end))
            start += window_size - overlap
        else:
            res.append((max(0, T - window_size), T))
            break
    return res
