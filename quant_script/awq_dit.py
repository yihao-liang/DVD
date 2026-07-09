import functools
import gc
from collections import defaultdict

import torch
import torch.nn as nn
import tqdm

import quant_script.common  # noqa: F401  (installs llm-awq sys.path)
from awq.quantize.auto_clip import apply_clip, auto_clip_block
from awq.quantize.auto_scale import apply_scale, auto_scale_block
from awq.quantize.quantizer import pseudo_quantize_tensor
from awq.utils.module import append_str_prefix, get_op_name

DEFAULT_Q_CONFIG = dict(zero_point=True, q_group_size=128)


def get_named_linears(module):
    return {n: m for n, m in module.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def run_awq_dit(dit, calib_samples, w_bit=4, q_config=DEFAULT_Q_CONFIG,
                auto_scale=True, mse_range=True, token_stride=4):
    """AWQ search over WanModel DiT blocks.
    calib_samples: list of dicts with keys x, context, t_mod, freqs (CPU tensors).
    token_stride: subsample cached linear-input tokens to bound memory."""
    inps = [s["x"].clone() for s in calib_samples]
    kwargs_list = [dict(context=s["context"], t_mod=s["t_mod"], freqs=s["freqs"]) for s in calib_samples]

    def cache_hook(m, x, y, name, feat_dict):
        t = x[0].detach()
        feat_dict[name].append(t.flatten(0, -2)[::token_stride].cpu())

    awq_results = {"scale": [], "clip": []}
    blocks = dit.blocks
    for i in tqdm.tqdm(range(len(blocks)), desc="AWQ over DiT blocks"):
        block = blocks[i].cuda()
        named_linears = get_named_linears(block)

        input_feat = defaultdict(list)
        handles = [
            named_linears[n].register_forward_hook(
                functools.partial(cache_hook, name=n, feat_dict=input_feat))
            for n in named_linears
        ]
        for j in range(len(inps)):
            kw = {k: v.cuda() for k, v in kwargs_list[j].items()}
            inps[j] = block(inps[j].cuda(), **kw).cpu()
        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        torch.cuda.empty_cache()

        if auto_scale:
            scales_list = auto_scale_block(block, {}, w_bit=w_bit, q_config=q_config,
                                           input_feat=input_feat)
            apply_scale(blocks[i], scales_list, input_feat_dict=input_feat)
            awq_results["scale"] += append_str_prefix(scales_list, get_op_name(dit, block) + ".")
        if mse_range:
            clip_list = auto_clip_block(block, w_bit=w_bit, q_config=q_config,
                                        input_feat=input_feat)
            apply_clip(block, clip_list)
            awq_results["clip"] += append_str_prefix(clip_list, get_op_name(dit, block) + ".")

        blocks[i] = block.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()
    return awq_results


@torch.no_grad()
def pseudo_quantize_dit(dit, w_bit=4, q_config=DEFAULT_Q_CONFIG):
    """Fake-quantize every Linear inside DiT blocks (weights only). Embeddings/head untouched."""
    n = 0
    for block in dit.blocks:
        for _, m in get_named_linears(block).items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bit=w_bit, **q_config)
            m.cpu()
            n += 1
    print(f"pseudo-quantized {n} linears to w{w_bit} g{q_config['q_group_size']}")
    return dit
