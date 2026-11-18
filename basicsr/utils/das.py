"""Distribution-Aware Smoothing (DAS), Section III-C of DHQuant.

The equivalent channel rescaling follows SmoothQuant. In SwinIR, scales are
folded into norm1 -> qkv and norm2 -> fc1 so inference needs no extra operator.
"""

from functools import partial

import torch
from torch import nn

from basicsr.utils import get_root_logger


@torch.no_grad()
def collect_activation_scales(model, cali_data, device):
    """Collect maximum absolute inputs per channel in one calibration pass."""
    model.eval()
    act_scales = {}

    def record_inputs(module, inputs, output, name):
        tensor = inputs[0]
        maximum = tensor.reshape(-1, tensor.shape[-1]).abs().max(dim=0)[0].float().cpu()
        act_scales[name] = (torch.max(act_scales[name], maximum)
                            if name in act_scales else maximum)

    if isinstance(cali_data, dict):
        batches = [cali_data['lq']]
    elif isinstance(cali_data, torch.Tensor):
        batches = [cali_data]
    elif isinstance(cali_data, (list, tuple)):
        batches = cali_data
    else:
        raise TypeError('DAS calibration expects an image tensor, an lq dict, or tensor batches.')

    hooks = []
    try:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and (name.endswith('qkv') or name.endswith('fc1')):
                hooks.append(module.register_forward_hook(partial(record_inputs, name=name)))
        for batch in batches:
            model(batch.to(device))
    finally:
        for hook in hooks:
            hook.remove()
    return act_scales


@torch.no_grad()
def smooth_layer_norm_and_linears(ln, fcs, act_scales, alpha=0.5):
    """Apply Eqs. (3)-(5), preserving Linear(LayerNorm(x)) and Linear bias."""
    if not 0 <= alpha <= 1:
        raise ValueError('DAS alpha must lie in [0, 1].')
    if not isinstance(fcs, (list, tuple)):
        fcs = [fcs]
    if not isinstance(ln, nn.LayerNorm):
        raise TypeError('DAS folding requires an affine LayerNorm.')
    for fc in fcs:
        if not isinstance(fc, nn.Linear) or ln.weight.numel() != fc.in_features:
            raise ValueError('LayerNorm and Linear channel dimensions must agree.')
        if fc.in_features != act_scales.numel():
            raise ValueError('DAS statistics do not match the Linear input channels.')

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    act_scales = act_scales.to(device=device, dtype=dtype)
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    ).max(dim=0)[0].clamp(min=1e-5)
    scales = (act_scales.pow(alpha) / weight_scales.pow(1 - alpha)).clamp(min=1e-5)

    ln.weight.div_(scales)
    if ln.bias is not None:
        ln.bias.div_(scales)
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))
    return scales


@torch.no_grad()
def apply_das(model, cali_data, device, alpha=0.5):
    """Smooth a full-precision SwinIR student before HCG initialization."""
    from basicsr.archs.swinir_arch import SwinTransformerBlock

    act_scales = collect_activation_scales(model, cali_data, device)
    count = 0
    for name, module in model.named_modules():
        if not isinstance(module, SwinTransformerBlock):
            continue
        for ln, fc, suffix in ((module.norm1, module.attn.qkv, 'attn.qkv'),
                               (module.norm2, module.mlp.fc1, 'mlp.fc1')):
            smooth_layer_norm_and_linears(ln, fc, act_scales[f'{name}.{suffix}'], alpha)
            count += 1
    if count == 0:
        raise ValueError('DAS found no SwinTransformerBlock in the backbone.')
    get_root_logger().info(f'DAS applied to {count} LayerNorm/Linear pairs, alpha={alpha}.')
    return count
