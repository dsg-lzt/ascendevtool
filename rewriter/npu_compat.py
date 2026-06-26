import torch
import torch.nn.functional as F

try:
    import torch_npu
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
except Exception:
    pass

_orig_cross = torch.cross

def _cross_npu(input, other, dim=-1):
    if input.device.type == 'npu':
        return torch.stack([
            input[..., 1] * other[..., 2] - input[..., 2] * other[..., 1],
            input[..., 2] * other[..., 0] - input[..., 0] * other[..., 2],
            input[..., 0] * other[..., 1] - input[..., 1] * other[..., 0],
        ], dim=dim)
    return _orig_cross(input, other, dim=dim)

torch.cross = _cross_npu

_orig_sdpa = F.scaled_dot_product_attention


def _manual_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if query.device.type != 'npu':
        return _orig_sdpa(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    L, S = query.size(-2), key.size(-2)
    s = scale if scale is not None else query.size(-1) ** -0.5
    attn = torch.matmul(query, key.transpose(-2, -1)) * s
    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        attn = attn + attn_mask
    if is_causal:
        causal_mask = torch.triu(torch.ones(L, S, device=query.device, dtype=torch.bool), diagonal=1)
        attn.masked_fill_(causal_mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, value)


F.scaled_dot_product_attention = _manual_sdpa
