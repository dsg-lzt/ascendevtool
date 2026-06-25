import os
import json
import tempfile
import warnings
from typing import Optional
from argparse import Namespace
import yaml
from addict import Dict

# 纯 PyTorch 实现 scaled_dot_product_attention（替代 NPU 不支持的融合版本）
import torch
import torch.nn.functional as F

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

BASE_KEY = "_base_"
RESERVED_KEYS = ["filename", "text"]


class ConfigDict(Dict):
    r"""ConfigDict based on Dict, which use to convert the config
        file into config dict
    """
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                f"`{self.__class__.__name__}` object has no attribute `{name}`"
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex


class Config:
    @staticmethod
    def fromfile(filename: str):
        cfg_dict = Config._file2dict(filename)
        return ConfigDict(cfg_dict)

    @staticmethod
    def _file2dict(filename: str):
        with open(filename, "r") as f:
            content = f.read()
        cfg_dict = yaml.safe_load(content)
        if cfg_dict is None:
            return {}
        if BASE_KEY in cfg_dict:
            base_dir = os.path.dirname(filename)
            base_files = cfg_dict[BASE_KEY]
            if isinstance(base_files, str):
                base_files = [base_files]
            for bf in base_files:
                bf_path = os.path.join(base_dir, bf)
                base_cfg = Config._file2dict(bf_path)
                cfg_dict = Config._merge_dict(base_cfg, cfg_dict)
            del cfg_dict[BASE_KEY]
        return cfg_dict

    @staticmethod
    def _merge_dict(base, override):
        result = base.copy()
        for k, v in override.items():
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                result[k] = Config._merge_dict(result[k], v)
            else:
                result[k] = v
        return result
