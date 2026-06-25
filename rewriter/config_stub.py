import os
import json
import tempfile
import warnings
from typing import Optional
from argparse import Namespace
import yaml
from addict import Dict

# NPU 原生 attention 替换 scaled_dot_product_attention（解决 can not cast format 报错）
import torch
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)

_orig_sdpa = torch.nn.functional.scaled_dot_product_attention

def _npu_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if query.device.type == 'npu':
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()
        m = attn_mask
        if m is not None and m.dim() == 2:
            m = m.unsqueeze(0)
        out, _ = torch_npu.npu_fused_infer_attention_score(
            q, k, v,
            num_heads=q.size(2),
            input_layout="BSH",
            scale=scale if scale is not None else q.size(-1) ** -0.5,
            atten_mask=m,
        )
        return out.transpose(1, 2).contiguous()
    return _orig_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale)

torch.nn.functional.scaled_dot_product_attention = _npu_sdpa

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
