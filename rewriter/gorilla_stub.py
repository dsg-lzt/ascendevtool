from __future__ import annotations
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import torch
import torch_npu
import os
import re

torch.npu.set_compile_mode(jit_compile=False)
# 允许 NPU 内部格式转换，避免 scaled_dot_product_attention 报错
if hasattr(torch.npu, 'config') and hasattr(torch.npu.config, 'allow_internal_format'):
    torch.npu.config.allow_internal_format = True

# 禁用 flash attention，避免 NPU scaled_dot_product_attention 报错
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)

class _CfgObj:
    pass

def _dict_to_obj(d):
    if isinstance(d, dict):
        obj = _CfgObj()
        for k, v in d.items():
            setattr(obj, k, _dict_to_obj(v))
        return obj
    return d

class _GorillaConfig:
    @staticmethod
    def fromfile(path):
        import yaml
        from addict import Dict
        import os
        if not os.path.isfile(path):
            base = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(base, path)
        with open(path, 'r') as f:
            cfg_dict = yaml.safe_load(f)
        return Dict(cfg_dict)

class _GorillaUtils:
    @staticmethod
    def set_cuda_visible_devices(gpu_ids):
        devices = gpu_ids.split(",") if isinstance(gpu_ids, str) else [str(g) for g in gpu_ids]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        os.environ["ASCEND_VISIBLE_DEVICES"] = "0"
        if torch.npu.is_available():
            torch.npu.set_device(0)

class _GorillaSolver:
    @staticmethod
    def load_checkpoint(model, filename, map_location=None):
        checkpoint = torch.load(filename, map_location=map_location or "cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state = checkpoint["model_state_dict"]
            else:
                state = checkpoint
        else:
            state = checkpoint
        model.load_state_dict(state, strict=False)

import sys

class _GorillaModule(sys.modules[__name__].__class__):
    Config = _GorillaConfig
    utils = _GorillaUtils
    solver = _GorillaSolver

sys.modules[__name__].__class__ = _GorillaModule
Config = _GorillaConfig
utils = _GorillaUtils
solver = _GorillaSolver
