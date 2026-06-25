from __future__ import annotations
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import torch
import torch_npu
import os
import re

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
        from mmcv import Config as _MMCVConfig
        return _MMCVConfig.fromfile(path)

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
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint, strict=False)

import sys

class _GorillaModule(sys.modules[__name__].__class__):
    Config = _GorillaConfig
    utils = _GorillaUtils
    solver = _GorillaSolver

sys.modules[__name__].__class__ = _GorillaModule
Config = _GorillaConfig
utils = _GorillaUtils
solver = _GorillaSolver
