from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class OpSolution:
    op_name: str
    strategy: str
    solution: str
    replacement_code: Optional[str] = None
    local_module: Optional[str] = None
    local_func: Optional[str] = None


def _cann_resource_dir() -> Path:
    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH") or ""
    if ascend_home:
        return Path(ascend_home) / "tools" / "ms_fmk_transplt" / "resource"
    return Path("/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/resource")


def load_supported_ops(torch_version: str = "2.6") -> Set[str]:
    path = _cann_resource_dir() / f"supported_op_{torch_version}.json"
    if not path.is_file():
        return set()
    with open(path, "r") as f:
        data = json.load(f)
    op_list = data.get("op_list", data)
    return set(op_list.keys())


_DECOMPOSE_RULES: Dict[str, str] = {
    "pointnet2._ext.gather_points": """
def ascend_gather_points(xyz, idx):
    batch_size, num_dims, _ = xyz.size()
    _, num_points, nsamples = idx.size()
    idx_expanded = idx.long().unsqueeze(1).expand(-1, num_dims, -1, -1)
    xyz_expanded = xyz.unsqueeze(3).expand(-1, -1, -1, nsamples)
    gathered = torch.gather(xyz_expanded, 2, idx_expanded)
    return gathered.contiguous()
""",
    "pointnet2._ext.group_points": """
def ascend_group_points(xyz, idx):
    batch_size, num_dims, num_points = xyz.size()
    _, nsamples, npoints = idx.size()
    idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
    idx = idx.long() + idx_base
    idx = idx.view(-1)
    xyz = xyz.transpose(2, 1).contiguous()
    grouped = xyz.view(batch_size * num_points, -1)[idx, :]
    grouped = grouped.view(batch_size, nsamples, npoints, num_dims)
    grouped = grouped.permute(0, 3, 1, 2).contiguous()
    return grouped
""",
    "torch.nn.DataParallel": """
import torch.nn as nn


def ascend_data_parallel(model, device_ids=None, output_device=None):
    if device_ids is None:
        device_ids = list(range(torch.npu.device_count()))
    if not device_ids:
        return model
    model = model.to(f'npu:{device_ids[0]}')
    if len(device_ids) == 1:
        return model
    return nn.DataParallel(model, device_ids=device_ids, output_device=output_device)
""",
}

_MATH_EQUIVALENCE_RULES: Dict[str, str] = {
    "torch.nn.functional.glu": "x.chunk(2, dim=-1)[0] * torch.sigmoid(x.chunk(2, dim=-1)[1])",
}


def decompose_op(op_name: str) -> Optional[str]:
    return _DECOMPOSE_RULES.get(op_name)


def get_math_equivalent(op_name: str) -> Optional[str]:
    return _MATH_EQUIVALENCE_RULES.get(op_name)
