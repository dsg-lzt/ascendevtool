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
    batch_size, num_dims, num_points = xyz.size()
    idx_2d = idx.dim() == 2
    if idx_2d:
        idx = idx.unsqueeze(1).unsqueeze(-1)
    else:
        idx = idx.unsqueeze(-1)
    idx_expanded = idx.long().expand(-1, num_dims, -1, -1)
    xyz_expanded = xyz.unsqueeze(3).expand(-1, -1, -1, idx_expanded.size(-1))
    gathered = torch.gather(xyz_expanded, 2, idx_expanded)
    if idx_2d:
        gathered = gathered.squeeze(-1)
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
def ascend_data_parallel(model, device_ids=None, output_device=None):
    try:
        import torch_npu
        if torch.npu.is_available() and torch.npu.device_count() > 0:
            model = model.to('npu:0')
            return model
    except Exception:
        pass
    return model.to('cpu')
""",
    "pointnet2._ext.furthest_point_sampling": """
def ascend_furthest_point_sampling(xyz, npoint):
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids
""",
    "pointnet2._ext.three_nn": """
def ascend_three_nn(unknown, known):
    B, N, _ = unknown.shape
    _, M, _ = known.shape
    dists = torch.zeros(B, N, 3, device=unknown.device)
    idx = torch.zeros(B, N, 3, dtype=torch.long, device=unknown.device)
    for b in range(B):
        u = unknown[b]
        k = known[b]
        d = torch.cdist(u, k)
        d_topk, i_topk = torch.topk(d, 3, dim=1, largest=False)
        dists[b] = d_topk
        idx[b] = i_topk
    return dists, idx
""",
    "pointnet2._ext.three_interpolate": """
def ascend_three_interpolate(features, idx, weight):
    B, C, M = features.shape
    _, N, _ = idx.shape
    out = torch.zeros(B, C, N, device=features.device)
    for b in range(B):
        for n in range(N):
            w = weight[b, n]
            for k in range(3):
                out[b, :, n] += features[b, :, idx[b, n, k]] * w[k]
    return out
""",
    "pointnet2._ext.ball_query": """
def ascend_ball_query(new_xyz, xyz, radius, nsample):
    B, M, _ = new_xyz.shape
    _, N, _ = xyz.shape
    idx = torch.zeros(B, M, nsample, dtype=torch.long, device=xyz.device)
    for b in range(B):
        for m in range(M):
            d = torch.sum((xyz[b] - new_xyz[b, m:m+1]) ** 2, dim=1)
            d_n = d / (torch.max(d) + 1e-6)
            valid = d < radius * radius
            valid_idx = torch.nonzero(valid).squeeze(1)
            if valid_idx.numel() == 0:
                idx[b, m, :] = 0
            elif valid_idx.numel() >= nsample:
                rand_choice = valid_idx[torch.randperm(valid_idx.numel())[:nsample]]
                idx[b, m, :] = rand_choice
            else:
                tile = valid_idx.repeat((nsample + valid_idx.numel() - 1) // valid_idx.numel())
                idx[b, m, :] = tile[:nsample]
    return idx
""",
    "pointnet2._ext.gather_points_grad": """
def ascend_gather_points_grad(grad_out, idx, N):
    B, C, M = grad_out.shape
    grad_points = grad_out.new_zeros(B, C, N)
    grad_points.scatter_add_(2, idx.unsqueeze(1).expand(-1, C, -1), grad_out)
    return grad_points
""",
    "pointnet2._ext.group_points_grad": """
def ascend_group_points_grad(grad_out, idx, N):
    B, C, nsample, npoint = grad_out.shape
    grad_xyz = grad_out.new_zeros(B, C, N)
    grad_xyz = grad_xyz.view(B, C, N)
    idx = idx.long()
    for b in range(B):
        for s in range(nsample):
            grad_xyz[b].scatter_add_(1, idx[b, s:s+1].expand(C, -1), grad_out[b, :, s, :])
    return grad_xyz
""",
    "pointnet2._ext.three_interpolate_grad": """
def ascend_three_interpolate_grad(grad_out, idx, weight, M):
    B, C, N = grad_out.shape
    grad_features = grad_out.new_zeros(B, C, M)
    for b in range(B):
        for n in range(N):
            w = weight[b, n]
            for k in range(3):
                grad_features[b, :, idx[b, n, k]] += grad_out[b, :, n] * w[k]
    return grad_features
""",
}

_MATH_EQUIVALENCE_RULES: Dict[str, str] = {
    "torch.nn.functional.glu": "x.chunk(2, dim=-1)[0] * torch.sigmoid(x.chunk(2, dim=-1)[1])",
}


def decompose_op(op_name: str) -> Optional[str]:
    return _DECOMPOSE_RULES.get(op_name)


def get_math_equivalent(op_name: str) -> Optional[str]:
    return _MATH_EQUIVALENCE_RULES.get(op_name)
