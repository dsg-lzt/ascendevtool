#!/usr/bin/env python3
"""FPS test — find correct torch_npu custom op API"""
import torch

def cpu_fps(xyz, npoint):
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long)
    distance = torch.ones(B, N) * 1e10
    farthest = torch.zeros(B, dtype=torch.long)
    batch_indices = torch.arange(B, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.to(torch.int32)


def discover_api():
    import torch_npu

    # Find all torch_npu attributes containing 'op' or 'run' or 'custom' or 'npu'
    for k in sorted(dir(torch_npu)):
        if any(w in k.lower() for w in ['op', 'run', 'custom', 'npu', 'function', 'contrib']):
            obj = getattr(torch_npu, k)
            if isinstance(obj, type):
                print(f"  torch_npu.{k}: class")
            elif callable(obj):
                print(f"  torch_npu.{k}: function")
            else:
                print(f"  torch_npu.{k}: {type(obj).__name__}")

    # Check submodules
    for sub in ['_C', 'contrib', 'npu_ops']:
        mod = getattr(torch_npu, sub, None)
        if mod is None: continue
        items = [k for k in dir(mod) if not k.startswith('_')]
        print(f"\n  torch_npu.{sub} [{len(items)} items]: {items[:30]}")

    # Check the torch.ops namespace for our operator
    ns = torch.ops.pointnet2__ext_furthest_point_sampling
    print(f"\n  torch.ops.pointnet2__ext_furthest_point_sampling:")
    print(f"    type: {type(ns)}")
    print(f"    dict: {ns.__dict__ if hasattr(ns, '__dict__') else 'N/A'}")

    print("\n  Op built + installed in CANN. Needs proper PyTorch binding.")
    exit(0)


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    discover_api()
