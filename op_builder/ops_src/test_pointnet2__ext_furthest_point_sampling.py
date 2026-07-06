#!/usr/bin/env python3
"""FPS (Furthest Point Sampling) Ascend C 算子测试"""
import torch
import numpy as np
import time

def cpu_fps(xyz, npoint):
    """PyTorch FPS 参考实现"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.to(torch.int32)


def test_fps_op():
    for B, N, M in [(2, 128, 32), (2, 512, 64), (4, 256, 48)]:
        xyz = torch.randn(B, N, 3, dtype=torch.float32).npu()
        npt = torch.tensor([M], dtype=torch.int32).npu()

        ref = cpu_fps(xyz.cpu(), M)

        try:
            out = torch.ops.pointnet2__ext_furthest_point_sampling.pointnet2__ext_furthest_point_sampling(xyz, npt)
            ok = True
        except Exception as e:
            print(f"  OP call failed: {e}")
            ok = False

        if ok:
            ok = torch.allclose(ref.long(), out.cpu().long(), atol=0)
            if not ok:
                mismatch = (ref.long() != out.cpu().long()).sum().item()
                print(f"  Mismatch: {mismatch} / {B*M}")

        print(f"  B={B} N={N} M={M}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            return False
    return True


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test_fps_op()
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
