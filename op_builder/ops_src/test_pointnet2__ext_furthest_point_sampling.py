#!/usr/bin/env python3
"""FPS Ascend C operator test"""
import torch
import numpy as np

def cpu_fps(xyz, npoint):
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
    for B, N, M in [(1, 256, 32), (2, 512, 64), (4, 128, 16)]:
        xyz = torch.randn(B, N, 3, dtype=torch.float32).npu()
        ref = cpu_fps(xyz.cpu(), M)

        try:
            out = torch.ops.pointnet2__ext_furthest_point_sampling.pointnet2__ext_furthest_point_sampling(xyz, npoint=M)
            ok = True
        except Exception as e:
            print(f"  B={B} N={N} M={M}: OP fail - {e}")
            return False

        ok = torch.equal(ref.long(), out.cpu().long())
        print(f"  B={B} N={N} M={M}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"    ref[:5]={ref[0][:5].tolist()} out[:5]={out.cpu()[0][:5].tolist()}")
            return False
    return True


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test_fps_op()
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
