#!/usr/bin/env python3
"""FPS test — use torch_npu OpCommand via TORCH_LIBRARY"""
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


def run_npu_fps(xyz, npoint):
    import torch_npu
    from torch_npu.contrib.function import npu_function
    from torch_npu.framework import OpCommand as op_cmd
    
    out = torch.empty(xyz.size(0), npoint, dtype=torch.float32, device=xyz.device)
    cmd = op_cmd()
    cmd.Name("pointnet2__ext_furthest_point_sampling")
    cmd.Input(xyz)
    cmd.Output(out)
    cmd.Attr("npoint", int(npoint))
    cmd.Run()
    return out.long()


def test():
    import torch_npu
    tests = [(1, 128, 32), (1, 512, 64), (2, 256, 48),
             (4, 128, 16), (1, 1024, 128), (2, 500, 100),
             (4, 200, 50), (1, 64, 8), (8, 100, 20), (3, 300, 150)]
    passed = 0
    for B, N, M in tests:
        xyz = torch.randn(B, N, 3).npu()
        ref = cpu_fps(xyz.cpu(), M)
        try:
            out = run_npu_fps(xyz, M)
            ok = torch.equal(ref, out.cpu())
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {'PASS' if ok else 'FAIL'}")
            if ok: passed += 1
            else:
                mismatch = (ref != out.cpu()).sum().item()
                print(f"    mismatch: {mismatch}/{B*M}")
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {type(e).__name__}: {str(e)[:100]}")
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)

if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test()
    print(f"{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
