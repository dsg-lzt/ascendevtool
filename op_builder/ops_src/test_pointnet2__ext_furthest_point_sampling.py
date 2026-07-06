#!/usr/bin/env python3
"""FPS Ascend C operator test"""
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


def test_fps_op():
    import torch_npu
    for B, N, M in [(1, 256, 32), (2, 512, 64), (4, 128, 16)]:
        xyz = torch.randn(B, N, 3, dtype=torch.float32).npu()
        ref = cpu_fps(xyz.cpu(), M)

        try:
            # 尝试不同命名空间
            op_func = None
            for ns in [torch.ops.npu, torch.ops.pointnet2__ext_furthest_point_sampling]:
                for name in ['pointnet2__ext_furthest_point_sampling', 'farthest_point_sampling']:
                    if hasattr(ns, name):
                        op_func = getattr(ns, name)
                        break
                if op_func:
                    break
            if op_func is None:
                print(f"  B={B} N={N} M={M}: OP not found in torch.ops")
                print(f"  Available npu ops: {dir(torch.ops.npu)[:10]}...")
                return False
            out = op_func(xyz, M).long()
            ok = torch.equal(ref.long(), out.cpu().long())
        except Exception as e:
            print(f"  B={B} N={N} M={M}: OP fail - {e}")
            return False

        print(f"  B={B} N={N} M={M}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            return False
    return True


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test_fps_op()
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
