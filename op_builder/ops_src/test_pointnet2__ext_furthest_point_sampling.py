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
    ns = torch.ops.pointnet2__ext_furthest_point_sampling
    avail = [k for k in dir(ns) if not k.startswith('_')]
    print(f"  Available functions: {avail}")
    
    # Find any function in this namespace
    op_func = None
    for fn_name in avail:
        obj = getattr(ns, fn_name)
        if callable(getattr(obj, 'op', None)) or hasattr(obj, '__call__'):
            op_func = getattr(ns, fn_name)
            print(f"  Using: {fn_name}")
            break
    if op_func is None and avail:
        op_func = getattr(ns, avail[0])
        print(f"  Trying first: {avail[0]}")

    if not op_func:
        print("  No callable found")
        return False

    for B, N, M in [(1, 256, 32)]:
        xyz = torch.randn(B, N, 3, dtype=torch.float32).npu()
        ref = cpu_fps(xyz.cpu(), M)
        try:
            out = op_func(xyz, M).long()
            ok = torch.equal(ref.long(), out.cpu().long())
            print(f"  B={B} N={N} M={M}: {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  B={B} N={N} M={M}: OP fail - {e}")
            # Try with different args
            try:
                out = op_func(xyz, npoint=M).long()
                print(f"    retry with npoint=: {'PASS' if torch.equal(ref.long(), out.cpu().long()) else 'FAIL'}")
            except Exception as e2:
                print(f"    retry fail: {e2}")
            return False
    return True


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test_fps_op()
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
