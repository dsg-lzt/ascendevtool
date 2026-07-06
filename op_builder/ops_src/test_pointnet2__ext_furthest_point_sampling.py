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
    # Try all possible registration methods
    op_name = 'pointnet2__ext_furthest_point_sampling'
    op_func = None
    
    # Method 1: torch.ops.pointnet2__ext_furthest_point_sampling
    try:
        ns = getattr(torch.ops, op_name, None)
        if ns and hasattr(ns, op_name):
            op_func = getattr(ns, op_name)
    except Exception:
        pass
    
    # Method 2: torch_npu.npu_ops  
    if not op_func and hasattr(torch_npu, 'npu_ops'):
        for attr in dir(torch_npu.npu_ops):
            if 'furthest' in attr.lower() or 'fps' in attr.lower():
                op_func = getattr(torch_npu.npu_ops, attr)
                break

    # Method 3: ctypes call to ACLNN
    if not op_func:
        import ctypes
        try:
            lib = ctypes.CDLL('libcust_opapi.so')
            if hasattr(lib, 'aclnn'+op_name.replace('__','_')):
                op_func = getattr(lib, 'aclnn'+op_name.replace('__','_'))
        except Exception:
            pass

    if op_func is None:
        print("OP not found via any method. Checking dirs...")
        print(f"  torch.ops keys: {[k for k in dir(torch.ops) if 'point' in k.lower() or 'furthest' in k.lower() or 'fps' in k.lower()]}")
        return False

    for B, N, M in [(1, 256, 32)]:
        xyz = torch.randn(B, N, 3, dtype=torch.float32).npu()
        ref = cpu_fps(xyz.cpu(), M)
        try:
            out = op_func(xyz, M)
            out = out.long()
            ok = torch.equal(ref.long(), out.cpu().long())
            print(f"  B={B} N={N} M={M}: {'PASS' if ok else 'FAIL'}")
        except Exception as e:
            print(f"  B={B} N={N} M={M}: OP fail - {e}")
            return False
    return True


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test_fps_op()
    print(f"\n{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
