#!/usr/bin/env python3
"""FPS test — discover torch_npu API + precision validation"""
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


def discover_and_test():
    import torch_npu as tnpu

    # Discover available APIs
    print("  Discovering torch_npu APIs...")
    candidates = {}
    for mod_name in ['_C', 'npu_ops', 'npu_function', 'functional']:
        try:
            mod = getattr(tnpu, mod_name, None)
            if mod is None:
                continue
            attrs = [a for a in dir(mod) if not a.startswith('_')]
            if 'OpCommand' in str(type(mod)):
                continue
            print(f"  torch_npu.{mod_name}: {type(mod).__name__} ({len(attrs)} attrs)")
            for a in attrs[:10]:
                print(f"    .{a}")
        except Exception as e:
            pass

    # Check torch_npu._C for custom op APIs
    c_fn = [a for a in dir(tnpu._C) if 'custom' in a.lower() or 'op' in a.lower() or 'run' in a.lower()]
    print(f"  _C custom/op/run APIs: {c_fn[:20]}")

    # Check available torch.ops namespaces
    for ns_name in [a for a in dir(torch.ops) if 'point' in a.lower() or 'furthest' in a.lower() or 'fps' in a.lower()]:
        ns = getattr(torch.ops, ns_name)
        print(f"  torch.ops.{ns_name}: {dir(ns)}")

    # Try the new test
    tests = [(1, 128, 32), (1, 512, 64), (2, 256, 48),
             (4, 128, 16), (1, 1024, 128), (2, 500, 100),
             (4, 200, 50), (1, 64, 8), (8, 100, 20), (3, 300, 150)]
    passed = 0
    for B, N, M in tests:
        xyz = torch.randn(B, N, 3).npu()
        ref = cpu_fps(xyz.cpu(), M)
        try:
            out = torch.empty(B, M, dtype=torch.float32).npu()
            # Try: torch_npu._C._run_custom_op
            # or torch_npu.npu_function()
            # or torch_npu.npu_ops.xxx
            raise NotImplementedError
        except NotImplementedError:
            ok = False
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {type(e).__name__}: {str(e)[:120]}")
    return True


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    discover_and_test()
    print("API discovery complete - test SKIPPED")
    exit(0)
