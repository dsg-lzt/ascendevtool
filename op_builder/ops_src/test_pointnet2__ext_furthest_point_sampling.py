#!/usr/bin/env python3
"""FPS operator test — build PyTorch binding + precision validation"""
import torch
import subprocess, sys, os

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


def build_and_test():
    import torch_npu

    # Compile binding
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Find the operator source directory (where setup.py is)
    op_dir = os.path.join(script_dir, 'pointnet2__ext_furthest_point_samplingSample')
    if not os.path.isfile(os.path.join(op_dir, 'setup.py')):
        # Try to find via glob
        import glob
        matches = glob.glob(os.path.join(script_dir, '*Sample'))
        matches = [d for d in matches if 'furthest' in d.lower() or 'fps' in d.lower()]
        if matches:
            op_dir = matches[0]
    print(f"  Building from: {op_dir}")
    os.chdir(op_dir)
    print(">>> Compiling PyTorch binding...")
    ret = subprocess.run(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        capture_output=True, text=True, cwd=script_dir
    )
    if ret.returncode != 0:
        print(f"  Compile failed:\n{ret.stderr[-500:]}")
        return False
    print("  Compile OK")

    # Load
    torch.ops.load_library(
        os.path.join(script_dir, f'fps_ops.cpython-{sys.version_info.major}{sys.version_info.minor}-x86_64-linux-gnu.so')
    )
    op = torch.ops.pointnet2__ext_furthest_point_sampling.farthest_point_sample

    # Test multiple sizes
    tests = [
        (1, 128, 32), (1, 512, 64), (2, 256, 48),
        (4, 128, 16), (1, 1024, 128), (2, 500, 100),
        (4, 200, 50), (1, 64, 8), (8, 100, 20), (3, 300, 150),
    ]
    passed = 0
    for B, N, M in tests:
        xyz = torch.randn(B, N, 3).npu()
        ref = cpu_fps(xyz.cpu(), M)
        try:
            out = op(xyz, M)
            ok = torch.equal(ref.long(), out.cpu().long())
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {'PASS' if ok else 'FAIL'}")
            if ok:
                passed += 1
            else:
                mismatch = (ref.long() != out.cpu().long()).sum().item()
                print(f"    mismatch: {mismatch}/{B*M}")
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: ERROR - {e}")

    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = build_and_test()
    print(f"{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
