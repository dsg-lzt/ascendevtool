#!/usr/bin/env python3
"""FPS test — register custom op via torch.utils.cpp_extension.load_inline"""
import torch
from torch.utils.cpp_extension import load_inline

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


cpp_source = """
#include <torch/extension.h>
#include <ATen/native/npu/frame/OpCommand.h>

static auto op_executed = false;

void farthest_point_sampling_op(const at::Tensor& xyz, int64_t npoint, const at::Tensor& out) {
    auto cmd = at::native::OpCommand();
    cmd.Name("pointnet2__ext_furthest_point_sampling")
       .Input(xyz)
       .Output(out)
       .Attr("npoint", npoint)
       .Run();
}

at::Tensor farthest_point_sampling(const at::Tensor& xyz, int64_t npoint) {
    auto out = at::empty({xyz.size(0), npoint}, xyz.options().dtype(at::kFloat));
    farthest_point_sampling_op(xyz, npoint, out);
    return out;
}

TORCH_LIBRARY(pointnet2_fps_ops, m) {
    m.def("farthest_point_sample", &farthest_point_sampling);
}
"""


def test():
    import torch_npu

    # Ensure ninja is available
    import subprocess, sys
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ninja', '-q'], capture_output=True)
    
    # Compile inline (no sys.path, no extra includes needed)
    print("  Compiling inline...")
    mod = load_inline(
        name='fps_ops',
        cpp_sources=[cpp_source],
        functions=['farthest_point_sample'],
        verbose=False,
    )
    print("  Compiled OK")
    
    op = torch.ops.pointnet2_fps_ops.farthest_point_sample

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
            out = op(xyz, M).long()
            ok = torch.equal(ref, out.cpu())
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {'PASS' if ok else 'FAIL'}")
            if ok: passed += 1
            else:
                mismatch = (ref != out.cpu()).sum().item()
                print(f"    mismatch: {mismatch}/{B*M}")
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test()
    print(f"{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
