#!/usr/bin/env python3
"""FPS test — compile + register via torch.utils.cpp_extension.load_inline"""
import torch, torch_npu, os, subprocess, sys

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


def test():
    import torch_npu
    # Find torch_npu include directory
    npu_include = os.path.join(os.path.dirname(torch_npu.__file__), 'include')
    print(f"  torch_npu include: {npu_include}")
    print(f"  contents: {os.listdir(npu_include) if os.path.isdir(npu_include) else 'NOT FOUND'}")

    # Find torch include
    torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')
    print(f"  torch include: {torch_include}")

    # Ninja should already be installed
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ninja', '-q'], capture_output=True)

    from torch.utils.cpp_extension import load_inline

    cpp_source = """
#include <torch/extension.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

at::Tensor fps_npu(const at::Tensor& xyz, int64_t npoint) {
    auto out = at::empty({xyz.size(0), npoint}, xyz.options().dtype(at::kFloat));
    auto cmd = at_npu::native::OpCommand();
    cmd.Name("pointnet2__ext_furthest_point_sampling")
       .Input(xyz)
       .Output(out)
       .Attr("npoint", npoint)
       .Run();
    return out;
}

TORCH_LIBRARY(fps_test_ops, m) {
    m.def("farthest_point_sample", &fps_npu);
}
"""

    print("  Compiling...")
    import torch_npu
    npu_dir = os.path.dirname(torch_npu.__file__)
    npu_include = os.path.join(npu_dir, 'include')
    
    # Find actual torch_npu .so
    import glob
    npu_so = None
    for pattern in [f'{npu_dir}/lib/*.so', f'{npu_dir}/../torch_npu.libs/*.so']:
        for f in glob.glob(pattern):
            if 'torch_npu' in os.path.basename(f):
                npu_so = f
                break
        if npu_so: break
    if npu_so is None:
        # Find in npu_dir recursively
        for f in glob.glob(f'{npu_dir}/**/*.so', recursive=True):
            if 'torch_npu' in os.path.basename(f):
                npu_so = f
                break
    print(f"  npu_so: {npu_so}")
    print(f"  libs in npu_dir: {os.listdir(npu_dir)}")
    
    lib_dir = os.path.dirname(npu_so) if npu_so else npu_dir
    
    mod = load_inline(
        name='fps_test_ops',
        cpp_sources=[cpp_source],
        extra_include_paths=[npu_include],
        extra_ldflags=[f'-L{lib_dir}', f'-l:{os.path.basename(npu_so)}' if npu_so else '-ltorch_npu'],
        verbose=False,
    )
    print("  Compiled OK")

    op = torch.ops.fps_test_ops.farthest_point_sample

    tests = [(1, 128, 32), (1, 512, 64), (2, 256, 48), (4, 128, 16),
             (1, 1024, 128), (2, 500, 100), (4, 200, 50), (1, 64, 8)]
    passed = 0
    for B, N, M in tests:
        xyz = torch.randn(B, N, 3).npu()
        ref = cpu_fps(xyz.cpu(), M)
        try:
            out = op(xyz, M).long()
            ok = torch.equal(ref, out.cpu())
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {'PASS' if ok else 'FAIL'}")
            if ok: passed += 1
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    print("=== FPS Operator Test ===")
    ok = test()
    print(f"{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
