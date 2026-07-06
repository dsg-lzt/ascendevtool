#!/usr/bin/env python3
"""FPS test — compile + load + validate"""
import torch, torch_npu, os, subprocess, sys, glob, tempfile

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
#include <torch_npu/csrc/framework/OpCommand.h>
at::Tensor fps_npu(const at::Tensor& xyz, int64_t npoint) {
    auto out = at::empty({xyz.size(0), npoint}, xyz.options().dtype(at::kInt));
    at_npu::native::OpCommand cmd;
    cmd.Name("pointnet2__ext_furthest_point_sampling")
       .Input(xyz).Output(out).Attr("npoint", npoint).Run();
    return out;
}
TORCH_LIBRARY(fps_test_ops, m) { m.def("farthest_point_sample", &fps_npu); }
"""

def test():
    subprocess.run([sys.executable,'-m','pip','install','ninja','-q'], capture_output=True)
    npu_dir = os.path.dirname(torch_npu.__file__)
    npu_inc = os.path.join(npu_dir, 'include')
    npu_so = glob.glob(f'{npu_dir}/lib/libtorch_npu.so')
    npu_so = npu_so[0] if npu_so else None
    lib_dir = os.path.dirname(npu_so) if npu_so else npu_dir

    from torch.utils.cpp_extension import load_inline
    build_dir = tempfile.mkdtemp()
    load_inline(name='fps_test_ops', cpp_sources=[cpp_source],
        extra_include_paths=[npu_inc],
        extra_ldflags=[f'-L{lib_dir}', f'-l:{os.path.basename(npu_so)}'],
        build_directory=build_dir, is_python_module=False, verbose=False)
    torch.ops.load_library(os.path.join(build_dir, 'fps_test_ops.so'))
    print("  Compiled + loaded OK")

    op = torch.ops.fps_test_ops.farthest_point_sample
    tests = [(1,128,32),(1,512,64),(2,256,48),(4,128,16),
             (1,1024,128),(2,500,100),(4,200,50),(1,64,8),(8,100,20),(3,300,150)]
    passed=0
    for B,N,M in tests:
        xyz=torch.randn(B,N,3).npu()
        ref=cpu_fps(xyz.cpu(),M)
        try:
            out=op(xyz,M)
            ok=torch.equal(ref,out.cpu())
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {'PASS' if ok else 'FAIL'}{'' if ok else f' mismatch={(ref!=out.cpu()).sum()}/{B*M}'}")
            if ok: passed+=1
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n  Result: {passed}/{len(tests)} passed")
    return passed==len(tests)

if __name__=="__main__":
    print("=== FPS Operator Test ===")
    ok=test()
    print(f"{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
