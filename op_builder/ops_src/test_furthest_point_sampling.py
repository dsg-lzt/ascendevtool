#!/usr/bin/env python3
"""FPS test - reference kernel validation"""
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
    auto out = at::zeros({xyz.size(0), npoint}, xyz.options().dtype(at::kInt));
    at_npu::native::OpCommand cmd;
    cmd.Name("FurthestPointSampling").Input(xyz).Output(out).Attr("npoint", npoint).Run();
    return out;
}
TORCH_LIBRARY(fps_ops, m) { m.def("farthest_point_sample", &fps_npu); }
"""

def test():
    subprocess.run([sys.executable,'-m','pip','install','ninja','-q'], capture_output=True)
    os.environ['PATH'] = os.path.dirname(sys.executable) + ':' + os.environ.get('PATH','')
    npu_dir = os.path.dirname(torch_npu.__file__)
    npu_inc = os.path.join(npu_dir, 'include')
    npu_so = glob.glob(f'{npu_dir}/lib/libtorch_npu.so')
    npu_so = npu_so[0] if npu_so else None
    lib_dir = os.path.dirname(npu_so) if npu_so else npu_dir

    from torch.utils.cpp_extension import load_inline
    build_dir = tempfile.mkdtemp()
    load_inline(name='fps_ops', cpp_sources=[cpp_source],
        extra_include_paths=[npu_inc],
        extra_ldflags=[f'-L{lib_dir}', f'-l:{os.path.basename(npu_so)}'],
        build_directory=build_dir, is_python_module=False, verbose=False)
    torch.ops.load_library(os.path.join(build_dir, 'fps_ops.so'))
    op = torch.ops.fps_ops.farthest_point_sample

    # ---- warmup: first call triggers operator compilation ----
    print("  [warmup] compiling FPS operator...", flush=True)
    wm_out = op(torch.randn(1,64,3).npu(), 8)
    torch.npu.synchronize()
    print(f"  [warmup] raw output[:8] = {wm_out.cpu()[0,:8].tolist()}", flush=True)

    # warmup multi-batch
    wm_out2 = op(torch.randn(8,100,3).npu(), 20)
    torch.npu.synchronize()
    print(f"  [warmup] B=8 batch0[:6] = {wm_out2.cpu()[0,:6].tolist()}", flush=True)
    print(f"  [warmup] B=8 batch1[:6] = {wm_out2.cpu()[1,:6].tolist()}", flush=True)
    print(f"  [warmup] B=8 batch2[:6] = {wm_out2.cpu()[2,:6].tolist()}", flush=True)
    print("  [warmup] done.", flush=True)

    # ---- test B=8 by calling 1 batch at a time ----
    B,N,M = 8,100,20
    xyz=torch.randn(B,N,3).npu()
    ref=cpu_fps(xyz.cpu(),M)
    out = torch.zeros(B, M, dtype=torch.int32)
    for bi in range(B):
        out_bi = op(xyz[bi:bi+1], M)
        torch.npu.synchronize()
        out[bi:bi+1] = out_bi.cpu()
    ok = torch.equal(ref, out)
    if ok:
        print(f"  [B8-per-batch] PASS")
    else:
        diff=(ref!=out); tw=diff.sum().item(); pb=diff.sum(dim=1).tolist()
        print(f"  [B8-per-batch] FAIL m={tw}/160 per_batch={pb}")

    # ---- original B=8 call for comparison ----
    xyz2=torch.randn(B,N,3).npu()
    ref2=cpu_fps(xyz2.cpu(),M)
    out2=op(xyz2,M)
    torch.npu.synchronize()
    ok2=torch.equal(ref2,out2.cpu())
    if ok2:
        print(f"  [B8-bulk] PASS")
    else:
        diff2=(ref2!=out2.cpu()); tw2=diff2.sum().item(); pb2=diff2.sum(dim=1).tolist()
        print(f"  [B8-bulk] FAIL m={tw2}/160 per_batch={pb2}")
    # ---- end debug test ----
    passed=0
    tests = [(1,128,32),(1,512,64),(2,256,48),(4,128,16),(1,1024,128),
             (2,500,100),(4,200,50),(1,64,8),(8,100,20),(3,300,150)]
    passed=0
    for B,N,M in tests:
        xyz=torch.randn(B,N,3).npu()
        ref=cpu_fps(xyz.cpu(),M)
        try:
            out=op(xyz,M)
            torch.npu.synchronize()
            out_cpu = out.cpu()
            ok=torch.equal(ref,out_cpu)
            if ok:
                print(f"  B={B:3d} N={N:4d} M={M:3d}: PASS")
                passed+=1
            else:
                diff = (ref != out_cpu)
                total_wrong = diff.sum().item()
                per_batch = diff.sum(dim=1).tolist()
                mismatch_idx = diff.nonzero(as_tuple=False)
                detail = ""
                if mismatch_idx.shape[0] > 0:
                    first = mismatch_idx[0].tolist()
                    detail = f" 1st_mismatch: batch={first[0]} idx={first[1]} ref={ref[first[0],first[1]].item()} out={out_cpu[first[0],first[1]].item()}"
                print(f"  B={B:3d} N={N:4d} M={M:3d}: FAIL m={total_wrong}/{B*M} per_batch={per_batch}{detail}")
        except Exception as e:
            print(f"  B={B:3d} N={N:4d} M={M:3d}: {type(e).__name__}: {str(e)[:120]}")
    print(f"\n  {passed}/{len(tests)} passed")
    return passed==len(tests)

if __name__=="__main__":
    print("=== FPS Operator Test ===")
    ok=test()
    print(f"{'ALL PASS' if ok else 'FAILED'}")
    exit(0 if ok else 1)
