#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <torch_npu/csrc/aten/CustomOps.h>

#include "torch_npu/csrc/core/npu/NPUException.h"

at::Tensor npu_fps(const at::Tensor& xyz, int64_t npoint) {
    TORCH_CHECK(xyz.dim() == 3, "xyz must be 3D (B, N, 3)");
    TORCH_CHECK(xyz.size(2) == 3, "xyz last dim must be 3");
    
    int64_t B = xyz.size(0);
    int64_t N = xyz.size(1);
    auto out = at::empty({B, npoint}, xyz.options().dtype(at::kLong));
    
    at_npu::native::OpCommand cmd;
    cmd.Name("pointnet2__ext_furthest_point_sampling")
       .Input(xyz)
       .Output(out)
       .Attr("npoint", npoint)
       .Run();
    
    return out;
}

TORCH_LIBRARY(pointnet2__ext_furthest_point_sampling, m) {
    m.def("farthest_point_sample", &npu_fps);
}
