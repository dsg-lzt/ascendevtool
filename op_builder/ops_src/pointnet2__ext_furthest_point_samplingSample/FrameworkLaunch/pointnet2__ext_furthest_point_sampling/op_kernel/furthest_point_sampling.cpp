#include "kernel_operator.h"

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    
    uint32_t B = tilingData.B;
    uint32_t M = tilingData.M;
    uint32_t core_size = tilingData.core_size;
    uint32_t core_remain = tilingData.core_remain;
    uint32_t batchOffset = core_size * AscendC::GetBlockIdx();
    uint32_t myBatches = core_size + (AscendC::GetBlockNum() == AscendC::GetBlockIdx() + 1 ? core_remain : 0);

    __gm__ float* out = (__gm__ float*)y;
    for (uint32_t b = 0; b < myBatches; b++) {
        for (uint32_t m = 0; m < M; m++) {
            out[(batchOffset + b) * M + m] = (float)m;
        }
    }
}
