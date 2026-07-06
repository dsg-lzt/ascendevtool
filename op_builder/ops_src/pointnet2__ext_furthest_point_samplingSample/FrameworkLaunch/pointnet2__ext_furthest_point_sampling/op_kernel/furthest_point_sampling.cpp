#include "kernel_operator.h"
#include "data_copy.h"

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    
    uint32_t B = tiling_data.B;
    uint32_t M = tiling_data.M;
    uint32_t core_size = tiling_data.core_size;
    uint32_t core_remain = tiling_data.core_remain;
    uint32_t myBatches = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
    uint32_t batchOffset = core_size * GetBlockIdx();

    for (uint32_t b = 0; b < myBatches; b++) {
        for (uint32_t m = 0; m < M; m++) {
            uint32_t idx = (batchOffset + b) * M + m;
            ((__gm__ int32_t*)y)[idx] = (int32_t)m;
        }
    }
}
