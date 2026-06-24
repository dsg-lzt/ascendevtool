#include "kernel_operator.h"

extern "C" __global__ __aicore__ void pointnet2_utils_three_interpolate(GM_ADDR x0, GM_ADDR x1, GM_ADDR y0, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}