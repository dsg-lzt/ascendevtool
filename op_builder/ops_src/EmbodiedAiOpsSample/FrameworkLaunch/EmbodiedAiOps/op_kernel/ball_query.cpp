#include "kernel_operator.h"

extern "C" __global__ __aicore__ void ball_query(GM_ADDR xyz, GM_ADDR center_xyz, GM_ADDR idx, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
}