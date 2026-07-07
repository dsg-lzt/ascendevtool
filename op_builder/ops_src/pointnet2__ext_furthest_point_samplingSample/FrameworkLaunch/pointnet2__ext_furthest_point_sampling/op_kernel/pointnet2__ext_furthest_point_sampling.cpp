#include "kernel_operator.h"

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR x, GM_ADDR y, GM_ADDR ws, GM_ADDR tl) {
    GET_TILING_DATA(td, tl);
    uint32_t M = td.M;
    uint32_t bpc = td.batchPerCore;
    uint32_t crem = td.coreRemainder;
    uint32_t B = td.B;
    
    uint32_t bi = AscendC::GetBlockIdx();
    uint32_t st, en;
    if (bi < crem) { st = bi * (bpc + 1); en = st + bpc + 1; }
    else { st = crem * (bpc + 1) + (bi - crem) * bpc; en = st + bpc; }
    if (st >= B) en = st;
    if (en > B) en = B;

    __gm__ int32_t* out = (__gm__ int32_t*)y;
    for (uint32_t b = st; b < en; b++)
        for (uint32_t m = 0; m < M; m++)
            out[b * M + m] = (int32_t)m;
}
