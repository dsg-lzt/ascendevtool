/**
 * FurthestPointSampling - AscendC Kernel
 *
 * Input:  points  [B, N, 3]  float32|float16
 * Output: indices [B, M]     int32
 * Workspace: float[B, N]     per-batch minDist
 *
 * NOTE: Temporarily writes known pattern to diagnose pointer arithmetic.
 */
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);

    int32_t B = static_cast<int32_t>(td.B);
    int32_t N = static_cast<int32_t>(td.N);
    int32_t M = static_cast<int32_t>(td.M);
    int32_t bpc = static_cast<int32_t>(td.batchPerCore);
    int32_t crem= static_cast<int32_t>(td.coreRemainder);
    int32_t dtLn= static_cast<int32_t>(td.dataTypeLength);

    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs, be;
    if (cid < static_cast<uint32_t>(crem)) {
        bs = static_cast<int32_t>(cid) * (bpc + 1);
        be = bs + bpc + 1;
    } else {
        bs = crem * (bpc + 1) + (static_cast<int32_t>(cid) - crem) * bpc;
        be = bs + bpc;
    }
    if (bs >= B || be <= bs) return;
    if (be > B) be = B;

    __gm__ int32_t* outGm = reinterpret_cast<__gm__ int32_t*>(output);
    __gm__ float*   wsGm  = reinterpret_cast<__gm__ float*>(workspace);

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* bo = outGm + b * M;
        __gm__ float*   bw = wsGm  + b * N;

        // Write KNOWN pattern: first element = b, rest = 42
        bo[0] = b;
        for (int32_t m = 1; m < M; m++) bo[m] = 42;
        for (int32_t i = 0; i < N; i++) bw[i] = 1.0f;
    }
}
