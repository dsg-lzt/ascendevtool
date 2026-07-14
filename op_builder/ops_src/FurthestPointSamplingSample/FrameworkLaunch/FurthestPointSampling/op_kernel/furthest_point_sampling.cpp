/**
 * FurthestPointSampling - AscendC Kernel
 *
 * Input:  points  [B, N, 3]  float32|float16
 * Output: indices [B, M]     int32
 * Workspace: float[B, N]     per-batch minDist (always float precision)
 *
 * Design: no UB buffers, no templates — direct __gm__ typed arithmetic.
 * Distance computed as separate scalar ops to avoid FMA-induced divergence.
 */
#include "kernel_operator.h"

constexpr int32_t C = 3;  /* coordinate dimension */

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

    /* base typed GM pointers */
    __gm__ int32_t* outGm = reinterpret_cast<__gm__ int32_t*>(output);
    __gm__ float*   wsGm  = reinterpret_cast<__gm__ float*>(workspace);

    for (int32_t b = bs; b < be; b++) {
        /* batch-level pointers via typed arithmetic on __gm__ pointers */
        __gm__ int32_t* bo = outGm + b * M;
        __gm__ float*   bw = wsGm  + b * N;

        for (int32_t i = 0; i < N; i++) bw[i] = 3.402823e+38f;

        int32_t farthest = 0;

        if (dtLn == 4) {                           /* ---- float32 ---- */
            __gm__ float* bi = reinterpret_cast<__gm__ float*>(input) + b * N * C;

            for (int32_t m = 0; m < M; m++) {
                bo[m] = farthest;
                float cx = bi[farthest * C];
                float cy = bi[farthest * C + 1];
                float cz = bi[farthest * C + 2];

                float bestVal = -1.0f;
                int32_t bestIdx = 0;

                for (int32_t i = 0; i < N; i++) {
                    float tx = bi[i*C]   - cx;
                    float ty = bi[i*C+1] - cy;
                    float tz = bi[i*C+2] - cz;
                    float d = tx*tx + ty*ty + tz*tz;
                    if (d < bw[i]) bw[i] = d;
                    if (bw[i] > bestVal) { bestVal = bw[i]; bestIdx = i; }
                }
                farthest = bestIdx;
                bw[farthest] = 0.0f;
            }

        } else {                                   /* ---- float16 ---- */
            __gm__ half* bi = reinterpret_cast<__gm__ half*>(input) + b * N * C;

            for (int32_t m = 0; m < M; m++) {
                bo[m] = farthest;
                float cx = static_cast<float>(bi[farthest * C]);
                float cy = static_cast<float>(bi[farthest * C + 1]);
                float cz = static_cast<float>(bi[farthest * C + 2]);

                float bestVal = -1.0f;
                int32_t bestIdx = 0;

                for (int32_t i = 0; i < N; i++) {
                    float tx = static_cast<float>(bi[i*C])   - cx;
                    float ty = static_cast<float>(bi[i*C+1]) - cy;
                    float tz = static_cast<float>(bi[i*C+2]) - cz;
                    float d = tx*tx + ty*ty + tz*tz;
                    if (d < bw[i]) bw[i] = d;
                    if (bw[i] > bestVal) { bestVal = bw[i]; bestIdx = i; }
                }
                farthest = bestIdx;
                bw[farthest] = 0.0f;
            }
        }
    }
}
