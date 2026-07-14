/**
 * FurthestPointSampling - AscendC Kernel
 * minDist in GM (not UB) to avoid pipeline sync issues.
 * Each batch uses its own GM workspace region.
 */
#include "kernel_operator.h"

constexpr int32_t C = 3;

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

    /* minDist in GM workspace — each core uses local offset within own ws */
    __gm__ float*   wsGm  = reinterpret_cast<__gm__ float*>(workspace);
    __gm__ int32_t* outGm = reinterpret_cast<__gm__ int32_t*>(output);

    for (int32_t b = bs; b < be; b++) {
        int32_t lb = b - bs;   /* local batch index within this core */
        __gm__ float*   bw = wsGm  + lb * N;   /* local offset (per-core ws) */
        __gm__ int32_t* bo = outGm + b  * M;   /* global offset (shared output) */

        /* init minDist in GM */
        for (int32_t i = 0; i < N; i++) bw[i] = 3.4028234663852886e+38f;

        int32_t farthest = 0;

        if (dtLn == 4) {
            __gm__ float* bi = reinterpret_cast<__gm__ float*>(input) + b * N * C;
            for (int32_t m = 0; m < M; m++) {
                bo[m] = farthest;
                float cx = bi[farthest*C], cy = bi[farthest*C+1], cz = bi[farthest*C+2];
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float d = (bi[i*C]-cx)*(bi[i*C]-cx) + (bi[i*C+1]-cy)*(bi[i*C+1]-cy) + (bi[i*C+2]-cz)*(bi[i*C+2]-cz);
                    if (d < bw[i]) bw[i] = d;
                    if (bw[i] > bestVal) { bestVal = bw[i]; bestIdx = i; }
                }
                farthest = bestIdx;
                bw[farthest] = 0.0f;
            }
        } else {
            __gm__ half* bi = reinterpret_cast<__gm__ half*>(input) + b * N * C;
            for (int32_t m = 0; m < M; m++) {
                bo[m] = farthest;
                float cx = (float)bi[farthest*C], cy = (float)bi[farthest*C+1], cz = (float)bi[farthest*C+2];
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float d = ((float)bi[i*C]-cx)*((float)bi[i*C]-cx) + ((float)bi[i*C+1]-cy)*((float)bi[i*C+1]-cy) + ((float)bi[i*C+2]-cz)*((float)bi[i*C+2]-cz);
                    if (d < bw[i]) bw[i] = d;
                    if (bw[i] > bestVal) { bestVal = bw[i]; bestIdx = i; }
                }
                farthest = bestIdx;
                bw[farthest] = 0.0f;
            }
        }
    }
}
