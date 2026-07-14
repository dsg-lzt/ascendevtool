#include "kernel_operator.h"

/**
 * FurthestPointSampling — no workspace, O(M²*N) recompute approach.
 * For each candidate i, compute distances to ALL previously selected
 * points and track the minimum. Pick argmax of minima.
 */

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);

    int32_t B=td.B, N=td.N, M=td.M, dt=td.dataTypeLength;
    int32_t bpc=td.batchPerCore, crem=td.coreRemainder;
    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs, be;
    if (cid < (uint32_t)crem) { bs = cid*(bpc+1); be = bs+bpc+1; }
    else { bs = crem*(bpc+1)+(int32_t)(cid-crem)*bpc; be = bs+bpc; }
    if (bs >= B || be <= bs) return;
    if (be > B) be = B;

    __gm__ int32_t* out = reinterpret_cast<__gm__ int32_t*>(output);

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* o = out + b * M;
        o[0] = 0;  /* first selected point is always index 0 */

        if (dt == 4) {
            __gm__ float* p = reinterpret_cast<__gm__ float*>(input) + b * N * 3;
            for (int32_t m = 1; m < M; m++) {
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    /* min distance from i to all selected points (j=0..m-1) */
                    float minD = 3.402823e+38f;
                    for (int32_t j = 0; j < m; j++) {
                        int32_t sj = o[j];  /* j-th selected point index */
                        float dx = p[i*3] - p[sj*3];
                        float dy = p[i*3+1] - p[sj*3+1];
                        float dz = p[i*3+2] - p[sj*3+2];
                        float d = dx*dx + dy*dy + dz*dz;
                        if (d < minD) minD = d;
                    }
                    if (minD > bestVal) { bestVal = minD; bestIdx = i; }
                }
                o[m] = bestIdx;
            }
        } else {
            __gm__ half* p = reinterpret_cast<__gm__ half*>(input) + b * N * 3;
            for (int32_t m = 1; m < M; m++) {
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float minD = 3.402823e+38f;
                    for (int32_t j = 0; j < m; j++) {
                        int32_t sj = o[j];
                        float dx = (float)p[i*3] - (float)p[sj*3];
                        float dy = (float)p[i*3+1] - (float)p[sj*3+1];
                        float dz = (float)p[i*3+2] - (float)p[sj*3+2];
                        float d = dx*dx + dy*dy + dz*dz;
                        if (d < minD) minD = d;
                    }
                    if (minD > bestVal) { bestVal = minD; bestIdx = i; }
                }
                o[m] = bestIdx;
            }
        }
    }
}
