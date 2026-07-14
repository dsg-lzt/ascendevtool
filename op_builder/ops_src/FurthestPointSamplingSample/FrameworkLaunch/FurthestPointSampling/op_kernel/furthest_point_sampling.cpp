/**
 * FurthestPointSampling - AscendC Kernel
 *
 * Input:  points  [B, N, 3]  float32|float16
 * Output: indices [B, M]     int32
 * Work:   float[B, N]        per-batch min distance (always float)
 *
 * NOTE: all GM pointer arithmetic done at the top-level entry function
 * using byte offsets against raw GM_ADDR to avoid template & cast issues.
 */
#include "kernel_operator.h"

#define F32_MAX  3.402823e+38f
#define F32_INIT -1.0f

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);

    int32_t B       = (int32_t)td.B;
    int32_t N       = (int32_t)td.N;
    int32_t M       = (int32_t)td.M;
    int32_t bpc     = (int32_t)td.batchPerCore;
    int32_t crem    = (int32_t)td.coreRemainder;
    int32_t dtLen   = (int32_t)td.dataTypeLength;

    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs, be;
    if (cid < (uint32_t)crem) {
        bs = cid * (bpc + 1);
        be = bs + bpc + 1;
    } else {
        bs = crem * (bpc + 1) + ((int32_t)cid - crem) * bpc;
        be = bs + bpc;
    }
    if (bs >= B) return;
    if (be > B) be = B;
    if (be <= bs) return;

    int32_t inStride = N * COORD_DIM * dtLen;
    int32_t outStride = M * (int32_t)sizeof(int32_t);
    int32_t wsStride = N * (int32_t)sizeof(float);

    for (int32_t b = bs; b < be; b++) {
        /* ---- input/output/ws base pointers for this batch (byte offset) ---- */
        uint8_t*  batchIn  = (uint8_t* __gm__)input    + b * inStride;
        int32_t*  batchOut = (int32_t* __gm__)((uint8_t* __gm__)output   + b * outStride);
        float*    batchWs  = (float*   __gm__)((uint8_t* __gm__)workspace + b * wsStride);

        /* initialise workspace */
        for (int32_t i = 0; i < N; i++) batchWs[i] = F32_MAX;

        int32_t farthest = 0;

        for (int32_t m = 0; m < M; m++) {
            batchOut[m] = farthest;

            if (dtLen == 4) {
                float* p = (float* __gm__)batchIn;
                float cx = p[farthest * COORD_DIM];
                float cy = p[farthest * COORD_DIM + 1];
                float cz = p[farthest * COORD_DIM + 2];

                float bestVal = F32_INIT;
                int32_t bestIdx = 0;

                for (int32_t i = 0; i < N; i++) {
                    float dx = p[i * COORD_DIM] - cx;
                    float dy = p[i * COORD_DIM + 1] - cy;
                    float dz = p[i * COORD_DIM + 2] - cz;
                    float d  = dx * dx + dy * dy + dz * dz;

                    if (d < batchWs[i]) batchWs[i] = d;
                    if (batchWs[i] > bestVal) { bestVal = batchWs[i]; bestIdx = i; }
                }
                farthest = bestIdx;
                batchWs[farthest] = 0.0f;

            } else {
                __gm__ half* p = (__gm__ half*)(uint8_t* __gm__)batchIn;
                float cx = (float)p[farthest * COORD_DIM];
                float cy = (float)p[farthest * COORD_DIM + 1];
                float cz = (float)p[farthest * COORD_DIM + 2];

                float bestVal = F32_INIT;
                int32_t bestIdx = 0;

                for (int32_t i = 0; i < N; i++) {
                    float dx = (float)p[i * COORD_DIM] - cx;
                    float dy = (float)p[i * COORD_DIM + 1] - cy;
                    float dz = (float)p[i * COORD_DIM + 2] - cz;
                    float d  = dx * dx + dy * dy + dz * dz;

                    if (d < batchWs[i]) batchWs[i] = d;
                    if (batchWs[i] > bestVal) { bestVal = batchWs[i]; bestIdx = i; }
                }
                farthest = bestIdx;
                batchWs[farthest] = 0.0f;
            }
        }
    }
}
