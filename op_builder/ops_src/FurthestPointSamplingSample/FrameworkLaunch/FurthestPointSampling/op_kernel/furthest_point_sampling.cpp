#include "kernel_operator.h"

constexpr int32_t COORD_DIM = 3;

__aicore__ void fps_float32_core(
    __gm__ float* inBatch, __gm__ int32_t* outBatch,
    __gm__ float* minDist, uint32_t N, uint32_t M)
{
    for (uint32_t i = 0; i < N; i++) minDist[i] = 3.402823e+38f;

    int32_t farthest = 0;

    for (uint32_t m = 0; m < M; m++) {
        outBatch[m] = farthest;

        float cx = inBatch[farthest * COORD_DIM];
        float cy = inBatch[farthest * COORD_DIM + 1];
        float cz = inBatch[farthest * COORD_DIM + 2];

        float bestVal = -1.0f;
        int32_t bestIdx = 0;

        for (uint32_t i = 0; i < N; i++) {
            float dx = inBatch[i * COORD_DIM] - cx;
            float dy = inBatch[i * COORD_DIM + 1] - cy;
            float dz = inBatch[i * COORD_DIM + 2] - cz;
            float d = dx * dx + dy * dy + dz * dz;

            if (d < minDist[i]) minDist[i] = d;
            if (minDist[i] > bestVal) {
                bestVal = minDist[i];
                bestIdx = static_cast<int32_t>(i);
            }
        }

        farthest = bestIdx;
        minDist[farthest] = 0.0f;
    }
}

__aicore__ void fps_float16_core(
    __gm__ half* inBatch, __gm__ int32_t* outBatch,
    __gm__ float* minDist, uint32_t N, uint32_t M)
{
    for (uint32_t i = 0; i < N; i++) minDist[i] = 3.402823e+38f;

    int32_t farthest = 0;

    for (uint32_t m = 0; m < M; m++) {
        outBatch[m] = farthest;

        float cx = static_cast<float>(inBatch[farthest * COORD_DIM]);
        float cy = static_cast<float>(inBatch[farthest * COORD_DIM + 1]);
        float cz = static_cast<float>(inBatch[farthest * COORD_DIM + 2]);

        float bestVal = -1.0f;
        int32_t bestIdx = 0;

        for (uint32_t i = 0; i < N; i++) {
            float dx = static_cast<float>(inBatch[i * COORD_DIM]) - cx;
            float dy = static_cast<float>(inBatch[i * COORD_DIM + 1]) - cy;
            float dz = static_cast<float>(inBatch[i * COORD_DIM + 2]) - cz;
            float d = dx * dx + dy * dy + dz * dz;

            if (d < minDist[i]) minDist[i] = d;
            if (minDist[i] > bestVal) {
                bestVal = minDist[i];
                bestIdx = static_cast<int32_t>(i);
            }
        }

        farthest = bestIdx;
        minDist[farthest] = 0.0f;
    }
}

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(td, tiling);

    uint32_t B = td.B;
    uint32_t N = td.N;
    uint32_t M = td.M;
    uint32_t bpc = td.batchPerCore;
    uint32_t crem = td.coreRem;

    uint32_t coreId = AscendC::GetBlockIdx();
    uint32_t batchStart, batchEnd;
    if (coreId < crem) {
        batchStart = coreId * (bpc + 1);
        batchEnd = batchStart + bpc + 1;
    } else {
        batchStart = crem * (bpc + 1) + (coreId - crem) * bpc;
        batchEnd = batchStart + bpc;
    }
    if (batchStart >= B) return;
    if (batchEnd > B) batchEnd = B;

    if (TILING_KEY_IS(0)) {
        __gm__ float* inBase = reinterpret_cast<__gm__ float*>(input);
        __gm__ int32_t* outBase = reinterpret_cast<__gm__ int32_t*>(output);
        __gm__ float* wsBase = reinterpret_cast<__gm__ float*>(workspace);

        for (uint32_t b = batchStart; b < batchEnd; b++) {
            fps_float32_core(
                inBase + b * N * COORD_DIM,
                outBase + b * M,
                wsBase + b * N,
                N, M);
        }
    } else if (TILING_KEY_IS(1)) {
        __gm__ half* inBase = reinterpret_cast<__gm__ half*>(input);
        __gm__ int32_t* outBase = reinterpret_cast<__gm__ int32_t*>(output);
        __gm__ float* wsBase = reinterpret_cast<__gm__ float*>(workspace);

        for (uint32_t b = batchStart; b < batchEnd; b++) {
            fps_float16_core(
                inBase + b * N * COORD_DIM,
                outBase + b * M,
                wsBase + b * N,
                N, M);
        }
    }
}
