/**
 * FurthestPointSampling - AscendC Kernel
 *
 * Input:  points  [B, N, 3]  float32|float16
 * Output: indices [B, M]     int32
 *
 * minDist in UB LocalTensor (reliable read-write). Input read from GM (read-only).
 */
#include "kernel_operator.h"

constexpr int32_t C = 3;

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR /*workspace*/, GM_ADDR tilingArg)
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

    /* UB buffer for minDist */
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufMd;
    pipe.InitBuffer(bufMd, N * sizeof(float));
    AscendC::LocalTensor<float> mdLocal = bufMd.Get<float>();

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* bo = reinterpret_cast<__gm__ int32_t*>(output) + b * M;

        /* initialize UB minDist: Duplicate is the reliable vector path */
        const float kBig = 3.4028234663852886e+38f;
        AscendC::Duplicate(mdLocal, kBig, N);

        int32_t farthest = 0;

        if (dtLn == 4) {
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

                    float cur = mdLocal.GetValue(i);
                    if (d < cur) { cur = d; mdLocal.SetValue(i, d); }
                    if (cur > bestVal) { bestVal = cur; bestIdx = i; }
                }
                farthest = bestIdx;
                mdLocal.SetValue(farthest, 0.0f);
            }

        } else {
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

                    float cur = mdLocal.GetValue(i);
                    if (d < cur) { cur = d; mdLocal.SetValue(i, d); }
                    if (cur > bestVal) { bestVal = cur; bestIdx = i; }
                }
                farthest = bestIdx;
                mdLocal.SetValue(farthest, 0.0f);
            }
        }
    }
}
