/**
 * FurthestPointSampling - minDist in UB with Duplicate init.
 * Input from GM (read-only), output to GM (write), no workspace usage.
 */
#include "kernel_operator.h"

constexpr int32_t C3 = 3;
const float kFmax = 3.4028234663852886e+38f;

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);

    int32_t B = static_cast<int32_t>(td.B);
    int32_t N = static_cast<int32_t>(td.N);
    int32_t M = static_cast<int32_t>(td.M);
    int32_t bpc = static_cast<int32_t>(td.batchPerCore);
    int32_t crem = static_cast<int32_t>(td.coreRemainder);
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

    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bufMd;
    pipe.InitBuffer(bufMd, N * sizeof(float));
    AscendC::LocalTensor<float> md = bufMd.Get<float>();

    __gm__ int32_t* outGm = reinterpret_cast<__gm__ int32_t*>(output);

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* bo = outGm + b * M;
        for (int32_t i = 0; i < N; i++) md.SetValue(i, 3.402823e+38f);

        int32_t farthest = 0;
        int32_t dtLn = static_cast<int32_t>(td.dataTypeLength);

        if (dtLn == 4) {
            __gm__ float* bi = reinterpret_cast<__gm__ float*>(input) + b * N * C3;
            for (int32_t m = 0; m < M; m++) {
                bo[m] = farthest;
                float cx = bi[farthest*C3], cy = bi[farthest*C3+1], cz = bi[farthest*C3+2];
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float tx = bi[i*C3]-cx, ty = bi[i*C3+1]-cy, tz = bi[i*C3+2]-cz;
                    float d = tx*tx + ty*ty + tz*tz;
                    float cur = md.GetValue(i);
                    if (d < cur) { cur = d; md.SetValue(i, d); }
                    if (cur > bestVal) { bestVal = cur; bestIdx = i; }
                }
                farthest = bestIdx;
                md.SetValue(farthest, 0.0f);
            }
        } else {
            __gm__ half* bi = reinterpret_cast<__gm__ half*>(input) + b * N * C3;
            for (int32_t m = 0; m < M; m++) {
                bo[m] = farthest;
                float cx = static_cast<float>(bi[farthest*C3]);
                float cy = static_cast<float>(bi[farthest*C3+1]);
                float cz = static_cast<float>(bi[farthest*C3+2]);
                float bestVal = -1.0f;
                int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float tx = static_cast<float>(bi[i*C3])-cx;
                    float ty = static_cast<float>(bi[i*C3+1])-cy;
                    float tz = static_cast<float>(bi[i*C3+2])-cz;
                    float d = tx*tx + ty*ty + tz*tz;
                    float cur = md.GetValue(i);
                    if (d < cur) { cur = d; md.SetValue(i, d); }
                    if (cur > bestVal) { bestVal = cur; bestIdx = i; }
                }
                farthest = bestIdx;
                md.SetValue(farthest, 0.0f);
            }
        }
    }
}
