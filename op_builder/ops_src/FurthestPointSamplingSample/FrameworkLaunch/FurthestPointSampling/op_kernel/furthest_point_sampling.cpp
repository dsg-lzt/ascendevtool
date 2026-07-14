#include "kernel_operator.h"

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);

    int32_t B=td.B, N=td.N, M=td.M, bpc=td.batchPerCore, crem=td.coreRemainder, dt=td.dataTypeLength;
    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs, be;
    if (cid < (uint32_t)crem) { bs = cid*(bpc+1); be = bs+bpc+1; }
    else { bs = crem*(bpc+1)+(int32_t)(cid-crem)*bpc; be = bs+bpc; }
    if (bs >= B || be <= bs) return;
    if (be > B) be = B;

    __gm__ float*   ws = reinterpret_cast<__gm__ float*>(workspace);
    __gm__ int32_t* out = reinterpret_cast<__gm__ int32_t*>(output);

    if (dt == 4) {
        __gm__ float* in = reinterpret_cast<__gm__ float*>(input);
        for (int32_t b = bs; b < be; b++) {
            __gm__ int32_t* o = out + b * M;
            __gm__ float*   w = ws  + b * N;
            for (int32_t i = 0; i < N; i++) w[i] = 3.402823e+38f;
            int32_t fid = 0;
            for (int32_t m = 0; m < M; m++) {
                o[m] = fid;
                float cx = in[b*N*3 + fid*3], cy = in[b*N*3 + fid*3+1], cz = in[b*N*3 + fid*3+2];
                float bestVal = -1.0f; int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float dx = in[b*N*3 + i*3] - cx, dy = in[b*N*3 + i*3+1] - cy, dz = in[b*N*3 + i*3+2] - cz;
                    float d = dx*dx + dy*dy + dz*dz;
                    if (d < w[i]) w[i] = d;
                    if (w[i] > bestVal) { bestVal = w[i]; bestIdx = i; }
                }
                fid = bestIdx; w[fid] = 0.0f;
            }
        }
    } else {
        __gm__ half* in = reinterpret_cast<__gm__ half*>(input);
        for (int32_t b = bs; b < be; b++) {
            __gm__ int32_t* o = out + b * M;
            __gm__ float*   w = ws  + b * N;
            for (int32_t i = 0; i < N; i++) w[i] = 3.402823e+38f;
            int32_t fid = 0;
            for (int32_t m = 0; m < M; m++) {
                o[m] = fid;
                float cx = (float)in[b*N*3 + fid*3], cy = (float)in[b*N*3 + fid*3+1], cz = (float)in[b*N*3 + fid*3+2];
                float bestVal = -1.0f; int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float dx = (float)in[b*N*3 + i*3] - cx, dy = (float)in[b*N*3 + i*3+1] - cy, dz = (float)in[b*N*3 + i*3+2] - cz;
                    float d = dx*dx + dy*dy + dz*dz;
                    if (d < w[i]) w[i] = d;
                    if (w[i] > bestVal) { bestVal = w[i]; bestIdx = i; }
                }
                fid = bestIdx; w[fid] = 0.0f;
            }
        }
    }
}
