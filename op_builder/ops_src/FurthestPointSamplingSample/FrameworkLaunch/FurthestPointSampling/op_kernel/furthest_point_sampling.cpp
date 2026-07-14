#include "kernel_operator.h"

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);

    int32_t B=td.B, N=td.N, M=td.M, bpc=td.batchPerCore, crem=td.coreRemainder, dt=td.dataTypeLength;
    int32_t wsN = td.wsStride;
    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs, be;
    if (cid < (uint32_t)crem) { bs = cid*(bpc+1); be = bs+bpc+1; }
    else { bs = crem*(bpc+1)+(int32_t)(cid-crem)*bpc; be = bs+bpc; }
    if (bs >= B || be <= bs) return;
    if (be > B) be = B;

    __gm__ float*   ws = reinterpret_cast<__gm__ float*>(workspace);
    __gm__ int32_t* out = reinterpret_cast<__gm__ int32_t*>(output);

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* o = out + b * M;
        __gm__ float*   w = ws  + b * wsN;
        for (int32_t i = 0; i < N; i++) w[i] = 3.402823e+38f;
        int32_t fid = 0;
        if (dt == 4) {
            __gm__ float* in = reinterpret_cast<__gm__ float*>(input) + b * N * 3;
            for (int32_t m = 0; m < M; m++) {
                o[m] = fid;
                float cx = in[fid*3], cy = in[fid*3+1], cz = in[fid*3+2];
                float bestVal = -1.0f; int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float d = (in[i*3]-cx)*(in[i*3]-cx)+(in[i*3+1]-cy)*(in[i*3+1]-cy)+(in[i*3+2]-cz)*(in[i*3+2]-cz);
                    if (d < w[i]) w[i] = d;
                    if (w[i] > bestVal) { bestVal = w[i]; bestIdx = i; }
                }
                fid = bestIdx; w[fid] = 0.0f;
            }
        } else {
            __gm__ half* in = reinterpret_cast<__gm__ half*>(input) + b * N * 3;
            for (int32_t m = 0; m < M; m++) {
                o[m] = fid;
                float cx = (float)in[fid*3], cy = (float)in[fid*3+1], cz = (float)in[fid*3+2];
                float bestVal = -1.0f; int32_t bestIdx = 0;
                for (int32_t i = 0; i < N; i++) {
                    float d = ((float)in[i*3]-cx)*((float)in[i*3]-cx)+((float)in[i*3+1]-cy)*((float)in[i*3+1]-cy)+((float)in[i*3+2]-cz)*((float)in[i*3+2]-cz);
                    if (d < w[i]) w[i] = d;
                    if (w[i] > bestVal) { bestVal = w[i]; bestIdx = i; }
                }
                fid = bestIdx; w[fid] = 0.0f;
            }
        }
    }
}
