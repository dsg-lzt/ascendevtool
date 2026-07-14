#include "kernel_operator.h"
constexpr int32_t C3 = 3;
constexpr int32_t SZ = 128;  /* padded buffer for alignment */

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);
    int32_t B=td.B, N=td.N, M=td.M, bpc=td.batchPerCore, crem=td.coreRemainder, dt=td.dataTypeLength;
    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs=0, be=0;
    if (cid < (uint32_t)crem) { bs = cid*(bpc+1); be = bs+bpc+1; }
    else { bs = crem*(bpc+1)+(cid-crem)*bpc; be = bs+bpc; }
    if (bs >= B || be <= bs) return;
    if (be > B) be = B;

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* bo = reinterpret_cast<__gm__ int32_t*>(output) + b * M;
        __gm__ float*   bw = reinterpret_cast<__gm__ float*>(workspace) + b * N;
        for (int32_t i = 0; i < N; i++) bw[i] = 3.402823e+38f;

        int32_t fid = 0;
        if (dt == 4) {
            __gm__ float* p = reinterpret_cast<__gm__ float*>(input) + b * N * C3;
            for (int32_t m = 0; m < M; m++) {
                bo[m] = fid;
                float cx=p[fid*C3], cy=p[fid*C3+1], cz=p[fid*C3+2];
                float bv=-1.0f; int32_t bi=0;
                for (int32_t i = 0; i < N; i++) {
                    float d = (p[i*C3]-cx)*(p[i*C3]-cx)+(p[i*C3+1]-cy)*(p[i*C3+1]-cy)+(p[i*C3+2]-cz)*(p[i*C3+2]-cz);
                    if (d < bw[i]) bw[i] = d;
                    if (bw[i] > bv) { bv = bw[i]; bi = i; }
                }
                fid = bi; bw[fid] = 0.0f;
            }
        } else {
            __gm__ half* p = reinterpret_cast<__gm__ half*>(input) + b * N * C3;
            for (int32_t m = 0; m < M; m++) {
                bo[m] = fid;
                float cx=(float)p[fid*C3], cy=(float)p[fid*C3+1], cz=(float)p[fid*C3+2];
                float bv=-1.0f; int32_t bi=0;
                for (int32_t i = 0; i < N; i++) {
                    float d = ((float)p[i*C3]-cx)*((float)p[i*C3]-cx)+((float)p[i*C3+1]-cy)*((float)p[i*C3+1]-cy)+((float)p[i*C3+2]-cz)*((float)p[i*C3+2]-cz);
                    if (d < bw[i]) bw[i] = d;
                    if (bw[i] > bv) { bv = bw[i]; bi = i; }
                }
                fid = bi; bw[fid] = 0.0f;
            }
        }
    }
}
