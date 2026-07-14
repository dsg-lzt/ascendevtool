#include "kernel_operator.h"
constexpr int32_t C3 = 3;
const float FMAX = 3.4028234663852886e+38f;

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tilingArg)
{
    GET_TILING_DATA(td, tilingArg);
    int32_t B=td.B, N=td.N, M=td.M, bpc=td.batchPerCore, crem=td.coreRemainder, dt=td.dataTypeLength;
    uint32_t cid = AscendC::GetBlockIdx();
    int32_t bs, be;
    if (cid < (uint32_t)crem) { bs = cid*(bpc+1); be = bs+bpc+1; }
    else { bs = crem*(bpc+1)+(cid-crem)*bpc; be = bs+bpc; }
    if (bs >= B || be <= bs) return;
    if (be > B) be = B;

    /* UB for minDist */
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> bMd;
    pipe.InitBuffer(bMd, N * sizeof(float) + M * sizeof(float) + N * sizeof(float));
    AscendC::LocalTensor<float> md = bMd.Get<float>();

    for (int32_t b = bs; b < be; b++) {
        __gm__ int32_t* o = (__gm__ int32_t*)((uint8_t* __gm__)output + b*M*4);
        AscendC::Duplicate(md, FMAX, N);
        int32_t fid = 0;
        if (dt == 4) {
            __gm__ float* p = (__gm__ float*)((uint8_t* __gm__)input + b*N*12);
            for (int32_t m = 0; m < M; m++) {
                o[m] = fid;
                float cx=p[fid*C3], cy=p[fid*C3+1], cz=p[fid*C3+2];
                int32_t bi = 0; float bv = -1.0f;
                for (int32_t i = 0; i < N; i++) {
                    float d = (p[i*C3]-cx)*(p[i*C3]-cx)+(p[i*C3+1]-cy)*(p[i*C3+1]-cy)+(p[i*C3+2]-cz)*(p[i*C3+2]-cz);
                    float v = md.GetValue(i);
                    if (d < v) { md.SetValue(i, d); v = d; }
                    if (v > bv) { bv = v; bi = i; }
                }
                fid = bi; md.SetValue(fid, 0.0f);
            }
        } else {
            __gm__ half* p = (__gm__ half*)((uint8_t* __gm__)input + b*N*6);
            for (int32_t m = 0; m < M; m++) {
                o[m] = fid;
                float cx=(float)p[fid*C3], cy=(float)p[fid*C3+1], cz=(float)p[fid*C3+2];
                int32_t bi = 0; float bv = -1.0f;
                for (int32_t i = 0; i < N; i++) {
                    float d = ((float)p[i*C3]-cx)*((float)p[i*C3]-cx)+((float)p[i*C3+1]-cy)*((float)p[i*C3+1]-cy)+((float)p[i*C3+2]-cz)*((float)p[i*C3+2]-cz);
                    float v = md.GetValue(i);
                    if (d < v) { md.SetValue(i, d); v = d; }
                    if (v > bv) { bv = v; bi = i; }
                }
                fid = bi; md.SetValue(fid, 0.0f);
            }
        }
    }
}
