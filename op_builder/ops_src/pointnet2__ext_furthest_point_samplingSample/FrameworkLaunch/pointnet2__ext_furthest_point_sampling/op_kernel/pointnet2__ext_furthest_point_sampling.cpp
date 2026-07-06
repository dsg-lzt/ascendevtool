#include "kernel_operator.h"

constexpr int32_t BUF_NUM = 2;

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR points, GM_ADDR sampled, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    if (!TILING_KEY_IS(0)) return;
    
    int32_t B = td.B, N = td.N, M = td.M;
    if (M > N) M = N;
    
    __gm__ float* in = (__gm__ float*)points;
    __gm__ int32_t* out = (__gm__ int32_t*)sampled;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> qX, qY, qZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> mdBuf, bufDist, bufTmp, bufSca;
    
    int32_t tileN = td.ubPointsNum;
    if (tileN > N) tileN = N;
    if (tileN < 64) tileN = 64;
    
    pipe.InitBuffer(qX, BUF_NUM, tileN * sizeof(float));
    pipe.InitBuffer(qY, BUF_NUM, tileN * sizeof(float));
    pipe.InitBuffer(qZ, BUF_NUM, tileN * sizeof(float));
    pipe.InitBuffer(mdBuf, N * sizeof(float));
    pipe.InitBuffer(bufDist, tileN * sizeof(float));
    pipe.InitBuffer(bufTmp, tileN * sizeof(float));
    pipe.InitBuffer(bufSca, tileN * sizeof(float));

    uint32_t batchStart = AscendC::GetBlockIdx() * td.batchPerCore;
    uint32_t batchEnd = batchStart + td.batchPerCore;
    if (batchStart >= (uint32_t)B) batchEnd = batchStart;
    if (batchEnd > (uint32_t)B) batchEnd = B;

    for (uint32_t b = batchStart; b < batchEnd; b++) {
        __gm__ float* batchIn = in + b * N * 3;
        __gm__ int32_t* batchOut = out + b * M;
        
        AscendC::LocalTensor<float> mdAll = mdBuf.Get<float>();
        for (int32_t i = 0; i < N; i++) mdAll.SetValue(i, td.initVal);

        batchOut[0] = 0;
        float selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        int32_t numTiles = (N + tileN - 1) / tileN;
        uint32_t selIdx = 0;

        for (int32_t m = 1; m < M; m++) {
            float globalMax = -65504.0f;
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; t++) {
                int32_t tStart = t * tileN;
                int32_t curN = (tStart + tileN <= N) ? tileN : (N - tStart);

                AscendC::LocalTensor<float> xTile = qX.AllocTensor<float>();
                AscendC::LocalTensor<float> yTile = qY.AllocTensor<float>();
                AscendC::LocalTensor<float> zTile = qZ.AllocTensor<float>();
                
                for (int32_t i = 0; i < curN; i++) {
                    int32_t gi = tStart + i;
                    xTile.SetValue(i, batchIn[gi * 3 + 0]);
                    yTile.SetValue(i, batchIn[gi * 3 + 1]);
                    zTile.SetValue(i, batchIn[gi * 3 + 2]);
                }
                qX.EnQue(xTile); qY.EnQue(yTile); qZ.EnQue(zTile);
                xTile = qX.DeQue<float>();
                yTile = qY.DeQue<float>();
                zTile = qZ.DeQue<float>();

                AscendC::LocalTensor<float> dist = bufDist.Get<float>();
                AscendC::LocalTensor<float> tmp  = bufTmp.Get<float>();
                AscendC::LocalTensor<float> sca  = bufSca.Get<float>();

                AscendC::Duplicate(sca, selX, curN);
                AscendC::Sub(dist, xTile, sca, curN);
                AscendC::Mul(dist, dist, dist, curN);
                AscendC::Duplicate(sca, selY, curN);
                AscendC::Sub(tmp, yTile, sca, curN);
                AscendC::Mul(tmp, tmp, tmp, curN);
                AscendC::Add(dist, dist, tmp, curN);
                AscendC::Duplicate(sca, selZ, curN);
                AscendC::Sub(tmp, zTile, sca, curN);
                AscendC::Mul(tmp, tmp, tmp, curN);
                AscendC::Add(dist, dist, tmp, curN);
                AscendC::Min(mdAll, mdAll, dist, curN);

                float localMaxF = -65504.0f;
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; i++) {
                    float fv = (float)mdAll.GetValue(i);
                    if (fv > localMaxF) { localMaxF = fv; localIdx = i; }
                }
                if (localMaxF > globalMax) { globalMax = localMaxF; globalIdx = tStart + localIdx; }

                qX.FreeTensor(xTile); qY.FreeTensor(yTile); qZ.FreeTensor(zTile);
            }
            selIdx = globalIdx;
            selX = batchIn[globalIdx * 3 + 0];
            selY = batchIn[globalIdx * 3 + 1];
            selZ = batchIn[globalIdx * 3 + 2];
            batchOut[m] = (int32_t)globalIdx;
            mdAll.SetValue(globalIdx, 0.0f);
        }
    }
}
