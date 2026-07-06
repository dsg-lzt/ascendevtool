#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t blockSize,
                                uint32_t coreSize, uint32_t coreRemain) {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        if (blockIdx < coreRemain) {
            batchBegin = blockIdx * (coreSize + 1);
            batchEnd = batchBegin + coreSize + 1;
        } else {
            batchBegin = blockIdx * coreSize + coreRemain;
            batchEnd = batchBegin + coreSize;
        }
        if (batchBegin >= B) batchBegin = B;
        if (batchEnd > B) batchEnd = B;
        myBatches = (batchEnd > batchBegin) ? (batchEnd - batchBegin) : 0;

        this->B = B; this->N = N; this->M = M;
        this->blockSize = blockSize;
        if (this->blockSize > N) this->blockSize = N;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), B * N * 3);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y), B * M);

        uint32_t padded = (this->blockSize * 3 + 31) / 32 * 32;
        pipe.InitBuffer(inQueue, BUFFER_NUM, padded * sizeof(float));
        pipe.InitBuffer(calcBuf, this->blockSize * 3 * sizeof(float));
    }

    __aicore__ inline void Process() {
        if (myBatches == 0) return;

        float distArr[2048];

        for (uint32_t b = batchBegin; b < batchEnd; b++) {
            uint32_t curN = (N < 2048) ? N : 2048;
            for (uint32_t i = 0; i < curN; i++) distArr[i] = 1e10f;

            int32_t farthest = 0;
            for (uint32_t m = 0; m < M; m++) {
                yGm.SetValue(b * M + m, farthest);

                float cx = xGm.GetValue(b * N * 3 + farthest * 3 + 0);
                float cy = xGm.GetValue(b * N * 3 + farthest * 3 + 1);
                float cz = xGm.GetValue(b * N * 3 + farthest * 3 + 2);

                float maxD = -1e10f;
                int32_t maxIdx = 0;
                uint32_t tileNum = (curN + blockSize - 1) / blockSize;

                for (uint32_t t = 0; t < tileNum; t++) {
                    CopyIn(b, t);
                    Compute(cx, cy, cz, &maxD, &maxIdx, &distArr[0]);
                }
                farthest = maxIdx;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t b, uint32_t tileIdx) {
        uint32_t off = tileIdx * blockSize;
        uint32_t curN = (N < 2048) ? N : 2048;
        uint32_t len = (off + blockSize <= curN) ? blockSize : (curN - off);
        if (len == 0) return;

        uint32_t padded = (len * 3 + 31) / 32 * 32;
        auto xLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[b * N * 3 + off * 3], padded);
        inQueue.EnQue(xLocal);
    }

    __aicore__ inline void Compute(float cx, float cy, float cz,
                                    float* maxD, int32_t* maxIdx, float* distArr) {
        auto xLocal = inQueue.DeQue<float>();
        uint32_t off = 0;
        uint32_t curN = (N < 2048) ? N : 2048;
        uint32_t tileIdx = 0;
        uint32_t len = (off + blockSize <= curN) ? blockSize : (curN - off);

        AscendC::LocalTensor<float> cBuf = calcBuf.Get<float>(len * 3);
        for (uint32_t j = 0; j < len; j++) {
            float dx = xLocal.GetValue(j * 3 + 0) - cx;
            float dy = xLocal.GetValue(j * 3 + 1) - cy;
            float dz = xLocal.GetValue(j * 3 + 2) - cz;
            float nd = dx * dx + dy * dy + dz * dz;

            uint32_t gIdx = off + j;
            if (nd < distArr[gIdx]) distArr[gIdx] = nd;
            if (distArr[gIdx] > *maxD) { *maxD = distArr[gIdx]; *maxIdx = gIdx; }
        }
        inQueue.FreeTensor(xLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> yGm;

    uint32_t B, N, M, blockSize;
    uint32_t batchBegin, batchEnd, myBatches;
};

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    KernelFPS op;
    op.Init(x, y,
            tilingData.B, tilingData.N, tilingData.M,
            tilingData.block_size,
            tilingData.core_size, tilingData.core_remain);
    op.Process();
}
