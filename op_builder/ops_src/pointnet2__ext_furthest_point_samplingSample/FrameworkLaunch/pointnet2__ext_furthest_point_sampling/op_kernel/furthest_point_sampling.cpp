#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFF_NUM = 2;
constexpr int32_t COORD_DIM = 3;

template<typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t blockSize,
                                uint32_t coreSize, uint32_t coreRemain) {
        this->B = B; this->N = N; this->M = M;
        this->blockSize = blockSize;

        uint32_t blockIdx = GetBlockIdx();
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

        if (myBatches > 0) {
            uint32_t readSize = blockSize * COORD_DIM;
            uint32_t padded = (readSize + 31) / 32 * 32;
            xyzGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), B * COORD_DIM * N);
            idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(y), B * M);

            pipe.InitBuffer(inQueue, BUFF_NUM, padded * sizeof(T));
            pipe.InitBuffer(calcBuf, readSize * 2 * sizeof(T));
        }
    }

    __aicore__ inline void Process() {
        if (myBatches == 0) return;
        for (uint32_t b = batchBegin; b < batchEnd; b++) {
            ProcessBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t b) {
        float distArr[2048];
        uint32_t curN = (N < 2048) ? N : 2048;
        for (uint32_t i = 0; i < curN; i++) distArr[i] = 1e10f;

        int32_t farthest = 0;
        T cx = 0, cy = 0, cz = 0;

        for (uint32_t m = 0; m < M; m++) {
            idxGm.SetValue(b * M + m, farthest);

            uint64_t cOff = static_cast<uint64_t>(b) * N * COORD_DIM + farthest * COORD_DIM;
            auto tBuf = calcBuf.Get<T>(COORD_DIM);
            DataCopy(tBuf, xyzGm[cOff], COORD_DIM);
            cx = tBuf.GetValue(0);
            cy = tBuf.GetValue(1);
            cz = tBuf.GetValue(2);

            float maxD = -1e10f;
            int32_t maxIdx = 0;
            uint32_t tileNum = (curN + blockSize - 1) / blockSize;

            for (uint32_t t = 0; t < tileNum; t++) {
                uint32_t off = t * blockSize;
                uint32_t len = (off + blockSize <= curN) ? blockSize : (curN - off);
                if (len == 0) continue;

                LoadBlock(b, off, len);

                LocalTensor<T> xBuf = calcBuf.Get<T>(0, len);
                LocalTensor<T> dBuf = calcBuf.Get<T>(len * COORD_DIM, len);

                for (uint32_t j = 0; j < len; j++) {
                    T dx = xBuf.GetValue(j * COORD_DIM) - cx;
                    T dy = xBuf.GetValue(j * COORD_DIM + 1) - cy;
                    T dz = xBuf.GetValue(j * COORD_DIM + 2) - cz;
                    float nd = (float)dx * (float)dx + (float)dy * (float)dy + (float)dz * (float)dz;
                    if (nd < distArr[off + j]) distArr[off + j] = nd;
                    if (distArr[off + j] > maxD) { maxD = distArr[off + j]; maxIdx = off + j; }
                }
            }
            farthest = maxIdx;
        }
    }

    __aicore__ inline void LoadBlock(uint32_t b, uint32_t r_begin, uint32_t r_count) {
        uint64_t baseOff = static_cast<uint64_t>(b) * N * COORD_DIM + r_begin * COORD_DIM;
        uint32_t padded = (r_count * COORD_DIM + 31) / 32 * 32;

        auto inLocal = inQueue.AllocTensor<T>();
        DataCopy(inLocal, xyzGm[baseOff], padded);
        inQueue.EnQue(inLocal);
        inLocal = inQueue.DeQue<T>();

        auto xBuf = calcBuf.Get<T>(0, r_count * COORD_DIM);
        DataCopy(xBuf, inLocal, r_count * COORD_DIM);

        inQueue.FreeTensor(inLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFF_NUM> inQueue;
    TBuf<QuePosition::VECCALC> calcBuf;

    GlobalTensor<T> xyzGm;
    GlobalTensor<int32_t> idxGm;

    uint32_t B, N, M, blockSize;
    uint32_t batchBegin, batchEnd, myBatches;
};

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(10000001)) {
        KernelFPS<half> op;
        op.Init(x, y, tilingData.B, tilingData.N, tilingData.M,
                tilingData.block_size,
                tilingData.core_size, tilingData.core_remain);
        op.Process();
    } else {
        KernelFPS<float> op;
        op.Init(x, y, tilingData.B, tilingData.N, tilingData.M,
                tilingData.block_size,
                tilingData.core_size, tilingData.core_remain);
        op.Process();
    }
}
