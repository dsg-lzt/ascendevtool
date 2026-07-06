#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFPS {
public:
    __aicore__ inline void Init(GM_ADDR xyz_gm, GM_ADDR idx_gm,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t block_size, uint32_t core_size,
                                uint32_t core_remain) {
        uint32_t myBatches = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        this->batchOffset = core_size * GetBlockIdx();
        this->B = myBatches;
        this->N = N;
        this->M = M;
        this->blockSize = block_size;
        if (this->blockSize > N) this->blockSize = N;
        this->GmStride = B * N * 3;

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockSize * 3 * sizeof(float));
        pipe.InitBuffer(calQueue, BUFFER_NUM, this->blockSize * 3 * sizeof(float));
        pipe.InitBuffer(distQueue, BUFFER_NUM, this->blockSize * sizeof(float));
        pipe.InitBuffer(maskQueue, BUFFER_NUM, this->blockSize * sizeof(half));
    }

    __aicore__ inline void Process() {
        auto* xyzGm = (__gm__ float*)x0;
        auto* idxGm = (__gm__ int32_t*)x1;

        for (uint32_t b = 0; b < B; b++) {
            float distances[2048] __attribute__((aligned(32)));
            for (uint32_t n = 0; n < N && n < 2048; n++) distances[n] = 1e10f;

            uint32_t farthest = 0;
            for (uint32_t m = 0; m < M; m++) {
                idxGm[(batchOffset + b) * M + m] = farthest;

                float cx = xyzGm[(batchOffset + b) * N * 3 + farthest * 3 + 0];
                float cy = xyzGm[(batchOffset + b) * N * 3 + farthest * 3 + 1];
                float cz = xyzGm[(batchOffset + b) * N * 3 + farthest * 3 + 2];

                UpdateDistances(xyzGm, distances, b, cx, cy, cz, N);
                farthest = FindMaxIndex(distances, N);
            }
        }
    }

private:
    __aicore__ inline void UpdateDistances(__gm__ float* xyzGm, float* dist, uint32_t b,
                                           float cx, float cy, float cz, uint32_t N) {
        for (uint32_t t = 0; t < tileNum; t++) {
            uint32_t off = t * blockSize;
            uint32_t len = (off + blockSize <= N) ? blockSize : (N - off);
            if (len == 0) break;

            auto inLocal = inQueue.AllocTensor<float>();
            DataCopy(inLocal, xyzGm[(batchOffset + b) * N * 3 + off * 3], len * 3);
            inQueue.EnQue(inLocal);
            inLocal = inQueue.DeQue<float>();

            auto calLocal = calQueue.AllocTensor<float>();
            auto dLocal = distQueue.AllocTensor<float>();
            auto mLocal = maskQueue.AllocTensor<half>();

            for (uint32_t j = 0; j < len; j++) {
                float dx = inLocal.GetValue(j * 3 + 0) - cx;
                float dy = inLocal.GetValue(j * 3 + 1) - cy;
                float dz = inLocal.GetValue(j * 3 + 2) - cz;
                float nd = dx * dx + dy * dy + dz * dz;
                if (nd < dist[off + j]) dist[off + j] = nd;
            }

            calQueue.EnQue(calLocal);
            distQueue.EnQue(dLocal);
            maskQueue.EnQue(mLocal);
            calLocal = calQueue.DeQue<float>();
            dLocal = distQueue.DeQue<float>();
            mLocal = maskQueue.DeQue<half>();
            calQueue.FreeTensor(calLocal);
            distQueue.FreeTensor(dLocal);
            maskQueue.FreeTensor(mLocal);
            inQueue.FreeTensor(inLocal);
        }
    }

    __aicore__ inline uint32_t FindMaxIndex(float* dist, uint32_t N) {
        float maxVal = -1e10f;
        uint32_t maxIdx = 0;
        for (uint32_t i = 0; i < N; i++) {
            if (dist[i] > maxVal) {
                maxVal = dist[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> calQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> distQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> maskQueue;
    uint32_t B, N, M, blockSize, tileNum, batchOffset, GmStride;
};

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR x0, GM_ADDR x1, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFPS op;
    op.Init(x0, x1,
            tiling_data.B, tiling_data.N, tiling_data.M,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}
