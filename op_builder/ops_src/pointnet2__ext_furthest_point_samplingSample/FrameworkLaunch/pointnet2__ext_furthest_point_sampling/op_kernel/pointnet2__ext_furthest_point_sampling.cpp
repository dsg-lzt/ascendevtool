#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelFPS {
public:
    __aicore__ inline void Init(GM_ADDR xyz_gm, GM_ADDR out_gm,
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
        this->tileNum = (N + this->blockSize - 1) / this->blockSize;

        this->xyzGm = (__gm__ float*)xyz_gm;
        this->outGm = (__gm__ float*)out_gm;

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockSize * 3 * sizeof(float));
    }

    __aicore__ inline void Process() {
        for (uint32_t b = 0; b < B; b++) {
            float distArr[4096] __attribute__((aligned(64)));
            uint32_t curN = (N < 4096) ? N : 4096;
            for (uint32_t i = 0; i < curN; i++) distArr[i] = 1e10f;

            uint32_t farthest = 0;
            for (uint32_t m = 0; m < M; m++) {
                uint32_t gid = (batchOffset + b) * M + m;
                outGm[gid] = (float)farthest;

                float cx = xyzGm[(batchOffset + b) * N * 3 + farthest * 3 + 0];
                float cy = xyzGm[(batchOffset + b) * N * 3 + farthest * 3 + 1];
                float cz = xyzGm[(batchOffset + b) * N * 3 + farthest * 3 + 2];

                float maxD = -1e10f;
                uint32_t maxIdx = 0;

                for (uint32_t t = 0; t < tileNum; t++) {
                    uint32_t off = t * blockSize;
                    uint32_t len = (off + blockSize <= curN) ? blockSize : (curN - off);
                    if (len == 0) continue;

                    auto inLocal = inQueue.AllocTensor<float>();
                    DataCopy(inLocal, xyzGm + (batchOffset + b) * N * 3 + off * 3, len * 3);
                    inQueue.EnQue(inLocal);
                    inLocal = inQueue.DeQue<float>();

                    for (uint32_t j = 0; j < len; j++) {
                        float dx = inLocal.GetValue(j * 3 + 0) - cx;
                        float dy = inLocal.GetValue(j * 3 + 1) - cy;
                        float dz = inLocal.GetValue(j * 3 + 2) - cz;
                        float nd = dx * dx + dy * dy + dz * dz;
                        if (nd < distArr[off + j]) distArr[off + j] = nd;
                        if (distArr[off + j] > maxD) {
                            maxD = distArr[off + j];
                            maxIdx = off + j;
                        }
                    }

                    inQueue.FreeTensor(inLocal);
                }
                farthest = maxIdx;
            }
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    __gm__ float* xyzGm;
    __gm__ float* outGm;
    uint32_t B, N, M, blockSize, tileNum, batchOffset;
};

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR xyz, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFPS op;
    op.Init(xyz, out,
            tiling_data.B, tiling_data.N, tiling_data.M,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}
