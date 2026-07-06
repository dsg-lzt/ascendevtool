#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
typedef float DT;

class KernelFPS {
public:
    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y0,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t block_size, uint32_t core_size,
                                uint32_t core_remain) {
        uint32_t myBatches = core_size + (GetBlockNum() == GetBlockIdx() + 1 ? core_remain : 0);
        uint32_t batchOffset = core_size * GetBlockIdx();

        this->B = myBatches;
        this->N = N;
        this->M = M;
        this->blockSize = block_size;
        if (this->blockSize > N) this->blockSize = N;
        this->tileNum = (N + this->blockSize - 1) / this->blockSize;

        auto xyzGmAddr = (__gm__ DT *)x0 + batchOffset * N * 3;
        auto idxGmAddr = (__gm__ int32_t *)y0 + batchOffset * M;

        xyzGm.SetGlobalBuffer(xyzGmAddr, B * N * 3);
        idxGm.SetGlobalBuffer(idxGmAddr, B * M);

        pipe.InitBuffer(inQueue, BUFFER_NUM, this->blockSize * 3 * sizeof(DT));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->blockSize * sizeof(DT));
        pipe.InitBuffer(distBuf, N * sizeof(DT));
        pipe.InitBuffer(centroidBuf, sizeof(DT) * 3);
        pipe.InitBuffer(tmpBuf1, this->blockSize * sizeof(DT));
        pipe.InitBuffer(tmpBuf3, this->blockSize * 3 * sizeof(DT));
    }

    __aicore__ inline void Process() {
        for (uint32_t b = 0; b < B; b++) {
            ProcessOneBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessOneBatch(uint32_t b) {
        auto dist = distBuf.Get<DT>();
        Duplicate(dist, DT(1e10f), N);

        int32_t farthest_i = 0;
        DT farthest_coord[3];

        for (uint32_t m = 0; m < M; m++) {
            idxGm.SetValue(b * M + m, farthest_i);

            farthest_coord[0] = xyzGm.GetValue(b * N * 3 + farthest_i * 3);
            farthest_coord[1] = xyzGm.GetValue(b * N * 3 + farthest_i * 3 + 1);
            farthest_coord[2] = xyzGm.GetValue(b * N * 3 + farthest_i * 3 + 2);

            DT maxDist = -1.0f;
            int32_t maxIdx = 0;

            for (uint32_t t = 0; t < tileNum; t++) {
                uint32_t offset = t * blockSize;
                uint32_t length = (offset + blockSize <= N) ? blockSize : (N - offset);

                auto inLocal = inQueue.AllocTensor<DT>();
                DataCopy(inLocal, xyzGm[b * N * 3 + offset * 3], length * 3);
                inQueue.EnQue(inLocal);
                inLocal = inQueue.DeQue<DT>();

                auto outLocal = outQueue.AllocTensor<DT>();
                auto tmp1 = tmpBuf1.Get<DT>();
                auto tmp3 = tmpBuf3.Get<DT>();

                auto cent = centroidBuf.Get<DT>();
                Duplicate(cent, farthest_coord[0], 3);
                cent.SetValue(0, farthest_coord[0]);
                cent.SetValue(1, farthest_coord[1]);
                cent.SetValue(2, farthest_coord[2]);

                Sub(tmp3, inLocal, cent, length * 3);
                Mul(tmp3, tmp3, tmp3, length * 3);

                for (uint32_t j = 0; j < length; j++) {
                    DT s = tmp3.GetValue(j * 3) + tmp3.GetValue(j * 3 + 1) + tmp3.GetValue(j * 3 + 2);
                    tmp1.SetValue(j, s);
                }

                for (uint32_t j = 0; j < length; j++) {
                    DT oldDist = dist.GetValue(offset + j);
                    DT newDist = tmp1.GetValue(j);
                    if (newDist < oldDist) {
                        dist.SetValue(offset + j, newDist);
                    }
                }

                for (uint32_t j = 0; j < length; j++) {
                    DT d = dist.GetValue(offset + j);
                    if (d > maxDist) {
                        maxDist = d;
                        maxIdx = offset + j;
                    }
                }

                outQueue.EnQue(outLocal);
                outLocal = outQueue.DeQue<DT>();
                outQueue.FreeTensor(outLocal);
                inQueue.FreeTensor(inLocal);
            }

            farthest_i = maxIdx;
        }
    }

private:
    TPipe pipe;
    GlobalTensor<DT> xyzGm;
    GlobalTensor<int32_t> idxGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> distBuf;
    TBuf<QuePosition::VECCALC> centroidBuf;
    TBuf<QuePosition::VECCALC> tmpBuf1;
    TBuf<QuePosition::VECCALC> tmpBuf3;
    uint32_t B, N, M, blockSize, tileNum;
};

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR x0, GM_ADDR x1, GM_ADDR y0, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFPS op;
    op.Init(x0, x1, y0,
            tiling_data.B, tiling_data.N, tiling_data.M,
            tiling_data.block_size, tiling_data.core_size,
            tiling_data.core_remain);
    op.Process();
}
