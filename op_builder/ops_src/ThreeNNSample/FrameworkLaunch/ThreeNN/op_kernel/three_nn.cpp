#include "kernel_operator.h"

using namespace AscendC;

#define TILING_KEY_FP16 10000001
#define TILING_KEY_FP32 10000002

constexpr int32_t BUFF_NUM = 2;
constexpr int32_t COORD_DIM = 3;
constexpr int32_t K = 3;

template<typename T>
class KernelThreeNN {
public:
    __aicore__ inline KernelThreeNN() {}

    __aicore__ inline void Init(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR idx,
                                 uint32_t iB, uint32_t iN, uint32_t iM,
                                 uint32_t iNblock, uint32_t iAlignNum,
                                 uint32_t iUsedCoreNum, uint32_t iTotalQuery,
                                 uint32_t iCoreQuerySize, uint32_t iCoreRemain,
                                 uint32_t iFmtB3N) {
        B = iB; N = iN; M = iM;
        N_block = iNblock; alignNum = iAlignNum;
        usedCoreNum = iUsedCoreNum; totalQuery = iTotalQuery;
        coreQuerySize = iCoreQuerySize; coreRemain = iCoreRemain;
        fmtB3N = iFmtB3N;

        uint32_t blockIdx = GetBlockIdx();

        if (blockIdx < coreRemain) {
            coreQueryBegin = blockIdx * (coreQuerySize + 1);
            coreQueryEnd = coreQueryBegin + coreQuerySize + 1;
        } else {
            coreQueryBegin = blockIdx * coreQuerySize + coreRemain;
            coreQueryEnd = coreQueryBegin + coreQuerySize;
        }
        coreQueryNum = coreQueryEnd - coreQueryBegin;

        if (coreQueryNum > 0 && blockIdx < usedCoreNum) {
            uint32_t maxLen = (N_block > M) ? N_block : M;
            uint32_t paddedBlock = fmtB3N
                ? (N_block + alignNum - 1) / alignNum * alignNum
                : (N_block * COORD_DIM + alignNum - 1) / alignNum * alignNum;

            xyz1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xyz1), B * COORD_DIM * N);
            xyz2Gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(xyz2), B * COORD_DIM * M);
            distGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dist), B * M * K);
            idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(idx), B * M * K);

            pipe.InitBuffer(xyz1Que, BUFF_NUM, paddedBlock * sizeof(T));
            pipe.InitBuffer(vecBuf, (COORD_DIM + 7 * N_block) * sizeof(T));
            pipe.InitBuffer(seqBuf, maxLen * COORD_DIM * sizeof(int32_t));
            pipe.InitBuffer(outBuf, K * sizeof(T) + K * sizeof(int32_t));

            BuildSeq(maxLen);
        }
    }

    __aicore__ inline void Process() {
        if (coreQueryNum == 0 || GetBlockIdx() >= usedCoreNum) return;

        for (uint32_t qi = 0; qi < coreQueryNum; qi++) {
            uint32_t gQi = coreQueryBegin + qi;
            uint32_t b = gQi / M;
            uint32_t q = gQi % M;
            ProcessQueryPoint(b, q);
        }
    }

private:
    __aicore__ inline void BuildSeq(uint32_t maxLen) {
        idxSeqBuf = seqBuf.Get<int32_t>(maxLen * COORD_DIM);
        idxSeq0 = idxSeqBuf;
        idxSeq1 = idxSeqBuf[maxLen];
        idxSeq2 = idxSeqBuf[maxLen * 2];

        for (uint32_t i = 0; i < maxLen; i++) {
            idxSeq0.SetValue(i, static_cast<int32_t>(i * COORD_DIM * sizeof(T)));
            idxSeq1.SetValue(i, static_cast<int32_t>(sizeof(T) + i * COORD_DIM * sizeof(T)));
            idxSeq2.SetValue(i, static_cast<int32_t>(2 * sizeof(T) + i * COORD_DIM * sizeof(T)));
        }
    }

    __aicore__ inline void ProcessQueryPoint(uint32_t b, uint32_t q) {
        LocalTensor<T> vBuf = vecBuf.Get<T>(COORD_DIM + 7 * N_block);
        LocalTensor<T> qBuf = vBuf;
        LocalTensor<T> x_buf = qBuf[COORD_DIM];
        LocalTensor<T> y_buf = x_buf[N_block];
        LocalTensor<T> z_buf = y_buf[N_block];
        LocalTensor<T> dx_buf = z_buf[N_block];
        LocalTensor<T> dy_buf = dx_buf[N_block];
        LocalTensor<T> dz_buf = dy_buf[N_block];
        LocalTensor<T> dist_buf = dz_buf[N_block];

        T qx, qy, qz;
        if (fmtB3N) {
            uint64_t baseOff = static_cast<uint64_t>(b) * COORD_DIM * M;
            DataCopy(qBuf, xyz2Gm[baseOff + q], 1);
            qx = (T)qBuf.GetValue(0);
            DataCopy(qBuf, xyz2Gm[baseOff + M + q], 1);
            qy = (T)qBuf.GetValue(0);
            DataCopy(qBuf, xyz2Gm[baseOff + 2 * M + q], 1);
            qz = (T)qBuf.GetValue(0);
        } else {
            uint64_t qOff = (static_cast<uint64_t>(b) * M + q) * COORD_DIM;
            DataCopy(qBuf, xyz2Gm[qOff], COORD_DIM);
            qx = (T)qBuf.GetValue(0);
            qy = (T)qBuf.GetValue(1);
            qz = (T)qBuf.GetValue(2);
        }

        T d0 = (T)65504, d1 = (T)65504, d2 = (T)65504;
        int32_t id0 = 0, id1 = 0, id2 = 0;

        uint32_t N_blk_cnt = (N + N_block - 1) / N_block;
        for (uint32_t blk = 0; blk < N_blk_cnt; blk++) {
            uint32_t r_begin = blk * N_block;
            uint32_t r_count = (r_begin + N_block <= N) ? N_block : (N - r_begin);

            if (fmtB3N) {
                LoadBlockB3N(b, r_begin, r_count, x_buf, y_buf, z_buf);
            } else {
                LoadBlockDeinterleave(b, r_begin, r_count, x_buf, y_buf, z_buf);
            }

            Muls(dx_buf, x_buf, (T)-1.0, r_count);
            Muls(dy_buf, y_buf, (T)-1.0, r_count);
            Muls(dz_buf, z_buf, (T)-1.0, r_count);
            Adds(dx_buf, dx_buf, qx, r_count);
            Adds(dy_buf, dy_buf, qy, r_count);
            Adds(dz_buf, dz_buf, qz, r_count);
            Mul(dx_buf, dx_buf, dx_buf, r_count);
            Mul(dy_buf, dy_buf, dy_buf, r_count);
            Mul(dz_buf, dz_buf, dz_buf, r_count);
            Add(dist_buf, dx_buf, dy_buf, r_count);
            Add(dist_buf, dist_buf, dz_buf, r_count);

            for (uint32_t i = 0; i < r_count; i++) {
                float fd = (float)dist_buf.GetValue(i);
                float fd0 = (float)d0;
                float fd1 = (float)d1;
                float fd2 = (float)d2;
                int32_t gid = static_cast<int32_t>(r_begin + i);
                if (fd < fd0) {
                    d2 = d1; id2 = id1;
                    d1 = d0; id1 = id0;
                    d0 = dist_buf.GetValue(i); id0 = gid;
                } else if (fd < fd1) {
                    d2 = d1; id2 = id1;
                    d1 = dist_buf.GetValue(i); id1 = gid;
                } else if (fd < fd2) {
                    d2 = dist_buf.GetValue(i); id2 = gid;
                }
            }
        }

        LocalTensor<T> oDist = outBuf.Get<T>(K);
        oDist.SetValue(0, d0);
        oDist.SetValue(1, d1);
        oDist.SetValue(2, d2);

        uint32_t oByte = K * sizeof(T);
        LocalTensor<uint8_t> oByteBuf = outBuf.Get<uint8_t>(oByte + K * sizeof(int32_t));
        LocalTensor<int32_t> oIdx = oByteBuf[oByte].ReinterpretCast<int32_t>();
        oIdx.SetValue(0, id0);
        oIdx.SetValue(1, id1);
        oIdx.SetValue(2, id2);

        uint64_t outOff = (static_cast<uint64_t>(b) * M + q) * K;
        DataCopy(distGm[outOff], oDist, K);
        DataCopy(idxGm[outOff], oIdx, K);
    }

    __aicore__ inline void LoadBlockB3N(uint32_t b, uint32_t r_begin, uint32_t r_count,
                                         LocalTensor<T>& x_buf, LocalTensor<T>& y_buf, LocalTensor<T>& z_buf) {
        uint64_t baseOff = static_cast<uint64_t>(b) * COORD_DIM * N;
        uint32_t r_aligned = (r_count + alignNum - 1) / alignNum * alignNum;

        auto tap = xyz1Que.AllocTensor<T>();
        DataCopy(tap, xyz1Gm[baseOff + r_begin], r_aligned);
        xyz1Que.EnQue(tap);
        tap = xyz1Que.DeQue<T>();
        DataCopy(x_buf, tap, r_count);
        xyz1Que.FreeTensor(tap);

        tap = xyz1Que.AllocTensor<T>();
        DataCopy(tap, xyz1Gm[baseOff + N + r_begin], r_aligned);
        xyz1Que.EnQue(tap);
        tap = xyz1Que.DeQue<T>();
        DataCopy(y_buf, tap, r_count);
        xyz1Que.FreeTensor(tap);

        tap = xyz1Que.AllocTensor<T>();
        DataCopy(tap, xyz1Gm[baseOff + 2 * N + r_begin], r_aligned);
        xyz1Que.EnQue(tap);
        tap = xyz1Que.DeQue<T>();
        DataCopy(z_buf, tap, r_count);
        xyz1Que.FreeTensor(tap);
    }

    __aicore__ inline void LoadBlockDeinterleave(uint32_t b, uint32_t r_begin, uint32_t r_count,
                                                   LocalTensor<T>& x_buf, LocalTensor<T>& y_buf, LocalTensor<T>& z_buf) {
        uint32_t xyz1Off = b * N * COORD_DIM + r_begin * COORD_DIM;
        uint32_t r_aligned = (r_count * COORD_DIM + alignNum - 1) / alignNum * alignNum;

        auto tap = xyz1Que.AllocTensor<T>();
        DataCopy(tap, xyz1Gm[xyz1Off], r_aligned);
        xyz1Que.EnQue(tap);
        tap = xyz1Que.DeQue<T>();

        Gather(x_buf, tap, idxSeq0.ReinterpretCast<uint32_t>(), 0, r_count);
        Gather(y_buf, tap, idxSeq1.ReinterpretCast<uint32_t>(), 0, r_count);
        Gather(z_buf, tap, idxSeq2.ReinterpretCast<uint32_t>(), 0, r_count);

        xyz1Que.FreeTensor(tap);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFF_NUM> xyz1Que;
    TBuf<QuePosition::VECCALC> vecBuf;
    TBuf<QuePosition::VECCALC> seqBuf;
    TBuf<QuePosition::VECCALC> outBuf;

    GlobalTensor<T> xyz1Gm;
    GlobalTensor<T> xyz2Gm;
    GlobalTensor<T> distGm;
    GlobalTensor<int32_t> idxGm;

    LocalTensor<int32_t> idxSeqBuf;
    LocalTensor<int32_t> idxSeq0, idxSeq1, idxSeq2;

    uint32_t B, N, M, N_block, alignNum;
    uint32_t usedCoreNum, totalQuery;
    uint32_t coreQuerySize, coreRemain;
    uint32_t coreQueryBegin, coreQueryEnd, coreQueryNum;
    uint32_t fmtB3N;
};

extern "C" __global__ __aicore__ void three_nn(
        GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR idx,
        GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(TILING_KEY_FP16)) {
        KernelThreeNN<half> op;
        op.Init(xyz1, xyz2, dist, idx,
                tilingData.B, tilingData.N, tilingData.M,
                tilingData.N_block, tilingData.alignNum,
                tilingData.usedCoreNum, tilingData.totalQuery,
                tilingData.coreQuerySize, tilingData.coreRemain,
                tilingData.formatB3N);
        op.Process();
    } else {
        KernelThreeNN<float> op;
        op.Init(xyz1, xyz2, dist, idx,
                tilingData.B, tilingData.N, tilingData.M,
                tilingData.N_block, tilingData.alignNum,
                tilingData.usedCoreNum, tilingData.totalQuery,
                tilingData.coreQuerySize, tilingData.coreRemain,
                tilingData.formatB3N);
        op.Process();
    }
}
