#include "kernel_operator.h"

constexpr int32_t BUF_NUM = 2;
constexpr int32_t COORD_DIM = 3;

template<typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t tileN, uint32_t bpc, uint32_t crem, float iv) {
        B_ = B; N_ = N; M_ = M; tileN_ = tileN; initVal_ = (T)iv;
        if (tileN_ > N_) tileN_ = N_;
        if (tileN_ < 64) tileN_ = 64;

        inGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p), B_ * N_ * COORD_DIM);
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(s), B_ * M_);

        uint32_t bi = AscendC::GetBlockIdx();
        if (bi < crem) { start_ = bi * (bpc + 1); end_ = start_ + bpc + 1; }
        else { start_ = crem * (bpc + 1) + (bi - crem) * bpc; end_ = start_ + bpc; }
        if (start_ >= B_) end_ = start_;
        if (end_ > B_) end_ = B_;

        pipe.InitBuffer(qX, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(md, N_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        for (uint32_t b = start_; b < end_; b++) ProcessBatch(b);
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t b) {
        AscendC::LocalTensor<T> mdAll = md.Get<T>();
        for (uint32_t i = 0; i < N_; i++) mdAll.SetValue(i, initVal_);

        uint64_t bo = (uint64_t)b * N_ * COORD_DIM;
        AscendC::LocalTensor<T> cBuf = qX.AllocTensor<T>();
        AscendC::DataCopy(cBuf, inGm[bo], COORD_DIM);
        qX.EnQue(cBuf);
        cBuf = qX.DeQue<T>();
        T cx = cBuf.GetValue(0), cy = cBuf.GetValue(1), cz = cBuf.GetValue(2);
        outGm.SetValue(b * M_, 0);

        int32_t nTiles = (N_ + tileN_ - 1) / tileN_;
        for (int32_t m = 1; m < (int32_t)M_; m++) {
            float gMax = -65504.0f;
            uint32_t gIdx = 0;

            for (int32_t t = 0; t < nTiles; t++) {
                int32_t ts = t * tileN_;
                int32_t cN = (ts + (int32_t)tileN_ <= (int32_t)N_) ? tileN_ : (N_ - ts);
                if (cN <= 0) continue;

                uint64_t off = bo + ts * COORD_DIM;
                AscendC::LocalTensor<T> xTile = qX.AllocTensor<T>();
                AscendC::DataCopy(xTile, inGm[off], cN * COORD_DIM);
                qX.EnQue(xTile);
                xTile = qX.DeQue<T>();

                float lM = -65504.0f;
                uint32_t lI = 0;
                for (int32_t i = 0; i < cN; i++) {
                    float dx = (float)xTile.GetValue(i * COORD_DIM) - (float)cx;
                    float dy = (float)xTile.GetValue(i * COORD_DIM + 1) - (float)cy;
                    float dz = (float)xTile.GetValue(i * COORD_DIM + 2) - (float)cz;
                    float nd = dx * dx + dy * dy + dz * dz;
                    float od = (float)mdAll.GetValue(ts + i);
                    if (nd < od) mdAll.SetValue(ts + i, (T)nd);
                    float cd = (float)mdAll.GetValue(ts + i);
                    if (cd > lM) { lM = cd; lI = i; }
                }
                if (lM > gMax) { gMax = lM; gIdx = ts + lI; }
                qX.FreeTensor(xTile);
            }

            outGm.SetValue(b * M_ + m, (int32_t)gIdx);
            uint64_t co = bo + gIdx * COORD_DIM;
            cBuf = qX.AllocTensor<T>();
            AscendC::DataCopy(cBuf, inGm[co], COORD_DIM);
            qX.EnQue(cBuf);
            cBuf = qX.DeQue<T>();
            cx = cBuf.GetValue(0); cy = cBuf.GetValue(1); cz = cBuf.GetValue(2);
            mdAll.SetValue(gIdx, (T)0.0f);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> qX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> md;
    AscendC::GlobalTensor<T> inGm;
    AscendC::GlobalTensor<int32_t> outGm;
    uint32_t B_, N_, M_, tileN_, start_, end_;
    T initVal_;
};

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(td, tiling);
    if (TILING_KEY_IS(0)) {
        KernelFPS<float> op;
        op.Init(input, output, td.B, td.N, td.M, td.tileN, td.batchPerCore, td.coreRem, td.initVal);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelFPS<half> op;
        op.Init(input, output, td.B, td.N, td.M, td.tileN, td.batchPerCore, td.coreRem, td.initVal);
        op.Process();
    }
}
