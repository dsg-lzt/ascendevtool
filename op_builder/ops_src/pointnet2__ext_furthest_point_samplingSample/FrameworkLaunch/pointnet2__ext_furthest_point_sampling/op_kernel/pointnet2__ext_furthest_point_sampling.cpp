#include "kernel_operator.h"

constexpr int32_t BUF_NUM = 2;
constexpr int32_t COORD_DIM = 3;

template<typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t tileN, uint32_t bpc,
                                uint32_t crem, float iv) {
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
        pipe.InitBuffer(mdBuf, N_ * sizeof(T));
        pipe.InitBuffer(wBuf, tileN_ * sizeof(T) * 2);
    }

    __aicore__ inline void Process() {
        for (uint32_t b = start_; b < end_; b++) ProcessBatch(b);
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t b) {
        AscendC::LocalTensor<T> mdAll = mdBuf.Get<T>();
        for (uint32_t i = 0; i < N_; i++) mdAll.SetValue(i, initVal_);

        AscendC::LocalTensor<T> cBuf = wBuf.Get<T>(COORD_DIM);
        AscendC::DataCopy(cBuf, inGm[b * N_ * COORD_DIM], COORD_DIM);
        T cx = cBuf.GetValue(0), cy = cBuf.GetValue(1), cz = cBuf.GetValue(2);
        outGm.SetValue(b * M_, 0);

        int32_t nTiles = ((int32_t)N_ + (int32_t)tileN_ - 1) / (int32_t)tileN_;

        for (int32_t m = 1; m < (int32_t)M_; m++) {
            float gMax = -65504.0f;
            uint32_t gIdx = 0, selIdx = 0;

            for (int32_t t = 0; t < nTiles; t++) {
                int32_t tStart = t * tileN_;
                int32_t curN = (tStart + (int32_t)tileN_ <= (int32_t)N_) ? tileN_ : (N_ - tStart);
                if (curN <= 0) continue;

                AscendC::LocalTensor<T> xTile = qX.AllocTensor<T>();
                uint64_t off = static_cast<uint64_t>(b) * N_ * COORD_DIM + tStart * COORD_DIM;
                AscendC::DataCopy(xTile, inGm[off], curN * COORD_DIM);
                qX.EnQue(xTile);
                xTile = qX.DeQue<T>();

                AscendC::LocalTensor<T> dBuf = wBuf.Get<T>(curN);
                AscendC::LocalTensor<T> sBuf = wBuf.Get<T>(curN + tileN_);

                float lMaxF = -65504.0f;
                uint32_t lIdx = 0;

                for (int32_t i = 0; i < curN; i++) {
                    T dx = xTile.GetValue(i * COORD_DIM) - cx;
                    T dy = xTile.GetValue(i * COORD_DIM + 1) - cy;
                    T dz = xTile.GetValue(i * COORD_DIM + 2) - cz;
                    float nd = (float)dx * (float)dx + (float)dy * (float)dy + (float)dz * (float)dz;
                    T ndT = (T)nd;
                    dBuf.SetValue(i, ndT);

                    T oldD = mdAll.GetValue(tStart + i);
                    if ((float)ndT < (float)oldD) mdAll.SetValue(tStart + i, ndT);
                    float fv = (float)mdAll.GetValue(tStart + i);
                    if (fv > lMaxF) { lMaxF = fv; lIdx = i; }
                }

                if (lMaxF > gMax) { gMax = lMaxF; gIdx = tStart + lIdx; }

                qX.FreeTensor(xTile);
            }

            outGm.SetValue(b * M_ + m, (int32_t)gIdx);
            AscendC::DataCopy(cBuf, inGm[b * N_ * COORD_DIM + gIdx * COORD_DIM], COORD_DIM);
            cx = cBuf.GetValue(0); cy = cBuf.GetValue(1); cz = cBuf.GetValue(2);
            mdAll.SetValue(gIdx, (T)0.0f);
        }
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> qX;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> mdBuf, wBuf;
    AscendC::GlobalTensor<T> inGm;
    AscendC::GlobalTensor<int32_t> outGm;
    uint32_t B_, N_, M_, tileN_, start_, end_;
    T initVal_;
};

extern "C" __global__ __aicore__ void pointnet2__ext_furthest_point_sampling(
    GM_ADDR p, GM_ADDR s, GM_ADDR ws, GM_ADDR tl) {
    GET_TILING_DATA(td, tl);
    if (TILING_KEY_IS(0)) {
        KernelFPS<float> op;
        op.Init(p, s, td.B, td.N, td.M, td.ubPointsNum, td.batchPerCore, td.coreRemainder, td.initVal);
        op.Process();
    }
}
