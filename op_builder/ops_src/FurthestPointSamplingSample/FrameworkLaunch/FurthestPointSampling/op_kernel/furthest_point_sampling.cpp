#include "kernel_operator.h"
#include "furthest_point_sampling_tiling.h"

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

        inGm = reinterpret_cast<__gm__ T*>(p);
        outGm = reinterpret_cast<__gm__ int32_t*>(s);

        uint32_t bi = AscendC::GetBlockIdx();
        if (bi < crem) { start_ = bi * (bpc + 1); end_ = start_ + bpc + 1; }
        else { start_ = crem * (bpc + 1) + (bi - crem) * bpc; end_ = start_ + bpc; }
        if (start_ >= B_) end_ = start_;
        if (end_ > B_) end_ = B_;
    }

    __aicore__ inline void Process() {
        for (uint32_t b = start_; b < end_; b++) ProcessBatch(b);
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t b) {
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> qX, qY, qZ;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> mdBuf, bufDist, bufTmp, bufSca;

        pipe.InitBuffer(qX, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qY, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(qZ, BUF_NUM, tileN_ * sizeof(T));
        pipe.InitBuffer(mdBuf,   N_ * sizeof(T));
        pipe.InitBuffer(bufDist, tileN_ * sizeof(T));
        pipe.InitBuffer(bufTmp,  tileN_ * sizeof(T));
        pipe.InitBuffer(bufSca,  tileN_ * sizeof(T));

        AscendC::LocalTensor<T> mdAll = mdBuf.Get<T>();
        __gm__ T* batchIn = inGm + b * N_ * COORD_DIM;
        __gm__ int32_t* batchOut = outGm + b * M_;

        for (uint32_t i = 0; i < N_; i++) mdAll.SetValue(i, initVal_);

        T selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        batchOut[0] = 0;
        uint32_t selIdx = 0;

        int32_t nTiles = (N_ + tileN_ - 1) / tileN_;
        for (int32_t m = 1; m < (int32_t)M_; m++) {
            T globalMax = (T)(-65504.0);
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < nTiles; t++) {
                int32_t ts = t * tileN_;
                int32_t cN = (ts + tileN_ <= N_) ? tileN_ : (N_ - ts);
                if (cN <= 0) continue;

                AscendC::LocalTensor<T> xLoc = qX.AllocTensor<T>();
                AscendC::LocalTensor<T> yLoc = qY.AllocTensor<T>();
                AscendC::LocalTensor<T> zLoc = qZ.AllocTensor<T>();
                for (int32_t i = 0; i < cN; i++) {
                    int32_t gi = ts + i;
                    xLoc.SetValue(i, batchIn[gi * COORD_DIM + 0]);
                    yLoc.SetValue(i, batchIn[gi * COORD_DIM + 1]);
                    zLoc.SetValue(i, batchIn[gi * COORD_DIM + 2]);
                }
                qX.EnQue(xLoc); qY.EnQue(yLoc); qZ.EnQue(zLoc);
                AscendC::LocalTensor<T> xTile = qX.DeQue<T>();
                AscendC::LocalTensor<T> yTile = qY.DeQue<T>();
                AscendC::LocalTensor<T> zTile = qZ.DeQue<T>();
                AscendC::LocalTensor<T> dist = bufDist.Get<T>();
                AscendC::LocalTensor<T> tmp  = bufTmp.Get<T>();
                AscendC::LocalTensor<T> sca  = bufSca.Get<T>();

                AscendC::Duplicate(sca, selX, cN);
                AscendC::Sub(dist, xTile, sca, cN);
                AscendC::Mul(dist, dist, dist, cN);
                AscendC::Duplicate(sca, selY, cN);
                AscendC::Sub(tmp, yTile, sca, cN);
                AscendC::Mul(tmp, tmp, tmp, cN);
                AscendC::Add(dist, dist, tmp, cN);
                AscendC::Duplicate(sca, selZ, cN);
                AscendC::Sub(tmp, zTile, sca, cN);
                AscendC::Mul(tmp, tmp, tmp, cN);
                AscendC::Add(dist, dist, tmp, cN);

                AscendC::Min(mdAll, mdAll, dist, cN);

                float localMax = -65504.0f;
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < cN; i++) {
                    float v = (float)mdAll.GetValue(i);
                    if (v > localMax) { localMax = v; localIdx = i; }
                }
                if (localMax > (float)globalMax) {
                    globalMax = (T)localMax;
                    globalIdx = ts + localIdx;
                }
                qX.FreeTensor(xTile); qY.FreeTensor(yTile); qZ.FreeTensor(zTile);
            }

            selIdx = globalIdx;
            int32_t off = selIdx * COORD_DIM;
            selX = batchIn[off + 0]; selY = batchIn[off + 1]; selZ = batchIn[off + 2];
            batchOut[m] = (int32_t)selIdx;
            mdAll.SetValue(selIdx, (T)0.0f);
        }
    }

    __gm__ T* inGm;
    __gm__ int32_t* outGm;
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
