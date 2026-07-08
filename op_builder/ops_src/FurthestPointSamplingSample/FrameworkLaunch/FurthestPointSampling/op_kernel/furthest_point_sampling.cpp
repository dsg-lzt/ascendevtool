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
        B_ = static_cast<int32_t>(B);
        N_ = static_cast<int32_t>(N);
        M_ = static_cast<int32_t>(M);
        tileN_ = static_cast<int32_t>(tileN);
        if (tileN_ > N_) tileN_ = N_;
        if (tileN_ < 64) tileN_ = 64;
        initVal_ = static_cast<T>(iv);

        inGm = reinterpret_cast<__gm__ T*>(p);
        outGm = reinterpret_cast<__gm__ int32_t*>(s);

        int32_t blockIdx = AscendC::GetBlockIdx();
        start_ = blockIdx * static_cast<int32_t>(bpc);
        end_ = start_ + static_cast<int32_t>(bpc);
        if (start_ >= B_) end_ = start_;
        if (end_ > B_) end_ = B_;
    }

    __aicore__ inline void Process() {
        for (int32_t b = start_; b < end_; ++b) ProcessBatch(b);
    }

private:
    __aicore__ inline void ProcessBatch(int32_t batchIdx) {
        __gm__ T* batchIn = inGm + batchIdx * N_ * COORD_DIM;
        __gm__ int32_t* batchOut = outGm + batchIdx * M_;

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
        for (int32_t i = 0; i < N_; ++i) mdAll.SetValue(i, initVal_);

        T selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        batchOut[0] = 0;
        uint32_t selIdx = 0;
        int32_t numTiles = (N_ + tileN_ - 1) / tileN_;

        for (int32_t m = 1; m < M_; ++m) {
            T globalMax = static_cast<T>(-65504.0);
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);
                if (curN <= 0) continue;

                AscendC::LocalTensor<T> xLoc = qX.AllocTensor<T>();
                AscendC::LocalTensor<T> yLoc = qY.AllocTensor<T>();
                AscendC::LocalTensor<T> zLoc = qZ.AllocTensor<T>();
                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
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
                T localMax = static_cast<T>(-65504.0);
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    T v = mdAll.GetValue(i);
                    float fv = static_cast<float>(v);
                    if (fv > localMaxF) { localMaxF = fv; localMax = v; localIdx = i; }
                }
                if (localMaxF > static_cast<float>(globalMax)) {
                    globalMax = localMax;
                    globalIdx = tStart + localIdx;
                }
                qX.FreeTensor(xTile); qY.FreeTensor(yTile); qZ.FreeTensor(zTile);
            }

            selIdx = globalIdx;
            int32_t off = selIdx * COORD_DIM;
            selX = batchIn[off + 0]; selY = batchIn[off + 1]; selZ = batchIn[off + 2];
            batchOut[m] = static_cast<int32_t>(selIdx);
            mdAll.SetValue(selIdx, static_cast<T>(0.0f));
        }
    }

    int32_t B_, N_, M_, tileN_, start_, end_;
    T initVal_;
    __gm__ T* inGm;
    __gm__ int32_t* outGm;
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
