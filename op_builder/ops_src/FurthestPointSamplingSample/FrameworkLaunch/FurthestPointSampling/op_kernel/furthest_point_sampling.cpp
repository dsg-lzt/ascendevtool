#include "kernel_operator.h"

constexpr int32_t COORD_DIM = 3;

template<typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t tileN, uint32_t bpc, uint32_t crem, float iv) {
        B_ = B; N_ = N; M_ = M; initVal_ = (T)iv;

        inGm = reinterpret_cast<__gm__ T*>(p);
        outGm = reinterpret_cast<__gm__ int32_t*>(s);

        uint32_t bi = AscendC::GetBlockIdx();
        if (bi < crem) { start_ = bi * (bpc + 1); end_ = start_ + bpc + 1; }
        else { start_ = crem * (bpc + 1) + (bi - crem) * bpc; end_ = start_ + bpc; }
        if (start_ >= B_) end_ = start_;
        if (end_ > B_) end_ = B_;
    }

    __aicore__ inline void Process() {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> mdBuf;
        pipe.InitBuffer(mdBuf, N_ * sizeof(T));
        AscendC::LocalTensor<T> md = mdBuf.Get<T>();

        for (uint32_t b = start_; b < end_; b++) {
            __gm__ T* bi = inGm + b * N_ * COORD_DIM;
            __gm__ int32_t* bo = outGm + b * M_;

            for (uint32_t i = 0; i < N_; i++) md.SetValue(i, initVal_);

            T sx = bi[0], sy = bi[1], sz = bi[2];
            bo[0] = 0;
            uint32_t selIdx = 0;

            for (int32_t m = 1; m < (int32_t)M_; m++) {
                float gMax = -1e38f;
                uint32_t gIdx = 0;

                for (uint32_t i = 0; i < N_; i++) {
                    float dx = (float)bi[i * COORD_DIM + 0] - (float)sx;
                    float dy = (float)bi[i * COORD_DIM + 1] - (float)sy;
                    float dz = (float)bi[i * COORD_DIM + 2] - (float)sz;
                    float nd = dx * dx + dy * dy + dz * dz;

                    float od = (float)md.GetValue(i);
                    if (nd < od) md.SetValue(i, (T)nd);

                    float cd = (float)md.GetValue(i);
                    if (cd > gMax) { gMax = cd; gIdx = i; }
                }

                if (gIdx >= N_) gIdx = 0;

                selIdx = gIdx;
                uint32_t off = selIdx * COORD_DIM;
                sx = bi[off + 0]; sy = bi[off + 1]; sz = bi[off + 2];
                bo[m] = (int32_t)selIdx;
                md.SetValue(selIdx, (T)0.0f);
            }
        }
    }

private:
    __gm__ T* inGm;
    __gm__ int32_t* outGm;
    uint32_t B_, N_, M_, start_, end_;
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
