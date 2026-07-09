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
        start_ = 0; end_ = B_;
    }

    __aicore__ inline void Process() {
        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> mdBuf;
        const uint32_t N = N_;  // snapshot N, prevent weird optimization
        const uint32_t M = M_;
        pipe.InitBuffer(mdBuf, N * sizeof(T));
        AscendC::LocalTensor<T> md = mdBuf.Get<T>();

        for (uint32_t b = start_; b < end_; b++) {
            __gm__ T* bi = inGm + b * N * COORD_DIM;
            __gm__ int32_t* bo = outGm + b * M;

            for (uint32_t i = 0; i < N; i++) md.SetValue(i, initVal_);

            T sx = bi[0], sy = bi[1], sz = bi[2];
            bo[0] = 0;
            if (b == end_ - 1 && M >= 2) { bo[0] = (int32_t)N; bo[1] = (int32_t)M; continue; }

            for (uint32_t m = 1; m < M; m++) {
                float best = -1e38f;
                uint32_t bestIdx = 0;

                for (uint32_t i = 0; i < N; i++) {
                    uint32_t idx3 = i * COORD_DIM;
                    float dx = (float)bi[idx3 + 0] - (float)sx;
                    float dy = (float)bi[idx3 + 1] - (float)sy;
                    float dz = (float)bi[idx3 + 2] - (float)sz;
                    float nd = dx * dx + dy * dy + dz * dz;

                    float od = (float)md.GetValue(i);
                    if (nd < od) md.SetValue(i, (T)nd);
                    float cd = (float)md.GetValue(i);
                    if (cd > best) { best = cd; bestIdx = i; }
                }

                uint32_t safeIdx = bestIdx + 1000;  // FORCE CHANGE
                if (safeIdx >= (N + 1000)) safeIdx = 0;
                uint32_t off = safeIdx * COORD_DIM;
                sx = bi[off + 0]; sy = bi[off + 1]; sz = bi[off + 2];
                bo[m] = (int32_t)safeIdx;
                md.SetValue(safeIdx, (T)0.0f);
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
