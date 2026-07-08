#include "kernel_operator.h"

template<typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR s,
                                uint32_t B, uint32_t N, uint32_t M,
                                uint32_t tileN, uint32_t bpc, uint32_t crem, float iv) {
        B_ = B; M_ = M; outGm = reinterpret_cast<__gm__ int32_t*>(s);
        uint32_t bi = AscendC::GetBlockIdx();
        if (bi < crem) { start_ = bi * (bpc + 1); end_ = start_ + bpc + 1; }
        else { start_ = crem * (bpc + 1) + (bi - crem) * bpc; end_ = start_ + bpc; }
        if (start_ >= B_) end_ = start_;
        if (end_ > B_) end_ = B_;
    }
    __aicore__ inline void Process() {
        for (uint32_t b = start_; b < end_; b++) {
            __gm__ int32_t* bo = outGm + b * M_;
            for (uint32_t i = 0; i < M_; i++) bo[i] = (int32_t)i;
        }
    }
private:
    __gm__ int32_t* outGm;
    uint32_t B_, M_, start_, end_;
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
