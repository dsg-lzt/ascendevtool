/**
 * FurthestPointSampling - AscendC Kernel
 *
 * Input:  points  [B, N, 3]  float32 | float16
 * Output: indices [B, M]     int32
 *
 * minDist stored in GM workspace for correct tile-level access.
 */
#include "kernel_operator.h"

constexpr int32_t COORD_DIM = 3;

template <typename T>
class KernelFPS {
public:
    __aicore__ inline KernelFPS() {}

    __aicore__ inline void Init(
        GM_ADDR pointsGm, GM_ADDR sampledGm, GM_ADDR wsGm,
        uint32_t ubPN, uint32_t /*ubMDN*/,
        uint32_t B, uint32_t N, uint32_t M,
        uint32_t C, uint32_t batchPerCore,
        uint32_t coreRemainder, uint32_t wsStride,
        float initVal)
    {
        B_ = static_cast<int32_t>(B);
        N_ = static_cast<int32_t>(N);
        M_ = static_cast<int32_t>(M);
        C_ = static_cast<int32_t>(C);
        tileN_ = static_cast<int32_t>(ubPN);
        initVal_ = static_cast<T>(initVal);

        inputGm_ = reinterpret_cast<__gm__ T*>(pointsGm);
        outputGm_ = reinterpret_cast<__gm__ int32_t*>(sampledGm);
        wsGm_ = reinterpret_cast<__gm__ float*>(wsGm);

        int32_t blockIdx = AscendC::GetBlockIdx();
        batchStart_ = 0;
        batchEnd_ = 0;

        if (static_cast<uint32_t>(blockIdx) < coreRemainder) {
            batchStart_ = blockIdx * static_cast<int32_t>(batchPerCore + 1);
            batchEnd_ = batchStart_ + static_cast<int32_t>(batchPerCore) + 1;
        } else {
            batchStart_ = static_cast<int32_t>(coreRemainder * (batchPerCore + 1)
                + (blockIdx - static_cast<int32_t>(coreRemainder)) * batchPerCore);
            batchEnd_ = batchStart_ + static_cast<int32_t>(batchPerCore);
        }
        if (batchStart_ >= static_cast<int32_t>(B)) batchEnd_ = batchStart_;
        if (batchEnd_ > static_cast<int32_t>(B)) batchEnd_ = B;
    }

    __aicore__ inline void Process() {
        for (int32_t b = batchStart_; b < batchEnd_; ++b) {
            ProcessBatch(b);
        }
    }

private:
    __aicore__ inline void ProcessBatch(int32_t batchIdx) {
        __gm__ T* batchIn = inputGm_ + batchIdx * N_ * C_;
        __gm__ int32_t* batchOut = outputGm_ + batchIdx * M_;
        __gm__ float* batchWs = wsGm_ + batchIdx * N_;

        AscendC::TPipe pipe;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufDist;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufTmp;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufSca;
        AscendC::TBuf<AscendC::QuePosition::VECCALC> bufMd;

        pipe.InitBuffer(bufDist, tileN_ * sizeof(T));
        pipe.InitBuffer(bufTmp, tileN_ * sizeof(T));
        pipe.InitBuffer(bufSca, tileN_ * sizeof(T));
        pipe.InitBuffer(bufMd, tileN_ * sizeof(float));

        AscendC::LocalTensor<T> dist = bufDist.Get<T>();
        AscendC::LocalTensor<T> tmp = bufTmp.Get<T>();
        AscendC::LocalTensor<T> sca = bufSca.Get<T>();
        AscendC::LocalTensor<float> mdLocal = bufMd.Get<float>();

        T selX = batchIn[0], selY = batchIn[1], selZ = batchIn[2];
        batchOut[0] = 0;

        int32_t numTiles = (N_ + tileN_ - 1) / tileN_;

        for (int32_t m = 1; m < M_; ++m) {
            float globalMax = -65504.0f;
            uint32_t globalIdx = 0;

            for (int32_t t = 0; t < numTiles; ++t) {
                int32_t tStart = t * tileN_;
                int32_t curN = (tStart + tileN_ <= N_) ? tileN_ : (N_ - tStart);

                // Load current minDist from GM workspace at correct offset
                // Also load xyz for this tile
                for (int32_t i = 0; i < curN; ++i) {
                    int32_t gi = tStart + i;
                    float mdVal;
                    if (m == 1) {
                        mdVal = (sizeof(T) == 4) ? 3.402823e+38f : 65504.0f;
                    } else {
                        mdVal = batchWs[tStart + i];
                    }
                    mdLocal.SetValue(i, mdVal);

                    dist.SetValue(i, static_cast<T>(0));
                    float dx = static_cast<float>(batchIn[gi * C_ + 0]) - static_cast<float>(selX);
                    float dy = static_cast<float>(batchIn[gi * C_ + 1]) - static_cast<float>(selY);
                    float dz = static_cast<float>(batchIn[gi * C_ + 2]) - static_cast<float>(selZ);
                    float d = dx * dx + dy * dy + dz * dz;
                    if (d < mdLocal.GetValue(i)) {
                        mdLocal.SetValue(i, d);
                    }
                }

                // Find max in this tile
                float localMax = -65504.0f;
                uint32_t localIdx = 0;
                for (int32_t i = 0; i < curN; ++i) {
                    float fv = mdLocal.GetValue(i);
                    if (fv > localMax) { localMax = fv; localIdx = static_cast<uint32_t>(i); }
                }
                if (localMax > globalMax) {
                    globalMax = localMax;
                    globalIdx = static_cast<uint32_t>(tStart) + localIdx;
                }

                // Write updated minDist back to GM workspace
                for (int32_t i = 0; i < curN; ++i) {
                    batchWs[tStart + i] = mdLocal.GetValue(i);
                }
            }

            int32_t off = static_cast<int32_t>(globalIdx) * C_;
            selX = batchIn[off + 0];
            selY = batchIn[off + 1];
            selZ = batchIn[off + 2];
            batchOut[m] = static_cast<int32_t>(globalIdx);

            batchWs[static_cast<int32_t>(globalIdx)] = 0.0f;
        }
    }

    int32_t B_, N_, M_, C_, tileN_;
    T initVal_;
    int32_t batchStart_, batchEnd_;
    __gm__ T* inputGm_;
    __gm__ int32_t* outputGm_;
    __gm__ float* wsGm_;
};

extern "C" __global__ __aicore__ void furthest_point_sampling(
    GM_ADDR points, GM_ADDR sampled, GM_ADDR workspace, GM_ADDR tiling)
{
    if (TILING_KEY_IS(0)) {
        KernelFPS<float> op;
        GET_TILING_DATA(fpsTiling, tiling);
        op.Init(points, sampled, workspace,
                fpsTiling.ubPointsNum, fpsTiling.ubMinDistNum,
                fpsTiling.B, fpsTiling.N, fpsTiling.M,
                fpsTiling.C, fpsTiling.batchPerCore,
                fpsTiling.coreRemainder, fpsTiling.wsStride,
                fpsTiling.initVal);
        op.Process();
    } else if (TILING_KEY_IS(1)) {
        KernelFPS<half> op;
        GET_TILING_DATA(fpsTiling, tiling);
        op.Init(points, sampled, workspace,
                fpsTiling.ubPointsNum, fpsTiling.ubMinDistNum,
                fpsTiling.B, fpsTiling.N, fpsTiling.M,
                fpsTiling.C, fpsTiling.batchPerCore,
                fpsTiling.coreRemainder, fpsTiling.wsStride,
                fpsTiling.initVal);
        op.Process();
    }
}

#ifndef ASCENDC_CPU_DEBUG
void furthest_point_sampling_do(uint32_t blockDim, void* l2ctrl, void* stream,
                                uint8_t* points, uint8_t* sampled,
                                uint8_t* workspace, uint8_t* tiling)
{
    furthest_point_sampling<<<blockDim, l2ctrl, stream>>>(
        points, sampled, workspace, tiling);
}
#endif
