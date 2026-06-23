#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

class KernelMaxPool2d
{
public:
    __aicore__ inline KernelMaxPool2d() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t n, uint32_t c1, uint32_t h, uint32_t w, uint32_t c0,
                                uint32_t out_h, uint32_t out_w,
                                uint32_t row_total, uint32_t out_total,
                                uint32_t tile_ow,
                                uint32_t core_row_size, uint32_t core_row_remain)
    {
        this->n = n;
        this->c1 = c1;
        this->h = h;
        this->w = w;
        this->c0 = c0;
        this->out_h = out_h;
        this->out_w = out_w;
        this->rowTotal = row_total;
        this->out_total = out_total;
        this->tileOw = tile_ow;
        this->coreRowSize = core_row_size;
        this->coreRowRemain = core_row_remain;

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum = AscendC::GetBlockNum();

        this->coreRowStart = this->coreRowSize * blockIdx;
        this->coreRowCount = this->coreRowSize + ((blockIdx + 1 == blockNum) ? this->coreRowRemain : 0);
        if (this->coreRowStart >= this->rowTotal)
        {
            this->coreRowCount = 0;
        }
        else if (this->coreRowStart + this->coreRowCount > this->rowTotal)
        {
            this->coreRowCount = this->rowTotal - this->coreRowStart;
        }

        const uint32_t inputTotal = this->n * this->c1 * this->h * this->w * this->c0;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, inputTotal);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y, this->out_total);

        const uint32_t inScalarLenMax = this->tileOw * 2 * this->c0;
        const uint32_t outScalarLenMax = this->tileOw * this->c0;
        pipe.InitBuffer(inQueueRow0, BUFFER_NUM, inScalarLenMax * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueRow1, BUFFER_NUM, inScalarLenMax * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, outScalarLenMax * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, outScalarLenMax * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf1, outScalarLenMax * sizeof(DTYPE_Y));
    }

    __aicore__ inline void Process()
    {
        if (this->coreRowCount == 0 || this->out_w == 0)
        {
            return;
        }
        for (uint32_t r = 0; r < this->coreRowCount; ++r)
        {
            const uint32_t rowIdx = this->coreRowStart + r;

            uint32_t t = rowIdx;
            const uint32_t oh = t % this->out_h;
            t = t / this->out_h;
            const uint32_t c1Idx = t % this->c1;
            const uint32_t batch = t / this->c1;

            const uint32_t inRowStride = this->w * this->c0;
            const uint32_t outRowStride = this->out_w * this->c0;
            const uint32_t inBase0 = (((batch * this->c1 + c1Idx) * this->h + (oh << 1)) * this->w) * this->c0;
            const uint32_t inBase1 = inBase0 + inRowStride;
            const uint32_t outBase = (((batch * this->c1 + c1Idx) * this->out_h + oh) * this->out_w) * this->c0;

            const uint32_t tileNum = this->out_w / this->tileOw + ((this->out_w % this->tileOw) > 0);
            if (tileNum == 0)
            {
                continue;
            }

            // 预取第 0 个 tile
            CopyIn(inBase0, inBase1, 0);
            for (uint32_t tileIdx = 0; tileIdx < tileNum; ++tileIdx)
            {
                if (tileIdx + 1 < tileNum)
                {
                    CopyIn(inBase0, inBase1, tileIdx + 1);
                }
                Compute(tileIdx);
                CopyOut(outBase, tileIdx);
            }
        }
    }

private:
    __aicore__ inline uint32_t TileLenOw(uint32_t tileIdx) const
    {
        const uint32_t owStart = tileIdx * this->tileOw;
        const uint32_t remain = this->out_w - owStart;
        return remain >= this->tileOw ? this->tileOw : remain;
    }

    __aicore__ inline void CopyIn(uint32_t inBase0, uint32_t inBase1, uint32_t tileIdx)
    {
        const uint32_t tileLenOw = TileLenOw(tileIdx);
        const uint32_t inScalarLen = tileLenOw * 2 * this->c0;
        const uint32_t inOffset = (tileIdx * this->tileOw * 2) * this->c0;

        AscendC::LocalTensor<DTYPE_X> xRow0 = inQueueRow0.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> xRow1 = inQueueRow1.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xRow0, xGm[inBase0 + inOffset], inScalarLen);
        AscendC::DataCopy(xRow1, xGm[inBase1 + inOffset], inScalarLen);
        inQueueRow0.EnQue(xRow0);
        inQueueRow1.EnQue(xRow1);
    }

    __aicore__ inline void Compute(uint32_t tileIdx)
    {
        const uint32_t tileLenOw = TileLenOw(tileIdx);
        const uint32_t outScalarLen = tileLenOw * this->c0;

        AscendC::LocalTensor<DTYPE_X> xRow0 = inQueueRow0.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> xRow1 = inQueueRow1.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> tmp0 = tmpBuf0.Get<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> tmp1 = tmpBuf1.Get<DTYPE_Y>();

        for (uint32_t ow = 0; ow < tileLenOw; ++ow)
        {
            const uint32_t inW0 = (ow << 1) * this->c0;
            const uint32_t inW1 = inW0 + this->c0;
            const uint32_t outOff = ow * this->c0;

            AscendC::LocalTensor<DTYPE_X> a00 = xRow0[inW0];
            AscendC::LocalTensor<DTYPE_X> a01 = xRow0[inW1];
            AscendC::LocalTensor<DTYPE_X> a10 = xRow1[inW0];
            AscendC::LocalTensor<DTYPE_X> a11 = xRow1[inW1];

            AscendC::LocalTensor<DTYPE_Y> t0 = tmp0[outOff];
            AscendC::LocalTensor<DTYPE_Y> t1 = tmp1[outOff];
            AscendC::LocalTensor<DTYPE_Y> yv = yLocal[outOff];

            AscendC::Max(t0, a00, a01, this->c0);
            AscendC::Max(t1, a10, a11, this->c0);
            AscendC::Max(yv, t0, t1, this->c0);
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueRow0.FreeTensor(xRow0);
        inQueueRow1.FreeTensor(xRow1);
    }

    __aicore__ inline void CopyOut(uint32_t outBase, uint32_t tileIdx)
    {
        const uint32_t tileLenOw = TileLenOw(tileIdx);
        const uint32_t outScalarLen = tileLenOw * this->c0;
        const uint32_t outOffset = (tileIdx * this->tileOw) * this->c0;

        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[outBase + outOffset], yLocal, outScalarLen);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueRow0;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueRow1;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf1;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t n;
    uint32_t c1;
    uint32_t h;
    uint32_t w;
    uint32_t c0;
    uint32_t out_h;
    uint32_t out_w;
    uint32_t rowTotal;
    uint32_t out_total;

    uint32_t tileOw;
    uint32_t coreRowSize;
    uint32_t coreRowRemain;

    uint32_t coreRowStart;
    uint32_t coreRowCount;
};

extern "C" __global__ __aicore__ void max_pool2d(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaxPool2d op;
    op.Init(x, y,
            tiling_data.n, tiling_data.c1, tiling_data.h, tiling_data.w, tiling_data.c0,
            tiling_data.out_h, tiling_data.out_w,
            tiling_data.row_total, tiling_data.out_total,
            tiling_data.tile_ow,
            tiling_data.core_row_size, tiling_data.core_row_remain);
    op.Process();
}