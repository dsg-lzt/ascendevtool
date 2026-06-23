#include "three_nn_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        ThreeNNTilingData tiling;
        constexpr int32_t P0 = 0;
        constexpr int32_t P1 = 1;
        constexpr int32_t D0 = 0;
        constexpr int32_t D1 = 1;
        constexpr int32_t D2 = 2;
        constexpr int32_t DIM_NUM = 3;
        constexpr int32_t COORD_DIM = 3;
        constexpr uint32_t BLOCK_SIZE = 32;
        constexpr uint32_t SAFE_DIV = 8;

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        auto usedCoreNum = ascendcPlatform.GetCoreNumAiv();
        tiling.set_UBSize(ub_size);

        auto dataType = context->GetInputDesc(P0)->GetDataType();
        if (dataType == ge::DT_FLOAT16) {
            tiling.set_tilingKey(static_cast<int64_t>(ThreeNNTilingKey::FP16_NORM));
        } else {
            tiling.set_tilingKey(static_cast<int64_t>(ThreeNNTilingKey::FP32_NORM));
        }

        auto xyz1Shape = context->GetInputShape(P0)->GetStorageShape();
        auto xyz2Shape = context->GetInputShape(P1)->GetStorageShape();

        uint32_t B = 1, N = 0, M = 0;
        uint32_t formatB3N = 0;

        if (xyz1Shape.GetDimNum() == DIM_NUM) {
            if (xyz1Shape.GetDim(D1) == COORD_DIM && xyz1Shape.GetDim(D2) != COORD_DIM) {
                formatB3N = 1;
                B = xyz1Shape.GetDim(D0);
                N = xyz1Shape.GetDim(D2);
                M = xyz2Shape.GetDim(D2);
            } else {
                formatB3N = 0;
                B = xyz1Shape.GetDim(D0);
                N = xyz1Shape.GetDim(D1);
                M = xyz2Shape.GetDim(D1);
            }
        } else {
            B = 1;
            N = xyz1Shape.GetDim(D0);
            M = xyz2Shape.GetDim(D0);
        }

        uint32_t sizeofdatatype = (dataType == ge::DT_FLOAT16) ? 2 : 4;
        uint32_t alignNum = BLOCK_SIZE / sizeofdatatype;

        uint64_t usableUB = ub_size / SAFE_DIV;
        uint32_t N_block = static_cast<uint32_t>(usableUB / (COORD_DIM * 2 * sizeofdatatype));
        N_block = N_block / alignNum * alignNum;
        if (N_block < alignNum) { N_block = alignNum; }
        if (N_block > N) { N_block = N; }

        uint32_t totalQuery = B * M;

        usedCoreNum = (usedCoreNum < totalQuery) ? usedCoreNum : totalQuery;
        usedCoreNum = (usedCoreNum >= 1) ? usedCoreNum : 1;

        uint32_t coreQuerySize = totalQuery / usedCoreNum;
        uint32_t coreRemain = totalQuery % usedCoreNum;

        tiling.set_B(B);
        tiling.set_N(N);
        tiling.set_M(M);
        tiling.set_N_block(N_block);
        tiling.set_alignNum(alignNum);
        tiling.set_usedCoreNum(usedCoreNum);
        tiling.set_totalQuery(totalQuery);
        tiling.set_coreQuerySize(coreQuerySize);
        tiling.set_coreRemain(coreRemain);
        tiling.set_formatB3N(formatB3N);

        context->SetBlockDim(usedCoreNum);
        context->SetTilingKey(tiling.get_tilingKey());

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const auto *xyz2Shape = context->GetInputShape(1);
        auto *distShape = context->GetOutputShape(0);
        auto *idxShape = context->GetOutputShape(1);
        *distShape = *xyz2Shape;
        *idxShape = *xyz2Shape;
        return GRAPH_SUCCESS;
    }
}

namespace ops
{
    class ThreeNN : public OpDef
    {
    public:
        explicit ThreeNN(const char *name) : OpDef(name)
        {
            this->Input("xyz1")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("xyz2")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("dist")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32, ge::DT_INT32})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310b")
                          .AddConfig("ascend910b")
                          .AddConfig("ascend310p");
        }
    };

    OP_ADD(ThreeNN);
}
