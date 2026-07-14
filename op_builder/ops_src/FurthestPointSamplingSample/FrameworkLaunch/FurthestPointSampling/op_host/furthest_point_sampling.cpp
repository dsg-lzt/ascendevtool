/**
 * FurthestPointSampling - Host-side Op Definition & Tiling
 *
 * Input:  points  [B, N, 3]   float32 | float16
 * Output: indices [B, M]      int32
 * Attr:   npoint  (int)
 */
#include "furthest_point_sampling_tiling.h"
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"

constexpr int32_t COORD_DIM = 3;

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    const gert::StorageShape *ss = context->GetInputShape(0);
    const gert::Shape &shape = ss->GetOriginShape();
    int32_t B = static_cast<int32_t>(shape.GetDim(0));
    int32_t N = static_cast<int32_t>(shape.GetDim(1));

    int32_t M = 0;
    auto attrs = context->GetAttrs();
    const int64_t *mPtr = attrs->GetInt(0);
    if (mPtr) M = static_cast<int32_t>(*mPtr);

    uint32_t dataTypeLength = 0;
    ge::TypeUtils::GetDataTypeLength(
        context->GetInputDesc(0)->GetDataType(), dataTypeLength);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = 1;

    uint64_t ubElm = ubSize / dataTypeLength;
    /* pad N to 128 for alignment */
    int32_t padN = ((N + 127) / 128) * 128;
    if (padN < 128) padN = 128;

    uint32_t tileN = static_cast<uint32_t>((ubElm - padN) / 9);
    if (tileN > static_cast<uint32_t>(padN)) tileN = static_cast<uint32_t>(padN);
    if (tileN < 64) tileN = 64;
    uint32_t numTiles = (padN + tileN - 1) / tileN;

    uint32_t batchesPerCore = (B + coreNum - 1) / coreNum;
    uint32_t coreRemainder = B % coreNum;

    constexpr float kFp32Init = 3.402823e+38f;
    constexpr float kFp16Init = 65504.0f;

    int32_t tilingKey = (dataTypeLength == 4) ? 0 : 1;

    FpsTilingData tiling;
    tiling.set_B(B);
    tiling.set_N(N);
    tiling.set_padN(padN);
    tiling.set_M(M);
    tiling.set_C(COORD_DIM);
    tiling.set_dataTypeLength(dataTypeLength);
    tiling.set_ubPointsNum(tileN);
    tiling.set_ubMinDistNum(tileN);
    tiling.set_tileLoopNum(numTiles);
    tiling.set_batchPerCore(batchesPerCore);
    tiling.set_coreRemainder(coreRemainder);
    tiling.set_wsOffset(padN);
    tiling.set_wsStride(padN);
    tiling.set_initVal((dataTypeLength == 4) ? kFp32Init : kFp16Init);

    context->SetTilingKey(static_cast<uint64_t>(tilingKey));
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = 0;

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *inputShape = context->GetInputShape(0);
    gert::Shape *outputShape = context->GetOutputShape(0);
    int32_t B = static_cast<int32_t>(inputShape->GetDim(0));

    int32_t M = 0;
    auto attrs = context->GetAttrs();
    const int64_t *mPtr = attrs->GetInt(0);
    if (mPtr) M = static_cast<int32_t>(*mPtr);

    outputShape->SetDim(0, B);
    outputShape->SetDim(1, M);
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class FurthestPointSampling : public OpDef {
public:
    explicit FurthestPointSampling(const char *name) : OpDef(name)
    {
        this->Input("points")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("npoint").AttrType(REQUIRED).Int();

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend310p");
    }
};

OP_ADD(FurthestPointSampling);

}  // namespace ops
