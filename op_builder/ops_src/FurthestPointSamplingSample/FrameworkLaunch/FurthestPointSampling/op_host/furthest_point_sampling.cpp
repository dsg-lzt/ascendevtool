
#include "furthest_point_sampling_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"

constexpr int32_t COORD_DIM = 3;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const gert::StorageShape* ss = context->GetInputShape(0);
    const gert::Shape& shape = ss->GetOriginShape();
    int32_t B = shape.GetDim(0);
    int32_t N = shape.GetDim(1);

    int32_t M = 0;
    auto attrs = context->GetAttrs();
    const int64_t* mp = attrs->GetInt(0);
    if (mp) M = *mp;
    if (M > N) M = N;
    if (M == 0) M = 1;

    uint32_t dtLen = 0;
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), dtLen);

    auto plat = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    plat.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    uint32_t coreNum = plat.GetCoreNum();
    if (coreNum == 0) coreNum = 1;

    uint64_t ubElm = ubSize / dtLen;
    uint32_t tileN = (ubElm - N) / 9;
    if (tileN > (uint32_t)N) tileN = N;
    if (tileN < 64) tileN = 64;
    uint32_t nTiles = (N + tileN - 1) / tileN;
    uint32_t bpc = (B + coreNum - 1) / coreNum;
    uint32_t crem = B % coreNum;

    FurthestPointSamplingTilingData tiling;
    tiling.set_B(B); tiling.set_N(N); tiling.set_M(M);
    tiling.set_dataTypeLength(dtLen);
    tiling.set_tileN(tileN);
    tiling.set_tileLoopNum(nTiles);
    tiling.set_batchPerCore(bpc);
    tiling.set_coreRem(crem);
    tiling.set_initVal(dtLen == 4 ? 3.402823e+38f : 65504.0f);

    context->SetTilingKey(dtLen == 4 ? 0 : 1);
    context->SetBlockDim(coreNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t* ws = context->GetWorkspaceSizes(1);
    ws[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* s = context->GetInputShape(0);
    gert::Shape* o = context->GetOutputShape(0);
    int32_t B = s->GetDim(0);
    int32_t M = 128;
    auto attrs = context->GetAttrs();
    const int64_t* mp = attrs->GetInt(0);
    if (mp) M = *mp;
    o->SetDim(0, B); o->SetDim(1, M);
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class FurthestPointSampling : public OpDef {
public:
    explicit FurthestPointSampling(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("npoint").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");

    }
};

OP_ADD(FurthestPointSampling);
}
