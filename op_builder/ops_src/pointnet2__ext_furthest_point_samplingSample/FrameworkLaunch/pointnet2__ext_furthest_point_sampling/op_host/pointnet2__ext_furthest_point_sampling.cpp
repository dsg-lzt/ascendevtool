#include "pointnet2__ext_furthest_point_sampling_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    FpsCustomTilingData tiling;
    constexpr int ATTR_NPOINT = 0;
    auto attrs = context->GetAttrs();
    int32_t M_raw = *(attrs->GetAttrPointer<int32_t>(ATTR_NPOINT));
    uint32_t M = static_cast<uint32_t>(M_raw);

    const gert::StorageShape* xyz_shape = context->GetInputShape(0);
    uint32_t B = xyz_shape->GetStorageShape().GetDim(0);
    uint32_t N = xyz_shape->GetStorageShape().GetDim(1);
    if (M > N) M = N;
    if (M == 0) M = 1;

    tiling.set_B(B); tiling.set_N(N); tiling.set_M(M); tiling.set_totalLength(B * M);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNumAiv = platform.GetCoreNumAiv();
    uint32_t usedCoreNum = (B < coreNumAiv) ? B : coreNumAiv;
    if (usedCoreNum == 0) usedCoreNum = 1;
    uint32_t core_size = B / usedCoreNum;
    uint32_t core_remain = B % usedCoreNum;
    uint32_t block_size = 128;
    if (block_size > N) block_size = N;

    tiling.set_tileNum((N + block_size - 1) / block_size);
    tiling.set_block_size(block_size);
    tiling.set_core_size(core_size);
    tiling.set_core_remain(core_remain);
    tiling.set_usedCoreNum(usedCoreNum);

    context->SetBlockDim(usedCoreNum);
    // 根据数据类型设置 tiling key (fp16=10000001, fp32=10000002)
    auto input_desc = context->GetInputDesc(0);
    if (input_desc.GetDataType() == ge::DT_FLOAT16) {
        context->SetTilingKey(10000001);
    } else {
        context->SetTilingKey(10000002);
    }
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xyz_shape = context->GetInputShape(0);
    uint32_t B = xyz_shape->GetDim(0);
    auto attrs = context->GetAttrs();
    int32_t M_raw = *(attrs->GetAttrPointer<int32_t>(0));
    uint32_t M = static_cast<uint32_t>(M_raw);
    if (M == 0) M = 1;
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, B);
    y_shape->SetDim(1, M);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class FurthestPointSampling : public OpDef {
public:
    explicit FurthestPointSampling(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y").ParamType(REQUIRED).DataType({ge::DT_INT32}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("npoint").Int();
        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};
OP_ADD(FurthestPointSampling);
}
