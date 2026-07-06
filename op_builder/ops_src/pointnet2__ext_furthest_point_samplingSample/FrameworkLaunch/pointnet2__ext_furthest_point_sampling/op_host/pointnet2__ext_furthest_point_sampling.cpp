#include "pointnet2__ext_furthest_point_sampling_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    pointnet2__ext_furthest_point_samplingTilingData tiling;

    const gert::StorageShape* xyz_shape = context->GetInputShape(0);
    uint32_t B = xyz_shape->GetStorageShape().GetDim(0);
    uint32_t N = xyz_shape->GetStorageShape().GetDim(1);
    uint32_t M = 256;
    if (M > N) M = N;

    tiling.set_B(B);
    tiling.set_N(N);
    tiling.set_M(M);
    tiling.set_totalLength(B * M);

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
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xyz_shape = context->GetInputShape(0);
    uint32_t B = xyz_shape->GetDim(0);
    uint32_t M = 256;
    gert::Shape* y_shape = context->GetOutputShape(0);
    y_shape->SetDimNum(2);
    y_shape->SetDim(0, B);
    y_shape->SetDim(1, M);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class pointnet2__ext_furthest_point_sampling : public OpDef {
public:
    explicit pointnet2__ext_furthest_point_sampling(const char* name) : OpDef(name) {
        this->Input("x0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y0")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
    }
};

OP_ADD(pointnet2__ext_furthest_point_sampling);
}
