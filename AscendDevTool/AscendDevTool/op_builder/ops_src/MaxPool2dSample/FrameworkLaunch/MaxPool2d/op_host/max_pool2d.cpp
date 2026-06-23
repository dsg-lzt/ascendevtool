
#include "max_pool2d_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling
{
    static ge::graphStatus TilingFunc(gert::TilingContext *context)
    {
        MaxPool2dTilingData tiling;

        constexpr uint32_t BLOCK_BYTES = 32;

        const auto x_desc = context->GetInputDesc(0);
        const auto x_shape_storage = context->GetInputShape(0)->GetStorageShape();
        if (x_shape_storage.GetDimNum() != 4)
        {
            return ge::GRAPH_FAILED;
        }

        const uint32_t n = static_cast<uint32_t>(x_shape_storage.GetDim(0));
        const uint32_t c = static_cast<uint32_t>(x_shape_storage.GetDim(1));
        const uint32_t h = static_cast<uint32_t>(x_shape_storage.GetDim(2));
        const uint32_t w = static_cast<uint32_t>(x_shape_storage.GetDim(3));
        if (h < 2 || w < 2)
        {
            return ge::GRAPH_FAILED;
        }

        uint32_t sizeofdatatype = 4;
        const auto dt = x_desc->GetDataType();
        if (dt == ge::DT_INT8)
        {
            sizeofdatatype = 1;
        }
        else if (dt == ge::DT_FLOAT16 || dt == ge::DT_BF16)
        {
            sizeofdatatype = 2;
        }
        else
        {
            sizeofdatatype = 4;
        }

        const uint32_t c0 = BLOCK_BYTES / sizeofdatatype;
        const uint32_t c1 = c / c0;

        const uint32_t out_h = h / 2;
        const uint32_t out_w = w / 2;
        const uint32_t row_total = n * c1 * out_h;
        const uint32_t out_total = row_total * out_w;

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint64_t ub_size = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        uint32_t aivNum = static_cast<uint32_t>(ascendcPlatform.GetCoreNum());
        if (row_total == 0)
        {
            aivNum = 1;
        }
        else
        {
            aivNum = (aivNum < row_total) ? aivNum : row_total;
            aivNum = aivNum >= 1 ? aivNum : 1;
        }

        const uint64_t usableUb = ub_size / 2;
        const uint64_t bytesPerOw = static_cast<uint64_t>(5) * static_cast<uint64_t>(sizeofdatatype);
        uint32_t tile_ow = 1;
        if (bytesPerOw > 0)
        {
            tile_ow = static_cast<uint32_t>(usableUb / bytesPerOw);
            tile_ow = tile_ow >= 1 ? tile_ow : 1;
        }
        if (tile_ow > out_w)
        {
            tile_ow = out_w;
        }
        if (c0 > 1 && tile_ow >= c0)
        {
            tile_ow = (tile_ow / c0) * c0;
            tile_ow = tile_ow >= 1 ? tile_ow : 1;
        }

        const uint32_t core_row_size = row_total / aivNum;
        const uint32_t core_row_remain = row_total - aivNum * core_row_size;

        tiling.set_n(n);
        tiling.set_c1(c1);
        tiling.set_h(h);
        tiling.set_w(w);
        tiling.set_c0(c0);
        tiling.set_out_h(out_h);
        tiling.set_out_w(out_w);
        tiling.set_row_total(row_total);
        tiling.set_out_total(out_total);
        tiling.set_tile_ow(tile_ow);
        tiling.set_core_row_size(core_row_size);
        tiling.set_core_row_remain(core_row_remain);

        context->SetBlockDim(aivNum);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge
{
    static ge::graphStatus InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape *x1_shape = context->GetInputShape(0);
        gert::Shape *y_shape = context->GetOutputShape(0);
        *y_shape = *x1_shape;
        // 约定输入为 NCHW，做固定 2x2、stride=2 的 MaxPool。
        // 输出维度：N, C, H/2, W/2
        // 说明：这里假设框架侧的 Shape 对象支持 SetDim/SetDimNum；如果编译报接口不存在，
        // 请把 shape 推导放到 op 配置层或改用对应的 shape_utils 工具函数。
        if (x1_shape->GetDimNum() == 4)
        {
            const int64_t n = x1_shape->GetDim(0);
            const int64_t c = x1_shape->GetDim(1);
            const int64_t h = x1_shape->GetDim(2);
            const int64_t w = x1_shape->GetDim(3);
            const int64_t out_h = h / 2;
            const int64_t out_w = w / 2;
            y_shape->SetDimNum(4);
            y_shape->SetDim(0, n);
            y_shape->SetDim(1, c);
            y_shape->SetDim(2, out_h);
            y_shape->SetDim(3, out_w);
        }
        return GRAPH_SUCCESS;
    }
    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto inputDataType = context->GetInputDataType(0);
        context->SetOutputDataType(0, inputDataType);
        return ge::GRAPH_SUCCESS;
    }
}

namespace ops
{
    class MaxPool2d : public OpDef
    {
    public:
        explicit MaxPool2d(const char *name) : OpDef(name)
        {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT8, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

            this->AICore()
                .SetTiling(optiling::TilingFunc);
            this->AICore().AddConfig("ascend310p").AddConfig("ascend310b").AddConfig("ascend910b");
        }
    };

    OP_ADD(MaxPool2d);
}
