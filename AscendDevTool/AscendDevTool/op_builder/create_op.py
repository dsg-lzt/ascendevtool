#!/usr/bin/env python3
import os
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Emboded Ops Builder: 自动算子骨架生成工具")
    parser.add_argument("--op_name", type=str, required=True, help="自定义算子的名称(建议小写加下划线,如 custom_relu)")
    parser.add_argument("--inputs", type=str, default="Tensor x", help="算子输入参数，用逗号分隔 (默认: Tensor x)")
    parser.add_argument("--outputs", type=str, default="Tensor", help="算子输出参数，用逗号分隔 (默认: Tensor)")
    
    args = parser.parse_args()
    
    # 强制清理名字，去掉可能引起问题的符号
    op_name = args.op_name.strip().lower()
    
    # 获取此脚本所在目录 (op_builder 目录)    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ops_dir = os.path.join(base_dir, "src", "ops", op_name)
    
    if os.path.exists(ops_dir):
        print(f"Error: 算子目录 '{ops_dir}' 已存在。不再重复生成。")
        sys.exit(1)
        
    os.makedirs(ops_dir)
    print(f"[*] 创建算子专属目录: {ops_dir}")

    # ===== 1. adapter.cpp 生成：严格闭包隔离 =====
    adapter_content = f"""// {op_name} adapter 绑定入口
#include <torch/extension.h>

namespace embodied_ops {{
namespace {op_name}_impl {{

// 拦截并强制将 Tensor 转换为 Contiguous 的内存布局以应对 NPU 要求
at::Tensor {op_name}_forward({args.inputs}) {{
    // TODO: 实现 Host 端 Tiling 及 ACLNN 分发代码
    // return y;
}}

}} // namespace {op_name}_impl
}} // namespace embodied_ops

// ====================== 强制分散注册 ======================
// 利用 Fragment 将私有实现注入到全局 embodied_ops 中
TORCH_LIBRARY_IMPL(embodied_ops, PrivateUse1, m) {{
    m.impl("{op_name}", TORCH_FN(embodied_ops::{op_name}_impl::{op_name}_forward));
}}
"""
    
    # ===== 2. kernel_impl.cpp 桩代码 =====
    kernel_content = f"""// {op_name} 的 Ascend C Device 端核函数
#include "kernel_operator.h"
using namespace AscendC;

class Kernel_{op_name} {{
public:
    __aicore__ inline Kernel_{op_name}() {{}}
    __aicore__ inline void Init(/* GM_ADDR x ... */) {{
        // TODO: 配置 Tiling 与 DataCopy
    }}
    __aicore__ inline void Process() {{
        // TODO: CopyIn -> Compute -> CopyOut 异步三阶段
    }}
}};

extern "C" __global__ __aicore__ void custom_{op_name}(GM_ADDR x, GM_ADDR workspace, GM_ADDR tiling) {{
    // 获取由 Host 分发的专属 {op_name} TilingData
    // {op_name}TilingData *tiling_data = reinterpret_cast<{op_name}TilingData*>(tiling);

    Kernel_{op_name} op;
    op.Init();
    op.Process();
}}
"""

    adapter_path = os.path.join(ops_dir, "adapter.cpp")
    kernel_path = os.path.join(ops_dir, "kernel_impl.cpp")
    
    with open(adapter_path, "w", encoding="utf-8") as f:
        f.write(adapter_content)
        
    with open(kernel_path, "w", encoding="utf-8") as f:
        f.write(kernel_content)

    print(f"[*] 生成 C++ Adapter: {adapter_path}")
    print(f"[*] 生成 Kernel 实现: {kernel_path}")
    
    # 提醒用户去登记 Schema
    print("\n--------------------------------------------------------------")
    print(f"✅ 生成成功！为了确保 PyTorch 能认出 {op_name}，请手动执行以下操作：")
    print(f"在 op_builder/src/op_registry.cpp 中加入 Schema 声明：")
    print(f"    m.def(\"{op_name}({args.inputs}) -> {args.outputs}\");")
    print("--------------------------------------------------------------")

if __name__ == "__main__":
    main()
