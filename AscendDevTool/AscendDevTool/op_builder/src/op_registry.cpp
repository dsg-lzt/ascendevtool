// 全局算子定义文件 - op_builder/src/op_registry.cpp
// 本文件集中注册所有 custom op 的 schema，避免重复定义

#include <torch/extension.h>

// 所有算子必须使用 embodied_ops 这个命名空间
TORCH_LIBRARY(embodied_ops, m)
{
    // Schema 在此处集中管理
    // 后续可由 create_op.py 自动追加修改此文件
    // example: m.def("custom_relu(Tensor x) -> Tensor");
}
