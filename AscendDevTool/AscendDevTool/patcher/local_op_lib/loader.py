import os
import torch
import torch_npu

_HAS_LOADED = False

def init_custom_ops():
    '''
    懒加载统一的自研算子动态库。
    只会在第一个自定义算子被拉起替换时加载一次。
    '''
    global _HAS_LOADED
    if not _HAS_LOADED:
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # 由于 loader 在 patcher/local_op_lib/loader.py，我们要回转到 build 目录
        so_path = os.path.join(curr_dir, "../../op_builder/build/libembodied_custom_ops.so")
        so_path = os.path.normpath(so_path)
        
        if not os.path.exists(so_path):
            raise FileNotFoundError(f"【严重错误】未找到自定义算子库: {so_path}。\n"
                                    "可能是您没有在 op_builder 目录下执行 cmake 构建。")
            
        torch.classes.load_library(so_path)
        _HAS_LOADED = True

def get_op(op_name: str):
    '''
    由 monkey patcher 调用的算子代理工厂函数。
    向外暴露具体的实现方法。
    '''
    init_custom_ops()
    # Pytorch custom operators expose attributes on `torch.ops.namespace`
    try:
        op_func = getattr(torch.ops.embodied_ops, op_name)
    except AttributeError:
        raise AttributeError(f"在 embodied_ops 命名空间内找不到算子 '{op_name}'。可能没有正确完成 C++ 端注册。")
    
    return op_func
