# 这里存放我们通过 Ascend C 开发或者用 Python 临时拼凑的平替算子实现
from typing import Any, Dict

def ascend_jit_annotate(type_hint, value):
    """
    替换 torch.jit.annotate
    由于端侧部署可能直接跳过 JIT 编译，此处可作 Dummy 处理，直接返回 value
    """
    # print(f"[Monkey Patch] 命中 ascend_jit_annotate")
    return value

def ascend_jit_is_scripting():
    """
    替换 torch.jit.is_scripting
    返回 False 让模型走常规 Python 控制流
    """
    # print(f"[Monkey Patch] 命中 ascend_jit_is_scripting")
    return False
