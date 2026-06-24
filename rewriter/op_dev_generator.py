from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

OP_BUILDER_DIR = Path(__file__).resolve().parents[1] / "op_builder"


def generate_operator_scaffolding(
    op_name: str,
    input_dtypes: Optional[List[str]] = None,
    output_dtypes: Optional[List[str]] = None,
    num_inputs: int = 2,
    num_outputs: int = 1,
) -> Optional[Path]:
    if input_dtypes is None:
        input_dtypes = ["float16", "float32"]
    if output_dtypes is None:
        output_dtypes = ["float16", "float32"]

    json_path = OP_BUILDER_DIR / f"_{op_name}_gen.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    op_def = [{
        "op": op_name,
        "language": "cpp",
        "input_desc": [
            {
                "name": f"x{i}",
                "param_type": "required",
                "format": ["ND"],
                "type": input_dtypes,
            }
            for i in range(num_inputs)
        ],
        "output_desc": [
            {
                "name": f"y{i}",
                "param_type": "required",
                "format": ["ND"],
                "type": output_dtypes,
            }
            for i in range(num_outputs)
        ],
    }]

    with open(json_path, "w") as f:
        json.dump(op_def, f, indent=2)

    return json_path


def generate_python_stub(op_name: str) -> str:
    safe_name = op_name.replace(".", "_").replace("-", "_")
    return f'''from __future__ import annotations

import torch


def ascend_{safe_name}(*args, **kwargs):
    raise NotImplementedError(
        "算子 {op_name} 的 Ascend C 实现尚未完成。"
        "请先完善 op_builder/ops_src/{op_name}Sample/FrameworkLaunch/"
        " 下的 kernel 实现，然后编译。"
    )


def patch_{safe_name}():
    import {op_name.split(".")[0] if "." in op_name else "torch"} as _m
    _m.{op_name.split(".")[-1]} = ascend_{safe_name}
'''
