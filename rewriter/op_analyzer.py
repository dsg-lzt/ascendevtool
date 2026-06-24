from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

from rewriter.op_database import OpSolution, decompose_op, get_math_equivalent


_MAP_STRATEGIES: Dict[str, str] = {
    "pointnet2._ext.three_nn": "patcher.local_op_lib.ascend_pointnet2_ops.ascend_three_nn(mapped to op_builder ThreeNNSample)",
    "pointnet2._ext.ball_query": "patcher.local_op_lib.ascend_pointnet2_ops.ascend_ball_query(mapped to op_builder BallQuerySample)",
}


def analyze_unsupported_ops(csv_path: Path, known_local_ops: Dict[str, tuple]) -> List[OpSolution]:
    solutions: List[OpSolution] = []

    if not csv_path.is_file():
        return solutions

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        seen = set()
        for row in reader:
            op = row.get("OP", "").strip()
            if not op or op in seen:
                continue
            seen.add(op)

            solution = _analyze_single_op(op, known_local_ops)
            solutions.append(solution)

    return solutions


def _analyze_single_op(op: str, known_local_ops: Dict[str, tuple]) -> OpSolution:
    if op in known_local_ops:
        module, func = known_local_ops[op]
        return OpSolution(op_name=op, strategy="已入库", solution=f"已有本地实现: {module}.{func}", local_module=module, local_func=func)

    if op in _MAP_STRATEGIES:
        return OpSolution(op_name=op, strategy="可映射", solution=_MAP_STRATEGIES[op])

    decomposed = decompose_op(op)
    if decomposed:
        return OpSolution(op_name=op, strategy="可拆分", solution=f"可拆分为原生 PyTorch 算子组合", replacement_code=decomposed)

    math_eq = get_math_equivalent(op)
    if math_eq:
        return OpSolution(op_name=op, strategy="数学等价改写", solution=f"可数学等价替换: {math_eq}", replacement_code=math_eq)

    return OpSolution(op_name=op, strategy="需开发", solution=f"需开发 Ascend C 算子 ({op})")
