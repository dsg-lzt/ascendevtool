from __future__ import annotations

from typing import Optional

from rewriter.op_database import decompose_op


def attempt_decompose(op_name: str) -> Optional[str]:
    return decompose_op(op_name)
