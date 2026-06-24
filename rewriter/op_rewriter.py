from __future__ import annotations

from typing import Optional

from rewriter.op_database import get_math_equivalent


def attempt_math_rewrite(op_name: str) -> Optional[str]:
    return get_math_equivalent(op_name)
