from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import libcst as cst
from libcst import matchers as m

from rewriter.op_database import OpSolution


_OP_TO_REPLACEMENT_FUNC: Dict[str, str] = {
    "pointnet2._ext.gather_points": "ascend_gather_points",
    "_ext.gather_points": "ascend_gather_points",
    "pointnet2._ext.group_points": "ascend_group_points",
    "_ext.group_points": "ascend_group_points",
    "pointnet2._ext.gather_points_grad": "ascend_gather_points_grad",
    "_ext.gather_points_grad": "ascend_gather_points_grad",
    "pointnet2._ext.group_points_grad": "ascend_group_points_grad",
    "_ext.group_points_grad": "ascend_group_points_grad",
    "pointnet2._ext.furthest_point_sampling": "ascend_furthest_point_sampling",
    "_ext.furthest_point_sampling": "ascend_furthest_point_sampling",
    "pointnet2._ext.three_interpolate": "ascend_three_interpolate",
    "_ext.three_interpolate": "ascend_three_interpolate",
    "pointnet2._ext.three_interpolate_grad": "ascend_three_interpolate_grad",
    "_ext.three_interpolate_grad": "ascend_three_interpolate_grad",
    "pointnet2._ext.three_nn": "ascend_three_nn",
    "_ext.three_nn": "ascend_three_nn",
    "pointnet2._ext.ball_query": "ascend_ball_query",
    "_ext.ball_query": "ascend_ball_query",
}


_FUNC_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class OpCallReplacer(cst.CSTTransformer):
    def __init__(self) -> None:
        super().__init__()
        self.changes: int = 0
        self._added_import: Dict[str, List[str]] = {}

    def _needs_import(self, module: str, symbol: str) -> None:
        if module not in self._added_import:
            self._added_import[module] = []
        if symbol not in self._added_import[module]:
            self._added_import[module].append(symbol)

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        full = self._call_full_name(original_node)
        if full is None:
            return updated_node
        replacement = _OP_TO_REPLACEMENT_FUNC.get(full)
        if replacement is None:
            return updated_node
        self.changes += 1
        self._needs_import("ascend_pointnet2_ops", replacement)
        return updated_node.with_changes(
            func=cst.Name(replacement)
        )

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        if not self._added_import:
            return updated_node
        new_body = list(updated_node.body)
        for module, symbols in self._added_import.items():
            names = [cst.ImportAlias(name=cst.Name(s)) for s in symbols]
            import_stmt = cst.SimpleStatementLine(body=[cst.ImportFrom(
                module=cst.parse_expression(module),
                names=names,
            )])
            insert_idx = 0
            for i, stmt in enumerate(new_body):
                if isinstance(stmt, cst.SimpleStatementLine) and stmt.body:
                    inner = stmt.body[0]
                    if isinstance(inner, (cst.Import, cst.ImportFrom)):
                        insert_idx = i + 1
            new_body.insert(insert_idx, import_stmt)
        self.changes += 1
        return updated_node.with_changes(body=tuple(new_body))

    @staticmethod
    def _call_full_name(node: cst.Call) -> Optional[str]:
        func = node.func
        if isinstance(func, cst.Attribute):
            parts = []
            cur = func
            while isinstance(cur, cst.Attribute):
                parts.append(cur.attr.value)
                cur = cur.value
            if isinstance(cur, cst.Name):
                parts.append(cur.value)
            parts.reverse()
            return ".".join(parts)
        return None


def _remove_broken_imports(source: str) -> str:
    import re as _re
    lines = source.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("import ") and stripped.endswith(" as _ext"):
            mod = stripped[7:].split(" as ")[0].strip()
            if "pointnet2._ext" in mod or "pointnet2_3090" in mod:
                continue
        if stripped.startswith("from ") and "pointnet2._ext" in stripped and "import" in stripped:
            continue
        if stripped.startswith("from ") and "pointnet2_3090" in stripped and "import" in stripped:
            continue
        result.append(line)
    return "\n".join(result)


def patch_source_file(
    source: str,
    replacement_import: str = "ascend_pointnet2_ops",
) -> Tuple[str, int]:
    return _patch_source_file_regex(source, replacement_import)


def _patch_source_file_regex(
    source: str,
    replacement_import: str = "ascend_pointnet2_ops",
) -> Tuple[str, int]:
    changes = 0

    # ── gorilla 替换 ──
    if "import gorilla" in source:
        source = source.replace("import gorilla", "# import gorilla  # replaced by AscendDevTool")
        source = re.sub(
            r"gorilla\.Config\.fromfile",
            "Config.fromfile",
            source,
        )
        source = re.sub(
            r"model\s*=\s*model\s*\.\s*(?:cuda|npu)\s*\(\s*\)\s*\n\s*model\s*\.\s*eval\s*\(\s*\)\s*\n\s*gorilla\.solver\.load_checkpoint\s*\(\s*model\s*=\s*model\s*,\s*filename\s*=\s*(\S+)\s*\)",
            r"model.load_state_dict(torch.load(\1, map_location='cpu')['model'])\nmodel = model.npu()\nmodel.eval()",
            source,
        )
        source = re.sub(
            r"gorilla\.solver\.load_checkpoint\s*\(\s*model\s*=\s*model\s*,\s*filename\s*=\s*(\S+)\s*\)",
            r"model.load_state_dict(torch.load(\1, map_location='cpu')['model'])",
            source,
        )
        source = re.sub(
            r"gorilla\.utils\.set_cuda_visible_devices\s*\([^)]*\)",
            r"os.environ.setdefault('ASCEND_VISIBLE_DEVICES', '0')",
            source,
        )
                # 确保 model.load_state_dict 在 model.cuda/npu 之前
        src_lines = source.split("\n")
        cuda_i = -1
        for i, ln in enumerate(src_lines):
            s = ln.strip()
            if re.match(r"model\s*=\s*model\s*\.\s*(?:cuda|npu)\s*\(\s*\)", s):
                cuda_i = i
            if "model.load_state_dict(torch.load(" in s and cuda_i >= 0:
                cuda_line = src_lines.pop(cuda_i)
                src_lines.insert(i, cuda_line)
                break
        source = "\n".join(src_lines)

# 添加 from config.config import Config
        if "from config.config import Config" not in source:
            source = source.replace(
                "# import gorilla  # replaced by AscendDevTool",
                "from config.config import Config\n# import gorilla  # replaced by AscendDevTool",
            )
        # 注入 NPU 编译模式设置（解决 scaled_dot_product_attention 报错）
        if "torch.npu.set_compile_mode" not in source:
            source = re.sub(
                r"(import torch_npu\s*\n)",
                r"\1torch.npu.set_compile_mode(jit_compile=False)\ntorch.npu.config.allow_internal_format = True\n",
                source,
            )
        changes += 1

    for op_path, replacement_func in _OP_TO_REPLACEMENT_FUNC.items():
        short_name = op_path.split(".")[-1]
        pattern = rf"_ext\.{re.escape(short_name)}\b"
        if re.search(pattern, source):
            source, count = re.subn(pattern, f"_ext.{replacement_func}", source)
            changes += count

    if changes > 0:
        source = re.sub(
            r"(\s*)import\s+pointnet2\._ext\s+as\s+_ext",
            rf"\1import {replacement_import} as _ext",
            source,
        )
        source = re.sub(
            r"(\s*)import\s+pointnet2_3090\.pointnet2_utils\s+as\s+_ext",
            rf"\1import {replacement_import} as _ext",
            source,
        )

    return source, changes


def _generate_replacement_module(output_dir: Path, solutions: List[OpSolution]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    # 放到 Pose_Estimation_Model 目录下，确保 import 能找到
    dest_dirs = list(output_dir.rglob("Pose_Estimation_Model"))
    dest = dest_dirs[0] if dest_dirs else output_dir
    path = dest / "ascend_pointnet2_ops.py"

    lines = [
        "from __future__ import annotations",
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
    ]

    seen_functions = set()

    for s in solutions:
        safe_name = s.op_name.split(".")[-1] if "." in s.op_name else s.op_name
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", safe_name)

        if s.op_name.startswith("torch.nn"):
            func_name = "ascend_data_parallel"
        else:
            func_name = f"ascend_{safe_name}"

        if s.strategy == "需开发":
            is_duplicate_name = False
            for other in solutions:
                if other is s:
                    continue
                if other.strategy in ("可拆分",) and other.op_name.split(".")[-1] == s.op_name.split(".")[-1]:
                    is_duplicate_name = True
                    break
            if is_duplicate_name:
                continue

        if func_name in seen_functions:
            continue
        seen_functions.add(func_name)

        if s.strategy == "可拆分" and s.replacement_code:
            func_code = s.replacement_code.strip()
            if s.op_name.startswith("torch.nn"):
                lines.append(f"# ── {s.op_name} ──")
                lines.append(func_code)
            else:
                func_lines = func_code.split("\n")
                for line in func_lines:
                    if line.strip().startswith("def ascend_"):
                        lines.append(f"# ── decomposed: {s.op_name} (PyTorch fallback, 建议后续用 Ascend C 替换) ──")
                        lines.append(line)
                        # 注入运行时警告
                        indent = line[:len(line) - len(line.lstrip())]
                        func_name = line.strip().split("(")[0].replace("def ", "")
                        lines.append(f'{indent}    import warnings')
                        lines.append(f'{indent}    warnings.warn("AscendDevTool: 算子 {s.op_name} 当前使用 PyTorch fallback，建议后续用 Ascend C 重写以提升性能")')
                    else:
                        lines.append(line)
                lines.append("")
        elif s.strategy in ("可映射",):
            lines.append(f"# ── mapped: {s.op_name} (Ascend C via op_builder) ──")
            lines.append(f"def {func_name}(*args, **kwargs):")
            lines.append(f'    raise NotImplementedError("Ascend C 算子 {s.op_name} 待编译部署")')
            lines.append("")
        elif s.strategy == "需开发":
            lines.append(f"# ── needs-dev: {s.op_name} ──")
            lines.append(f"def {func_name}(*args, **kwargs):")
            lines.append(f'    raise NotImplementedError("算子 {s.op_name} 需要开发")')
            lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _generate_gorilla_stub(output_dir: Path) -> Path:
    import shutil
    src = Path(__file__).parent / "gorilla_stub.py"
    paths = list(output_dir.rglob("Pose_Estimation_Model"))
    dest = paths[0] if paths else output_dir
    dest_path = dest / "gorilla.py"
    shutil.copy(src, dest_path)
    return dest_path


def _generate_config_module(output_dir: Path) -> Path:
    import shutil
    src = Path(__file__).parent / "config_stub.py"
    paths = list(output_dir.rglob("Pose_Estimation_Model"))
    dest = paths[0] if paths else output_dir
    config_dir = dest / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    dest_path = config_dir / "config.py"
    shutil.copy(src, dest_path)  # 每次都覆盖，确保最新
    return dest_path


def apply_rewrites_to_source(
    source_dir: Path,
    output_dir: Path,
    solutions: List[OpSolution],
) -> Tuple[List[str], int]:
    import shutil

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir, ignore=shutil.ignore_patterns("__pycache__", ".git", "*.pyc"))

    replacement_path = _generate_replacement_module(output_dir, solutions)
    gorilla_path = _generate_gorilla_stub(output_dir)
    config_path = _generate_config_module(output_dir)
    generated_files = [str(replacement_path), str(gorilla_path), str(config_path)]

    py_files = list(output_dir.rglob("*.py"))
    total_changes = 0

    for py_file in py_files:
        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        new_source, changes = patch_source_file(source)
        if changes == 0:
            continue

        py_file.write_text(new_source, encoding="utf-8")
        generated_files.append(str(py_file))
        total_changes += changes

    return generated_files, total_changes
