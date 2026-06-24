from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from rewriter.op_analyzer import analyze_unsupported_ops
from rewriter.op_database import OpSolution
from rewriter.op_dev_agent import batch_generate_dev_tasks
from rewriter.op_dev_generator import generate_operator_scaffolding
from rewriter.code_patcher import apply_rewrites_to_source


@dataclass(frozen=True)
class RewriteResult:
    status: str
    total_ops: int
    mapped_count: int
    decomposable_count: int
    math_rewrite_count: int
    need_develop_count: int
    solutions: List[OpSolution]
    generated_ops: List[str]
    patched_files: List[str]  # new
    total_patches: int  # new


def _load_local_ops_csv(path: Path) -> Dict[str, tuple]:
    mapping: Dict[str, tuple] = {}
    if not path.is_file():
        return mapping
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op = row.get("OP", "").strip()
            module = row.get("Local_Module", "").strip()
            func = row.get("Local_Func", "").strip()
            if op and module and func:
                mapping[op] = (module, func)
    return mapping


def _write_decompose_stubs(solutions: List[OpSolution], output_dir: Path) -> List[str]:
    generated: List[str] = []
    decomposable = [s for s in solutions if s.strategy == "可拆分" and s.replacement_code]
    if decomposable:
        output_dir.mkdir(parents=True, exist_ok=True)
        stub_path = output_dir / "decomposed_ops.py"
        lines = [
            "from __future__ import annotations",
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
        ]
        for s in decomposable:
            lines.append(f"# ── {s.solution} ──")
            lines.append(s.replacement_code.strip())
            lines.append("")
        stub_path.write_text("\n".join(lines), encoding="utf-8")
        generated.append(str(stub_path))
    return generated


def _write_direct_map_csv(solutions: List[OpSolution], local_ops_csv: Path) -> List[str]:
    generated: List[str] = []
    mapped = [s for s in solutions if s.strategy == "可映射"]
    if not mapped:
        return generated

    existing = set()
    if local_ops_csv.is_file():
        with open(local_ops_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row.get("OP", "").strip())

    new_entries = []
    for s in mapped:
        if s.solution and "(" in s.solution:
            name = s.solution[:s.solution.index("(")].strip()
        else:
            name = s.solution[:50].strip() if s.solution else ""
        if name and name not in existing:
            new_entries.append([name, "", ""])

    if new_entries:
        local_ops_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(local_ops_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not local_ops_csv.is_file() or local_ops_csv.stat().st_size == 0:
                writer.writerow(["OP", "Local_Module", "Local_Func"])
            for entry in new_entries:
                writer.writerow(entry)
        generated.append(str(local_ops_csv))
    return generated


@dataclass(frozen=True)
class DevTaskResult:
    status: str
    total_ops: int
    generated_tasks: List[str]


def run_operator_development(
    unsupported_csv: Path,
    local_ops_csv: Path,
    output_dir: Path,
    op_name: Optional[str] = None,
) -> DevTaskResult:
    known = _load_local_ops_csv(local_ops_csv)
    solutions = analyze_unsupported_ops(unsupported_csv, known)

    need_dev_ops = [s.op_name for s in solutions if s.strategy == "需开发"]

    if op_name:
        if op_name not in need_dev_ops:
            return DevTaskResult(
                status=f"算子 '{op_name}' 不在待开发列表中",
                total_ops=0,
                generated_tasks=[],
            )
        need_dev_ops = [op_name]

    if not need_dev_ops:
        return DevTaskResult(
            status="没有需要开发的算子",
            total_ops=0,
            generated_tasks=[],
        )

    tasks = batch_generate_dev_tasks(need_dev_ops, output_dir)

    return DevTaskResult(
        status=f"已为 {len(need_dev_ops)} 个算子生成开发任务，{len(tasks)} 个成功",
        total_ops=len(need_dev_ops),
        generated_tasks=[str(t) for t in tasks],
    )


def rewrite_unsupported_ops(
    unsupported_csv: Path,
    local_ops_csv: Path,
    output_dir: Optional[Path] = None,
    source_dir: Optional[Path] = None,
) -> RewriteResult:
    known = _load_local_ops_csv(local_ops_csv)
    solutions = analyze_unsupported_ops(unsupported_csv, known)

    mapped = sum(1 for s in solutions if s.strategy in ("已入库", "可映射"))
    decomposable = sum(1 for s in solutions if s.strategy == "可拆分")
    math_rewrite = sum(1 for s in solutions if s.strategy == "数学等价改写")
    need_dev = sum(1 for s in solutions if s.strategy == "需开发")

    generated_ops: List[str] = []
    patched_files: List[str] = []
    total_patches = 0

    if output_dir:
        generated_ops.extend(_write_decompose_stubs(solutions, output_dir))

    if source_dir and output_dir:
        patched_files, total_patches = apply_rewrites_to_source(source_dir, output_dir, solutions)
        generated_ops.extend(patched_files)

    for s in solutions:
        if s.strategy == "需开发":
            try:
                safe_name = s.op_name.replace(".", "_").replace("-", "_").replace("/", "_")
                path = generate_operator_scaffolding(safe_name)
                if path:
                    generated_ops.append(str(path))
            except Exception:
                pass

    status_lines = [
        f"分析完成：{len(solutions)} 个不支持的算子",
        f"  已入库/可映射: {mapped}",
        f"  可拆分: {decomposable}",
        f"  数学等价改写: {math_rewrite}",
        f"  需开发: {need_dev}",
    ]
    if total_patches:
        status_lines.append(f"已替换 {total_patches} 处调用，修改 {len(patched_files)} 个文件")
    if generated_ops:
        status_lines.append(f"已生成 {len(generated_ops)} 个文件/脚手架")

    return RewriteResult(
        status="\n".join(status_lines),
        total_ops=len(solutions),
        mapped_count=mapped,
        decomposable_count=decomposable,
        math_rewrite_count=math_rewrite,
        need_develop_count=need_dev,
        solutions=solutions,
        generated_ops=generated_ops,
        patched_files=patched_files,
        total_patches=total_patches,
    )
