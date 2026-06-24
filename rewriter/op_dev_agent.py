from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

OP_BUILDER_DIR = Path(__file__).resolve().parents[1] / "op_builder"
OPS_SRC_DIR = OP_BUILDER_DIR / "ops_src"
CANNBOT_SKILLS_DIR = Path("/home/lzt/ascend_project/cannbot-skills")

RELEVANT_SKILLS = [
    "ascendc-api-best-practices",
    "ascendc-simt-best-practices",
    "ascendc-simt-tiling-design",
    "ascendc-direct-invoke-template",
    "ascendc-registry-invoke-template",
    "ascendc-performance-best-practices",
    "ascendc-code-review",
]


def _find_msopgen() -> Optional[str]:
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = os.path.join(path, "msopgen")
        if os.access(candidate, os.X_OK):
            return candidate
    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH") or ""
    if ascend_home:
        for sub in ["bin", "tools/ccec_compiler/bin", "compiler/bin"]:
            candidate = os.path.join(ascend_home, sub, "msopgen")
            if os.access(candidate, os.X_OK):
                return candidate
    return None


def _generate_json_for_ops(op_names: List[str], output_path: Path) -> Path:
    op_defs = []
    for name in op_names:
        safe = name.replace(".", "_").replace("-", "_").replace("/", "_")
        op_defs.append({
            "op": safe,
            "language": "cpp",
            "input_desc": [
                {"name": "x0", "param_type": "required", "format": ["ND"], "type": ["float16", "float32"]},
                {"name": "x1", "param_type": "required", "format": ["ND"], "type": ["float16", "float32"]},
            ],
            "output_desc": [
                {"name": "y0", "param_type": "required", "format": ["ND"], "type": ["float16", "float32"]},
            ],
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(op_defs, f, indent=2)
    return output_path


def _run_msopgen(op_name: str, json_path: Path) -> Optional[Path]:
    msopgen = _find_msopgen()
    if msopgen is None:
        return None

    safe_name = op_name.replace(".", "_").replace("-", "_").replace("/", "_")
    target_dir = OPS_SRC_DIR / f"{safe_name}Sample" / "FrameworkLaunch"
    target_dir.mkdir(parents=True, exist_ok=True)

    tmp_workspace = Path.home() / "ascend_gen_workspace"
    tmp_workspace.mkdir(parents=True, exist_ok=True)
    tmp_workspace.chmod(0o755)

    tmp_json = tmp_workspace / f"{safe_name}.json"
    tmp_json.write_text(json_path.read_text())
    tmp_json.chmod(0o644)

    soc_ver = os.environ.get("SOC_VERSION", "Ascend310P")
    cmd = f"{msopgen} gen -i {tmp_json.name} -c ai_core-{soc_ver} -lan cpp -out {safe_name} -op {safe_name}"
    try:
        result = subprocess.run(cmd, shell=True, cwd=tmp_workspace, capture_output=True, text=True, timeout=120)
    except Exception:
        return None

    generated = tmp_workspace / safe_name
    if not generated.is_dir():
        return None

    import shutil
    dest = target_dir / safe_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(generated, dest)
    shutil.copy(tmp_json, target_dir / f"{safe_name}.json")

    shutil.rmtree(generated, ignore_errors=True)
    tmp_json.unlink(missing_ok=True)

    return dest


def _find_original_cuda_source(op_name: str) -> Optional[Path]:
    parts = op_name.split(".")
    short = parts[-1] if parts else op_name
    search_dir = Path("/home/lzt/ascend_project/SAM-6D")
    candidates = list(search_dir.rglob(f"*{short}*.cu")) + list(search_dir.rglob(f"*{short}*.cpp"))
    if candidates:
        return candidates[0]
    return None


def generate_agent_prompt(op_name: str, output_dir: Path) -> Optional[Path]:
    safe_name = op_name.replace(".", "_").replace("-", "_").replace("/", "_")

    project_dir = OPS_SRC_DIR / f"{safe_name}Sample" / "FrameworkLaunch" / safe_name
    kernel_file = project_dir / "op_kernel" / f"{safe_name}.cpp"
    host_file = project_dir / "op_host" / f"{safe_name}.cpp"
    tiling_file = project_dir / "op_host" / f"{safe_name}_tiling.h"

    if not kernel_file.is_file():
        return None

    kernel_src = kernel_file.read_text(encoding="utf-8")
    host_src = host_file.read_text(encoding="utf-8") if host_file.is_file() else "N/A"
    tiling_src = tiling_file.read_text(encoding="utf-8") if tiling_file.is_file() else "N/A"

    cuda_path = _find_original_cuda_source(op_name)
    cuda_src = ""
    if cuda_path:
        try:
            cuda_src = cuda_path.read_text(encoding="utf-8")
        except Exception:
            pass

    skills_context = ""
    for skill_name in RELEVANT_SKILLS:
        skill_md = CANNBOT_SKILLS_DIR / "ops" / skill_name / "SKILL.md"
        if skill_md.is_file():
            content = skill_md.read_text(encoding="utf-8")
            skills_context += f"\n## {skill_name}\n\n{content}\n\n---\n"

    op_builder_skills = Path(__file__).resolve().parents[1] / "op_builder" / "skill_docs"
    for doc in sorted(op_builder_skills.glob("*.md")):
        skills_context += f"\n## {doc.stem}\n\n{doc.read_text(encoding='utf-8')}\n\n---\n"

    prompt = f"""# TASK: Implement Ascend C Kernel for `{op_name}`

## Project Files

### Kernel Stub ({kernel_file})
```cpp
{kernel_src}
```

### Host Code ({host_file})
```cpp
{host_src}
```

### Tiling Header ({tiling_file})
```cpp
{tiling_src}
```

### Original CUDA Source ({cuda_path or 'not found'})
```cpp
{cuda_src[:5000]}
```

## Instructions

1. Read the above files to understand the operator semantics
2. Implement the Ascend C kernel following the SPMD model:
   - Use `DataCopy` for GM→UB and UB→GM transfers
   - Use Ascend C Vector APIs for computation (Add, Mul, Sub, Max, Min, etc.)
   - Use `Reduce` for block-level reduction operations
   - Use double-buffering (`DataCopyParams`) for overlapping compute with memory
   - Use `SetBlockDim` / `SetWorkSpace` in the host for multi-core distribution
3. Follow the existing patterns in the op_builder/ops_src/ directory
4. Write the complete implementation using the Edit tool

## Reference: CANNbot Skills + Ascend C API Docs

{skills_context[:20000]}
"""

    prompt_file = output_dir / f"AGENT_PROMPT_{safe_name}.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(prompt, encoding="utf-8")
    return prompt_file


def generate_dev_task(op_name: str, output_dir: Path) -> Optional[Path]:
    safe_name = op_name.replace(".", "_").replace("-", "_").replace("/", "_")
    return develop_single_op(op_name, output_dir)


def develop_single_op(op_name: str, output_dir: Path) -> Optional[Path]:
    safe_name = op_name.replace(".", "_").replace("-", "_").replace("/", "_")

    json_path = OP_BUILDER_DIR / f"_dev_{safe_name}.json"
    _generate_json_for_ops([op_name], json_path)

    _run_msopgen(op_name, json_path)
    json_path.unlink(missing_ok=True)

    prompt_file = generate_agent_prompt(op_name, output_dir)

    task_file = output_dir / f"DEV_TASK_{safe_name}.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Ascend C 算子开发任务：{op_name}",
        "",
        f"- 原始算子名：`{op_name}`",
        f"- Ascend C 工程：`op_builder/ops_src/{safe_name}Sample/FrameworkLaunch/{safe_name}/`",
        f"- Agent 开发提示：`{prompt_file}`",
        "",
        f"## 步骤",
        f"1. op_kernel/{safe_name}.cpp — 实现 Device 侧 Kernel 逻辑",
        f"2. op_host/{safe_name}.cpp — 实现 Host 侧 Tiling + 调用逻辑",
        f"3. 运行 `bash build.sh` 编译",
        f"4. 编译成功后将算子加入 `patcher/local_op_lib/local_ops.csv`",
    ]
    task_file.write_text("\n".join(lines), encoding="utf-8")
    return task_file


def batch_generate_dev_tasks(
    op_names: List[str],
    output_dir: Path,
) -> List[Path]:
    generated: List[Path] = []
    for op_name in op_names:
        try:
            task_path = generate_dev_task(op_name, output_dir)
            if task_path:
                generated.append(task_path)
        except Exception:
            continue
    return generated
