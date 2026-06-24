from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from migrator.torch_to_npu import transform_source


@dataclass
class FileMigrationStats:
    path: Path
    changes: int
    error: Optional[str] = None


@dataclass(frozen=True)
class MigrateResult:
    status: str
    total_files: int
    modified_files: int
    total_changes: int
    file_stats: List[FileMigrationStats]


def migrate_file(input_path: Path, output_path: Path) -> FileMigrationStats:
    try:
        source = input_path.read_text(encoding="utf-8")
    except Exception as e:
        return FileMigrationStats(path=input_path, changes=0, error=f"读取失败: {e}")

    if not source.strip():
        return FileMigrationStats(path=input_path, changes=0)

    try:
        new_source, changes = transform_source(source)
    except Exception as e:
        return FileMigrationStats(path=input_path, changes=0, error=f"转换失败: {e}")

    if changes == 0:
        return FileMigrationStats(path=input_path, changes=0)

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(new_source, encoding="utf-8")
    except Exception as e:
        return FileMigrationStats(path=input_path, changes=changes, error=f"写入失败: {e}")

    return FileMigrationStats(path=input_path, changes=changes)


def migrate_directory(source_dir: Path, target_dir: Path) -> MigrateResult:
    py_files = sorted(source_dir.rglob("*.py"))
    file_stats: List[FileMigrationStats] = []

    for py_file in py_files:
        rel = py_file.relative_to(source_dir)
        out = target_dir / rel
        stats = migrate_file(py_file, out)
        file_stats.append(stats)

    modified_files = sum(1 for s in file_stats if s.changes > 0 and s.error is None)
    total_changes = sum(s.changes for s in file_stats if s.error is None)

    if not file_stats:
        return MigrateResult(
            status="源目录下未找到任何 .py 文件",
            total_files=0,
            modified_files=0,
            total_changes=0,
            file_stats=[],
        )

    errors = [s for s in file_stats if s.error]
    status_lines = [
        f"迁移完成：共扫描 {len(file_stats)} 个 .py 文件",
        f"修改 {modified_files} 个文件，共 {total_changes} 处变更",
    ]
    if errors:
        status_lines.append(f"{len(errors)} 个文件处理异常")
        for e in errors[:5]:
            status_lines.append(f"  - {e.path.name}: {e.error}")
        if len(errors) > 5:
            status_lines.append(f"  ... 及其他 {len(errors) - 5} 个")

    return MigrateResult(
        status="\n".join(status_lines),
        total_files=len(file_stats),
        modified_files=modified_files,
        total_changes=total_changes,
        file_stats=file_stats,
    )
