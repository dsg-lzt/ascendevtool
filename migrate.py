#!/usr/bin/env python3
"""一键完成 算子替换 + CUDA→NPU 迁移"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rewriter.rewriter_core import rewrite_unsupported_ops
from migrator.torch_to_npu import transform_source


def main():
    parser = argparse.ArgumentParser(description="AscendDevTool 一键迁移")
    parser.add_argument("source", help="待迁移模型源码目录")
    parser.add_argument("csv", help="unsupported_api.csv 路径")
    parser.add_argument("-o", "--output", required=True, help="迁移输出目录")
    parser.add_argument("--local-csv", default=None, help="local_ops.csv 路径")
    args = parser.parse_args()

    src_dir = Path(args.source)
    csv_path = Path(args.csv)
    out_dir = Path(args.output)

    if not src_dir.is_dir():
        print(f"错误: 源码目录不存在: {args.source}")
        sys.exit(1)
    if not csv_path.is_file():
        print(f"错误: CSV 文件不存在: {args.csv}")
        sys.exit(1)

    local_csv = Path(args.local_csv) if args.local_csv else (
        Path(__file__).resolve().parents[1] / "patcher" / "local_op_lib" / "local_ops.csv"
    )

    # 1. 算子替换
    print(">>> 算子替换分析...")
    result = rewrite_unsupported_ops(csv_path, local_csv, out_dir, src_dir)
    print(result.status)

    # 2. CUDA→NPU 迁移
    print(">>> CUDA→NPU 迁移...")
    total = 0
    skipped = 0
    for f in out_dir.rglob("*.py"):
        if "ascend_pointnet2" in f.name or "gorilla" in f.name:
            skipped += 1
            continue
        src = f.read_text(encoding="utf-8")
        new_src, changes = transform_source(src)
        if changes > 0:
            f.write_text(new_src, encoding="utf-8")
            total += changes
    print(f"迁移完成: {total} 处变更 (跳过 {skipped} 个生成文件)")

    print(f"\n输出目录: {out_dir}")


if __name__ == "__main__":
    main()
