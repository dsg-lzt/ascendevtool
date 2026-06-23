import csv
import os
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


DEFAULT_TORCH_VERSION = "default"
SUPPORTED_TORCH_VERSIONS = [
    "1.11.0",
    "2.1.0",
    "2.2.0",
    "2.3.1",
    "2.4.0",
    "2.5.1",
    "2.6.0",
]


def get_torch_version_options() -> List[str]:
    return [DEFAULT_TORCH_VERSION, *SUPPORTED_TORCH_VERSIONS]


def _workspace_root() -> Path:
    # gui/scanner_core.py -> <workspace>/gui/scanner_core.py
    return Path(__file__).resolve().parents[1]


def _find_pytorch_analyse_sh() -> Optional[Path]:
    candidates = [
        Path("/usr/local/Ascend/ascend-toolkit/latest/tools/ms_fmk_transplt/pytorch_analyse.sh"),
        Path("/usr/local/Ascend/cann/tools/ms_fmk_transplt/pytorch_analyse.sh"),
    ]
    ascend_home = os.environ.get("ASCEND_TOOLKIT_HOME") or os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates.insert(0, Path(ascend_home) / "tools" / "ms_fmk_transplt" / "pytorch_analyse.sh")
    home_ascend = Path.home() / "Ascend" / "ascend-toolkit" / "latest" / "tools" / "ms_fmk_transplt" / "pytorch_analyse.sh"
    candidates.append(home_ascend)
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return c
    for c in candidates:
        if c.is_file():
            return c
    return None


def _detect_torch_version() -> Optional[str]:
    try:
        import torch  # type: ignore

        v = getattr(torch, "__version__", None)
        if not v:
            return None
        # strip local version suffixes: 2.1.0+cpu -> 2.1.0
        return str(v).split("+")[0]
    except Exception:
        return None


def _resolve_torch_version(version_choice: Optional[str]) -> Tuple[str, str]:
    choice = (version_choice or DEFAULT_TORCH_VERSION).strip()
    if choice and choice != DEFAULT_TORCH_VERSION:
        return choice, f"用户指定版本：{choice}"

    detected = _detect_torch_version()
    if detected:
        return detected, f"默认模式：检测到本机 torch 版本 {detected}"
    return "2.6.0", "默认模式：未检测到 torch，使用 2.6.0"


def _safe_token_for_dir(value: str, fallback: str) -> str:
    # Keep folder names predictable and shell-friendly.
    normalized = re.sub(r"[^0-9A-Za-z._-]", "_", value).strip("._-")
    if not normalized:
        return fallback
    return normalized


def _safe_version_for_dir(version: str) -> str:
    return _safe_token_for_dir(version, "2_6_0").replace(".", "_")


def _safe_source_name_for_dir(path: Path) -> str:
    return _safe_token_for_dir(path.name, "source")


def _iter_report_csv_files(reports_dir: Path) -> List[Path]:
    if not reports_dir.exists():
        return []
    return sorted(reports_dir.rglob("unsupported_api.csv"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_unsupported_ops_from_csv(csv_path: Path) -> Counter:
    counter: Counter = Counter()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        # canonical column: OP
        for row in reader:
            op = (row.get("OP") or row.get("op") or "").strip()
            if not op:
                continue
            counter[op] += 1
    return counter


def _summarize_counter(counter: Counter) -> Tuple[int, int, List[Tuple[str, int]]]:
    total_occurrences = int(sum(counter.values()))
    total_unique = int(len(counter))
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
    return total_unique, total_occurrences, items


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ScanResult:
    status: str
    total_unique_ops: int
    total_occurrences: int
    items: List[Tuple[str, int]]


def _run_official_scan_if_available(
    input_dir: Path,
    output_dir: Path,
    torch_version_choice: Optional[str] = DEFAULT_TORCH_VERSION,
) -> Tuple[bool, str]:
    tool = _find_pytorch_analyse_sh()
    if tool is None:
        return False, "未发现 CANN 扫描脚本 pytorch_analyse.sh（将尝试读取已有报告）"

    version, version_note = _resolve_torch_version(torch_version_choice)

    # 经验：官方脚本内部固定使用 python3，常见问题是系统 python 缺少依赖（例如 libcst），
    # 导致扫描失败或在某些环境下表现为“长时间无反馈”。
    # 这里优先用当前进程的 Python 解释器执行 analyse 入口，同时把 stdout/stderr 写入日志，
    # 并增加超时，避免 GUI 永久处于 busy。
    analyse_py = tool.parent / "analysis" / "pytorch_analyse.py"
    if not analyse_py.is_file():
        cmd = [str(tool), "-i", str(input_dir), "-o", str(output_dir), "-v", version, "-m", "torch_apis"]
        env = None
    else:
        cmd = [
            sys.executable,
            str(analyse_py),
            "-i",
            str(input_dir),
            "-o",
            str(output_dir),
            "-v",
            version,
            "-m",
            "torch_apis",
        ]
        env = os.environ.copy()
        # 让 analyse 包内相对导入可用
        env["PYTHONPATH"] = f"{tool.parent}:{env.get('PYTHONPATH', '')}"

    timeout_raw = os.environ.get("ASCEND_SCAN_TIMEOUT_SEC", "1800")
    try:
        timeout_sec = int(timeout_raw)
    except Exception:
        timeout_sec = 1800

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    # pytorch_analyse 在输出目录已存在时会交互询问是否覆盖；GUI 场景下 stdin 被关闭会导致 EOF。
    # 为避免交互提示并保留历史报告，这里每次扫描输出到一个新的时间戳子目录。
    source_name = _safe_source_name_for_dir(input_dir)
    run_output_dir = output_dir / f"run_{source_name}_torch{_safe_version_for_dir(version)}_{ts}"
    log_path = output_dir / f"gui_scan_{ts}.log"

    try:
        _ensure_dir(output_dir)
        _ensure_dir(run_output_dir)
        with log_path.open("w", encoding="utf-8", newline="\n") as log_f:
            cmd_for_log = list(cmd)
            try:
                o_idx = cmd_for_log.index("-o")
                if o_idx + 1 < len(cmd_for_log):
                    cmd_for_log[o_idx + 1] = str(run_output_dir)
            except ValueError:
                pass

            # 实际执行也替换输出目录
            cmd_to_run = list(cmd_for_log)

            log_f.write(f"cmd: {' '.join(cmd_to_run)}\n")
            log_f.write(f"timeout_sec: {timeout_sec} (<=0 means no-timeout)\n")
            log_f.flush()

            proc = subprocess.Popen(
                cmd_to_run,
                stdout=log_f,
                stderr=log_f,
                stdin=subprocess.PIPE,
                text=True,
                env=env,
            )
            try:
                # 官方工具在输出目录存在时会读取 stdin 做覆盖确认；这里默认喂入 continue，
                # 若工具未提示也不会影响正常运行。
                if timeout_sec <= 0:
                    proc.communicate(input="continue\n")
                else:
                    proc.communicate(input="continue\n", timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    pass
                return False, f"官方扫描超时（>{timeout_sec}s）已终止，详见日志：{log_path}"

        if proc.returncode != 0:
            tail = ""
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                tail_lines = lines[-20:] if len(lines) > 20 else lines
                tail = "\n".join(tail_lines).strip()
            except Exception:
                tail = ""
            extra = f"\n--- log tail ---\n{tail}" if tail else ""
            msg = f"官方扫描执行失败（退出码 {proc.returncode}），详见日志：{log_path}{extra}"
            return False, msg

        return True, (
            f"官方扫描完成（torch 版本参数: {version}，{version_note}），"
            f"输出目录：{run_output_dir}，日志：{log_path}"
        )

    except FileNotFoundError:
        return False, "未找到 pytorch_analyse.sh（将尝试读取已有报告）"
    except Exception as e:
        return False, f"调用官方扫描异常：{type(e).__name__}: {e}"


def _display_path_for_status(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


def _build_result_from_csv(csv_path: Path, status: str) -> ScanResult:
    counter = _load_unsupported_ops_from_csv(csv_path)
    total_unique, total_occurrences, items = _summarize_counter(counter)
    return ScanResult(
        status=status,
        total_unique_ops=total_unique,
        total_occurrences=total_occurrences,
        items=items,
    )


def load_unsupported_ops_from_path(report_path: Path) -> ScanResult:
    root = _workspace_root()

    if not report_path.exists():
        return ScanResult(
            status=f"结果路径不存在：{report_path}",
            total_unique_ops=0,
            total_occurrences=0,
            items=[],
        )

    csv_path: Optional[Path] = None
    if report_path.is_file():
        if report_path.suffix.lower() == ".csv":
            csv_path = report_path
    else:
        files = _iter_report_csv_files(report_path)
        if files:
            csv_path = files[0]

    if csv_path is None:
        return ScanResult(
            status=f"在路径下未找到 unsupported_api.csv：{report_path}",
            total_unique_ops=0,
            total_occurrences=0,
            items=[],
        )

    display = _display_path_for_status(csv_path, root)
    return _build_result_from_csv(csv_path, f"已加载历史扫描结果：{display}")


def scan_unsupported_ops(source_dir: Path, torch_version_choice: Optional[str] = DEFAULT_TORCH_VERSION) -> ScanResult:
    root = _workspace_root()
    reports_dir = root / "scanner" / "reports"
    _ensure_dir(reports_dir)

    status_parts: List[str] = []
    ran, msg = _run_official_scan_if_available(source_dir, reports_dir, torch_version_choice)
    status_parts.append(msg)

    report_files = _iter_report_csv_files(reports_dir)
    if not report_files:
        return ScanResult(
            status="；".join(status_parts)
            + "。未找到任何 unsupported_api.csv 报告，请先安装/配置 CANN 并确保扫描输出到 scanner/reports/",
            total_unique_ops=0,
            total_occurrences=0,
            items=[],
        )

    csv_path = report_files[0]
    status_parts.append(f"已读取报告：{_display_path_for_status(csv_path, root)}")
    if ran is False:
        status_parts.append("（未执行官方扫描，仅汇总现有报告）")

    return _build_result_from_csv(csv_path, "；".join(status_parts))
