from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Property, QThread, Signal, Slot

from gui.qt_models import UnsupportedOpsModel
from gui.scanner_core import (
    DEFAULT_TORCH_VERSION,
    ScanResult,
    get_torch_version_options,
    load_unsupported_ops_from_path,
    scan_unsupported_ops,
)
from migrator.migrator_core import MigrateResult, migrate_directory
from rewriter.rewriter_core import RewriteResult, rewrite_unsupported_ops, DevTaskResult, run_operator_development


class ScanWorker(QObject):
    finished = Signal(object)

    def __init__(self, source_dir: str, torch_version: str) -> None:
        super().__init__()
        self._source_dir = source_dir
        self._torch_version = torch_version

    @Slot()
    def run(self) -> None:
        try:
            result = scan_unsupported_ops(Path(self._source_dir), self._torch_version)
        except Exception as e:
            result = ScanResult(
                status=f"扫描异常：{type(e).__name__}: {e}",
                total_unique_ops=0,
                total_occurrences=0,
                items=[],
            )
        self.finished.emit(result)


class LoadResultWorker(QObject):
    finished = Signal(object)

    def __init__(self, report_path: str) -> None:
        super().__init__()
        self._report_path = report_path

    @Slot()
    def run(self) -> None:
        try:
            result = load_unsupported_ops_from_path(Path(self._report_path))
        except Exception as e:
            result = ScanResult(
                status=f"加载历史结果异常：{type(e).__name__}: {e}",
                total_unique_ops=0,
                total_occurrences=0,
                items=[],
            )
        self.finished.emit(result)


class MigrateWorker(QObject):
    finished = Signal(object)

    def __init__(self, source_dir: str, target_dir: str) -> None:
        super().__init__()
        self._source_dir = source_dir
        self._target_dir = target_dir

    @Slot()
    def run(self) -> None:
        try:
            result = migrate_directory(Path(self._source_dir), Path(self._target_dir))
        except Exception as e:
            result = MigrateResult(
                status=f"迁移异常：{type(e).__name__}: {e}",
                total_files=0,
                modified_files=0,
                total_changes=0,
                file_stats=[],
            )
        self.finished.emit(result)


class RewriteWorker(QObject):
    finished = Signal(object)

    def __init__(self, unsupported_csv: str, local_ops_csv: str, output_dir: str, source_dir: str) -> None:
        super().__init__()
        self._unsupported_csv = unsupported_csv
        self._local_ops_csv = local_ops_csv
        self._output_dir = output_dir
        self._source_dir = source_dir

    @Slot()
    def run(self) -> None:
        try:
            result = rewrite_unsupported_ops(
                Path(self._unsupported_csv),
                Path(self._local_ops_csv),
                Path(self._output_dir) if self._output_dir else None,
                Path(self._source_dir) if self._source_dir else None,
            )
        except Exception as e:
            result = RewriteResult(
                status=f"算子分析异常：{type(e).__name__}: {e}",
                total_ops=0,
                mapped_count=0,
                decomposable_count=0,
                math_rewrite_count=0,
                need_develop_count=0,
                solutions=[],
                generated_ops=[],
                patched_files=[],
                total_patches=0,
            )
        self.finished.emit(result)


class DevWorker(QObject):
    finished = Signal(object)

    def __init__(self, unsupported_csv: str, local_ops_csv: str, output_dir: str, op_name: str = "") -> None:
        super().__init__()
        self._unsupported_csv = unsupported_csv
        self._local_ops_csv = local_ops_csv
        self._output_dir = output_dir
        self._op_name = op_name

    @Slot()
    def run(self) -> None:
        try:
            result = run_operator_development(
                Path(self._unsupported_csv),
                Path(self._local_ops_csv),
                Path(self._output_dir) if self._output_dir else Path.home() / "ascend_dev_tasks",
                self._op_name if self._op_name else None,
            )
        except Exception as e:
            result = DevTaskResult(
                status=f"开发任务生成异常：{type(e).__name__}: {e}",
                total_ops=0,
                generated_tasks=[],
            )
        self.finished.emit(result)


class InferenceWorker(QObject):
    finished = Signal(object)

    def __init__(self, script_path: str, cwd: str) -> None:
        super().__init__()
        self._script_path = script_path
        self._cwd = cwd

    @Slot()
    def run(self) -> None:
        try:
            sp = Path(self._script_path)
            if not sp.is_file():
                self.finished.emit({"output": "", "exit_code": -1, "error": f"脚本不存在: {self._script_path}"})
                return

            if sp.suffix == ".sh":
                cmd = ["bash", str(sp)]
            elif sp.suffix == ".py":
                cmd = [sys.executable, str(sp)]
            else:
                cmd = [sys.executable, str(sp)]

            env = os.environ.copy()
            env.setdefault("PYTHONIOENCODING", "utf-8")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,
                cwd=self._cwd or str(sp.parent),
                env=env,
            )

            output = ""
            if proc.stdout:
                output += proc.stdout
            if proc.stderr:
                if output:
                    output += "\n"
                output += proc.stderr

            self.finished.emit({
                "output": output,
                "exit_code": proc.returncode,
                "error": "",
            })
        except subprocess.TimeoutExpired:
            self.finished.emit({"output": "", "exit_code": -1, "error": "推理超时（超过30分钟）"})
        except Exception as e:
            self.finished.emit({"output": "", "exit_code": -1, "error": f"{type(e).__name__}: {e}"})


class Backend(QObject):
    sourceDirChanged = Signal()
    targetDirChanged = Signal()
    reportPathChanged = Signal()
    torchVersionChanged = Signal()
    statusChanged = Signal()
    busyChanged = Signal()
    summaryChanged = Signal()
    unsupportedOpsChanged = Signal()
    unsupportedOpsJsonChanged = Signal()
    migrateStatusChanged = Signal()
    migrateSummaryChanged = Signal()
    rewriteStatusChanged = Signal()
    rewriteSummaryChanged = Signal()
    devStatusChanged = Signal()
    devSummaryChanged = Signal()
    devOpListChanged = Signal()
    inferenceScriptPathChanged = Signal()
    inferenceOutputChanged = Signal()
    inferenceStatusChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._source_dir = ""
        self._target_dir = ""
        self._report_path = ""
        self._torch_version_options = get_torch_version_options()
        self._torch_version = DEFAULT_TORCH_VERSION
        self._status = ""
        self._busy = False
        self._total_unique_ops = 0
        self._total_occurrences = 0
        self._model = UnsupportedOpsModel()
        self._unsupported_ops: List[Dict[str, Any]] = []
        self._unsupported_ops_json = "[]"
        self._thread: Optional[QThread] = None
        self._worker: Optional[ScanWorker] = None
        self._migrate_status = ""
        self._migrate_total_files = 0
        self._migrate_modified_files = 0
        self._migrate_total_changes = 0
        self._rewrite_status = ""
        self._rewrite_total_ops = 0
        self._rewrite_mapped = 0
        self._rewrite_decomposable = 0
        self._rewrite_need_dev = 0
        self._dev_status = ""
        self._dev_total_ops = 0
        self._dev_generated_tasks = 0
        self._dev_op_list_json = "[]"
        self._inference_script_path = ""
        self._inference_output = ""
        self._inference_status = ""

    def getSourceDir(self) -> str:  # noqa: N802
        return self._source_dir

    def setSourceDir(self, v: str) -> None:  # noqa: N802
        if v == self._source_dir:
            return
        self._source_dir = v
        self.sourceDirChanged.emit()

    sourceDir = Property(str, getSourceDir, setSourceDir, notify=sourceDirChanged)

    def getTargetDir(self) -> str:  # noqa: N802
        return self._target_dir

    def setTargetDir(self, v: str) -> None:  # noqa: N802
        if v == self._target_dir:
            return
        self._target_dir = v
        self.targetDirChanged.emit()

    targetDir = Property(str, getTargetDir, setTargetDir, notify=targetDirChanged)

    def getReportPath(self) -> str:  # noqa: N802
        return self._report_path

    def setReportPath(self, v: str) -> None:  # noqa: N802
        if v == self._report_path:
            return
        self._report_path = v
        self.reportPathChanged.emit()

    reportPath = Property(str, getReportPath, setReportPath, notify=reportPathChanged)

    def getTorchVersionOptions(self) -> List[str]:  # noqa: N802
        return self._torch_version_options

    torchVersionOptions = Property(list, getTorchVersionOptions, constant=True)

    def getTorchVersion(self) -> str:  # noqa: N802
        return self._torch_version

    def setTorchVersion(self, v: str) -> None:  # noqa: N802
        value = (v or "").strip() or DEFAULT_TORCH_VERSION
        if value not in self._torch_version_options:
            value = DEFAULT_TORCH_VERSION
        if value == self._torch_version:
            return
        self._torch_version = value
        self.torchVersionChanged.emit()

    torchVersion = Property(str, getTorchVersion, setTorchVersion, notify=torchVersionChanged)

    def getStatus(self) -> str:  # noqa: N802
        return self._status

    status = Property(str, getStatus, notify=statusChanged)

    def getBusy(self) -> bool:  # noqa: N802
        return self._busy

    busy = Property(bool, getBusy, notify=busyChanged)

    def getTotalUniqueOps(self) -> int:  # noqa: N802
        return self._total_unique_ops

    totalUniqueOps = Property(int, getTotalUniqueOps, notify=summaryChanged)

    def getTotalOccurrences(self) -> int:  # noqa: N802
        return self._total_occurrences

    totalOccurrences = Property(int, getTotalOccurrences, notify=summaryChanged)

    def getUnsupportedOpsModel(self) -> QObject:  # noqa: N802
        return self._model

    unsupportedOpsModel = Property(QObject, getUnsupportedOpsModel, constant=True)

    def getUnsupportedOps(self) -> List[Dict[str, Any]]:  # noqa: N802
        return self._unsupported_ops

    unsupportedOps = Property(list, getUnsupportedOps, notify=unsupportedOpsChanged)

    def getUnsupportedOpsJson(self) -> str:  # noqa: N802
        return self._unsupported_ops_json

    unsupportedOpsJson = Property(str, getUnsupportedOpsJson, notify=unsupportedOpsJsonChanged)

    def getMigrateStatus(self) -> str:  # noqa: N802
        return self._migrate_status

    migrateStatus = Property(str, getMigrateStatus, notify=migrateStatusChanged)

    def getMigrateTotalFiles(self) -> int:  # noqa: N802
        return self._migrate_total_files

    migrateTotalFiles = Property(int, getMigrateTotalFiles, notify=migrateSummaryChanged)

    def getMigrateModifiedFiles(self) -> int:  # noqa: N802
        return self._migrate_modified_files

    migrateModifiedFiles = Property(int, getMigrateModifiedFiles, notify=migrateSummaryChanged)

    def getMigrateTotalChanges(self) -> int:  # noqa: N802
        return self._migrate_total_changes

    migrateTotalChanges = Property(int, getMigrateTotalChanges, notify=migrateSummaryChanged)

    def getRewriteStatus(self) -> str:  # noqa: N802
        return self._rewrite_status

    rewriteStatus = Property(str, getRewriteStatus, notify=rewriteStatusChanged)

    def getRewriteTotalOps(self) -> int:  # noqa: N802
        return self._rewrite_total_ops

    rewriteTotalOps = Property(int, getRewriteTotalOps, notify=rewriteSummaryChanged)

    def getRewriteMapped(self) -> int:  # noqa: N802
        return self._rewrite_mapped

    rewriteMapped = Property(int, getRewriteMapped, notify=rewriteSummaryChanged)

    def getRewriteDecomposable(self) -> int:  # noqa: N802
        return self._rewrite_decomposable

    rewriteDecomposable = Property(int, getRewriteDecomposable, notify=rewriteSummaryChanged)

    def getRewriteNeedDev(self) -> int:  # noqa: N802
        return self._rewrite_need_dev

    rewriteNeedDev = Property(int, getRewriteNeedDev, notify=rewriteSummaryChanged)

    def getDevStatus(self) -> str:  # noqa: N802
        return self._dev_status

    devStatus = Property(str, getDevStatus, notify=devStatusChanged)

    def getDevTotalOps(self) -> int:  # noqa: N802
        return self._dev_total_ops

    devTotalOps = Property(int, getDevTotalOps, notify=devSummaryChanged)

    def getDevGeneratedTasks(self) -> int:  # noqa: N802
        return self._dev_generated_tasks

    devGeneratedTasks = Property(int, getDevGeneratedTasks, notify=devSummaryChanged)

    def getDevOpListJson(self) -> str:  # noqa: N802
        return self._dev_op_list_json

    devOpListJson = Property(str, getDevOpListJson, notify=devOpListChanged)

    def _refresh_dev_op_list(self) -> None:
        try:
            from rewriter.op_analyzer import analyze_unsupported_ops
            from pathlib import Path

            reports_dir = Path(__file__).resolve().parents[1] / "scanner" / "reports"
            csv_files = list(reports_dir.rglob("unsupported_api.csv"))
            if not csv_files:
                self._dev_op_list_json = "[]"
                return

            local_csv = Path(__file__).resolve().parents[1] / "patcher" / "local_op_lib" / "local_ops.csv"
            from rewriter.rewriter_core import _load_local_ops_csv
            known = _load_local_ops_csv(local_csv)
            solutions = analyze_unsupported_ops(csv_files[0], known)

            ops = []
            for s in solutions:
                if s.strategy == "需开发":
                    safe = s.op_name.replace(".", "_").replace("-", "_").replace("/", "_")
                    ops_src = Path(__file__).resolve().parents[1] / "op_builder" / "ops_src"
                    has_proj = (ops_src / f"{safe}Sample").is_dir()
                    ops.append({
                        "op": s.op_name,
                        "safeName": safe,
                        "hasProject": has_proj,
                    })
            self._dev_op_list_json = json.dumps(ops, ensure_ascii=False)
        except Exception:
            self._dev_op_list_json = "[]"
        self.devOpListChanged.emit()

    def _set_busy(self, v: bool) -> None:
        if v == self._busy:
            return
        self._busy = v
        self.busyChanged.emit()

    def _set_status(self, v: str) -> None:
        if v == self._status:
            return
        self._status = v
        self.statusChanged.emit()

    def _apply_result(self, result: ScanResult) -> None:
        self._set_status(result.status)
        self._total_unique_ops = result.total_unique_ops
        self._total_occurrences = result.total_occurrences
        self._model.setItems(result.items)
        self._unsupported_ops = [{"op": op, "count": count} for op, count in result.items]
        try:
            self._unsupported_ops_json = json.dumps(self._unsupported_ops, ensure_ascii=False)
        except Exception:
            self._unsupported_ops_json = "[]"
        self.summaryChanged.emit()
        self.unsupportedOpsChanged.emit()
        self.unsupportedOpsJsonChanged.emit()

    @Slot()
    def scanModels(self) -> None:  # noqa: N802
        if self._busy:
            return
        if not self._source_dir:
            self._set_status("请先选择待迁移模型目录")
            return
        if not Path(self._source_dir).exists():
            self._set_status("待迁移模型目录不存在")
            return

        self._set_busy(True)
        self._set_status("正在扫描/汇总，请稍候...")

        thread = QThread()
        worker = ScanWorker(self._source_dir, self._torch_version)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_scan_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    @Slot()
    def loadScannedResult(self) -> None:  # noqa: N802
        if self._busy:
            return
        if not self._report_path:
            self._set_status("请先选择已扫描结果路径")
            return
        if not Path(self._report_path).exists():
            self._set_status("已扫描结果路径不存在")
            return

        self._set_busy(True)
        self._set_status("正在加载历史扫描结果，请稍候...")

        thread = QThread()
        worker = LoadResultWorker(self._report_path)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_scan_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    @Slot(object)
    def _on_scan_finished(self, result: object) -> None:
        if isinstance(result, ScanResult):
            self._apply_result(result)
        else:
            self._set_status("扫描返回结果格式异常")
        self._set_busy(False)

    @Slot()
    def _on_thread_finished(self) -> None:
        self._worker = None
        self._thread = None

    @Slot()
    def migrateToNpu(self) -> None:  # noqa: N802
        if self._busy:
            return
        if not self._source_dir:
            self._migrate_status = "请先选择待迁移模型目录"
            self.migrateStatusChanged.emit()
            return
        if not self._target_dir:
            self._migrate_status = "请先选择迁移目标目录"
            self.migrateStatusChanged.emit()
            return
        if not Path(self._source_dir).exists():
            self._migrate_status = "待迁移模型目录不存在"
            self.migrateStatusChanged.emit()
            return

        self._set_busy(True)
        self._migrate_status = "正在迁移代码到 NPU，请稍候..."
        self.migrateStatusChanged.emit()

        thread = QThread()
        worker = MigrateWorker(self._source_dir, self._target_dir)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_migrate_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    @Slot(object)
    def _on_migrate_finished(self, result: object) -> None:
        if isinstance(result, MigrateResult):
            self._migrate_status = result.status
            self._migrate_total_files = result.total_files
            self._migrate_modified_files = result.modified_files
            self._migrate_total_changes = result.total_changes
        else:
            self._migrate_status = "迁移返回结果格式异常"
        self.migrateStatusChanged.emit()
        self.migrateSummaryChanged.emit()
        self._set_busy(False)

    @Slot()
    def analyzeOps(self) -> None:  # noqa: N802
        if self._busy:
            return
        if not self._source_dir:
            self._rewrite_status = "请先选择待迁移模型目录"
            self.rewriteStatusChanged.emit()
            return

        reports_dir = Path(__file__).resolve().parents[1] / "scanner" / "reports"
        csv_files = list(reports_dir.rglob("unsupported_api.csv"))
        if not csv_files:
            self._rewrite_status = "未找到 unsupported_api.csv，请先执行扫描"
            self.rewriteStatusChanged.emit()
            return

        unsupported_csv = str(csv_files[0])
        local_csv = str(Path(__file__).resolve().parents[1] / "patcher" / "local_op_lib" / "local_ops.csv")
        output_dir = self._target_dir or str(Path(__file__).resolve().parents[1] / "rewriter" / "output")

        self._set_busy(True)
        self._rewrite_status = "正在分析算子替换方案，请稍候..."
        self.rewriteStatusChanged.emit()

        thread = QThread()
        worker = RewriteWorker(unsupported_csv, local_csv, output_dir, self._source_dir)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_rewrite_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    @Slot(object)
    def _on_rewrite_finished(self, result: object) -> None:
        if isinstance(result, RewriteResult):
            self._rewrite_status = result.status
            self._rewrite_total_ops = result.total_ops
            self._rewrite_mapped = result.mapped_count
            self._rewrite_decomposable = result.decomposable_count
            self._rewrite_need_dev = result.need_develop_count
        else:
            self._rewrite_status = "算子分析返回结果格式异常"
        self.rewriteStatusChanged.emit()
        self.rewriteSummaryChanged.emit()
        self._refresh_dev_op_list()
        self._set_busy(False)

    @Slot()
    def developOps(self) -> None:  # noqa: N802
        if self._busy:
            return

        reports_dir = Path(__file__).resolve().parents[1] / "scanner" / "reports"
        csv_files = list(reports_dir.rglob("unsupported_api.csv"))
        if not csv_files:
            self._dev_status = "未找到 unsupported_api.csv，请先执行扫描"
            self.devStatusChanged.emit()
            return

        unsupported_csv = str(csv_files[0])
        local_csv = str(Path(__file__).resolve().parents[1] / "patcher" / "local_op_lib" / "local_ops.csv")
        output_dir = self._target_dir or str(Path(__file__).resolve().parents[1] / "rewriter" / "dev_tasks")

        self._set_busy(True)
        self._dev_status = "正在生成算子开发任务..."
        self.devStatusChanged.emit()

        thread = QThread()
        worker = DevWorker(unsupported_csv, local_csv, output_dir)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_dev_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    @Slot(object)
    def _on_dev_finished(self, result: object) -> None:
        if isinstance(result, DevTaskResult):
            self._dev_status = result.status
            self._dev_total_ops = result.total_ops
            self._dev_generated_tasks = len(result.generated_tasks)
        else:
            self._dev_status = "开发任务生成结果格式异常"
        self.devStatusChanged.emit()
        self.devSummaryChanged.emit()
        self._refresh_dev_op_list()
        self._set_busy(False)

    @Slot(str)
    def developSingleOp(self, op_name: str) -> None:  # noqa: N802
        if self._busy:
            return

        reports_dir = Path(__file__).resolve().parents[1] / "scanner" / "reports"
        csv_files = list(reports_dir.rglob("unsupported_api.csv"))
        if not csv_files:
            self._dev_status = "未找到 unsupported_api.csv，请先执行扫描"
            self.devStatusChanged.emit()
            return

        unsupported_csv = str(csv_files[0])
        local_csv = str(Path(__file__).resolve().parents[1] / "patcher" / "local_op_lib" / "local_ops.csv")
        output_dir = self._target_dir or str(Path(__file__).resolve().parents[1] / "rewriter" / "dev_tasks")

        self._set_busy(True)
        self._dev_status = f"正在生成 [{op_name}] 的开发任务..."
        self.devStatusChanged.emit()

        thread = QThread()
        worker = DevWorker(unsupported_csv, local_csv, output_dir, op_name)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_dev_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    def getInferenceScriptPath(self) -> str:  # noqa: N802
        return self._inference_script_path

    def setInferenceScriptPath(self, v: str) -> None:  # noqa: N802
        if v == self._inference_script_path:
            return
        self._inference_script_path = v
        self.inferenceScriptPathChanged.emit()

    inferenceScriptPath = Property(str, getInferenceScriptPath, setInferenceScriptPath, notify=inferenceScriptPathChanged)

    def getInferenceOutput(self) -> str:  # noqa: N802
        return self._inference_output

    inferenceOutput = Property(str, getInferenceOutput, notify=inferenceOutputChanged)

    def getInferenceStatus(self) -> str:  # noqa: N802
        return self._inference_status

    inferenceStatus = Property(str, getInferenceStatus, notify=inferenceStatusChanged)

    @Slot()
    def runInference(self) -> None:  # noqa: N802
        if self._busy:
            return
        if not self._inference_script_path:
            self._inference_status = "请先选择推理脚本"
            self.inferenceStatusChanged.emit()
            return

        sp = Path(self._inference_script_path)
        if not sp.is_file():
            self._inference_status = "推理脚本不存在"
            self.inferenceStatusChanged.emit()
            return

        self._set_busy(True)
        self._inference_status = "推理运行中..."
        self._inference_output = ""
        self.inferenceStatusChanged.emit()
        self.inferenceOutputChanged.emit()

        cwd = self._target_dir or str(sp.parent)

        thread = QThread()
        worker = InferenceWorker(str(sp), cwd)
        self._thread = thread
        self._worker = worker

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_inference_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        thread.start()

    @Slot(object)
    def _on_inference_finished(self, result: dict) -> None:
        output = result.get("output", "")
        exit_code = result.get("exit_code", -1)
        error = result.get("error", "")

        if error:
            self._inference_output = error
            self._inference_status = f"推理失败: {error}"
        else:
            self._inference_output = output
            if exit_code == 0:
                self._inference_status = "推理完成 (exit 0)"
            else:
                self._inference_status = f"推理结束 (exit {exit_code})"

        self.inferenceOutputChanged.emit()
        self.inferenceStatusChanged.emit()
        self._set_busy(False)

    @Slot()
    def shutdown(self) -> None:
        thread = self._thread
        if thread is None:
            return
        try:
            thread.quit()
            thread.wait(2000)
        except Exception:
            pass
