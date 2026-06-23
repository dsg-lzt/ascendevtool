from __future__ import annotations

import json
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
    def shutdown(self) -> None:
        thread = self._thread
        if thread is None:
            return
        try:
            thread.quit()
            thread.wait(2000)
        except Exception:
            pass
