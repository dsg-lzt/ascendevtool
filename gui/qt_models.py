from __future__ import annotations

from typing import Any, Dict, List, Tuple

from PySide6.QtCore import QAbstractListModel, QByteArray, QModelIndex, Qt


class UnsupportedOpsModel(QAbstractListModel):
    OP_ROLE = Qt.UserRole + 1
    COUNT_ROLE = Qt.UserRole + 2

    def __init__(self) -> None:
        super().__init__()
        self._items: List[Tuple[str, int]] = []

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:  # noqa: N802
        if not index.isValid() or index.row() < 0 or index.row() >= len(self._items):
            return None
        op, count = self._items[index.row()]
        if role == Qt.DisplayRole:
            return op
        if role == self.OP_ROLE:
            return op
        if role == self.COUNT_ROLE:
            return count
        return None

    def roleNames(self) -> Dict[int, bytes]:  # noqa: N802
        return {
            self.OP_ROLE: QByteArray(b"op"),
            self.COUNT_ROLE: QByteArray(b"count"),
        }

    def setItems(self, items: List[Tuple[str, int]]) -> None:  # noqa: N802
        self.beginResetModel()
        self._items = items
        self.endResetModel()
