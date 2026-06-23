import sys
from pathlib import Path


def main() -> int:
    try:
        from PySide6.QtWidgets import QApplication  # type: ignore
        from PySide6.QtQml import QQmlApplicationEngine  # type: ignore
    except Exception as e:
        print(
            "缺少 GUI 依赖：请先安装 PySide6。\n"
            "建议：pip install -r gui/requirements.txt\n\n"
            f"导入错误：{type(e).__name__}: {e}",
            file=sys.stderr,
        )
        return 1

    # 延迟导入：只有 PySide6 可用时才加载 Backend（其内部会 import PySide6）
    from gui.backend import Backend

    app = QApplication(sys.argv)
    engine = QQmlApplicationEngine()

    backend = Backend()
    engine.rootContext().setContextProperty("backend", backend)

    app.aboutToQuit.connect(backend.shutdown)

    qml_path = Path(__file__).resolve().parent / "qml" / "Main.qml"
    engine.load(str(qml_path))
    if not engine.rootObjects():
        return 2

    return app.exec()
