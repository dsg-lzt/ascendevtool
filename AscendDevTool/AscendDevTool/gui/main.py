"""GUI entrypoint.

This file stays small on purpose:
- re-exports scanning helpers for non-GUI usage
- keeps `python gui/main.py` working (adds workspace root to sys.path)
"""

from __future__ import annotations

import sys
from pathlib import Path

# When executed as a script (`python gui/main.py`), sys.path only contains the gui/ folder.
# Add workspace root so `import gui.*` works.
workspace_root = Path(__file__).resolve().parents[1]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from gui.scanner_core import ScanResult, scan_unsupported_ops


def main() -> int:
    from gui.app_main import main as _app_main

    return _app_main()


if __name__ == "__main__":
    raise SystemExit(main())