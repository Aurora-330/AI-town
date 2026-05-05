"""Pytest shared setup for backend/tests.

确保测试从 backend/tests/ 子目录运行时，仍能直接导入
backend 根下的模块，例如 agents、prompt_builder、tools 等。
"""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
