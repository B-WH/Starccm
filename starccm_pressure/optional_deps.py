"""性能敏感路径使用的可选依赖加载器。"""

from __future__ import annotations

import importlib
from typing import Any


def load_ckdtree() -> Any | None:
    """在安装 SciPy 时返回 scipy.spatial.cKDTree，否则返回 None。"""
    try:
        scipy_spatial = importlib.import_module("scipy.spatial")
    except Exception:
        return None
    return getattr(scipy_spatial, "cKDTree", None)
