"""Optional dependency loaders used by performance-sensitive paths."""

from __future__ import annotations

import importlib
from typing import Any


def load_ckdtree() -> Any | None:
    """Return scipy.spatial.cKDTree when SciPy is installed, otherwise None."""
    try:
        scipy_spatial = importlib.import_module("scipy.spatial")
    except Exception:
        return None
    return getattr(scipy_spatial, "cKDTree", None)
