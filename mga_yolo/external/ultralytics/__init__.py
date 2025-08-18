"""Bridge package for vendored Ultralytics.

This outer-level ``ultralytics`` package exposes the inner vendored package located at
``ultralytics/ultralytics`` so absolute imports like ``import ultralytics`` and
``from ultralytics.models import ...`` resolve correctly even though the source tree
is nested.

It injects the inner subpackages into ``sys.modules`` under the expected absolute
names (``ultralytics.models``, ``ultralytics.nn`` etc.) and re-exports public symbols
defined by the inner package ``__all__``.
"""
from __future__ import annotations

import importlib as _importlib
import sys as _sys

_INNER_PKG_NAME = "mga_yolo.external.ultralytics.ultralytics"

# Subpackages to expose at top-level (preload before inner __init__ which expects absolute names)
_SUBPACKAGES = ["engine", "utils", "nn", "data", "models", "trackers", "hub", "solutions", "cfg", "assets"]
for _name in _SUBPACKAGES:
    full_inner = f"{_INNER_PKG_NAME}.{_name}"
    try:
        _mod = _importlib.import_module(full_inner)
        # expose as top-level ultralytics.<subpkg>
        _sys.modules[f"ultralytics.{_name}"] = _mod
    except Exception:
        pass

# Now import inner package (will find its absolute imports satisfied)
_inner = _importlib.import_module(_INNER_PKG_NAME)

if hasattr(_inner, "__all__"):
    for _sym in _inner.__all__:  # type: ignore[attr-defined]
        try:
            globals()[_sym] = getattr(_inner, _sym)
        except AttributeError:
            continue
    __all__ = list(_inner.__all__)  # type: ignore[attr-defined]
else:
    __all__ = []
