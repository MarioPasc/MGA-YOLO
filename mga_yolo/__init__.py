"""
Mask-Guided-Attention YOLO (MGA-YOLO)
------------------------------------
Research-grade fork of Ultralytics-YOLO that injects mask-guided CBAM.

Author: Mario Pascual González <mpascual@uma.es>
"""
from importlib.metadata import version as _get_version

__all__ = ["__version__"]
try:
    __version__: str = _get_version("mga-yolo")
except Exception:            # package not installed (editable checkout)
    __version__ = "0.0.0"
