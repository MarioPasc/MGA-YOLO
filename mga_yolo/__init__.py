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

import logging

# Global logger for the entire optimization module
LOGGER = logging.getLogger("mga_yolo")
LOGGER.setLevel(logging.INFO)

# Configure a default stream handler (can be overridden in cli or main)
_handler = logging.StreamHandler()
_formatter = logging.Formatter(
    "[MGA-YOLO]: %(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_handler.setFormatter(_formatter)
LOGGER.addHandler(_handler)

