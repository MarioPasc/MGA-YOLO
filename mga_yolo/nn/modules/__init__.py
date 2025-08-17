"""MGA-YOLO custom nn modules package.
Initializes logging for segmentation-related modules.
"""
import logging

LOGGER = logging.getLogger("mga_yolo.segmentation")
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.DEBUG)

__all__ = ["LOGGER"]
