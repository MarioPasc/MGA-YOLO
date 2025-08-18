"""MGA-YOLO custom nn modules package.
Initializes logging for segmentation-related modules.
"""
import logging

SEG_LOGGER = logging.getLogger("mga_yolo.segmentation")
if not SEG_LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    SEG_LOGGER.addHandler(handler)
SEG_LOGGER.setLevel(logging.DEBUG)

ATT_LOGGER = logging.getLogger("mga_yolo.attention")
if not ATT_LOGGER.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(formatter)
    ATT_LOGGER.addHandler(handler)
ATT_LOGGER.setLevel(logging.DEBUG)

__all__ = ["SEG_LOGGER", "ATT_LOGGER"]
