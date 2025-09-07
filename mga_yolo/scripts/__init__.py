"""Shared utilities for experiment orchestration scripts.

Defines a module-level logger used by performance comparison tooling.
"""
from __future__ import annotations

import logging

LOGGER = logging.getLogger("ModelComparisonExperiment")
if not LOGGER.handlers:  # Configure only once
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[ModelComparisonExperiment][%(asctime)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

__all__ = ["LOGGER"]
