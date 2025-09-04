"""
Utilities for handling MGA-specific configuration parameters.

This module provides a single entry point to:
- discover configuration keys starting with a given prefix (default: 'MGA_'),
- export them as environment variables for downstream components, and
- return a cleaned configuration dict with those keys removed.

Design notes:
- Environment variables must be strings. We convert values as follows:
  * str: used as-is
  * bool/int/float: str(value)
  * list/tuple/dict: JSON-encoded via json.dumps
  * None: skipped (not exported)
- If an environment variable already exists and overwrite=False, we keep the existing value
  and log the decision.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping
import json
import os

from mga_yolo.external.ultralytics.ultralytics.utils import LOGGER


def _to_env_string(value: Any) -> str | None:
    """Convert arbitrary config value into a stable environment variable string.

    Returns None if the value should not be exported (e.g., None).
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    # For structured values (lists, tuples, dicts), prefer JSON for round-trip safety.
    try:
        return json.dumps(value)
    except Exception:
        # Fallback to plain string representation if JSON fails.
        return str(value)


def apply_env_from_config(
    config: Mapping[str, Any], *, prefix: str = "MGA_", overwrite: bool = True
) -> Dict[str, Any]:
    """
    Extract MGA_* keys from config, set them as environment variables, and return a cleaned config.

    Args:
        config: Input configuration mapping (will not be mutated).
        prefix: Key prefix to match for environment export. Defaults to 'MGA_'.
        overwrite: If True, overwrite existing environment variables. If False, keep existing
                   environment values and do not export new ones for matching keys.

    Returns:
        A new dict with all keys starting with `prefix` removed.

    Logging:
        - Logs each exported variable as: [MGA] ENV set: NAME=VALUE (truncated if long)
        - Summarizes total exported and skipped (existing) counts.
    """
    cleaned: Dict[str, Any] = dict(config)

    exported = 0
    skipped_existing = 0
    for k in list(cleaned.keys()):
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        v = cleaned.pop(k)
        env_val = _to_env_string(v)
        if env_val is None:
            LOGGER.info(f"[MGA] ENV skip (None): {k}")
            continue

        if not overwrite and k in os.environ:
            skipped_existing += 1
            LOGGER.info(f"[MGA] ENV keep existing: {k}={os.environ.get(k, '')}")
            continue

        os.environ[k] = env_val
        exported += 1
        # Limit value length in log to avoid flooding
        shown = env_val if len(env_val) <= 256 else env_val[:253] + "..."
        LOGGER.info(f"[MGA] ENV set: {k}={shown}")

    if exported or skipped_existing:
        LOGGER.info(f"[MGA] ENV summary: set={exported}, kept_existing={skipped_existing}")

    return cleaned
