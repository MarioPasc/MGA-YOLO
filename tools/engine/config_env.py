from __future__ import annotations
"""
Utilities for exporting BASE_FM_* configuration keys as environment variables.

Single entry point:
- apply_env_from_config(config, prefix="BASE_FM_", overwrite=True)

Rules:
- str/bool/int/float -> str(value)
- list/tuple/dict -> JSON-encoded via json.dumps
- None -> skipped
"""

from typing import Any, Dict, Mapping
import json
import os

from ultralytics.utils import LOGGER


def _to_env_string(value: Any) -> str | None:
    """Convert arbitrary config value into an environment variable string."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    try:
        return json.dumps(value)
    except Exception:
        return str(value)


def apply_env_from_config(
    config: Mapping[str, Any], *, prefix: str = "BASE_FM_", overwrite: bool = True
) -> Dict[str, Any]:
    """
    Extract BASE_FM_* keys, export to os.environ, and return a cleaned config.

    Args:
        config: Input configuration mapping (not mutated).
        prefix: Key prefix to export. Default "BASE_FM_".
        overwrite: Overwrite existing env vars if True.

    Returns:
        A new dict with keys starting with `prefix` removed.
    """
    cleaned: Dict[str, Any] = dict(config)
    exported = 0
    kept = 0

    for k in list(cleaned.keys()):
        if not isinstance(k, str) or not k.startswith(prefix):
            continue
        v = cleaned.pop(k)
        env_val = _to_env_string(v)
        if env_val is None:
            LOGGER.info(f"[BASE_FM] ENV skip (None): {k}")
            continue

        if not overwrite and k in os.environ:
            kept += 1
            LOGGER.info(f"[BASE_FM] ENV keep existing: {k}={os.environ.get(k, '')}")
            continue

        os.environ[k] = env_val
        exported += 1
        shown = env_val if len(env_val) <= 256 else env_val[:253] + "..."
        LOGGER.info(f"[BASE_FM] ENV set: {k}={shown}")

    if exported or kept:
        LOGGER.info(f"[BASE_FM] ENV summary: set={exported}, kept_existing={kept}")

    return cleaned
