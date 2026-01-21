"""Detects strategy decay (declining expectancy, confidence collapse).

Minimal implementation: returns False when disabled.
"""
from typing import Dict, Any
from config.phase12_config import get as _get


def detect_decay(metrics: Dict[str, Any], threshold: float = 0.1) -> Dict[str, Any]:
    enabled = _get("walk_forward", "enabled", False)
    if not enabled:
        return {"decay": False, "reason": "disabled"}

    # placeholder: simple rule
    expectancy = metrics.get("expectancy", 0.0)
    if expectancy < -threshold:
        return {"decay": True, "reason": "expectancy_drop", "value": expectancy}
    return {"decay": False}


__all__ = ["detect_decay"]
