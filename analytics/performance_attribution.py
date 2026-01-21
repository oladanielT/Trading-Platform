"""Performance attribution and regime-level breakdown.

Disabled by default; minimal conservative stubs.
"""
from typing import Dict, Any
from config.phase12_config import get as _get


def attribute(strategy_pnl: Dict[str, float]) -> Dict[str, Any]:
    enabled = _get("performance_attribution", "enabled", False)
    if not enabled:
        return {"attributed": False}
    # simple proportional attribution
    total = sum(strategy_pnl.values())
    if total == 0:
        return {"attributed": True, "shares": {k: 0.0 for k in strategy_pnl}}
    return {"attributed": True, "shares": {k: v / total for k, v in strategy_pnl.items()}}


__all__ = ["attribute"]
