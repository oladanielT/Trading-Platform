"""Trade clustering utilities (good vs bad conditions).

Minimal clustering stub.
"""
from typing import List, Dict, Any
from config.phase12_config import get as _get


def cluster(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    enabled = _get("performance_attribution", "enabled", False)
    if not enabled:
        return {"clustered": False}
    # naive split by pnl sign
    good = [t for t in trades if t.get("pnl", 0) > 0]
    bad = [t for t in trades if t.get("pnl", 0) <= 0]
    return {"clustered": True, "good_count": len(good), "bad_count": len(bad)}


__all__ = ["cluster"]
