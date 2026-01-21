"""
optimizer.py - Strategy parameter suggestion utility

Contract:
- Accepts backtest results and current strategy parameters
- Produces non-executing, auditable recommendations for parameter adjustments
- Returns recommendations only (no execution)
- Lightweight, rule-based heuristics so it's deterministic and easy to test

Primary API:
    recommend_parameters(current_params: Dict, backtest_results: Dict, param_bounds: Optional[Dict] = None) -> Dict

Returned dict:
    {
      "recommendations": [
        {
          "param": "position_size",
          "current": 0.01,
          "suggested": 0.005,
          "action": "scale_down" | "scale_up" | "set" | "adjust_behavior",
          "confidence": 0.85,              # 0.0 .. 1.0
          "reason": "short explanation",
        },
        ...
      ],
      "summary": {...}  # computed stats used by the heuristic
    }
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Recommendation:
    param: str
    current: Any
    suggested: Any
    action: str
    confidence: float  # 0.0 .. 1.0
    reason: str


def _safe_get_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metrics object returned by backtest engine / metrics collector.
    Falls back to deriving lightweight stats from trades list.
    """
    metrics = results.get("metrics") or {}
    trades = results.get("trades") or []

    out = {
        "total_trades": metrics.get("total_trades") or len(trades),
        "realized_pnl": metrics.get("realized_pnl") or sum(float(t.get("pnl", t.get("pnl_realized", 0.0))) for t in trades),
        "avg_trade_pnl": metrics.get("avg_trade_pnl"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "starting_equity": metrics.get("starting_equity"),
        "latest_equity": metrics.get("latest_equity") or metrics.get("ending_equity"),
    }

    # derive missing simple metrics from trades if possible
    if out["avg_trade_pnl"] is None and trades:
        out["avg_trade_pnl"] = (out["realized_pnl"] / out["total_trades"]) if out["total_trades"] > 0 else 0.0

    if out["win_rate_pct"] is None and trades:
        wins = sum(1 for t in trades if float(t.get("pnl", t.get("pnl_realized", 0.0))) > 0)
        out["win_rate_pct"] = (wins / out["total_trades"] * 100.0) if out["total_trades"] > 0 else 0.0

    if out["max_drawdown_pct"] is None:
        # if no drawdown provided, be conservative and leave as None
        out["max_drawdown_pct"] = metrics.get("max_drawdown_pct")

    return out


def _clip(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None and value < low:
        return float(low)
    if high is not None and value > high:
        return float(high)
    return value


def recommend_parameters(
    current_params: Dict[str, Any],
    backtest_results: Dict[str, Any],
    param_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Dict[str, Any]:
    """
    Produce recommendations based on provided backtest results and current params.

    Args:
        current_params: dictionary of current strategy parameters (e.g. position_size, stop_loss, lookback, signal_threshold, leverage)
        backtest_results: results from backtesting.engine.run_backtest (must include "metrics" and/or "trades")
        param_bounds: optional dict mapping param -> (min, max) to keep suggestions sane

    Returns:
        dict with keys:
          - recommendations: list of Recommendation-like dicts
          - summary: computed normalized metrics used for heuristics
    """
    params = copy.deepcopy(current_params or {})
    bounds = param_bounds or {}

    m = _safe_get_metrics(backtest_results)

    total = int(m.get("total_trades") or 0)
    pnl = float(m.get("realized_pnl") or 0.0)
    avg_pnl = None if m.get("avg_trade_pnl") is None else float(m.get("avg_trade_pnl"))
    win_rate = None if m.get("win_rate_pct") is None else float(m.get("win_rate_pct"))
    max_dd_pct = None if m.get("max_drawdown_pct") is None else float(m.get("max_drawdown_pct"))

    recs: List[Recommendation] = []

    # Heuristic 1: severe drawdown -> reduce risk aggressively
    if max_dd_pct is not None and max_dd_pct >= 20.0:
        # recommend halving position size if present
        if "position_size" in params and isinstance(params["position_size"], (int, float)):
            cur = float(params["position_size"])
            suggested = _clip(cur * 0.5, *bounds.get("position_size", (None, None)))
            reason = f"max_drawdown_pct={max_dd_pct:.2f}% is high -> reduce position size to limit risk"
            recs.append(Recommendation("position_size", cur, suggested, "scale_down", 0.95, reason))

        # recommend increasing stop loss aggressiveness if exists
        if "stop_loss" in params and isinstance(params["stop_loss"], (int, float)):
            cur = float(params["stop_loss"])
            # make stop loss tighter (smaller absolute pct)
            suggested = _clip(cur * 0.7, *bounds.get("stop_loss", (None, None)))
            reason = "high drawdown suggests tightening stop_loss"
            recs.append(Recommendation("stop_loss", cur, suggested, "set", 0.9, reason))

        # suggest reducing leverage
        if "leverage" in params and isinstance(params["leverage"], (int, float)):
            cur = float(params["leverage"])
            suggested = _clip(max(1.0, cur * 0.5), *bounds.get("leverage", (1.0, None)))
            recs.append(Recommendation("leverage", cur, suggested, "set", 0.9, "high drawdown -> reduce leverage"))

    # Heuristic 2: consistently profitable with good win rate -> consider increasing risk modestly
    if pnl > 0 and win_rate is not None and win_rate >= 60.0:
        if "position_size" in params and isinstance(params["position_size"], (int, float)):
            cur = float(params["position_size"])
            suggested = _clip(cur * 1.1, *bounds.get("position_size", (None, None)))
            reason = f"positive PnL and high win_rate ({win_rate:.1f}%) -> modestly increase position_size"
            recs.append(Recommendation("position_size", cur, suggested, "scale_up", 0.6, reason))

        # if stop_loss exists, consider widening slightly to avoid noise exit when performance is good
        if "stop_loss" in params and isinstance(params["stop_loss"], (int, float)):
            cur = float(params["stop_loss"])
            suggested = _clip(cur * 1.05, *bounds.get("stop_loss", (None, None)))
            recs.append(Recommendation("stop_loss", cur, suggested, "set", 0.5, "good performance -> slightly widen stop_loss"))

    # Heuristic 3: poor average trade and low win rate -> reduce aggression and improve filters
    if (avg_pnl is not None and avg_pnl < 0.0) and (win_rate is not None and win_rate < 50.0):
        # reduce position size
        if "position_size" in params and isinstance(params["position_size"], (int, float)):
            cur = float(params["position_size"])
            suggested = _clip(cur * 0.6, *bounds.get("position_size", (None, None)))
            recs.append(Recommendation("position_size", cur, suggested, "scale_down", 0.9, "negative avg_trade_pnl & low win_rate -> trim risk"))

        # tighten signal (increase threshold) if parameter exists
        if "signal_threshold" in params and isinstance(params["signal_threshold"], (int, float)):
            cur = float(params["signal_threshold"])
            suggested = _clip(cur * 1.2, *bounds.get("signal_threshold", (None, None)))
            recs.append(Recommendation("signal_threshold", cur, suggested, "set", 0.8, "low quality signals -> raise threshold to reduce false entries"))

        # consider larger lookback to reduce noise
        if "lookback" in params and isinstance(params["lookback"], (int, float)):
            cur = float(params["lookback"])
            suggested = _clip(max(1, cur * 1.5), *bounds.get("lookback", (1, None)))
            recs.append(Recommendation("lookback", cur, suggested, "set", 0.6, "increase lookback to smooth noisy signals"))

    # Heuristic 4: too few trades -> increase sensitivity / lower thresholds
    if total < 20:
        if "signal_threshold" in params and isinstance(params["signal_threshold"], (int, float)):
            cur = float(params["signal_threshold"])
            suggested = _clip(cur * 0.85, *bounds.get("signal_threshold", (None, None)))
            recs.append(Recommendation("signal_threshold", cur, suggested, "set", 0.5, "low trade count -> lower threshold to increase trade frequency"))

        if "lookback" in params and isinstance(params["lookback"], (int, float)):
            cur = float(params["lookback"])
            suggested = _clip(max(1, cur * 0.8), *bounds.get("lookback", (1, None)))
            recs.append(Recommendation("lookback", cur, suggested, "set", 0.45, "low trade count -> reduce lookback to react faster"))

    # Heuristic 5: if avg_trade_pnl positive but win_rate low (few big winners) -> prefer increasing take_profit or letting winners run
    if (avg_pnl is not None and avg_pnl > 0) and (win_rate is not None and win_rate < 40):
        if "take_profit" in params and isinstance(params["take_profit"], (int, float)):
            cur = float(params["take_profit"])
            suggested = _clip(cur * 1.2, *bounds.get("take_profit", (None, None)))
            recs.append(Recommendation("take_profit", cur, suggested, "set", 0.6, "few big winners -> increase take_profit to capture larger moves"))

    # Generic sanity: if no pars found to adjust, provide soft guidance
    if not recs:
        # Provide conservative generic guidance
        recs.append(Recommendation("advice", None, None, "adjust_behavior", 0.3, "no strong signals from backtest; consider more data or run parameter sweep"))

    # Convert to serializable dicts
    recommendations = [asdict(r) for r in recs]

    summary = {
        "derived_metrics": m,
        "param_count": len(params),
        "param_bounds_provided": bool(param_bounds),
    }

    return {"recommendations": recommendations, "summary": summary}


# Convenience sync wrapper for common call sites
def suggest(current_params: Dict[str, Any], backtest_results: Dict[str, Any], param_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> Dict[str, Any]:
    return recommend_parameters(current_params, backtest_results, param_bounds)


# Example usage (local)
if __name__ == "__main__":
    # sample current params and fake backtest results
    cur = {"position_size": 0.01, "stop_loss": 0.02, "take_profit": 0.04, "lookback": 20, "signal_threshold": 0.5, "leverage": 2}
    fake_results = {"metrics": {"total_trades": 12, "realized_pnl": -500.0, "avg_trade_pnl": -41.6, "win_rate_pct": 35.0, "max_drawdown_pct": 18.0}}
    out = suggest(cur, fake_results, param_bounds={"position_size": (0.0001, 1.0), "stop_loss": (0.001, 0.5), "leverage": (1.0, 10.0)})
    import json
    print(json.dumps(out, indent=2))
```# filepath: /home/oladan/Trading-Platform/ai/optimizer.py
"""
optimizer.py - Strategy parameter suggestion utility

Contract:
- Accepts backtest results and current strategy parameters
- Produces non-executing, auditable recommendations for parameter adjustments
- Returns recommendations only (no execution)
- Lightweight, rule-based heuristics so it's deterministic and easy to test

Primary API:
    recommend_parameters(current_params: Dict, backtest_results: Dict, param_bounds: Optional[Dict] = None) -> Dict

Returned dict:
    {
      "recommendations": [
        {
          "param": "position_size",
          "current": 0.01,
          "suggested": 0.005,
          "action": "scale_down" | "scale_up" | "set" | "adjust_behavior",
          "confidence": 0.85,              # 0.0 .. 1.0
          "reason": "short explanation",
        },
        ...
      ],
      "summary": {...}  # computed stats used by the heuristic
    }
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Recommendation:
    param: str
    current: Any
    suggested: Any
    action: str
    confidence: float  # 0.0 .. 1.0
    reason: str


def _safe_get_metrics(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize metrics object returned by backtest engine / metrics collector.
    Falls back to deriving lightweight stats from trades list.
    """
    metrics = results.get("metrics") or {}
    trades = results.get("trades") or []

    out = {
        "total_trades": metrics.get("total_trades") or len(trades),
        "realized_pnl": metrics.get("realized_pnl") or sum(float(t.get("pnl", t.get("pnl_realized", 0.0))) for t in trades),
        "avg_trade_pnl": metrics.get("avg_trade_pnl"),
        "win_rate_pct": metrics.get("win_rate_pct"),
        "max_drawdown_pct": metrics.get("max_drawdown_pct"),
        "starting_equity": metrics.get("starting_equity"),
        "latest_equity": metrics.get("latest_equity") or metrics.get("ending_equity"),
    }

    # derive missing simple metrics from trades if possible
    if out["avg_trade_pnl"] is None and trades:
        out["avg_trade_pnl"] = (out["realized_pnl"] / out["total_trades"]) if out["total_trades"] > 0 else 0.0

    if out["win_rate_pct"] is None and trades:
        wins = sum(1 for t in trades if float(t.get("pnl", t.get("pnl_realized", 0.0))) > 0)
        out["win_rate_pct"] = (wins / out["total_trades"] * 100.0) if out["total_trades"] > 0 else 0.0

    if out["max_drawdown_pct"] is None:
        # if no drawdown provided, be conservative and leave as None
        out["max_drawdown_pct"] = metrics.get("max_drawdown_pct")

    return out


def _clip(value: float, low: Optional[float], high: Optional[float]) -> float:
    if low is not None and value < low:
        return float(low)
    if high is not None and value > high:
        return float(high)
    return value


def recommend_parameters(
    current_params: Dict[str, Any],
    backtest_results: Dict[str, Any],
    param_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Dict[str, Any]:
    """
    Produce recommendations based on provided backtest results and current params.

    Args:
        current_params: dictionary of current strategy parameters (e.g. position_size, stop_loss, lookback, signal_threshold, leverage)
        backtest_results: results from backtesting.engine.run_backtest (must include "metrics" and/or "trades")
        param_bounds: optional dict mapping param -> (min, max) to keep suggestions sane

    Returns:
        dict with keys:
          - recommendations: list of Recommendation-like dicts
          - summary: computed normalized metrics used for heuristics
    """
    params = copy.deepcopy(current_params or {})
    bounds = param_bounds or {}

    m = _safe_get_metrics(backtest_results)

    total = int(m.get("total_trades") or 0)
    pnl = float(m.get("realized_pnl") or 0.0)
    avg_pnl = None if m.get("avg_trade_pnl") is None else float(m.get("avg_trade_pnl"))
    win_rate = None if m.get("win_rate_pct") is None else float(m.get("win_rate_pct"))
    max_dd_pct = None if m.get("max_drawdown_pct") is None else float(m.get("max_drawdown_pct"))

    recs: List[Recommendation] = []

    # Heuristic 1: severe drawdown -> reduce risk aggressively
    if max_dd_pct is not None and max_dd_pct >= 20.0:
        # recommend halving position size if present
        if "position_size" in params and isinstance(params["position_size"], (int, float)):
            cur = float(params["position_size"])
            suggested = _clip(cur * 0.5, *bounds.get("position_size", (None, None)))
            reason = f"max_drawdown_pct={max_dd_pct:.2f}% is high -> reduce position size to limit risk"
            recs.append(Recommendation("position_size", cur, suggested, "scale_down", 0.95, reason))

        # recommend increasing stop loss aggressiveness if exists
        if "stop_loss" in params and isinstance(params["stop_loss"], (int, float)):
            cur = float(params["stop_loss"])
            # make stop loss tighter (smaller absolute pct)
            suggested = _clip(cur * 0.7, *bounds.get("stop_loss", (None, None)))
            reason = "high drawdown suggests tightening stop_loss"
            recs.append(Recommendation("stop_loss", cur, suggested, "set", 0.9, reason))

        # suggest reducing leverage
        if "leverage" in params and isinstance(params["leverage"], (int, float)):
            cur = float(params["leverage"])
            suggested = _clip(max(1.0, cur * 0.5), *bounds.get("leverage", (1.0, None)))
            recs.append(Recommendation("leverage", cur, suggested, "set", 0.9, "high drawdown -> reduce leverage"))

    # Heuristic 2: consistently profitable with good win rate -> consider increasing risk modestly
    if pnl > 0 and win_rate is not None and win_rate >= 60.0:
        if "position_size" in params and isinstance(params["position_size"], (int, float)):
            cur = float(params["position_size"])
            suggested = _clip(cur * 1.1, *bounds.get("position_size", (None, None)))
            reason = f"positive PnL and high win_rate ({win_rate:.1f}%) -> modestly increase position_size"
            recs.append(Recommendation("position_size", cur, suggested, "scale_up", 0.6, reason))

        # if stop_loss exists, consider widening slightly to avoid noise exit when performance is good
        if "stop_loss" in params and isinstance(params["stop_loss"], (int, float)):
            cur = float(params["stop_loss"])
            suggested = _clip(cur * 1.05, *bounds.get("stop_loss", (None, None)))
            recs.append(Recommendation("stop_loss", cur, suggested, "set", 0.5, "good performance -> slightly widen stop_loss"))

    # Heuristic 3: poor average trade and low win rate -> reduce aggression and improve filters
    if (avg_pnl is not None and avg_pnl < 0.0) and (win_rate is not None and win_rate < 50.0):
        # reduce position size
        if "position_size" in params and isinstance(params["position_size"], (int, float)):
            cur = float(params["position_size"])
            suggested = _clip(cur * 0.6, *bounds.get("position_size", (None, None)))
            recs.append(Recommendation("position_size", cur, suggested, "scale_down", 0.9, "negative avg_trade_pnl & low win_rate -> trim risk"))

        # tighten signal (increase threshold) if parameter exists
        if "signal_threshold" in params and isinstance(params["signal_threshold"], (int, float)):
            cur = float(params["signal_threshold"])
            suggested = _clip(cur * 1.2, *bounds.get("signal_threshold", (None, None)))
            recs.append(Recommendation("signal_threshold", cur, suggested, "set", 0.8, "low quality signals -> raise threshold to reduce false entries"))

        # consider larger lookback to reduce noise
        if "lookback" in params and isinstance(params["lookback"], (int, float)):
            cur = float(params["lookback"])
            suggested = _clip(max(1, cur * 1.5), *bounds.get("lookback", (1, None)))
            recs.append(Recommendation("lookback", cur, suggested, "set", 0.6, "increase lookback to smooth noisy signals"))

    # Heuristic 4: too few trades -> increase sensitivity / lower thresholds
    if total < 20:
        if "signal_threshold" in params and isinstance(params["signal_threshold"], (int, float)):
            cur = float(params["signal_threshold"])
            suggested = _clip(cur * 0.85, *bounds.get("signal_threshold", (None, None)))
            recs.append(Recommendation("signal_threshold", cur, suggested, "set", 0.5, "low trade count -> lower threshold to increase trade frequency"))

        if "lookback" in params and isinstance(params["lookback"], (int, float)):
            cur = float(params["lookback"])
            suggested = _clip(max(1, cur * 0.8), *bounds.get("lookback", (1, None)))
            recs.append(Recommendation("lookback", cur, suggested, "set", 0.45, "low trade count -> reduce lookback to react faster"))

    # Heuristic 5: if avg_trade_pnl positive but win_rate low (few big winners) -> prefer increasing take_profit or letting winners run
    if (avg_pnl is not None and avg_pnl > 0) and (win_rate is not None and win_rate < 40):
        if "take_profit" in params and isinstance(params["take_profit"], (int, float)):
            cur = float(params["take_profit"])
            suggested = _clip(cur * 1.2, *bounds.get("take_profit", (None, None)))
            recs.append(Recommendation("take_profit", cur, suggested, "set", 0.6, "few big winners -> increase take_profit to capture larger moves"))

    # Generic sanity: if no pars found to adjust, provide soft guidance
    if not recs:
        # Provide conservative generic guidance
        recs.append(Recommendation("advice", None, None, "adjust_behavior", 0.3, "no strong signals from backtest; consider more data or run parameter sweep"))

    # Convert to serializable dicts
    recommendations = [asdict(r) for r in recs]

    summary = {
        "derived_metrics": m,
        "param_count": len(params),
        "param_bounds_provided": bool(param_bounds),
    }

    return {"recommendations": recommendations, "summary": summary}


# Convenience sync wrapper for common call sites
def suggest(current_params: Dict[str, Any], backtest_results: Dict[str, Any], param_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None) -> Dict[str, Any]:
    return recommend_parameters(current_params, backtest_results, param_bounds)


# Example usage (local)
if __name__ == "__main__":
    # sample current params and fake backtest results
    cur = {"position_size": 0.01, "stop_loss": 0.02, "take_profit": 0.04, "lookback": 20, "signal_threshold": 0.5, "leverage": 2}
    fake_results = {"metrics": {"total_trades": 12, "realized_pnl": -500.0, "avg_trade_pnl": -41.6, "win_rate_pct": 35.0, "max_drawdown_pct": 18.0}}
    out = suggest(cur, fake_results, param_bounds={"position_size": (0.0001, 1.0), "stop_loss": (0.001, 0.5), "leverage": (1.0, 10.0)})
    import json
    print(json.dumps(out, indent=2))