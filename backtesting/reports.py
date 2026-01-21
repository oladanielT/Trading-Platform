"""
reports.py - Backtest reporting utilities

Responsibilities:
- Generate textual summary metrics and plots from backtest results produced by backtesting.engine.BacktestEngine
- Read results returned by engine.run_backtest ({"trades": [...], "metrics": {...}})
- Optionally accept equity time-series if available to render an equity curve and drawdown chart
- Thread-safe file output and safe for headless environments (uses Agg backend)

This module provides:
- summarize_results(results) -> dict
- generate_plots(results, equity_series=None, out_dir="reports") -> dict(paths)
- save_report(results, equity_series=None, out_dir="reports") -> dict(paths + summary_path)

Notes:
- No trading logic is present here: plotting and aggregation only.
- matplotlib is required for plotting (pip install matplotlib). If matplotlib is missing,
  plotting functions raise an informative ImportError.
"""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Use Agg backend for headless servers
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
except Exception:
    plt = None  # type: ignore

_LOCK = threading.RLock()


def _ts_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _safe_mkdir(path: str) -> None:
    with _LOCK:
        os.makedirs(path, exist_ok=True)


def _compute_cumulative_pnl_from_trades(trades: Sequence[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """
    Returns list of (ts_ms, cumulative_pnl) ordered by ts ascending.
    Trades should contain exit_ts and pnl fields (pnl may be positive/negative).
    """
    sorted_trades = sorted((t for t in trades if t.get("exit_ts") is not None or t.get("exit_ts_ms") is not None),
                           key=lambda t: t.get("exit_ts", t.get("exit_ts_ms", 0)))
    entries: List[Tuple[int, float]] = []
    cum = 0.0
    for t in sorted_trades:
        ts = int(t.get("exit_ts", t.get("exit_ts_ms", t.get("exit_ts_ms", 0) or 0)))
        pnl = float(t.get("pnl", t.get("pnl_realized", 0.0)))
        cum += pnl
        entries.append((ts, cum))
    return entries


def _compute_drawdown(series: Sequence[Tuple[int, float]]) -> Tuple[List[Tuple[int, float]], float, float]:
    """
    Given series [(ts, value)], compute drawdown series [(ts, drawdown_abs)] and return max drawdown (abs, pct).
    Drawdown is peak - current_value. pct is (peak-current)/peak*100.
    """
    dd_series: List[Tuple[int, float]] = []
    peak = None
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    for ts, val in series:
        if peak is None or val > peak:
            peak = val
        dd = (peak - val) if peak is not None else 0.0
        dd_series.append((ts, dd))
        if dd > max_dd_abs:
            max_dd_abs = dd
            max_dd_pct = (dd / peak * 100.0) if peak and peak > 0 else 0.0
    return dd_series, max_dd_abs, max_dd_pct


def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a compact summary from engine.run_backtest results.

    Expected input:
      {
        "trades": [ { "exit_ts": ms, "pnl": float, ... }, ... ],
        "metrics": { ... }  # optional aggregated metrics from MetricsCollector
      }

    Returns dict with keys:
      - total_trades, wins, losses, win_rate_pct
      - realized_pnl, avg_trade_pnl
      - max_drawdown_abs, max_drawdown_pct (if metrics contains them or derivable)
      - starting_equity, ending_equity (from metrics if present)
    """
    trades = results.get("trades", []) or []
    metrics = results.get("metrics", {}) or {}

    total = len(trades)
    wins = sum(1 for t in trades if float(t.get("pnl", t.get("pnl_realized", 0.0))) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl", t.get("pnl_realized", 0.0))) <= 0)
    realized_pnl = sum(float(t.get("pnl", t.get("pnl_realized", 0.0))) for t in trades)
    avg_trade = (realized_pnl / total) if total > 0 else None
    win_rate = (wins / total * 100.0) if total > 0 else None

    # Prefer drawdown numbers from provided metrics if available
    max_dd_abs = metrics.get("max_drawdown_abs")
    max_dd_pct = metrics.get("max_drawdown_pct")
    starting_equity = metrics.get("starting_equity")
    latest_equity = metrics.get("latest_equity") or metrics.get("ending_equity")

    # If drawdown not provided, estimate from trades cumulative pnl
    if (max_dd_abs is None or max_dd_pct is None) and trades:
        cum = _compute_cumulative_pnl_from_trades(trades)
        if cum:
            _, max_dd_abs_est, max_dd_pct_est = _compute_drawdown(cum)
            max_dd_abs = max_dd_abs or max_dd_abs_est
            max_dd_pct = max_dd_pct or max_dd_pct_est

    summary = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "realized_pnl": realized_pnl,
        "avg_trade_pnl": avg_trade,
        "starting_equity": starting_equity,
        "ending_equity": latest_equity,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_pct": max_dd_pct,
    }
    return summary


def generate_plots(
    results: Dict[str, Any],
    equity_series: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "reports",
) -> Dict[str, str]:
    """
    Generate and persist plots for backtest results.

    Args:
      results: output of engine.run_backtest
      equity_series: optional list of {"ts": ms, "equity": float} to plot an equity curve.
                     If not provided, a cumulative realized-PnL curve derived from trades is used.
      out_dir: directory to write plots (created if missing)

    Returns:
      dict of plot name -> filepath
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    _safe_mkdir(out_dir)
    plots: Dict[str, str] = {}

    trades = results.get("trades", []) or []

    # 1) Equity curve
    if equity_series:
        series = sorted([(int(s["ts"]), float(s["equity"])) for s in equity_series], key=lambda x: x[0])
        times = [ _ts_to_dt(ts) for ts, _ in series ]
        values = [ val for _, val in series ]
        title = "Equity Curve (provided samples)"
    else:
        cum = _compute_cumulative_pnl_from_trades(trades)
        if not cum:
            # nothing to plot
            series = []
            times = []
            values = []
        else:
            times = [ _ts_to_dt(ts) for ts, _ in cum ]
            values = [ v for _, v in cum ]
        title = "Cumulative Realized PnL (derived from trades)"

    if times and values:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, values, label="Equity / Cumulative PnL", color="#2b8cbe")
        ax.set_title(title)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.autofmt_xdate()
        path = os.path.join(out_dir, "equity_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["equity_curve"] = path

        # drawdown plot
        series_pairs = list(zip([int(t.timestamp() * 1000) for t in times], values))
        dd_series, max_dd_abs, max_dd_pct = _compute_drawdown(list(zip([int(t.timestamp() * 1000) for t in times], values)))
        dd_times = [ _ts_to_dt(ts) for ts, _ in dd_series ]
        dd_vals = [ v for _, v in dd_series ]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(dd_times, dd_vals, color="#de2d26", alpha=0.4)
        ax.set_title(f"Drawdown (max {max_dd_abs:.2f}, {max_dd_pct:.2f}%)")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Drawdown")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.autofmt_xdate()
        path = os.path.join(out_dir, "drawdown.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["drawdown"] = path

    # 2) Trade PnL histogram
    pnls = [float(t.get("pnl", t.get("pnl_realized", 0.0))) for t in trades]
    if pnls:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pnls, bins=40, color="#31a354", alpha=0.9)
        ax.set_title("Trade PnL Distribution")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3)
        path = os.path.join(out_dir, "trade_pnl_hist.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["trade_pnl_hist"] = path

        # PnL over time scatter
        times = [ _ts_to_dt(int(t.get("exit_ts", t.get("exit_ts_ms", 0)))) for t in trades ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(times, pnls, c=["#3182bd" if p >= 0 else "#de2d26" for p in pnls], alpha=0.8)
        ax.set_title("Trade PnL over Time")
        ax.set_xlabel("Exit Time (UTC)")
        ax.set_ylabel("PnL")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.autofmt_xdate()
        path = os.path.join(out_dir, "trade_pnl_ts.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["trade_pnl_ts"] = path

    return plots


def save_report(
    results: Dict[str, Any],
    equity_series: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "reports",
) -> Dict[str, str]:
    """
    Create a report directory with:
      - summary.json (summary metrics)
      - plots (png files)
      - full_results.json (original results object)

    Returns dict with paths to generated artifacts.
    """
    _safe_mkdir(out_dir)
    summary = summarize_results(results)
    artifacts: Dict[str, str] = {}

    # Save full results
    results_path = os.path.join(out_dir, "full_results.json")
    with _LOCK:
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, default=str, indent=2)
    artifacts["full_results"] = results_path

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")
    with _LOCK:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, default=str, indent=2)
    artifacts["summary"] = summary_path

    # Generate plots
    try:
        plot_paths = generate_plots(results, equity_series=equity_series, out_dir=out_dir)
        artifacts.update(plot_paths)
    except ImportError:
        # matplotlib not installed; skip plots but do not fail
        artifacts["plots"] = "matplotlib not installed; skipped plots"

    return artifacts


# Example usage
if __name__ == "__main__":
    # Minimal demo: consume a results dict similar to engine.run_backtest
    demo_results = {
        "trades": [
            {"trade_id": "t1", "exit_ts": 1672531200000, "pnl": 100.0},
            {"trade_id": "t2", "exit_ts": 1672534800000, "pnl": -50.0},
            {"trade_id": "t3", "exit_ts": 1672538400000, "pnl": 200.0},
        ],
        "metrics": {
            "starting_equity": 100000.0,
            "latest_equity": 100250.0,
            "max_drawdown_abs": 50.0,
            "max_drawdown_pct": 0.05,
        },
    }
    out = save_report(demo_results, equity_series=None, out_dir="reports_demo")
    print("Generated artifacts:", out)

_LOCK = threading.RLock()


def _ts_to_dt(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)


def _safe_mkdir(path: str) -> None:
    with _LOCK:
        os.makedirs(path, exist_ok=True)


def _compute_cumulative_pnl_from_trades(trades: Sequence[Dict[str, Any]]) -> List[Tuple[int, float]]:
    """
    Returns list of (ts_ms, cumulative_pnl) ordered by ts ascending.
    Trades should contain exit_ts and pnl fields (pnl may be positive/negative).
    """
    sorted_trades = sorted((t for t in trades if t.get("exit_ts") is not None or t.get("exit_ts_ms") is not None),
                           key=lambda t: t.get("exit_ts", t.get("exit_ts_ms", 0)))
    entries: List[Tuple[int, float]] = []
    cum = 0.0
    for t in sorted_trades:
        ts = int(t.get("exit_ts", t.get("exit_ts_ms", t.get("exit_ts_ms", 0) or 0)))
        pnl = float(t.get("pnl", t.get("pnl_realized", 0.0)))
        cum += pnl
        entries.append((ts, cum))
    return entries


def _compute_drawdown(series: Sequence[Tuple[int, float]]) -> Tuple[List[Tuple[int, float]], float, float]:
    """
    Given series [(ts, value)], compute drawdown series [(ts, drawdown_abs)] and return max drawdown (abs, pct).
    Drawdown is peak - current_value. pct is (peak-current)/peak*100.
    """
    dd_series: List[Tuple[int, float]] = []
    peak = None
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    for ts, val in series:
        if peak is None or val > peak:
            peak = val
        dd = (peak - val) if peak is not None else 0.0
        dd_series.append((ts, dd))
        if dd > max_dd_abs:
            max_dd_abs = dd
            max_dd_pct = (dd / peak * 100.0) if peak and peak > 0 else 0.0
    return dd_series, max_dd_abs, max_dd_pct


def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute a compact summary from engine.run_backtest results.

    Expected input:
      {
        "trades": [ { "exit_ts": ms, "pnl": float, ... }, ... ],
        "metrics": { ... }  # optional aggregated metrics from MetricsCollector
      }

    Returns dict with keys:
      - total_trades, wins, losses, win_rate_pct
      - realized_pnl, avg_trade_pnl
      - max_drawdown_abs, max_drawdown_pct (if metrics contains them or derivable)
      - starting_equity, ending_equity (from metrics if present)
    """
    trades = results.get("trades", []) or []
    metrics = results.get("metrics", {}) or {}

    total = len(trades)
    wins = sum(1 for t in trades if float(t.get("pnl", t.get("pnl_realized", 0.0))) > 0)
    losses = sum(1 for t in trades if float(t.get("pnl", t.get("pnl_realized", 0.0))) <= 0)
    realized_pnl = sum(float(t.get("pnl", t.get("pnl_realized", 0.0))) for t in trades)
    avg_trade = (realized_pnl / total) if total > 0 else None
    win_rate = (wins / total * 100.0) if total > 0 else None

    # Prefer drawdown numbers from provided metrics if available
    max_dd_abs = metrics.get("max_drawdown_abs")
    max_dd_pct = metrics.get("max_drawdown_pct")
    starting_equity = metrics.get("starting_equity")
    latest_equity = metrics.get("latest_equity") or metrics.get("ending_equity")

    # If drawdown not provided, estimate from trades cumulative pnl
    if (max_dd_abs is None or max_dd_pct is None) and trades:
        cum = _compute_cumulative_pnl_from_trades(trades)
        if cum:
            _, max_dd_abs_est, max_dd_pct_est = _compute_drawdown(cum)
            max_dd_abs = max_dd_abs or max_dd_abs_est
            max_dd_pct = max_dd_pct or max_dd_pct_est

    summary = {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "realized_pnl": realized_pnl,
        "avg_trade_pnl": avg_trade,
        "starting_equity": starting_equity,
        "ending_equity": latest_equity,
        "max_drawdown_abs": max_dd_abs,
        "max_drawdown_pct": max_dd_pct,
    }
    return summary


def generate_plots(
    results: Dict[str, Any],
    equity_series: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "reports",
) -> Dict[str, str]:
    """
    Generate and persist plots for backtest results.

    Args:
      results: output of engine.run_backtest
      equity_series: optional list of {"ts": ms, "equity": float} to plot an equity curve.
                     If not provided, a cumulative realized-PnL curve derived from trades is used.
      out_dir: directory to write plots (created if missing)

    Returns:
      dict of plot name -> filepath
    """
    if plt is None:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")

    _safe_mkdir(out_dir)
    plots: Dict[str, str] = {}

    trades = results.get("trades", []) or []

    # 1) Equity curve
    if equity_series:
        series = sorted([(int(s["ts"]), float(s["equity"])) for s in equity_series], key=lambda x: x[0])
        times = [ _ts_to_dt(ts) for ts, _ in series ]
        values = [ val for _, val in series ]
        title = "Equity Curve (provided samples)"
    else:
        cum = _compute_cumulative_pnl_from_trades(trades)
        if not cum:
            # nothing to plot
            series = []
            times = []
            values = []
        else:
            times = [ _ts_to_dt(ts) for ts, _ in cum ]
            values = [ v for _, v in cum ]
        title = "Cumulative Realized PnL (derived from trades)"

    if times and values:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(times, values, label="Equity / Cumulative PnL", color="#2b8cbe")
        ax.set_title(title)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.autofmt_xdate()
        path = os.path.join(out_dir, "equity_curve.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["equity_curve"] = path

        # drawdown plot
        series_pairs = list(zip([int(t.timestamp() * 1000) for t in times], values))
        dd_series, max_dd_abs, max_dd_pct = _compute_drawdown(list(zip([int(t.timestamp() * 1000) for t in times], values)))
        dd_times = [ _ts_to_dt(ts) for ts, _ in dd_series ]
        dd_vals = [ v for _, v in dd_series ]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(dd_times, dd_vals, color="#de2d26", alpha=0.4)
        ax.set_title(f"Drawdown (max {max_dd_abs:.2f}, {max_dd_pct:.2f}%)")
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Drawdown")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.autofmt_xdate()
        path = os.path.join(out_dir, "drawdown.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["drawdown"] = path

    # 2) Trade PnL histogram
    pnls = [float(t.get("pnl", t.get("pnl_realized", 0.0))) for t in trades]
    if pnls:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(pnls, bins=40, color="#31a354", alpha=0.9)
        ax.set_title("Trade PnL Distribution")
        ax.set_xlabel("PnL")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle="--", alpha=0.3)
        path = os.path.join(out_dir, "trade_pnl_hist.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["trade_pnl_hist"] = path

        # PnL over time scatter
        times = [ _ts_to_dt(int(t.get("exit_ts", t.get("exit_ts_ms", 0)))) for t in trades ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(times, pnls, c=["#3182bd" if p >= 0 else "#de2d26" for p in pnls], alpha=0.8)
        ax.set_title("Trade PnL over Time")
        ax.set_xlabel("Exit Time (UTC)")
        ax.set_ylabel("PnL")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=timezone.utc))
        fig.autofmt_xdate()
        path = os.path.join(out_dir, "trade_pnl_ts.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        plots["trade_pnl_ts"] = path

    return plots


def save_report(
    results: Dict[str, Any],
    equity_series: Optional[List[Dict[str, Any]]] = None,
    out_dir: str = "reports",
) -> Dict[str, str]:
    """
    Create a report directory with:
      - summary.json (summary metrics)
      - plots (png files)
      - full_results.json (original results object)

    Returns dict with paths to generated artifacts.
    """
    _safe_mkdir(out_dir)
    summary = summarize_results(results)
    artifacts: Dict[str, str] = {}

    # Save full results
    results_path = os.path.join(out_dir, "full_results.json")
    with _LOCK:
        with open(results_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, default=str, indent=2)
    artifacts["full_results"] = results_path

    # Save summary
    summary_path = os.path.join(out_dir, "summary.json")
    with _LOCK:
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, default=str, indent=2)
    artifacts["summary"] = summary_path

    # Generate plots
    try:
        plot_paths = generate_plots(results, equity_series=equity_series, out_dir=out_dir)
        artifacts.update(plot_paths)
    except ImportError:
        # matplotlib not installed; skip plots but do not fail
        artifacts["plots"] = "matplotlib not installed; skipped plots"

    return artifacts


# Example usage
if __name__ == "__main__":
    # Minimal demo: consume a results dict similar to engine.run_backtest
    demo_results = {
        "trades": [
            {"trade_id": "t1", "exit_ts": 1672531200000, "pnl": 100.0},
            {"trade_id": "t2", "exit_ts": 1672534800000, "pnl": -50.0},
            {"trade_id": "t3", "exit_ts": 1672538400000, "pnl": 200.0},
        ],
        "metrics": {
            "starting_equity": 100000.0,
            "latest_equity": 100250.0,
            "max_drawdown_abs": 50.0,
            "max_drawdown_pct": 0.05,
        },
    }
    out = save_report(demo_results, equity_series=None, out_dir="reports_demo")
    print("Generated artifacts:", out)