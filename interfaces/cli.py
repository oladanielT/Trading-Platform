"""
CLI for managing strategies (backtest mode), viewing PnL and active trades,
and integrating with monitoring/logger.

Contracts:
- Start/stop strategy runs (runs backtests asynchronously as background tasks)
- Show current PnL and active trades for running jobs
- Interact with monitoring module (log events / trades / summaries)

Notes:
- This CLI runs backtests (replay) only — it does not connect to live/exchange.
- Strategy classes must follow the simple contract used by backtesting.engine:
    on_start(engine), on_bar(bar, engine), on_end(engine)
  and be importable as module.ClassName (e.g. mypkg.strats.mystrat.MyStrategy)
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtesting.engine import BacktestEngine, ExecutionConfig, BacktestError
from monitoring.logger import log_event, log_trade, configure as logger_config

# Simple in-memory registry for active background backtests
@dataclass
class RunInfo:
    name: str
    task: asyncio.Task
    engine: BacktestEngine
    started_at: float
    bars_path: Optional[str]
    strategy_repr: str
    result: Optional[Dict[str, Any]] = None
    cancelled: bool = False
    error: Optional[str] = None


_ACTIVE_RUNS: Dict[str, RunInfo] = {}


async def _load_bars_from_file(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bars file not found: {path}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    # Expect a list of bars
    if not isinstance(data, list):
        raise ValueError("bars file must contain a JSON list of bar dicts")
    return data


def _load_strategy(spec: str):
    """
    spec: "module.sub:ClassName" or "module.sub.ClassName"
    """
    if ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
    elif "." in spec and spec.count(".") >= 1:
        parts = spec.split(".")
        mod_name = ".".join(parts[:-1])
        cls_name = parts[-1]
    else:
        raise ValueError("strategy spec must be module.Class or module:Class")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls


async def _run_backtest_job(name: str, bars: List[Dict[str, Any]], strategy_obj: Any, exec_cfg: ExecutionConfig, bars_path: Optional[str]):
    """
    Background runner wrapper. Uses BacktestEngine to run and captures results,
    logging summary and trades to monitoring.logger.
    """
    engine = BacktestEngine(exec_cfg)
    started = time.time()
    log_event("job.start", {"job": name, "mode": "backtest", "bars_path": bool(bars_path), "strategy": str(type(strategy_obj))})
    try:
        # run in thread to avoid blocking event loop
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, engine.run_backtest, bars, strategy_obj)
        # capture results in registry
        ri = _ACTIVE_RUNS.get(name)
        if ri:
            ri.result = results
        # log trades and summary
        trades = results.get("trades", []) or []
        for t in trades:
            # keep logging lightweight; avoid leaking secrets
            log_trade({"trade_id": t.get("trade_id") or t.get("order_id"), "symbol": t.get("symbol"), "pnl": t.get("pnl", t.get("pnl_realized")), "size": t.get("size") or t.get("executed_size")})
        summary = results.get("metrics", {}) or {}
        log_event("job.completed", {"job": name, "duration_sec": round(time.time() - started, 3), "trades": len(trades), "summary": summary})
        return results
    except asyncio.CancelledError:
        ri = _ACTIVE_RUNS.get(name)
        if ri:
            ri.cancelled = True
        log_event("job.cancelled", {"job": name})
        raise
    except Exception as exc:
        ri = _ACTIVE_RUNS.get(name)
        if ri:
            ri.error = str(exc)
        log_event("job.error", {"job": name, "error": str(exc)})
        raise


def _ensure_logger_configured():
    # lightweight default configuration for CLI monitoring output
    logger_config(level="INFO", log_file=None)


# --------------------------
# CLI command implementations
# --------------------------
async def cmd_start(args: argparse.Namespace):
    _ensure_logger_configured()
    name = args.name
    if not name:
        print("A job name is required (--name).", file=sys.stderr)
        return 1
    if name in _ACTIVE_RUNS:
        print(f"Job '{name}' is already running.", file=sys.stderr)
        return 1

    # load bars
    if args.bars:
        try:
            bars = await _load_bars_from_file(args.bars)
        except Exception as exc:
            print(f"Failed to load bars: {exc}", file=sys.stderr)
            return 1
        bars_path = args.bars
    else:
        # If no bars provided, use a tiny synthetic dataset similar to engine demo
        from datetime import datetime, timezone
        base_ts = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        symbol = args.symbol or "BTC/USDT"
        prices = [42000 + i * 200 for i in range(10)]
        bars = []
        for i, p in enumerate(prices):
            bars.append({"symbol": symbol, "timestamp": base_ts + i * 60_000, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 1.0})
        bars_path = None

    # load strategy class
    try:
        cls = _load_strategy(args.strategy)
    except Exception as exc:
        print(f"Failed to import strategy '{args.strategy}': {exc}", file=sys.stderr)
        return 1

    try:
        strat_obj = cls()  # require no-arg constructor (simple contract)
    except Exception as exc:
        print(f"Failed to instantiate strategy '{args.strategy}': {exc}", file=sys.stderr)
        return 1

    exec_cfg = ExecutionConfig(execute_at=args.execute_at or "open", slippage_pct=float(args.slippage or 0.0), commission_pct=float(args.commission or 0.0), initial_cash=float(args.initial_cash or 100000.0))

    # create background task
    loop = asyncio.get_running_loop()
    task = loop.create_task(_run_backtest_job(name, bars, strat_obj, exec_cfg, bars_path))
    ri = RunInfo(name=name, task=task, engine=BacktestEngine(exec_cfg), started_at=time.time(), bars_path=bars_path, strategy_repr=f"{args.strategy}")
    _ACTIVE_RUNS[name] = ri

    def _on_done(t: asyncio.Task):
        try:
            res = t.result()
            ri.result = res
        except asyncio.CancelledError:
            ri.cancelled = True
        except Exception as exc:
            ri.error = str(exc)

    task.add_done_callback(_on_done)
    print(f"Started job '{name}' (background). Use 'status' to inspect or 'stop' to cancel.")
    return 0


async def cmd_status(args: argparse.Namespace):
    """
    Print a compact table of active runs and their current metrics.
    """
    _ensure_logger_configured()
    if not _ACTIVE_RUNS:
        print("No active jobs.")
        return 0

    out = []
    for name, ri in _ACTIVE_RUNS.items():
        status = "running" if not ri.task.done() else ("cancelled" if ri.cancelled else ("error" if ri.error else "completed"))
        duration = round(time.time() - ri.started_at, 1)
        # try to get metrics from result or engine
        metrics = None
        trades = []
        if ri.result:
            metrics = ri.result.get("metrics")
            trades = ri.result.get("trades", []) or []
        else:
            # if still running, attempt to grab engine metrics if available (engine used inside job; our RunInfo.engine is unused for now)
            try:
                metrics = ri.engine.get_metrics()
                trades = ri.engine.get_simulated_trades()
            except Exception:
                metrics = None
                trades = []
        out.append({"name": name, "status": status, "duration_s": duration, "trades": len(trades), "metrics": metrics or {}})

    print(json.dumps(out, default=str, indent=2))
    return 0


async def cmd_stop(args: argparse.Namespace):
    _ensure_logger_configured()
    name = args.name
    if not name:
        print("A job name is required (--name).", file=sys.stderr)
        return 1
    ri = _ACTIVE_RUNS.get(name)
    if not ri:
        print(f"No such job '{name}'.", file=sys.stderr)
        return 1
    if ri.task.done():
        print(f"Job '{name}' already finished.")
        return 0
    ri.task.cancel()
    try:
        await ri.task
    except asyncio.CancelledError:
        pass
    print(f"Job '{name}' cancelled.")
    return 0


async def cmd_show_results(args: argparse.Namespace):
    name = args.name
    if not name:
        print("A job name is required (--name).", file=sys.stderr)
        return 1
    ri = _ACTIVE_RUNS.get(name)
    if not ri:
        print(f"No such job '{name}'.", file=sys.stderr)
        return 1
    if ri.result:
        print(json.dumps(ri.result, default=str, indent=2))
        return 0
    else:
        if ri.task.done():
            if ri.error:
                print(f"Job finished with error: {ri.error}", file=sys.stderr)
                return 1
            print("Job finished but no result captured.")
            return 1
        else:
            print("Job is still running. Use 'status' to inspect.", file=sys.stderr)
            return 1


# --------------------------
# CLI entrypoint
# --------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tp-cli", description="Trading Platform CLI (backtest runner & monitoring)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="Start a strategy backtest in background")
    sp.add_argument("--name", required=True, help="Unique job name")
    sp.add_argument("--strategy", required=True, help="Strategy import path (module.Class or module:Class)")
    sp.add_argument("--bars", help="Path to JSON file with bars list (optional)")
    sp.add_argument("--symbol", help="Symbol to synthesize if no bars file provided")
    sp.add_argument("--execute-at", choices=("open", "close"), default="open")
    sp.add_argument("--slippage", type=float, default=0.0)
    sp.add_argument("--commission", type=float, default=0.0)
    sp.add_argument("--initial-cash", type=float, default=100000.0)

    sp2 = sub.add_parser("status", help="Show active jobs and basic metrics")
    # no args

    sp3 = sub.add_parser("stop", help="Stop a running job")
    sp3.add_argument("--name", required=True, help="Job name to stop")

    sp4 = sub.add_parser("results", help="Show full results for a finished job")
    sp4.add_argument("--name", required=True, help="Job name")

    return p


async def _main(argv: List[str]):
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = args.cmd
    if cmd == "start":
        return await cmd_start(args)
    if cmd == "status":
        return await cmd_status(args)
    if cmd == "stop":
        return await cmd_stop(args)
    if cmd == "results":
        return await cmd_show_results(args)
    print("Unknown command", file=sys.stderr)
    return 2


def main():
    try:
        return_code = asyncio.run(_main(sys.argv[1:]))
        sys.exit(return_code or 0)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
```# filepath: /home/oladan/Trading-Platform/interface/cli.py
"""
CLI for managing strategies (backtest mode), viewing PnL and active trades,
and integrating with monitoring/logger.

Contracts:
- Start/stop strategy runs (runs backtests asynchronously as background tasks)
- Show current PnL and active trades for running jobs
- Interact with monitoring module (log events / trades / summaries)

Notes:
- This CLI runs backtests (replay) only — it does not connect to live/exchange.
- Strategy classes must follow the simple contract used by backtesting.engine:
    on_start(engine), on_bar(bar, engine), on_end(engine)
  and be importable as module.ClassName (e.g. mypkg.strats.mystrat.MyStrategy)
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtesting.engine import BacktestEngine, ExecutionConfig, BacktestError
from monitoring.logger import log_event, log_trade, configure as logger_config

# Simple in-memory registry for active background backtests
@dataclass
class RunInfo:
    name: str
    task: asyncio.Task
    engine: BacktestEngine
    started_at: float
    bars_path: Optional[str]
    strategy_repr: str
    result: Optional[Dict[str, Any]] = None
    cancelled: bool = False
    error: Optional[str] = None


_ACTIVE_RUNS: Dict[str, RunInfo] = {}


async def _load_bars_from_file(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"bars file not found: {path}")
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    # Expect a list of bars
    if not isinstance(data, list):
        raise ValueError("bars file must contain a JSON list of bar dicts")
    return data


def _load_strategy(spec: str):
    """
    spec: "module.sub:ClassName" or "module.sub.ClassName"
    """
    if ":" in spec:
        mod_name, cls_name = spec.split(":", 1)
    elif "." in spec and spec.count(".") >= 1:
        parts = spec.split(".")
        mod_name = ".".join(parts[:-1])
        cls_name = parts[-1]
    else:
        raise ValueError("strategy spec must be module.Class or module:Class")
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    return cls


async def _run_backtest_job(name: str, bars: List[Dict[str, Any]], strategy_obj: Any, exec_cfg: ExecutionConfig, bars_path: Optional[str]):
    """
    Background runner wrapper. Uses BacktestEngine to run and captures results,
    logging summary and trades to monitoring.logger.
    """
    engine = BacktestEngine(exec_cfg)
    started = time.time()
    log_event("job.start", {"job": name, "mode": "backtest", "bars_path": bool(bars_path), "strategy": str(type(strategy_obj))})
    try:
        # run in thread to avoid blocking event loop
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, engine.run_backtest, bars, strategy_obj)
        # capture results in registry
        ri = _ACTIVE_RUNS.get(name)
        if ri:
            ri.result = results
        # log trades and summary
        trades = results.get("trades", []) or []
        for t in trades:
            # keep logging lightweight; avoid leaking secrets
            log_trade({"trade_id": t.get("trade_id") or t.get("order_id"), "symbol": t.get("symbol"), "pnl": t.get("pnl", t.get("pnl_realized")), "size": t.get("size") or t.get("executed_size")})
        summary = results.get("metrics", {}) or {}
        log_event("job.completed", {"job": name, "duration_sec": round(time.time() - started, 3), "trades": len(trades), "summary": summary})
        return results
    except asyncio.CancelledError:
        ri = _ACTIVE_RUNS.get(name)
        if ri:
            ri.cancelled = True
        log_event("job.cancelled", {"job": name})
        raise
    except Exception as exc:
        ri = _ACTIVE_RUNS.get(name)
        if ri:
            ri.error = str(exc)
        log_event("job.error", {"job": name, "error": str(exc)})
        raise


def _ensure_logger_configured():
    # lightweight default configuration for CLI monitoring output
    logger_config(level="INFO", log_file=None)


# --------------------------
# CLI command implementations
# --------------------------
async def cmd_start(args: argparse.Namespace):
    _ensure_logger_configured()
    name = args.name
    if not name:
        print("A job name is required (--name).", file=sys.stderr)
        return 1
    if name in _ACTIVE_RUNS:
        print(f"Job '{name}' is already running.", file=sys.stderr)
        return 1

    # load bars
    if args.bars:
        try:
            bars = await _load_bars_from_file(args.bars)
        except Exception as exc:
            print(f"Failed to load bars: {exc}", file=sys.stderr)
            return 1
        bars_path = args.bars
    else:
        # If no bars provided, use a tiny synthetic dataset similar to engine demo
        from datetime import datetime, timezone
        base_ts = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        symbol = args.symbol or "BTC/USDT"
        prices = [42000 + i * 200 for i in range(10)]
        bars = []
        for i, p in enumerate(prices):
            bars.append({"symbol": symbol, "timestamp": base_ts + i * 60_000, "open": p, "high": p * 1.01, "low": p * 0.99, "close": p, "volume": 1.0})
        bars_path = None

    # load strategy class
    try:
        cls = _load_strategy(args.strategy)
    except Exception as exc:
        print(f"Failed to import strategy '{args.strategy}': {exc}", file=sys.stderr)
        return 1

    try:
        strat_obj = cls()  # require no-arg constructor (simple contract)
    except Exception as exc:
        print(f"Failed to instantiate strategy '{args.strategy}': {exc}", file=sys.stderr)
        return 1

    exec_cfg = ExecutionConfig(execute_at=args.execute_at or "open", slippage_pct=float(args.slippage or 0.0), commission_pct=float(args.commission or 0.0), initial_cash=float(args.initial_cash or 100000.0))

    # create background task
    loop = asyncio.get_running_loop()
    task = loop.create_task(_run_backtest_job(name, bars, strat_obj, exec_cfg, bars_path))
    ri = RunInfo(name=name, task=task, engine=BacktestEngine(exec_cfg), started_at=time.time(), bars_path=bars_path, strategy_repr=f"{args.strategy}")
    _ACTIVE_RUNS[name] = ri

    def _on_done(t: asyncio.Task):
        try:
            res = t.result()
            ri.result = res
        except asyncio.CancelledError:
            ri.cancelled = True
        except Exception as exc:
            ri.error = str(exc)

    task.add_done_callback(_on_done)
    print(f"Started job '{name}' (background). Use 'status' to inspect or 'stop' to cancel.")
    return 0


async def cmd_status(args: argparse.Namespace):
    """
    Print a compact table of active runs and their current metrics.
    """
    _ensure_logger_configured()
    if not _ACTIVE_RUNS:
        print("No active jobs.")
        return 0

    out = []
    for name, ri in _ACTIVE_RUNS.items():
        status = "running" if not ri.task.done() else ("cancelled" if ri.cancelled else ("error" if ri.error else "completed"))
        duration = round(time.time() - ri.started_at, 1)
        # try to get metrics from result or engine
        metrics = None
        trades = []
        if ri.result:
            metrics = ri.result.get("metrics")
            trades = ri.result.get("trades", []) or []
        else:
            # if still running, attempt to grab engine metrics if available (engine used inside job; our RunInfo.engine is unused for now)
            try:
                metrics = ri.engine.get_metrics()
                trades = ri.engine.get_simulated_trades()
            except Exception:
                metrics = None
                trades = []
        out.append({"name": name, "status": status, "duration_s": duration, "trades": len(trades), "metrics": metrics or {}})

    print(json.dumps(out, default=str, indent=2))
    return 0


async def cmd_stop(args: argparse.Namespace):
    _ensure_logger_configured()
    name = args.name
    if not name:
        print("A job name is required (--name).", file=sys.stderr)
        return 1
    ri = _ACTIVE_RUNS.get(name)
    if not ri:
        print(f"No such job '{name}'.", file=sys.stderr)
        return 1
    if ri.task.done():
        print(f"Job '{name}' already finished.")
        return 0
    ri.task.cancel()
    try:
        await ri.task
    except asyncio.CancelledError:
        pass
    print(f"Job '{name}' cancelled.")
    return 0


async def cmd_show_results(args: argparse.Namespace):
    name = args.name
    if not name:
        print("A job name is required (--name).", file=sys.stderr)
        return 1
    ri = _ACTIVE_RUNS.get(name)
    if not ri:
        print(f"No such job '{name}'.", file=sys.stderr)
        return 1
    if ri.result:
        print(json.dumps(ri.result, default=str, indent=2))
        return 0
    else:
        if ri.task.done():
            if ri.error:
                print(f"Job finished with error: {ri.error}", file=sys.stderr)
                return 1
            print("Job finished but no result captured.")
            return 1
        else:
            print("Job is still running. Use 'status' to inspect.", file=sys.stderr)
            return 1


# --------------------------
# CLI entrypoint
# --------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tp-cli", description="Trading Platform CLI (backtest runner & monitoring)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("start", help="Start a strategy backtest in background")
    sp.add_argument("--name", required=True, help="Unique job name")
    sp.add_argument("--strategy", required=True, help="Strategy import path (module.Class or module:Class)")
    sp.add_argument("--bars", help="Path to JSON file with bars list (optional)")
    sp.add_argument("--symbol", help="Symbol to synthesize if no bars file provided")
    sp.add_argument("--execute-at", choices=("open", "close"), default="open")
    sp.add_argument("--slippage", type=float, default=0.0)
    sp.add_argument("--commission", type=float, default=0.0)
    sp.add_argument("--initial-cash", type=float, default=100000.0)

    sp2 = sub.add_parser("status", help="Show active jobs and basic metrics")
    # no args

    sp3 = sub.add_parser("stop", help="Stop a running job")
    sp3.add_argument("--name", required=True, help="Job name to stop")

    sp4 = sub.add_parser("results", help="Show full results for a finished job")
    sp4.add_argument("--name", required=True, help="Job name")

    return p


async def _main(argv: List[str]):
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = args.cmd
    if cmd == "start":
        return await cmd_start(args)
    if cmd == "status":
        return await cmd_status(args)
    if cmd == "stop":
        return await cmd_stop(args)
    if cmd == "results":
        return await cmd_show_results(args)
    print("Unknown command", file=sys.stderr)
    return 2


def main():
    try:
        return_code = asyncio.run(_main(sys.argv[1:]))
        sys.exit(return_code or 0)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()