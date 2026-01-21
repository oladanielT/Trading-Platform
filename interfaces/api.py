"""
REST API (read-only) for platform metrics and run status.

Contract:
- Read-only endpoints only (GET).
- Expose safe info about backtest runs, metrics and recent logs.
- No control over strategies, no execution or AI decision endpoints.
- Avoid leaking sensitive data (api keys, secrets, tokens).
"""
from __future__ import annotations

import asyncio
import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Import run registry and logger manager from existing modules
# NOTE: these modules are part of the same project; keep usage read-only
from interface import cli as cli_module  # provides _ACTIVE_RUNS
from monitoring import logger as mon_logger  # provides get_recent, configure

app = FastAPI(title="Trading Platform Read API", version="0.1.0")

# keys to redact in any returned payloads
_SENSITIVE_KEYS = {k.lower() for k in ("api_key", "apiKey", "secret", "password", "private_key", "token")}

# simple asyncio lock to guard aggregated reads
_READ_LOCK = asyncio.Lock()


def _redact(obj: Any) -> Any:
    """Recursively redact sensitive keys from dict-like objects."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                out[k] = "<REDACTED>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _run_status(ri: cli_module.RunInfo) -> Dict[str, Any]:
    """Produce a compact, safe summary for a RunInfo entry."""
    status = "running" if not ri.task.done() else ("cancelled" if ri.cancelled else ("error" if ri.error else "completed"))
    duration = round((datetime.datetime.utcnow().timestamp() - ri.started_at), 1) if ri.started_at else None
    # prefer result metrics if available, else empty
    metrics = None
    trades_count = 0
    if ri.result:
        metrics = ri.result.get("metrics", {}) or {}
        trades_count = len((ri.result.get("trades") or []) or [])
    else:
        # try to access engine if possible (best-effort, safe)
        try:
            metrics = ri.engine.get_metrics()
            trades_count = len(ri.engine.get_simulated_trades())
        except Exception:
            metrics = {}
            trades_count = 0

    return _redact({
        "name": ri.name,
        "status": status,
        "started_at": datetime.datetime.utcfromtimestamp(ri.started_at).isoformat() + "Z" if ri.started_at else None,
        "duration_s": duration,
        "bars_path": ri.bars_path,
        "strategy": ri.strategy_repr,
        "trades_count": trades_count,
        "metrics": metrics,
        "error": ri.error,
        "cancelled": ri.cancelled,
    })


@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat() + "Z"}


@app.get("/runs")
async def list_runs():
    """List all known runs with compact safe metadata."""
    async with _READ_LOCK:
        runs = []
        for name, ri in list(cli_module._ACTIVE_RUNS.items()):
            runs.append(_run_status(ri))
    return JSONResponse(runs)


@app.get("/runs/{name}")
async def get_run(name: str):
    """Return detailed run info (safe)."""
    ri = cli_module._ACTIVE_RUNS.get(name)
    if not ri:
        raise HTTPException(status_code=404, detail="run not found")
    async with _READ_LOCK:
        result = {
            "meta": _run_status(ri),
            "result": _redact(ri.result) if ri.result else None,
        }
    return JSONResponse(result)


@app.get("/runs/{name}/trades")
async def get_run_trades(name: str):
    """Return simulated/realized trades for a run (if available)."""
    ri = cli_module._ACTIVE_RUNS.get(name)
    if not ri:
        raise HTTPException(status_code=404, detail="run not found")
    async with _READ_LOCK:
        trades = []
        if ri.result and ri.result.get("trades"):
            trades = ri.result.get("trades") or []
        else:
            # attempt to fetch from engine if still running
            try:
                trades = ri.engine.get_simulated_trades()
            except Exception:
                trades = []
    # redact and return
    return JSONResponse(_redact(trades))


@app.get("/trades/active")
async def active_trades():
    """Aggregate active (running) jobs' in-flight trades (best-effort)."""
    out = []
    async with _READ_LOCK:
        for ri in list(cli_module._ACTIVE_RUNS.values()):
            if ri.task and not ri.task.done():
                try:
                    trades = ri.engine.get_simulated_trades()
                except Exception:
                    trades = []
                if trades:
                    out.append({"job": ri.name, "trades": _redact(trades)})
    return JSONResponse(out)


@app.get("/logs/recent")
async def recent_logs(limit: int = Query(50, ge=1, le=500)):
    """Return recent structured logs from monitoring.logger (in-memory snapshot)."""
    # Ensure the default manager is configured (safe no-op if already configured)
    try:
        mon_logger.configure(level="INFO", log_file=None)
    except Exception:
        pass
    recent = mon_logger._default_manager.get_recent(limit)
    return JSONResponse(_redact(recent))


@app.get("/metrics/overall")
async def metrics_overall(run: Optional[str] = Query(None, description="Optional run name to scope metrics")):
    """
    Return overall metrics. If `run` provided, return that run's metrics (if available).
    Otherwise return aggregated metrics summary for all finished runs.
    """
    async with _READ_LOCK:
        if run:
            ri = cli_module._ACTIVE_RUNS.get(run)
            if not ri:
                raise HTTPException(status_code=404, detail="run not found")
            if ri.result:
                return JSONResponse(_redact(ri.result.get("metrics", {})))
            # if still running, try engine
            try:
                return JSONResponse(_redact(ri.engine.get_metrics()))
            except Exception:
                raise HTTPException(status_code=503, detail="metrics unavailable for running job")
        # aggregate finished runs metrics
        agg = {"runs": {}, "summary": {}}
        total_trades = 0
        total_pnl = 0.0
        runs_with_metrics = 0
        for name, ri in list(cli_module._ACTIVE_RUNS.items()):
            if ri.result and ri.result.get("metrics"):
                m = ri.result.get("metrics", {})
                agg["runs"][name] = _redact(m)
                runs_with_metrics += 1
                try:
                    total_trades += int(m.get("total_trades") or 0)
                    total_pnl += float(m.get("realized_pnl") or 0.0)
                except Exception:
                    pass
        agg["summary"] = {"runs_reported": runs_with_metrics, "total_trades": total_trades, "total_realized_pnl": total_pnl}
    return JSONResponse(agg)


# Prevent accidental execution control endpoints
@app.post("/{full_path:path}")
async def deny_all_post(full_path: str):
    raise HTTPException(status_code=405, detail="Write operations are not allowed via this API")


# Entrypoint for running the API server (development)
def run_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run("interface.api:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    run_server()
```# filepath: /home/oladan/Trading-Platform/interface/api.py
"""
REST API (read-only) for platform metrics and run status.

Contract:
- Read-only endpoints only (GET).
- Expose safe info about backtest runs, metrics and recent logs.
- No control over strategies, no execution or AI decision endpoints.
- Avoid leaking sensitive data (api keys, secrets, tokens).
"""
from __future__ import annotations

import asyncio
import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import uvicorn

# Import run registry and logger manager from existing modules
# NOTE: these modules are part of the same project; keep usage read-only
from interface import cli as cli_module  # provides _ACTIVE_RUNS
from monitoring import logger as mon_logger  # provides get_recent, configure

app = FastAPI(title="Trading Platform Read API", version="0.1.0")

# keys to redact in any returned payloads
_SENSITIVE_KEYS = {k.lower() for k in ("api_key", "apiKey", "secret", "password", "private_key", "token")}

# simple asyncio lock to guard aggregated reads
_READ_LOCK = asyncio.Lock()


def _redact(obj: Any) -> Any:
    """Recursively redact sensitive keys from dict-like objects."""
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                out[k] = "<REDACTED>"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    return obj


def _run_status(ri: cli_module.RunInfo) -> Dict[str, Any]:
    """Produce a compact, safe summary for a RunInfo entry."""
    status = "running" if not ri.task.done() else ("cancelled" if ri.cancelled else ("error" if ri.error else "completed"))
    duration = round((datetime.datetime.utcnow().timestamp() - ri.started_at), 1) if ri.started_at else None
    # prefer result metrics if available, else empty
    metrics = None
    trades_count = 0
    if ri.result:
        metrics = ri.result.get("metrics", {}) or {}
        trades_count = len((ri.result.get("trades") or []) or [])
    else:
        # try to access engine if possible (best-effort, safe)
        try:
            metrics = ri.engine.get_metrics()
            trades_count = len(ri.engine.get_simulated_trades())
        except Exception:
            metrics = {}
            trades_count = 0

    return _redact({
        "name": ri.name,
        "status": status,
        "started_at": datetime.datetime.utcfromtimestamp(ri.started_at).isoformat() + "Z" if ri.started_at else None,
        "duration_s": duration,
        "bars_path": ri.bars_path,
        "strategy": ri.strategy_repr,
        "trades_count": trades_count,
        "metrics": metrics,
        "error": ri.error,
        "cancelled": ri.cancelled,
    })


@app.get("/health")
async def health():
    return {"ok": True, "time": datetime.datetime.utcnow().isoformat() + "Z"}


@app.get("/runs")
async def list_runs():
    """List all known runs with compact safe metadata."""
    async with _READ_LOCK:
        runs = []
        for name, ri in list(cli_module._ACTIVE_RUNS.items()):
            runs.append(_run_status(ri))
    return JSONResponse(runs)


@app.get("/runs/{name}")
async def get_run(name: str):
    """Return detailed run info (safe)."""
    ri = cli_module._ACTIVE_RUNS.get(name)
    if not ri:
        raise HTTPException(status_code=404, detail="run not found")
    async with _READ_LOCK:
        result = {
            "meta": _run_status(ri),
            "result": _redact(ri.result) if ri.result else None,
        }
    return JSONResponse(result)


@app.get("/runs/{name}/trades")
async def get_run_trades(name: str):
    """Return simulated/realized trades for a run (if available)."""
    ri = cli_module._ACTIVE_RUNS.get(name)
    if not ri:
        raise HTTPException(status_code=404, detail="run not found")
    async with _READ_LOCK:
        trades = []
        if ri.result and ri.result.get("trades"):
            trades = ri.result.get("trades") or []
        else:
            # attempt to fetch from engine if still running
            try:
                trades = ri.engine.get_simulated_trades()
            except Exception:
                trades = []
    # redact and return
    return JSONResponse(_redact(trades))


@app.get("/trades/active")
async def active_trades():
    """Aggregate active (running) jobs' in-flight trades (best-effort)."""
    out = []
    async with _READ_LOCK:
        for ri in list(cli_module._ACTIVE_RUNS.values()):
            if ri.task and not ri.task.done():
                try:
                    trades = ri.engine.get_simulated_trades()
                except Exception:
                    trades = []
                if trades:
                    out.append({"job": ri.name, "trades": _redact(trades)})
    return JSONResponse(out)


@app.get("/logs/recent")
async def recent_logs(limit: int = Query(50, ge=1, le=500)):
    """Return recent structured logs from monitoring.logger (in-memory snapshot)."""
    # Ensure the default manager is configured (safe no-op if already configured)
    try:
        mon_logger.configure(level="INFO", log_file=None)
    except Exception:
        pass
    recent = mon_logger._default_manager.get_recent(limit)
    return JSONResponse(_redact(recent))


@app.get("/metrics/overall")
async def metrics_overall(run: Optional[str] = Query(None, description="Optional run name to scope metrics")):
    """
    Return overall metrics. If `run` provided, return that run's metrics (if available).
    Otherwise return aggregated metrics summary for all finished runs.
    """
    async with _READ_LOCK:
        if run:
            ri = cli_module._ACTIVE_RUNS.get(run)
            if not ri:
                raise HTTPException(status_code=404, detail="run not found")
            if ri.result:
                return JSONResponse(_redact(ri.result.get("metrics", {})))
            # if still running, try engine
            try:
                return JSONResponse(_redact(ri.engine.get_metrics()))
            except Exception:
                raise HTTPException(status_code=503, detail="metrics unavailable for running job")
        # aggregate finished runs metrics
        agg = {"runs": {}, "summary": {}}
        total_trades = 0
        total_pnl = 0.0
        runs_with_metrics = 0
        for name, ri in list(cli_module._ACTIVE_RUNS.items()):
            if ri.result and ri.result.get("metrics"):
                m = ri.result.get("metrics", {})
                agg["runs"][name] = _redact(m)
                runs_with_metrics += 1
                try:
                    total_trades += int(m.get("total_trades") or 0)
                    total_pnl += float(m.get("realized_pnl") or 0.0)
                except Exception:
                    pass
        agg["summary"] = {"runs_reported": runs_with_metrics, "total_trades": total_trades, "total_realized_pnl": total_pnl}
    return JSONResponse(agg)


# Prevent accidental execution control endpoints
@app.post("/{full_path:path}")
async def deny_all_post(full_path: str):
    raise HTTPException(status_code=405, detail="Write operations are not allowed via this API")


# Entrypoint for running the API server (development)
def run_server(host: str = "127.0.0.1", port: int = 8000):
    uvicorn.run("interface.api:app", host=host, port=port, reload=False, log_level="info")


if __name__ == "__main__":
    run_server()