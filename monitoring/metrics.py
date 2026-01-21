"""
metrics.py - Performance metrics collector for the trading platform.

Responsibilities:
- Track realized/unrealized PnL, win rate, drawdowns, trade counts.
- Maintain equity time-series for drawdown calculation.
- Produce daily summary reports (per-calendar-day).
- Thread-safe; no trading or execution logic included.

Design notes:
- The module records "closed" trades (realized PnL) via record_trade(...) and
  equity samples via record_equity(...). It does not decide when to close trades.
- All public methods are thread-safe. Async helpers are provided that run
  blocking work in the default thread pool (asyncio) to be safe for async code.
- Daily summaries are based on the UTC date of the trade/ equity timestamps.
"""

from __future__ import annotations

import asyncio
import copy
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from statistics import mean
from typing import Deque, Dict, List, Optional, Tuple, Any


@dataclass
class TradeRecord:
    """Lightweight representation of a closed trade (realized)."""
    trade_id: Optional[str]
    symbol: str
    entry_ts: int  # milliseconds since epoch (UTC)
    exit_ts: int   # milliseconds since epoch (UTC)
    pnl: float     # realized pnl (positive = profit)
    return_pct: Optional[float] = None  # percent return on trade (optional)
    size: Optional[float] = None
    side: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquitySample:
    ts: int       # milliseconds since epoch (UTC)
    equity: float


@dataclass
class DailySummary:
    day: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    avg_trade_pnl: Optional[float] = None
    win_rate: Optional[float] = None
    starting_equity: Optional[float] = None
    ending_equity: Optional[float] = None
    max_drawdown_abs: Optional[float] = None
    max_drawdown_pct: Optional[float] = None


class MetricsCollector:
    """
    Thread-safe metrics collector.

    Key methods:
      - record_trade(trade_record: TradeRecord)
      - record_equity(ts_ms: int, equity: float)
      - get_overall_metrics() -> dict
      - daily_summary(day: date) -> DailySummary

    Internal invariants:
      - trade records are immutable after recording (stored as dataclasses)
      - equity samples are stored in time-ordered deque (bounded by retention)
    """

    def __init__(self, equity_retention: int = 10_000):
        """
        Args:
            equity_retention: maximum number of equity samples to keep in memory.
        """
        self._lock = threading.RLock()
        self._trades: List[TradeRecord] = []
        self._equity: Deque[EquitySample] = deque(maxlen=equity_retention)

        # Derived quick-stats caches (recomputed on demand)
        self._cached_overall: Optional[Dict[str, Any]] = None
        self._cache_dirty = True

    # -----------------------
    # Recording methods
    # -----------------------
    def record_trade(
        self,
        trade_id: Optional[str],
        symbol: str,
        entry_ts_ms: int,
        exit_ts_ms: int,
        pnl: float,
        return_pct: Optional[float] = None,
        size: Optional[float] = None,
        side: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a realized (closed) trade.

        Args:
            trade_id: optional unique id for the trade
            symbol: trading instrument (e.g. "BTC/USDT")
            entry_ts_ms: entry timestamp in milliseconds (UTC)
            exit_ts_ms: exit timestamp in milliseconds (UTC)
            pnl: realized PnL (positive = profit)
            return_pct: optional percent return for this trade
            size: absolute position size
            side: "buy" or "sell" or None
            info: optional extra metadata (will be shallow-copied)
        """
        tr = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            entry_ts=entry_ts_ms,
            exit_ts=exit_ts_ms,
            pnl=float(pnl),
            return_pct=(None if return_pct is None else float(return_pct)),
            size=size,
            side=side,
            info=copy.copy(info or {}),
        )
        with self._lock:
            self._trades.append(tr)
            self._cache_dirty = True

    def record_equity(self, ts_ms: int, equity: float) -> None:
        """
        Record an equity/sample point.

        Args:
            ts_ms: timestamp in milliseconds (UTC)
            equity: account equity value at timestamp
        """
        sample = EquitySample(ts=ts_ms, equity=float(equity))
        with self._lock:
            # keep samples sorted by timestamp; simplest approach is append if in order,
            # otherwise insert to maintain order (rare).
            if not self._equity or ts_ms >= self._equity[-1].ts:
                self._equity.append(sample)
            else:
                # insert to keep ascending order (deque has no insert at arbitrary pos, so convert)
                tmp = list(self._equity)
                # find insertion point
                idx = 0
                while idx < len(tmp) and tmp[idx].ts < ts_ms:
                    idx += 1
                tmp.insert(idx, sample)
                self._equity = deque(tmp, maxlen=self._equity.maxlen)
            self._cache_dirty = True

    # -----------------------
    # Computation helpers
    # -----------------------
    def _compute_drawdowns(self, samples: List[EquitySample]) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute maximum drawdown (absolute and percentage) for provided equity samples.

        Returns:
            (max_drawdown_abs, max_drawdown_pct) where pct is in [0,100] or None if not computable.
        """
        if not samples:
            return None, None

        peak = samples[0].equity
        max_dd = 0.0
        max_dd_pct = 0.0
        for s in samples:
            if s.equity > peak:
                peak = s.equity
            dd = peak - s.equity
            dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        return max_dd, max_dd_pct

    def _group_trades_by_day(self) -> Dict[date, List[TradeRecord]]:
        """Group recorded trades by their exit date (UTC)."""
        groups: Dict[date, List[TradeRecord]] = defaultdict(list)
        with self._lock:
            for tr in self._trades:
                dt = datetime.fromtimestamp(tr.exit_ts / 1000.0, tz=timezone.utc).date()
                groups[dt].append(tr)
        return groups

    def _equity_range_for_day(self, day: date) -> List[EquitySample]:
        """Return equity samples whose timestamp's UTC date equals 'day'."""
        out: List[EquitySample] = []
        with self._lock:
            for s in self._equity:
                d = datetime.fromtimestamp(s.ts / 1000.0, tz=timezone.utc).date()
                if d == day:
                    out.append(s)
        return out

    # -----------------------
    # Public reporting
    # -----------------------
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Return dictionary with aggregated metrics across all recorded data:
            - total_trades, wins, losses, win_rate
            - realized_pnl, avg_trade_pnl
            - max_drawdown_abs, max_drawdown_pct
            - latest_equity, starting_equity (from samples)
        """
        with self._lock:
            if not self._cache_dirty and self._cached_overall is not None:
                return copy.deepcopy(self._cached_overall)

            trades = list(self._trades)
            total = len(trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            losses = sum(1 for t in trades if t.pnl <= 0)
            realized_pnl = sum(t.pnl for t in trades)
            avg_trade = (realized_pnl / total) if total > 0 else None
            win_rate = (wins / total * 100.0) if total > 0 else None

            samples = list(self._equity)
            latest_equity = samples[-1].equity if samples else None
            starting_equity = samples[0].equity if samples else None
            max_dd_abs, max_dd_pct = self._compute_drawdowns(samples)

            out = {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": win_rate,
                "realized_pnl": realized_pnl,
                "avg_trade_pnl": avg_trade,
                "starting_equity": starting_equity,
                "latest_equity": latest_equity,
                "max_drawdown_abs": max_dd_abs,
                "max_drawdown_pct": max_dd_pct,
            }
            self._cached_overall = copy.deepcopy(out)
            self._cache_dirty = False
            return out

    def daily_summary(self, day: date) -> DailySummary:
        """
        Produce a DailySummary for the provided UTC date.

        Aggregation rules:
          - Trades counted by exit timestamp localised to UTC date.
          - Equity starting/ending taken as first/last sample on that day if available.
          - Max drawdown computed from equity samples within the day.
        """
        trades_by_day = self._group_trades_by_day()
        trades = trades_by_day.get(day, [])

        samples = self._equity_range_for_day(day)
        starting_equity = samples[0].equity if samples else None
        ending_equity = samples[-1].equity if samples else None
        max_dd_abs, max_dd_pct = self._compute_drawdowns(samples)

        trades_count = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl <= 0)
        realized_pnl = sum(t.pnl for t in trades)
        avg_trade_pnl = (realized_pnl / trades_count) if trades_count > 0 else None
        win_rate = (wins / trades_count * 100.0) if trades_count > 0 else None

        return DailySummary(
            day=day,
            trades=trades_count,
            wins=wins,
            losses=losses,
            realized_pnl=realized_pnl,
            avg_trade_pnl=avg_trade_pnl,
            win_rate=win_rate,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            max_drawdown_abs=max_dd_abs,
            max_drawdown_pct=max_dd_pct,
        )

    # -----------------------
    # Utility methods
    # -----------------------
    def reset(self) -> None:
        """Clear all recorded trades and equity samples."""
        with self._lock:
            self._trades.clear()
            self._equity.clear()
            self._cache_dirty = True
            self._cached_overall = None

    def export_trades(self) -> List[Dict[str, Any]]:
        """Return a shallow-serializable list of recorded trades (safe copy)."""
        with self._lock:
            return [copy.deepcopy(t.__dict__) for t in self._trades]

    def export_equity(self) -> List[Dict[str, Any]]:
        """Return a shallow-serializable list of equity samples (safe copy)."""
        with self._lock:
            return [copy.deepcopy(s.__dict__) for s in self._equity]

    # -----------------------
    # Async helpers (convenience)
    # -----------------------
    async def async_record_trade(
        self,
        trade_id: Optional[str],
        symbol: str,
        entry_ts_ms: int,
        exit_ts_ms: int,
        pnl: float,
        return_pct: Optional[float] = None,
        size: Optional[float] = None,
        side: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.record_trade,
            trade_id,
            symbol,
            entry_ts_ms,
            exit_ts_ms,
            pnl,
            return_pct,
            size,
            side,
            info,
        )

    async def async_record_equity(self, ts_ms: int, equity: float) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.record_equity, ts_ms, equity)

    async def async_get_overall_metrics(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_overall_metrics)

    async def async_daily_summary(self, day: date) -> DailySummary:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.daily_summary, day)

    # -----------------------
    # Convenience helpers
    # -----------------------
    @staticmethod
    def now_ms() -> int:
        """Return current UTC timestamp in milliseconds."""
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


# Example usage
if __name__ == "__main__":
    import time as _time

    mc = MetricsCollector()

    # Simulate equity progression and trades across two days
    base = mc.now_ms()
    day0 = datetime.fromtimestamp(base / 1000.0, tz=timezone.utc).date()
    # record equity samples (monotonic increase then drop)
    mc.record_equity(base, 100_000.0)
    mc.record_equity(base + 60_000, 100_500.0)
    mc.record_equity(base + 120_000, 100_200.0)
    mc.record_trade("t1", "BTC/USDT", base, base + 60_000, pnl=500.0, return_pct=0.005, size=0.01, side="buy")
    mc.record_trade("t2", "ETH/USDT", base + 30_000, base + 120_000, pnl=-300.0, return_pct=-0.003, size=0.5, side="sell")

    # advance a day
    next_day_ms = base + 86_400_000
    mc.record_equity(next_day_ms, 100_100.0)
    mc.record_trade("t3", "BTC/USDT", next_day_ms, next_day_ms + 60_000, pnl=100.0, return_pct=0.001, size=0.005, side="buy")

    print("Overall metrics:", mc.get_overall_metrics())
    print("Day 0 summary:", mc.daily_summary(day0))
    print("Next day summary:", mc.daily_summary(datetime.fromtimestamp(next_day_ms / 1000.0, tz=timezone.utc).date()))
    symbol: str
    entry_ts: int  # milliseconds since epoch (UTC)
    exit_ts: int   # milliseconds since epoch (UTC)
    pnl: float     # realized pnl (positive = profit)
    return_pct: Optional[float] = None  # percent return on trade (optional)
    size: Optional[float] = None
    side: Optional[str] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EquitySample:
    ts: int       # milliseconds since epoch (UTC)
    equity: float


@dataclass
class DailySummary:
    day: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    avg_trade_pnl: Optional[float] = None
    win_rate: Optional[float] = None
    starting_equity: Optional[float] = None
    ending_equity: Optional[float] = None
    max_drawdown_abs: Optional[float] = None
    max_drawdown_pct: Optional[float] = None


class MetricsCollector:
    """
    Thread-safe metrics collector.

    Key methods:
      - record_trade(trade_record: TradeRecord)
      - record_equity(ts_ms: int, equity: float)
      - get_overall_metrics() -> dict
      - daily_summary(day: date) -> DailySummary

    Internal invariants:
      - trade records are immutable after recording (stored as dataclasses)
      - equity samples are stored in time-ordered deque (bounded by retention)
    """

    def __init__(self, equity_retention: int = 10_000):
        """
        Args:
            equity_retention: maximum number of equity samples to keep in memory.
        """
        self._lock = threading.RLock()
        self._trades: List[TradeRecord] = []
        self._equity: Deque[EquitySample] = deque(maxlen=equity_retention)

        # Derived quick-stats caches (recomputed on demand)
        self._cached_overall: Optional[Dict[str, Any]] = None
        self._cache_dirty = True

    # -----------------------
    # Recording methods
    # -----------------------
    def record_trade(
        self,
        trade_id: Optional[str],
        symbol: str,
        entry_ts_ms: int,
        exit_ts_ms: int,
        pnl: float,
        return_pct: Optional[float] = None,
        size: Optional[float] = None,
        side: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a realized (closed) trade.

        Args:
            trade_id: optional unique id for the trade
            symbol: trading instrument (e.g. "BTC/USDT")
            entry_ts_ms: entry timestamp in milliseconds (UTC)
            exit_ts_ms: exit timestamp in milliseconds (UTC)
            pnl: realized PnL (positive = profit)
            return_pct: optional percent return for this trade
            size: absolute position size
            side: "buy" or "sell" or None
            info: optional extra metadata (will be shallow-copied)
        """
        tr = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            entry_ts=entry_ts_ms,
            exit_ts=exit_ts_ms,
            pnl=float(pnl),
            return_pct=(None if return_pct is None else float(return_pct)),
            size=size,
            side=side,
            info=copy.copy(info or {}),
        )
        with self._lock:
            self._trades.append(tr)
            self._cache_dirty = True

    def record_equity(self, ts_ms: int, equity: float) -> None:
        """
        Record an equity/sample point.

        Args:
            ts_ms: timestamp in milliseconds (UTC)
            equity: account equity value at timestamp
        """
        sample = EquitySample(ts=ts_ms, equity=float(equity))
        with self._lock:
            # keep samples sorted by timestamp; simplest approach is append if in order,
            # otherwise insert to maintain order (rare).
            if not self._equity or ts_ms >= self._equity[-1].ts:
                self._equity.append(sample)
            else:
                # insert to keep ascending order (deque has no insert at arbitrary pos, so convert)
                tmp = list(self._equity)
                # find insertion point
                idx = 0
                while idx < len(tmp) and tmp[idx].ts < ts_ms:
                    idx += 1
                tmp.insert(idx, sample)
                self._equity = deque(tmp, maxlen=self._equity.maxlen)
            self._cache_dirty = True

    # -----------------------
    # Computation helpers
    # -----------------------
    def _compute_drawdowns(self, samples: List[EquitySample]) -> Tuple[Optional[float], Optional[float]]:
        """
        Compute maximum drawdown (absolute and percentage) for provided equity samples.

        Returns:
            (max_drawdown_abs, max_drawdown_pct) where pct is in [0,100] or None if not computable.
        """
        if not samples:
            return None, None

        peak = samples[0].equity
        max_dd = 0.0
        max_dd_pct = 0.0
        for s in samples:
            if s.equity > peak:
                peak = s.equity
            dd = peak - s.equity
            dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
        return max_dd, max_dd_pct

    def _group_trades_by_day(self) -> Dict[date, List[TradeRecord]]:
        """Group recorded trades by their exit date (UTC)."""
        groups: Dict[date, List[TradeRecord]] = defaultdict(list)
        with self._lock:
            for tr in self._trades:
                dt = datetime.fromtimestamp(tr.exit_ts / 1000.0, tz=timezone.utc).date()
                groups[dt].append(tr)
        return groups

    def _equity_range_for_day(self, day: date) -> List[EquitySample]:
        """Return equity samples whose timestamp's UTC date equals 'day'."""
        out: List[EquitySample] = []
        with self._lock:
            for s in self._equity:
                d = datetime.fromtimestamp(s.ts / 1000.0, tz=timezone.utc).date()
                if d == day:
                    out.append(s)
        return out

    # -----------------------
    # Public reporting
    # -----------------------
    def get_overall_metrics(self) -> Dict[str, Any]:
        """
        Return dictionary with aggregated metrics across all recorded data:
            - total_trades, wins, losses, win_rate
            - realized_pnl, avg_trade_pnl
            - max_drawdown_abs, max_drawdown_pct
            - latest_equity, starting_equity (from samples)
        """
        with self._lock:
            if not self._cache_dirty and self._cached_overall is not None:
                return copy.deepcopy(self._cached_overall)

            trades = list(self._trades)
            total = len(trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            losses = sum(1 for t in trades if t.pnl <= 0)
            realized_pnl = sum(t.pnl for t in trades)
            avg_trade = (realized_pnl / total) if total > 0 else None
            win_rate = (wins / total * 100.0) if total > 0 else None

            samples = list(self._equity)
            latest_equity = samples[-1].equity if samples else None
            starting_equity = samples[0].equity if samples else None
            max_dd_abs, max_dd_pct = self._compute_drawdowns(samples)

            out = {
                "total_trades": total,
                "wins": wins,
                "losses": losses,
                "win_rate_pct": win_rate,
                "realized_pnl": realized_pnl,
                "avg_trade_pnl": avg_trade,
                "starting_equity": starting_equity,
                "latest_equity": latest_equity,
                "max_drawdown_abs": max_dd_abs,
                "max_drawdown_pct": max_dd_pct,
            }
            self._cached_overall = copy.deepcopy(out)
            self._cache_dirty = False
            return out

    def daily_summary(self, day: date) -> DailySummary:
        """
        Produce a DailySummary for the provided UTC date.

        Aggregation rules:
          - Trades counted by exit timestamp localised to UTC date.
          - Equity starting/ending taken as first/last sample on that day if available.
          - Max drawdown computed from equity samples within the day.
        """
        trades_by_day = self._group_trades_by_day()
        trades = trades_by_day.get(day, [])

        samples = self._equity_range_for_day(day)
        starting_equity = samples[0].equity if samples else None
        ending_equity = samples[-1].equity if samples else None
        max_dd_abs, max_dd_pct = self._compute_drawdowns(samples)

        trades_count = len(trades)
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl <= 0)
        realized_pnl = sum(t.pnl for t in trades)
        avg_trade_pnl = (realized_pnl / trades_count) if trades_count > 0 else None
        win_rate = (wins / trades_count * 100.0) if trades_count > 0 else None

        return DailySummary(
            day=day,
            trades=trades_count,
            wins=wins,
            losses=losses,
            realized_pnl=realized_pnl,
            avg_trade_pnl=avg_trade_pnl,
            win_rate=win_rate,
            starting_equity=starting_equity,
            ending_equity=ending_equity,
            max_drawdown_abs=max_dd_abs,
            max_drawdown_pct=max_dd_pct,
        )

    # -----------------------
    # Utility methods
    # -----------------------
    def reset(self) -> None:
        """Clear all recorded trades and equity samples."""
        with self._lock:
            self._trades.clear()
            self._equity.clear()
            self._cache_dirty = True
            self._cached_overall = None

    def export_trades(self) -> List[Dict[str, Any]]:
        """Return a shallow-serializable list of recorded trades (safe copy)."""
        with self._lock:
            return [copy.deepcopy(t.__dict__) for t in self._trades]

    def export_equity(self) -> List[Dict[str, Any]]:
        """Return a shallow-serializable list of equity samples (safe copy)."""
        with self._lock:
            return [copy.deepcopy(s.__dict__) for s in self._equity]

    # -----------------------
    # Async helpers (convenience)
    # -----------------------
    async def async_record_trade(
        self,
        trade_id: Optional[str],
        symbol: str,
        entry_ts_ms: int,
        exit_ts_ms: int,
        pnl: float,
        return_pct: Optional[float] = None,
        size: Optional[float] = None,
        side: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            self.record_trade,
            trade_id,
            symbol,
            entry_ts_ms,
            exit_ts_ms,
            pnl,
            return_pct,
            size,
            side,
            info,
        )

    async def async_record_equity(self, ts_ms: int, equity: float) -> None:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.record_equity, ts_ms, equity)

    async def async_get_overall_metrics(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_overall_metrics)

    async def async_daily_summary(self, day: date) -> DailySummary:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.daily_summary, day)

    # -----------------------
    # Convenience helpers
    # -----------------------
    @staticmethod
    def now_ms() -> int:
        """Return current UTC timestamp in milliseconds."""
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


# Example usage
if __name__ == "__main__":
    import time as _time

    mc = MetricsCollector()

    # Simulate equity progression and trades across two days
    base = mc.now_ms()
    day0 = datetime.fromtimestamp(base / 1000.0, tz=timezone.utc).date()
    # record equity samples (monotonic increase then drop)
    mc.record_equity(base, 100_000.0)
    mc.record_equity(base + 60_000, 100_500.0)
    mc.record_equity(base + 120_000, 100_200.0)
    mc.record_trade("t1", "BTC/USDT", base, base + 60_000, pnl=500.0, return_pct=0.005, size=0.01, side="buy")
    mc.record_trade("t2", "ETH/USDT", base + 30_000, base + 120_000, pnl=-300.0, return_pct=-0.003, size=0.5, side="sell")

    # advance a day
    next_day_ms = base + 86_400_000
    mc.record_equity(next_day_ms, 100_100.0)
    mc.record_trade("t3", "BTC/USDT", next_day_ms, next_day_ms + 60_000, pnl=100.0, return_pct=0.001, size=0.005, side="buy")

    print("Overall metrics:", mc.get_overall_metrics())
    print("Day 0 summary:", mc.daily_summary(day0))
    print("Next day summary:", mc.daily_summary(datetime.fromtimestamp(next_day_ms / 1000.0, tz=timezone.utc).date()))