from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_metrics: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    trades_train: List[Dict[str, Any]] = field(default_factory=list)
    trades_test: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    windows: List[WindowResult]
    aggregate_metrics: Dict[str, Any] = field(default_factory=dict)


class WalkForwardEngine:
    """
    Simple walk-forward engine that splits trade history into rolling
    train/test windows based on timestamps (milliseconds since epoch).

    Inputs are lists of trade dicts with 'entry_ts' or 'exit_ts' in ms.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def _to_ts(self, t):
        # accept int timestamps or datetime
        if isinstance(t, datetime):
            return int(t.timestamp() * 1000)
        return int(t)

    def run_walk_forward(
        self,
        trades: List[Dict[str, Any]],
        train_window_days: int = 30,
        test_window_days: int = 7,
        step_days: int = 7,
    ) -> WalkForwardResult:
        if not trades:
            return WalkForwardResult(windows=[], aggregate_metrics={})

        # Ensure trades sorted by exit_ts (or entry)
        trades_sorted = sorted(trades, key=lambda t: t.get('exit_ts', t.get('entry_ts', 0)))
        first_ts = trades_sorted[0].get('entry_ts') or trades_sorted[0].get('exit_ts')
        first_dt = datetime.utcfromtimestamp(first_ts / 1000.0)

        windows: List[WindowResult] = []

        cursor = first_dt
        last_trade_ts = trades_sorted[-1].get('exit_ts') or trades_sorted[-1].get('entry_ts')
        last_dt = datetime.utcfromtimestamp(last_trade_ts / 1000.0)

        while cursor + timedelta(days=train_window_days + test_window_days) <= last_dt + timedelta(days=1):
            train_start_dt = cursor
            train_end_dt = cursor + timedelta(days=train_window_days) - timedelta(milliseconds=1)
            test_start_dt = train_end_dt + timedelta(milliseconds=1)
            test_end_dt = test_start_dt + timedelta(days=test_window_days) - timedelta(milliseconds=1)

            train_start = int(train_start_dt.timestamp() * 1000)
            train_end = int(train_end_dt.timestamp() * 1000)
            test_start = int(test_start_dt.timestamp() * 1000)
            test_end = int(test_end_dt.timestamp() * 1000)

            trades_train = [t for t in trades_sorted if (t.get('entry_ts') or t.get('exit_ts')) and (t.get('entry_ts', t.get('exit_ts')) >= train_start and t.get('entry_ts', t.get('exit_ts')) <= train_end)]
            trades_test = [t for t in trades_sorted if (t.get('entry_ts') or t.get('exit_ts')) and (t.get('entry_ts', t.get('exit_ts')) >= test_start and t.get('entry_ts', t.get('exit_ts')) <= test_end)]

            window = WindowResult(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                trades_train=trades_train,
                trades_test=trades_test,
            )
            windows.append(window)

            cursor = cursor + timedelta(days=step_days)

        # Aggregate metrics across windows (placeholder sums/averages)
        agg = {
            'expectancy': None,
            'sharpe': None,
            'max_drawdown': None,
            'win_rate': None,
            'trade_count': sum(len(w.trades_test) for w in windows)
        }

        return WalkForwardResult(windows=windows, aggregate_metrics=agg)
