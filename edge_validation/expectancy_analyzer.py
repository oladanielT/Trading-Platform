from typing import List, Dict, Any
from dataclasses import dataclass
import math

@dataclass
class ExpectancyMetrics:
    expectancy: float
    profit_factor: float
    sharpe: float
    sortino: float
    max_drawdown: float
    time_to_recovery: float
    trade_count: int


def compute_expectancy(trades: List[Dict[str, Any]], per_trade_returns: bool = True) -> ExpectancyMetrics:
    """
    Compute expectancy and related metrics from a list of trades.
    Each trade dict must include 'pnl' and 'exit_ts' (ms) optionally.
    """
    if not trades:
        return ExpectancyMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    pnls = [float(t.get('pnl', 0.0)) for t in trades]
    n = len(pnls)
    total = sum(pnls)
    expectancy = total / n if n > 0 else 0.0

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    mean = expectancy
    std = math.sqrt(sum((p - mean) ** 2 for p in pnls) / (n - 1)) if n > 1 else 0.0
    sharpe = (mean / std * math.sqrt(n)) if std > 0 else 0.0

    # Sortino: downside deviation
    downside = [p for p in pnls if p < 0]
    dd_std = math.sqrt(sum((p - 0) ** 2 for p in downside) / (len(downside) - 1)) if len(downside) > 1 else 0.0
    sortino = (mean / dd_std * math.sqrt(n)) if dd_std > 0 else 0.0

    # Equity curve and max drawdown
    cum = []
    running = 0.0
    for p in pnls:
        running += p
        cum.append(running)
    peak = -float('inf')
    max_dd = 0.0
    for v in cum:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    # Time to recovery: naive count of trades until cumulative returns exceed previous peak
    time_to_recover = 0.0
    if max_dd > 0:
        # find trough index and when recovered
        trough_idx = 0
        peak = -float('inf')
        for i, v in enumerate(cum):
            if v > peak:
                peak = v
            if peak - v == max_dd:
                trough_idx = i
                break
        recover_idx = None
        for j in range(trough_idx + 1, len(cum)):
            if cum[j] >= peak:
                recover_idx = j
                break
        if recover_idx is None:
            time_to_recover = float('inf')
        else:
            time_to_recover = recover_idx - trough_idx

    return ExpectancyMetrics(
        expectancy=expectancy,
        profit_factor=profit_factor,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_dd,
        time_to_recovery=time_to_recover,
        trade_count=n
    )
