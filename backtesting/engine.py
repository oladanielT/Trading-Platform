"""
Backtesting engine.

Responsibilities:
- Replay historical market data (OHLCV bars)
- Accept a strategy instance (strategy must implement on_start, on_bar, on_end as needed)
- Provide a minimal execution API (market orders) the strategy can call
- Return simulated trades (realized) and compute PnL, win rate, drawdowns (via MetricsCollector)
- No strategy logic is implemented here — engine only executes orders given by the strategy

Design notes / Strategy contract:
- Strategy is a plain object with optional methods:
    - on_start(engine): called once before replay starts
    - on_bar(bar: dict, engine): called for every bar. `bar` is normalized:
        { "symbol", "timestamp" (ms), "open", "high", "low", "close", "volume" }
    - on_end(engine): called once after replay ends
- To execute, strategy should call engine.place_market_order(symbol, size, side)
    - side: "buy" or "sell"
    - size: positive float (base asset units)
    - Orders are executed immediately at the engine's configured execution price for that bar (default: open)
- Engine records realized trades (only when position is reduced/closed) and equity samples.
- Engine supports simple commission and slippage models (percentage-based).
"""

from __future__ import annotations

import copy
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from monitoring.metrics import MetricsCollector, TradeRecord  # uses monitoring/metrics.py in project


@dataclass
class ExecutionConfig:
    """Configuration for execution / simulation fidelity."""
    execute_at: str = "open"        # "open" or "close" - price field from bar used for execution
    slippage_pct: float = 0.0       # slippage as fraction of price (e.g. 0.001 = 0.1%)
    commission_pct: float = 0.0     # commission as fraction of notional (e.g. 0.001)
    initial_cash: float = 100_000.0 # starting cash / equity
    record_equity_every_bar: bool = True  # whether to record equity sample each bar


class BacktestError(RuntimeError):
    pass


class BacktestEngine:
    """
    Simple, thread-safe backtesting engine.

    Public API:
      - run_backtest(bars: List[Dict], strategy) -> Dict[str, Any]
      - place_market_order(symbol, size, side) -> Dict (execution report)
      - get_simulated_trades() -> List[Dict]
      - get_metrics() -> Dict

    Note: This engine is intentionally conservative and simple. It does not
    implement order books, limit orders, advanced matching, margining or risk
    checks. It is intended for strategy replay and metric calculation only.
    """

    def __init__(self, exec_cfg: Optional[ExecutionConfig] = None):
        self.exec_cfg = exec_cfg or ExecutionConfig()
        self._lock = threading.RLock()

        # runtime state
        self._cash = float(self.exec_cfg.initial_cash)
        # positions: symbol -> {"size": float, "avg_price": float}
        self._positions: Dict[str, Dict[str, float]] = {}
        self._current_bar: Optional[Dict[str, Any]] = None

        # metrics recorder
        self._metrics = MetricsCollector()
        # realized trades also kept locally for return
        self._realized_trades: List[TradeRecord] = []

        # helper for lifecycle
        self._running = False

    # -------------------------
    # Helpers / bookkeeping
    # -------------------------
    @staticmethod
    def _now_ms() -> int:
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)

    def _exec_price_for_bar(self, bar: Dict[str, Any]) -> float:
        """Return the configured execution price for the provided bar."""
        if self.exec_cfg.execute_at == "open" and "open" in bar and bar["open"] is not None:
            return float(bar["open"])
        # fallback to close
        return float(bar.get("close", bar.get("open", 0.0)))

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply simple slippage (percentage) depending on side."""
        if not self.exec_cfg.slippage_pct:
            return price
        if side.lower() == "buy":
            return price * (1.0 + abs(self.exec_cfg.slippage_pct))
        else:
            return price * (1.0 - abs(self.exec_cfg.slippage_pct))

    def _apply_commission(self, notional: float) -> float:
        """Return commission amount for a trade notional."""
        return abs(notional) * abs(self.exec_cfg.commission_pct)

    def _position_value(self, symbol: str, market_price: float) -> float:
        pos = self._positions.get(symbol)
        if not pos:
            return 0.0
        return pos["size"] * market_price

    def _total_equity(self, market_prices: Optional[Dict[str, float]] = None) -> float:
        """Compute current equity = cash + mark-to-market value of positions."""
        total = self._cash
        if market_prices is None:
            # use last seen bar price for each position if available
            if self._current_bar:
                p = self._current_bar
                symbol = p.get("symbol")
                if symbol and symbol in self._positions:
                    total += self._position_value(symbol, float(p.get("close", p.get("open", 0.0))))
            # otherwise cannot compute; just return cash
            return total
        for sym, pos in self._positions.items():
            price = market_prices.get(sym)
            if price is None:
                continue
            total += pos["size"] * price
        return total

    # -------------------------
    # Execution API (for strategies)
    # -------------------------
    def place_market_order(self, symbol: str, size: float, side: str) -> Dict[str, Any]:
        """
        Place a market order (immediate execution within engine).

        Args:
            symbol: instrument symbol (must match bar["symbol"])
            size: positive base-asset size to buy/sell
            side: "buy" or "sell"

        Returns:
            execution report dict:
                {
                    "order_id": str,
                    "symbol": str,
                    "executed_size": float,
                    "executed_price": float,
                    "notional": float,
                    "commission": float,
                    "pnl_realized": float,  # >0 if profit realized by this execution
                    "cash_after": float,
                }
        """
        with self._lock:
            if not self._running:
                raise BacktestError("Engine not running; cannot place orders")

            if not self._current_bar:
                raise BacktestError("No current market data to execute against")

            if size <= 0:
                raise ValueError("size must be positive")

            exec_price = self._exec_price_for_bar(self._current_bar)
            exec_price = self._apply_slippage(exec_price, side)

            # determine direction: buy increases size, sell decreases
            dir_mult = 1.0 if side.lower() == "buy" else -1.0
            delta = dir_mult * float(size)

            # position before
            pos = self._positions.get(symbol, {"size": 0.0, "avg_price": 0.0})
            prev_size = pos["size"]
            prev_avg = pos["avg_price"]

            # compute notional and commission
            notional = exec_price * delta  # signed notional (positive for buy, negative for sell)
            commission = self._apply_commission(exec_price * abs(delta))

            # Cash impact:
            # For buy (delta>0): cash -= exec_price * delta + commission
            # For sell (delta<0): cash += abs(exec_price * delta) - commission
            # Use signed notional:
            self._cash -= notional + (commission if delta > 0 else -commission)  # careful sign
            # The above is equivalent to:
            # if delta>0: cash -= notional + commission
            # else: cash -= notional - commission (note notional negative), simplifies to cash += abs(notional) - commission

            realized_pnl = 0.0
            trade_record: Optional[TradeRecord] = None

            # If this trade reduces or flips position, compute realized pnl for the portion closed
            # Cases:
            # - same direction: increases position, update avg_price
            # - opposite direction but smaller: reduces position, realize pnl on closed size
            # - opposite direction and larger: close existing and open new position (realize pnl for old, set new avg)
            new_size = prev_size + delta

            if prev_size == 0.0:
                # opening new position
                new_avg = exec_price
            elif prev_size * new_size > 0:
                # same sign after trade -> increased/decreased but same direction
                # update average price for the net opened part (weighted avg)
                opened = delta
                if opened > 0:
                    # both positive
                    new_avg = (prev_avg * prev_size + exec_price * opened) / (prev_size + opened)
                else:
                    # both negative (short add)
                    new_avg = (prev_avg * abs(prev_size) + exec_price * abs(opened)) / (abs(prev_size) + abs(opened))
            else:
                # position reduced or flipped
                # closed_size = portion of prev position closed (cannot exceed abs(prev_size))
                closed_size = min(abs(delta), abs(prev_size))
                # pnl = (exit_price - entry_price) * closed_size * sign_of_prev_position
                side_sign = 1.0 if prev_size > 0 else -1.0
                realized_pnl = (exec_price - prev_avg) * closed_size * side_sign
                # subtract commission (already applied to cash)
                # record trade
                trade_record = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    symbol=symbol,
                    entry_ts=self._now_ms(),   # engine doesn't track original entry timestamp per-lot; use current time
                    exit_ts=self._current_bar.get("timestamp", self._now_ms()),
                    pnl=float(realized_pnl - commission),
                    return_pct=(realized_pnl / (abs(prev_avg) * closed_size) * 100.0) if (prev_avg and closed_size) else None,
                    size=closed_size,
                    side="sell" if prev_size > 0 else "buy",
                    info={"exec_price": exec_price, "commission": commission},
                )
                # if flip, calculate new avg for remaining/new position
                remaining = prev_size + delta  # could be negative (flipped)
                if abs(remaining) > 0:
                    # new avg for the net remaining portion uses the price of the side that remains;
                    # if we flipped, the new position's avg is the exec_price (simple model)
                    if prev_size * new_size < 0:
                        new_avg = exec_price
                    else:
                        new_avg = prev_avg  # should not come here
                else:
                    new_avg = 0.0

            # persist position update
            if abs(new_size) < 1e-12:
                # closed
                if symbol in self._positions:
                    del self._positions[symbol]
            else:
                self._positions[symbol] = {"size": new_size, "avg_price": float(new_avg)}

            # record realized trade if any
            if trade_record:
                self._metrics.record_trade(
                    trade_id=trade_record.trade_id,
                    symbol=trade_record.symbol,
                    entry_ts_ms=trade_record.entry_ts,
                    exit_ts_ms=trade_record.exit_ts,
                    pnl=trade_record.pnl,
                    return_pct=trade_record.return_pct,
                    size=trade_record.size,
                    side=trade_record.side,
                    info=trade_record.info,
                )
                self._realized_trades.append(trade_record)

            # optionally record equity sample at this bar
            if self.exec_cfg.record_equity_every_bar and self._current_bar:
                price_for_equity = float(self._current_bar.get("close", exec_price))
                equity = self._cash + sum(self._position_value(s, price_for_equity) for s in self._positions.keys())
                self._metrics.record_equity(self._current_bar.get("timestamp", self._now_ms()), equity)

            report = {
                "order_id": str(uuid.uuid4()),
                "symbol": symbol,
                "executed_size": delta,
                "executed_price": exec_price,
                "notional": exec_price * delta,
                "commission": commission,
                "pnl_realized": realized_pnl - commission if realized_pnl else 0.0,
                "cash_after": self._cash,
            }
            return report

    # -------------------------
    # Backtest runner
    # -------------------------
    def run_backtest(self, bars: List[Dict[str, Any]], strategy: Any) -> Dict[str, Any]:
        """
        Run the backtest.

        Args:
            bars: list of normalized bar dictionaries (ascending by timestamp).
                  Each bar should contain at least: symbol, timestamp (ms), open, high, low, close, volume
            strategy: strategy object (see module docstring for required methods)

        Returns:
            A dict containing:
                - "trades": list of realized trade dicts
                - "metrics": aggregated metrics (from MetricsCollector)
        """
        with self._lock:
            if self._running:
                raise BacktestError("Backtest already running")
            self._running = True

        try:
            # strategy lifecycle: on_start
            if hasattr(strategy, "on_start") and callable(strategy.on_start):
                strategy.on_start(self)

            # iterate bars
            for bar in bars:
                with self._lock:
                    self._current_bar = copy.deepcopy(bar)

                # record equity sample at bar start if desired
                if self.exec_cfg.record_equity_every_bar and self._current_bar:
                    price_for_equity = float(self._current_bar.get("close", self._exec_price_for_bar(self._current_bar)))
                    equity = self._cash + sum(self._position_value(s, price_for_equity) for s in self._positions.keys())
                    self._metrics.record_equity(self._current_bar.get("timestamp", self._now_ms()), equity)

                # call strategy
                if hasattr(strategy, "on_bar") and callable(strategy.on_bar):
                    # strategy can call engine.place_market_order(...)
                    strategy.on_bar(copy.deepcopy(self._current_bar), self)

            # strategy lifecycle: on_end
            if hasattr(strategy, "on_end") and callable(strategy.on_end):
                strategy.on_end(self)

            # final equity sample
            if self.exec_cfg.record_equity_every_bar:
                with self._lock:
                    final_ts = self._current_bar.get("timestamp", self._now_ms()) if self._current_bar else self._now_ms()
                    # use last close for mark-to-market
                    last_price = float(self._current_bar.get("close", 0.0)) if self._current_bar else 0.0
                    equity = self._cash + sum(self._position_value(s, last_price) for s in self._positions.keys())
                    self._metrics.record_equity(final_ts, equity)

            # collect output
            trades_out = [t.__dict__ for t in self._realized_trades]
            metrics_out = self._metrics.get_overall_metrics()
            return {"trades": trades_out, "metrics": metrics_out}
        finally:
            with self._lock:
                self._running = False
                self._current_bar = None

    # -------------------------
    # Accessors
    # -------------------------
    def get_simulated_trades(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [t.__dict__ for t in list(self._realized_trades)]

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics.get_overall_metrics()

    def reset(self) -> None:
        """Reset engine state (positions, cash, metrics)."""
        with self._lock:
            self._cash = float(self.exec_cfg.initial_cash)
            self._positions.clear()
            self._current_bar = None
            self._metrics.reset()
            self._realized_trades.clear()
            self._running = False

    # -------------------------
    # Example usage
    # -------------------------
if __name__ == "__main__":
    # Minimal example demonstrating engine usage with a toy strategy.
    class BuyAndHoldStrategy:
        def __init__(self, symbol: str, size: float):
            self.symbol = symbol
            self.size = size
            self._bought = False

        def on_start(self, engine: BacktestEngine):
            pass

        def on_bar(self, bar: Dict[str, Any], engine: BacktestEngine):
            # Buy once on first bar
            if not self._bought:
                engine.place_market_order(self.symbol, size=self.size, side="buy")
                self._bought = True

        def on_end(self, engine: BacktestEngine):
            pass

    # create toy bars
    bars = []
    symbol = "BTC/USDT"
    base_ts = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    prices = [42000, 42200, 41800, 43000, 44000]
    for i, p in enumerate(prices):
        bars.append({
            "symbol": symbol,
            "timestamp": base_ts + i * 60_000,
            "open": p,
            "high": p * 1.01,
            "low": p * 0.99,
            "close": p,
            "volume": 1.0,
        })

    engine = BacktestEngine(ExecutionConfig(execute_at="open", slippage_pct=0.0005, commission_pct=0.00075, initial_cash=100_000.0))
    strat = BuyAndHoldStrategy(symbol=symbol, size=0.1)
    results = engine.run_backtest(bars, strat)
    print("Simulated trades:", results["trades"])
    print("Metrics:", results["metrics"])