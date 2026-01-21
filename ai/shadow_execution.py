"""
Shadow Execution Engine

Runs strategy signals without executing trades to evaluate performance
before live deployment.

Purpose:
- Generate strategy signals using live market data
- Track what trades WOULD have been made
- Compare against actual live trading decisions
- Measure divergence and theoretical performance

SAFETY: This module NEVER executes real trades.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
    """A trade that would have been executed (but wasn't)."""
    strategy_id: str
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    regime: Optional[str] = None
    trade_id: Optional[str] = None
    signal: Optional[Any] = None
    
    def close(self, exit_price: float, exit_time: datetime) -> float:
        """Close the shadow trade and calculate PnL."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        
        if self.side == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
        
        return self.pnl


@dataclass
class ShadowMetrics:
    """Performance metrics from shadow execution."""
    strategy_id: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_trade_pnl: float = 0.0
    correlation_with_live: float = 0.0  # Correlation with live decisions
    divergence_count: int = 0  # How many times shadow disagreed with live
    evaluation_period_days: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_pnl': self.total_pnl,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'avg_trade_pnl': self.avg_trade_pnl,
            'correlation_with_live': self.correlation_with_live,
            'divergence_count': self.divergence_count,
            'evaluation_period_days': self.evaluation_period_days
        }


class ShadowExecutor:
    """
    Shadow execution engine for strategy evaluation.
    
    CRITICAL: This engine NEVER executes real trades.
    - Generates signals only
    - Tracks theoretical performance
    - Compares with live trading
    - Provides promotion criteria data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize shadow executor.

        Accepts optional `config` dict for `strategy_promotion.shadow_execution`.
        Backwards-compatible constructor used by tests which pass a config.
        """
        cfg = config or {}
        # Nest path used in tests
        se_cfg = cfg.get('strategy_promotion', {}).get('shadow_execution', {}) if isinstance(cfg, dict) else {}

        # Behavior flags
        self.track_signals = bool(se_cfg.get('track_signals', True))
        self.compare_live = bool(se_cfg.get('compare_with_live', True))
        self.max_signal_divergence = float(se_cfg.get('max_signal_divergence', 0.15))

        self.shadow_trades: Dict[str, List[ShadowTrade]] = {}
        # open_positions maps trade_id -> ShadowTrade (allows multiple opens)
        self.open_positions: Dict[str, ShadowTrade] = {}
        # signal counts per strategy for total_signals metric
        self.signal_counts: Dict[str, int] = {}

        logger.info("[SHADOW] Initialized")
    
    def evaluate(
        self,
        strategy_id: str,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate strategy and generate shadow signal.
        
        Args:
            strategy_id: Strategy identifier
            strategy_func: Strategy function that returns signals
            market_data: Current market data (OHLCV, indicators)
            current_price: Current market price
            regime: Current market regime
            
        Returns:
            {
                'signal': 'long' | 'short' | 'close' | None,
                'size': float,
                'confidence': float,
                'metadata': dict
            }
        """
        """
        Backwards-compatible evaluate wrapper.

        Supports two calling styles:
        1. Modern: evaluate(strategy_id, strategy_func=callable, market_data=..., current_price=..., regime=...)
        2. Legacy tests: evaluate(strategy_id, signal, entry_price, timestamp)

        Legacy mode returns a `ShadowTrade` instance; modern mode returns the strategy signal dict.
        """
        # Legacy signature: (strategy_id, signal, entry_price, timestamp)
        if ('signal' in kwargs) or (len(args) >= 1 and not callable(args[0])):
            # Prefer kwargs if provided
            if 'signal' in kwargs:
                signal = kwargs.get('signal')
                entry_price = kwargs.get('entry_price')
                timestamp = kwargs.get('timestamp', datetime.now(timezone.utc))
            else:
                signal = args[0]
                entry_price = args[1] if len(args) > 1 else kwargs.get('entry_price')
                timestamp = args[2] if len(args) > 2 else kwargs.get('timestamp', datetime.now(timezone.utc))

            # Normalize signal into side/action
            side = 'long' if (signal == 1 or signal == 'long') else 'short'
            # Generate trade id
            trade_id = str(uuid.uuid4())
            trade = ShadowTrade(
                strategy_id=strategy_id,
                symbol='UNKNOWN',
                side=side,
                size=1.0,
                entry_price=float(entry_price),
                entry_time=timestamp if isinstance(timestamp, datetime) else datetime.now(timezone.utc),
                regime=kwargs.get('regime'),
                trade_id=trade_id
            )
            trade.signal = signal

            # Store open position
            self.open_positions[trade_id] = trade
            # Track signal count
            self.signal_counts[strategy_id] = self.signal_counts.get(strategy_id, 0) + 1

            return trade

        # Modern signature: expect callable in kwargs or args[0]
        try:
            strategy_func = kwargs.get('strategy_func') or (args[0] if args else None)
            market_data = kwargs.get('market_data') or (args[1] if len(args) > 1 else {})
            current_price = kwargs.get('current_price') or (args[2] if len(args) > 2 else None)
            regime = kwargs.get('regime') or (args[3] if len(args) > 3 else None)

            if not callable(strategy_func):
                return {'signal': None}

            signal = strategy_func(market_data)
            if signal is None:
                return {'signal': None}

            # Track open/close as before
            if signal.get('action') == 'open':
                self._open_shadow_position(
                    strategy_id=strategy_id,
                    symbol=signal.get('symbol', 'UNKNOWN'),
                    side=signal.get('side', 'long'),
                    size=signal.get('size', 0.0),
                    price=current_price,
                    regime=regime
                )
            elif signal.get('action') == 'close':
                self._close_shadow_position(
                    strategy_id=strategy_id,
                    price=current_price
                )

            return signal
        except Exception as e:
            logger.error(f"[SHADOW] Error evaluating {strategy_id}: {e}")
            return {'signal': None, 'error': str(e)}
    
    def _open_shadow_position(
        self,
        strategy_id: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        regime: Optional[str] = None
    ) -> None:
        """Open a shadow position (no real execution)."""
        # Close existing position if any
        if strategy_id in self.open_positions:
            self._close_shadow_position(strategy_id, price)
        
        # Create shadow trade
        trade = ShadowTrade(
            strategy_id=strategy_id,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=datetime.now(timezone.utc),
            regime=regime
        )
        
        self.open_positions[strategy_id] = trade
        
        logger.debug(
            f"[SHADOW] Opened {side} position: {strategy_id} "
            f"@ {price:.2f}, size={size:.4f}"
        )
    
    def _close_shadow_position(
        self,
        strategy_id: str,
        price: float,
        trade_id: Optional[str] = None,
    ) -> Optional[float]:
        """Close a shadow position and record PnL.

        If `trade_id` provided, close that specific trade; otherwise close any open trade for the strategy.
        """
        # Find trade by trade_id if given
        if trade_id:
            if trade_id not in self.open_positions:
                return None
            trade = self.open_positions.pop(trade_id)
        else:
            # Pop any open trade for the strategy (choose most recent)
            found_id = None
            found_trade = None
            for tid, t in list(self.open_positions.items()):
                if t.strategy_id == strategy_id:
                    found_id = tid
                    found_trade = t
            if found_id is None:
                return None
            trade = self.open_positions.pop(found_id)

        pnl = trade.close(price, datetime.now(timezone.utc))

        # Store completed trade
        if strategy_id not in self.shadow_trades:
            self.shadow_trades[strategy_id] = []
        self.shadow_trades[strategy_id].append(trade)

        logger.debug(
            f"[SHADOW] Closed position: {strategy_id} "
            f"@ {price:.2f}, PnL={pnl:.2f}"
        )

        return pnl
    
    def calculate_metrics(
        self,
        strategy_id: str,
        evaluation_days: int = 30
    ) -> ShadowMetrics:
        """
        Calculate performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy to evaluate
            evaluation_days: Number of days to analyze
            
        Returns:
            ShadowMetrics with calculated performance
        """
        # total_signals = all evaluate calls recorded
        total_signals = self.signal_counts.get(strategy_id, 0)

        if strategy_id not in self.shadow_trades:
            metrics_empty = ShadowMetrics(strategy_id=strategy_id, total_trades=0, evaluation_period_days=evaluation_days)
            setattr(metrics_empty, 'total_signals', total_signals)
            setattr(metrics_empty, 'avg_pnl', 0.0)
            return metrics_empty
        
        trades = self.shadow_trades[strategy_id]
        
        if not trades:
            return ShadowMetrics(strategy_id=strategy_id)
        
        # Calculate basic metrics
        total_trades = len(trades)
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        if not pnls:
            return ShadowMetrics(
                strategy_id=strategy_id,
                total_trades=total_trades
            )
        
        winning_trades = sum(1 for p in pnls if p > 0)
        losing_trades = sum(1 for p in pnls if p <= 0)
        total_pnl = sum(pnls)
        avg_trade_pnl = total_pnl / len(pnls) if pnls else 0.0
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        # Sharpe ratio
        import numpy as np
        sharpe = 0.0
        if len(pnls) > 1:
            std = np.std(pnls)
            if std > 0:
                sharpe = (avg_trade_pnl / std) * np.sqrt(252)
        
        # Max drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        metrics = ShadowMetrics(
            strategy_id=strategy_id,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_trade_pnl=avg_trade_pnl,
            evaluation_period_days=evaluation_days
        )
        # Attach legacy-friendly fields
        setattr(metrics, 'total_signals', total_signals)
        setattr(metrics, 'avg_pnl', metrics.avg_trade_pnl)

        return metrics
    
    def compare_with_live(
        self,
        strategy_id: str,
        live_signals_or_metrics: Any
    ) -> Dict[str, Any]:
        """
        Compare shadow signals with actual live trading decisions.
        
        Args:
            strategy_id: Strategy to compare
            live_signals: List of actual live trading signals
            
        Returns:
            {
                'correlation': float,
                'divergence_count': int,
                'agreement_rate': float
            }
        """
        # If no shadow trades, return neutral
        shadow_trades = self.shadow_trades.get(strategy_id, [])
        if not shadow_trades:
            return {'pnl_difference': 0.0, 'signal_difference': 0.0}

        shadow_total_pnl = sum((t.pnl or 0.0) for t in shadow_trades)

        # Accept either a metrics dict (legacy tests) or list of live signals
        if isinstance(live_signals_or_metrics, dict):
            live_metrics = live_signals_or_metrics
            live_pnl = float(live_metrics.get('total_pnl', 0.0))
            pnl_diff = 0.0 if live_pnl == 0 else (shadow_total_pnl - live_pnl) / max(abs(live_pnl), 1.0)
            signal_diff = (len(shadow_trades) - int(live_metrics.get('total_trades', 0)))
            return {'pnl_difference': pnl_diff, 'signal_difference': signal_diff}

        # Otherwise, try list-of-signals comparison (simple agreement rate)
        agreements = 0
        divergences = 0
        for shadow_trade in shadow_trades:
            matched = False
            for live_signal in live_signals_or_metrics:
                live_ts = live_signal.get('timestamp', shadow_trade.entry_time)
                try:
                    time_diff = abs((shadow_trade.entry_time - live_ts).total_seconds())
                except Exception:
                    time_diff = 0
                if time_diff < 300:
                    matched = True
                    if shadow_trade.side == live_signal.get('side'):
                        agreements += 1
                    else:
                        divergences += 1
                    break
            if not matched:
                divergences += 1

        total = agreements + divergences
        agreement_rate = agreements / total if total > 0 else 0.0
        return {'correlation': agreement_rate, 'divergence_count': divergences, 'agreement_rate': agreement_rate}
    
    def get_open_positions(self, strategy_id: Optional[str] = None) -> List[ShadowTrade]:
        """Get currently open shadow positions."""
        if strategy_id:
            return [self.open_positions.get(strategy_id)] if strategy_id in self.open_positions else []
        return list(self.open_positions.values())
    
    def get_closed_trades(self, strategy_id: str) -> List[ShadowTrade]:
        """Get all closed shadow trades for a strategy."""
        return self.shadow_trades.get(strategy_id, [])
    
    def reset(self, strategy_id: Optional[str] = None) -> None:
        """Reset shadow tracking (for testing or new evaluation period)."""
        if strategy_id:
            self.shadow_trades.pop(strategy_id, None)
            self.open_positions.pop(strategy_id, None)
        else:
            self.shadow_trades.clear()
            self.open_positions.clear()
        
        logger.info(f"[SHADOW] Reset: {strategy_id or 'all'}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        # tracked strategies should include any strategy with signals or trades
        tracked = set(self.shadow_trades.keys()) | set(self.signal_counts.keys())
        return {
            'tracked_strategies': list(tracked),
            'open_positions': len(self.open_positions),
            'total_shadow_trades': sum(len(trades) for trades in self.shadow_trades.values())
        }

    def close_trade(self, strategy_id: str, trade_id: str, exit_price: float, timestamp: Optional[datetime] = None) -> Optional[ShadowTrade]:
        """Close a trade by `trade_id` (legacy test API).

        Returns the closed ShadowTrade or None if not found.
        """
        ts = timestamp or datetime.now(timezone.utc)
        pnl = self._close_shadow_position(strategy_id=strategy_id, price=exit_price, trade_id=trade_id)
        # Return the trade object (search in closed trades)
        closed = None
        for t in self.shadow_trades.get(strategy_id, []):
            if t.trade_id == trade_id:
                closed = t
                break
        return closed
