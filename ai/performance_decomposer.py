"""
performance_decomposer.py - Strategy Performance Decomposition

Breaks down trading performance by strategy, symbol, and market regime
to identify which strategies work best under which conditions.

Design:
- Non-invasive: collects data from trade records without modifying execution
- Per-symbol, per-strategy, per-regime decomposition
- Calculates expectancy, profit factor, drawdown contribution
- Backward compatible with existing metrics system
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

logger = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a single strategy."""
    strategy_name: str
    symbol: Optional[str] = None
    regime: Optional[str] = None
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    
    max_drawdown: float = 0.0
    consecutive_losses: int = 0
    max_consecutive_losses: int = 0
    
    expectancy: float = 0.0  # (Win% * AvgWin) - (Loss% * AvgLoss)
    profit_factor: float = 0.0  # GrossProfit / GrossLoss
    
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.winning_trades / self.total_trades) * 100.0
    
    @property
    def loss_rate(self) -> float:
        """Calculate loss rate percentage."""
        if self.total_trades == 0:
            return 0.0
        return (self.losing_trades / self.total_trades) * 100.0
    
    @property
    def avg_trade(self) -> float:
        """Average PnL per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades
    
    @property
    def sharpe_ratio(self) -> float:
        """Approximate Sharpe ratio (trades as returns)."""
        if self.total_trades < 2:
            return 0.0
        try:
            returns = []
            # Reconstruct from summary (approximation)
            for _ in range(self.winning_trades):
                returns.append(self.avg_win)
            for _ in range(self.losing_trades):
                returns.append(-self.avg_loss)
            
            if not returns or len(returns) < 2:
                return 0.0
            
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
            std_dev = variance ** 0.5
            
            if std_dev == 0:
                return 0.0
            
            return mean_return / std_dev
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'regime': self.regime,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'loss_rate': self.loss_rate,
            'total_pnl': self.total_pnl,
            'avg_trade': self.avg_trade,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'max_drawdown': self.max_drawdown,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'sharpe_ratio': self.sharpe_ratio,
            'last_updated': self.last_updated.isoformat(),
        }


class PerformanceDecomposer:
    """
    Decomposes trading performance across multiple dimensions.
    
    Tracks:
    - Per-strategy metrics
    - Per-symbol metrics
    - Per-regime metrics (TRENDING, RANGING, VOLATILE, etc.)
    - Combined decompositions (strategy x symbol, strategy x regime, etc.)
    """
    
    def __init__(self):
        """Initialize decomposer."""
        self._lock = __import__('threading').RLock()
        
        # Main tracking dictionaries
        self._performance_by_key: Dict[str, StrategyPerformance] = {}
        self._trade_history: List[Dict[str, Any]] = []
        
        # Configuration
        self._include_regime_decomposition = True
        self._include_symbol_decomposition = True
    
    def _make_key(self, strategy: str, symbol: Optional[str] = None, regime: Optional[str] = None) -> str:
        """Generate tracking key."""
        parts = [strategy]
        if symbol:
            parts.append(f"sym:{symbol}")
        if regime:
            parts.append(f"reg:{regime}")
        return "|".join(parts)
    
    def record_trade(
        self,
        strategy_name: str,
        symbol: str,
        pnl: float,
        entry_ts: int,
        exit_ts: int,
        regime: Optional[str] = None,
        size: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a completed trade.
        
        Args:
            strategy_name: Name of strategy that generated the trade
            symbol: Trading pair (e.g., 'BTC/USDT')
            pnl: Realized profit/loss
            entry_ts: Entry timestamp (ms since epoch)
            exit_ts: Exit timestamp (ms since epoch)
            regime: Market regime when trade was closed (optional)
            size: Trade size (optional)
            metadata: Additional metadata (optional)
        """
        with self._lock:
            trade_record = {
                'strategy': strategy_name,
                'symbol': symbol,
                'pnl': pnl,
                'entry_ts': entry_ts,
                'exit_ts': exit_ts,
                'regime': regime,
                'size': size,
                'metadata': metadata or {},
            }
            self._trade_history.append(trade_record)
            
            # Update performance across all relevant keys
            self._update_performance(trade_record)
            
            logger.debug(
                f"Recorded trade: strategy={strategy_name}, symbol={symbol}, "
                f"pnl={pnl:.2f}, regime={regime}"
            )
    
    def _update_performance(self, trade: Dict[str, Any]) -> None:
        """Update all relevant performance keys for a trade."""
        strategy = trade['strategy']
        symbol = trade['symbol']
        regime = trade.get('regime')
        pnl = trade['pnl']
        
        # Keys to update
        keys_to_update = [
            self._make_key(strategy),  # Overall strategy
            self._make_key(strategy, symbol),  # Strategy + symbol
        ]
        
        if regime and self._include_regime_decomposition:
            keys_to_update.append(self._make_key(strategy, regime=regime))  # Strategy + regime
            keys_to_update.append(self._make_key(strategy, symbol, regime))  # All three
        
        if self._include_symbol_decomposition and symbol:
            # Also track by symbol only (aggregate across strategies)
            keys_to_update.append(self._make_key("ALL_STRATEGIES", symbol))
        
        for key in keys_to_update:
            if key not in self._performance_by_key:
                # Extract metadata from key
                parts = key.split('|')
                strat = parts[0]
                sym = None
                reg = None
                for part in parts[1:]:
                    if part.startswith('sym:'):
                        sym = part[4:]
                    elif part.startswith('reg:'):
                        reg = part[4:]
                
                self._performance_by_key[key] = StrategyPerformance(
                    strategy_name=strat,
                    symbol=sym,
                    regime=reg,
                )
            
            perf = self._performance_by_key[key]
            perf.total_trades += 1
            perf.total_pnl += pnl
            
            if pnl > 0:
                perf.winning_trades += 1
                perf.avg_win = (perf.avg_win * (perf.winning_trades - 1) + pnl) / perf.winning_trades
                perf.consecutive_losses = 0
            else:
                perf.losing_trades += 1
                perf.avg_loss = (perf.avg_loss * (perf.losing_trades - 1) - pnl) / perf.losing_trades
                perf.consecutive_losses += 1
                perf.max_consecutive_losses = max(perf.max_consecutive_losses, perf.consecutive_losses)
            
            # Update expectancy and profit factor
            perf.expectancy = self._calculate_expectancy(perf)
            perf.profit_factor = self._calculate_profit_factor(perf)
            perf.last_updated = datetime.now(timezone.utc)
    
    def _calculate_expectancy(self, perf: StrategyPerformance) -> float:
        """Calculate expectancy: (Win% * AvgWin) - (Loss% * AvgLoss)."""
        if perf.total_trades == 0:
            return 0.0
        
        win_rate = perf.winning_trades / perf.total_trades
        loss_rate = perf.losing_trades / perf.total_trades
        
        return (win_rate * perf.avg_win) - (loss_rate * perf.avg_loss)
    
    def _calculate_profit_factor(self, perf: StrategyPerformance) -> float:
        """Calculate profit factor: GrossProfit / abs(GrossLoss)."""
        gross_profit = perf.avg_win * perf.winning_trades if perf.winning_trades > 0 else 0.0
        gross_loss = perf.avg_loss * perf.losing_trades if perf.losing_trades > 0 else 0.0
        
        if gross_loss == 0:
            return 0.0 if gross_profit == 0 else float('inf')
        
        return gross_profit / abs(gross_loss)
    
    def get_strategy_performance(
        self,
        strategy_name: str,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> Optional[StrategyPerformance]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_name: Name of strategy
            symbol: Optional symbol filter
            regime: Optional regime filter
            
        Returns:
            StrategyPerformance object or None if not found
        """
        with self._lock:
            key = self._make_key(strategy_name, symbol, regime)
            return self._performance_by_key.get(key)
    
    def get_all_performance(self) -> Dict[str, StrategyPerformance]:
        """Get all performance records."""
        with self._lock:
            return dict(self._performance_by_key)
    
    def get_top_strategies(self, metric: str = 'expectancy', top_n: int = 10) -> List[StrategyPerformance]:
        """
        Get top-performing strategies by a specific metric.
        
        Args:
            metric: 'expectancy', 'profit_factor', 'sharpe_ratio', 'win_rate', 'total_pnl'
            top_n: Number of top strategies to return
            
        Returns:
            List of StrategyPerformance objects sorted by metric (descending)
        """
        with self._lock:
            # Prefer overall strategy-level records (no symbol, no regime)
            overall = [p for p in self._performance_by_key.values() if p.symbol is None and p.regime is None]

            # Sort by requested metric
            if metric == 'expectancy':
                overall.sort(key=lambda p: p.expectancy, reverse=True)
            elif metric == 'profit_factor':
                overall.sort(key=lambda p: p.profit_factor, reverse=True)
            elif metric == 'sharpe_ratio':
                overall.sort(key=lambda p: p.sharpe_ratio, reverse=True)
            elif metric == 'win_rate':
                overall.sort(key=lambda p: p.win_rate, reverse=True)
            elif metric == 'total_pnl':
                overall.sort(key=lambda p: p.total_pnl, reverse=True)
            else:
                raise ValueError(f"Unknown metric: {metric}")

            return overall[:top_n]
    
    def get_regime_effectiveness(self, strategy_name: str) -> Dict[str, StrategyPerformance]:
        """
        Get strategy performance across all regimes.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Dict mapping regime -> StrategyPerformance
        """
        with self._lock:
            result = {}
            for key, perf in self._performance_by_key.items():
                if perf.strategy_name == strategy_name and perf.regime:
                    result[perf.regime] = perf
            return result
    
    def get_symbol_breakdown(self, strategy_name: str) -> Dict[str, StrategyPerformance]:
        """
        Get strategy performance across all symbols.
        
        Args:
            strategy_name: Name of strategy
            
        Returns:
            Dict mapping symbol -> StrategyPerformance
        """
        with self._lock:
            result = {}
            for key, perf in self._performance_by_key.items():
                if perf.strategy_name == strategy_name and perf.symbol and not perf.regime:
                    result[perf.symbol] = perf
            return result
    
    def get_trade_history(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get trade history with optional filters.
        
        Args:
            strategy_name: Filter by strategy
            symbol: Filter by symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of trade records
        """
        with self._lock:
            trades = self._trade_history
            
            if strategy_name:
                trades = [t for t in trades if t['strategy'] == strategy_name]
            if symbol:
                trades = [t for t in trades if t['symbol'] == symbol]
            
            if limit:
                trades = trades[-limit:]
            
            return list(trades)
    
    def get_regime_specific_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get aggregated metrics by regime across all strategies.
        
        NEW FOR PHASE 10: Enhanced regime analysis.
        
        Returns:
            Dict mapping regime -> {metric: value}
        """
        with self._lock:
            regime_metrics = defaultdict(lambda: {
                'total_trades': 0,
                'total_pnl': 0.0,
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_sharpe': 0.0,
                'avg_expectancy': 0.0,
                'max_dd': 0.0
            })
            
            for key, perf in self._performance_by_key.items():
                if perf.regime:
                    regime = perf.regime
                    regime_metrics[regime]['total_trades'] += perf.total_trades
                    regime_metrics[regime]['total_pnl'] += perf.total_pnl
                    regime_metrics[regime]['winning_trades'] += perf.winning_trades
                    regime_metrics[regime]['losing_trades'] += perf.losing_trades
                    regime_metrics[regime]['max_dd'] = max(
                        regime_metrics[regime]['max_dd'],
                        perf.max_drawdown
                    )
            
            # Calculate averages
            for regime in regime_metrics:
                if regime_metrics[regime]['total_trades'] > 0:
                    regime_metrics[regime]['win_rate'] = (
                        regime_metrics[regime]['winning_trades'] / 
                        regime_metrics[regime]['total_trades']
                    )
            
            return dict(regime_metrics)
    
    def get_drawdown_clustering(self) -> Dict[str, List[float]]:
        """
        Get drawdown clusters by regime.
        
        NEW FOR PHASE 10: Identify which regimes contribute most to drawdowns.
        
        Returns:
            Dict mapping regime -> [drawdown_values]
        """
        with self._lock:
            regime_drawdowns = defaultdict(list)
            
            for key, perf in self._performance_by_key.items():
                if perf.regime and perf.max_drawdown > 0:
                    regime_drawdowns[perf.regime].append(perf.max_drawdown)
            
            return dict(regime_drawdowns)
    
    def get_expectancy_persistence(self, strategy_name: str, lookback: int = 50) -> Dict[str, float]:
        """
        Calculate expectancy persistence over recent trades.
        
        NEW FOR PHASE 10: Detect if strategy edge is persisting or degrading.
        
        Args:
            strategy_name: Strategy to analyze
            lookback: Number of recent trades to analyze
            
        Returns:
            {
                'recent_expectancy': float,
                'historical_expectancy': float,
                'persistence_score': float  # 0-1, higher = more persistent
            }
        """
        with self._lock:
            # Get strategy trades
            trades = [t for t in self._trade_history if t['strategy'] == strategy_name]
            
            if len(trades) < lookback:
                return {
                    'recent_expectancy': 0.0,
                    'historical_expectancy': 0.0,
                    'persistence_score': 0.0
                }
            
            # Recent window
            recent_trades = trades[-lookback:]
            recent_pnls = [t['pnl'] for t in recent_trades]
            recent_exp = sum(recent_pnls) / len(recent_pnls) if recent_pnls else 0.0
            
            # Historical
            all_pnls = [t['pnl'] for t in trades]
            hist_exp = sum(all_pnls) / len(all_pnls) if all_pnls else 0.0
            
            # Persistence = how close recent is to historical
            if abs(hist_exp) > 0:
                persistence = 1.0 - abs(recent_exp - hist_exp) / abs(hist_exp)
                persistence = max(0.0, min(1.0, persistence))  # Clip to [0, 1]
            else:
                persistence = 0.5  # Neutral if no historical expectancy
            
            return {
                'recent_expectancy': recent_exp,
                'historical_expectancy': hist_exp,
                'persistence_score': persistence
            }
    
    def reset(self) -> None:
        """Reset all performance data."""
        with self._lock:
            self._performance_by_key.clear()
            self._trade_history.clear()
            logger.info("PerformanceDecomposer reset")


# Module-level convenience instance
_default_decomposer = PerformanceDecomposer()


def record_trade(
    strategy_name: str,
    symbol: str,
    pnl: float,
    entry_ts: int,
    exit_ts: int,
    regime: Optional[str] = None,
    size: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Convenience function using default decomposer."""
    _default_decomposer.record_trade(strategy_name, symbol, pnl, entry_ts, exit_ts, regime, size, metadata)


def get_strategy_performance(
    strategy_name: str,
    symbol: Optional[str] = None,
    regime: Optional[str] = None,
) -> Optional[StrategyPerformance]:
    """Convenience function using default decomposer."""
    return _default_decomposer.get_strategy_performance(strategy_name, symbol, regime)


def get_decomposer() -> PerformanceDecomposer:
    """Get the default decomposer instance."""
    return _default_decomposer
