"""
Walk-Forward Validation Engine

Implements rolling in-sample / out-of-sample splits with:
- No lookahead bias
- Multiple market regime coverage
- Performance tracking across time windows

SAFETY: This module is OFFLINE ONLY - no live trading integration.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class WindowConfig:
    """Configuration for walk-forward window splits."""
    train_pct: float = 0.6
    test_pct: float = 0.4
    step_size: float = 0.1  # Rolling window step
    min_train_samples: int = 100
    min_test_samples: int = 30


@dataclass
class WindowResult:
    """Results from a single walk-forward window."""
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    in_sample_trades: int
    out_sample_trades: int
    in_sample_sharpe: float
    out_sample_sharpe: float
    in_sample_expectancy: float
    out_sample_expectancy: float
    in_sample_max_dd: float
    out_sample_max_dd: float
    regime_coverage: Dict[str, int]
    overfitting_score: float  # Higher = more overfit


class WalkForwardEngine:
    """
    Walk-forward validation engine for strategy robustness testing.
    
    CRITICAL: This is an OFFLINE research tool. It does NOT interact with:
    - Live trading systems
    - Order execution
    - Position sizing
    - Runtime optimization
    
    Purpose: Validate strategy edge before human promotes to live testing.
    """
    
    def __init__(self, window_config: Optional[WindowConfig] = None):
        """
        Initialize walk-forward engine.
        
        Args:
            window_config: Window split configuration
        """
        self.config = window_config or WindowConfig()
        self.results: List[WindowResult] = []
        
    def run(
        self,
        strategy_func,
        data: Dict[str, Any],
        regime_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation on a strategy.
        
        Args:
            strategy_func: Function that takes data and returns trades
            data: Historical market data (OHLCV + indicators)
            regime_labels: Optional regime labels per timestamp
            
        Returns:
            {
                'windows': List[WindowResult],
                'overall_consistency': float,  # 0-1
                'regime_consistency': Dict[str, float],
                'oos_degradation': float,  # Negative = worse OOS
                'passed': bool
            }
        """
        if not data or 'close' not in data:
            raise ValueError("Data must contain 'close' prices")
            
        n_samples = len(data['close'])
        
        if n_samples < self.config.min_train_samples + self.config.min_test_samples:
            raise ValueError(
                f"Insufficient data: {n_samples} samples, "
                f"need {self.config.min_train_samples + self.config.min_test_samples}"
            )
        
        # Calculate window sizes
        train_size = int(n_samples * self.config.train_pct)
        test_size = int(n_samples * self.config.test_pct)
        step = int(n_samples * self.config.step_size)
        
        # Enforce minimums
        train_size = max(train_size, self.config.min_train_samples)
        test_size = max(test_size, self.config.min_test_samples)
        step = max(step, 10)
        
        window_results = []
        window_id = 0
        
        # Rolling windows
        start = 0
        while start + train_size + test_size <= n_samples:
            train_end = start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            # Extract windows (NO LOOKAHEAD)
            train_data = self._slice_data(data, start, train_end)
            test_data = self._slice_data(data, test_start, test_end)
            
            train_regimes = None
            test_regimes = None
            if regime_labels:
                train_regimes = regime_labels[start:train_end]
                test_regimes = regime_labels[test_start:test_end]
            
            # Run strategy on both windows
            train_trades = strategy_func(train_data)
            test_trades = strategy_func(test_data)
            
            # Analyze performance
            train_metrics = self._calculate_metrics(train_trades, train_regimes)
            test_metrics = self._calculate_metrics(test_trades, test_regimes)
            
            # Detect overfitting
            overfitting_score = self._calculate_overfitting(train_metrics, test_metrics)
            
            window_result = WindowResult(
                window_id=window_id,
                train_start=start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                in_sample_trades=len(train_trades),
                out_sample_trades=len(test_trades),
                in_sample_sharpe=train_metrics['sharpe'],
                out_sample_sharpe=test_metrics['sharpe'],
                in_sample_expectancy=train_metrics['expectancy'],
                out_sample_expectancy=test_metrics['expectancy'],
                in_sample_max_dd=train_metrics['max_dd'],
                out_sample_max_dd=test_metrics['max_dd'],
                regime_coverage=test_metrics['regime_counts'],
                overfitting_score=overfitting_score
            )
            
            window_results.append(window_result)
            window_id += 1
            
            # Step forward
            start += step
        
        # Aggregate results
        self.results = window_results
        return self._aggregate_results(window_results, regime_labels)
    
    def _slice_data(self, data: Dict[str, Any], start: int, end: int) -> Dict[str, Any]:
        """Extract time slice from data without lookahead."""
        sliced = {}
        for key, values in data.items():
            if isinstance(values, (list, np.ndarray)):
                sliced[key] = values[start:end]
            else:
                sliced[key] = values  # Scalar configs
        return sliced
    
    def _calculate_metrics(
        self,
        trades: List[Dict],
        regime_labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculate performance metrics for a set of trades."""
        if not trades:
            return {
                'sharpe': 0.0,
                'expectancy': 0.0,
                'max_dd': 0.0,
                'win_rate': 0.0,
                'regime_counts': {}
            }
        
        pnls = [t.get('pnl', 0.0) for t in trades]
        
        # Sharpe
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Expectancy
        expectancy = np.mean(pnls) if pnls else 0.0
        
        # Max Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win Rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        
        # Regime Coverage
        regime_counts = {}
        if regime_labels:
            for regime in regime_labels:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        return {
            'sharpe': sharpe,
            'expectancy': expectancy,
            'max_dd': max_dd,
            'win_rate': win_rate,
            'regime_counts': regime_counts
        }
    
    def _calculate_overfitting(
        self,
        train_metrics: Dict,
        test_metrics: Dict
    ) -> float:
        """
        Calculate overfitting score.
        
        Returns:
            Score from 0 (no overfit) to 1 (severe overfit)
        """
        # Sharpe degradation
        sharpe_diff = train_metrics['sharpe'] - test_metrics['sharpe']
        sharpe_penalty = max(0, sharpe_diff / max(abs(train_metrics['sharpe']), 1.0))
        
        # Expectancy degradation
        exp_diff = train_metrics['expectancy'] - test_metrics['expectancy']
        exp_penalty = max(0, exp_diff / max(abs(train_metrics['expectancy']), 0.01))
        
        # Win rate collapse
        wr_diff = train_metrics['win_rate'] - test_metrics['win_rate']
        wr_penalty = max(0, wr_diff)
        
        # Combined score (0-1 range)
        overfit_score = np.clip((sharpe_penalty + exp_penalty + wr_penalty) / 3.0, 0, 1)
        
        return overfit_score
    
    def _aggregate_results(
        self,
        window_results: List[WindowResult],
        regime_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Aggregate walk-forward window results."""
        if not window_results:
            return {
                'windows': [],
                'overall_consistency': 0.0,
                'regime_consistency': {},
                'oos_degradation': -1.0,
                'passed': False
            }
        
        # Calculate consistency (OOS Sharpe std dev)
        oos_sharpes = [w.out_sample_sharpe for w in window_results]
        consistency = 1.0 - min(np.std(oos_sharpes), 1.0) if len(oos_sharpes) > 1 else 0.0
        
        # OOS degradation (average)
        degradations = [
            w.in_sample_sharpe - w.out_sample_sharpe
            for w in window_results
        ]
        avg_degradation = np.mean(degradations)
        
        # Regime-specific consistency
        regime_consistency = {}
        if regime_labels:
            unique_regimes = set(regime_labels)
            for regime in unique_regimes:
                # Count windows covering this regime
                regime_windows = [
                    w for w in window_results
                    if regime in w.regime_coverage
                ]
                if regime_windows:
                    regime_sharpes = [w.out_sample_sharpe for w in regime_windows]
                    regime_consistency[regime] = 1.0 - min(np.std(regime_sharpes), 1.0)
        
        # Pass criteria (OOS consistency >= 70%)
        passed = consistency >= 0.70 and avg_degradation < 1.0
        
        return {
            'windows': window_results,
            'overall_consistency': consistency,
            'regime_consistency': regime_consistency,
            'oos_degradation': avg_degradation,
            'avg_oos_sharpe': np.mean(oos_sharpes),
            'min_oos_sharpe': np.min(oos_sharpes),
            'max_overfitting_score': max(w.overfitting_score for w in window_results),
            'passed': passed
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about walk-forward runs."""
        if not self.results:
            return {'status': 'no_runs', 'windows': 0}
        
        return {
            'status': 'completed',
            'windows': len(self.results),
            'avg_train_trades': np.mean([w.in_sample_trades for w in self.results]),
            'avg_test_trades': np.mean([w.out_sample_trades for w in self.results]),
            'consistency_range': {
                'min_oos_sharpe': min(w.out_sample_sharpe for w in self.results),
                'max_oos_sharpe': max(w.out_sample_sharpe for w in self.results)
            },
            'overfitting_detected': any(w.overfitting_score > 0.5 for w in self.results)
        }
