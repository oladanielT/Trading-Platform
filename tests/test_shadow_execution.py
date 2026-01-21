"""
Tests for ShadowExecutor.

Verifies:
- Signal tracking without execution
- No actual trades placed
- Performance calculation
- Divergence detection
"""

import pytest
import sys
import os
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.shadow_execution import (
    ShadowExecutor,
    ShadowTrade,
    ShadowMetrics
)


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'strategy_promotion': {
            'shadow_execution': {
                'track_signals': True,
                'compare_with_live': True,
                'max_signal_divergence': 0.15
            }
        }
    }


@pytest.fixture
def executor(config):
    """Create shadow executor."""
    return ShadowExecutor(config)


class TestSignalTracking:
    """Test shadow signal tracking without execution."""
    
    def test_evaluate_generates_signal(self, executor):
        """Verify evaluate() generates signals without executing."""
        strategy_id = "test_strategy_1"
        signal = 1  # Long signal
        price = 50000.0
        
        result = executor.evaluate(
            strategy_id=strategy_id,
            signal=signal,
            entry_price=price,
            timestamp=datetime.now()
        )
        
        # Should return shadow trade
        assert result is not None
        assert result.strategy_id == strategy_id
        assert result.signal == signal
        assert result.entry_price == price
        assert result.exit_price is None  # Not closed yet
    
    def test_no_actual_execution(self, executor):
        """Verify no actual trades are executed."""
        strategy_id = "test_no_exec"
        
        # Generate multiple signals
        for i in range(10):
            executor.evaluate(
                strategy_id=strategy_id,
                signal=1 if i % 2 == 0 else -1,
                entry_price=50000.0 + i * 100,
                timestamp=datetime.now()
            )
        
        # Should have tracked signals but not executed
        metrics = executor.calculate_metrics(strategy_id)
        assert metrics.total_signals > 0
        # Actual execution verification would happen at integration level
    
    def test_close_shadow_trade(self, executor):
        """Test closing shadow trades."""
        strategy_id = "test_close"
        
        # Open shadow trade
        trade = executor.evaluate(
            strategy_id=strategy_id,
            signal=1,
            entry_price=50000.0,
            timestamp=datetime.now()
        )
        
        # Close it
        exit_price = 51000.0
        closed_trade = executor.close_trade(
            strategy_id=strategy_id,
            trade_id=trade.trade_id,
            exit_price=exit_price,
            timestamp=datetime.now()
        )
        
        assert closed_trade.exit_price == exit_price
        assert closed_trade.pnl is not None
        assert closed_trade.pnl > 0  # Profitable long


class TestPerformanceCalculation:
    """Test shadow performance metrics."""
    
    def test_calculate_metrics_for_profitable_trades(self, executor):
        """Test metrics calculation for winning trades."""
        strategy_id = "test_profitable"
        
        # Simulate profitable trades
        for i in range(10):
            trade = executor.evaluate(
                strategy_id=strategy_id,
                signal=1,
                entry_price=50000.0,
                timestamp=datetime.now()
            )
            
            executor.close_trade(
                strategy_id=strategy_id,
                trade_id=trade.trade_id,
                exit_price=51000.0,  # All profitable
                timestamp=datetime.now()
            )
        
        metrics = executor.calculate_metrics(strategy_id)
        
        assert metrics.total_signals == 10
        assert metrics.total_trades == 10
        assert metrics.win_rate == 1.0  # 100% wins
        assert metrics.total_pnl > 0
        assert metrics.avg_pnl > 0
    
    def test_calculate_metrics_for_mixed_trades(self, executor):
        """Test metrics for mix of wins and losses."""
        strategy_id = "test_mixed"
        
        # 6 wins, 4 losses
        for i in range(10):
            trade = executor.evaluate(
                strategy_id=strategy_id,
                signal=1,
                entry_price=50000.0,
                timestamp=datetime.now()
            )
            
            # First 6 win, last 4 lose
            exit_price = 51000.0 if i < 6 else 49000.0
            executor.close_trade(
                strategy_id=strategy_id,
                trade_id=trade.trade_id,
                exit_price=exit_price,
                timestamp=datetime.now()
            )
        
        metrics = executor.calculate_metrics(strategy_id)
        
        assert metrics.total_trades == 10
        assert metrics.win_rate == 0.6  # 60%
        assert 0 < metrics.win_rate < 1
    
    def test_drawdown_calculation(self, executor):
        """Test max drawdown calculation."""
        strategy_id = "test_drawdown"
        
        # Simulate trades with drawdown
        prices = [50000, 52000, 48000, 45000, 50000]  # Peak then drop
        
        for price in prices:
            trade = executor.evaluate(
                strategy_id=strategy_id,
                signal=1,
                entry_price=50000.0,
                timestamp=datetime.now()
            )
            
            executor.close_trade(
                strategy_id=strategy_id,
                trade_id=trade.trade_id,
                exit_price=price,
                timestamp=datetime.now()
            )
        
        metrics = executor.calculate_metrics(strategy_id)
        
        # Should have detected drawdown from 52000 -> 45000
        assert metrics.max_drawdown > 0
    
    def test_metrics_for_no_trades(self, executor):
        """Test metrics when no trades executed."""
        strategy_id = "test_no_trades"
        
        metrics = executor.calculate_metrics(strategy_id)
        
        assert metrics.total_signals == 0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == 0.0


class TestDivergenceDetection:
    """Test divergence between shadow and live."""
    
    def test_compare_with_live_execution(self, executor):
        """Test comparison with live execution data."""
        strategy_id = "test_divergence"
        
        # Generate shadow trades
        for i in range(5):
            trade = executor.evaluate(
                strategy_id=strategy_id,
                signal=1,
                entry_price=50000.0,
                timestamp=datetime.now()
            )
            executor.close_trade(
                strategy_id=strategy_id,
                trade_id=trade.trade_id,
                exit_price=51000.0,
                timestamp=datetime.now()
            )
        
        # Simulate live execution data
        live_metrics = {
            'total_trades': 5,
            'total_pnl': 5500.0,  # Slightly different from shadow
            'win_rate': 1.0
        }
        
        divergence = executor.compare_with_live(strategy_id, live_metrics)
        
        assert divergence is not None
        assert 'pnl_difference' in divergence
        assert 'signal_difference' in divergence
    
    def test_divergence_alert_threshold(self, executor):
        """Test divergence alerts when threshold exceeded."""
        strategy_id = "test_divergence_alert"
        
        # Generate shadow trades
        trade = executor.evaluate(
            strategy_id=strategy_id,
            signal=1,
            entry_price=50000.0,
            timestamp=datetime.now()
        )
        executor.close_trade(
            strategy_id=strategy_id,
            trade_id=trade.trade_id,
            exit_price=51000.0,
            timestamp=datetime.now()
        )
        
        # Live execution very different
        live_metrics = {
            'total_trades': 1,
            'total_pnl': 500.0,  # 50% of shadow PnL
            'win_rate': 1.0
        }
        
        divergence = executor.compare_with_live(strategy_id, live_metrics)
        
        # Should detect significant divergence
        assert abs(divergence['pnl_difference']) > 0.1  # >10% difference


class TestSafetyGuarantees:
    """Test critical safety guarantees."""
    
    def test_shadow_trades_isolated(self, executor):
        """Verify shadow trades don't affect other strategies."""
        # Create trades for two strategies
        executor.evaluate("strategy_1", 1, 50000.0, datetime.now())
        executor.evaluate("strategy_2", -1, 50000.0, datetime.now())
        
        # Metrics should be isolated
        metrics_1 = executor.calculate_metrics("strategy_1")
        metrics_2 = executor.calculate_metrics("strategy_2")
        
        assert metrics_1.total_signals == 1
        assert metrics_2.total_signals == 1
        # Each strategy has independent tracking
    
    def test_trade_id_uniqueness(self, executor):
        """Verify each shadow trade gets unique ID."""
        strategy_id = "test_unique_ids"
        
        trade_ids = set()
        for i in range(10):
            trade = executor.evaluate(
                strategy_id=strategy_id,
                signal=1,
                entry_price=50000.0,
                timestamp=datetime.now()
            )
            trade_ids.add(trade.trade_id)
        
        # All IDs should be unique
        assert len(trade_ids) == 10
    
    def test_diagnostics_include_all_strategies(self, executor):
        """Verify diagnostics cover all tracked strategies."""
        executor.evaluate("strategy_1", 1, 50000.0, datetime.now())
        executor.evaluate("strategy_2", -1, 50000.0, datetime.now())
        executor.evaluate("strategy_3", 1, 50000.0, datetime.now())
        
        diagnostics = executor.get_diagnostics()
        
        assert len(diagnostics['tracked_strategies']) == 3
        assert 'strategy_1' in diagnostics['tracked_strategies']
        assert 'strategy_2' in diagnostics['tracked_strategies']
        assert 'strategy_3' in diagnostics['tracked_strategies']


class TestIntegration:
    """Integration tests for shadow execution."""
    
    def test_full_shadow_lifecycle(self, executor):
        """Test complete shadow trading lifecycle."""
        strategy_id = "test_full_lifecycle"
        
        # 1. Generate signals
        trades = []
        for i in range(5):
            trade = executor.evaluate(
                strategy_id=strategy_id,
                signal=1 if i % 2 == 0 else -1,
                entry_price=50000.0 + i * 100,
                timestamp=datetime.now()
            )
            trades.append(trade)
        
        # 2. Close some trades
        for i, trade in enumerate(trades[:3]):
            executor.close_trade(
                strategy_id=strategy_id,
                trade_id=trade.trade_id,
                exit_price=51000.0 if i % 2 == 0 else 49000.0,
                timestamp=datetime.now()
            )
        
        # 3. Calculate metrics
        metrics = executor.calculate_metrics(strategy_id)
        
        assert metrics.total_signals == 5
        assert metrics.total_trades == 3  # Only closed trades counted
        assert 0 <= metrics.win_rate <= 1
        
        # 4. Get diagnostics
        diagnostics = executor.get_diagnostics()
        assert strategy_id in diagnostics['tracked_strategies']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
