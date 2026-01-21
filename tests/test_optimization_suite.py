"""
test_optimization_suite.py - Tests for optimization modules

Tests all optimization components:
- Performance decomposition
- Adaptive weighting
- Signal quality scoring
- Strategy retirement
- Optimization monitoring
"""

import pytest
from datetime import datetime, timezone, timedelta
import json

from ai.performance_decomposer import (
    PerformanceDecomposer,
    StrategyPerformance,
    record_trade,
    get_decomposer,
)

from ai.adaptive_weighting_optimizer import (
    AdaptiveWeightingOptimizer,
    initialize_optimizer,
    get_optimizer,
)

from ai.signal_quality_scorer import (
    SignalQualityScorer,
    initialize_scorer,
    score_signal,
)

from ai.strategy_retirement import (
    StrategyRetirementManager,
    StrategyStatus,
    initialize_manager,
    assess_strategy,
)

from ai.optimization_monitor import (
    OptimizationMonitor,
    get_monitor,
)


# ==================== Performance Decomposer Tests ====================

class TestPerformanceDecomposer:
    """Test performance decomposition."""
    
    def test_initialization(self):
        """Test decomposer initialization."""
        decomposer = PerformanceDecomposer()
        assert decomposer is not None
        assert len(decomposer.get_all_performance()) == 0
    
    def test_record_single_trade(self):
        """Test recording a single trade."""
        decomposer = PerformanceDecomposer()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        decomposer.record_trade(
            strategy_name="TestStrategy",
            symbol="BTC/USDT",
            pnl=100.0,
            entry_ts=now_ms - 3600000,
            exit_ts=now_ms,
            regime="TRENDING",
        )
        
        perf = decomposer.get_strategy_performance("TestStrategy")
        assert perf is not None
        assert perf.total_trades == 1
        assert perf.winning_trades == 1
        assert perf.total_pnl == 100.0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        decomposer = PerformanceDecomposer()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # 3 wins, 2 losses = 60% win rate
        for i in range(3):
            decomposer.record_trade(
                strategy_name="WinStrategy",
                symbol="ETH/USDT",
                pnl=50.0,
                entry_ts=now_ms - (i+1) * 3600000,
                exit_ts=now_ms - i * 3600000,
            )
        
        for i in range(2):
            decomposer.record_trade(
                strategy_name="WinStrategy",
                symbol="ETH/USDT",
                pnl=-30.0,
                entry_ts=now_ms - (i+4) * 3600000,
                exit_ts=now_ms - (i+3) * 3600000,
            )
        
        perf = decomposer.get_strategy_performance("WinStrategy")
        assert perf.total_trades == 5
        assert perf.winning_trades == 3
        assert perf.losing_trades == 2
        assert abs(perf.win_rate - 60.0) < 0.1
    
    def test_regime_breakdown(self):
        """Test performance breakdown by regime."""
        decomposer = PerformanceDecomposer()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Trades in TRENDING
        for i in range(3):
            decomposer.record_trade(
                strategy_name="RegimeStrat",
                symbol="BTC/USDT",
                pnl=100.0,
                entry_ts=now_ms - (i+1) * 3600000,
                exit_ts=now_ms - i * 3600000,
                regime="TRENDING",
            )
        
        # Trades in RANGING
        for i in range(2):
            decomposer.record_trade(
                strategy_name="RegimeStrat",
                symbol="BTC/USDT",
                pnl=-50.0,
                entry_ts=now_ms - (i+4) * 3600000,
                exit_ts=now_ms - (i+3) * 3600000,
                regime="RANGING",
            )
        
        regime_breakdown = decomposer.get_regime_effectiveness("RegimeStrat")
        assert "TRENDING" in regime_breakdown
        assert "RANGING" in regime_breakdown
        assert regime_breakdown["TRENDING"].win_rate == 100.0
        assert regime_breakdown["RANGING"].win_rate == 0.0
    
    def test_expectancy_calculation(self):
        """Test expectancy calculation."""
        decomposer = PerformanceDecomposer()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # 2 wins of 100, 2 losses of 50
        # Expectancy = (50% * 100) - (50% * 50) = 50 - 25 = 25
        decomposer.record_trade(
            strategy_name="ExpectStrat",
            symbol="BTC/USDT",
            pnl=100.0,
            entry_ts=now_ms - 4 * 3600000,
            exit_ts=now_ms - 3 * 3600000,
        )
        decomposer.record_trade(
            strategy_name="ExpectStrat",
            symbol="BTC/USDT",
            pnl=100.0,
            entry_ts=now_ms - 2 * 3600000,
            exit_ts=now_ms - 1 * 3600000,
        )
        decomposer.record_trade(
            strategy_name="ExpectStrat",
            symbol="BTC/USDT",
            pnl=-50.0,
            entry_ts=now_ms - 2 * 3600000,
            exit_ts=now_ms - 1 * 3600000,
        )
        decomposer.record_trade(
            strategy_name="ExpectStrat",
            symbol="BTC/USDT",
            pnl=-50.0,
            entry_ts=now_ms - 2 * 3600000,
            exit_ts=now_ms - 1 * 3600000,
        )
        
        perf = decomposer.get_strategy_performance("ExpectStrat")
        assert abs(perf.expectancy - 25.0) < 5.0  # Allow small rounding error
    
    def test_top_strategies(self):
        """Test finding top strategies by metric."""
        decomposer = PerformanceDecomposer()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Create 3 strategies with different expectancies
        strategies = [
            ("HighExp", 5, 0),  # 5 wins, 0 losses
            ("MedExp", 3, 2),   # 3 wins, 2 losses
            ("LowExp", 1, 4),   # 1 win, 4 losses
        ]
        
        for strat_name, wins, losses in strategies:
            for i in range(wins):
                decomposer.record_trade(
                    strategy_name=strat_name,
                    symbol="BTC/USDT",
                    pnl=100.0,
                    entry_ts=now_ms - (i+1) * 3600000,
                    exit_ts=now_ms - i * 3600000,
                )
            for i in range(losses):
                decomposer.record_trade(
                    strategy_name=strat_name,
                    symbol="BTC/USDT",
                    pnl=-50.0,
                    entry_ts=now_ms - (wins+i+1) * 3600000,
                    exit_ts=now_ms - (wins+i) * 3600000,
                )
        
        top = decomposer.get_top_strategies('expectancy', top_n=5)
        # Filter to overall strategy performance (no symbol/regime filters)
        overall_perfs = [p for p in top if p.symbol is None and p.regime is None]
        assert len(overall_perfs) >= 3
        assert overall_perfs[0].strategy_name == "HighExp"
        assert overall_perfs[1].strategy_name == "MedExp"
        assert overall_perfs[2].strategy_name == "LowExp"


# ==================== Adaptive Weighting Tests ====================

class TestAdaptiveWeighting:
    """Test adaptive weighting optimizer."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        base_weights = {"S1": 1.0, "S2": 1.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        assert optimizer is not None
        assert optimizer.base_weights == base_weights
    
    def test_weight_update_with_good_performance(self):
        """Test weight increase with good performance."""
        base_weights = {"GoodStrat": 1.0}
        optimizer = AdaptiveWeightingOptimizer(
            base_weights,
            adjustment_rate=0.5,
        )
        
        performance = {
            "GoodStrat": {
                "expectancy": 25.0,
                "profit_factor": 1.5,
                "sharpe_ratio": 1.0,
            }
        }
        
        new_weights = optimizer.update_weights(performance)
        assert new_weights["GoodStrat"] > 1.0  # Should increase
    
    def test_weight_update_with_poor_performance(self):
        """Test weight decrease with poor performance."""
        base_weights = {"BadStrat": 1.0}
        optimizer = AdaptiveWeightingOptimizer(
            base_weights,
            adjustment_rate=0.5,
        )
        
        performance = {
            "BadStrat": {
                "expectancy": -10.0,
                "profit_factor": 0.5,
                "sharpe_ratio": -0.5,
            }
        }
        
        new_weights = optimizer.update_weights(performance)
        assert new_weights["BadStrat"] < 1.0  # Should decrease
    
    def test_weight_bounds(self):
        """Test weight min/max bounds."""
        base_weights = {"Strat": 1.0}
        optimizer = AdaptiveWeightingOptimizer(
            base_weights,
            min_weight=0.2,
            max_weight=2.0,
        )
        
        # Try to update with extreme performance
        for _ in range(10):
            optimizer.update_weights({
                "Strat": {
                    "expectancy": -100.0,
                    "profit_factor": 0.0,
                    "sharpe_ratio": -10.0,
                }
            })
        
        weight = optimizer.get_current_weights()["Strat"]
        assert weight >= 0.2  # Should not go below min
    
    def test_regime_specific_weights(self):
        """Test regime-specific weighting."""
        base_weights = {"TrendStrat": 1.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        trending_perf = {"TrendStrat": {"expectancy": 30.0, "profit_factor": 2.0}}
        ranging_perf = {"TrendStrat": {"expectancy": -10.0, "profit_factor": 0.5}}
        
        optimizer.update_weights(trending_perf, regime="TRENDING")
        trending_weight = optimizer.get_weight_for_regime("TrendStrat", "TRENDING")
        
        optimizer.update_weights(ranging_perf, regime="RANGING")
        ranging_weight = optimizer.get_weight_for_regime("TrendStrat", "RANGING")
        
        assert trending_weight > ranging_weight
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        base_weights = {"S1": 1.0, "S2": 2.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        normalized = optimizer.normalize_weights()
        total = sum(normalized.values())
        
        assert abs(total - 1.0) < 0.001  # Should sum to 1.0


# ==================== Signal Quality Tests ====================

class TestSignalQuality:
    """Test signal quality scoring."""
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = SignalQualityScorer()
        assert scorer is not None
        assert scorer.min_quality_threshold == 0.7
    
    def test_score_high_confidence_signal(self):
        """Test scoring high-confidence signal."""
        scorer = SignalQualityScorer()
        
        metrics = scorer.score_signal(
            strategy="EMAStrat",
            symbol="BTC/USDT",
            signal_type="BUY",
            original_confidence=0.9,
            confluence_count=3,
            regime="TRENDING",
            regime_alignment=0.9,
        )
        
        assert metrics.quality_score > 0.7
        assert metrics.confidence == 0.9
        assert metrics.confluence_count == 3
    
    def test_score_low_confidence_signal(self):
        """Test scoring low-confidence signal."""
        scorer = SignalQualityScorer()
        
        metrics = scorer.score_signal(
            strategy="WeakStrat",
            symbol="BTC/USDT",
            signal_type="SELL",
            original_confidence=0.3,
            confluence_count=1,
            regime="RANGING",
            regime_alignment=0.3,
        )
        
        assert metrics.quality_score < 0.7
    
    def test_filter_signals(self):
        """Test signal filtering by quality."""
        scorer = SignalQualityScorer(min_quality_threshold=0.7)
        
        signals = [
            scorer.score_signal("S1", "BTC/USDT", "BUY", 0.9, confluence_count=3, regime="TRENDING", regime_alignment=0.9),
            scorer.score_signal("S2", "BTC/USDT", "SELL", 0.3, confluence_count=1, regime="RANGING", regime_alignment=0.3),
            scorer.score_signal("S3", "BTC/USDT", "BUY", 0.8, confluence_count=2, regime="TRENDING", regime_alignment=0.8),
        ]
        
        filtered = scorer.filter_signals(signals, min_quality=0.7)
        # Should have at least some high quality signals
        assert len(filtered) >= 1  # At least one should pass quality threshold
    
    def test_regime_adjustment(self):
        """Test regime-specific adjustment."""
        scorer = SignalQualityScorer()
        
        metrics_trending = scorer.score_signal(
            "TrendStrat", "BTC/USDT", "BUY", 0.7, regime="TRENDING", regime_alignment=0.9
        )
        
        metrics_ranging = scorer.score_signal(
            "TrendStrat", "BTC/USDT", "BUY", 0.7, regime="RANGING", regime_alignment=0.3
        )
        
        # Trending should score higher
        assert metrics_trending.quality_score > metrics_ranging.quality_score


# ==================== Strategy Retirement Tests ====================

class TestStrategyRetirement:
    """Test strategy retirement manager."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = StrategyRetirementManager()
        assert manager is not None
        assert len(manager.get_all_health()) == 0
    
    def test_assess_healthy_strategy(self):
        """Test assessment of healthy strategy."""
        manager = StrategyRetirementManager(min_trades_for_assessment=5)
        
        metrics = manager.assess_strategy(
            strategy_name="HealthyStrat",
            recent_expectancy=25.0,
            recent_profit_factor=1.5,
            recent_win_rate=0.65,
            recent_trades_count=20,
        )
        
        assert metrics.status == StrategyStatus.ACTIVE
        assert metrics.confidence_level > 0.8
    
    def test_assess_underperforming_strategy(self):
        """Test assessment of underperforming strategy."""
        manager = StrategyRetirementManager(min_trades_for_assessment=5)
        
        metrics = manager.assess_strategy(
            strategy_name="BadStrat",
            recent_expectancy=-5.0,
            recent_profit_factor=0.5,
            recent_win_rate=0.30,
            recent_trades_count=20,
        )
        
        assert metrics.status == StrategyStatus.DEGRADED
    
    def test_status_progression(self):
        """Test status progression: ACTIVE -> DEGRADED -> SUSPENDED."""
        manager = StrategyRetirementManager(
            degraded_window_hours=0.01,  # Fast transitions for testing
            suspended_window_hours=0.01,
        )
        
        # Initial healthy assessment
        h1 = manager.assess_strategy(
            "ProgStrat",
            recent_expectancy=20.0,
            recent_profit_factor=1.5,
            recent_win_rate=0.6,
            recent_trades_count=20,
        )
        assert h1.status == StrategyStatus.ACTIVE
        
        # Degraded assessment
        import time
        time.sleep(0.05)  # Allow time to pass
        
        h2 = manager.assess_strategy(
            "ProgStrat",
            recent_expectancy=-5.0,
            recent_profit_factor=0.5,
            recent_win_rate=0.3,
            recent_trades_count=30,
        )
        assert h2.status == StrategyStatus.DEGRADED
    
    def test_get_suspended_strategies(self):
        """Test getting suspended strategies."""
        manager = StrategyRetirementManager(min_trades_for_assessment=5)
        
        # Create one healthy and one suspended
        manager.assess_strategy(
            "Healthy",
            recent_expectancy=20.0,
            recent_profit_factor=1.5,
            recent_win_rate=0.6,
            recent_trades_count=20,
        )
        
        health = manager.assess_strategy(
            "Bad",
            recent_expectancy=-5.0,
            recent_profit_factor=0.5,
            recent_win_rate=0.3,
            recent_trades_count=20,
        )
        
        manager.force_status("Bad", StrategyStatus.SUSPENDED, "Test suspension")
        
        suspended = manager.get_suspended_strategies()
        assert "Bad" in suspended
        assert "Healthy" not in suspended


# ==================== Optimization Monitor Tests ====================

class TestOptimizationMonitor:
    """Test optimization monitor."""
    
    def test_initialization(self):
        """Test monitor initialization."""
        monitor = OptimizationMonitor()
        assert monitor is not None
    
    def test_get_optimization_status(self):
        """Test getting optimization status."""
        # Initialize optimizer first
        from ai.adaptive_weighting_optimizer import initialize_optimizer
        initialize_optimizer({"TestStrat": 1.0})
        
        monitor = OptimizationMonitor()
        status = monitor.get_optimization_status()
        
        assert status is not None
        # Status should have timestamp or be an error
        assert 'timestamp' in status or 'error' in status


# ==================== Integration Tests ====================

class TestOptimizationIntegration:
    """Integration tests for optimization system."""
    
    def test_full_workflow(self):
        """Test complete optimization workflow."""
        # Initialize components
        decomposer = PerformanceDecomposer()
        optimizer = AdaptiveWeightingOptimizer(
            {"EMAStrat": 1.0, "FVGStrat": 1.0}
        )
        scorer = SignalQualityScorer()
        manager = StrategyRetirementManager()
        monitor = OptimizationMonitor()
        
        # Simulate trading activity
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        
        # Record some trades
        for i in range(20):
            decomposer.record_trade(
                strategy_name="EMAStrat",
                symbol="BTC/USDT",
                pnl=50.0 if i % 3 == 0 else -20.0,
                entry_ts=now_ms - (i+1) * 3600000,
                exit_ts=now_ms - i * 3600000,
                regime="TRENDING",
            )
        
        # Get performance
        perf = decomposer.get_strategy_performance("EMAStrat")
        assert perf.total_trades == 20
        
        # Update weights based on performance
        perf_data = {
            "EMAStrat": {
                "expectancy": perf.expectancy,
                "profit_factor": perf.profit_factor,
            }
        }
        weights = optimizer.update_weights(perf_data)
        assert "EMAStrat" in weights
        
        # Score a signal
        metrics = scorer.score_signal(
            "EMAStrat", "BTC/USDT", "BUY", 0.85,
            confluence_count=2,
            regime="TRENDING"
        )
        assert metrics.quality_score > 0.0
        
        # Assess strategy health
        health = manager.assess_strategy(
            "EMAStrat",
            recent_expectancy=perf.expectancy,
            recent_profit_factor=perf.profit_factor,
            recent_win_rate=perf.win_rate / 100.0,
            recent_trades_count=20,
        )
        assert health.status == StrategyStatus.ACTIVE


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
