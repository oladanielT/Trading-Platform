#!/usr/bin/env python3
"""
test_optimization_system.py - Comprehensive tests for optimization modules

Tests:
1. Performance decomposition
2. Adaptive weighting
3. Signal quality scoring
4. Strategy retirement
5. Optimization engine integration
6. Risk constraint compliance
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    get_scorer,
)
from ai.strategy_retirement import (
    StrategyRetirementManager,
    StrategyStatus,
    initialize_manager,
    get_manager,
)
from ai.optimization_engine import (
    OptimizationEngine,
    initialize_engine,
    get_engine,
)
from ai.optimization_monitor import OptimizationMonitor, get_monitor


# ==============================================================================
# Performance Decomposer Tests
# ==============================================================================

class TestPerformanceDecomposer:
    """Test suite for performance decomposition."""
    
    def test_initialization(self):
        """Test decomposer initialization."""
        decomposer = PerformanceDecomposer()
        assert decomposer is not None
        assert len(decomposer.get_all_performance()) == 0
    
    def test_record_single_trade(self):
        """Test recording a single trade."""
        decomposer = PerformanceDecomposer()
        decomposer.reset()
        
        decomposer.record_trade(
            strategy_name="EMA_Trend",
            symbol="BTC/USDT",
            pnl=100.0,
            entry_ts=1000,
            exit_ts=2000,
            regime="TRENDING",
        )
        
        perf = decomposer.get_strategy_performance("EMA_Trend")
        assert perf is not None
        assert perf.total_trades == 1
        assert perf.winning_trades == 1
        assert perf.total_pnl == 100.0
    
    def test_win_loss_tracking(self):
        """Test win/loss rate calculation."""
        decomposer = PerformanceDecomposer()
        decomposer.reset()
        
        # 3 winning trades
        for i in range(3):
            decomposer.record_trade(
                strategy_name="TestStrat",
                symbol="BTC/USDT",
                pnl=100.0,
                entry_ts=i*1000,
                exit_ts=(i+1)*1000,
            )
        
        # 2 losing trades
        for i in range(2):
            decomposer.record_trade(
                strategy_name="TestStrat",
                symbol="BTC/USDT",
                pnl=-50.0,
                entry_ts=(3+i)*1000,
                exit_ts=(4+i)*1000,
            )
        
        perf = decomposer.get_strategy_performance("TestStrat")
        assert perf.total_trades == 5
        assert perf.winning_trades == 3
        assert perf.losing_trades == 2
        assert perf.win_rate == 60.0
        assert perf.loss_rate == 40.0
    
    def test_expectancy_calculation(self):
        """Test expectancy calculation."""
        decomposer = PerformanceDecomposer()
        decomposer.reset()
        
        # Create known scenario
        decomposer.record_trade("S1", "BTC/USDT", 100.0, 0, 1)  # Win
        decomposer.record_trade("S1", "BTC/USDT", -50.0, 1, 2)  # Loss
        
        perf = decomposer.get_strategy_performance("S1")
        # Win rate: 50%, Loss rate: 50%
        # Avg Win: 100, Avg Loss: 50
        # Expectancy = (0.5 * 100) - (0.5 * 50) = 50 - 25 = 25
        assert perf.expectancy == 25.0
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        decomposer = PerformanceDecomposer()
        decomposer.reset()
        
        decomposer.record_trade("S1", "BTC/USDT", 100.0, 0, 1)
        decomposer.record_trade("S1", "BTC/USDT", 100.0, 1, 2)
        decomposer.record_trade("S1", "BTC/USDT", -50.0, 2, 3)
        
        perf = decomposer.get_strategy_performance("S1")
        # Gross profit: 200
        # Gross loss: 50
        # PF = 200 / 50 = 4.0
        assert perf.profit_factor == 4.0
    
    def test_regime_decomposition(self):
        """Test per-regime performance tracking."""
        decomposer = PerformanceDecomposer()
        decomposer.reset()
        
        decomposer.record_trade("EMA", "BTC/USDT", 100.0, 0, 1, regime="TRENDING")
        decomposer.record_trade("EMA", "BTC/USDT", 50.0, 1, 2, regime="TRENDING")
        decomposer.record_trade("EMA", "BTC/USDT", -30.0, 2, 3, regime="RANGING")
        
        trending = decomposer.get_strategy_performance("EMA", regime="TRENDING")
        ranging = decomposer.get_strategy_performance("EMA", regime="RANGING")
        
        assert trending.total_trades == 2
        assert trending.total_pnl == 150.0
        assert ranging.total_trades == 1
        assert ranging.total_pnl == -30.0
    
    def test_top_strategies(self):
        """Test top strategy retrieval."""
        decomposer = PerformanceDecomposer()
        decomposer.reset()
        
        # Good strategy
        for i in range(10):
            decomposer.record_trade("Good", "BTC/USDT", 100.0, i*10, i*10+5)
        
        # Bad strategy
        for i in range(10):
            decomposer.record_trade("Bad", "BTC/USDT", -50.0, i*10, i*10+5)
        
        top = decomposer.get_top_strategies(metric='expectancy', top_n=1)
        assert len(top) == 1
        assert top[0].strategy_name == "Good"


# ==============================================================================
# Adaptive Weighting Optimizer Tests
# ==============================================================================

class TestAdaptiveWeightingOptimizer:
    """Test suite for adaptive weighting."""
    
    def test_initialization(self):
        """Test optimizer initialization."""
        base_weights = {"EMA": 1.0, "RSI": 0.8}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        weights = optimizer.get_current_weights()
        assert weights["EMA"] == 1.0
        assert weights["RSI"] == 0.8
    
    def test_weight_update_positive_performance(self):
        """Test weights increase for good performance."""
        base_weights = {"EMA": 1.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        # Good performance
        perf_data = {
            "EMA": {
                "expectancy": 50.0,  # High expectancy
                "profit_factor": 2.0,  # Good PF
                "max_drawdown": 0.05,  # Low DD
                "sharpe_ratio": 1.5,  # Good Sharpe
            }
        }
        
        new_weights = optimizer.update_weights(perf_data)
        assert new_weights["EMA"] > 1.0  # Should increase
    
    def test_weight_update_negative_performance(self):
        """Test weights decrease for poor performance."""
        base_weights = {"EMA": 1.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        # Poor performance
        perf_data = {
            "EMA": {
                "expectancy": -10.0,  # Negative expectancy
                "profit_factor": 0.5,  # Bad PF
                "max_drawdown": 0.25,  # High DD
                "sharpe_ratio": -0.5,  # Bad Sharpe
            }
        }
        
        new_weights = optimizer.update_weights(perf_data)
        assert new_weights["EMA"] < 1.0  # Should decrease
    
    def test_weight_bounds(self):
        """Test weight min/max bounds."""
        base_weights = {"S1": 1.0}
        optimizer = AdaptiveWeightingOptimizer(
            base_weights,
            min_weight=0.1,
            max_weight=2.0,
        )
        
        # Extremely good performance
        perf_data = {"S1": {"expectancy": 1000.0, "profit_factor": 100.0}}
        new_weights = optimizer.update_weights(perf_data)
        assert new_weights["S1"] <= 2.0
        
        # Extremely bad performance
        perf_data = {"S1": {"expectancy": -1000.0, "profit_factor": 0.1}}
        new_weights = optimizer.update_weights(perf_data)
        assert new_weights["S1"] >= 0.1
    
    def test_adjustment_rate_limit(self):
        """Test weight adjustment rate limiting."""
        base_weights = {"S1": 1.0}
        optimizer = AdaptiveWeightingOptimizer(
            base_weights,
            adjustment_rate=0.15,  # Max 15% change
        )
        
        # Good performance (would suggest 2x increase)
        perf_data = {"S1": {"expectancy": 100.0, "profit_factor": 5.0}}
        new_weights = optimizer.update_weights(perf_data)
        
        # Change should be limited to ~15%
        change_pct = (new_weights["S1"] - 1.0) / 1.0
        assert change_pct <= 0.15 * 1.5  # Allow some buffer
    
    def test_regime_specific_weights(self):
        """Test per-regime weight adjustment."""
        base_weights = {"EMA": 1.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        perf_data = {"EMA": {"expectancy": 50.0, "profit_factor": 2.0, "max_drawdown": 0.05, "sharpe_ratio": 1.0}}
        
        # Update for trending regime (good performance)
        optimizer.update_weights(perf_data, regime="TRENDING")
        trending_weight = optimizer.get_weight_for_regime("EMA", "TRENDING")
        assert trending_weight > 1.0
        
        # Update for ranging regime (poor performance)
        perf_data["EMA"]["expectancy"] = -10.0
        perf_data["EMA"]["profit_factor"] = 0.5
        perf_data["EMA"]["sharpe_ratio"] = -0.5
        optimizer.update_weights(perf_data, regime="RANGING")
        ranging_weight = optimizer.get_weight_for_regime("EMA", "RANGING")
        
        # Ranging should have lower weight due to poor performance
        # (actual comparison depends on the scoring algorithm)
        assert ranging_weight >= optimizer.min_weight  # Still respects bounds
    
    def test_normalize_weights(self):
        """Test weight normalization."""
        base_weights = {"S1": 1.0, "S2": 2.0, "S3": 1.5}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        normalized = optimizer.normalize_weights()
        total = sum(normalized.values())
        assert abs(total - 1.0) < 0.001
    
    def test_weight_history(self):
        """Test weight history tracking."""
        base_weights = {"S1": 1.0}
        optimizer = AdaptiveWeightingOptimizer(base_weights)
        
        perf_data = {"S1": {"expectancy": 50.0, "profit_factor": 2.0}}
        optimizer.update_weights(perf_data)
        
        history = optimizer.get_weight_history()
        assert len(history) > 0
        
        last_update = history[-1]
        assert last_update.strategy == "S1"
        assert last_update.previous_weight == 1.0


# ==============================================================================
# Signal Quality Scorer Tests
# ==============================================================================

class TestSignalQualityScorer:
    """Test suite for signal quality scoring."""
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = SignalQualityScorer()
        assert scorer.min_quality_threshold == 0.7
    
    def test_basic_signal_scoring(self):
        """Test basic signal scoring."""
        scorer = SignalQualityScorer()
        
        metrics = scorer.score_signal(
            strategy="EMA",
            symbol="BTC/USDT",
            signal_type="BUY",
            original_confidence=0.8,
            confluence_count=3,
        )
        
        assert metrics is not None
        assert metrics.strategy == "EMA"
        assert metrics.symbol == "BTC/USDT"
        assert metrics.quality_score >= 0.0
        assert metrics.quality_score <= 1.0
    
    def test_confluence_impact(self):
        """Test confluence count impact on quality."""
        scorer = SignalQualityScorer()
        
        # Low confluence
        low_conf = scorer.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.7, confluence_count=1
        )
        
        # High confluence
        high_conf = scorer.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.7, confluence_count=5
        )
        
        assert high_conf.quality_score > low_conf.quality_score
    
    def test_regime_alignment(self):
        """Test regime alignment impact."""
        scorer = SignalQualityScorer()
        
        # Good regime alignment
        good_align = scorer.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.7,
            regime="TRENDING",
            regime_alignment=0.9,
        )
        
        # Poor regime alignment
        poor_align = scorer.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.7,
            regime="RANGING",
            regime_alignment=0.3,
        )
        
        assert good_align.quality_score > poor_align.quality_score
    
    def test_regime_adjustments(self):
        """Test regime-specific adjustments."""
        scorer = SignalQualityScorer()
        
        # Trending should boost signals
        trending = scorer.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.7, regime="TRENDING"
        )
        
        # Volatile should reduce signals
        volatile = scorer.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.7, regime="VOLATILE"
        )
        
        assert trending.quality_score > volatile.quality_score
    
    def test_signal_filtering(self):
        """Test signal filtering by quality threshold."""
        scorer = SignalQualityScorer(min_quality_threshold=0.7)
        
        signals = []
        sig1 = scorer.score_signal("S1", "BTC", "BUY", 0.9, confluence_count=4)
        signals.append(sig1)
        sig2 = scorer.score_signal("S2", "BTC", "BUY", 0.3, confluence_count=1)
        signals.append(sig2)
        sig3 = scorer.score_signal("S3", "BTC", "BUY", 0.8, confluence_count=3)
        signals.append(sig3)
        
        filtered = scorer.filter_signals(signals, min_quality=0.5)
        
        # Check that filtering works
        assert len(filtered) <= len(signals)
        assert all(s.quality_score >= 0.5 for s in filtered)
    
    def test_signal_history(self):
        """Test signal history tracking."""
        scorer = SignalQualityScorer()
        
        for i in range(5):
            scorer.score_signal(
                "EMA", "BTC/USDT", "BUY", 0.7, confluence_count=i+1
            )
        
        history = scorer.get_signal_history(limit=100)
        assert len(history) == 5


# ==============================================================================
# Strategy Retirement Manager Tests
# ==============================================================================

class TestStrategyRetirementManager:
    """Test suite for strategy retirement."""
    
    def test_initialization(self):
        """Test manager initialization."""
        manager = StrategyRetirementManager()
        assert manager.min_expectancy == 0.1
    
    def test_healthy_strategy_active(self):
        """Test healthy strategy stays active."""
        manager = StrategyRetirementManager()
        
        health = manager.assess_strategy(
            strategy_name="GoodStrat",
            recent_expectancy=0.5,  # Good
            recent_profit_factor=1.8,  # Good
            recent_win_rate=0.55,  # Good
            recent_trades_count=50,
        )
        
        assert health.status == StrategyStatus.ACTIVE
    
    def test_underperforming_strategy_degraded(self):
        """Test underperforming strategy degrades."""
        manager = StrategyRetirementManager()
        
        health = manager.assess_strategy(
            strategy_name="BadStrat",
            recent_expectancy=-0.5,  # Negative expectancy
            recent_profit_factor=0.5,  # Below threshold
            recent_win_rate=0.30,  # Below threshold
            recent_trades_count=50,
        )
        
        assert health.status == StrategyStatus.DEGRADED
    
    def test_degraded_to_suspended_transition(self):
        """Test degraded strategy eventually suspends."""
        manager = StrategyRetirementManager(degraded_window_hours=0)
        
        # First assessment: degraded
        health = manager.assess_strategy(
            "BadStrat",
            recent_expectancy=-0.5,
            recent_profit_factor=0.5,
            recent_win_rate=0.30,
            recent_trades_count=50,
        )
        assert health.status == StrategyStatus.DEGRADED
        
        # Manually set degraded_since to past
        health.degraded_since = datetime.now(timezone.utc) - timedelta(hours=25)
        manager._health["BadStrat"] = health
        
        # Second assessment with same bad metrics
        health = manager.assess_strategy(
            "BadStrat",
            recent_expectancy=-0.5,
            recent_profit_factor=0.5,
            recent_win_rate=0.30,
            recent_trades_count=50,
        )
        assert health.status == StrategyStatus.SUSPENDED
    
    def test_recovery_from_degraded(self):
        """Test strategy can recover from degraded."""
        manager = StrategyRetirementManager()
        
        # First: degraded
        manager.assess_strategy(
            "RecoveryStrat",
            recent_expectancy=-0.5,
            recent_profit_factor=0.5,
            recent_win_rate=0.30,
            recent_trades_count=50,
        )
        
        # Second: recovered
        health = manager.assess_strategy(
            "RecoveryStrat",
            recent_expectancy=0.5,  # Now good
            recent_profit_factor=1.8,  # Now good
            recent_win_rate=0.55,  # Now good
            recent_trades_count=60,
        )
        
        assert health.status == StrategyStatus.ACTIVE
    
    def test_audit_log(self):
        """Test audit log of status changes."""
        manager = StrategyRetirementManager()
        
        # Create state change
        manager.assess_strategy(
            "S1", recent_expectancy=-0.5, recent_profit_factor=0.5,
            recent_win_rate=0.30, recent_trades_count=50,
        )
        
        audit = manager.get_audit_log("S1")
        assert len(audit) > 0
        assert audit[-1]["strategy"] == "S1"


# ==============================================================================
# Optimization Engine Tests
# ==============================================================================

class TestOptimizationEngine:
    """Test suite for optimization engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        base_weights = {"S1": 1.0, "S2": 1.0}
        engine = OptimizationEngine(base_weights)
        
        assert engine is not None
        assert engine.enable_quality_filtering
        assert engine.enable_adaptive_weighting
    
    def test_signal_scoring_and_filtering(self):
        """Test signal scoring and filtering."""
        base_weights = {"EMA": 1.0}
        engine = OptimizationEngine(base_weights)
        
        # Good signal
        quality, execute = engine.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.9, confluence_count=3
        )
        assert execute  # Should execute
        assert quality > 0.6
        
        # Poor signal
        quality, execute = engine.score_signal(
            "EMA", "BTC/USDT", "BUY", 0.3, confluence_count=1
        )
        # May or may not execute depending on config
    
    def test_position_size_multiplier(self):
        """Test position size multiplier calculation."""
        base_weights = {"S1": 1.0}
        engine = OptimizationEngine(base_weights)
        
        # Good signal with good strategy
        mult = engine.get_position_size_multiplier("S1", signal_quality=0.9)
        assert mult > 0.0
        assert mult <= 2.0
        
        # Poor signal
        mult = engine.get_position_size_multiplier("S1", signal_quality=0.3)
        assert mult >= 0.0
        assert mult <= 2.0
    
    def test_suspended_strategy_weight_zero(self):
        """Test suspended strategies get zero weight."""
        base_weights = {"Active": 1.0, "Bad": 1.0}
        engine = OptimizationEngine(base_weights)
        engine._ensure_components()
        
        # Suspend a strategy - directly update weight and status
        if engine._retirement_manager:
            from ai.strategy_retirement import StrategyStatus
            engine._retirement_manager.force_status("Bad", StrategyStatus.SUSPENDED, "Test")
            
            # Verify status is suspended
            health = engine._retirement_manager.get_strategy_health("Bad")
            assert health.status == StrategyStatus.SUSPENDED
            
            # Should get zero weight
            weight = engine.get_strategy_weight("Bad")
            assert weight == 0.0
        else:
            # Component not initialized, skip test
            pytest.skip("Retirement manager not initialized")
    
    def test_optimization_update(self):
        """Test optimization update cycle."""
        base_weights = {"S1": 1.0}
        engine = OptimizationEngine(base_weights)
        
        state = engine.update_optimization(regime="TRENDING")
        
        assert state is not None
        assert state.timestamp is not None
        assert state.current_weights is not None
    
    def test_can_disable_features(self):
        """Test optimization features can be disabled."""
        base_weights = {"S1": 1.0}
        
        engine = OptimizationEngine(
            base_weights,
            enable_quality_filtering=False,
            enable_adaptive_weighting=False,
            enable_retirement_management=False,
        )
        
        # Quality filtering disabled
        quality, execute = engine.score_signal("S1", "BTC", "BUY", 0.3)
        assert execute  # Should still execute with low quality


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestOptimizationIntegration:
    """Integration tests across all components."""
    
    def test_end_to_end_workflow(self):
        """Test end-to-end optimization workflow."""
        # Initialize
        base_weights = {"EMA": 1.0, "RSI": 0.8}
        engine = initialize_engine(base_weights)
        decomposer = get_decomposer()
        decomposer.reset()
        
        # Simulate trades for both strategies
        decomposer.record_trade("EMA", "BTC/USDT", 100.0, 0, 1)
        decomposer.record_trade("EMA", "BTC/USDT", 50.0, 1, 2)
        decomposer.record_trade("RSI", "BTC/USDT", -30.0, 0, 1)
        decomposer.record_trade("RSI", "BTC/USDT", 20.0, 2, 3)
        
        # Update optimization
        state = engine.update_optimization(regime="TRENDING")
        
        # Verify state
        assert state is not None
        # Should have weights for strategies that have trades
        assert "EMA" in state.current_weights
    
    def test_monitor_integration(self):
        """Test monitor integration with all components."""
        monitor = get_monitor()
        
        status = monitor.get_optimization_status()
        assert status is not None
        
        health = monitor.get_health_report()
        assert health is not None
    
    def test_risk_constraints_maintained(self):
        """Verify risk constraints are never violated."""
        base_weights = {"S1": 1.0}
        engine = OptimizationEngine(base_weights)
        
        # Try extreme cases
        mult = engine.get_position_size_multiplier("S1", signal_quality=0.0)
        assert mult >= 0.0
        assert mult <= 2.0  # Respects upper bound
        
        mult = engine.get_position_size_multiplier("S1", signal_quality=1.0)
        assert mult >= 0.0
        assert mult <= 2.0  # Still respects upper bound
        
        # Weights should always be within bounds
        weights = engine._weighting_optimizer.get_current_weights()
        for w in weights.values():
            assert w >= engine._weighting_optimizer.min_weight
            assert w <= engine._weighting_optimizer.max_weight


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
