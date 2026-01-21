"""
Tests for adaptive weighting in MultiStrategyAggregator.
"""

from datetime import datetime, timedelta
import pandas as pd
import pytest

from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.base import BaseStrategy, PositionState, Signal


def _market(rows=50, start=100.0, step=1.0):
    """Generate simple market data."""
    base = datetime.utcnow() - timedelta(minutes=rows)
    data = []
    price = start
    for i in range(rows):
        price += step
        data.append(
            {
                "timestamp": base + timedelta(minutes=i),
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
                "volume": 1_000 + i,
            }
        )
    return pd.DataFrame(data)


class MockStrategy(BaseStrategy):
    """Mock strategy for testing."""
    
    def __init__(self, name, action="buy", confidence=0.5):
        super().__init__(name=name)
        self.action = action
        self.confidence = confidence
    
    def analyze(self, market_data, symbol=None):
        return {"action": self.action, "confidence": self.confidence}
    
    def generate_signal(self, market_data, position):
        return Signal(
            timestamp=datetime.utcnow(),
            action=self.action,
            confidence=self.confidence,
            rationale="mock",
        )


class TestAdaptiveWeightingDisabled:
    """Test that adaptive weighting is opt-in and defaults to disabled."""
    
    def test_default_initialization_no_adaptive_weights(self):
        """Verify adaptive weighting is disabled by default."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        strat2 = MockStrategy("strat2", "sell", 0.4)
        
        agg = MultiStrategyAggregator([strat1, strat2])
        
        assert agg.enable_adaptive_weights is False
        assert len(agg._adaptive_weights) == 0
        assert len(agg._total_predictions) == 0
    
    def test_disabled_adaptive_weights_use_static_weights(self):
        """Verify static weights used when adaptive disabled."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        strat2 = MockStrategy("strat2", "buy", 0.4)
        
        agg = MultiStrategyAggregator(
            [strat1, strat2],
            weights={"strat1": 2.0, "strat2": 1.0},
            enable_adaptive_weights=False,
        )
        
        result = agg.analyze(_market())
        
        assert result is not None
        assert result["action"] == "buy"
        # Should use static weights: 0.6*2.0 + 0.4*1.0 = 1.6 / 3.0 = 0.533
        assert 0.5 < result["confidence"] < 0.6
    
    def test_record_outcome_no_effect_when_disabled(self):
        """Verify record_outcome has no effect when adaptive disabled."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator([strat1], enable_adaptive_weights=False)
        
        # Record outcomes should have no effect
        agg.record_outcome("strat1", True)
        agg.record_outcome("strat1", True)
        
        assert agg._total_predictions["strat1"] == 0
        assert agg._correct_predictions["strat1"] == 0


class TestAdaptiveWeightingEnabled:
    """Test adaptive weighting when enabled."""
    
    def test_initialization_with_adaptive_enabled(self):
        """Verify adaptive weighting initializes correctly."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            enable_adaptive_weights=True,
            min_samples_for_adaptation=10,
            adaptation_learning_rate=0.2,
        )
        
        assert agg.enable_adaptive_weights is True
        assert agg.min_samples_for_adaptation == 10
        assert agg.adaptation_learning_rate == 0.2
    
    def test_fallback_to_static_weights_insufficient_samples(self):
        """Verify fallback to static weights when samples < min_samples."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        strat2 = MockStrategy("strat2", "buy", 0.4)
        
        agg = MultiStrategyAggregator(
            [strat1, strat2],
            weights={"strat1": 2.0, "strat2": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
        )
        
        # Record only 10 outcomes (below threshold)
        for _ in range(10):
            agg.record_outcome("strat1", True)
        
        # Should still use static weights
        weight = agg._get_weight("strat1")
        assert weight == 2.0  # Static weight
    
    def test_adaptive_weights_after_sufficient_samples(self):
        """Verify adaptive weights applied after reaching min_samples."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            weights={"strat1": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=1.0,  # Full adjustment for testing
        )
        
        # Record 20 correct outcomes (100% accuracy)
        for _ in range(20):
            agg.record_outcome("strat1", True)
        
        # Should use adaptive weight now
        weight = agg._get_weight("strat1")
        # 100% accuracy -> performance_score = 1.5
        # adaptive_weight = 1.0 * 1.5 = 1.5
        assert weight > 1.0
        assert abs(weight - 1.5) < 0.01
    
    def test_adaptive_weights_decrease_with_poor_performance(self):
        """Verify adaptive weights decrease with poor accuracy."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            weights={"strat1": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=1.0,
        )
        
        # Record 20 incorrect outcomes (0% accuracy)
        for _ in range(20):
            agg.record_outcome("strat1", False)
        
        # Should decrease weight
        weight = agg._get_weight("strat1")
        # 0% accuracy -> performance_score = 0.5
        # adaptive_weight = 1.0 * 0.5 = 0.5
        assert weight < 1.0
        assert abs(weight - 0.5) < 0.01
    
    def test_adaptive_weights_50_percent_accuracy_neutral(self):
        """Verify 50% accuracy keeps weight neutral."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            weights={"strat1": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=1.0,
        )
        
        # Record 10 correct, 10 incorrect (50% accuracy)
        for _ in range(10):
            agg.record_outcome("strat1", True)
        for _ in range(10):
            agg.record_outcome("strat1", False)
        
        # Should keep weight neutral
        weight = agg._get_weight("strat1")
        # 50% accuracy -> performance_score = 1.0
        # adaptive_weight = 1.0 * 1.0 = 1.0
        assert abs(weight - 1.0) < 0.01


class TestAdaptiveWeightingLearningRate:
    """Test gradual adjustment via learning rate."""
    
    def test_learning_rate_gradual_adjustment(self):
        """Verify learning rate causes gradual weight adjustment."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            weights={"strat1": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=0.1,  # 10% learning rate
        )
        
        # Record 20 perfect outcomes
        for _ in range(20):
            agg.record_outcome("strat1", True)
        
        # First update: 0.9 * 1.0 + 0.1 * 1.5 = 1.05
        first_weight = agg._get_weight("strat1")
        assert 1.0 < first_weight < 1.5
        
        # Record 10 more perfect outcomes
        for _ in range(10):
            agg.record_outcome("strat1", True)
        
        # Should move closer to 1.5 but not reach it immediately
        second_weight = agg._get_weight("strat1")
        assert second_weight > first_weight
        assert second_weight < 1.5


class TestAdaptiveWeightingMultiStrategy:
    """Test adaptive weighting with multiple strategies."""
    
    def test_independent_adaptation_per_strategy(self):
        """Verify each strategy has independent adaptive weight."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        strat2 = MockStrategy("strat2", "sell", 0.4)
        
        agg = MultiStrategyAggregator(
            [strat1, strat2],
            weights={"strat1": 1.0, "strat2": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=1.0,
        )
        
        # strat1: 100% accuracy
        for _ in range(20):
            agg.record_outcome("strat1", True)
        
        # strat2: 0% accuracy
        for _ in range(20):
            agg.record_outcome("strat2", False)
        
        # Weights should differ
        weight1 = agg._get_weight("strat1")
        weight2 = agg._get_weight("strat2")
        
        assert weight1 > 1.0
        assert weight2 < 1.0
        assert weight1 > weight2


class TestAdaptiveWeightingContributions:
    """Test adaptive weight info in contributions."""
    
    def test_contributions_include_adaptive_info_when_enabled(self):
        """Verify contributions include adaptive weight details."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        strat2 = MockStrategy("strat2", "buy", 0.4)
        
        agg = MultiStrategyAggregator(
            [strat1, strat2],
            enable_adaptive_weights=True,
            min_samples_for_adaptation=10,
        )
        
        # Record some outcomes
        for _ in range(15):
            agg.record_outcome("strat1", True)
        
        result = agg.analyze(_market())
        
        assert result is not None
        contrib = result["contributions"][0]
        
        # Should have adaptive_info
        assert "adaptive_info" in contrib
        assert contrib["adaptive_info"]["samples"] == 15
        assert contrib["adaptive_info"]["accuracy"] > 0.99
        assert contrib["adaptive_info"]["using_adaptive"] is True
    
    def test_contributions_show_fallback_when_insufficient_samples(self):
        """Verify contributions show using_adaptive=False when below threshold."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
        )
        
        # Record only 5 outcomes
        for _ in range(5):
            agg.record_outcome("strat1", True)
        
        result = agg.analyze(_market())
        
        contrib = result["contributions"][0]
        assert contrib["adaptive_info"]["samples"] == 5
        assert contrib["adaptive_info"]["using_adaptive"] is False


class TestAdaptiveWeightingLogging:
    """Test logging of adaptive weight adjustments."""
    
    def test_weight_adjustment_logged(self, caplog):
        """Verify weight adjustments are logged."""
        import logging
        caplog.set_level(logging.INFO)
        
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            weights={"strat1": 1.0},
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=1.0,
        )
        
        # Record outcomes to trigger adaptation
        for _ in range(20):
            agg.record_outcome("strat1", True)
        
        # Check for adaptive weight log
        adaptive_logs = [
            rec for rec in caplog.records
            if "[AdaptiveWeights]" in rec.message and "strat1" in rec.message
        ]
        
        assert len(adaptive_logs) > 0
        # Should log accuracy, samples, weights
        log_msg = adaptive_logs[0].message
        assert "samples=20" in log_msg
        assert "accuracy=100.00%" in log_msg
        assert "adaptive_weight" in log_msg


class TestAdaptiveWeightingEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_samples_returns_neutral_score(self):
        """Verify performance score is 1.0 with zero samples."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            enable_adaptive_weights=True,
        )
        
        score = agg._calculate_performance_score("strat1")
        assert score == 1.0
    
    def test_unknown_strategy_uses_default_weight(self):
        """Verify unknown strategy falls back to default weight."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator([strat1], enable_adaptive_weights=True)
        
        # Get weight for non-existent strategy
        weight = agg._get_weight("unknown_strat")
        assert weight == 1.0  # Default weight
    
    def test_static_weights_respected_as_baseline(self):
        """Verify static weights used as baseline for adaptation."""
        strat1 = MockStrategy("strat1", "buy", 0.6)
        
        agg = MultiStrategyAggregator(
            [strat1],
            weights={"strat1": 2.0},  # Static weight 2.0
            enable_adaptive_weights=True,
            min_samples_for_adaptation=20,
            adaptation_learning_rate=1.0,
        )
        
        # 100% accuracy
        for _ in range(20):
            agg.record_outcome("strat1", True)
        
        # Adaptive weight = 2.0 * 1.5 = 3.0
        weight = agg._get_weight("strat1")
        assert abs(weight - 3.0) < 0.01
