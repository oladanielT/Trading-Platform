"""
Example demonstrating adaptive weighting in multi-strategy aggregation.

This shows how strategies with better performance get higher weights over time.
"""

from datetime import datetime, timedelta
import pandas as pd

from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.base import BaseStrategy


class TrendFollower(BaseStrategy):
    """Simple trend following strategy (accurate in trending markets)."""
    
    def __init__(self):
        super().__init__(name="TrendFollower")
    
    def analyze(self, market_data, symbol=None):
        # Simple trend detection
        if len(market_data) < 5:
            return {"action": "hold", "confidence": 0.0}
        
        recent = market_data.tail(5)
        if recent["close"].is_monotonic_increasing:
            return {"action": "buy", "confidence": 0.7}
        elif recent["close"].is_monotonic_decreasing:
            return {"action": "sell", "confidence": 0.7}
        return {"action": "hold", "confidence": 0.3}
    
    def generate_signal(self, market_data, position):
        raise NotImplementedError()


class MeanReversion(BaseStrategy):
    """Simple mean reversion strategy (accurate in ranging markets)."""
    
    def __init__(self):
        super().__init__(name="MeanReversion")
    
    def analyze(self, market_data, symbol=None):
        if len(market_data) < 10:
            return {"action": "hold", "confidence": 0.0}
        
        recent = market_data.tail(10)
        mean = recent["close"].mean()
        current = recent["close"].iloc[-1]
        
        # Buy when below mean, sell when above
        deviation = abs(current - mean) / mean
        if current < mean * 0.98:
            return {"action": "buy", "confidence": min(0.8, deviation * 10)}
        elif current > mean * 1.02:
            return {"action": "sell", "confidence": min(0.8, deviation * 10)}
        return {"action": "hold", "confidence": 0.3}
    
    def generate_signal(self, market_data, position):
        raise NotImplementedError()


def generate_trending_market(length=100):
    """Generate uptrend market data."""
    base = datetime.utcnow() - timedelta(minutes=length)
    data = []
    price = 100.0
    
    for i in range(length):
        price += 0.5  # Steady uptrend
        data.append({
            "timestamp": base + timedelta(minutes=i),
            "open": price,
            "high": price + 0.3,
            "low": price - 0.3,
            "close": price,
            "volume": 1000,
        })
    
    return pd.DataFrame(data)


def generate_ranging_market(length=100):
    """Generate ranging market data."""
    base = datetime.utcnow() - timedelta(minutes=length)
    data = []
    price = 100.0
    
    for i in range(length):
        # Oscillate around 100
        price = 100.0 + 5 * (i % 10 - 5) / 5
        data.append({
            "timestamp": base + timedelta(minutes=i),
            "open": price,
            "high": price + 0.5,
            "low": price - 0.5,
            "close": price,
            "volume": 1000,
        })
    
    return pd.DataFrame(data)


def simulate_adaptive_learning():
    """Simulate adaptive weighting learning process."""
    
    print("=" * 70)
    print("ADAPTIVE WEIGHTING DEMONSTRATION")
    print("=" * 70)
    print()
    
    # Create aggregator with adaptive weighting enabled
    aggregator = MultiStrategyAggregator(
        strategies=[TrendFollower(), MeanReversion()],
        weights={"TrendFollower": 1.0, "MeanReversion": 1.0},
        enable_adaptive_weights=True,
        min_samples_for_adaptation=10,
        adaptation_learning_rate=0.2,  # Faster learning for demo
    )
    
    print("Initial Configuration:")
    print(f"  - Adaptive Weighting: ENABLED")
    print(f"  - Min Samples: 10")
    print(f"  - Learning Rate: 0.2")
    print(f"  - Static Weights: TrendFollower=1.0, MeanReversion=1.0")
    print()
    
    # Phase 1: Trending market (TrendFollower should perform better)
    print("PHASE 1: Trending Market (50 bars)")
    print("-" * 70)
    
    trending_data = generate_trending_market(50)
    
    for i in range(0, 50, 10):
        window = trending_data.iloc[:i+10]
        result = aggregator.analyze(window)
        
        # Simulate outcome: TrendFollower correct, MeanReversion wrong
        aggregator.record_outcome("TrendFollower", was_correct=True)
        aggregator.record_outcome("MeanReversion", was_correct=False)
        
        if i >= 10:  # After min_samples reached
            tf_weight = aggregator._get_weight("TrendFollower")
            mr_weight = aggregator._get_weight("MeanReversion")
            print(f"Bar {i+10:3d}: TrendFollower weight={tf_weight:.3f}, "
                  f"MeanReversion weight={mr_weight:.3f}")
    
    print()
    
    # Show final statistics
    tf_samples = aggregator._total_predictions["TrendFollower"]
    tf_correct = aggregator._correct_predictions["TrendFollower"]
    mr_samples = aggregator._total_predictions["MeanReversion"]
    mr_correct = aggregator._correct_predictions["MeanReversion"]
    
    print(f"Phase 1 Results:")
    print(f"  TrendFollower: {tf_correct}/{tf_samples} correct ({tf_correct/tf_samples*100:.0f}%)")
    print(f"  MeanReversion: {mr_correct}/{mr_samples} correct ({mr_correct/mr_samples*100:.0f}%)")
    print()
    
    # Phase 2: Ranging market (MeanReversion should perform better)
    print("PHASE 2: Ranging Market (50 bars)")
    print("-" * 70)
    
    ranging_data = generate_ranging_market(50)
    
    for i in range(0, 50, 10):
        window = ranging_data.iloc[:i+10]
        result = aggregator.analyze(window)
        
        # Simulate outcome: MeanReversion correct, TrendFollower wrong
        aggregator.record_outcome("TrendFollower", was_correct=False)
        aggregator.record_outcome("MeanReversion", was_correct=True)
        
        tf_weight = aggregator._get_weight("TrendFollower")
        mr_weight = aggregator._get_weight("MeanReversion")
        print(f"Bar {i+10:3d}: TrendFollower weight={tf_weight:.3f}, "
              f"MeanReversion weight={mr_weight:.3f}")
    
    print()
    
    # Final statistics
    tf_samples = aggregator._total_predictions["TrendFollower"]
    tf_correct = aggregator._correct_predictions["TrendFollower"]
    mr_samples = aggregator._total_predictions["MeanReversion"]
    mr_correct = aggregator._correct_predictions["MeanReversion"]
    
    print(f"Phase 2 Results:")
    print(f"  TrendFollower: {tf_correct}/{tf_samples} correct ({tf_correct/tf_samples*100:.0f}%)")
    print(f"  MeanReversion: {mr_correct}/{mr_samples} correct ({mr_correct/mr_samples*100:.0f}%)")
    print()
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("Observations:")
    print("  1. Weights adapted based on performance in each market regime")
    print("  2. TrendFollower weighted higher during trending phase")
    print("  3. MeanReversion weighted higher during ranging phase")
    print("  4. Gradual adjustment prevents overreaction to noise")
    print()
    print("Key Insight:")
    print("  Adaptive weighting allows the system to automatically discover")
    print("  which strategies work best in different market conditions.")
    print()


if __name__ == "__main__":
    simulate_adaptive_learning()
