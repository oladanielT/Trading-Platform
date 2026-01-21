# Adaptive Weighting Quick Reference

## Quick Start

### Enable Adaptive Weighting (Backtest/Paper Only)

```python
from strategies.multi_strategy_aggregator import MultiStrategyAggregator

aggregator = MultiStrategyAggregator(
    strategies=[strategy1, strategy2, strategy3],
    enable_adaptive_weights=True,              # Enable adaptation
    min_samples_for_adaptation=20,             # Wait for 20 samples
    adaptation_learning_rate=0.1,              # 10% learning rate
)
```

### Record Outcomes

```python
# After each prediction, record whether it was correct
aggregator.record_outcome("strategy1", was_correct=True)
aggregator.record_outcome("strategy2", was_correct=False)
```

### Check Current Weights

```python
# Adaptive weight (if available) or static fallback
weight = aggregator._get_weight("strategy1")
```

## Default Behavior (Production Safe)

```python
# Adaptive weighting DISABLED by default
aggregator = MultiStrategyAggregator(strategies=[s1, s2])
# No adaptation occurs, static weights always used
```

## Configuration Presets

### Conservative (Stable Markets)

```python
enable_adaptive_weights=True
min_samples_for_adaptation=50
adaptation_learning_rate=0.05
```

### Balanced (General Purpose)

```python
enable_adaptive_weights=True
min_samples_for_adaptation=20
adaptation_learning_rate=0.1
```

### Aggressive (Fast Adaptation)

```python
enable_adaptive_weights=True
min_samples_for_adaptation=10
adaptation_learning_rate=0.3
```

## Weight Mapping

| Accuracy | Performance Score | Weight Multiplier |
| -------- | ----------------- | ----------------- |
| 0%       | 0.5               | 0.5× static       |
| 25%      | 0.75              | 0.75× static      |
| 50%      | 1.0               | 1.0× static       |
| 75%      | 1.25              | 1.25× static      |
| 100%     | 1.5               | 1.5× static       |

## Key Safety Features

✅ **Disabled by default** - No impact unless explicitly enabled  
✅ **Minimum samples** - Requires sufficient data before adapting  
✅ **Gradual adjustment** - Learning rate prevents sudden changes  
✅ **Static fallback** - Uses static weights when insufficient data  
✅ **Independent tracking** - Each strategy tracked separately

## Common Patterns

### Backtesting Loop

```python
for i, bar in enumerate(historical_data):
    # Get aggregated signal
    result = aggregator.analyze(bar)

    # Execute trade
    executed_trade = execute(result)

    # After knowing outcome, record it
    if i > 0:  # After first bar
        for contrib in result["contributions"]:
            was_correct = evaluate_prediction(contrib, actual_outcome)
            aggregator.record_outcome(contrib["strategy"], was_correct)
```

### Viewing Adaptive Info

```python
result = aggregator.analyze(market_data)

for contrib in result["contributions"]:
    if "adaptive_info" in contrib:
        print(f"{contrib['strategy']}:")
        print(f"  Weight: {contrib['weight']}")
        print(f"  Samples: {contrib['adaptive_info']['samples']}")
        print(f"  Accuracy: {contrib['adaptive_info']['accuracy']:.2%}")
        print(f"  Using Adaptive: {contrib['adaptive_info']['using_adaptive']}")
```

## Testing

```bash
# Run adaptive weighting tests
pytest tests/test_adaptive_weighting.py -v

# Run demo
PYTHONPATH=. python3 examples/adaptive_weighting_demo.py
```

## Files

- **Implementation**: `strategies/multi_strategy_aggregator.py`
- **Tests**: `tests/test_adaptive_weighting.py` (16 tests)
- **Documentation**: `ADAPTIVE_WEIGHTING.md`
- **Demo**: `examples/adaptive_weighting_demo.py`

## Logging

### Adaptation Logs

```
INFO [AdaptiveWeights] strategy=FVGStrategy samples=25 accuracy=72.00%
     static_weight=1.000 adaptive_weight=1.122
```

### Contribution Logs

```
INFO [MultiStrategy] final=buy 0.650 | contributions=[
    {'strategy': 'FVGStrategy', 'weight': 1.122,
     'adaptive_info': {'samples': 25, 'accuracy': 0.72, 'using_adaptive': True}},
    ...
]
```

---

**Version**: Phase 8 Adaptive Weighting  
**Status**: ✅ Complete - 136/136 tests passing  
**Safety**: Opt-in only, production safe
