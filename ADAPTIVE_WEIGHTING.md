# Adaptive Weighting for Multi-Strategy Aggregation

## Overview

The `MultiStrategyAggregator` now supports optional **adaptive weighting**, which automatically adjusts strategy weights based on historical performance. This feature is **opt-in only** and does not affect live execution by default, ensuring safe backtesting and validation before deployment.

## Key Features

### 1. Opt-In Design

- **Default Behavior**: Adaptive weighting is **disabled** by default
- **No Impact on Live**: Must explicitly enable; existing code unaffected
- **Backward Compatible**: All existing functionality preserved

### 2. Performance-Based Adaptation

- Tracks accuracy of each strategy's predictions
- Adjusts weights based on historical success rate
- Maps accuracy to weight multipliers:
  - **0% accuracy** → 0.5× weight (poor performer downweighted)
  - **50% accuracy** → 1.0× weight (neutral)
  - **100% accuracy** → 1.5× weight (strong performer upweighted)

### 3. Intelligent Fallback

- Requires minimum sample count before activating (default: 20)
- Falls back to static weights if insufficient data
- Prevents premature conclusions from small samples

### 4. Gradual Adjustment

- Uses learning rate for smooth weight transitions (default: 0.1)
- Prevents sudden weight swings from noise
- Allows system to adapt to changing market conditions

### 5. Comprehensive Logging

- Logs weight adjustments with accuracy metrics
- Includes adaptive info in contribution details
- Clear visibility into adaptation decisions

## Configuration

### Initialization Parameters

```python
MultiStrategyAggregator(
    strategies: List[BaseStrategy],
    weights: Optional[Dict[str, float]] = None,
    enable_adaptive_weights: bool = False,              # Enable adaptation
    min_samples_for_adaptation: int = 20,               # Min samples before adapting
    adaptation_learning_rate: float = 0.1,              # Learning rate (0.0-1.0)
)
```

### Parameter Details

- **`enable_adaptive_weights`**: Master switch for adaptive weighting

  - `False` (default): Use static weights only
  - `True`: Enable adaptive weight adjustment

- **`min_samples_for_adaptation`**: Minimum prediction count before adaptation

  - Default: 20
  - Higher values = more conservative (require more data)
  - Lower values = more aggressive (adapt faster)

- **`adaptation_learning_rate`**: Speed of weight adjustment
  - Range: 0.0 (no learning) to 1.0 (instant adjustment)
  - Default: 0.1 (gradual 10% adjustment per update)
  - Formula: `new_weight = old_weight * (1 - lr) + target_weight * lr`

## Usage Examples

### Example 1: Basic Adaptive Weighting

```python
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.pattern_strategies import FVGStrategy, ABCStrategy

# Create aggregator with adaptive weighting
aggregator = MultiStrategyAggregator(
    strategies=[FVGStrategy(), ABCStrategy()],
    weights={"FVGStrategy": 1.0, "ABCStrategy": 1.0},
    enable_adaptive_weights=True,
    min_samples_for_adaptation=20,
    adaptation_learning_rate=0.1,
)

# Use in backtesting loop
for bar in historical_data:
    result = aggregator.analyze(bar, symbol="BTC/USDT")

    # Execute trade based on result
    # ... trade execution logic ...

    # After knowing outcome, record it
    if trade_was_profitable:
        aggregator.record_outcome("FVGStrategy", was_correct=True)
        aggregator.record_outcome("ABCStrategy", was_correct=True)
    else:
        aggregator.record_outcome("FVGStrategy", was_correct=False)
        aggregator.record_outcome("ABCStrategy", was_correct=False)
```

### Example 2: Conservative Adaptation

```python
# Higher threshold, slower learning for stable markets
aggregator = MultiStrategyAggregator(
    strategies=[strat1, strat2, strat3],
    enable_adaptive_weights=True,
    min_samples_for_adaptation=50,    # Wait for 50 samples
    adaptation_learning_rate=0.05,     # Very gradual adjustment
)
```

### Example 3: Aggressive Adaptation

```python
# Lower threshold, faster learning for rapid adaptation
aggregator = MultiStrategyAggregator(
    strategies=[strat1, strat2],
    enable_adaptive_weights=True,
    min_samples_for_adaptation=10,    # Adapt after 10 samples
    adaptation_learning_rate=0.3,     # Faster adjustment
)
```

### Example 4: Disabled (Default Behavior)

```python
# No adaptive weighting - uses static weights only
aggregator = MultiStrategyAggregator(
    strategies=[strat1, strat2],
    weights={"strat1": 2.0, "strat2": 1.0},
    # enable_adaptive_weights defaults to False
)

# record_outcome calls have no effect
aggregator.record_outcome("strat1", True)  # No-op
```

## Weight Calculation Formula

### Performance Score Calculation

```python
accuracy = correct_predictions / total_predictions

# Map accuracy to performance score [0.5, 1.5]
performance_score = 0.5 + accuracy
```

### Adaptive Weight Calculation

```python
target_weight = static_weight * performance_score

# Gradual adjustment using learning rate
new_adaptive_weight = (
    current_adaptive_weight * (1 - learning_rate) +
    target_weight * learning_rate
)
```

### Weight Selection Logic

```python
if enable_adaptive_weights and total_samples >= min_samples_for_adaptation:
    return adaptive_weight
else:
    return static_weight  # Fallback
```

## Log Output Examples

### Adaptive Weight Adjustment Logs

```
INFO [AdaptiveWeights] strategy=FVGStrategy samples=25 accuracy=72.00% static_weight=1.000 adaptive_weight=1.122
INFO [AdaptiveWeights] strategy=ABCStrategy samples=25 accuracy=48.00% static_weight=1.000 adaptive_weight=0.980
```

### Contribution Details with Adaptive Info

```python
{
    "action": "buy",
    "confidence": 0.65,
    "contributions": [
        {
            "strategy": "FVGStrategy",
            "action": "buy",
            "confidence": 0.7,
            "weight": 1.122,
            "adaptive_info": {
                "static_weight": 1.0,
                "samples": 25,
                "accuracy": 0.72,
                "using_adaptive": True
            }
        },
        {
            "strategy": "ABCStrategy",
            "action": "buy",
            "confidence": 0.6,
            "weight": 0.980,
            "adaptive_info": {
                "static_weight": 1.0,
                "samples": 25,
                "accuracy": 0.48,
                "using_adaptive": True
            }
        }
    ]
}
```

## Testing

### Test Coverage

All tests in `tests/test_adaptive_weighting.py`:

- **TestAdaptiveWeightingDisabled** (3 tests)

  - Default disabled behavior
  - Static weights when disabled
  - record_outcome no-op when disabled

- **TestAdaptiveWeightingEnabled** (5 tests)

  - Initialization with adaptive enabled
  - Fallback to static weights with insufficient samples
  - Weight increase with high accuracy
  - Weight decrease with poor performance
  - Neutral weight at 50% accuracy

- **TestAdaptiveWeightingLearningRate** (1 test)

  - Gradual adjustment via learning rate

- **TestAdaptiveWeightingMultiStrategy** (1 test)

  - Independent adaptation per strategy

- **TestAdaptiveWeightingContributions** (2 tests)

  - Contributions include adaptive info
  - Fallback indicator when below threshold

- **TestAdaptiveWeightingLogging** (1 test)

  - Weight adjustments logged correctly

- **TestAdaptiveWeightingEdgeCases** (3 tests)
  - Zero samples neutral score
  - Unknown strategy default weight
  - Static weights as baseline

### Running Tests

```bash
# Run adaptive weighting tests only
pytest tests/test_adaptive_weighting.py -v

# Run full test suite (136 tests)
pytest tests/ -v
```

## Best Practices

### For Backtesting

1. **Enable Adaptive Weighting**: Use to discover which strategies perform best

   ```python
   enable_adaptive_weights=True
   min_samples_for_adaptation=20
   adaptation_learning_rate=0.1
   ```

2. **Track Results**: Monitor weight evolution in logs
3. **Validate**: Compare adaptive vs static performance

### For Paper Trading

1. **Start Conservative**: Use higher thresholds

   ```python
   min_samples_for_adaptation=50
   adaptation_learning_rate=0.05
   ```

2. **Monitor Closely**: Review adaptive_info in contributions
3. **Build History**: Accumulate sufficient samples before live

### For Live Trading

1. **Default to Disabled**: Keep `enable_adaptive_weights=False`
2. **Manual Override**: Set static weights based on backtest findings
3. **Gradual Rollout**: Only enable after extensive paper trading validation

## Integration with RegimeBasedStrategy

The adaptive aggregator works seamlessly with regime-based strategy:

```python
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.multi_strategy_aggregator import MultiStrategyAggregator
from strategies.pattern_strategies import FVGStrategy, ABCStrategy

# Create adaptive aggregator
aggregator = MultiStrategyAggregator(
    strategies=[FVGStrategy(), ABCStrategy()],
    enable_adaptive_weights=True,  # Enable for backtesting
    min_samples_for_adaptation=20,
)

# Use as underlying strategy in regime-based system
regime_strategy = RegimeBasedStrategy(
    allowed_regimes=["TRENDING"],
    enable_readiness_gate=True,
    enable_readiness_scaling=True,
    enable_pattern_strategies=True,
    pattern_aggregator=aggregator,
    trading_mode="backtest",
)

# Aggregator automatically receives symbol and market data
result = regime_strategy.analyze(market_data, symbol="BTC/USDT")

# Record outcomes for adaptation (in backtest loop)
# ... after trade completes ...
aggregator.record_outcome("FVGStrategy", was_correct=True)
```

## Performance Considerations

### Memory Usage

- Tracks counts per strategy (minimal overhead)
- No historical data storage (only aggregated stats)
- Suitable for long-running processes

### CPU Usage

- Weight updates: O(N) where N = number of strategies
- Negligible overhead (simple arithmetic)
- Updates only on record_outcome calls

### Accuracy of Adaptation

- Requires meaningful outcome feedback
- Works best with clear win/loss signals
- Consider market regime changes when interpreting

## Limitations & Future Enhancements

### Current Limitations

1. Binary outcomes only (correct/incorrect)
2. Equal weighting of all historical samples
3. No time-based decay of old samples
4. No cross-strategy correlation analysis

### Potential Enhancements

- [ ] Weighted outcomes (magnitude of error)
- [ ] Time-based sample decay (recent samples weighted more)
- [ ] Regime-specific adaptation (different weights per regime)
- [ ] Confidence-based outcome weighting
- [ ] Correlation detection between strategies

## API Reference

### Methods

#### `record_outcome(strategy_name: str, was_correct: bool)`

Record outcome for a strategy prediction.

**Parameters:**

- `strategy_name`: Name of the strategy
- `was_correct`: Whether the prediction was correct

**Example:**

```python
aggregator.record_outcome("FVGStrategy", was_correct=True)
```

#### `_calculate_performance_score(strat_name: str) -> float`

Calculate performance score based on accuracy (internal method).

**Returns:** Float in range [0.5, 1.5]

#### `_update_adaptive_weights()`

Update all adaptive weights based on current performance (internal method).

#### `_get_weight(strat_name: str) -> float`

Get current weight for strategy (adaptive if available, else static).

**Returns:** Float weight value

---

**Status**: ✅ Complete and tested  
**Test Coverage**: 16/16 tests passing  
**Integration**: Compatible with all existing functionality  
**Safety**: Opt-in only, no impact on live execution by default
