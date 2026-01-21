# Regime-Based Meta-Strategy

## Overview

The regime-based strategy system provides intelligent market condition filtering for trading strategies. It classifies market regimes and controls when underlying strategies can generate actionable signals, reducing overtrading and improving risk-adjusted returns.

## Architecture

### Components

1. **RegimeDetector** (`strategies/regime_detector.py`)

   - Stateless market regime classifier
   - Analyzes price behavior, volatility, and trend characteristics
   - No machine learning, fully deterministic and explainable

2. **RegimeBasedStrategy** (`strategies/regime_strategy.py`)
   - Meta-strategy that wraps existing strategies
   - Filters signals based on market regime
   - Delegates to underlying strategy only in favorable conditions

## Market Regimes

### TRENDING

- **Characteristics**: Clear directional movement with strong trend
- **Indicators**: High trend strength, good price efficiency
- **Strategy Behavior**: Allow trend-following strategies (EMA, momentum, etc.)

### RANGING

- **Characteristics**: Price oscillating within a range
- **Indicators**: Low trend strength, moderate volatility
- **Strategy Behavior**: Typically HOLD (trend strategies perform poorly)

### VOLATILE

- **Characteristics**: High volatility with unpredictable movements
- **Indicators**: Volatility above high threshold, rapid price changes
- **Strategy Behavior**: HOLD (high risk, low signal quality)

### QUIET

- **Characteristics**: Low volatility with minimal price movement
- **Indicators**: Very low volatility, weak trends
- **Strategy Behavior**: HOLD (low opportunity, potential false signals)

## Regime Detection Metrics

### 1. Realized Volatility

- Standard deviation of log returns (annualized)
- Measures price variability
- Thresholds: Low < 0.01, High > 0.03

### 2. Trend Strength

- Price slope normalized by volatility
- Measures directional consistency
- Thresholds: Weak < 0.5, Strong > 2.0

### 3. Price Efficiency

- Net movement / Total movement
- Measures trend quality (0 = random, 1 = perfect trend)
- Threshold: > 0.5 for strong trends

### 4. ATR (Average True Range)

- Normalized by price level
- Provides volatility context
- Used for confirming volatility regime

## Usage

### Basic Usage

```python
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.regime_detector import MarketRegime

# Create regime-based strategy (defaults to EMA trend underlying)
strategy = RegimeBasedStrategy(
    allowed_regimes=[MarketRegime.TRENDING],
    regime_confidence_threshold=0.5
)

# Generate signal
signal = strategy.generate_signal(market_data, position_state)

print(f"Signal: {signal.signal.value}")
print(f"Confidence: {signal.confidence}")
print(f"Regime: {signal.metadata['regime']}")
```

### Custom Underlying Strategy

```python
from strategies.ema_trend import EMATrendStrategy

# Create custom underlying strategy
ema_strategy = EMATrendStrategy(
    fast_period=20,
    slow_period=50,
    signal_confidence=0.8
)

# Wrap with regime filter
regime_strategy = RegimeBasedStrategy(
    underlying_strategy=ema_strategy,
    allowed_regimes=[MarketRegime.TRENDING]
)
```

### Custom Regime Detection Parameters

```python
from strategies.regime_detector import RegimeDetector

# Create custom regime detector
detector = RegimeDetector(
    lookback_period=150,
    volatility_threshold_high=0.04,
    volatility_threshold_low=0.008,
    trend_threshold_strong=2.5
)

# Use in strategy
strategy = RegimeBasedStrategy(
    regime_detector=detector,
    allowed_regimes=[MarketRegime.TRENDING, MarketRegime.QUIET]
)
```

### Multiple Allowed Regimes

```python
# Allow both trending and ranging
strategy = RegimeBasedStrategy(
    allowed_regimes=[
        MarketRegime.TRENDING,
        MarketRegime.RANGING
    ]
)

# Or use strings
strategy.set_allowed_regimes(['trending', 'ranging'])
```

### Monitoring Regime Without Trading

```python
# Get current regime status
status = strategy.get_regime_status(market_data)

print(f"Current Regime: {status['regime']}")
print(f"Is Allowed: {status['is_allowed']}")
print(f"Metrics: {status['metrics']}")
```

## Signal Metadata

The RegimeBasedStrategy enriches signal metadata with comprehensive information:

```python
{
    'regime': 'trending',
    'regime_confidence': 0.85,
    'regime_metrics': {
        'volatility': 0.025,
        'trend_strength': 2.3,
        'price_efficiency': 0.62,
        'atr_pct': 1.8
    },
    'regime_explanation': 'Strong directional trend detected...',
    'decision': 'delegate_to_underlying_strategy',
    'decision_reason': 'Regime trending is favorable...',
    'underlying_strategy': 'EMATrendStrategy',
    'underlying_signal': 'BUY',
    'underlying_confidence': 0.75,
    'confidence_adjusted': True,
    'allowed_regimes': ['trending']
}
```

## Configuration Options

### RegimeDetector Parameters

| Parameter                   | Default | Description                    |
| --------------------------- | ------- | ------------------------------ |
| `lookback_period`           | 100     | Number of candles to analyze   |
| `volatility_threshold_high` | 0.03    | High volatility threshold (3%) |
| `volatility_threshold_low`  | 0.01    | Low volatility threshold (1%)  |
| `trend_threshold_strong`    | 2.0     | Strong trend threshold         |
| `trend_threshold_weak`      | 0.5     | Weak trend threshold           |
| `efficiency_threshold`      | 0.5     | Price efficiency threshold     |

### RegimeBasedStrategy Parameters

| Parameter                               | Default          | Description                           |
| --------------------------------------- | ---------------- | ------------------------------------- |
| `underlying_strategy`                   | EMATrendStrategy | Strategy to delegate to               |
| `allowed_regimes`                       | [TRENDING]       | Regimes where signals allowed         |
| `regime_confidence_threshold`           | 0.5              | Min regime classification confidence  |
| `reduce_confidence_in_marginal_regimes` | True             | Scale confidence by regime confidence |

## Integration with Main Trading Loop

Update `main.py` to use RegimeBasedStrategy:

```python
from strategies.regime_strategy import RegimeBasedStrategy

# In _setup_strategy method:
if strategy_name == 'regime_based':
    self.strategy = RegimeBasedStrategy(
        allowed_regimes=strategy_config.get('allowed_regimes', ['trending']),
        regime_confidence_threshold=strategy_config.get('regime_confidence_threshold', 0.5)
    )
```

Update `config.yaml`:

```yaml
strategy:
  name: regime_based
  allowed_regimes:
    - trending
  regime_confidence_threshold: 0.5
  reduce_confidence_in_marginal_regimes: true
```

## Benefits

### 1. Reduced Overtrading

- Filters out signals in unfavorable market conditions
- Avoids choppy/ranging markets that hurt trend strategies

### 2. Improved Risk Management

- Stays out of highly volatile markets
- Reduces drawdowns during uncertain conditions

### 3. Better Risk-Adjusted Returns

- Higher win rate by only trading in favorable regimes
- Lower transaction costs from fewer trades

### 4. Fully Explainable

- No black-box ML models
- Clear metrics and decision logic
- Auditable regime classifications

### 5. Production-Safe

- Stateless and deterministic
- No future data leakage
- Comprehensive logging and metadata

## Testing

Run the example file to see regime detection in action:

```bash
cd /home/oladan/Trading-Platform
python examples/regime_strategy_example.py
```

## Performance Considerations

### Lookback Period

- Longer periods (150-200): More stable, slower to adapt
- Shorter periods (50-100): More responsive, may whipsaw

### Volatility Thresholds

- Adjust based on asset class and timeframe
- Crypto: Higher thresholds (0.03-0.05)
- Forex: Lower thresholds (0.01-0.02)

### Allowed Regimes

- Conservative: Only TRENDING
- Moderate: TRENDING + QUIET
- Aggressive: All except VOLATILE

## Limitations

1. **Lagging Indicator**: Regime detection uses historical data, may lag regime changes
2. **Threshold Sensitivity**: Performance depends on proper threshold calibration
3. **Asset-Specific**: Optimal parameters vary by asset and timeframe
4. **No Prediction**: Detects current regime, doesn't predict future regimes

## Future Enhancements

Potential improvements (maintaining no-ML constraint):

1. **Adaptive Thresholds**: Auto-adjust based on historical regime distributions
2. **Regime Persistence**: Consider regime stability/duration
3. **Multi-Timeframe**: Combine regime detection across timeframes
4. **Regime-Specific Strategies**: Different strategies for different regimes
5. **Transition Detection**: Identify regime changes early

## References

- Volatility estimation: Standard deviation of log returns
- Trend strength: Linear regression with normalization
- Price efficiency: Similar to Efficiency Ratio (ER) indicator
- ATR: Wilder's Average True Range

---

**Status**: Production-ready, fully tested, no dependencies on risk/execution modules

**Compliance**: Follows SYSTEM_SPEC.md contract, stateless, no trading logic
