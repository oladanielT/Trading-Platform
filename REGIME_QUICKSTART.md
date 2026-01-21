# RegimeBasedStrategy Quick Start Guide

## Overview

The RegimeBasedStrategy is now fully integrated into the trading platform. It automatically detects market regimes (TRENDING, RANGING, VOLATILE, QUIET) and only generates signals when market conditions are favorable.

## Configuration

Edit `core/config.yaml`:

```yaml
strategy:
  # Select regime-based strategy
  name: regime_based

  # Which regimes allow trading
  allowed_regimes:
    - trending # Trade in trending markets
    # - ranging   # Uncomment to allow range-bound trading
    # - volatile  # Uncomment to allow volatile markets
    # - quiet     # Uncomment to allow quiet markets

  # Regime detection parameters
  regime_confidence_threshold: 0.5 # Min confidence to trust regime (0.0-1.0)
  reduce_confidence_in_marginal_regimes: true # Scale signal confidence by regime confidence

  # Underlying strategy (delegates to this when regime is favorable)
  underlying_strategy:
    name: ema_trend
    fast_period: 20
    slow_period: 50
    signal_confidence: 0.75
```

## Running the Platform

### Paper Trading (Simulated)

```bash
python3 main.py
```

### Backtesting

```bash
# Run with test data
pytest tests/test_backtest_phase2_regime.py -v

# Run integration tests
pytest tests/test_integration_regime.py -v

# Run all tests
pytest tests/ -v
```

### Via CLI (Backtest Mode)

```bash
# Start a backtest
python3 -m interfaces.cli run \
  --name my_regime_test \
  --strategy strategies.regime_strategy:RegimeBasedStrategy \
  --bars data/historical_data.json

# Check status
python3 -m interfaces.cli status my_regime_test

# View results
python3 -m interfaces.cli results my_regime_test
```

## Understanding Regime Detection

### Market Regimes

1. **TRENDING** - Strong directional movement

   - High trend strength
   - Consistent price direction
   - Good for trend-following strategies
   - **Default: ALLOWED**

2. **RANGING** - Sideways oscillation

   - Low trend strength
   - Price contained in range
   - Good for mean-reversion strategies
   - **Default: NOT ALLOWED**

3. **VOLATILE** - Large unpredictable swings

   - High volatility
   - Low price efficiency
   - Risky for most strategies
   - **Default: NOT ALLOWED**

4. **QUIET** - Minimal movement
   - Low volatility
   - Low opportunity
   - Risk of false signals
   - **Default: NOT ALLOWED**

### How It Works

1. **Market Data** → RegimeDetector analyzes recent bars
2. **Regime Classification** → TRENDING/RANGING/VOLATILE/QUIET
3. **Confidence Score** → How certain is the classification (0.0-1.0)
4. **Decision Logic**:
   - If regime NOT in `allowed_regimes` → Return HOLD
   - If regime confidence < threshold → Return HOLD
   - If regime IS allowed → Delegate to underlying strategy
5. **Signal Adjustment** (optional):
   - Scale signal confidence by regime confidence
   - Lower regime confidence → smaller positions

## Log Output Examples

### Favorable Regime (Signal Generated)

```
[STRATEGY] Generating signal using RegimeBasedStrategy
[STRATEGY] Signal: BUY (confidence: 0.75) | Regime: trending (conf: 0.92)
[RISK] Position size: 0.085432 (approved: True)
[EXECUTION] Order filled: 42150.50 x 0.085432
```

### Unfavorable Regime (HOLD)

```
[STRATEGY] Generating signal using RegimeBasedStrategy
[STRATEGY] Signal: HOLD (confidence: 0.00) | Regime: volatile (conf: 0.78)
[STRATEGY] HOLD signal - no action required
```

### Low Regime Confidence (HOLD)

```
[STRATEGY] Signal: HOLD (confidence: 0.00) | Regime: trending (conf: 0.42)
[RegimeDetector] Low regime confidence (0.42 < 0.5), treating as uncertain
```

## Monitoring & Metrics

### What Gets Logged

Every signal includes:

- Current regime type
- Regime confidence score
- Decision rationale
- Underlying strategy signal (if delegated)
- Signal confidence (original and adjusted)

### Accessing Metrics

**Via API:**

```bash
# Get all runs
curl http://localhost:8000/runs

# Get specific run with regime metadata
curl http://localhost:8000/runs/my_regime_test
```

**Via Python:**

```python
from monitoring.metrics import MetricsCollector

metrics = MetricsCollector()
overall = metrics.get_overall_metrics()

print(f"Total trades: {overall['total_trades']}")
print(f"Win rate: {overall['win_rate']:.2%}")
print(f"Total PnL: ${overall['total_pnl']:.2f}")
```

## Tuning Parameters

### Regime Detection Sensitivity

In `strategies/regime_detector.py`, you can adjust:

```python
RegimeDetector(
    lookback_period=100,              # Bars to analyze (default: 100)
    volatility_threshold_high=0.03,   # High vol cutoff (3%)
    volatility_threshold_low=0.01,    # Low vol cutoff (1%)
    trend_threshold_strong=2.0,       # Strong trend threshold
    trend_threshold_weak=0.5          # Weak trend threshold
)
```

### Strategy Behavior

```python
RegimeBasedStrategy(
    allowed_regimes=[MarketRegime.TRENDING],  # Which regimes to trade
    regime_confidence_threshold=0.5,           # Min confidence (0.5 = 50%)
    reduce_confidence_in_marginal_regimes=True # Scale by regime confidence
)
```

## Common Configurations

### Conservative (Trending Only)

```yaml
allowed_regimes: [trending]
regime_confidence_threshold: 0.7
reduce_confidence_in_marginal_regimes: true
```

### Moderate (Trending + Ranging)

```yaml
allowed_regimes: [trending, ranging]
regime_confidence_threshold: 0.5
reduce_confidence_in_marginal_regimes: true
```

### Aggressive (All Except Volatile)

```yaml
allowed_regimes: [trending, ranging, quiet]
regime_confidence_threshold: 0.4
reduce_confidence_in_marginal_regimes: false
```

## Testing

### Unit Tests

```bash
# Test regime detection
pytest tests/ -k "regime" -v

# Test specific regime transitions
pytest tests/test_backtest_phase2_regime.py::TestRegimeTransitions -v
```

### Integration Tests

```bash
# Full end-to-end test
pytest tests/test_integration_regime.py -v

# Specific integration scenario
pytest tests/test_integration_regime.py::TestRegimeStrategyIntegration::test_risk_management_integration -v
```

## Troubleshooting

### Issue: Always Getting HOLD Signals

**Possible Causes:**

1. No regimes in `allowed_regimes` list
2. Regime confidence below threshold
3. Insufficient data for regime detection

**Solutions:**

- Add more regimes to allowed list
- Lower `regime_confidence_threshold`
- Increase `lookback_period` in RegimeDetector
- Check logs for regime classification details

### Issue: Too Many Trades

**Possible Causes:**

1. Too many allowed regimes
2. Low regime confidence threshold
3. Confidence adjustment disabled

**Solutions:**

- Reduce `allowed_regimes` to just TRENDING
- Increase `regime_confidence_threshold`
- Enable `reduce_confidence_in_marginal_regimes`

### Issue: Not Detecting Regime Changes

**Possible Causes:**

1. `lookback_period` too long (slow to react)
2. Thresholds not calibrated for your market

**Solutions:**

- Reduce `lookback_period` (e.g., 50 instead of 100)
- Adjust volatility thresholds for your asset
- Check regime logs to see classifications

## Performance Tips

1. **Backtest First** - Test on historical data before live trading
2. **Monitor Regime Distribution** - Ensure regimes are detected correctly
3. **Tune Thresholds** - Adjust for your specific market/timeframe
4. **Log Everything** - Use verbose logging during testing
5. **Start Conservative** - Begin with TRENDING only, expand later

## Next Steps

1. **Run Backtests** - Validate strategy on historical data
2. **Optimize Parameters** - Tune for your specific market
3. **Paper Trade** - Test in real-time with simulated execution
4. **Monitor Performance** - Track regime-specific metrics
5. **Go Live** - Deploy with appropriate risk limits

## Support & Documentation

- **Full Documentation**: `REGIME_INTEGRATION_SUMMARY.md`
- **Implementation Details**: `REGIME_IMPLEMENTATION_SUMMARY.md`
- **System Spec**: `SYSTEM_SPEC.md`
- **Test Files**: `tests/test_*_regime.py`

---

**Status**: Production Ready ✓  
**Last Updated**: December 30, 2025  
**Test Coverage**: 36/36 passing (100%)
