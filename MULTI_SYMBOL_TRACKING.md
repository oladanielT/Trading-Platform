# Multi-Symbol Tracking Implementation

## Overview

Extended the regime-based trading system to support independent tracking of multiple trading pairs simultaneously. Each symbol maintains its own readiness state, confidence scores, and regime metrics to prevent cross-contamination between different markets.

## Key Features

### 1. Per-Symbol Readiness Tracking

- **Independent State**: Each symbol maintains separate `_readiness_score` and `_readiness_state`
- **No Cross-Contamination**: BTC/USDT's readiness doesn't affect ETH/USDT
- **Default Fallback**: Single-symbol usage preserved for backward compatibility

### 2. Symbol-Aware Logging

- All regime and readiness logs now include the `symbol` parameter
- Clear visibility into which trading pair triggered each decision
- Example: `[RegimeBasedStrategy] symbol=BTC/USDT, readiness=READY (0.85)`

### 3. Multi-Strategy Aggregation

- `MultiStrategyAggregator` propagates symbol to underlying strategies
- Automatically detects if strategies support `symbol` parameter
- Maintains independent strategy states per symbol

### 4. Backward Compatibility

- Default symbol fallback (`"DEFAULT"`) for single-pair usage
- All existing code works without modification
- Feature flags preserve legacy behavior

## Architecture Changes

### RegimeBasedStrategy

```python
# Before (single symbol)
self._readiness_score: float = 0.0
self._readiness_state: ReadinessState = ReadinessState.NOT_READY

# After (multi-symbol)
self._readiness_score: Dict[str, float] = {}  # keyed by symbol
self._readiness_state: Dict[str, ReadinessState] = {}  # keyed by symbol
self._default_symbol: str = "DEFAULT"  # fallback for single-pair usage
```

### Analyze Method Signature

```python
def analyze(self, market_data: pd.DataFrame, symbol: Optional[str] = None) -> Optional[dict]:
    """
    Analyze market data and generate trading signals.

    Args:
        market_data: OHLCV price data
        symbol: Trading pair identifier (e.g., "BTC/USDT")
                If None, uses default symbol for backward compatibility

    Returns:
        Dict with action, confidence, and symbol (if provided)
    """
```

### MultiStrategyAggregator

- Inspects underlying strategy `analyze()` signatures
- Passes `symbol` parameter only to compatible strategies
- Includes symbol in result dict and logs

## Usage Examples

### Single Symbol (Legacy)

```python
strategy = RegimeBasedStrategy(allowed_regimes=["TRENDING"])
result = strategy.analyze(market_data)
# Uses default symbol internally, works as before
```

### Multiple Symbols

```python
strategy = RegimeBasedStrategy(
    allowed_regimes=["TRENDING"],
    enable_readiness_gate=True,
    trading_mode="paper"
)

# Analyze BTC
btc_result = strategy.analyze(btc_data, symbol="BTC/USDT")
# Readiness state stored under "BTC/USDT"

# Analyze ETH independently
eth_result = strategy.analyze(eth_data, symbol="ETH/USDT")
# Separate readiness state under "ETH/USDT"

# Check per-symbol readiness
btc_readiness = strategy._readiness_state["BTC/USDT"]
eth_readiness = strategy._readiness_state["ETH/USDT"]
```

### With Multi-Strategy Aggregation

```python
from strategies.pattern_strategies import FVGStrategy, ABCStrategy
from strategies.multi_strategy_aggregator import MultiStrategyAggregator

aggregator = MultiStrategyAggregator([
    FVGStrategy(),
    ABCStrategy(),
])

regime_strategy = RegimeBasedStrategy(
    allowed_regimes=["TRENDING"],
    trading_mode="paper",
    enable_pattern_strategies=True,
    pattern_aggregator=aggregator
)

# Symbol automatically propagated through aggregator to underlying strategies
btc_result = regime_strategy.analyze(btc_data, symbol="BTC/USDT")
# All pattern strategies receive symbol="BTC/USDT"
```

## Testing

### Test Coverage

- **test_multi_symbol_tracking.py**: 4 tests validating independent symbol tracking
  - Per-symbol readiness state isolation
  - Default symbol fallback behavior
  - Symbol-aware logging in aggregator
  - Independence of readiness scores across symbols

### Updated Existing Tests

- **test_regime_readiness.py**: Updated to use dict-based readiness state
- **test_readiness_gating.py**: Updated to set per-symbol readiness

### All Tests Passing

```bash
$ pytest tests/ -v
120 passed, 9 warnings in 5.81s
```

## Log Output Example

```
INFO [RegimeBasedStrategy] symbol=BTC/USDT, regime=TRENDING (0.92), readiness=READY (0.87)
INFO [ReadinessScaling] symbol=BTC/USDT, scaling_factor=1.00 (readiness=0.87, READY)
INFO [MultiStrategyAggregator] symbol=BTC/USDT, weighted_vote: buy=0.65, sell=0.00, hold=0.35
INFO [RegimeBasedStrategy] symbol=ETH/USDT, regime=TRENDING (0.78), readiness=FORMING (0.52)
INFO [ReadinessScaling] symbol=ETH/USDT, scaling_factor=0.85 (readiness=0.52, FORMING)
INFO [MultiStrategyAggregator] symbol=ETH/USDT, weighted_vote: buy=0.45, sell=0.10, hold=0.45
```

## Benefits

1. **Multi-Pair Trading**: Trade multiple symbols independently without interference
2. **Per-Symbol Confidence**: Each market assessed on its own merits
3. **Clear Attribution**: Logs clearly show which symbol triggered actions
4. **No Breaking Changes**: Existing single-pair code works unchanged
5. **Production Ready**: Full test coverage ensures reliability

## Future Enhancements

- [ ] Per-symbol runtime summaries (currently aggregates all symbols)
- [ ] Symbol-specific pattern strategy tuning
- [ ] Cross-symbol correlation detection
- [ ] Portfolio-level readiness scoring

## Migration Notes

### For Existing Code

No changes required. The system automatically uses the default symbol fallback for backward compatibility.

### For New Multi-Symbol Code

Add `symbol` parameter to all `analyze()` calls:

```python
# Old
result = strategy.analyze(market_data)

# New (multi-symbol)
result = strategy.analyze(market_data, symbol="BTC/USDT")
```

## Implementation Files Modified

- `strategies/regime_strategy.py`: Per-symbol readiness dicts, symbol param in analyze()
- `strategies/multi_strategy_aggregator.py`: Symbol propagation to underlying strategies
- `tests/test_multi_symbol_tracking.py`: New test suite for multi-symbol functionality
- `tests/test_regime_readiness.py`: Updated for dict-based readiness state
- `tests/test_readiness_gating.py`: Updated for per-symbol readiness

---

**Status**: ✅ Complete and tested
**Version**: Phase 8 Multi-Symbol Extension
**Date**: 2025
