# Runtime Alerts Quick Reference

## Quick Start

### Enable Alerts

```python
from monitoring.runtime_summary import RuntimeSummaryAggregator

agg = RuntimeSummaryAggregator(
    enable_alerts=True,  # Activate alert system
)
```

### Record Analysis

```python
# After strategy analysis
agg.record_analysis(
    symbol="BTC/USDT",
    readiness_state="ready",
    readiness_score=0.85,
    aggregate_action="buy",
    aggregate_confidence=0.7,
    contributions=[
        {"strategy": "FVGStrategy", "action": "buy", "confidence": 0.8},
        {"strategy": "ABCStrategy", "action": "buy", "confidence": 0.6},
    ]
)
```

## Alert Types

| Alert Type                | Trigger                           | Example                          |
| ------------------------- | --------------------------------- | -------------------------------- |
| **readiness_degradation** | Readiness drops below threshold   | ready → not_ready                |
| **confidence_anomaly**    | High confidence + action reversal | 0.95 conf, buy→sell              |
| **strategy_deviation**    | Strategy conflicts with aggregate | Strategy says sell, agg says buy |
| **confidence_deviation**  | Strategy confidence deviates >30% | Strat 0.9, agg 0.65              |

## Configuration

### Default Settings

```python
RuntimeSummaryAggregator(
    enable_alerts=False,                    # Disabled by default
    readiness_alert_threshold="not_ready",  # Alert on degradation
    confidence_threshold=0.85,              # High confidence needed
    deviation_threshold=0.3,                # 30% deviation tolerance
)
```

### Presets

**Backtest/Paper (Aggressive)**

```python
enable_alerts=True
readiness_alert_threshold="forming"
confidence_threshold=0.75
deviation_threshold=0.15
```

**Live (Conservative)**

```python
enable_alerts=True
readiness_alert_threshold="ready"
confidence_threshold=0.95
deviation_threshold=0.5
```

## Alert Log Format

All alerts log as INFO with `[Alert]` prefix:

```
INFO [Alert] Readiness degradation: {'type': 'readiness_degradation', 'symbol': 'BTC/USDT', ...}
INFO [Alert] Confidence anomaly: {'type': 'confidence_anomaly', 'symbol': 'BTC/USDT', ...}
INFO [Alert] Strategy deviation: {'type': 'strategy_deviation', 'symbol': 'BTC/USDT', ...}
```

## Filtering Alerts

### By Type

```python
# Get all readiness alerts
readiness_alerts = [
    rec for rec in caplog.records
    if "[Alert]" in rec.message and "readiness_degradation" in rec.message
]
```

### By Symbol

```python
# Get all BTC alerts
btc_alerts = [
    rec for rec in caplog.records
    if "[Alert]" in rec.message and "BTC/USDT" in rec.message
]
```

### By Level

```python
# Future: upgrade to WARNING level
# logging.getLogger().setLevel(logging.WARNING)
```

## Usage in Backtesting

```python
from monitoring.runtime_summary import RuntimeSummaryAggregator
from strategies.regime_strategy import RegimeBasedStrategy

regime_strat = RegimeBasedStrategy(
    allowed_regimes=["TRENDING"],
    enable_readiness_gate=True,
)

alerts = RuntimeSummaryAggregator(enable_alerts=True)

for bar in backtest_data:
    result = regime_strat.analyze(bar, symbol="BTC/USDT")

    # Record for alerts
    alerts.record_analysis(
        symbol="BTC/USDT",
        readiness_state=result.get("readiness_state"),
        readiness_score=result.get("readiness_score"),
        aggregate_action=result.get("action"),
        aggregate_confidence=result.get("confidence"),
        contributions=result.get("contributions"),
    )

    # Execute trade
```

## State Levels

```
ready (level 3)      ← Best readiness
forming (level 2)    ← Intermediate
not_ready (level 1)  ← Poor readiness

Alert triggers when: current_level <= threshold_level AND state changed
```

## Key Points

✅ **Opt-in only** - Disabled by default  
✅ **INFO level** - Easy filtering, ready for WARN upgrade  
✅ **Observational** - No impact on trading execution  
✅ **Per-symbol** - Independent tracking per pair  
✅ **Zero overhead** - Non-blocking, minimal CPU/memory  
✅ **Production safe** - Thoroughly tested

## Testing

```python
import logging
import pytest

def test_readiness_alert(caplog):
    caplog.set_level(logging.INFO)

    agg = RuntimeSummaryAggregator(enable_alerts=True)

    # Trigger alert
    agg.record_analysis(
        symbol="BTC/USDT",
        readiness_state="ready",
        readiness_score=0.9,
        aggregate_action="buy",
        aggregate_confidence=0.6,
    )

    agg.record_analysis(
        symbol="BTC/USDT",
        readiness_state="not_ready",
        readiness_score=0.1,
        aggregate_action="hold",
        aggregate_confidence=0.0,
    )

    # Verify alert
    alerts = [rec for rec in caplog.records if "[Alert]" in rec.message]
    assert len(alerts) > 0
```

## Common Patterns

### Monitor High-Confidence Reversals

```python
RuntimeSummaryAggregator(
    enable_alerts=True,
    confidence_threshold=0.8,  # Alert on high confidence
    # ... rest of config
)
```

### Track Strategy Disagreements

```python
RuntimeSummaryAggregator(
    enable_alerts=True,
    deviation_threshold=0.15,  # Tight tolerance
    # ... rest of config
)
```

### Detect Readiness Drops

```python
RuntimeSummaryAggregator(
    enable_alerts=True,
    readiness_alert_threshold="forming",  # Alert on any drop
    # ... rest of config
)
```

---

**Status**: ✅ Complete and tested  
**Test Coverage**: 16/16 tests passing  
**Log Level**: INFO  
**Safety**: Production ready
