# RuntimeSummaryAggregator Alert System

## Overview

The `RuntimeSummaryAggregator` has been enhanced with an optional **alert system** that emits INFO-level logs for critical trading conditions. This allows real-time monitoring and early detection of system anomalies without impacting execution.

## Alert Types

### 1. Readiness Degradation Alerts

Emitted when system readiness drops below a configured threshold.

**Conditions:**

- Readiness state transitions to a lower level (e.g., ready → not_ready)
- Current readiness level ≤ alert threshold

**Alert Payload:**

```python
{
    "type": "readiness_degradation",
    "symbol": "BTC/USDT",
    "previous_state": "ready",
    "current_state": "not_ready",
    "readiness_score": 0.1,
    "threshold": "not_ready"
}
```

**Log Example:**

```
INFO [Alert] Readiness degradation:
{'type': 'readiness_degradation', 'symbol': 'BTC/USDT',
 'previous_state': 'ready', 'current_state': 'not_ready',
 'readiness_score': 0.1, 'threshold': 'not_ready'}
```

### 2. Confidence Anomaly Alerts

Emitted when confidence exceeds threshold with unexpected action change.

**Conditions:**

- Confidence > confidence_threshold
- Action changed from previous cycle (not hold)
- Indicates sudden signal reversal with high conviction

**Alert Payload:**

```python
{
    "type": "confidence_anomaly",
    "symbol": "BTC/USDT",
    "confidence": 0.95,
    "threshold": 0.85,
    "previous_action": "buy",
    "current_action": "sell",
    "confidence_delta": 0.35
}
```

**Log Example:**

```
INFO [Alert] Confidence anomaly:
{'type': 'confidence_anomaly', 'symbol': 'BTC/USDT',
 'confidence': 0.95, 'threshold': 0.85,
 'previous_action': 'buy', 'current_action': 'sell',
 'confidence_delta': 0.35}
```

### 3. Strategy Deviation Alerts

Emitted when individual strategy signals deviate significantly from aggregate.

**Sub-types:**

#### a) Action Deviation

Strategy recommends different action than aggregate.

**Conditions:**

- Strategy action ≠ aggregate action
- Neither is "hold"

**Alert Payload:**

```python
{
    "type": "strategy_deviation",
    "symbol": "BTC/USDT",
    "strategy": "ABCStrategy",
    "aggregate_action": "buy",
    "aggregate_confidence": 0.65,
    "strategy_action": "sell",
    "strategy_confidence": 0.6
}
```

#### b) Confidence Deviation

Strategy confidence differs significantly from aggregate.

**Conditions:**

- Confidence difference > deviation_threshold
- Aggregate confidence > 0.3 (sufficient conviction)

**Alert Payload:**

```python
{
    "type": "confidence_deviation",
    "symbol": "BTC/USDT",
    "strategy": "FVGStrategy",
    "aggregate_confidence": 0.65,
    "strategy_confidence": 0.9,
    "deviation": 0.25,
    "threshold": 0.3
}
```

## Configuration

### Initialization Parameters

```python
RuntimeSummaryAggregator(
    interval_seconds=300,                          # Summary emission interval
    logger_fn=log_event,                           # Custom logger function
    enable_alerts=False,                           # Enable alert system (default False)
    readiness_alert_threshold="not_ready",         # Alert threshold level
    confidence_threshold=0.85,                     # High confidence threshold
    deviation_threshold=0.3,                       # Strategy deviation threshold
)
```

### Parameter Details

- **`enable_alerts`**: Master switch for alert system

  - `False` (default): No alerts emitted
  - `True`: Alerts active

- **`readiness_alert_threshold`**: Readiness level that triggers alerts

  - Options: `"ready"`, `"forming"`, `"not_ready"`
  - Default: `"not_ready"` (alert when ready drops to not_ready or below)

- **`confidence_threshold`**: Anomaly threshold for confidence

  - Range: 0.0 to 1.0
  - Default: 0.85 (alert on sudden high-confidence reversals)

- **`deviation_threshold`**: Deviation tolerance for strategies
  - Range: 0.0 to 1.0
  - Default: 0.3 (alert when strategy differs by >30%)

## Usage Examples

### Example 1: Enable All Alerts (Backtesting)

```python
from monitoring.runtime_summary import RuntimeSummaryAggregator

aggregator = RuntimeSummaryAggregator(
    enable_alerts=True,
    readiness_alert_threshold="not_ready",
    confidence_threshold=0.85,
    deviation_threshold=0.3,
)

# In analysis loop
for symbol, market_data in analysis_loop:
    result = regime_strategy.analyze(market_data, symbol=symbol)

    # Record analysis for alert detection
    aggregator.record_analysis(
        symbol=symbol,
        readiness_state=result.get("readiness_state", "not_ready"),
        readiness_score=result.get("readiness_score", 0.0),
        aggregate_action=result.get("action", "hold"),
        aggregate_confidence=result.get("confidence", 0.0),
        contributions=result.get("contributions"),  # Strategy details
    )

    # Alerts logged automatically
```

### Example 2: Conservative Alert Settings (Live)

```python
aggregator = RuntimeSummaryAggregator(
    enable_alerts=True,
    readiness_alert_threshold="ready",     # Only alert if falls to "forming" or below
    confidence_threshold=0.95,             # Very high confidence needed
    deviation_threshold=0.5,               # Large deviation tolerance
)
```

### Example 3: Aggressive Monitoring (Paper Trading)

```python
aggregator = RuntimeSummaryAggregator(
    enable_alerts=True,
    readiness_alert_threshold="forming",   # Alert on any readiness drop
    confidence_threshold=0.75,             # Lower confidence threshold
    deviation_threshold=0.15,              # Tight deviation tolerance
)
```

### Example 4: Alerts Disabled (Default)

```python
# No alerts emitted
aggregator = RuntimeSummaryAggregator()  # enable_alerts=False by default
```

## Alert Log Levels

All alerts are logged at **INFO level** to allow easy filtering:

```python
import logging

# Capture only alerts
alerts = logging.getLogger()
alerts.setLevel(logging.INFO)

# Filter handler for alerts only
class AlertFilter(logging.Filter):
    def filter(self, record):
        return "[Alert]" in record.getMessage()

handler = logging.StreamHandler()
handler.addFilter(AlertFilter())
alerts.addHandler(handler)
```

### Future Level Upgrades

The current INFO level is suitable for testing and development. In production, alerts can be upgraded to different levels:

```python
# Example upgrade pattern (not yet implemented)
ALERT_LEVELS = {
    "readiness_degradation": logging.WARNING,    # Important state change
    "confidence_anomaly": logging.WARNING,       # Unusual signal
    "strategy_deviation": logging.INFO,          # Informational
}
```

## Integration with Trading Systems

### With RegimeBasedStrategy

```python
from strategies.regime_strategy import RegimeBasedStrategy

regime_strategy = RegimeBasedStrategy(
    allowed_regimes=["TRENDING"],
    enable_readiness_gate=True,
    enable_readiness_scaling=True,
)

aggregator = RuntimeSummaryAggregator(enable_alerts=True)

# In backtesting/paper trading loop
for bar in data:
    result = regime_strategy.analyze(bar, symbol="BTC/USDT")

    # Record for alerts
    aggregator.record_analysis(
        symbol="BTC/USDT",
        readiness_state=result.get("readiness_state", "not_ready"),
        readiness_score=result.get("readiness_score", 0.0),
        aggregate_action=result.get("action", "hold"),
        aggregate_confidence=result.get("confidence", 0.0),
        contributions=result.get("contributions", []),
    )

    # Execute trade based on result
```

### With MultiStrategyAggregator

```python
from strategies.multi_strategy_aggregator import MultiStrategyAggregator

agg = MultiStrategyAggregator(
    strategies=[strategy1, strategy2, strategy3],
    enable_adaptive_weights=True,
)

aggregator = RuntimeSummaryAggregator(enable_alerts=True)

# In loop
result = agg.analyze(market_data, symbol="BTC/USDT")

aggregator.record_analysis(
    symbol="BTC/USDT",
    readiness_state="ready",  # From regime layer
    readiness_score=0.85,
    aggregate_action=result["action"],
    aggregate_confidence=result["confidence"],
    contributions=result["contributions"],
)
```

## Testing Alerts

### Capture Alert Logs

```python
import logging
import pytest

def test_alerts(caplog):
    caplog.set_level(logging.INFO)

    aggregator = RuntimeSummaryAggregator(enable_alerts=True)

    # Trigger alert
    aggregator.record_analysis(
        symbol="BTC/USDT",
        readiness_state="not_ready",
        readiness_score=0.1,
        aggregate_action="hold",
        aggregate_confidence=0.0,
    )

    # Check logs
    alert_logs = [rec for rec in caplog.records if "[Alert]" in rec.message]
    assert len(alert_logs) > 0
```

### Filter Alert Types

```python
def test_specific_alert(caplog):
    caplog.set_level(logging.INFO)

    aggregator = RuntimeSummaryAggregator(enable_alerts=True)

    # ... trigger alert ...

    # Get specific alert type
    confidence_alerts = [
        rec for rec in caplog.records
        if "[Alert]" in rec.message and "confidence_anomaly" in rec.message
    ]
    assert len(confidence_alerts) > 0
```

## Performance Considerations

### Memory Usage

- Per-symbol state tracking (minimal overhead)
- No history storage (aggregated stats only)
- Suitable for multi-pair monitoring

### CPU Usage

- Alert checks: O(N) where N = number of strategies
- Negligible overhead (simple comparisons)
- Non-blocking logging

### Impact on Trading

- Alerts are **observational only**
- No impact on trading logic or execution
- Can be safely enabled in live trading

## Readiness State Levels

The alert system recognizes three readiness states with hierarchy:

```
ready (3)
   ↓
forming (2)
   ↓
not_ready (1)
```

Alerts trigger when:

- Current level ≤ threshold level AND state changed

Examples:

- Threshold "ready": Alert if ready→forming or ready→not_ready
- Threshold "forming": Alert if forming→not_ready
- Threshold "not_ready": Alert on any transition TO not_ready

## Troubleshooting

### Alerts Not Appearing

1. **Check enable_alerts**: Must be `True`

   ```python
   agg = RuntimeSummaryAggregator(enable_alerts=True)
   ```

2. **Check log level**: Should be INFO or lower

   ```python
   logging.getLogger().setLevel(logging.INFO)
   ```

3. **Check thresholds**: May be too conservative
   ```python
   # Lower threshold for more alerts
   readiness_alert_threshold="forming"
   confidence_threshold=0.75
   ```

### Too Many Alerts

1. **Increase thresholds**:

   ```python
   confidence_threshold=0.95  # Only very high confidence
   deviation_threshold=0.5    # Allow more deviation
   ```

2. **Disable specific alert types**: Not yet implemented, use log filtering
   ```python
   # Filter in handler instead
   ```

## API Reference

### record_analysis()

```python
def record_analysis(
    symbol: str,
    readiness_state: str,
    readiness_score: float,
    aggregate_action: str,
    aggregate_confidence: float,
    contributions: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Record strategy analysis results for alert detection."""
```

**Parameters:**

- `symbol`: Trading pair (e.g., "BTC/USDT")
- `readiness_state`: Readiness level ("ready", "forming", "not_ready")
- `readiness_score`: Readiness score 0-1
- `aggregate_action`: Aggregated signal ("buy", "sell", "hold")
- `aggregate_confidence`: Signal confidence 0-1
- `contributions`: List of per-strategy dicts with `{strategy, action, confidence}`

---

**Status**: ✅ Complete and tested  
**Test Coverage**: 16/16 alert tests passing  
**Total Test Suite**: 152/152 tests passing  
**Log Level**: INFO (ready for production upgrade)  
**Safety**: Observational only, no impact on execution
