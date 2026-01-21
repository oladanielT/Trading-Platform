# Optimization System - Quick Reference

## Initialization

```python
from ai.optimization_engine import initialize_engine

# At startup
engine = initialize_engine(
    base_weights={
        "EMA_Trend": 1.0,
        "FVG_Pattern": 1.0,
        "RSI_Reversal": 0.8,
    }
)
```

## Trade Execution (with Optimization)

```python
from ai.optimization_engine import get_engine

engine = get_engine()

# 1. Score signal
quality, should_execute = engine.score_signal(
    strategy="EMA_Trend",
    symbol="BTC/USDT",
    signal_type="BUY",
    original_confidence=0.75,
    confluence_count=3,
)

# 2. Skip low-quality signals
if not should_execute:
    return None

# 3. Size position with optimization
base_size = calculate_base_position_size()
multiplier = engine.get_position_size_multiplier(
    strategy="EMA_Trend",
    signal_quality=quality,
)
final_size = base_size * multiplier

# 4. Execute (risk module does final checks)
execute_order(final_size)
```

## Periodic Optimization Update (e.g., Hourly)

```python
# Get current regime
regime = detect_market_regime()

# Update all components
state = engine.update_optimization(regime=regime)

# Log changes
if state.weight_changes_this_period:
    logger.info(f"Weight updates: {state.weight_changes_this_period}")
if state.suspended_strategies:
    logger.warning(f"Suspended: {state.suspended_strategies}")
```

## Monitoring

```python
from ai.optimization_monitor import get_monitor

monitor = get_monitor()

# Get summaries
status = monitor.get_optimization_status()
health = monitor.get_health_report()
allocation = monitor.get_allocation_report()

# Export
monitor.export_json('optimization_report.json')
```

## Performance Decomposition

```python
from ai.performance_decomposer import record_trade, get_decomposer

# Record trades
record_trade(
    strategy_name="EMA_Trend",
    symbol="BTC/USDT",
    pnl=250.0,
    entry_ts=1704067200000,
    exit_ts=1704153600000,
    regime="TRENDING",
)

# Query performance
decomposer = get_decomposer()
perf = decomposer.get_strategy_performance("EMA_Trend")
print(f"Expectancy: {perf.expectancy:.2f}")
print(f"Profit Factor: {perf.profit_factor:.2f}")
print(f"Win Rate: {perf.win_rate:.2f}%")

# Get by regime
regimes = decomposer.get_regime_effectiveness("EMA_Trend")
for regime, perf in regimes.items():
    print(f"{regime}: {perf.expectancy:.2f}")
```

## Adaptive Weighting

```python
from ai.adaptive_weighting_optimizer import get_optimizer

optimizer = get_optimizer()

# Check current weights
weights = optimizer.get_current_weights()
print(weights)

# Get normalized allocation
normalized = optimizer.normalize_weights()
print(f"Total: {sum(normalized.values()):.1f}")

# View history
history = optimizer.get_weight_history()
for update in history[-5:]:
    print(f"{update.strategy}: {update.previous_weight:.3f} → {update.new_weight:.3f}")
```

## Signal Quality Scoring

```python
from ai.signal_quality_scorer import get_scorer

scorer = get_scorer()

# Score a signal
metrics = scorer.score_signal(
    strategy="EMA_Trend",
    symbol="BTC/USDT",
    signal_type="BUY",
    original_confidence=0.75,
    confluence_count=3,
    regime="TRENDING",
)

print(f"Quality: {metrics.quality_score:.3f}")
print(f"Can Execute: {metrics.quality_score >= scorer.min_quality_threshold}")

# Get quality stats
summary = scorer.get_quality_summary("EMA_Trend", "BTC/USDT")
print(f"Avg Quality: {summary['avg_quality_score']:.3f}")
print(f"High Quality %: {summary['high_quality_percentage']:.1f}%")
```

## Strategy Retirement

```python
from ai.strategy_retirement import get_manager, StrategyStatus

manager = get_manager()

# Assess strategy
health = manager.assess_strategy(
    strategy_name="EMA_Trend",
    recent_expectancy=0.5,
    recent_profit_factor=1.2,
    recent_win_rate=0.52,
    recent_trades_count=50,
)

print(f"Status: {health.status.value}")

# Get all statuses
active = manager.get_active_strategies()
degraded = manager.get_degraded_strategies()
suspended = manager.get_suspended_strategies()

# View audit log
audit = manager.get_audit_log("EMA_Trend")
for entry in audit[-3:]:
    print(f"{entry['old_status']} → {entry['new_status']}")

# Override if needed (admin)
manager.force_status("BadStrat", StrategyStatus.RETIRED, "Admin override")
```

## Monitoring Checklist

- [ ] Optimization engine initialized at startup
- [ ] `update_optimization()` called periodically (hourly)
- [ ] `score_signal()` called before every trade
- [ ] Position multiplier applied to position sizing
- [ ] Logs monitored for weight changes and suspensions
- [ ] Regular health reports generated
- [ ] Optimization metrics exported to monitoring system

## Configuration Knobs

| Parameter               | Default | Range    | Impact                                              |
| ----------------------- | ------- | -------- | --------------------------------------------------- |
| `adjustment_rate`       | 0.15    | 0.01-0.5 | How fast weights change (higher=more responsive)    |
| `min_weight`            | 0.1     | 0.0-1.0  | Minimum strategy weight (lower=can disable)         |
| `max_weight`            | 2.0     | 1.0-5.0  | Maximum strategy weight (higher=more concentration) |
| `min_quality_threshold` | 0.7     | 0.3-1.0  | Signal filter threshold (higher=fewer trades)       |
| `min_expectancy`        | 0.1     | 0.0-1.0  | Minimum expectancy for active status                |
| `min_profit_factor`     | 0.8     | 0.5-1.5  | Minimum PF for active status                        |
| `degraded_window_hours` | 24      | 1-168    | Hours before degraded→suspended                     |

## Troubleshooting

**Signals being over-filtered?**

```python
scorer = get_scorer()
scorer.min_quality_threshold = 0.6  # Lower from 0.7
```

**Strategies suspending too fast?**

```python
manager = get_manager()
manager.degraded_window_hours = 48  # Increase from 24
```

**Weights not adapting?**

```python
state = engine.update_optimization()
print(state.weight_changes_this_period)  # Check if any changes
```

**Strategy suspended by mistake?**

```python
manager = get_manager()
manager.force_status("Strategy", StrategyStatus.ACTIVE, "Manual override")
```

## Risk Guarantees

✅ **Optimization CANNOT**:

- Increase position sizes beyond risk limits
- Disable drawdown guards or circuit breakers
- Weaken readiness gating
- Create leverage

✅ **Optimization CAN**:

- Be disabled instantly
- Be reverted to base weights
- Be paused while troubleshooting
- Be configured conservatively

---

See [OPTIMIZATION_SYSTEM.md](OPTIMIZATION_SYSTEM.md) for full documentation.
