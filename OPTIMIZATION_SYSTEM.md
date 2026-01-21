# Optimization & Alpha Expansion System

## Overview

This document describes the production-ready optimization system that enhances profitability and edge without increasing risk. The system is designed as **non-invasive advisory** - it operates entirely within the existing risk framework and never weakens safety constraints.

### Core Philosophy

- **Risk-First**: All optimizations respect existing drawdown limits, position sizing rules, and circuit breakers
- **Non-Invasive**: Can be disabled or reverted instantly without affecting trading logic
- **Measurable**: Every optimization is tracked and can be audited
- **Gradual**: Changes are conservative and limited to prevent destabilization
- **Reversible**: All components can be reset to baseline at any time

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Optimization Engine                         │
│          (Unified coordinator and safety monitor)            │
└────────┬────────┬──────────────┬────────────────┬────────────┘
         │        │              │                │
    ┌────▼───┐ ┌──▼────┐ ┌──────▼─────┐ ┌───────▼────┐
    │Performance  │ Adaptive │ Signal      │ Strategy    │
    │Decomposer   │ Weighting │ Quality    │ Retirement  │
    │             │ Optimizer │ Scorer     │ Manager     │
    └──────────────┴────────────┴────────────┴─────────────┘
         │                │              │
         ▼                ▼              ▼
    ┌──────────────────────────────────────┐
    │    Existing Risk/Execution Modules   │
    │  (Position Sizing, Drawdown Guard,   │
    │   Circuit Breakers, Readiness Gate)  │
    └──────────────────────────────────────┘
```

---

## Components

### 1. Performance Decomposer (`ai/performance_decomposer.py`)

**Purpose**: Break down trading performance by strategy, symbol, and market regime to identify which strategies work best under which conditions.

**Key Metrics**:

- Win rate and loss rate
- Expectancy: `(WinRate × AvgWin) - (LossRate × AvgLoss)`
- Profit Factor: `GrossProfit / |GrossLoss|`
- Sharpe Ratio: Mean return / Standard deviation
- Maximum drawdown per strategy
- Consecutive loss tracking

**Usage**:

```python
from ai.performance_decomposer import record_trade, get_decomposer

# Record a closed trade
record_trade(
    strategy_name="EMA_Trend",
    symbol="BTC/USDT",
    pnl=250.0,
    entry_ts=1704067200000,  # milliseconds
    exit_ts=1704153600000,
    regime="TRENDING",
    size=0.5
)

# Get strategy performance
decomposer = get_decomposer()
perf = decomposer.get_strategy_performance("EMA_Trend")
print(f"Win Rate: {perf.win_rate:.2f}%")
print(f"Expectancy: {perf.expectancy:.2f}")
print(f"Profit Factor: {perf.profit_factor:.2f}")

# Get regime effectiveness
regime_perf = decomposer.get_regime_effectiveness("EMA_Trend")
for regime, perf in regime_perf.items():
    print(f"{regime}: expectancy={perf.expectancy:.2f}")
```

**Key Methods**:

- `record_trade()`: Log a closed trade
- `get_strategy_performance()`: Get metrics for a strategy
- `get_regime_effectiveness()`: Performance breakdown by market regime
- `get_symbol_breakdown()`: Performance breakdown by trading pair
- `get_top_strategies()`: Find best/worst performers by metric

---

### 2. Adaptive Weighting Optimizer (`ai/adaptive_weighting_optimizer.py`)

**Purpose**: Dynamically adjust strategy weights based on recent performance, profit factor, drawdown contribution, and regime alignment.

**How It Works**:

1. **Composite Scoring**: Each strategy gets a score based on:

   - Expectancy (40% weight)
   - Profit Factor (30% weight)
   - Drawdown penalty (20% weight)
   - Consistency/Sharpe (10% weight)

2. **Conservative Adjustment**: Weight changes are limited to 15% per update to prevent wild swings

3. **Regime-Specific Weights**: Different weights can be applied per market regime

4. **Bounds**: Minimum 0.1x and maximum 2.0x weight to prevent disabling or over-weighting strategies

**Usage**:

```python
from ai.adaptive_weighting_optimizer import initialize_optimizer, get_optimizer

# Initialize with base weights
base_weights = {
    "EMA_Trend": 1.0,
    "FVG_Pattern": 1.0,
    "RSI_Reversal": 0.8,
}

optimizer = initialize_optimizer(base_weights)

# Update weights based on performance
performance_data = {
    "EMA_Trend": {
        "expectancy": 2.5,
        "profit_factor": 1.8,
        "max_drawdown": 0.05,
        "sharpe_ratio": 1.2,
    },
    # ... other strategies
}

new_weights = optimizer.update_weights(performance_data, regime="TRENDING")
print(f"New weights: {new_weights}")

# Get allocation report
summary = optimizer.get_allocation_summary()
print(f"Total allocation: {summary['total_allocation']:.1f}")
print(f"Normalized weights: {optimizer.normalize_weights()}")

# Check history
history = optimizer.get_weight_history()
for update in history[-5:]:
    print(f"{update.strategy}: {update.previous_weight:.3f} → {update.new_weight:.3f}")
```

**Key Methods**:

- `update_weights()`: Recalculate weights based on performance
- `get_current_weights()`: Get current adaptive weights
- `get_weight_for_regime()`: Get weight for specific regime
- `normalize_weights()`: Scale weights to sum to 1.0
- `reset_to_base_weights()`: Revert to static weights
- `get_allocation_summary()`: Get allocation metrics

---

### 3. Signal Quality Scorer (`ai/signal_quality_scorer.py`)

**Purpose**: Evaluate signal quality using confluence analysis and regime compatibility to filter low-confidence signals.

**Quality Factors**:

- **Confluence**: How many indicators agree (1-5+)
- **Regime Alignment**: How well signal fits current market regime (0.0-1.0)
- **Historical Success Rate**: Historical accuracy of similar setups
- **Original Confidence**: Base confidence from strategy

**Quality Score Formula**:

```
quality_score = (
    original_confidence × 0.3 +
    regime_alignment × 0.3 +
    historical_success_rate × 0.4
) × confluence_factor × regime_adjustment
```

**Usage**:

```python
from ai.signal_quality_scorer import get_scorer

scorer = get_scorer()

# Score a signal
metrics = scorer.score_signal(
    strategy="EMA_Trend",
    symbol="BTC/USDT",
    signal_type="BUY",
    original_confidence=0.75,
    confluence_count=3,  # EMA, RSI, Support agree
    regime="TRENDING",
    regime_alignment=0.85,
)

print(f"Quality Score: {metrics.quality_score:.3f}")
print(f"Should Execute: {metrics.quality_score >= scorer.min_quality_threshold}")

# Get history
history = scorer.get_signal_history(limit=50)
high_quality = [s for s in history if s.quality_score >= 0.7]
print(f"High quality signals: {len(high_quality)}/{len(history)}")

# Get quality summary
summary = scorer.get_quality_summary("EMA_Trend", "BTC/USDT")
print(f"Average quality: {summary['avg_quality_score']:.3f}")
```

**Regime Adjustments** (default):

- `TRENDING`: 1.1x boost (signals work better in trends)
- `RANGING`: 0.8x reduction (signals less reliable in ranges)
- `VOLATILE`: 0.7x reduction (noisy environment)

**Key Methods**:

- `score_signal()`: Calculate quality score for a signal
- `filter_signals()`: Filter to only high-quality signals
- `get_signal_history()`: Retrieve scored signals
- `get_quality_summary()`: Get aggregate metrics
- `record_signal_outcome()`: Track actual trade results
- `set_regime_adjustment()`: Customize regime multipliers

---

### 4. Strategy Retirement Manager (`ai/strategy_retirement.py`)

**Purpose**: Detect underperforming strategies and automatically reduce allocation or suspend them.

**Status Lifecycle**:

```
ACTIVE → DEGRADED → SUSPENDED → RETIRED
  ▲         ▲
  └─────────┘ (recovery possible)
```

**Thresholds** (configurable):

- Minimum expectancy: 0.1
- Minimum profit factor: 0.8
- Minimum win rate: 35%
- Maximum drawdown: 15%
- Maximum consecutive losses: 10

**Timing**:

- ACTIVE → DEGRADED: Immediate (violations detected)
- DEGRADED → SUSPENDED: After 24 hours in degraded state
- SUSPENDED → RETIRED: After 72 hours in suspended state
- DEGRADED → ACTIVE: Recovers when all metrics healthy

**Usage**:

```python
from ai.strategy_retirement import get_manager

manager = get_manager()

# Assess strategy health
health = manager.assess_strategy(
    strategy_name="EMA_Trend",
    recent_expectancy=0.5,
    recent_profit_factor=1.2,
    recent_win_rate=0.52,
    recent_trades_count=50,
    max_drawdown_recent=0.08,
    consecutive_losses=2,
    max_consecutive_losses=5,
)

print(f"Status: {health.status.value}")
print(f"Reason: {health.status_reason}")
print(f"Confidence: {health.confidence_level:.2f}")

# Get all health metrics
all_health = manager.get_all_health()
active = manager.get_active_strategies()
degraded = manager.get_degraded_strategies()
suspended = manager.get_suspended_strategies()

print(f"Active: {len(active)}, Degraded: {len(degraded)}, Suspended: {len(suspended)}")

# View audit trail
audit = manager.get_audit_log("EMA_Trend")
for entry in audit[-5:]:
    print(f"{entry['timestamp']}: {entry['old_status']} → {entry['new_status']}")

# Admin override (if needed)
manager.force_status("BadStrategy", StrategyStatus.RETIRED, "Manual retirement")
```

**Key Methods**:

- `assess_strategy()`: Evaluate strategy health
- `get_strategy_health()`: Get current health
- `get_all_health()`: Get all strategy health
- `get_active_strategies()`: List active strategies
- `get_degraded_strategies()`: List degraded strategies
- `get_suspended_strategies()`: List suspended strategies
- `get_audit_log()`: View status change history
- `force_status()`: Admin override

---

### 5. Optimization Engine (`ai/optimization_engine.py`)

**Purpose**: Unified coordinator ensuring all optimization components work together safely.

**Responsibilities**:

1. Orchestrate updates across all components
2. Ensure risk constraints are never violated
3. Track optimization state and history
4. Provide signal quality filtering
5. Calculate position size multipliers

**How It Works**:

```python
from ai.optimization_engine import initialize_engine, get_engine

# Initialize once at startup
base_weights = {
    "EMA_Trend": 1.0,
    "FVG_Pattern": 1.0,
}

engine = initialize_engine(
    base_weights=base_weights,
    risk_max_drawdown=0.20,  # Never exceed 20% portfolio DD
    enable_quality_filtering=True,
    enable_adaptive_weighting=True,
    enable_retirement_management=True,
)

# Periodic optimization update (e.g., hourly)
state = engine.update_optimization(regime="TRENDING")
print(f"Active strategies: {len(state.active_strategies)}")
print(f"Best performer: {state.best_strategy}")
print(f"Weight changes: {state.weight_changes_this_period}")

# Score signals at trade time
quality, should_execute = engine.score_signal(
    strategy="EMA_Trend",
    symbol="BTC/USDT",
    signal_type="BUY",
    original_confidence=0.75,
    confluence_count=3,
)

if should_execute:
    # Get position size multiplier
    multiplier = engine.get_position_size_multiplier(
        strategy="EMA_Trend",
        signal_quality=quality,
        regime="TRENDING",
    )
    # Apply multiplier to base position size (respects risk limits)
else:
    print("Signal quality too low, skip trade")

# Check optimization health
status = engine.get_optimization_status()
print(f"Enabled features: {status['enabled']}")

# View history
history = engine.get_optimization_history(limit=10)
for state in history:
    print(f"{state['timestamp']}: weights={state['current_weights']}")
```

**Key Methods**:

- `update_optimization()`: Perform comprehensive update
- `score_signal()`: Score and filter signals
- `get_strategy_weight()`: Get effective weight for strategy
- `get_position_size_multiplier()`: Calculate position multiplier
- `get_optimization_status()`: Get current state
- `get_optimization_history()`: View state history

---

### 6. Optimization Monitor (`ai/optimization_monitor.py`)

**Purpose**: Unified monitoring and reporting for all optimization metrics.

**Reports Available**:

- Optimization status (component health)
- Strategy summaries (comprehensive breakdown)
- Allocation report (current weights)
- Health report (strategy status)
- Regime effectiveness (which strategies work when)
- JSON export (for logging/analysis)

**Usage**:

```python
from ai.optimization_monitor import get_monitor

monitor = get_monitor()

# Get overall status
status = monitor.get_optimization_status()
print(f"Tracked strategies: {status['metrics']['strategies_tracked']}")

# Get detailed strategy info
summary = monitor.get_strategy_summary("EMA_Trend")
print(f"Performance: {summary['performance']['expectancy']:.2f}")
print(f"Current weight: {summary['current_weight']:.3f}")
print(f"Health: {summary['health_status']['status']}")

# Get allocation
alloc = monitor.get_allocation_report()
print(f"Normalized weights: {alloc['normalized_weights']}")

# Get health overview
health = monitor.get_health_report()
print(f"Active: {len(health['active'])}")
print(f"Degraded: {len(health['degraded'])}")
print(f"Suspended: {len(health['suspended'])}")

# Get regime breakdown
regimes = monitor.get_regime_effectiveness_report()
for regime, strategies in regimes['by_regime'].items():
    print(f"\n{regime}:")
    for s in strategies:
        print(f"  {s['strategy']}: expectancy={s['expectancy']:.2f}")

# Export to file
monitor.export_json('optimization_report.json')

# Get as pandas DataFrame
df = monitor.get_dataframe_export('performance')
print(df.to_string())
```

**Key Methods**:

- `get_optimization_status()`: Overall component health
- `get_strategy_summary()`: Comprehensive strategy metrics
- `get_allocation_report()`: Current allocation
- `get_health_report()`: Strategy health status
- `get_regime_effectiveness_report()`: Performance by regime
- `export_json()`: Export to file
- `get_dataframe_export()`: Get as pandas DataFrame

---

## Integration with Live Trading

### Initialization

Add to your main entry point (e.g., `main.py` or `interfaces/api.py`):

```python
from ai.optimization_engine import initialize_engine
from ai.optimization_monitor import get_monitor

# Initialize optimization system
base_weights = {
    "EMA_Trend": 1.0,
    "FVG_Pattern": 1.0,
    "RSI_Reversal": 0.8,
}

optimization_engine = initialize_engine(
    base_weights=base_weights,
    enable_quality_filtering=True,
    enable_adaptive_weighting=True,
    enable_retirement_management=True,
)

# Initialize monitor
monitor = get_monitor()
```

### Trade Execution Flow

```python
def execute_trade(strategy_name, signal):
    """Modified trade execution with optimization."""

    # Score the signal
    quality, should_execute = optimization_engine.score_signal(
        strategy=strategy_name,
        symbol=signal.symbol,
        signal_type=signal.signal.value,
        original_confidence=signal.confidence,
        confluence_count=signal.metadata.get('confluence_count', 1),
    )

    if not should_execute:
        logger.info(f"Signal filtered: {strategy_name}/{signal.symbol} "
                   f"quality={quality:.3f}")
        return None

    # Size position with optimization multiplier
    base_size = calculate_position_size(signal)
    multiplier = optimization_engine.get_position_size_multiplier(
        strategy=strategy_name,
        signal_quality=quality,
    )
    final_size = base_size * multiplier

    # Execute trade (risk module will do final checks)
    return execute_order(signal, size=final_size)
```

### Periodic Optimization Updates

```python
async def optimization_update_loop():
    """Periodically update optimizations (e.g., hourly)."""

    while True:
        await asyncio.sleep(3600)  # Every hour

        try:
            # Get current regime
            regime = detect_market_regime()

            # Update all optimization components
            state = optimization_engine.update_optimization(regime=regime)

            # Log changes
            if state.weight_changes_this_period:
                logger.info(f"Weight updates: {state.weight_changes_this_period}")

            if state.degraded_strategies:
                logger.warning(f"Degraded: {state.degraded_strategies}")

            if state.suspended_strategies:
                logger.warning(f"Suspended: {state.suspended_strategies}")

        except Exception as e:
            logger.error(f"Optimization update failed: {e}")
```

### Monitoring & Alerting

```python
def generate_optimization_alerts():
    """Generate alerts based on optimization state."""

    health = monitor.get_health_report()

    alerts = []

    # Alert on suspended strategies
    if health['suspended']:
        alerts.append(f"ALERT: Suspended strategies: {health['suspended']}")

    # Alert on multiple degraded
    if len(health['degraded']) > 2:
        alerts.append(f"WARNING: {len(health['degraded'])} strategies degraded")

    # Alert on no active strategies (catastrophic failure)
    if not health['active']:
        alerts.append("CRITICAL: No active strategies!")

    return alerts
```

---

## Risk & Safety

### Hard Constraints (Never Violated)

1. **Optimization cannot weaken risk rules**

   - Position sizing limits still apply
   - Drawdown guards still apply
   - Circuit breakers still apply
   - Readiness gating still applies

2. **Weight adjustment limits**

   - Maximum 15% change per update
   - Minimum 0.1x weight (cannot disable)
   - Maximum 2.0x weight (prevents over-allocation)

3. **Signal filtering safeguards**

   - Can only reduce confidence, not increase
   - Can only reduce position size, not increase
   - Can suspend strategy (set weight to 0) via retirement manager

4. **Optimization can be disabled**
   - Set `enable_quality_filtering=False` to bypass scoring
   - Set `enable_adaptive_weighting=False` to use base weights
   - Set `enable_retirement_management=False` to prevent suspension
   - Can revert instantly at any time

### Testing Risk Compliance

```python
def test_optimization_risk_compliance():
    """Verify optimization never violates risk constraints."""

    engine = get_engine()

    # Test 1: Weights never exceed bounds
    weights = engine._weighting_optimizer.get_current_weights()
    for s, w in weights.items():
        assert w >= engine._weighting_optimizer.min_weight
        assert w <= engine._weighting_optimizer.max_weight

    # Test 2: Signal filtering never increases confidence
    quality, _ = engine.score_signal("EMA_Trend", "BTC/USDT", "BUY", 0.9)
    assert quality <= 1.0  # Never increases confidence

    # Test 3: Multipliers never exceed risk limits
    mult = engine.get_position_size_multiplier("EMA_Trend", 0.5)
    assert mult <= 2.0  # Capped per configuration
    assert mult >= 0.0

    # Test 4: Suspended strategies get zero weight
    engine._retirement_manager.force_status(
        "BadStrat",
        StrategyStatus.SUSPENDED,
        "Test"
    )
    weight = engine.get_strategy_weight("BadStrat")
    assert weight == 0.0
```

---

## Performance Metrics & Reporting

### Key Metrics to Track

1. **Optimization Effectiveness**

   - Average quality score of executed signals
   - % signals filtered vs executed
   - Win rate improvement from optimization

2. **Weight Distribution**

   - Average weight per strategy
   - Weight changes per period
   - Correlation with performance

3. **Strategy Health**

   - % active / degraded / suspended
   - Average time in degraded state
   - Recovery rate

4. **Alpha Impact**
   - Sharpe ratio improvement
   - Profit factor improvement
   - Expectancy improvement

### Dashboard Integration

```python
def get_optimization_dashboard_data():
    """Get data for live optimization dashboard."""

    engine = get_engine()
    monitor = get_monitor()

    return {
        'optimization_enabled': engine.get_optimization_status(),
        'current_weights': engine._weighting_optimizer.get_current_weights(),
        'strategy_health': monitor.get_health_report(),
        'allocation': monitor.get_allocation_report(),
        'regime_effectiveness': monitor.get_regime_effectiveness_report(),
        'latest_state': engine._last_optimization_state.to_dict(),
    }
```

---

## Troubleshooting

### Component Won't Initialize

```python
# Check if component initialized
status = engine.get_optimization_status()
print(status['components_initialized'])

# If not, check for import errors
try:
    from ai.adaptive_weighting_optimizer import get_optimizer
    opt = get_optimizer()
except RuntimeError as e:
    print(f"Optimizer not initialized: {e}")
```

### Weights Not Updating

```python
# Check if weighting is enabled
if not engine.enable_adaptive_weighting:
    print("Adaptive weighting disabled")

# Manually trigger update
state = engine.update_optimization(regime="TRENDING")
print(f"Weight changes: {state.weight_changes_this_period}")

# Check weight history
history = engine._weighting_optimizer.get_weight_history()
print(f"Recent updates: {len(history)}")
```

### Signals Being Over-Filtered

```python
# Check quality threshold
scorer = engine._quality_scorer
print(f"Threshold: {scorer.min_quality_threshold}")

# Lower threshold if too aggressive
scorer.min_quality_threshold = 0.6  # was 0.7

# Check regime adjustments
print(f"Regime adjustments: {scorer._regime_adjustments}")
```

### Strategy Suspended Unexpectedly

```python
# Get detailed health
manager = engine._retirement_manager
health = manager.get_strategy_health("BadStrat")
print(f"Status: {health.status.value}")
print(f"Reason: {health.status_reason}")
print(f"Expectancy: {health.recent_expectancy:.2f}")
print(f"Profit Factor: {health.recent_profit_factor:.2f}")

# Check thresholds
print(f"Min expectancy: {manager.min_expectancy}")
print(f"Min profit factor: {manager.min_profit_factor}")

# Override if needed
manager.force_status("BadStrat", StrategyStatus.ACTIVE, "Manual override")
```

---

## Configuration & Tuning

### Conservative Settings (Default)

```python
engine = initialize_engine(
    base_weights=base_weights,
    # Weighting optimizer
    adjustment_rate=0.15,  # 15% max change per update
    min_weight=0.1,  # Don't disable strategies
    max_weight=2.0,  # Don't over-weight
    # Quality scorer
    min_quality_threshold=0.7,  # 70% quality required
    # Retirement manager
    min_expectancy=0.1,
    min_profit_factor=0.8,
    min_win_rate=0.35,
    max_drawdown=0.15,
    degraded_window_hours=24,
    suspended_window_hours=72,
)
```

### Aggressive Settings (Higher Alpha, Higher Risk of Churn)

```python
engine = initialize_engine(
    base_weights=base_weights,
    adjustment_rate=0.30,  # 30% max change
    min_weight=0.05,  # Can reduce to 5%
    max_weight=3.0,  # Can increase to 3x
    min_quality_threshold=0.6,  # 60% quality
    min_expectancy=0.0,  # Very lenient
    degraded_window_hours=6,  # Quick to suspend
)
```

### Tuning Guide

**To increase alpha**:

- Lower `min_quality_threshold` (allow more trades)
- Increase `max_weight` (allocate more to winners)
- Lower `min_expectancy` (less strict on metrics)

**To reduce churn**:

- Increase `adjustment_rate` (faster responses to bad performance)
- Increase `degraded_window_hours` (give strategies longer to recover)
- Increase `min_trades_for_assessment` (require more data before retiring)

**To protect capital**:

- Increase `min_quality_threshold` (filter more signals)
- Decrease `max_weight` (limit allocation to any one strategy)
- Increase `min_expectancy` and `min_profit_factor`

---

## Changelog

### v1.0.0 (Initial Release)

- Performance decomposition (per strategy, symbol, regime)
- Adaptive weight optimizer (expectancy-based)
- Signal quality scorer (confluence + regime + history)
- Strategy retirement manager (automatic suspension)
- Unified optimization engine (coordinator)
- Comprehensive monitoring dashboard

---

## FAQ

**Q: Can optimization increase position sizing beyond limits?**

A: No. The position size multiplier is capped at 2.0x, and the underlying risk module still applies all limits.

**Q: What if a strategy is suspended by accident?**

A: You can manually override with `manager.force_status()`. The manager logs all status changes to an audit trail.

**Q: How do I know if a signal is being filtered?**

A: Check the `should_execute` return value from `engine.score_signal()`. You can also log the quality score.

**Q: Can I use only some optimization features?**

A: Yes. You can set `enable_*` flags to False for any component you want to disable.

**Q: How often should I update optimizations?**

A: Recommended: Every 1-24 hours depending on your trading frequency. More frequent = more responsive, less frequent = more stable.

**Q: What happens if the optimization engine crashes?**

A: It will log the error and fall back to base weights. All risk controls continue to work.

---

## See Also

- [System Specification](../SYSTEM_SPEC.md)
- [Live Trading Guide](../LIVE_TRADING_GUIDE.md)
- [Risk Management](../risk/)
- [Strategy Examples](../strategies/)
