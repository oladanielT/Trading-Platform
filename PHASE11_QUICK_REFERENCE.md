# Phase 11 Quick Reference

## Human-in-the-Loop Strategy Promotion & Capital Graduation

### 🎯 What It Does

Bridges Phase 10 (offline validation) and live trading with mandatory human approval and graduated capital allocation.

---

## 📋 Core Concepts

### Promotion States (FSM)

```
RESEARCH → APPROVED → SHADOW → PAPER → MICRO_LIVE → FULL_LIVE → RETIRED
   0%         0%        0%       0%        5%          100%        0%
```

### Capital Allocation

- **SHADOW/PAPER**: 0% (no execution)
- **MICRO_LIVE**: 1-5% max (configurable, default 5%)
- **FULL_LIVE**: 100% (subject to portfolio risk limits)

### Safety Rules

1. NO auto-promotion (human approval required)
2. NO state skipping (sequential only)
3. NO capital leaks (graduation enforced)
4. Automatic demotion if performance degrades

---

## 🚀 Quick Start

### 1. Submit Strategy for Approval

```python
from ai.strategy_promotion_manager import StrategyPromotionManager

manager = StrategyPromotionManager(config)

# From Alpha Lab report
manager.submit_candidate(
    strategy_id="momentum_v2",
    report_path="alpha_reports/validated/momentum_v2.json"
)

# Or direct (for testing)
manager.submit_candidate(
    strategy_id="momentum_v2",
    validation_results={'qualified': True}
)
```

### 2. Human Approves

```python
# API method
manager.approve("momentum_v2", approved_by="trader_name")

# OR file-based (move file)
# mv approvals/pending/momentum_v2_*.yaml approvals/approved/
```

### 3. Promote Through States

```python
# Shadow mode (signals tracked, no execution)
manager.promote_to_shadow("momentum_v2")

# Paper trading (simulated fills)
manager.promote_to_paper("momentum_v2")

# Micro-live (5% capital, real trades)
manager.promote_to_micro_live("momentum_v2")

# Full live (100% capital, portfolio rules apply)
manager.promote_to_full_live("momentum_v2")
```

### 4. Capital Graduation

```python
from ai.capital_graduation import CapitalGraduation

graduation = CapitalGraduation(config)
state = manager.get_state("momentum_v2")

# Check if execution allowed
if graduation.is_execution_allowed(state):
    # Get capital limit
    allowed = graduation.get_allowed_capital(
        state=state,
        total_equity=100000.0,
        proposed_size=10000.0
    )
    # If state=MICRO_LIVE: allowed=5000 (5% of 100k)
```

### 5. Monitor with Gatekeeper

```python
from ai.promotion_gatekeeper import PromotionGatekeeper

gatekeeper = PromotionGatekeeper(config)

# Evaluate performance
decision = gatekeeper.evaluate("momentum_v2", {
    'total_trades': 50,
    'max_drawdown': 0.08,
    'expectancy': 1.2,
    'win_rate': 0.55,
    'consecutive_losses': 2
})

if decision.should_demote:
    manager.demote("momentum_v2", decision.reason.value)
```

### 6. Shadow Execution

```python
from ai.shadow_execution import ShadowExecutor

executor = ShadowExecutor()

# Generate signals without executing
result = executor.evaluate(
    strategy_id="momentum_v2",
    strategy_func=my_strategy_function,
    market_data=current_data,
    current_price=50000.0
)

# Calculate metrics
metrics = executor.calculate_metrics("momentum_v2")
```

---

## ⚙️ Configuration

```yaml
# core/config.yaml
strategy_promotion:
  enabled: false # Must enable to use
  require_human_approval: true # Never disable
  auto_demote: true
  max_micro_live_capital: 0.05 # 5% max

  demotion_criteria:
    max_drawdown: 0.15 # 15%
    min_expectancy: 0.0 # Must be positive
    min_win_rate: 0.40 # 40%
    max_consecutive_losses: 5
    min_trades_before_eval: 20
    max_correlation: 0.85
    regime_mismatch_threshold: 10
```

---

## 🔍 Checking Status

### Query Strategy State

```python
# Get current state
state = manager.get_state("momentum_v2")
print(state)  # PromotionState.MICRO_LIVE

# Check if live
is_live = manager.is_live("momentum_v2")
print(is_live)  # True if MICRO_LIVE or FULL_LIVE

# Get state history
history = manager.get_history("momentum_v2")
for record in history:
    print(f"{record['timestamp']}: {record['action']} -> {record['state']}")
```

### Get All Strategies by State

```python
# Get all live strategies
live_strategies = manager.get_strategies_by_state(PromotionState.MICRO_LIVE)

# Get all pending approval
pending = manager.get_strategies_by_state(PromotionState.RESEARCH)
```

### Diagnostics

```python
# Manager diagnostics
manager_diag = manager.get_diagnostics()

# Gatekeeper criteria
criteria = gatekeeper.get_criteria()
print(f"Max drawdown: {criteria.max_drawdown}")

# Capital limits
limits = graduation.get_diagnostics()
```

---

## ⚠️ Error Handling

### Common Exceptions

```python
from ai.strategy_promotion_manager import (
    InvalidTransitionError,
    StrategyNotFoundError
)

try:
    # Try to skip state
    manager.promote_to_paper("momentum_v2")  # Currently in APPROVED
except InvalidTransitionError as e:
    print(f"Invalid transition: {e}")
    # Must go to SHADOW first
    manager.promote_to_shadow("momentum_v2")
    manager.promote_to_paper("momentum_v2")  # Now works

try:
    state = manager.get_state("nonexistent")
except StrategyNotFoundError:
    print("Strategy not found")
```

---

## 📊 Integration Example

```python
# Full trading loop integration
def trading_loop():
    manager = StrategyPromotionManager(config)
    graduation = CapitalGraduation(config)
    gatekeeper = PromotionGatekeeper(config)

    for strategy_id in active_strategies:
        # Get strategy state
        state = manager.get_state(strategy_id)

        # Check if can execute
        if not graduation.is_execution_allowed(state):
            continue  # Skip shadow/paper strategies

        # Generate signal
        signal = strategy.generate_signal(market_data)

        if signal:
            # Calculate position size
            proposed_size = calculate_position_size(signal)

            # Apply capital graduation
            allowed_size = graduation.get_allowed_capital(
                state=state,
                total_equity=account.equity,
                proposed_size=proposed_size
            )

            # Apply portfolio risk limits (final check)
            final_size = min(
                allowed_size,
                portfolio_risk.check_position_size(...)
            )

            # Execute
            execute_trade(strategy_id, final_size)

        # Monitor performance
        metrics = get_live_metrics(strategy_id)
        decision = gatekeeper.evaluate(strategy_id, metrics)

        if decision.should_demote:
            logger.warning(f"Demoting {strategy_id}: {decision.reason}")
            manager.demote(strategy_id, decision.reason.value)
```

---

## 🔒 Safety Checklist

Before Going Live:

- [ ] `require_human_approval: true` in config
- [ ] `enabled: false` initially (enable after testing)
- [ ] Approval process tested (file-based or API)
- [ ] Capital limits verified (max 5% for MICRO_LIVE)
- [ ] Demotion criteria configured
- [ ] Shadow execution tested (no real trades)
- [ ] Paper trading tested (simulated fills)
- [ ] Gatekeeper monitors all live strategies

---

## 📝 File Locations

```
ai/
├── strategy_promotion_manager.py  # FSM & approval
├── shadow_execution.py            # Signal tracking
├── capital_graduation.py          # Capital limits
└── promotion_gatekeeper.py        # Auto-demotion

approvals/
├── pending/     # Awaiting approval
├── approved/    # Approved strategies
└── rejected/    # Rejected strategies

core/config.yaml  # Configuration (lines 308-395)

tests/
├── test_strategy_promotion_manager.py
├── test_shadow_execution.py
├── test_capital_graduation.py
└── test_promotion_gatekeeper.py  # 18/18 passing ✅
```

---

## 🎓 Demotion Reasons

```python
from ai.promotion_gatekeeper import DemotionReason

# All possible demotion triggers
DemotionReason.EXCESSIVE_DRAWDOWN     # DD > 15%
DemotionReason.NEGATIVE_EXPECTANCY    # Expectancy < 0
DemotionReason.WIN_RATE_COLLAPSE      # WR < 40%
DemotionReason.CONSECUTIVE_LOSSES     # Losses >= 5
DemotionReason.HIGH_CORRELATION       # Corr > 0.85
DemotionReason.REGIME_MISMATCH        # Mismatches >= 10
DemotionReason.INSUFFICIENT_TRADES    # Trades < 20
DemotionReason.MANUAL                 # Human decision
```

---

## 💡 Pro Tips

1. **Start Slow**: Use SHADOW mode for 24+ hours before PAPER
2. **Monitor Closely**: Check demotion criteria daily in MICRO_LIVE
3. **Low Correlation**: Aim for <0.70 correlation with existing strategies
4. **Regime Awareness**: Verify performance in all 3 regimes before FULL_LIVE
5. **Capital Conservative**: Consider using <5% for MICRO_LIVE initially
6. **Audit Trail**: Never delete state_history - critical for debugging
7. **Test Demotion**: Manually trigger demotion to verify it works

---

## 🚨 Common Pitfalls

❌ **Trying to skip states** → Use sequential promotion only  
❌ **Disabling human approval** → Never set `require_human_approval: false`  
❌ **Ignoring demotion signals** → Let gatekeeper do its job  
❌ **Not testing shadow mode** → Always verify signals before paper  
❌ **Rushing to FULL_LIVE** → Spend time at each level  
❌ **Exceeding capital limits** → Respect graduation rules  
❌ **No regime testing** → Validate in TRENDING, RANGING, VOLATILE

---

## 📚 See Also

- **Full Documentation**: `PHASE11_COMPLETE.md`
- **Phase 10 Reference**: `PHASE10_QUICK_REFERENCE.md`
- **Config Guide**: `core/config.yaml`
- **Approval Workflow**: `approvals/README.md`
- **Working Tests**: `tests/test_promotion_gatekeeper.py`

---

**Version**: Phase 11 v1.0  
**Status**: Implementation Complete  
**Safety Level**: Maximum  
**Ready for**: Integration & Testing
