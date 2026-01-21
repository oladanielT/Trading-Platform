# Strategy Promotion Approvals

File-based approval system for human-in-the-loop strategy promotion.

## Directory Structure

```
approvals/
├── pending/     - Strategies awaiting approval
├── approved/    - Approved strategies (automatically archived)
└── rejected/    - Rejected strategies (for audit trail)
```

## Approval Workflow

### 1. Strategy Submits for Approval

When a strategy is submitted via `StrategyPromotionManager.submit_candidate()`, an approval file is created:

```
approvals/pending/<strategy_id>_<timestamp>.yaml
```

### 2. Human Reviews Approval File

The approval file contains all validation metrics and scorecards. Review and decide:

**To Approve:**

```bash
# Move to approved folder (name unchanged)
mv approvals/pending/strategy_xyz_20240101_120000.yaml approvals/approved/

# System will detect approval and transition strategy to APPROVED state
```

**To Reject:**

```bash
# Move to rejected folder
mv approvals/pending/strategy_xyz_20240101_120000.yaml approvals/rejected/

# System will detect rejection and transition strategy to REJECTED state
```

### 3. System Processes Approval

The promotion manager monitors the approvals directory and automatically:

- Detects files moved to `approved/` → calls `approve(strategy_id)`
- Detects files moved to `rejected/` → calls `reject(strategy_id)`

## Approval File Format

```yaml
strategy_id: "momentum_regime_v2"
submitted_at: "2024-01-01T12:00:00"
current_state: "RESEARCH"
requested_transition: "RESEARCH -> APPROVED"

# Validation results from Offline Alpha Lab
validation_results:
  walk_forward:
    passed: true
    is_overfitting_score: 0.12
    oos_sharpe: 1.8

  robustness:
    passed: true
    parameter_sensitivity: 0.85
    noise_resilience: 0.92

  performance_metrics:
    sharpe_ratio: 2.1
    max_drawdown: 0.08
    win_rate: 0.58
    expectancy: 1.4
    total_trades: 450

  regime_performance:
    TRENDING:
      sharpe: 2.8
      trades: 200
    RANGING:
      sharpe: 1.2
      trades: 150
    VOLATILE:
      sharpe: 0.8
      trades: 100

# Human decision (DO NOT MODIFY - move file instead)
status: "pending" # pending, approved, rejected
decision_by: ""
decision_at: ""
decision_notes: ""
```

## Safety Rules

1. **No Auto-Promotion**: System NEVER moves files automatically to approved/
2. **Human Decision Required**: Only human operators move approval files
3. **Immutable History**: Rejected approvals kept for audit trail
4. **One-Way Transitions**: Cannot un-reject or un-approve

## Monitoring Approvals

```bash
# Check pending approvals
ls -lh approvals/pending/

# View approval file details
cat approvals/pending/strategy_xyz_20240101_120000.yaml

# Check approval history
ls -lh approvals/approved/
ls -lh approvals/rejected/
```

## Integration with Promotion Manager

```python
from ai.strategy_promotion_manager import StrategyPromotionManager

# Initialize
manager = StrategyPromotionManager(config)

# Submit strategy for approval (creates pending file)
result = manager.submit_candidate(
    strategy_id="momentum_v2",
    validation_results=alpha_lab_results
)

# Human reviews and moves file to approved/ or rejected/

# System polls and processes
manager.process_pending_approvals()  # Called by main loop
```

## Best Practices

### Review Checklist

- [ ] Walk-forward validation passed (no overfitting)
- [ ] Robustness tests passed (stable under perturbations)
- [ ] Sharpe ratio > 1.5 across all regimes
- [ ] Max drawdown < 15%
- [ ] Win rate > 40%
- [ ] Sufficient trade count (>100 trades)
- [ ] Regime-specific performance acceptable
- [ ] Low correlation with existing strategies
- [ ] Economic rationale sound

### Common Rejection Reasons

- **Overfitting**: IS/OOS degradation >20%
- **Fragility**: High parameter sensitivity
- **Insufficient Data**: <100 trades in validation
- **Regime Mismatch**: Poor performance in >1 regime
- **High Correlation**: >0.85 correlation with existing strategies

## Emergency Override

If the system malfunctions, manually edit the strategy state:

```python
# Emergency manual state change (use with caution)
manager.force_state_change(
    strategy_id="momentum_v2",
    new_state=PromotionState.REJECTED,
    reason="Manual override: system malfunction"
)
```

**WARNING**: Force state changes bypass all safety checks. Use only in emergencies.
