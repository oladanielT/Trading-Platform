# Phase 10 Quick Reference

## 🎯 One-Liner

**Offline research lab for discovering robust alpha and rejecting overfit strategies BEFORE live trading.**

---

## ⚡ Quick Start (30 seconds)

1. **Enable in config** (manual only):

   ```yaml
   alpha_lab:
     enabled: true # Change from false
   ```

2. **Run experiment**:

   ```python
   from ai.offline_alpha_lab import OfflineAlphaLab
   import yaml

   with open('core/config.yaml') as f:
       config = yaml.safe_load(f)

   lab = OfflineAlphaLab(config)

   strategies = [{
       'name': 'test_strategy',
       'func': my_strategy_func,
       'params': {'param1': 10}
   }]

   results = lab.run_experiments(strategies, historical_data)
   ```

3. **Check results**:
   ```bash
   ls alpha_reports/validated/
   ls alpha_reports/rejected/
   ```

---

## 📊 What Gets Tested

| Test Type              | What It Checks                | Pass Threshold  |
| ---------------------- | ----------------------------- | --------------- |
| Walk-Forward           | OOS consistency               | ≥ 70%           |
| Parameter Perturbation | Sensitivity to ±30% changes   | Stability ≥ 0.7 |
| Noise Injection        | Resilience to 1% price jitter | Stability ≥ 0.7 |
| Monte Carlo            | Trade sequence dependency     | Stability ≥ 0.7 |
| Regime Shuffle         | Order independence            | Stability ≥ 0.7 |

---

## 🔧 Qualification Criteria (STRICT)

```python
min_expectancy: 0.0       # Must be positive
min_sharpe: 1.2           # Risk-adjusted returns
max_drawdown: 0.20        # 20% capital preservation
min_robustness: 0.7       # Stability score
min_oos_consistency: 0.70 # 70% OOS performance
min_trades: 30            # Statistical significance
```

**ALL must pass** - single failure = rejection

---

## 📁 File Structure

```
ai/
├── offline_alpha_lab.py        # Main orchestrator
├── walk_forward_engine.py      # OOS validation
├── robustness_tests.py         # Stability testing
└── performance_decomposer.py   # Enhanced metrics

alpha_reports/
├── validated/                  # ✅ Passed all tests
├── rejected/                   # ❌ Failed criteria
└── summary_*.json              # Experiment summaries

tests/
├── test_offline_alpha_lab.py
├── test_walk_forward_engine.py
└── test_robustness_tests.py
```

---

## 🚫 What It NEVER Does

❌ NO live trading execution  
❌ NO config auto-modification  
❌ NO auto-promotion of strategies  
❌ NO runtime hooks  
❌ NO position sizing  
❌ NO order placement

**Offline means offline.**

---

## 🎮 Usage Examples

### Example 1: Test Single Strategy

```python
strategies = [{
    'name': 'ema_cross',
    'func': lambda data, params: ema_crossover(data, params['fast'], params['slow']),
    'params': {'fast': 12, 'slow': 26}
}]

results = lab.run_experiments(strategies, data)
print(results['validated_strategies'])  # Passed
print(results['rejected_strategies'])   # Failed
```

### Example 2: Batch Testing

```python
strategies = []
for fast in [10, 12, 15]:
    for slow in [20, 26, 30]:
        strategies.append({
            'name': f'ema_{fast}_{slow}',
            'func': ema_strategy,
            'params': {'fast': fast, 'slow': slow}
        })

results = lab.run_experiments(strategies, data)
```

### Example 3: Walk-Forward Only

```python
from ai.walk_forward_engine import WalkForwardEngine

engine = WalkForwardEngine()
wf_results = engine.run(
    strategy_func=my_strategy,
    data=historical_data,
    regime_labels=['TRENDING', 'RANGING', ...]
)

print(f"OOS Consistency: {wf_results['overall_consistency']:.2%}")
print(f"Passed: {wf_results['passed']}")
```

---

## 📈 Report Contents

Each validated/rejected strategy gets a JSON report with:

```json
{
  "strategy_name": "...",
  "qualified": true/false,
  "rejection_reasons": [...],
  "performance": {
    "expectancy": 0.45,
    "sharpe": 1.85,
    "max_drawdown": 0.12,
    "win_rate": 0.58
  },
  "validation": {
    "oos_consistency": 0.78,
    "robustness_score": 0.82
  },
  "regime_performance": {
    "TRENDING": {...},
    "RANGING": {...}
  },
  "recommendations": [...]
}
```

---

## ✅ Run Tests

```bash
# All Phase 10 tests
pytest tests/test_walk_forward_engine.py \
       tests/test_robustness_tests.py \
       tests/test_offline_alpha_lab.py -v

# Quick test
pytest tests/test_offline_alpha_lab.py::TestIsolation -v
```

**Expected**: 50 tests, 100% passing

---

## 🔍 Verify Isolation

```bash
# Should return ZERO matches
grep -r "from execution\|OrderManager\|Broker" ai/offline_alpha_lab.py
grep -r "from execution\|OrderManager\|Broker" ai/walk_forward_engine.py
grep -r "from execution\|OrderManager\|Broker" ai/robustness_tests.py
```

---

## 🎯 Decision Tree

```
Strategy Idea
     ↓
Run in Alpha Lab
     ↓
  Validated? ──NO──→ Review rejection reasons → Improve → Retry
     ↓YES
Human Review
     ↓
  Approved? ──NO──→ Archive
     ↓YES
Paper Trading (manual promotion)
     ↓
Monitoring
     ↓
  Working? ──NO──→ Return to Alpha Lab
     ↓YES
Live Trading (manual promotion)
```

**Key**: Human in the loop at every promotion step.

---

## 📚 Key Modules

### OfflineAlphaLab

```python
lab = OfflineAlphaLab(config)
results = lab.run_experiments(strategies, data)
diag = lab.get_diagnostics()
```

### WalkForwardEngine

```python
engine = WalkForwardEngine(window_config)
results = engine.run(strategy_func, data, regime_labels)
diag = engine.get_diagnostics()
```

### RobustnessTestSuite

```python
suite = RobustnessTestSuite(robustness_config)
results = suite.run_full_suite(strategy_func, data, params)
diag = suite.get_diagnostics()
```

---

## 🚨 Common Issues

### "Alpha Lab is disabled"

**Solution**: Set `alpha_lab.enabled: true` in config

### "Insufficient data"

**Solution**: Need at least 130 samples (100 train + 30 test)

### "All strategies rejected"

**Solution**: Criteria are STRICT - review rejection reasons

### "Reports not found"

**Solution**: Check `alpha_reports/` directory (created automatically)

---

## 🎓 Best Practices

1. **Start with robust baseline** - Test known good strategy first
2. **Review rejections** - Learn from failures
3. **Use regime labels** - Better performance decomposition
4. **Batch test variations** - Find optimal parameters
5. **Never skip human review** - Validation ≠ approval

---

## 🔗 Integration Points

| System             | Interaction            |
| ------------------ | ---------------------- |
| Live Trading       | NONE (isolated)        |
| OptimizationEngine | NONE (no auto-updates) |
| PortfolioRisk      | NONE (offline only)    |
| Execution          | NONE (no orders)       |
| Reports            | READ ONLY (JSON files) |

---

**Status**: ✅ COMPLETE  
**Tests**: 50/50 passing  
**Safety**: Fully isolated
