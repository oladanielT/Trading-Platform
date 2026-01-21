# Phase 9 Quick Reference: Execution Reality

## 🎯 One-Line Summary

**Apply realistic market constraints (slippage, liquidity, fill probability) to reduce position sizes in adverse conditions, never increase them.**

---

## ⚡ Quick Start (30 seconds)

### Enable Execution Reality

1. **Edit** [core/config.yaml](core/config.yaml):

   ```yaml
   execution_reality:
     enabled: true # ← Change from false
   ```

2. **Run** platform:

   ```bash
   python3 main.py
   ```

3. **Monitor** logs for `[EXECUTION_REALITY]` messages

---

## 📊 What It Does

| Constraint            | When Applied         | Effect             | Default      |
| --------------------- | -------------------- | ------------------ | ------------ |
| **Liquidity Limit**   | Order > 2% of volume | Reduce size        | 0.02         |
| **Fill Probability**  | Expected fill < 60%  | Reject trade       | 0.6          |
| **Slippage Model**    | Always               | Estimate impact    | conservative |
| **Volatility Impact** | High volatility      | Reduce probability | 2.0x         |
| **Size Impact**       | Large orders         | Increase slippage  | 100 bps      |

---

## 🔌 Integration Points

**Location in Order Flow** (FINAL gate):

```
Strategy → Filter → Size → Drawdown → Portfolio →
Regime → EXECUTION REALITY ← HERE → Order
```

**Logging**:

- `[EXECUTION_REALITY] Evaluating order feasibility`
- `[EXECUTION_REALITY] Position size reduced: X → Y`
- `[EXECUTION_REALITY] Trade rejected: reason`

---

## ⚙️ Configuration

### Minimal (Use Defaults)

```yaml
execution_reality:
  enabled: true
```

### Custom

```yaml
execution_reality:
  enabled: true
  max_size_vs_volume: 0.02 # Max 2% of hourly volume
  min_fill_probability: 0.6 # Reject if <60%
  slippage_model: conservative # or 'moderate'/'aggressive'
  latency_penalty_ms: 500 # Extra latency buffer
```

### Models

- **conservative** (default): Pessimistic, safe, realistic
- **moderate**: Middle ground, for backtesting
- **aggressive**: Optimistic, for stress testing

---

## 📈 Expected Behavior

### Normal Market (Liquid, 1% Volatility)

- Small orders (0.1% volume): Approved, 10 bps slippage
- Medium orders (1% volume): Approved, 15 bps slippage
- Large orders (3% volume): Reduced, 25 bps slippage

### Stressed Market (Illiquid, 5% Volatility)

- Small orders (0.5% volume): Approved, 25 bps slippage
- Medium orders (2% volume): Rejected, fill prob 45%
- Large orders (10% volume): Rejected, fill prob 20%

---

## 🔐 Safety Guarantees

✅ **Size never increases**  
✅ **Zero-size orders skip safely**  
✅ **Disabled mode is transparent**  
✅ **No signal modification**  
✅ **Respects all upstream risk gates**

---

## 📊 Monitoring

### Get Status

```python
status = platform.get_execution_reality_status()
print(status['metrics']['approval_rate'])      # 95%
print(status['metrics']['avg_estimated_slippage_bps'])  # 12.5
```

### Check Logs

```bash
grep '\[EXECUTION_REALITY\]' logs/trading.log | tail -20
```

### View Diagnostics

```python
diag = platform.get_execution_reality_status()
print(f"Orders: {diag['metrics']['total_orders_evaluated']}")
print(f"Rejections: {diag['metrics']['total_orders_rejected']}")
for symbol, health in diag['symbol_health'].items():
    print(f"{symbol}: {health['approval_rate']:.1%}")
```

---

## 🎛️ Tuning

### Too Conservative (Rejecting Good Trades)

```yaml
execution_reality:
  min_fill_probability: 0.6 # ← Lower this (was 0.7)
  max_size_vs_volume: 0.03 # ← Increase (was 0.02)
  slippage_model: moderate # ← Switch to moderate
```

### Too Aggressive (Not Protecting)

```yaml
execution_reality:
  min_fill_probability: 0.7 # ← Raise this (was 0.6)
  max_size_vs_volume: 0.01 # ← Decrease (was 0.02)
  slippage_model: conservative # ← Already safe
```

---

## 🧪 Verification

### Run Tests

```bash
python3 -m pytest tests/test_execution_reality_engine.py -v
# Expected: 27/27 passed
```

### Smoke Test

```python
from execution.execution_reality_engine import ExecutionRealityEngine
engine = ExecutionRealityEngine({'enabled': True})
result = engine.evaluate_order(
    symbol='BTC/USDT',
    side='buy',
    requested_size=1.0,
    market_snapshot={
        'close': 100.0,
        'volume': 10000.0,
        'atr_percent': 1.0,
    }
)
print(f"Approved: {result['approved_size']:.6f}")
print(f"Slippage: {result['expected_slippage_bps']:.1f} bps")
```

---

## 📋 Configuration Defaults

| Setting              | Default      | Safe Range                       |
| -------------------- | ------------ | -------------------------------- |
| enabled              | false        | true/false                       |
| max_size_vs_volume   | 0.02         | 0.01-0.05                        |
| min_fill_probability | 0.6          | 0.5-0.8                          |
| slippage_model       | conservative | conservative/moderate/aggressive |
| latency_penalty_ms   | 500          | 0-2000                           |

---

## 🚀 Recommended Setups

### Live Trading

```yaml
execution_reality:
  enabled: true
  slippage_model: conservative
  max_size_vs_volume: 0.01
  min_fill_probability: 0.7
```

### Backtesting

```yaml
execution_reality:
  enabled: true
  slippage_model: moderate
  max_size_vs_volume: 0.02
  min_fill_probability: 0.6
```

### Development

```yaml
execution_reality:
  enabled: false
```

---

## 🔍 Common Issues

| Issue                  | Cause            | Solution                      |
| ---------------------- | ---------------- | ----------------------------- |
| 50% rejections         | Too conservative | Lower `min_fill_probability`  |
| Sizes 20% of requested | Illiquid market  | Increase `max_size_vs_volume` |
| Doesn't help           | Disabled         | Set `enabled: true`           |

---

## 📞 Status Check

```python
# Quick health check
diag = platform.get_execution_reality_status()
assert diag['enabled'] == True
assert diag['metrics']['approval_rate'] > 0.8
print("✓ Execution Reality is healthy")
```

---

## 🎓 How It Works (60 seconds)

1. **Liquidity Check**: Is order > 2% of recent volume?

   - YES: Reduce size
   - NO: Continue

2. **Slippage Estimate**: Calculate expected slippage

   - Larger orders = more slippage
   - Higher volatility = more slippage

3. **Fill Probability**: Estimate chance of full fill

   - Decreases with size and volatility
   - Bounded 0-100%

4. **Final Decision**:
   - If probability < 60%: REJECT
   - Otherwise: APPROVE reduced size

---

## 🏁 Key Points

- ✅ **Conservative by design** - protects capital
- ✅ **Disabled by default** - zero impact when off
- ✅ **Final gate** - acts right before order submission
- ✅ **Reversible** - set `enabled: false` to disable
- ✅ **Auditable** - full execution history logged
- ✅ **Safe** - 27 tests verify guarantees

---

**Phase 9 is ACTIVE and READY for use**. Enable when you need realistic market simulation.
