# Phase 7 Quick Reference: Portfolio-Level Intelligence

## 🎯 One-Line Summary

**Portfolio-level correlation and exposure control to prevent correlated drawdowns and overexposure.**

---

## ⚡ Quick Start

### Enable Portfolio Risk (30 seconds)

1. **Edit** [`core/config.yaml`](core/config.yaml):

   ```yaml
   portfolio_risk:
     enabled: true # ← Change from false
   ```

2. **Run** platform:

   ```bash
   python3 main.py
   ```

3. **Monitor** logs for `[PORTFOLIO_RISK]` messages

---

## 📊 What It Does

| Feature                  | Purpose                        | Default Limit         |
| ------------------------ | ------------------------------ | --------------------- |
| **Correlation Control**  | Prevent correlated positions   | Max correlation: 0.85 |
| **Symbol Exposure**      | Limit concentration per symbol | 25% of equity         |
| **Directional Exposure** | Cap total long/short exposure  | 60% of equity         |
| **Regime Exposure**      | Balance across market regimes  | 50% of equity         |

---

## 🔧 Key Configuration

```yaml
portfolio_risk:
  enabled: false # Toggle on/off
  correlation_lookback: 100 # Bars for correlation
  max_pairwise_correlation: 0.85 # 0.0-1.0
  max_symbol_exposure_pct: 0.25 # % of equity
  max_directional_exposure_pct: 0.60 # % of equity
  max_regime_exposure_pct: 0.50 # % of equity
```

---

## 📍 Integration Point

```
Signal → Optimizer → Position Size → Drawdown Check
                                          ↓
                              🆕 Portfolio Risk ← HERE
                                          ↓
                                      Execution
```

**Placement**: After all sizing and risk checks, before order execution

---

## 🎮 How It Works

### Position Validation Flow

```python
allowed, reason, multiplier = portfolio_risk_manager.validate_new_position(
    symbol='BTC/USDT',
    side='long',
    proposed_size=1000.0,
    regime='TRENDING',
    strategy='ema_trend'
)

# Returns:
# - allowed: True/False (can trade open?)
# - reason: "symbol_exposure_reduced_0.75" or "high_correlation_with_ETH/USDT"
# - multiplier: 0.0 - 1.0 (size reduction factor)
```

### Final Position Size

```python
final_size = base_size × optimization_mult × portfolio_mult
#            ↑           ↑                   ↑
#            Position    Optimizer           Portfolio Risk
#            Sizer       (Phase 6)           (Phase 7)
```

**Portfolio multiplier range**: `[0.0, 1.0]` (can only reduce/block, never increase)

---

## 📈 Example Scenarios

### Scenario 1: High Correlation Block

```
Open: BTC/USDT long $10,000
Proposed: ETH/USDT long $5,000
Correlation: 0.92 (above 0.85 threshold)

Action: Reduce ETH position to $2,500
Reason: "high_correlation_with_BTC/USDT"
Multiplier: 0.50x
```

### Scenario 2: Symbol Exposure Limit

```
Equity: $100,000
Max Symbol Exposure: 25% = $25,000

Open: BTC/USDT long $15,000
Proposed: BTC/USDT long $12,000
Total would be: $27,000 (27%) ❌

Action: Reduce to $10,000 (total = $25,000)
Reason: "symbol_exposure_reduced_0.83"
Multiplier: 0.83x
```

### Scenario 3: Directional Exposure Limit

```
Equity: $100,000
Max Long Exposure: 60% = $60,000

Open Longs: $45,000 (45%)
Proposed: Long $20,000
Total would be: $65,000 (65%) ❌

Action: Reduce to $15,000 (total = $60,000)
Reason: "directional_exposure_reduced_0.75"
Multiplier: 0.75x
```

### Scenario 4: Regime Exposure Limit

```
Equity: $100,000
Max Regime Exposure: 50% = $50,000

TRENDING Positions: $40,000 (40%)
Proposed: TRENDING $15,000
Total would be: $55,000 (55%) ❌

Action: Reduce to $10,000 (total = $50,000)
Reason: "regime_exposure_reduced_0.67"
Multiplier: 0.67x
```

---

## 📊 Monitoring

### Check Portfolio Statistics

```python
stats = portfolio_risk_manager.get_stats()

{
    'enabled': True,
    'open_positions': 3,
    'total_exposure': 0.35,        # 35% equity deployed
    'long_exposure': 0.30,         # 30% long
    'short_exposure': 0.05,        # 5% short
    'max_correlation': 0.72,       # Highest correlation
    'blocked_trades': 5,           # Total blocked
    'reduced_trades': 12,          # Total reduced
    'regime_exposures': {
        'TRENDING': 0.25,
        'RANGING': 0.10
    },
    'symbol_exposures': {
        'BTC/USDT': 0.15,
        'ETH/USDT': 0.10,
        'LTC/USDT': 0.10
    }
}
```

### Log Messages

```
[PORTFOLIO_RISK] Position registered: BTC/USDT long size=0.500000 regime=TRENDING
[PORTFOLIO_RISK] Validating position
[PORTFOLIO_RISK] Trade size reduced: multiplier=0.67 reason=symbol_exposure_reduced_0.67
[PORTFOLIO_RISK] Trade blocked: ETH/USDT reason=high_correlation_with_BTC/USDT
[PORTFOLIO_RISK] Position unregistered: BTC/USDT
```

---

## 🔒 Safety Guarantees

| Safety Feature                      | Status                                       |
| ----------------------------------- | -------------------------------------------- |
| **Disabled by default**             | ✅ Must opt-in                               |
| **Cannot increase leverage**        | ✅ Multiplier ≤ 1.0                          |
| **Cannot override drawdown guards** | ✅ Independent check                         |
| **Backward compatible**             | ✅ Zero behavior change when disabled        |
| **Graceful degradation**            | ✅ Trading continues if portfolio risk fails |

---

## 🧪 Testing

### Run Portfolio Risk Tests

```bash
pytest tests/test_portfolio_risk_manager.py -v
# 32 passed in 1.61s ✅
```

### Verify Backward Compatibility

```bash
pytest tests/test_optimization_system.py -q
# 36 passed, 1 skipped in 3.46s ✅

python3 -c "import main"
# ✓ main.py imports successfully ✅
```

---

## 📁 File Locations

| File                                                                           | Purpose             | Lines |
| ------------------------------------------------------------------------------ | ------------------- | ----- |
| [`risk/portfolio_risk_manager.py`](risk/portfolio_risk_manager.py)             | Main implementation | 640   |
| [`main.py`](main.py)                                                           | Integration points  | +100  |
| [`tests/test_portfolio_risk_manager.py`](tests/test_portfolio_risk_manager.py) | Test suite          | 670   |

---

## 🎯 Core Methods

```python
# Update market data for correlation
portfolio_risk_manager.update_market_data(symbol, candles)

# Register position after execution
portfolio_risk_manager.register_position(
    symbol='BTC/USDT',
    side='long',
    size=0.5,
    regime='TRENDING',
    strategy='ema_trend'
)

# Validate before execution (PRIMARY INTEGRATION)
allowed, reason, multiplier = portfolio_risk_manager.validate_new_position(
    symbol='ETH/USDT',
    side='long',
    proposed_size=1000.0,
    regime='TRENDING',
    strategy='pattern_bull_flag'
)

# Unregister when position closes
portfolio_risk_manager.unregister_position(symbol='BTC/USDT')

# Get correlation between symbols
correlation = portfolio_risk_manager.get_correlation('BTC/USDT', 'ETH/USDT')

# Get exposure metrics
symbol_exposure = portfolio_risk_manager.get_symbol_exposure('BTC/USDT')
long_exposure = portfolio_risk_manager.get_directional_exposure('long')
regime_exposure = portfolio_risk_manager.get_regime_exposure('TRENDING')

# Get portfolio statistics
stats = portfolio_risk_manager.get_stats()
```

---

## 🚨 Common Issues

### Issue 1: Portfolio Risk Not Working

**Symptom**: No `[PORTFOLIO_RISK]` logs

**Fix**: Check `core/config.yaml`:

```yaml
portfolio_risk:
  enabled: true # Must be true
```

### Issue 2: All Trades Blocked

**Symptom**: Every trade gets `reason=symbol_exposure_limit`

**Fix**: Limits too restrictive, increase:

```yaml
portfolio_risk:
  max_symbol_exposure_pct: 0.35 # Increase from 0.25
  max_directional_exposure_pct: 0.75 # Increase from 0.60
```

### Issue 3: Correlation Not Computing

**Symptom**: `max_correlation: 0.0` in stats

**Fix**: Need more data, reduce lookback:

```yaml
portfolio_risk:
  correlation_lookback: 50 # Reduce from 100
```

---

## 📖 Advanced Configuration

### Conservative (High Protection)

```yaml
portfolio_risk:
  enabled: true
  max_pairwise_correlation: 0.70 # Stricter correlation
  max_symbol_exposure_pct: 0.15 # Lower symbol limit
  max_directional_exposure_pct: 0.50 # Lower directional limit
  max_regime_exposure_pct: 0.40 # Lower regime limit
```

### Aggressive (Lower Protection)

```yaml
portfolio_risk:
  enabled: true
  max_pairwise_correlation: 0.95 # Looser correlation
  max_symbol_exposure_pct: 0.40 # Higher symbol limit
  max_directional_exposure_pct: 0.80 # Higher directional limit
  max_regime_exposure_pct: 0.70 # Higher regime limit
```

### Balanced (Default)

```yaml
portfolio_risk:
  enabled: true
  max_pairwise_correlation: 0.85
  max_symbol_exposure_pct: 0.25
  max_directional_exposure_pct: 0.60
  max_regime_exposure_pct: 0.50
```

---

## ✅ Checklist: First-Time Setup

- [ ] Enable in [`core/config.yaml`](core/config.yaml)
- [ ] Run tests: `pytest tests/test_portfolio_risk_manager.py -v`
- [ ] Start platform: `python3 main.py`
- [ ] Verify logs show `[PORTFOLIO_RISK]` messages
- [ ] Monitor `get_stats()` for exposure metrics
- [ ] Adjust limits if needed based on blocked/reduced trades

---

## 📚 Related Documentation

- [Full Implementation Guide](PHASE7_PORTFOLIO_INTELLIGENCE.md) - Comprehensive details
- [Optimization System](LIVE_TRADING_ACTIVATION_IMPLEMENTATION.md) - Phase 6 multiplier composition
- [Drawdown Guards](risk/drawdown_guard.py) - Position-level risk controls
- [Position Sizing](risk/position_sizing.py) - Base position sizing

---

**Phase 7 Status**: ✅ Production Ready

Portfolio-level intelligence is operational and ready for production deployment!
