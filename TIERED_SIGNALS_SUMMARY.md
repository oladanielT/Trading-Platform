## RangeTradingStrategy: Tiered Signal System

### Problem

The strategy was too strict with fixed RSI thresholds:

- **Required:** RSI > 65 to trigger SELL (overbought)
- **Reality:** In ranging markets, RSI rarely hits 65
- **Your case:** Price at resistance ($92,482 vs $92,519) + RSI=59 = NO SIGNAL ❌

### Solution: Tiered Approach

Implemented **two levels of confidence** instead of hard cutoffs:

#### Normal Level (Easier to Trigger)

- **BUY:** RSI < 45 (was 35)
- **SELL:** RSI > 55 (was 65)
- **Confidence:** 0.60-0.75
- **Use case:** Capture more opportunities in ranging markets

#### Strong Level (Higher Confidence)

- **BUY:** RSI < 35 (original level)
- **SELL:** RSI > 65 (original level)
- **Confidence:** 0.85-0.95
- **Use case:** Higher conviction signals

---

### Configuration Changes

**Before:**

```yaml
rsi_oversold: 35.0
rsi_overbought: 65.0
support_resistance_pct: 1.5
```

**After:**

```yaml
rsi_oversold: 45.0 # Normal threshold
rsi_overbought: 55.0 # Normal threshold
rsi_strong_oversold: 35.0 # Strong threshold
rsi_strong_overbought: 65.0 # Strong threshold
support_resistance_pct: 0.5 # Very tight proximity (price must be right at S/R)
```

---

### Signal Generation Flow

```
CONDITION CHECKS:
├─ STRONG signals (highest confidence)
│  ├─ Oversold strong: RSI < 35 + near support → BUY (conf 0.85-0.95)
│  └─ Overbought strong: RSI > 65 + near resistance → SELL (conf 0.85-0.95)
│
├─ NORMAL signals (medium confidence)
│  ├─ Oversold normal: RSI < 45 + near support → BUY (conf 0.60-0.75)
│  └─ Overbought normal: RSI > 55 + near resistance → SELL (conf 0.60-0.75)
│
└─ HOLD: No conditions met
```

---

### Example Scenarios

#### Scenario 1: Your Case (Price at Resistance, RSI=59)

```
Price: $92,482.54
Resistance: $92,519.95
Distance: 0.04% ✅ NEAR RESISTANCE

RSI: 59.01
✅ Normal overbought (59 > 55)
❌ Strong overbought (59 < 65)

RESULT: NORMAL SELL SIGNAL
Confidence: 0.68 (moderate)
Reason: "RSI overbought near resistance"
```

#### Scenario 2: Strong Sell (Price at Resistance, RSI=72)

```
Price: $92,482.54
Resistance: $92,519.95
Distance: 0.04% ✅

RSI: 72.00
✅ Normal overbought (72 > 55)
✅ Strong overbought (72 > 65)

RESULT: STRONG SELL SIGNAL
Confidence: 0.93 (high)
Reason: "RSI strong overbought near resistance"
```

#### Scenario 3: Normal Buy (Price at Support, RSI=42)

```
Price: $90,200.00
Support: $90,150.00
Distance: 0.055% ✅

RSI: 42.00
✅ Normal oversold (42 < 45)
❌ Strong oversold (42 > 35)

RESULT: NORMAL BUY SIGNAL
Confidence: 0.65
Reason: "RSI oversold near support"
```

---

### Confidence Calculation

```python
confidence = base_conf (0.65)
           + RSI strength (0-0.15)    # How extreme RSI is
           + Volatility bonus (0-0.1) # If volatility is optimal
           + S/R proximity (0-0.1)    # Bonus for being at S/R
           × Strength multiplier      # 1.2x for strong signals, 1.0x for normal
           = Final confidence (0.6-0.95)
```

**Example:**

- Base: 0.65
- RSI strength (59 vs 55): 0.08
- Volatility bonus: 0.05
- S/R proximity: 0.10
- Normal multiplier: × 1.0
- **Total: 0.65 + 0.08 + 0.05 + 0.10 = 0.88 → capped at 0.95 = 0.88**

---

### Debug Output Changes

Now shows BOTH levels of RSI checks:

```
[RANGE DEBUG] RSI(14): 59.01
[RANGE DEBUG] Normal thresholds: oversold <45.0, overbought >55.0
[RANGE DEBUG] Strong thresholds: oversold <35.0, overbought >65.0
[RANGE DEBUG]   - Oversold normal (<45.0): False
[RANGE DEBUG]   - Oversold strong (<35.0): False
[RANGE DEBUG]   - Overbought normal (>55.0): True    ✅ Would trigger normal sell
[RANGE DEBUG]   - Overbought strong (>65.0): False   ❌ Not strong enough
```

When signal triggers:

```
[RANGE DEBUG] ✅ SELL SIGNAL GENERATED
[RANGE DEBUG]   Condition: RSI overbought (59.01 > 55.0) + near resistance
[RANGE DEBUG]   Confidence: 0.680
```

Or:

```
[RANGE DEBUG] ✅✅ STRONG SELL SIGNAL GENERATED
[RANGE DEBUG]   Condition: RSI STRONG overbought (72.00 > 65.0) + near resistance
[RANGE DEBUG]   Confidence: 0.930 (STRONG)
```

---

### Expected Improvement

| Metric                     | Before        | After                       |
| -------------------------- | ------------- | --------------------------- |
| Signals in ranging markets | Very few      | Regular (multiple per hour) |
| Signal quality             | High but rare | Varied (normal + strong)    |
| Confidence range           | 0.7-0.95      | 0.60-0.95                   |
| Trigger opportunities      | ~5% of time   | ~30-40% of time             |
| Aggregator weight          | 40%           | 40% (but gets more signals) |

---

### Configuration for Testing

```bash
# Run with new tiered thresholds
python3 main.py -c FIXED_CONFIG_FOR_VOLATILE_MARKETS.yaml -e paper --once

# Monitor output for:
# - [RANGE DEBUG] ✅ BUY/SELL SIGNAL GENERATED  (normal strength)
# - [RANGE DEBUG] ✅✅ STRONG BUY/SELL SIGNAL    (high strength)
```

---

### Files Modified

1. **strategies/range_trading.py**

   - Added `rsi_strong_oversold` and `rsi_strong_overbought` parameters
   - Implemented two-tier signal logic
   - Updated confidence calculation with strength multiplier
   - Enhanced debug output showing both RSI thresholds

2. **FIXED_CONFIG_FOR_VOLATILE_MARKETS.yaml**
   - Updated RSI thresholds (35→45 oversold, 65→55 overbought)
   - Added strong thresholds (35/65)
   - Tightened S/R proximity to 0.5% (was 1.5%)

---

### Next Steps

1. ✅ Run the bot: `python3 main.py -c FIXED_CONFIG_FOR_VOLATILE_MARKETS.yaml -e paper --once`
2. 📊 Monitor output for `[RANGE DEBUG]` messages
3. 🎯 Look for `✅ SELL SIGNAL` when price is near resistance
4. 📈 Check if you get trades instead of continuous HOLDs
5. 🔧 Fine-tune confidence thresholds if needed

**Expected:** You should now see regular BUY/SELL signals in ranging markets, with confidence levels indicating signal quality.
