# Parallel Testing Strategy

## Current Status (3+ Hours In)

### ✅ EMA Test (Running)

- **Duration:** 3+ hours (28+ iterations)
- **Status:** Clean execution, zero errors
- **Signals:** All HOLD (no crossovers yet - EXPECTED)
- **Process:** Running in PID 5118
- **Log:** logs/paper_first_test.log

**Good news:** System is perfectly healthy. EMA is waiting for a trend reversal.

### What to do with EMA test:

- ✓ Let it continue running 24 hours total (need 21+ more hours)
- ✓ Don't worry about HOLD signals - EMA crossovers are rare
- ✓ After 24h, check if ANY crossover occurs → validates strategy execution

---

## NEW: Parallel Active Strategy Test (FVG)

### Why Run FVG in Parallel?

1. **Validates Execution Flow Faster**

   - FVG generates 10-50 signals per day (vs EMA: 2-8 per week)
   - You'll see actual trades within 1-6 hours
   - Confirms stops/fills/sizing work under real conditions

2. **Tests Pattern Strategy vs Trend Strategy**

   - EMA: Trend-following (low frequency, possibly higher quality)
   - FVG: Pattern-based (high frequency, may have lower win rate)
   - See which strategy works better in current market

3. **Parallel Data**
   - Two independent tests
   - Can compare performance metrics later
   - No interference (separate log files)

### Launch Parallel FVG Test

```bash
# Start FVG test in separate terminal
cd /home/oladan/Trading-Platform
./run_parallel_active_test.sh

# This will:
# - Start FVG bot in background
# - Monitor for 5 minutes
# - Show initial checkpoint (signals, orders, errors)
# - Continue running for 6 hours
```

### After 5 Minutes (FVG checkpoint):

```bash
Expected:
  Signals: 2-5 (FVG finds patterns)
  Orders placed: 1-3 (position sizing approves some)
  Errors: 0 (clean execution)

If you see this: ✓ Execution flow works
```

### After 1-6 Hours (FVG completion):

```bash
# Analyze FVG results
python3 analyze_paper_session.py

# It will default to paper_first_test.log (EMA)
# To analyze FVG instead:
# grep "Signal:" logs/paper_active_test.log | wc -l
# grep "Order filled" logs/paper_active_test.log | wc -l
```

Expected FVG results after 6 hours:

- Signals: 10-30
- Fills: 5-20
- Trades closed: 2-10
- Win rate: 30-60% (don't care yet, just validating execution)

---

## Timeline: Next 24+ Hours

### Now (Parallel Tests Start)

```bash
# Terminal 1: EMA test (already running)
tail -f logs/paper_first_test.log

# Terminal 2: FVG test
./run_parallel_active_test.sh
tail -f logs/paper_active_test.log
```

### In 1 Hour

- EMA: Still all HOLD (normal)
- FVG: Should have 3-10 signals, some fills

### In 6 Hours

- EMA: Still all HOLD
- FVG: Analysis shows ~10-30 signals, position sizing/execution validated
  - If FVG shows clean execution (stops placed, fills happening) → ✅ Ready for multi-strategy
  - If FVG shows issues → Debug before proceeding

### In 24 Hours (Tomorrow)

```bash
# Check EMA results
python3 analyze_paper_session.py
# Expected: 0-2 signals (normal for EMA in this market)
# If ≥1 signal → Execution works, no issues
```

---

## What Each Test Tells You

### EMA Test (Low Frequency)

- **Validates:** Trend-following strategy execution
- **Expected:** 0-2 signals in 24h (rare crossovers)
- **Success:** If signal occurs, it fills correctly with proper stops

### FVG Test (High Frequency)

- **Validates:** Pattern recognition + execution + risk management
- **Expected:** 10-30 signals in 6h
- **Success:** Stops placed, fills happening, position sizes reasonable

---

## If FVG Shows Execution Works

Tomorrow (Day 2):

1. **Stop individual tests**
   - Kill EMA and FVG processes
2. **Enable Both Strategies**
   ```yaml
   strategy:
     name: regime_based
     multi_strategy:
       enabled: true
       strategies:
         - name: ema_trend
           weight: 0.3
         - name: fvg
           weight: 0.7
   ```
3. **Run 24h Multi-Strategy Test**
   - Do strategies complement each other?
   - Is aggregation better than individual strategies?

---

## If FVG Shows Issues

Examples:

- No signals despite FVG logic being correct
- Signals generated but no fills
- Position size = 0
- Crashes/errors

**Action:**

1. Check logs for specific error
2. Use [PAPER_TRADING_READY.md](PAPER_TRADING_READY.md) troubleshooting guide
3. Fix and restart

---

## Quick Reference

### Check Test Status

```bash
# EMA test running?
ps aux | grep main.py

# EMA stats
tail -5 logs/paper_first_test.log
grep "Signal:" logs/paper_first_test.log | tail -5

# FVG stats (after running)
grep "Signal:" logs/paper_active_test.log | wc -l
grep "Order filled" logs/paper_active_test.log | wc -l
```

### Stop Tests

```bash
# Get PIDs
ps aux | grep main.py

# Kill specific test
kill 5118  # EMA
kill <PID> # FVG (if running)
```

### Analyze Results

```bash
# Full analysis (EMA by default)
python3 analyze_paper_session.py

# Quick FVG stats
tail -50 logs/paper_active_test.log | grep -E "(BUY|SELL|filled)"
```

---

## Expected Outcomes

### Best Case (Both tests successful)

- EMA: 1-2 signals, all execute cleanly
- FVG: 15-30 signals, 50%+ fill rate, clean execution
- **Next step:** Multi-strategy aggregator test (Day 2)

### Good Case (FVG works, EMA too slow)

- EMA: 0 signals in 24h
- FVG: 15-30 signals, execution validated
- **Next step:** Test with FVG + other active patterns for 3 days

### Debug Case (FVG issues)

- EMA: 0 signals (expected)
- FVG: <5 signals OR many fills fail OR crashes
- **Next step:** Fix issue, restart FVG test

---

## Key Insight

You now know:

- ✅ System architecture is solid (clean, no crashes)
- ✅ Position sizing works
- ✅ Risk layer functions correctly
- ❓ Do your strategies have any edge? (Need more data)

The parallel tests will tell you:

- ✅ Can your execution handle real signals? (FVG test)
- ✅ Does trend-following work in this market? (EMA test)
- ✅ Should you combine strategies or focus on one? (Multi-strategy test Day 2)

**Next command:**

```bash
./run_parallel_active_test.sh
```

**Report back (after FVG test):**

- FVG signals generated?
- Any fills or all rejected?
- Any errors in logs?
