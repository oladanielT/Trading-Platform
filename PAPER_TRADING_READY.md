# Paper Trading Pre-Flight Checklist - COMPLETE ✓

## Status: Ready for Launch

### ✓ Completed Tasks

1. **Stop-Loss Propagation** - FIXED

   - MultiStrategyAggregator forwards stop_loss/entry_price from strongest strategy
   - Metadata flows through to position sizer
   - Test: `test_stop_loss_propagation.py` passes

2. **Position Sizer Validation** - VERIFIED

   - PositionSizer correctly reads stop_loss from signal metadata
   - Calculates valid position sizes (2000 units for 50-pip stop @ 1% risk)
   - Test: Direct PositionSizer test passes

3. **Stop-Loss Distance Audit** - PASSED

   - Created: `tests/test_stop_distances.py`
   - EMA strategy: ATR-based stops (2.0× ATR)
   - FVG strategy: 3.36% stop distance ($2.82) ✓
   - All strategies produce valid, directionally-correct stops
   - Result: **ALL CHECKS PASSED - Ready for paper trading**

4. **Paper Session Config** - CREATED

   - File: `configs/config_paper_first_test.yaml`
   - Single symbol: BTC/USDT
   - Single strategy: ema_trend (simplest baseline)
   - Risk: 1% per trade, max 1 position, 15% drawdown kill-switch
   - Conservative settings for first run

5. **Launch Script** - CREATED

   - File: `run_paper_test.sh`
   - Monitors first 30 minutes with real-time log tail
   - Provides checkpoint stats (signals, sizing, orders, errors)
   - Clean shutdown with PID tracking

6. **Analysis Script** - CREATED
   - File: `analyze_paper_session.py`
   - Parses logs for signals, fills, PnL, errors
   - Calculates win rate, profit factor, equity change
   - Validates success criteria (≥1 signal, ≥1 fill, equity changed, no crashes)

---

## Next Steps (Execute Now)

### Hour 1: Launch Paper Session

```bash
# 1. Start paper trading bot (6-hour test)
cd /home/oladan/Trading-Platform
./run_paper_test.sh

# Script will:
# - Start bot in background
# - Show real-time logs
# - Stop monitoring after 30 minutes (bot continues running)
# - Display checkpoint stats
```

### What to Watch (First 30 Minutes)

**Healthy execution flow:**

```
[DATA] Fetched OHLCV for BTC/USDT
[STRATEGY] Signal: BUY (confidence: 0.75)
[STRATEGY] Metadata: entry_price=..., stop_loss=... (X% distance)
[RISK] Position sizing: approved=True, size=...
[EXECUTION] Submitting order...
[EXECUTION] Order filled: entry=...
[POSITION] Position opened: BTC/USDT LONG
```

**Red flags to stop immediately:**

- `ERROR: stop_loss missing` (should be fixed)
- `Position size = 0` (stop too wide or risk too high)
- `Unhandled exception` (code bug)
- No signals after 15 minutes (strategy not triggering)

### Hour 2-6: Let It Run

**Monitor periodically:**

```bash
# Check if bot is still running
ps aux | grep main.py

# Watch logs
tail -f logs/paper_first_test.log

# Quick stats
grep -c "Signal:" logs/paper_first_test.log
grep -c "Order filled" logs/paper_first_test.log
```

**Stop early if:**

- Repeated errors (crashes, exceptions)
- Zero fills after 2 hours
- Drawdown exceeds 10% (kill-switch should activate)

### After 6 Hours: Analyze Results

```bash
# Run analysis script
python3 analyze_paper_session.py

# Expected output:
# - 5-30 signals (depends on market volatility)
# - 50-80% fill rate (risk layer approvals)
# - 1-10 trades closed
# - Final equity ≠ $1000 (some PnL)
# - Zero errors
```

**Success criteria (minimal):**

- ✓ At least 1 signal generated
- ✓ At least 1 position sizing approved
- ✓ At least 1 order filled
- ✓ Equity changed
- ✓ No crashes

**Don't worry about:**

- Win rate (not important yet)
- Profitability (not important yet)
- Drawdown (unless >10%)

**Focus on:**

- Execution flow works end-to-end
- No unhandled exceptions
- Position sizes are reasonable (not 0, not huge)
- Stops are being placed/managed

---

## If Paper Session Succeeds

### Day 2 (Tomorrow)

1. **Enable Multi-Strategy** (if single-strategy works)

   ```yaml
   # Edit config_paper_first_test.yaml
   strategy:
     name: regime_based
     multi_strategy:
       enabled: true
       strategies:
         - name: ema_trend
           weight: 0.3
         - name: fvg
           weight: 0.3
         - name: continuation
           weight: 0.4
   ```

2. **Run 24-hour session**

   - Same config, longer duration
   - Compare: does aggregation improve results?

3. **Calculate correlation matrix**
   - Which strategies agree/conflict?
   - Are signals redundant or diverse?

### Week 2 (Days 3-7)

1. **Build TradeJournal class** (automated metrics)

   - Log every entry/exit with timestamps
   - Calculate: Sharpe, max drawdown, profit factor
   - Export to JSON for analysis

2. **Run 3-5 day paper session**

   - All strategies enabled
   - Collect 30-100 trades minimum
   - Generate statistical confidence

3. **Strategy validation**
   - Disable any strategy with win rate <40%
   - Add confluence filters to weak strategies
   - Tighten entry criteria if too many signals

### Week 3 (Testnet Transition)

1. **If paper metrics acceptable:**

   - Win rate >45% OR profit factor >1.2
   - Max drawdown <20%
   - Sharpe >0.5 (paper, zero-cost)

2. **Switch to testnet:**

   - Minimum position sizes
   - Real execution costs (spreads, slippage)
   - Run 7-14 days

3. **Compare testnet vs. paper:**

   - Sharpe ratio degradation
   - Win rate reduction
   - Slippage impact

   **If testnet Sharpe <50% of paper Sharpe:**

   - Execution costs too high
   - Reduce trading frequency
   - Focus on higher-confidence signals

---

## Troubleshooting Guide

### No Signals Generated

**Cause:** Strategy not triggering
**Fix:**

- Check if EMA crossover actually occurs in 6 hours
- Temporarily lower `signal_confidence` to 0.5
- Use volatile market hours (London/NY overlap)

### Signals But No Fills

**Cause:** Risk layer rejecting all positions
**Fix:**

- Check logs for rejection reason
- Verify stop_loss metadata is present
- Increase `max_leverage` if position_value > equity
- Disable `portfolio_risk` temporarily

### Position Size = 0

**Cause:** Stop too wide for 1% risk
**Fix:**

- Reduce `atr_multiplier` from 2.0 to 1.5
- Increase `risk_per_trade` from 0.01 to 0.02
- Check if stop_loss distance is >10% (unreasonable)

### Crashes/Exceptions

**Cause:** Code bug
**Fix:**

- Read full stack trace in logs
- Check for missing imports, attribute errors
- Verify data feed is working (fetch OHLCV)

---

## Files Created

### Test Scripts

- `tests/test_stop_distances.py` - Validates stop-loss logic
- `test_stop_loss_propagation.py` - End-to-end metadata flow test

### Configuration

- `configs/config_paper_first_test.yaml` - Paper session config

### Execution

- `run_paper_test.sh` - Launch script with monitoring
- `analyze_paper_session.py` - Post-session analysis

### Logs

- `logs/paper_first_test.log` - Session logs (created on run)

---

## Quick Commands Reference

```bash
# Pre-flight check
python3 tests/test_stop_distances.py

# Launch paper session
./run_paper_test.sh

# Monitor (if monitoring stopped)
tail -f logs/paper_first_test.log

# Check bot status
ps aux | grep main.py

# Stop bot
kill <PID>  # PID shown by run_paper_test.sh

# Analyze results
python3 analyze_paper_session.py

# Quick stats
grep -c "Signal:" logs/paper_first_test.log
grep -c "Order filled" logs/paper_first_test.log
grep -c "ERROR" logs/paper_first_test.log
```

---

## Success Metrics Timeline

### Today (6-hour test)

- ✓ Execution flow works
- ✓ No crashes
- ✓ Trades happen

### Day 2-3 (24-hour test)

- Signals: 10-50
- Fills: 5-30
- Win rate: 30-70% (any is fine)

### Week 2 (5-day test)

- Trades: 30-100
- Sharpe: >0.5 (paper)
- Max DD: <20%
- Profit factor: >1.2

### Week 3 (Testnet)

- Sharpe degradation: <50%
- Execution costs: <2-3% per trade
- Win rate: >45%

---

## Current State

**All systems ready for launch:**

- ✅ Stop-loss propagation fixed
- ✅ Position sizer validated
- ✅ Stop distances audited
- ✅ Config created
- ✅ Scripts deployed
- ✅ Pre-flight checks passed

**Next command:**

```bash
./run_paper_test.sh
```

**Good luck! Report back:**

- Signals generated (after 30 min)
- Any errors (if they occur)
- Final equity (after 6 hours)
