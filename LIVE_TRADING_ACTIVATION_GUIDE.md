# Live Trading Activation Guide

## Overview

This guide walks you through activating all strategies for live trading with small positions, conservative risk limits, and comprehensive monitoring. Follow this exactly as written to ensure safe activation.

## Pre-Activation Requirements

### 1. Backtesting Validation

- [ ] Run full backtesting suite with `pytest tests/ -v`
- [ ] Verify all tests pass (152/152)
- [ ] Review backtest metrics (see IMPLEMENTATION_SUMMARY.md)
- [ ] Confirm win rate >= 50% on historical data

### 2. Paper Mode Validation (1-2 weeks minimum)

- [ ] Set `environment: paper` in config
- [ ] Run trading bot for 1-2 weeks
- [ ] Collect at least 20-50 trades for statistical significance
- [ ] Monitor metrics: `python scripts/monitor_live_trading.py --section all`
- [ ] Check that observed win rate matches backtest expectations
- [ ] Verify readiness state changes match expected regime patterns
- [ ] Confirm no unexpected alerts or errors in logs

### 3. Testnet Validation (1 week)

- [ ] Set `environment: testnet` in config
- [ ] Set `exchange.testnet: true`
- [ ] Use testnet API credentials
- [ ] Run for 1 week with real exchange latency
- [ ] Verify order fill rates > 95%
- [ ] Check actual slippage vs backtested slippage
- [ ] Confirm fees don't exceed tolerance

## Configuration

### Step 1: Use Live Trading Config Template

```bash
cp config-live-trading.yaml core/config.yaml
```

Or manually edit `core/config.yaml`:

```yaml
# CRITICAL SETTINGS FOR LIVE
environment: paper # Start here, upgrade to testnet, then live
exchange:
  id: binance
  testnet: true # true for paper/testnet, false for live

trading:
  symbols:
    - BTC/USDT
    - ETH/USDT
    - BNB/USDT
    # Add up to 5 symbols for diversification

risk:
  max_position_size: 0.1 # 10% max - DO NOT EXCEED
  risk_per_trade: 0.02 # 2% - DO NOT EXCEED
  max_daily_drawdown: 0.05 # 5% daily limit
  max_total_drawdown: 0.20 # 20% total limit
```

### Step 2: Enable Multi-Strategy Aggregation

If not already enabled in your config, add:

```yaml
strategy:
  multi_strategy:
    enabled: true

    # All three timeframes active
    strategies:
      - name: ema_trend
        enabled: true
        params:
          fast_period: 20
          slow_period: 50
        weight: 1.0

      - name: ema_trend_medium
        enabled: true
        params:
          fast_period: 50
          slow_period: 200
        weight: 0.8

      - name: ema_trend_long
        enabled: true
        params:
          fast_period: 100
          slow_period: 300
        weight: 0.6

    # Enable adaptive weighting
    adaptive_weighting:
      enabled: true
      learning_rate: 0.1
      min_samples: 20
```

### Step 3: Enable Runtime Alerts

```yaml
monitoring:
  runtime_alerts:
    enabled: true
    readiness_threshold: not_ready
    confidence_threshold: 0.85
    deviation_threshold: 0.3
```

## Activation Stages

### STAGE 1: Paper Mode Validation

**Duration**: 1-2 weeks  
**Configuration**: `environment: paper`  
**Capital**: Virtual (any amount in config)  
**Goal**: Validate strategy performance matches backtests

```bash
# Start trading bot
python main.py

# In another terminal, monitor every hour
watch -n 3600 'python scripts/monitor_live_trading.py --section all'

# Or run manually for quick checks
python scripts/monitor_live_trading.py --section trades
```

**Pass Criteria**:

- ✓ Win rate >= 50% (ideally 55%+)
- ✓ Max single-day drawdown < 3%
- ✓ Consecutive losses < 5 within 100 trades
- ✓ Readiness state transitions match expected patterns
- ✓ Adaptive weight changes are gradual (no sudden swings)
- ✓ Trade distribution across symbols is balanced

**Example Passing Metrics**:

```
Total trades: 50
Win rate: 56%
Max drawdown: 1.8%
Net PnL: +$2,847
Profit factor: 1.45
```

**If Performance is Below Expectations**:

1. Check readiness thresholds:

   ```yaml
   allowed_regimes:
     - trending # Try adding 'quiet' if too restrictive
     - quiet # Try removing if too loose
   ```

2. Review confidence thresholds:

   ```yaml
   regime_confidence_threshold: 0.5 # Lower if too strict
   reduce_confidence_in_marginal_regimes: true
   ```

3. Adjust position sizing:
   ```yaml
   sizing_method: fixed_risk # Ensure using fixed risk
   use_confidence_scaling: true # Scale by confidence
   ```

### STAGE 2: Testnet Validation

**Duration**: 1 week minimum  
**Configuration**: `environment: testnet`, `exchange.testnet: true`  
**Capital**: Small amount ($1000-5000)  
**Goal**: Validate execution, fills, and slippage on real exchange

```bash
# Update config
environment: testnet
exchange:
  testnet: true
  api_key: "YOUR_TESTNET_API_KEY"
  secret_key: "YOUR_TESTNET_SECRET_KEY"

# Restart trading bot
python main.py

# Monitor during trading hours
python scripts/monitor_live_trading.py --section all
```

**Pass Criteria**:

- ✓ Order fill rate > 95%
- ✓ Actual slippage within ±0.5% of limit price
- ✓ Trade counts per day stable
- ✓ Win rate similar to paper (±5%)
- ✓ No connection errors or timeouts
- ✓ Fees reasonable (check logs)

**If Issues Found**:

1. Slippage too high?

   - Increase `order_timeout` from 30 to 60 seconds
   - Use limit orders instead of market orders
   - Consider post-only orders during low liquidity

2. Fill rate issues?
   - Reduce order size (increase number of smaller orders)
   - Adjust position sizing to be more conservative
   - Review market conditions during trading

### STAGE 3: Live Micro Positions

**Duration**: 2-4 weeks  
**Configuration**: `environment: live`, `exchange.testnet: false`  
**Capital**: Very small ($1000-5000)  
**Goal**: Validate live execution with real money before scaling

```bash
# Update config - VERIFY CREDENTIALS CAREFULLY
environment: live
exchange:
  testnet: false
  api_key: "YOUR_LIVE_API_KEY"
  secret_key: "YOUR_LIVE_SECRET_KEY"
  # Use production endpoints (auto-configured by ExchangeConnector)

# ALWAYS start a backup/monitoring terminal
tmux new-session -d -s monitoring
tmux send-keys -t monitoring 'watch -n 3600 "python scripts/monitor_live_trading.py --section all"' Enter

# Start trading
python main.py
```

**Monitoring Schedule**:

- Check every 4 hours during trading hours
- Check balance and open positions before market open
- Review alerts immediately
- Run full analysis daily

**Daily Checklist**:

```bash
# Check overall health
python scripts/monitor_live_trading.py --section trades

# Check drawdown status
python scripts/monitor_live_trading.py --section drawdown

# Check readiness correlation
python scripts/monitor_live_trading.py --section readiness

# Check for errors in logs
tail -50 logs/live_trading.log | grep -i error
tail -50 logs/live_trading.log | grep -i alert
```

**Pass Criteria for Scaling**:

- ✓ Win rate within 5% of backtested expectations
- ✓ No unexpected errors or crashes
- ✓ Slippage matches testnet observations
- ✓ Max drawdown < 3% on worst day
- ✓ No consecutive losses > 5
- ✓ Readiness state timing matches expectations
- ✓ All positions closed at appropriate times
- ✓ No emergency margin calls or liquidations
- ✓ Alerts are accurate and useful

**Capital Increase Schedule** (if passing criteria):

```
Week 1-2: $1,000 (micro validation)
Week 3-4: $2,500 (50% increase)
Week 5-6: $5,000 (100% increase)
Week 7-8: $10,000 (100% increase)
Week 9+:  Full capital (after 4+ weeks success)
```

**Critical Stop Rules** (Halt immediately):

- Win rate drops below 40%
- 3+ consecutive losses in a day
- Unrealistic slippage (> 1%)
- Margin call or liquidation risk
- Readiness state doesn't match behavior
- Any error in core logic execution

### STAGE 4: Live Scaling

**Duration**: Ongoing  
**Configuration**: Continue with `environment: live`  
**Capital**: Gradually increase per schedule  
**Goal**: Reach full capital allocation while maintaining performance

```bash
# Increase capital only after 2+ weeks at current level
# with consistent performance matching expectations

# Update config with new initial_capital
trading:
  initial_capital: 5000.0  # Increase by 50% increments

# Restart trading
python main.py
```

**Scaling Rules**:

1. Only increase capital after 2+ weeks at current level
2. Only increase if win rate >= 50%
3. Only increase if max DD < 3% on best weeks
4. Increase by 50% per stage (not more)
5. Can return to previous level if performance drops
6. Never exceed original full capital in first month

**Monitoring During Scaling**:

```bash
# Set up automated alerts
# Create a cron job for hourly monitoring
crontab -e

# Add this line:
0 * * * * python /path/to/scripts/monitor_live_trading.py --section all >> /tmp/trading_monitor.log 2>&1

# Check monitoring log
tail -20 /tmp/trading_monitor.log
```

### STAGE 5: Production Optimization (After 4+ weeks)

Once you've successfully run at full capital for 4+ weeks:

1. **Adjust thresholds based on observations**:

   ```yaml
   # If too many false alerts, increase thresholds
   confidence_threshold: 0.90  # from 0.85
   deviation_threshold: 0.4    # from 0.3

   # If missing important signals, decrease
   confidence_threshold: 0.80
   deviation_threshold: 0.25
   ```

2. **Consider increasing position sizes**:

   ```yaml
   risk:
     max_position_size: 0.15 # from 0.10
     risk_per_trade: 0.025 # from 0.02
   ```

3. **Fine-tune regime parameters**:
   ```yaml
   strategy:
     regime_confidence_threshold: 0.45 # from 0.5
     allow_volatile_trend_override: true
     volatile_trend_strength_min: 0.9 # from 1.0
   ```

## Monitoring and Logging

### Real-Time Monitoring

```bash
# Watch trading logs in real-time
tail -f logs/live_trading.log

# Filter for specific events
tail -f logs/live_trading.log | grep "Trade closed"
tail -f logs/live_trading.log | grep "\[Alert\]"
tail -f logs/live_trading.log | grep "Readiness"
```

### Key Log Markers to Watch

```log
# Trade execution
2025-12-31 10:00:00 - Trade closed: BTC/USDT LONG +$125.50

# Readiness changes
2025-12-31 10:05:00 - Readiness changed: BTC/USDT ready -> forming

# Alerts
2025-12-31 10:10:00 - [Alert] Confidence anomaly: BTC/USDT confidence 0.95 with action reversal

# Adaptive weights
2025-12-31 10:15:00 - [AdaptiveWeights] Strategy update: ema_trend 1.0 -> 1.1x (50 samples, 56% accuracy)

# Errors (watch for these!)
2025-12-31 10:20:00 - ERROR: Order timeout on BTC/USDT, retrying...
```

### Periodic Analysis

```bash
# Daily analysis
python scripts/monitor_live_trading.py --section all > daily_report_$(date +%Y%m%d).txt

# Weekly analysis
python scripts/monitor_trade_success.py --json > weekly_trades.json
python scripts/monitor_drawdown.py --json > weekly_drawdown.json
python scripts/monitor_readiness_correlation.py --json > weekly_readiness.json
```

### Alert Thresholds Configuration

```yaml
monitoring:
  runtime_alerts:
    # Readiness drop alert
    readiness_threshold: not_ready # Alert when readiness <= not_ready

    # High confidence + reversal alert
    confidence_threshold: 0.85 # Alert when confidence > 85%

    # Strategy deviation alert
    deviation_threshold: 0.3 # Alert when deviation > 30%
```

## Emergency Procedures

### If Losing Money Faster Than Expected

```bash
# 1. Check current positions
# View in exchange account or logs

# 2. Check win rate
python scripts/monitor_live_trading.py --section trades

# 3. If win rate < 40%:
#    a. Stop trading
#    b. Reduce capital to previous level
#    c. Return to paper mode for analysis

# 4. Check readiness state correlation
python scripts/monitor_live_trading.py --section readiness

# 5. If readiness doesn't predict outcomes:
#    a. Adjust allowed_regimes
#    b. Adjust regime_confidence_threshold
#    c. Re-run backtests
```

### If Drawdown Exceeds Limits

```bash
# 1. Check current drawdown
python scripts/monitor_live_trading.py --section drawdown

# 2. If > max_daily_drawdown (5%):
#    a. Stop opening new positions
#    b. Let existing positions run (don't panic sell)
#    c. Wait for recovery

# 3. If > max_total_drawdown (20%):
#    a. Halt trading
#    b. Analyze cause
#    c. Reduce capital
#    d. Return to paper mode
```

### If Strategy Performance Degrades

```bash
# Sudden win rate drop might indicate:

# 1. Market regime change
python scripts/monitor_readiness_correlation.py

# 2. Excessive slippage
tail -50 logs/live_trading.log | grep -i slippage

# 3. Strategy parameters no longer optimal
# → Verify backtest assumptions still hold
# → Consider re-optimizing on recent data

# 4. Adaptive weights diverging from static
# → Check weight changes in logs
# → Reduce learning_rate if too aggressive
```

## Success Indicators

### Green Flags (Keep Going)

✓ Win rate 50%+ and stable  
✓ Max DD < 3% on best weeks  
✓ Positive net PnL every week  
✓ Trade distribution balanced  
✓ Readiness state predictions accurate  
✓ Adaptive weights stable  
✓ Alerts are meaningful  
✓ No connection or execution errors

### Yellow Flags (Review)

⚠️ Win rate 40-50% (still valid but watch)  
⚠️ Max DD 3-5% (approaching limits)  
⚠️ Irregular win rate (high variance)  
⚠️ Too many false alerts (tune thresholds)  
⚠️ Slippage higher than backtest (1%+)

### Red Flags (Halt and Analyze)

✗ Win rate < 40% (likely issue)  
✗ Max DD > 5% (exceeds daily limit)  
✗ 5+ consecutive losses (circuit break)  
✗ Regular errors in logs  
✗ Readiness state doesn't predict outcomes  
✗ Margin call or liquidation risk  
✗ Extreme slippage (> 1%)

## Useful Commands

```bash
# Start trading bot
python main.py

# Monitor in background
tmux new-session -d -s trading
tmux send-keys -t trading 'python main.py' Enter
tmux send-keys -t trading 'python scripts/monitor_live_trading.py --section all' Enter

# View logs
tail -f logs/live_trading.log

# Get detailed metrics
python scripts/monitor_live_trading.py --section all
python scripts/monitor_trade_success.py
python scripts/monitor_drawdown.py
python scripts/monitor_readiness_correlation.py

# JSON output for analysis
python scripts/monitor_live_trading.py --json > metrics.json

# Check for specific events
grep "Trade closed" logs/live_trading.log | wc -l  # Count trades
grep "ERROR" logs/live_trading.log                  # Check errors
grep "\[Alert\]" logs/live_trading.log              # Check alerts
```

## Timeline Summary

```
Week 1-2:   Paper Mode         (20-50 trades)
Week 3:     Testnet Validation (continuous)
Week 4-7:   Live Micro ($1-5k) (gradual scaling)
Week 8-12:  Live Scaling       ($5-20k)
Week 13+:   Production         (full capital)
```

## Key Preservation Points

All the following existing logic is **fully preserved**:

- ✓ Readiness gating (no trades below threshold)
- ✓ Position sizing and risk limits
- ✓ Drawdown guards and circuit breakers
- ✓ Order execution and retry logic
- ✓ Strategy signal generation
- ✓ Regime detection and filtering

Nothing changes about how trades are executed—only the monitoring and analysis layers are enhanced.

---

**Ready to activate?** Start with Paper Mode and follow the stages exactly as described. Good luck!
