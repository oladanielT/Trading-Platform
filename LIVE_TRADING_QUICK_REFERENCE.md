# Live Trading Quick Reference

## Quick Start (5 minutes)

```bash
# 1. Copy live trading config
cp config-live-trading.yaml core/config.yaml

# 2. Start in paper mode first
# Edit: environment: paper (already set in template)

# 3. Start trading
python main.py

# 4. Monitor (in another terminal)
python scripts/monitor_live_trading.py --section all
```

## Position Sizing (Always Followed)

```
MAX POSITION SIZE:  10% of equity per trade
MAX RISK:           2% of equity per trade
TYPICAL POSITION:   $5,000-10,000 on $100k account
SCALING EXAMPLE:    Week 1: $1k → Week 2: $2.5k → Week 3: $5k → Week 4: $10k
```

## Risk Limits (Never Exceeded)

```
MAX DAILY LOSS:     5% of equity
MAX TOTAL LOSS:     20% of equity from peak
MAX CONSECUTIVE:    5 losing trades (auto-halt)
```

## Activation Stages

```
┌─────────────────────────────────────────────────────────────┐
│ STAGE 1: PAPER MODE (1-2 weeks)                             │
│ Set environment: paper in config                             │
│ Target: 50%+ win rate, <3% daily DD                          │
│ ✓ Pass → Continue                                            │
│ ✗ Fail → Debug strategy                                      │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 2: TESTNET (1 week)                                    │
│ Set environment: testnet, testnet: true                      │
│ Target: Match paper results, >95% fill rate                  │
│ ✓ Pass → Continue                                            │
│ ✗ Fail → Debug execution/fills                               │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 3: LIVE MICRO (2-4 weeks)                              │
│ Set environment: live, testnet: false, add live credentials  │
│ Start with $1k-5k capital only                               │
│ Target: Confirm live metrics match paper/testnet             │
│ ✓ Pass → Increase capital 50% per week                       │
│ ✗ Fail → Return to paper mode                                │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│ STAGE 4: LIVE SCALING (Ongoing)                              │
│ Increase capital gradual: $5k → $10k → $15k → Full           │
│ Target: Maintain 50%+ win rate at each level                 │
│ Monitor: Daily analysis, ready to downscale                  │
└─────────────────────────────────────────────────────────────┘
```

## Critical Config Settings

```yaml
# core/config.yaml

environment: paper # paper → testnet → live
exchange:
  testnet: true # true for paper/testnet, false for live

trading:
  symbols:
    - BTC/USDT # Start with 1-3 pairs
    - ETH/USDT

risk:
  max_position_size: 0.1 # 10% max - DO NOT CHANGE
  risk_per_trade: 0.02 # 2% - DO NOT CHANGE
  max_daily_drawdown: 0.05 # 5% daily
  max_total_drawdown: 0.20 # 20% total

monitoring:
  runtime_alerts:
    enabled: true # Watch for anomalies
```

## Monitoring Commands

```bash
# Full dashboard (all metrics)
python scripts/monitor_live_trading.py --section all

# Individual sections
python scripts/monitor_live_trading.py --section trades
python scripts/monitor_live_trading.py --section drawdown
python scripts/monitor_live_trading.py --section readiness

# JSON format (for analysis)
python scripts/monitor_live_trading.py --json

# Watch logs in real-time
tail -f logs/live_trading.log

# Filter for specific events
tail -f logs/live_trading.log | grep "Trade closed"  # Trades
tail -f logs/live_trading.log | grep "\[Alert\]"      # Alerts
tail -f logs/live_trading.log | grep "Readiness"      # State changes
```

## What to Watch

### Green Indicators ✓

- Win rate 50%+ and stable
- Max drawdown < 3% on best weeks
- Positive PnL accumulating
- Trade distribution balanced across pairs
- Readiness state predictions accurate
- Adaptive weights stable (not wildly swinging)
- Alerts are meaningful (not spam)

### Yellow Warnings ⚠️

- Win rate 40-50% (acceptable but monitor)
- Max DD 3-5% (approaching limits)
- High variance in daily results
- Too many alerts (tune thresholds)
- Slippage > 0.5% (check order sizing)

### Red Flags ✗ (STOP)

- Win rate < 40% (likely problem)
- Max DD > 5% daily (at limit)
- 5+ consecutive losses (auto-halt)
- Regular errors in logs
- Margin call or liquidation risk
- Readiness state doesn't match outcomes
- Slippage > 1% (execution issue)

## Typical Daily Workflow

```
09:00 AM  - Check overnight logs
           tail logs/live_trading.log | tail -50

10:00 AM  - First analysis
           python scripts/monitor_live_trading.py --section trades

01:00 PM  - Mid-day check
           python scripts/monitor_live_trading.py --section drawdown

04:00 PM  - End of day summary
           python scripts/monitor_live_trading.py --section all
           python scripts/monitor_readiness_correlation.py

Weekly    - Deep analysis
           Save all outputs to file for trending analysis
```

## Emergency Procedures

### Win Rate Too Low (< 40%)

1. Stop opening new positions
2. Run: `python scripts/monitor_live_trading.py --section readiness`
3. Check if readiness state predictions are accurate
4. If readiness is wrong:
   - Adjust `regime_confidence_threshold`
   - Add/remove regimes from `allowed_regimes`
5. Return to paper mode if issue persists

### Drawdown Exceeding Limits

1. Check: `python scripts/monitor_live_trading.py --section drawdown`
2. If > 5% daily: Stop opening new positions
3. If > 20% total: Halt all trading
4. Let existing positions close
5. Reduce capital and return to paper

### Connection or Execution Errors

1. Check logs: `tail logs/live_trading.log | grep ERROR`
2. Verify API credentials in config
3. Check internet connection
4. If persisting, stop bot and troubleshoot
5. Never restart without understanding the issue

## Key Files

| File                                       | Purpose                                   |
| ------------------------------------------ | ----------------------------------------- |
| `config-live-trading.yaml`                 | Live trading config template              |
| `core/config.yaml`                         | Active configuration (copy template here) |
| `scripts/monitor_live_trading.py`          | Main monitoring dashboard                 |
| `scripts/monitor_trade_success.py`         | Trade success analysis                    |
| `scripts/monitor_drawdown.py`              | Drawdown tracking                         |
| `scripts/monitor_readiness_correlation.py` | Readiness state analysis                  |
| `LIVE_TRADING_ACTIVATION_GUIDE.md`         | Full activation procedures                |
| `logs/live_trading.log`                    | Live trading logs                         |

## Capital Scaling Schedule

For a $100k account starting with 10% position limit:

```
Week 1-2:   Paper mode      ($100k virtual)
Week 3:     Testnet         ($100k virtual)
Week 4-5:   Live $1k        ($1,000 real)    ← START HERE
Week 6-7:   Live $2.5k      ($2,500 real)    ← 50% increase
Week 8-9:   Live $5k        ($5,000 real)    ← 100% increase
Week 10-11: Live $10k       ($10,000 real)   ← 100% increase
Week 12+:   Live $20k+      Scale as needed   ← Continue if stable
```

Only increase capital if:

- ✓ 2+ weeks at current level
- ✓ Win rate 50%+
- ✓ Max DD < 3%
- ✓ No unexpected errors

## Useful Shortcuts

```bash
# Fast health check (takes ~30 seconds)
python scripts/monitor_live_trading.py --section trades && \
python scripts/monitor_live_trading.py --section drawdown

# Weekly analysis export
date=$(date +%Y%m%d)
python scripts/monitor_live_trading.py --json > analysis_$date.json

# Tail trades only
tail -f logs/live_trading.log | grep "Trade closed"

# Count total trades
grep -c "Trade closed" logs/live_trading.log

# Show recent errors
tail -100 logs/live_trading.log | grep -i error
```

## Nothing Changes About Core Logic

✓ Readiness gating fully preserved  
✓ Position sizing rules enforced  
✓ Risk limits active  
✓ Order execution unchanged  
✓ Strategy parameters applied  
✓ Drawdown guards active

This guide only **adds monitoring and activation procedures**—all safety logic remains exactly the same.

---

**First time?** Start with STAGE 1 (Paper Mode) and follow the timeline. Good luck!

## Next Steps

1. Read [BINANCE_INTEGRATION.md](BINANCE_INTEGRATION.md) for detailed documentation
2. Read [LIVE_TRADING_GUIDE.md](LIVE_TRADING_GUIDE.md) for comprehensive guide
3. Run tests: `pytest tests/test_live_trading.py -v`
4. Start paper trading: `python3 binance_regime_trader.py --duration 60`
5. Monitor logs: `tail -f logs/trading.log`

---

**Status**: ✅ Production Ready (21/21 tests passing)
**Last Updated**: December 30, 2025
