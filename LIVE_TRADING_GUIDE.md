# Live Trading Guide

## Overview

This guide covers running the Binance Regime-Based Trader in **live trading mode** with real-time price fetching, automated signal generation, and paper-based order simulation.

## Quick Start

### 1. Minimal Setup (5 minutes)

```bash
# Clone or navigate to project
cd /home/oladan/Trading-Platform

# Install dependencies (if needed)
pip install -r requirements.txt

# Run live trading for 60 seconds
python3 binance_regime_trader.py --duration 60
```

### 2. First Live Session

Expected first-run output:

```
✓ Loaded configuration from core/config.yaml
✓ Binance client initialized successfully
✓ Strategy ready for BTC/USDT
✓ Strategy ready for ETH/USDT
✓ Strategy ready for BNB/USDT
✓ Strategy ready for XRP/USDT
✓ Strategy ready for ADA/USDT

=== LIVE TRADING SESSION STARTED ===
Initial Capital: $100,000.00
Duration: 60 seconds
Symbols: 5 (BTC, ETH, BNB, XRP, ADA)
Update Interval: 5 seconds

[15:30:45] Fetching prices for 5 symbols...
[15:30:46] ✓ Prices updated
  BTC/USDT: $43,250.00
  ETH/USDT: $2,345.67
  BNB/USDT: $310.50
  XRP/USDT: $0.65
  ADA/USDT: $1.15

[15:30:46] Generating signals...
  BTC/USDT: BUY (regime=trending, confidence=0.85)
  ETH/USDT: HOLD (regime=trending, confidence=0.75)
  BNB/USDT: SELL (regime=ranging, confidence=0.60) - FILTERED OUT
  XRP/USDT: BUY (regime=trending, confidence=0.72)
  ADA/USDT: HOLD (regime=ranging, confidence=0.55) - FILTERED OUT

[15:30:46] Executing 2 approved trades...
  ✓ BTC/USDT BUY 0.010 @ $43,250.00
  ✓ XRP/USDT BUY 6.77 @ $0.65

[15:30:46] Portfolio Status:
  Equity: $100,001.45
  Open Positions: 2
  Total Trades: 2
  Win Rate: 100.0% (2/2)

[15:31:51] Session Complete
  Duration: 65 seconds
  Total Return: +0.00%
  Total Trades: 2
  Symbols Traded: 2/5
  Report: logs/backtest_summary.txt
```

---

## Configuration

### Core Settings (core/config.yaml)

```yaml
exchange:
  testnet: true # Use testnet (safe for testing)
  paper_mode: true # Paper trading (no real execution)
  api_key: YOUR_API_KEY
  secret_key: YOUR_SECRET_KEY

trading:
  initial_capital: 100000.0 # Starting capital in USDT
  price_update_interval: 5.0 # Fetch prices every 5 seconds
  symbols:
    - BTC/USDT # Bitcoin
    - ETH/USDT # Ethereum
    - BNB/USDT # Binance Coin
    - XRP/USDT # Ripple
    - ADA/USDT # Cardano
  allocation_per_symbol: 0.2 # 20% of capital per pair

strategy:
  allowed_regimes:
    - trending # Trade in uptrends
    # - ranging                    # Disable range trading
    # - mean_reversion             # Disable mean-reversion trades
  regime_confidence_threshold: 0.5

backtesting:
  lookback_days: 30
  timeframe: "1h"
```

### Optimization Tips

**For low-latency trading:**

```yaml
trading:
  price_update_interval: 1.0 # Fetch every 1 second (more responsive)
```

**For high-frequency portfolio:**

```yaml
trading:
  price_update_interval: 0.5 # Fetch every 500ms
```

**For resource-constrained systems:**

```yaml
trading:
  price_update_interval: 10.0 # Fetch every 10 seconds (lower CPU)
```

---

## Running Live Sessions

### Command-Line Modes

#### 1. Indefinite Live Trading

```bash
python3 binance_regime_trader.py
# Runs until you press Ctrl+C
```

#### 2. Time-Limited Trading (Recommended for Testing)

```bash
# Trade for 5 minutes (300 seconds)
python3 binance_regime_trader.py --duration 300

# Trade for 1 hour
python3 binance_regime_trader.py --duration 3600

# Trade for 30 seconds (quick test)
python3 binance_regime_trader.py --duration 30
```

#### 3. Backtesting Mode (Validation)

```bash
# Compare live mode behavior with historical data
python3 binance_regime_trader.py --backtest
```

---

## Real-Time Monitoring

### Tracking Portfolio Metrics

During a live session, monitor these key metrics:

| Metric             | Meaning                 | Good Range          |
| ------------------ | ----------------------- | ------------------- |
| **Equity**         | Current portfolio value | > Initial capital   |
| **Total Return**   | Percentage gain/loss    | > 0%                |
| **Open Positions** | Number of active trades | 0-5 typical         |
| **Win Rate**       | % of profitable trades  | > 50%               |
| **Max Drawdown**   | Worst decline from peak | < -20% warning      |
| **Total Trades**   | Cumulative executions   | Increases over time |

### Log File Monitoring

Monitor live updates in `logs/trading.log`:

```bash
# Watch log in real-time
tail -f logs/trading.log

# Count trades executed
grep "BUY\|SELL" logs/trading.log | wc -l

# Find large moves
grep -i "error\|fail" logs/trading.log
```

### Example Live Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Watch trading session in real-time

echo "=== LIVE TRADING MONITOR ==="
echo "Starting session for 5 minutes..."

# Start trading in background
timeout 300 python3 binance_regime_trader.py > /tmp/trading.log 2>&1 &
PID=$!

# Monitor in real-time
while kill -0 $PID 2>/dev/null; do
    clear
    echo "=== LIVE TRADING SESSION ==="
    echo "Time: $(date)"
    echo ""

    # Show last 10 log lines
    echo "Recent Activity:"
    tail -10 /tmp/trading.log

    # Count trades
    TRADES=$(grep -c "BUY\|SELL" /tmp/trading.log || echo 0)
    echo ""
    echo "Total Trades: $TRADES"

    sleep 5
done

echo "Session ended."
```

---

## Regime-Based Safety Filtering

### Understanding Regimes

The strategy classifies market conditions into regimes:

| Regime       | Market Condition         | Signal Type                    | Risk Level           |
| ------------ | ------------------------ | ------------------------------ | -------------------- |
| **TRENDING** | Strong uptrend/downtrend | BUY/SELL                       | Low-Medium           |
| **RANGING**  | Sideways consolidation   | BUY/SELL at support/resistance | Medium               |
| **VOLATILE** | High volatility swings   | Caution signals                | High                 |
| **QUIET**    | Very low volatility      | Avoid trading                  | Very Low opportunity |

### Configuration Examples

**Conservative (Only Trade Strong Trends):**

```yaml
strategy:
  allowed_regimes:
    - trending
  regime_confidence_threshold: 0.7 # High confidence required
```

**Moderate (Trade Trends + Ranges):**

```yaml
strategy:
  allowed_regimes:
    - trending
    - ranging
  regime_confidence_threshold: 0.5
```

**Aggressive (Trade Almost Everything):**

```yaml
strategy:
  allowed_regimes:
    - trending
    - ranging
    - mean_reversion
  regime_confidence_threshold: 0.3 # Lower threshold
```

### Viewing Regime Classifications

Add this to your trading script to see regime analysis:

```python
from binance_regime_trader import LiveTradeManager

manager = LiveTradeManager()
await manager.initialize()

# Fetch prices
prices = await manager.fetch_prices()

# Get historical data
historical_data = await manager.fetch_historical_data()

# Generate signals (includes regime info)
signals = await manager.generate_signals(prices, historical_data)

# Print regime info
for symbol, signal in signals.items():
    print(f"{symbol}: regime={signal['regime']}, "
          f"regime_confidence={signal.get('regime_confidence', 0):.2f}")
```

---

## Trade Record Analysis

### Trade Data Structure

Every trade is recorded with metadata:

```python
Trade(
    symbol='BTC/USDT',
    timestamp=datetime(2025, 12, 30, 15, 30, 45),
    side='buy',                     # 'buy' or 'sell'
    price=43250.0,
    amount=0.01,
    regime='trending',              # Market regime at execution
    regime_confidence=0.85,         # Confidence in regime classification
    signal_confidence=0.85,         # Strategy confidence in the signal
    order_id=None                   # None in paper mode
)
```

### Analyzing Trade Performance

```bash
# View all trades from last session
tail logs/trading.log | grep "Trade\|Order"

# Count buy vs sell
grep "side='buy'" logs/trading.log | wc -l   # Buys
grep "side='sell'" logs/trading.log | wc -l  # Sells

# Profitable trades
grep "unrealized_pnl.*[1-9]" logs/trading.log | wc -l
```

---

## Troubleshooting

### Issue: No Trades Executed

**Possible Causes:**

1. All regimes are filtered out
2. Confidence thresholds too high
3. Strategy not initialized correctly

**Solution:**

```yaml
strategy:
  allowed_regimes:
    - trending
    - ranging
    - mean_reversion # Enable all regimes
  regime_confidence_threshold: 0.3 # Lower threshold
```

### Issue: Frequent HOLD Signals

**Possible Causes:**

1. Market is consolidating (ranging regime)
2. Volatility too low
3. Price at no clear support/resistance

**Solution:**

```python
# Adjust EMA periods for faster signals
config['strategy']['ema_short'] = 9   # Was 12
config['strategy']['ema_long'] = 26   # Was 26
```

### Issue: High Slippage in Paper Mode

Paper mode simulates orders at current price. In live mode:

- Use limit orders instead of market orders
- Set price bounds: `price ± slippage_pct`

```python
await manager.execute_trade(
    signal=signal,
    symbol='BTC/USDT',
    price=43250.0,
    order_type='limit',         # Use limit orders
    slippage_pct=0.1            # Allow 0.1% slippage
)
```

### Issue: Session Crashes

**Check logs:**

```bash
# View last error
tail -50 logs/trading.log | grep -i error
```

**Common errors:**

- API key invalid → Check `core/config.yaml`
- Network timeout → Check internet connection
- Insufficient balance → Check initial_capital in config

---

## Performance Optimization

### Reduce CPU Usage

```yaml
trading:
  price_update_interval: 10.0 # Fetch every 10 seconds (vs 5)
  symbols:
    - BTC/USDT # Trade fewer pairs
    - ETH/USDT
```

### Improve Signal Quality

```yaml
strategy:
  ema_short: 9
  ema_long: 26
  regime_confidence_threshold: 0.6 # Higher = stricter filtering
```

### Faster Order Execution

```python
# Reduce position sizing for faster fills
config['trading']['allocation_per_symbol'] = 0.1  # 10% vs 20%
```

---

## Graceful Shutdown

### Normal Shutdown (Ctrl+C)

```
$ python3 binance_regime_trader.py
...trading...
^C
Received SIGINT signal, shutting down gracefully...
Closing Binance client...
Updating final equity...
Generating final report...
✓ Report saved to logs/backtest_summary.txt
✓ Trades logged to logs/trading.log
Session ended.
```

### Emergency Shutdown (Kill Process)

```bash
# Find process
ps aux | grep binance_regime_trader

# Kill gracefully (SIGTERM)
kill <PID>

# Force kill (SIGKILL) if needed
kill -9 <PID>
```

---

## Testing Before Live Trading

### 1. Dry Run (Backtest Mode)

```bash
python3 binance_regime_trader.py --backtest
```

### 2. Paper Trading Test

```bash
# Run for 5 minutes in paper mode
python3 binance_regime_trader.py --duration 300
```

### 3. Verify Configuration

```bash
# Check config loads correctly
python3 -c "import yaml; print(yaml.safe_load(open('core/config.yaml')))"
```

### 4. Run Unit Tests

```bash
# Run all tests
pytest tests/test_live_trading.py -v

# Run specific test class
pytest tests/test_live_trading.py::TestLiveTradeManager -v
```

---

## Integration with Monitoring Tools

### Send Alerts on Signals

```python
import smtplib

async def send_signal_alert(symbol, signal):
    """Send email alert when signal generated."""
    msg = f"{symbol}: {signal['action']} @ {signal['price']}"

    # Send via your email service
    # (implement your alerting logic here)
    pass
```

### Log to External Service

```python
import requests

async def log_to_service(trade):
    """Send trade to external monitoring service."""
    await requests.post(
        'https://your-service.com/trades',
        json=trade.to_dict()
    )
```

---

## Advanced Usage

### Custom Strategy Integration

```python
from strategies.base import Strategy

class CustomStrategy(Strategy):
    def analyze(self, df):
        # Your custom logic
        return {
            'action': 'buy',
            'confidence': 0.75,
            'regime': 'trending',
            'regime_confidence': 0.85
        }

# Use in live trading
manager.strategies['BTC/USDT'] = CustomStrategy()
```

### Multi-Timeframe Analysis

```python
# Combine 1H and 4H signals
signals_1h = await strategy_1h.analyze(df_1h)
signals_4h = await strategy_4h.analyze(df_4h)

# Execute only if both agree
if signals_1h['action'] == signals_4h['action']:
    await manager.execute_trade(signals_1h, 'BTC/USDT', price)
```

---

## Support & Debugging

### Collect Debug Information

When reporting issues, include:

```bash
# System info
python3 --version
pip freeze | grep ccxt

# Config (sanitized)
cat core/config.yaml | grep -v "api_key\|secret"

# Recent logs
tail -100 logs/trading.log

# Test results
pytest tests/test_live_trading.py -v
```

### Check Binance Status

- Testnet status: https://testnet.binance.vision
- Live API status: https://www.binance.com/en/support/announcement

---

## Checklist: Before Going Live

- [ ] Test strategy in backtest mode (`--backtest`)
- [ ] Run paper trading test (`--duration 300`)
- [ ] Pass all unit tests (`pytest tests/test_live_trading.py -v`)
- [ ] Monitor for 5+ minutes without errors
- [ ] Verify signals align with market conditions
- [ ] Check logs for warnings or unusual activity
- [ ] Have kill command ready (for emergency shutdown)
- [ ] Start with small position sizes (`allocation_per_symbol: 0.05`)
- [ ] Monitor closely for first 24 hours
- [ ] Review final report (`logs/backtest_summary.txt`)

---

**Last Updated**: December 30, 2025
**Status**: ✅ Production Ready (21/21 tests passing)
