# Binance Regime-Based Multi-Pair Trading System

## Overview

This module integrates the **RegimeBasedStrategy** with **Binance Testnet/Live** trading, enabling automated multi-pair trading with comprehensive logging, backtesting, and order execution.

## Features

✅ **Binance Connectivity**

- Async client for Binance testnet/live API
- Multi-pair price fetching
- Historical OHLCV data retrieval
- Order placement (market/limit)

✅ **Regime-Based Strategy**

- 4 market regimes (TRENDING, RANGING, VOLATILE, QUIET)
- Confidence-based filtering
- EMA trend underlying strategy
- Configurable regime thresholds

✅ **Multi-Pair Trading**

- Concurrent backtesting across 5+ pairs
- Portfolio-level metrics
- Allocation per symbol (20% default)
- Risk management integration

✅ **Comprehensive Logging**

- Signal logging with regime metadata
- Order tracking
- Performance metrics
- Error handling with retries

✅ **Environment Modes**

- **Paper**: Simulated orders, no real execution
- **Testnet**: Binance testnet with test funds
- **Live**: Real trading (use with caution!)

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure ccxt is installed with async support
pip install ccxt --upgrade
```

### 2. Configuration

Edit `core/config.yaml` with your credentials:

```yaml
# Trading environment
environment: paper # paper | testnet | live

# Exchange configuration
exchange:
  id: binance
  testnet: true
  api_key: "YOUR_API_KEY_HERE"
  secret_key: "YOUR_SECRET_KEY_HERE"

# Trading parameters
trading:
  symbols:
    - BTC/USDT
    - ETH/USDT
    - BNB/USDT
    - XRP/USDT
    - ADA/USDT

  timeframe: 1h
  initial_capital: 100000.0
  fetch_limit: 500
  allocation_per_symbol: 0.2 # 20% per pair
```

**Current Testnet Credentials** (already configured):

- **API Key**: `Cq9XgedSyra43Yn6zEaGvdq9vf6FV91zCa6T4kffIaV4xxJbydvelVOibLVrxhvJ`
- **Secret Key**: `beumcOk2cGfjJgQbIH69rZSnRRjWSr2E7KKfwnTQCIhrWXhGwK1FXHHb9k7g4EAT`

### 3. Run Multi-Pair Backtest

```bash
# Run the Binance regime trader
python3 binance_regime_trader.py
```

**Expected Output**:

```
================================================================================
BINANCE REGIME-BASED MULTI-PAIR TRADER
================================================================================
Started at: 2025-12-30 10:30:15
================================================================================

✓ Loaded configuration from core/config.yaml
✓ Binance client initialized successfully
✓ Strategy ready for BTC/USDT
✓ Strategy ready for ETH/USDT
...

Fetching current prices for 5 symbols...
  BTC/USDT: $43,250.00
  ETH/USDT: $2,345.67
  ...

Running backtest for BTC/USDT...
✓ Loaded 500 candles (2025-10-01 to 2025-12-30)
Backtest completed for BTC/USDT:
  Total Return: 15.43%
  Total Trades: 23
  Win Rate: 65.22%
  Max Drawdown: -8.32%
  Sharpe Ratio: 1.85
...

================================================================================
MULTI-PAIR BACKTEST SUMMARY
================================================================================

BTC/USDT:
  Total Return:      15.43%
  Total Trades:          23
  Win Rate:          65.22%
  Profit Factor:       2.15
  Max Drawdown:      -8.32%
  Sharpe Ratio:        1.85
...

PORTFOLIO AGGREGATES:
  Avg Return:        12.35%
  Total Trades:         115
  Avg Win Rate:      62.18%
  Avg Sharpe:          1.72
  Symbols Tested:          5
================================================================================
```

---

## Module Structure

### 1. `data/binance_client.py`

**BinanceClient** - Async Binance API wrapper

**Methods**:

- `initialize()` - Connect to Binance
- `fetch_ticker(symbol)` - Get current price
- `fetch_ohlcv(symbol, timeframe, limit)` - Get historical candles
- `place_order(symbol, side, type, amount, price)` - Execute order
- `fetch_balance()` - Get account balance
- `fetch_multiple_tickers(symbols)` - Batch price fetch
- `fetch_multiple_ohlcv(symbols, timeframe, limit)` - Batch OHLCV fetch

**Features**:

- Retry logic for network errors
- Rate limiting support
- Paper/testnet/live mode switching
- Order simulation in paper mode

### 2. `binance_regime_trader.py`

**BinanceRegimeTrader** - Main orchestrator

**Methods**:

- `initialize()` - Setup client and strategies
- `fetch_current_prices()` - Get live prices for all pairs
- `fetch_historical_data(symbol)` - Load OHLCV for backtesting
- `run_backtest(symbol, ohlcv_data)` - Execute backtest
- `simulate_order_placement(symbol, signal, price)` - Place orders
- `run_multi_pair_backtest()` - Run all backtests concurrently
- `generate_summary()` - Create performance report
- `run()` - Main execution loop

**Workflow**:

1. Load config from `config.yaml`
2. Initialize Binance client
3. Create RegimeBasedStrategy for each pair
4. Fetch current market prices
5. Download historical data
6. Run backtests concurrently
7. Generate summary report
8. Log all activities

---

## Usage Examples

### Example 1: Basic Multi-Pair Backtest

```python
import asyncio
from binance_regime_trader import BinanceRegimeTrader

async def main():
    trader = BinanceRegimeTrader(config_path='core/config.yaml')
    await trader.run()

asyncio.run(main())
```

### Example 2: Fetch Live Prices Only

```python
import asyncio
from data.binance_client import BinanceClient

async def get_prices():
    client = BinanceClient(
        api_key='YOUR_KEY',
        secret_key='YOUR_SECRET',
        testnet=True,
        environment='paper'
    )

    await client.initialize()

    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    tickers = await client.fetch_multiple_tickers(symbols)

    for symbol, ticker in tickers.items():
        print(f"{symbol}: ${ticker['last']:.2f}")

    await client.close()

asyncio.run(get_prices())
```

### Example 3: Single Pair Backtest

```python
import asyncio
from binance_regime_trader import BinanceRegimeTrader

async def backtest_btc():
    trader = BinanceRegimeTrader()
    await trader.initialize()

    # Fetch historical data
    ohlcv = await trader.fetch_historical_data('BTC/USDT', '1h', 500)

    # Run backtest
    results = await trader.run_backtest('BTC/USDT', ohlcv)

    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    await trader.close()

asyncio.run(backtest_btc())
```

---

## Configuration Details

### Environment Modes

```yaml
# Paper Mode: Simulated orders, no API calls
environment: paper

# Testnet Mode: Binance testnet with test funds
environment: testnet
exchange:
  testnet: true
  api_key: 'TESTNET_KEY'
  secret_key: 'TESTNET_SECRET'

# Live Mode: Real trading (⚠️ use caution!)
environment: live
exchange:
  testnet: false
  api_key: 'LIVE_KEY'
  secret_key: 'LIVE_SECRET'
```

### Regime Strategy Settings

```yaml
strategy:
  name: regime_based

  # Which regimes to trade in
  allowed_regimes:
    - trending # Trade in trending markets
    # - ranging     # Trade in ranging markets
    # - volatile    # Trade in volatile markets
    # - quiet       # Trade in quiet markets

  # Minimum confidence to trust regime classification
  regime_confidence_threshold: 0.5

  # Scale signal confidence by regime confidence
  reduce_confidence_in_marginal_regimes: true

  # Underlying strategy (EMA Trend)
  underlying_strategy:
    name: ema_trend
    fast_period: 20
    slow_period: 50
    signal_confidence: 0.75
```

### Risk Management

```yaml
risk:
  max_position_size: 0.1 # 10% of equity per position
  risk_per_trade: 0.02 # 2% risk per trade
  sizing_method: fixed_risk # fixed_risk, kelly, etc.

  max_daily_drawdown: 0.05 # 5% daily loss limit
  max_total_drawdown: 0.20 # 20% total drawdown limit
  max_consecutive_losses: 5 # Max losing streak
```

---

## Logging and Monitoring

### Log Files

- **Main Log**: `logs/binance_trader.log`
- **Summary Report**: `logs/backtest_summary.txt`
- **Platform Log**: `logs/trading.log`

### Log Levels

```yaml
monitoring:
  log_level: INFO # DEBUG, INFO, WARNING, ERROR, CRITICAL
  log_file: logs/binance_trader.log
  metrics_enabled: true
```

### Sample Log Output

```
2025-12-30 10:30:15 [INFO] BinanceRegimeTrader - ✓ Binance client initialized
2025-12-30 10:30:16 [INFO] BinanceRegimeTrader - Creating RegimeBasedStrategy for BTC/USDT
2025-12-30 10:30:17 [INFO] BinanceClient - Fetched ticker for BTC/USDT: $43250.00
2025-12-30 10:30:18 [INFO] BinanceClient - Fetched 500 1h candles for BTC/USDT
2025-12-30 10:30:20 [INFO] BinanceRegimeTrader - Backtest completed for BTC/USDT:
2025-12-30 10:30:20 [INFO] BinanceRegimeTrader -   Total Return: 15.43%
2025-12-30 10:30:20 [INFO] BinanceRegimeTrader -   Total Trades: 23
2025-12-30 10:30:20 [INFO] BinanceRegimeTrader -   Win Rate: 65.22%
2025-12-30 10:30:21 [INFO] BinanceClient - 🔵 SIMULATED: BUY 0.025 BTC/USDT @ $43250.00
```

---

## Error Handling

### Network Errors

The client automatically retries network operations:

```python
# Automatic retry with exponential backoff
for attempt in range(max_retries):
    try:
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker
    except ccxt.NetworkError as e:
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (attempt + 1))
        else:
            raise
```

### Exchange Errors

```python
try:
    order = await client.place_order(...)
except ccxt.InsufficientFunds:
    logger.error("Insufficient funds")
except ccxt.InvalidOrder:
    logger.error("Invalid order parameters")
except ccxt.ExchangeError:
    logger.error("Exchange error")
```

### Graceful Shutdown

```python
try:
    await trader.run()
except KeyboardInterrupt:
    print("⚠ Interrupted by user")
finally:
    await trader.close()  # Cleanup connections
```

---

## Testing

### Run Unit Tests

```bash
# Test backtest engine with regime strategy
pytest tests/test_backtest_phase2_regime.py -v

# Test integration
pytest tests/test_integration_regime.py -v

# Run all tests
pytest tests/ -v
```

### Manual Testing

```bash
# Test Binance client connection
python3 -c "
import asyncio
from data.binance_client import BinanceClient

async def test():
    client = BinanceClient('key', 'secret', testnet=True, environment='paper')
    await client.initialize()
    ticker = await client.fetch_ticker('BTC/USDT')
    print(f'BTC: \${ticker[\"last\"]}')
    await client.close()

asyncio.run(test())
"

# Run full multi-pair trader
python3 binance_regime_trader.py
```

---

## Performance Metrics

### Backtest Metrics

Each backtest provides:

- **Total Return**: Overall profit/loss percentage
- **Total Trades**: Number of trades executed
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return

### Portfolio Metrics

Aggregated across all pairs:

- **Average Return**: Mean return across pairs
- **Total Trades**: Sum of all trades
- **Average Win Rate**: Mean win rate
- **Average Sharpe**: Mean Sharpe ratio
- **Symbols Tested**: Number of pairs backtested

---

## Troubleshooting

### Issue: Connection Errors

**Solution**: Verify API credentials and network connectivity

```bash
# Test API credentials
curl -X GET "https://testnet.binance.vision/api/v3/ping"
```

### Issue: Invalid Symbol

**Solution**: Check symbol format (use '/' separator)

```yaml
# Correct
symbols:
  - BTC/USDT
  - ETH/USDT

# Incorrect
symbols:
  - BTCUSDT  # Missing '/'
```

### Issue: Rate Limiting

**Solution**: Enable rate limiting in client

```python
client = BinanceClient(
    api_key='...',
    secret_key='...',
    testnet=True,
    environment='paper'
)
# Rate limiting enabled by default in ccxt
```

### Issue: Insufficient Test Funds

**Solution**: Request testnet funds from Binance

- Visit: https://testnet.binance.vision/
- Login and request test USDT/BTC

---

## Next Steps

1. **Run Initial Backtest**

   ```bash
   python3 binance_regime_trader.py
   ```

2. **Review Results**

   - Check `logs/backtest_summary.txt`
   - Analyze per-pair performance
   - Adjust strategy parameters

3. **Test on Testnet**

   - Switch to `environment: testnet`
   - Run with real API calls (test funds)
   - Monitor order execution

4. **Production Deployment** (⚠️ caution)
   - Switch to `environment: live`
   - Start with small position sizes
   - Monitor closely for 24-48 hours

---

## Live Trading vs Backtesting

### Backtesting Mode

- **Purpose**: Historical analysis and strategy validation
- **Data**: Historical OHLCV candles from Binance
- **Execution**: Paper simulation of orders
- **Use Case**: Strategy development and optimization
- **Command**: `python3 binance_regime_trader.py --backtest`

### Live Trading Mode

- **Purpose**: Real-time automated trading with continuous monitoring
- **Data**: Current market prices fetched every `price_update_interval` seconds
- **Execution**: Paper simulation OR real orders (configurable)
- **Use Case**: Production trading with regime-based safety filters
- **Command**: `python3 binance_regime_trader.py` (default is live mode)

### Key Differences

| Feature             | Backtesting         | Live Trading                |
| ------------------- | ------------------- | --------------------------- |
| Data Source         | Historical candles  | Real-time tickers           |
| Execution           | One-shot per candle | Continuous async loop       |
| Portfolio Tracking  | Per-symbol summary  | Real-time equity updates    |
| Signal Generation   | Once per backtest   | Every price update          |
| Position Management | Trade records only  | Active position tracking    |
| Trade Duration      | Instant             | Based on market conditions  |
| Risk Management     | Static allocation   | Dynamic equity-based sizing |

---

## Real-Time Price Fetching Configuration

The Live Trading system fetches prices at a configurable interval. Adjust in [core/config.yaml](core/config.yaml):

```yaml
trading:
  price_update_interval: 5 # Fetch prices every 5 seconds (default)
  symbols:
    - BTC/USDT
    - ETH/USDT
    - BNB/USDT
    - XRP/USDT
    - ADA/USDT
  allocation_per_symbol: 0.2 # 20% of capital per pair
```

**Impact of interval adjustment:**

- Smaller interval (1-2s): More responsive to market changes, higher CPU usage
- Default interval (5s): Balance between responsiveness and efficiency
- Larger interval (10-30s): Lower resource usage, slower reaction to signals

---

## Portfolio Metrics Tracking

### Real-Time Metrics (Updated Every Price Fetch)

The `PortfolioManager` tracks portfolio-level metrics:

```python
manager = LiveTradeManager()
await manager.initialize()

# During live trading, portfolio metrics are continuously updated:
metrics = manager.portfolio.get_metrics()

# Returns:
{
    'total_return': 0.0847,           # 8.47% from initial capital
    'total_return_pct': 8.47,         # Percentage format
    'total_equity': 108470.0,         # Current portfolio value
    'peak_equity': 108470.0,          # Highest equity reached
    'current_drawdown': 0.0,          # Current peak-to-trough decline
    'max_drawdown': -0.0325,          # -3.25% worst drawdown
    'total_trades': 47,               # Total trades executed
    'winning_trades': 31,             # Trades with profit > 0
    'win_rate': 0.66,                 # 66% win rate
    'num_symbols': 5                  # Symbols being traded
}
```

### Per-Symbol Metrics

Each symbol has individual metrics tracked in `PairMetrics`:

```python
for symbol, pair_metrics in manager.portfolio.pair_metrics.items():
    print(f"{symbol}:")
    print(f"  Total Return: {pair_metrics.total_return:.2%}")
    print(f"  Win Rate: {pair_metrics.win_rate:.1%}")
    print(f"  Trades: {pair_metrics.total_trades}")
    print(f"  Unrealized P&L: ${pair_metrics.unrealized_pnl:.2f}")
```

### Unrealized P&L Calculation

Unrealized profit/loss for open positions is calculated as:

```
Unrealized P&L = (Current Price - Entry Price) × Position Size
```

This is updated on every price fetch and reflected in metrics.

---

## Trade Execution and Order Logging

### Trade Record Structure

Each trade is recorded with comprehensive metadata:

```python
@dataclass
class Trade:
    symbol: str                  # BTC/USDT, ETH/USDT, etc.
    timestamp: datetime          # When the trade was executed
    side: str                    # 'buy' or 'sell'
    price: float                 # Execution price
    amount: float                # Position size
    regime: str                  # Market regime at time of trade
    regime_confidence: float     # Confidence in regime classification
    signal_confidence: float     # Strategy signal confidence
    order_id: Optional[str]      # Order ID (None in paper mode)
```

### Example Trade Execution Flow

```
1. Fetch current prices
   BTC/USDT: $43,250, ETH/USDT: $2,345, etc.

2. Generate signals for each symbol
   BTC/USDT → regime='trending', action='buy', confidence=0.85
   ETH/USDT → regime='trending', action='hold', confidence=0.75
   BNB/USDT → regime='ranging', action='sell', confidence=0.65

3. Filter by allowed regimes (configured in strategy)
   Only 'trending' allowed → BNB/USDT filtered out

4. Execute approved trades
   - BTC/USDT: BUY 0.01 at $43,250 (regime=trending, confidence=0.85)

5. Record trade with metadata
   Trade(
       symbol='BTC/USDT',
       timestamp=2025-12-30 15:30:45,
       side='buy',
       price=43250.0,
       amount=0.01,
       regime='trending',
       regime_confidence=0.85,
       signal_confidence=0.85,
       order_id=None  # Paper mode
   )

6. Update portfolio
   - Add position: BTC/USDT +0.01
   - Equity tracking: unrealized P&L calculated

7. Log to files
   - logs/trading.log: All trades and metrics
   - logs/backtest_summary.txt: Final report
```

### Regime-Based Safety Filtering

Only trading signals from allowed market regimes are executed:

```yaml
strategy:
  allowed_regimes:
    - trending # Always trade in trending markets
    # - ranging     # Disable ranging trades (commented out)
    # - mean_reversion  # Disable mean reversion trades
  regime_confidence_threshold: 0.5
```

**Filtering logic:**

1. Signal generated for symbol
2. Check signal's regime against allowed_regimes
3. If regime in allowed list → Execute trade
4. If regime NOT in allowed list → Skip trade (hold)

This prevents trading in regime conditions you're not confident in.

---

## Running Live Trading

### Command-Line Options

```bash
# Default: Run indefinitely in live trading mode
python3 binance_regime_trader.py

# Run for specific duration (seconds)
python3 binance_regime_trader.py --duration 300  # Trade for 5 minutes

# Run backtesting instead
python3 binance_regime_trader.py --backtest

# Combine options
python3 binance_regime_trader.py --duration 600  # Trade for 10 minutes (live mode)
```

### Example: 30-Second Live Trading Session

```bash
python3 binance_regime_trader.py --duration 30
```

Expected output:

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

[15:30:45] Fetching prices...
  BTC/USDT: $43,250.00
  ETH/USDT: $2,345.67
  BNB/USDT: $310.50
  XRP/USDT: $0.65
  ADA/USDT: $1.15

[15:30:45] Analyzing signals...
  BTC/USDT: BUY (regime=trending, confidence=0.85)
  ETH/USDT: HOLD (regime=trending, confidence=0.75)
  BNB/USDT: SELL (regime=ranging, confidence=0.60) - FILTERED OUT
  XRP/USDT: BUY (regime=trending, confidence=0.72)

[15:30:45] Executing 2 trades...
  ✓ BTC/USDT BUY 0.010 @ $43,250.00
  ✓ XRP/USDT BUY 6.77 @ $15.15

[15:30:45] Portfolio Update:
  Equity: $100,001.45
  Open Positions: 2
  Total Trades: 2

[15:31:50] Session Complete
  Duration: 30 seconds
  Total Return: +0.00%
  Total Trades: 2
  Symbols Traded: 2
```

### Graceful Shutdown

Press **Ctrl+C** at any time to gracefully shutdown:

```
^C
Received SIGINT signal, shutting down gracefully...
Closing Binance client...
Generating final report...
✓ Report saved to logs/backtest_summary.txt
Session ended.
```

---

---

## API Reference

### Trade (Dataclass)

```python
@dataclass
class Trade:
    symbol: str                  # Trading pair (e.g., 'BTC/USDT')
    timestamp: datetime          # Trade execution time
    side: str                    # 'buy' or 'sell'
    price: float                 # Execution price
    amount: float                # Position size
    regime: str                  # Market regime at execution
    regime_confidence: float     # Confidence in regime classification (0-1)
    signal_confidence: float     # Strategy signal confidence (0-1)
    order_id: Optional[str]      # Binance order ID (None in paper mode)

    def to_dict() -> Dict        # Convert to dictionary
```

### PairMetrics (Dataclass)

```python
@dataclass
class PairMetrics:
    symbol: str                  # Trading pair
    total_return: float          # Return in decimal (0.05 = 5%)
    total_trades: int            # Number of trades
    winning_trades: int          # Trades with profit > 0
    max_drawdown: float          # Worst peak-to-trough decline
    current_price: float         # Last market price
    entry_price: float           # Average entry price
    position_size: float         # Amount currently held
    unrealized_pnl: float        # P&L of open positions

    @property
    def win_rate() -> float      # Winning trades / Total trades
```

### PortfolioManager

```python
class PortfolioManager:
    def __init__(initial_capital: float)

    # Properties
    @property
    def current_equity() -> float           # Current portfolio value
    @property
    def peak_equity() -> float              # Highest equity reached
    @property
    def max_drawdown() -> float             # Worst drawdown from peak

    # Methods
    def add_trade(trade: Trade) -> None     # Record a trade
    def update_position(symbol: str, side: str, price: float, amount: float) -> None
    def update_equity(prices: Dict[str, float]) -> None  # Update from current prices
    def close_position(symbol: str) -> None # Close all positions for symbol
    def get_metrics() -> Dict               # Get aggregated portfolio metrics
```

### LiveTradeManager

```python
class LiveTradeManager:
    def __init__(
        config_path: str = 'core/config.yaml',
        price_update_interval: float = 5.0,
        environment: str = 'paper'
    )

    # Initialization
    async def initialize() -> None

    # Price fetching
    async def fetch_prices() -> Dict[str, float]

    # Signal generation
    async def generate_signals(
        prices: Dict[str, float],
        historical_data: Dict[str, DataFrame]
    ) -> Dict[str, Dict[str, Any]]

    # Trade execution
    async def execute_trade(signal: Dict, symbol: str, price: float) -> None

    # Main trading loops
    async def run_live_trading(duration_seconds: Optional[int] = None) -> None
    async def run_backtest() -> None

    # Reporting
    def generate_report() -> str

    # Cleanup
    async def shutdown() -> None
```

### BinanceClient

```python
class BinanceClient:
    def __init__(api_key, secret_key, testnet=True, environment='paper')
    async def initialize() -> None
    async def close() -> None
    async def fetch_ticker(symbol: str) -> Dict
    async def fetch_ohlcv(symbol, timeframe, limit, since) -> DataFrame
    async def place_order(symbol, side, order_type, amount, price) -> Dict
    async def fetch_balance() -> Dict
    async def fetch_multiple_tickers(symbols: List[str]) -> Dict
    async def fetch_multiple_ohlcv(symbols, timeframe, limit) -> Dict
```

---

## License

This module is part of the Trading Platform project.

---

## Support

For issues or questions:

1. Check logs in `logs/trading.log`
2. Review configuration in `core/config.yaml`
3. Run tests: `pytest tests/ -v`
4. Check test results: `tests/test_live_trading.py` (21 tests)
5. Check Binance API status: https://www.binance.com/en/support/announcement

---

**Last Updated**: December 30, 2025
**Test Status**: ✅ 21/21 tests passing
