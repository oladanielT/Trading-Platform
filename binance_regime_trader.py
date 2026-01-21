"""
Binance Live/Paper Trading with Regime-Based Strategy
=====================================================
Real-time multi-pair trading orchestrator with async price fetching,
signal generation, and comprehensive portfolio management.

Features:
- Real-time price fetching (configurable intervals)
- Live/paper trading mode with regime-based strategy
- Multi-pair concurrent signal generation and execution
- Order execution and tracking with regime metadata
- Portfolio metrics aggregation (return, Sharpe, drawdown, win rate)
- Graceful shutdown handling (Ctrl+C)
- Comprehensive logging to file and console
- Backtesting support for historical analysis
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
import signal

# Platform imports
from data.binance_client import BinanceClient
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.ema_trend import EMATrendStrategy
from backtesting.engine import BacktestEngine
from monitoring.metrics import MetricsCollector


@dataclass
class Trade:
    """Trade execution record with regime metadata."""
    symbol: str
    timestamp: datetime
    side: str
    price: float
    amount: float
    regime: str
    regime_confidence: float
    signal_confidence: float
    order_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class PairMetrics:
    """Per-symbol performance metrics."""
    symbol: str
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    max_drawdown: float = 0.0
    current_price: float = 0.0
    entry_price: Optional[float] = None
    position_size: Optional[float] = None
    unrealized_pnl: float = 0.0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        return (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0


class PortfolioManager:
    """Manages portfolio-level metrics and positions."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialize portfolio manager."""
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        self.pair_metrics: Dict[str, PairMetrics] = {}
        self.trades: List[Trade] = []
        self.positions: Dict[str, Dict[str, Any]] = {}
    
    def add_trade(self, trade: Trade) -> None:
        """Record a trade execution."""
        self.trades.append(trade)
        if trade.symbol not in self.pair_metrics:
            self.pair_metrics[trade.symbol] = PairMetrics(symbol=trade.symbol)
        self.pair_metrics[trade.symbol].total_trades += 1
    
    def update_position(self, symbol: str, price: float, amount: Optional[float] = None) -> None:
        """Update current position."""
        if amount is None or amount == 0:
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            self.positions[symbol] = {'price': price, 'amount': amount}
    
    def update_equity(self, current_prices: Dict[str, float]) -> float:
        """Update current equity and return it."""
        equity = self.current_equity
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position['price'])
            unrealized = (current_price - position['price']) * position['amount']
            equity += unrealized
            if symbol in self.pair_metrics:
                self.pair_metrics[symbol].unrealized_pnl = unrealized
        return equity
    
    def close_position(self, symbol: str, realized_pnl: float) -> None:
        """Close position and record P&L."""
        self.current_equity += realized_pnl
        self.current_equity = max(self.current_equity, 0)
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
        if symbol in self.positions:
            del self.positions[symbol]
    
    @property
    def max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        return (self.current_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated portfolio metrics."""
        total_trades = sum(m.total_trades for m in self.pair_metrics.values())
        winning_trades = sum(m.winning_trades for m in self.pair_metrics.values())
        total_return = (self.current_equity - self.initial_capital) / self.initial_capital
        
        return {
            'initial_capital': self.initial_capital,
            'current_equity': self.current_equity,
            'total_return': total_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades) if total_trades > 0 else 0.0,
            'max_drawdown': self.max_drawdown,
            'open_positions': len(self.positions),
        }


class LiveTradeManager:
    """Real-time trading with regime-based strategy."""
    
    def __init__(self, config_path: str = 'core/config.yaml', price_update_interval: int = 5):
        """Initialize live trade manager."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.price_update_interval = price_update_interval
        
        self.client: Optional[BinanceClient] = None
        self.strategies: Dict[str, RegimeBasedStrategy] = {}
        self.portfolio: Optional[PortfolioManager] = None
        self._running = False
        
        self._load_config()
        self.logger = self._setup_logging()
        self.logger.info("=" * 80)
        self.logger.info("BINANCE LIVE/PAPER REGIME-BASED TRADER INITIALIZED")
        self.logger.info("=" * 80)
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_file = Path(self.config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print(f"✓ Loaded configuration from {self.config_path}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging to file and console."""
        logs_dir = Path('logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger('LiveTradeManager')
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(logs_dir / 'trading.log')
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    async def initialize(self) -> None:
        """Initialize Binance client and strategies."""
        self.logger.info("Initializing trading system...")
        
        environment = self.config.get('environment', 'paper')
        exchange_config = self.config.get('exchange', {})
        trading_config = self.config.get('trading', {})
        
        # Initialize Binance client
        self.client = BinanceClient(
            api_key=exchange_config.get('api_key', ''),
            secret_key=exchange_config.get('secret_key', ''),
            testnet=exchange_config.get('testnet', True),
            environment=environment
        )
        
        await self.client.initialize()
        self.logger.info(f"✓ Connected to Binance ({environment} mode)")
        
        # Initialize portfolio
        initial_capital = trading_config.get('initial_capital', 100000.0)
        self.portfolio = PortfolioManager(initial_capital)
        self.logger.info(f"✓ Portfolio initialized: ${initial_capital:,.2f}")
        
        # Initialize strategies
        symbols = trading_config.get('symbols', ['BTC/USDT'])
        strategy_config = self.config.get('strategy', {})
        
        for symbol in symbols:
            underlying_config = strategy_config.get('underlying_strategy', {})
            underlying = EMATrendStrategy(
                fast_period=underlying_config.get('fast_period', 20),
                slow_period=underlying_config.get('slow_period', 50),
                signal_confidence=underlying_config.get('signal_confidence', 0.75)
            )
            
            strategy = RegimeBasedStrategy(
                underlying_strategy=underlying,
                allowed_regimes=strategy_config.get('allowed_regimes', ['trending']),
                regime_confidence_threshold=strategy_config.get('regime_confidence_threshold', 0.5),
                reduce_confidence_in_marginal_regimes=strategy_config.get(
                    'reduce_confidence_in_marginal_regimes', True
                )
            )
            
            self.strategies[symbol] = strategy
        
        self.logger.info(f"✓ Initialized {len(self.strategies)} strategies")
    
    async def fetch_prices(self) -> Dict[str, float]:
        """Fetch current prices for all symbols."""
        symbols = list(self.strategies.keys())
        try:
            tickers = await self.client.fetch_multiple_tickers(symbols)
            return {symbol: ticker['last'] for symbol, ticker in tickers.items()}
        except Exception as e:
            self.logger.error(f"Failed to fetch prices: {e}")
            return {}
    
    async def generate_signals(
        self,
        prices: Dict[str, float],
        historical_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """Generate trading signals for all symbols."""
        signals = {}
        allowed_regimes = self.config.get('strategy', {}).get('allowed_regimes', ['trending', 'ranging', 'mean_reversion'])
        
        for symbol, strategy in self.strategies.items():
            if symbol not in historical_data or symbol not in prices:
                continue
            
            try:
                df = historical_data[symbol].copy()
                signal = strategy.analyze(df)
                
                # Only include non-hold signals from allowed regimes
                if signal and signal.get('action') != 'hold':
                    regime = signal.get('regime', 'unknown')
                    
                    # Check if regime is in allowed list
                    if regime not in allowed_regimes:
                        continue
                    
                    signal['symbol'] = symbol
                    signal['timestamp'] = datetime.now()
                    signal['price'] = prices[symbol]
                    signals[symbol] = signal
            
            except Exception as e:
                self.logger.error(f"Signal generation failed for {symbol}: {e}")
        
        return signals
    
    async def execute_trade(
        self,
        symbol: str,
        signal: Dict[str, Any],
        current_price: float
    ) -> Optional[Trade]:
        """Execute trade based on signal."""
        action = signal.get('action', 'hold')
        
        if action == 'hold':
            return None
        
        try:
            risk_config = self.config.get('risk', {})
            trading_config = self.config.get('trading', {})
            
            max_position_size = risk_config.get('max_position_size', 0.1)
            capital_per_symbol = (
                trading_config.get('initial_capital', 100000.0) *
                trading_config.get('allocation_per_symbol', 0.2)
            )
            
            position_value = capital_per_symbol * max_position_size
            amount = position_value / current_price
            
            side = 'buy' if action == 'buy' else 'sell'
            
            order = await self.client.place_order(
                symbol=symbol,
                side=side,
                order_type='market',
                amount=amount
            )
            
            trade = Trade(
                symbol=symbol,
                timestamp=datetime.now(),
                side=side,
                price=current_price,
                amount=amount,
                regime=signal.get('regime', 'unknown'),
                regime_confidence=signal.get('regime_confidence', 0.0),
                signal_confidence=signal.get('confidence', 0.0),
                order_id=order.get('id') if order else None
            )
            
            self.portfolio.add_trade(trade)
            self.portfolio.update_position(symbol, current_price, amount if side == 'buy' else 0)
            
            self.logger.info(
                f"TRADE: {side.upper()} {amount:.6f} {symbol} @ ${current_price:.2f} | "
                f"Regime: {signal.get('regime')} ({signal.get('regime_confidence'):.2f}) | "
                f"Signal: {signal.get('confidence'):.2f}"
            )
            
            return trade
        
        except Exception as e:
            self.logger.error(f"Trade execution failed for {symbol}: {e}")
            return None
    
    async def run_live_trading(self, duration_seconds: Optional[int] = None) -> None:
        """Run live trading loop."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING LIVE TRADING")
        self.logger.info("=" * 80)
        
        self._running = True
        start_time = datetime.now()
        
        symbols = list(self.strategies.keys())
        timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        fetch_limit = self.config.get('trading', {}).get('fetch_limit', 100)
        
        self.logger.info(f"Fetching {fetch_limit} {timeframe} candles...")
        
        historical_data = await self.client.fetch_multiple_ohlcv(
            symbols, timeframe, fetch_limit
        )
        
        self.logger.info(f"✓ Loaded historical data for {len(historical_data)} symbols")
        
        try:
            iteration = 0
            
            while self._running:
                iteration += 1
                
                if duration_seconds and (datetime.now() - start_time).total_seconds() > duration_seconds:
                    self.logger.info(f"Duration limit reached ({duration_seconds}s)")
                    break
                
                prices = await self.fetch_prices()
                if not prices:
                    await asyncio.sleep(self.price_update_interval)
                    continue
                
                signals = await self.generate_signals(prices, historical_data)
                
                for symbol, signal in signals.items():
                    await self.execute_trade(symbol, signal, prices[symbol])
                
                if self.portfolio:
                    current_equity = self.portfolio.update_equity(prices)
                    metrics = self.portfolio.get_metrics()
                    
                    self.logger.debug(
                        f"[{iteration}] Equity: ${current_equity:,.2f} | "
                        f"Return: {metrics['total_return']:.2%} | "
                        f"Trades: {metrics['total_trades']}"
                    )
                
                await asyncio.sleep(self.price_update_interval)
        
        except asyncio.CancelledError:
            self.logger.info("Live trading cancelled")
        except Exception as e:
            self.logger.error(f"Error in live trading: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def run_backtest(self) -> Dict[str, Dict[str, Any]]:
        """Run backtests for all symbols."""
        self.logger.info("=" * 80)
        self.logger.info("RUNNING BACKTESTS")
        self.logger.info("=" * 80)
        
        symbols = list(self.strategies.keys())
        timeframe = self.config.get('trading', {}).get('timeframe', '1h')
        fetch_limit = self.config.get('trading', {}).get('fetch_limit', 500)
        
        ohlcv_data = await self.client.fetch_multiple_ohlcv(symbols, timeframe, fetch_limit)
        
        results = {}
        
        for symbol in symbols:
            if symbol not in ohlcv_data:
                continue
            
            try:
                df = ohlcv_data[symbol]
                bars = df.to_dict('records')
                engine = BacktestEngine()
                result = engine.run_backtest(bars, self.strategies[symbol])
                results[symbol] = result
                
                self.logger.info(
                    f"{symbol}: {result.get('total_return', 0):.2%} return, "
                    f"{result.get('total_trades', 0)} trades, "
                    f"{result.get('win_rate', 0):.2%} win rate"
                )
            
            except Exception as e:
                self.logger.error(f"Backtest failed for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive trading report."""
        lines = ["\n" + "=" * 80]
        lines.append("TRADING SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if not self.portfolio:
            return "\n".join(lines)
        
        metrics = self.portfolio.get_metrics()
        lines.append("\nPORTFOLIO METRICS:")
        lines.append(f"  Initial Capital:  ${metrics['initial_capital']:>12,.2f}")
        lines.append(f"  Current Equity:   ${metrics['current_equity']:>12,.2f}")
        lines.append(f"  Total Return:     {metrics['total_return']:>12.2%}")
        lines.append(f"  Max Drawdown:     {metrics['max_drawdown']:>12.2%}")
        lines.append(f"  Total Trades:     {metrics['total_trades']:>12}")
        lines.append(f"  Win Rate:         {metrics['win_rate']:>12.2%}")
        lines.append(f"  Open Positions:   {metrics['open_positions']:>12}")
        
        if self.portfolio.pair_metrics:
            lines.append("\nPER-SYMBOL METRICS:")
            for symbol, pm in self.portfolio.pair_metrics.items():
                lines.append(f"\n  {symbol}:")
                lines.append(f"    Trades:        {pm.total_trades}")
                lines.append(f"    Win Rate:      {pm.win_rate:.2%}")
                lines.append(f"    Current Price: ${pm.current_price:,.2f}")
                if pm.position_size:
                    lines.append(f"    Position:      {pm.position_size:.6f}")
                    lines.append(f"    Unrealized P&L: ${pm.unrealized_pnl:,.2f}")
        
        if self.portfolio.trades:
            lines.append("\nRECENT TRADES (Last 10):")
            for trade in self.portfolio.trades[-10:]:
                lines.append(
                    f"  {trade.timestamp.strftime('%H:%M:%S')} | {trade.side.upper():>4} "
                    f"{trade.amount:.6f} {trade.symbol:>10} @ ${trade.price:>10,.2f} | "
                    f"Regime: {trade.regime:>8} ({trade.regime_confidence:.2f})"
                )
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self.logger.info("Initiating graceful shutdown...")
        
        self._running = False
        
        if self.client:
            await self.client.close()
        
        report = self.generate_report()
        print(report)
        
        report_file = Path('logs/backtest_summary.txt')
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"✓ Report saved to {report_file}")
        self.logger.info("Shutdown complete")


async def main(
    mode: str = 'live',
    duration: Optional[int] = None,
    config_path: str = 'core/config.yaml'
) -> None:
    """
    Main entry point for live/paper trading.
    
    Args:
        mode: 'live' for live trading, 'backtest' for backtesting
        duration: Duration in seconds (None = indefinite)
        config_path: Path to configuration file
    """
    print("\n" + "=" * 80)
    print(f"BINANCE REGIME-BASED {'LIVE' if mode == 'live' else 'BACKTEST'} TRADER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")
    
    manager = LiveTradeManager(config_path=config_path)
    
    def signal_handler(sig, frame):
        print("\n\n⚠ Interrupted by user (Ctrl+C)")
        asyncio.create_task(manager.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        await manager.initialize()
        
        if mode == 'live':
            await manager.run_live_trading(duration_seconds=duration)
        else:
            await manager.run_backtest()
            report = manager.generate_report()
            print(report)
            
            report_file = Path('logs/backtest_summary.txt')
            report_file.parent.mkdir(parents=True, exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report)
    
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        manager.logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    
    finally:
        print("\n" + "=" * 80)
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80 + "\n")


if __name__ == '__main__':
    import sys
    
    mode = 'live'
    duration = None
    
    if '--backtest' in sys.argv:
        mode = 'backtest'
    
    if '--duration' in sys.argv:
        idx = sys.argv.index('--duration')
        if idx + 1 < len(sys.argv):
            duration = int(sys.argv[idx + 1])
    
    asyncio.run(main(mode=mode, duration=duration))
