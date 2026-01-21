"""
Binance Exchange Client
-----------------------
Async client for connecting to Binance (testnet/live) for price data and order execution.

Supports:
- Testnet and live environments
- Async price fetching for multiple symbols
- Historical OHLCV data retrieval
- Order placement simulation (testnet)
- Graceful error handling and retry logic
"""

import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Async Binance client for testnet/live trading.
    
    Features:
    - Async price fetching
    - OHLCV data retrieval
    - Order simulation/execution
    - Connection management
    - Error handling with retries
    """
    
    def __init__(
        self,
        api_key: str,
        secret_key: str,
        testnet: bool = True,
        environment: str = 'paper'
    ):
        """
        Initialize Binance client.
        
        Args:
            api_key: Binance API key
            secret_key: Binance secret key
            testnet: Use testnet endpoints if True
            environment: Trading environment ('paper', 'testnet', 'live')
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.testnet = testnet
        self.environment = environment
        self.exchange: Optional[ccxt.binance] = None
        self._initialized = False
        
        logger.info(
            f"BinanceClient created: environment={environment}, "
            f"testnet={testnet}"
        )
    
    async def initialize(self) -> None:
        """Initialize the exchange connection."""
        if self._initialized:
            logger.warning("Client already initialized")
            return
        
        # In paper mode, skip real connection
        if self.environment == 'paper':
            logger.info("✓ Binance client initialized in PAPER mode (no real connection)")
            self._initialized = True
            return
        
        try:
            config = {
                'apiKey': self.api_key,
                'secret': self.secret_key,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future' if self.environment != 'paper' else 'spot',
                }
            }
            
            # Set testnet URLs if in testnet mode
            if self.testnet and self.environment == 'testnet':
                config['urls'] = {
                    'api': {
                        'public': 'https://testnet.binance.vision/api',
                        'private': 'https://testnet.binance.vision/api',
                    }
                }
            
            self.exchange = ccxt.binance(config)
            
            # Load markets
            await self.exchange.load_markets()
            
            self._initialized = True
            logger.info(f"✓ Binance client initialized successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to initialize Binance client: {e}")
            raise
    
    async def close(self) -> None:
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close()
            self._initialized = False
            logger.info("Binance client connection closed")
    
    async def fetch_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current ticker/price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
        
        Returns:
            Ticker data with current price
        
        Raises:
            Exception: If fetching fails after retries
        """
        if not self._initialized:
            await self.initialize()
        
        # In paper mode, return simulated ticker
        if self.environment == 'paper':
            base_prices = {
                'BTC/USDT': 43250.0,
                'ETH/USDT': 2345.67,
                'BNB/USDT': 315.89,
                'XRP/USDT': 0.6234,
                'ADA/USDT': 0.4821,
            }
            price = base_prices.get(symbol, 100.0)
            return {
                'symbol': symbol,
                'last': price,
                'bid': price * 0.9999,
                'ask': price * 1.0001,
                'high': price * 1.05,
                'low': price * 0.95,
                'timestamp': int(datetime.now().timestamp() * 1000),
                'datetime': datetime.now().isoformat(),
            }
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                ticker = await self.exchange.fetch_ticker(symbol)
                logger.debug(f"Fetched ticker for {symbol}: ${ticker['last']:.2f}")
                return ticker
            
            except ccxt.NetworkError as e:
                logger.warning(
                    f"Network error fetching {symbol} ticker "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise
            
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching {symbol} ticker: {e}")
                raise
            
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol} ticker: {e}")
                raise
        
        raise Exception(f"Failed to fetch ticker for {symbol} after {max_retries} retries")
    
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = '1h',
        limit: int = 500,
        since: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candlestick timeframe ('1m', '5m', '1h', '1d', etc.)
            limit: Number of candles to fetch
            since: Start timestamp in milliseconds (optional)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        
        Raises:
            Exception: If fetching fails after retries
        """
        if not self._initialized:
            await self.initialize()
        
        # In paper mode, generate synthetic data
        if self.environment == 'paper':
            return self._generate_synthetic_ohlcv(symbol, timeframe, limit)
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    limit=limit,
                    since=since
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(
                    ohlcv,
                    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                )
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                logger.info(
                    f"Fetched {len(df)} {timeframe} candles for {symbol} "
                    f"({df['timestamp'].min()} to {df['timestamp'].max()})"
                )
                
                return df
            
            except ccxt.NetworkError as e:
                logger.warning(
                    f"Network error fetching {symbol} OHLCV "
                    f"(attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    raise
            
            except ccxt.ExchangeError as e:
                logger.error(f"Exchange error fetching {symbol} OHLCV: {e}")
                raise
            
            except Exception as e:
                logger.error(f"Unexpected error fetching {symbol} OHLCV: {e}")
                raise
        
        raise Exception(f"Failed to fetch OHLCV for {symbol} after {max_retries} retries")
    
    def _generate_synthetic_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for paper trading."""
        # Base prices for different symbols
        base_prices = {
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2300.0,
            'BNB/USDT': 310.0,
            'XRP/USDT': 0.62,
            'ADA/USDT': 0.48,
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate timestamps
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        seconds = timeframe_seconds.get(timeframe, 3600)
        
        now = datetime.now()
        timestamps = [
            now - pd.Timedelta(seconds=seconds * i)
            for i in range(limit - 1, -1, -1)
        ]
        
        # Generate price movement (random walk with trend)
        np.random.seed(hash(symbol) % (2**32))  # Consistent seed per symbol
        returns = np.random.normal(0.0002, 0.02, limit)  # Slight upward trend
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            volatility = 0.01
            open_price = close * (1 + np.random.normal(0, volatility))
            high = max(open_price, close) * (1 + abs(np.random.normal(0, volatility)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, volatility)))
            volume = np.random.lognormal(10, 2)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        
        logger.info(
            f"Generated {len(df)} synthetic {timeframe} candles for {symbol} "
            f"(${df['close'].iloc[0]:.2f} → ${df['close'].iloc[-1]:.2f})"
        )
        
        return df
    
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Place an order on Binance testnet/live.
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', etc.
            amount: Order size (in base currency)
            price: Limit price (required for limit orders)
        
        Returns:
            Order info dict
        
        Raises:
            Exception: If order placement fails
        """
        if not self._initialized:
            await self.initialize()
        
        # In paper mode, simulate order
        if self.environment == 'paper':
            return self._simulate_order(symbol, side, order_type, amount, price)
        
        try:
            params = {}
            
            if order_type == 'limit' and price is None:
                raise ValueError("Limit orders require a price")
            
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=amount,
                price=price,
                params=params
            )
            
            logger.info(
                f"✓ Order placed: {side.upper()} {amount} {symbol} @ "
                f"{price if price else 'MARKET'} (ID: {order.get('id')})"
            )
            
            return order
        
        except ccxt.InsufficientFunds as e:
            logger.error(f"✗ Insufficient funds for order: {e}")
            raise
        
        except ccxt.InvalidOrder as e:
            logger.error(f"✗ Invalid order: {e}")
            raise
        
        except ccxt.ExchangeError as e:
            logger.error(f"✗ Exchange error placing order: {e}")
            raise
        
        except Exception as e:
            logger.error(f"✗ Unexpected error placing order: {e}")
            raise
    
    def _simulate_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: Optional[float]
    ) -> Dict[str, Any]:
        """Simulate order execution in paper mode."""
        order_id = f"SIM-{int(datetime.now().timestamp() * 1000)}"
        
        simulated_order = {
            'id': order_id,
            'symbol': symbol,
            'type': order_type,
            'side': side,
            'amount': amount,
            'price': price,
            'cost': amount * price if price else None,
            'status': 'closed',
            'filled': amount,
            'timestamp': int(datetime.now().timestamp() * 1000),
            'datetime': datetime.now().isoformat(),
            'info': {'simulated': True}
        }
        
        logger.info(
            f"🔵 SIMULATED: {side.upper()} {amount} {symbol} @ "
            f"{price if price else 'MARKET'} (ID: {order_id})"
        )
        
        return simulated_order
    
    async def fetch_balance(self) -> Dict[str, Any]:
        """
        Fetch account balance.
        
        Returns:
            Balance information
        
        Raises:
            Exception: If fetching fails
        """
        if not self._initialized:
            await self.initialize()
        
        # In paper mode, return simulated balance
        if self.environment == 'paper':
            return {
                'USDT': {'free': 100000.0, 'used': 0.0, 'total': 100000.0},
                'BTC': {'free': 0.0, 'used': 0.0, 'total': 0.0}
            }
        
        try:
            balance = await self.exchange.fetch_balance()
            logger.debug(f"Fetched balance: {balance['total']}")
            return balance
        
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            raise
    
    async def fetch_multiple_tickers(
        self,
        symbols: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch tickers for multiple symbols concurrently.
        
        Args:
            symbols: List of trading pairs
        
        Returns:
            Dict mapping symbol to ticker data
        """
        tasks = [self.fetch_ticker(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        tickers = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch ticker for {symbol}: {result}")
            else:
                tickers[symbol] = result
        
        return tickers
    
    async def fetch_multiple_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        limit: int = 500
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple symbols concurrently.
        
        Args:
            symbols: List of trading pairs
            timeframe: Candlestick timeframe
            limit: Number of candles per symbol
        
        Returns:
            Dict mapping symbol to OHLCV DataFrame
        """
        tasks = [
            self.fetch_ohlcv(symbol, timeframe, limit)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        ohlcv_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch OHLCV for {symbol}: {result}")
            else:
                ohlcv_data[symbol] = result
        
        return ohlcv_data
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"BinanceClient(environment={self.environment}, "
            f"testnet={self.testnet}, initialized={self._initialized})"
        )


async def demo_binance_client():
    """Demo function to test Binance client."""
    # Initialize client
    client = BinanceClient(
        api_key='test_key',
        secret_key='test_secret',
        testnet=True,
        environment='paper'
    )
    
    try:
        await client.initialize()
        
        # Fetch single ticker
        ticker = await client.fetch_ticker('BTC/USDT')
        print(f"BTC/USDT: ${ticker['last']:.2f}")
        
        # Fetch multiple tickers
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        tickers = await client.fetch_multiple_tickers(symbols)
        
        print("\nCurrent prices:")
        for symbol, ticker in tickers.items():
            print(f"  {symbol}: ${ticker['last']:.2f}")
        
        # Fetch OHLCV
        df = await client.fetch_ohlcv('BTC/USDT', '1h', 100)
        print(f"\nFetched {len(df)} candles for BTC/USDT")
        print(df.tail())
        
    finally:
        await client.close()


if __name__ == '__main__':
    asyncio.run(demo_binance_client())
