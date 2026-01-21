"""
market_feed.py - Market Data Feed

Provides standardized market data feeds from exchange connectors.
Returns normalized pandas DataFrames with consistent schema and
handles missing data, timestamp normalization, and data validation.
Completely stateless with no side effects.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
import asyncio

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from data.exchange import ExchangeConnector


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketFeed:
    """
    Stateless market data feed processor.
    
    Fetches and normalizes market data from exchange connectors into
    standardized pandas DataFrames. Handles timestamp normalization,
    missing data detection, and data validation.
    """
    
    # Standard DataFrame schema
    STANDARD_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def __init__(self, exchange: ExchangeConnector):
        """
        Initialize market feed.
        
        Args:
            exchange: Connected ExchangeConnector instance
            
        Raises:
            ImportError: If pandas is not installed
            ValueError: If exchange is not connected
        """
        if pd is None or np is None:
            raise ImportError(
                "pandas and numpy required. Install with: "
                "pip install pandas numpy"
            )
        
        if not exchange.is_connected():
            raise ValueError("Exchange connector must be connected")
        
        self._exchange = exchange
        logger.info(f"MarketFeed initialized with exchange: {exchange.exchange_id}")
    
    async def fetch(
        self,
        symbol: str,
        timeframe: str = '1h',
        since: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch and normalize market data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '1h', '1d', etc.)
            since: Start datetime (UTC)
            limit: Maximum number of candles to fetch
            
        Returns:
            Standardized DataFrame with columns: timestamp, open, high, low, close, volume
            
        Raises:
            ValueError: If invalid parameters provided
            Exception: If data fetching fails
        """
        # Validate inputs
        self._validate_inputs(symbol, timeframe)
        
        # Convert since to milliseconds if provided
        since_ms = None
        if since:
            since_ms = int(since.timestamp() * 1000)
        else:
            # Calculate appropriate lookback period based on limit and timeframe
            # This ensures we fetch enough data for indicators (e.g., EMA200 needs 201+ candles)
            if limit is None:
                limit = 500  # Default to 500 candles if not specified
            
            # Convert timeframe to hours for calculation
            timeframe_hours = self._timeframe_to_hours(timeframe)
            lookback_hours = limit * timeframe_hours
            
            # Fetch data from (now - lookback_hours)
            since_ms = int((datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).timestamp() * 1000)
            logger.debug(f"Calculated lookback: {lookback_hours} hours ({limit} x {timeframe_hours}h candles)")
        
        # Fetch raw OHLCV data using fetch_ohlcv_range (with extended timeout for slow networks)
        logger.debug(
            f"Fetching {symbol} {timeframe} data "
            f"(since={since}, limit={limit})"
        )
        
        raw_ohlcv = await self._exchange.fetch_ohlcv_range(
            symbol=symbol,
            timeframe=timeframe,
            since_ms=since_ms,
            page_limit=limit,
            timeout=30.0  # Extended timeout for market data fetches (Binance can be slow)
        )
        
        if not raw_ohlcv:
            logger.warning(f"No data returned for {symbol} {timeframe}")
            return self._create_empty_dataframe()
        
        # Convert dict format to list format expected by _normalize_ohlcv
        if raw_ohlcv and isinstance(raw_ohlcv[0], dict):
            raw_ohlcv_list = [
                [d['timestamp'], d['open'], d['high'], d['low'], d['close'], d['volume']]
                for d in raw_ohlcv
            ]
        else:
            raw_ohlcv_list = raw_ohlcv
        
        # Normalize to DataFrame
        df = self._normalize_ohlcv(raw_ohlcv_list)
        
        # Validate and clean
        df = self._validate_and_clean(df, symbol, timeframe)
        
        logger.info(
            f"Fetched {len(df)} candles for {symbol} {timeframe} "
            f"({df['timestamp'].min()} to {df['timestamp'].max()})"
        )
        
        return df
    
    def _timeframe_to_hours(self, timeframe: str) -> float:
        """
        Convert timeframe string to hours.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '4h', '1d')
            
        Returns:
            Number of hours per candle
        """
        # Extract number and unit from timeframe
        import re
        match = re.match(r'^(\d+)([mhdwM])$', timeframe)
        if not match:
            logger.warning(f"Unknown timeframe format: {timeframe}, defaulting to 1 hour")
            return 1.0
        
        value = int(match.group(1))
        unit = match.group(2)
        
        # Convert to hours
        conversions = {
            'm': value / 60.0,      # minutes to hours
            'h': float(value),       # hours
            'd': value * 24.0,       # days to hours
            'w': value * 24.0 * 7,   # weeks to hours
            'M': value * 24.0 * 30,  # months to hours (approximate)
        }
        
        return conversions.get(unit, 1.0)
    
    async def fetch_range(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        limit_per_request: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch market data for a date range.
        
        Automatically handles pagination and combines results into
        a single normalized DataFrame.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start_time: Start datetime (UTC)
            end_time: End datetime (UTC, defaults to now)
            limit_per_request: Max candles per request
            
        Returns:
            Standardized DataFrame covering the entire range
        """
        self._validate_inputs(symbol, timeframe)
        
        if not end_time:
            end_time = datetime.now(timezone.utc)
        
        logger.info(
            f"Fetching range for {symbol} {timeframe}: "
            f"{start_time} to {end_time}"
        )
        
        # Fetch raw data using exchange's range method
        raw_ohlcv = await self._exchange.fetch_ohlcv_range(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit_per_request=limit_per_request
        )
        
        if not raw_ohlcv:
            logger.warning(f"No data in range for {symbol} {timeframe}")
            return self._create_empty_dataframe()
        
        # Normalize to DataFrame
        df = self._normalize_ohlcv(raw_ohlcv)
        
        # Validate and clean
        df = self._validate_and_clean(df, symbol, timeframe)
        
        # Check for gaps
        gaps = self._detect_gaps(df, timeframe)
        if gaps:
            logger.warning(f"Detected {len(gaps)} gaps in data for {symbol} {timeframe}")
        
        return df
    
    def _validate_inputs(self, symbol: str, timeframe: str) -> None:
        """
        Validate input parameters.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError(f"Invalid symbol: {symbol}")
        
        if not timeframe or not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Note: Timeframe validation skipped - ExchangeConnector doesn't expose supported timeframes
        # If timeframe is invalid, CCXT will raise an error when we fetch data
    
    def _normalize_ohlcv(self, raw_ohlcv: List[List]) -> pd.DataFrame:
        """
        Convert raw OHLCV data to normalized DataFrame.
        
        Args:
            raw_ohlcv: Raw OHLCV data from exchange
                      Format: [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            Normalized DataFrame with standard columns
        """
        if not raw_ohlcv:
            return self._create_empty_dataframe()
        
        # Create DataFrame from raw data
        df = pd.DataFrame(
            raw_ohlcv,
            columns=self.STANDARD_COLUMNS
        )
        
        # Normalize timestamp to datetime (UTC)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Ensure numeric types for price/volume columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def _validate_and_clean(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> pd.DataFrame:
        """
        Validate and clean DataFrame.
        
        Checks for:
        - Missing values
        - Invalid price data (negative, zero, null)
        - Price consistency (high >= low, etc.)
        - Duplicate timestamps
        
        Args:
            df: DataFrame to validate
            symbol: Trading pair symbol (for logging)
            timeframe: Timeframe (for logging)
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        original_len = len(df)
        
        # Remove duplicate timestamps (keep last)
        duplicates = df.duplicated(subset=['timestamp'], keep='last')
        if duplicates.any():
            logger.warning(
                f"Removing {duplicates.sum()} duplicate timestamps for {symbol}"
            )
            df = df[~duplicates]
        
        # Check for missing values
        missing = df[self.STANDARD_COLUMNS].isnull().sum()
        if missing.any():
            logger.warning(
                f"Missing values detected in {symbol} {timeframe}: "
                f"{missing[missing > 0].to_dict()}"
            )
            # Drop rows with any missing values
            df = df.dropna()
        
        # Validate price data
        price_cols = ['open', 'high', 'low', 'close']
        
        # Remove rows with non-positive prices
        for col in price_cols:
            invalid = df[col] <= 0
            if invalid.any():
                logger.warning(
                    f"Removing {invalid.sum()} rows with invalid {col} "
                    f"prices for {symbol}"
                )
                df = df[~invalid]
        
        # Validate OHLC consistency (high >= low, etc.)
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(
                f"Removing {invalid_ohlc.sum()} rows with inconsistent "
                f"OHLC data for {symbol}"
            )
            df = df[~invalid_ohlc]
        
        # Validate volume
        invalid_volume = df['volume'] < 0
        if invalid_volume.any():
            logger.warning(
                f"Removing {invalid_volume.sum()} rows with negative "
                f"volume for {symbol}"
            )
            df = df[~invalid_volume]
        
        # Reset index after cleaning
        df = df.reset_index(drop=True)
        
        removed = original_len - len(df)
        if removed > 0:
            logger.info(f"Cleaned {removed} invalid rows from {symbol} {timeframe}")
        
        return df
    
    def _detect_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Dict[str, Any]]:
        """
        Detect gaps in time series data.
        
        Args:
            df: DataFrame to check
            timeframe: Expected timeframe interval
            
        Returns:
            List of detected gaps with start/end timestamps
        """
        if df.empty or len(df) < 2:
            return []
        
        # Convert timeframe to timedelta
        timeframe_delta = self._timeframe_to_timedelta(timeframe)
        if not timeframe_delta:
            return []
        
        # Calculate time differences
        time_diffs = df['timestamp'].diff()
        
        # Detect gaps (differences > expected interval)
        tolerance = timeframe_delta * 1.5  # Allow 50% tolerance
        gaps = time_diffs > tolerance
        
        gap_list = []
        for idx in df[gaps].index:
            gap_info = {
                'index': idx,
                'start': df.loc[idx - 1, 'timestamp'],
                'end': df.loc[idx, 'timestamp'],
                'expected_interval': timeframe_delta,
                'actual_interval': time_diffs.loc[idx]
            }
            gap_list.append(gap_info)
        
        return gap_list
    
    def _timeframe_to_timedelta(self, timeframe: str) -> Optional[pd.Timedelta]:
        """
        Convert timeframe string to pandas Timedelta.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            pandas Timedelta or None if conversion fails
        """
        try:
            # Common timeframe mappings
            mappings = {
                'm': 'min',
                'h': 'h',
                'd': 'D',
                'w': 'W',
                'M': 'M'
            }
            
            # Extract number and unit
            import re
            match = re.match(r'(\d+)([mhdwM])', timeframe)
            if not match:
                return None
            
            amount, unit = match.groups()
            pandas_unit = mappings.get(unit)
            
            if not pandas_unit:
                return None
            
            return pd.Timedelta(f"{amount}{pandas_unit}")
            
        except Exception as e:
            logger.debug(f"Failed to convert timeframe '{timeframe}': {e}")
            return None
    
    def _create_empty_dataframe(self) -> pd.DataFrame:
        """
        Create empty DataFrame with standard schema.
        
        Returns:
            Empty DataFrame with correct columns and types
        """
        df = pd.DataFrame(columns=self.STANDARD_COLUMNS)
        
        # Set proper dtypes
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df[['open', 'high', 'low', 'close', 'volume']] = df[
            ['open', 'high', 'low', 'close', 'volume']
        ].astype(float)
        
        return df
    
    @staticmethod
    def resample(
        df: pd.DataFrame,
        target_timeframe: str,
        agg_func: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Resample data to different timeframe.
        
        Args:
            df: Input DataFrame
            target_timeframe: Target timeframe (e.g., '5m', '1h', '1d')
            agg_func: Custom aggregation functions (defaults to OHLC standard)
            
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Default aggregation functions for OHLC data
        if agg_func is None:
            agg_func = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
        
        # Set timestamp as index for resampling
        df_indexed = df.set_index('timestamp')
        
        # Resample
        resampled = df_indexed.resample(target_timeframe).agg(agg_func)
        
        # Reset index
        resampled = resampled.reset_index()
        
        # Remove rows with NaN (from incomplete periods)
        resampled = resampled.dropna()
        
        return resampled
    
    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic calculated indicators to DataFrame.
        
        Adds:
        - typical_price: (high + low + close) / 3
        - price_change: close - previous close
        - price_change_pct: percentage change in close
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with added indicator columns
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Price changes
        df['price_change'] = df['close'].diff()
        df['price_change_pct'] = df['close'].pct_change() * 100
        
        return df


# Convenience functions

async def fetch_market_data(
    exchange: ExchangeConnector,
    symbol: str,
    timeframe: str = '1h',
    since: Optional[datetime] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Convenience function to fetch market data.
    
    Args:
        exchange: Connected ExchangeConnector instance
        symbol: Trading pair symbol
        timeframe: Candle timeframe
        since: Start datetime (UTC)
        limit: Maximum number of candles
        
    Returns:
        Standardized DataFrame
    """
    feed = MarketFeed(exchange)
    return await feed.fetch(symbol, timeframe, since, limit)


# Example usage
if __name__ == "__main__":
    from environment import EnvironmentConfig, Environment
    from exchange import ExchangeConnector
    from datetime import timedelta
    
    async def main():
        print("=== Market Feed Example ===\n")
        
        # Create demo config (public endpoints)
        demo_config = EnvironmentConfig(
            mode=Environment.PAPER,
            api_key=None,
            api_secret=None
        )
        
        try:
            # Create and connect exchange
            print("Connecting to Binance...")
            exchange = ExchangeConnector('binance', demo_config)
            await exchange.connect()
            
            if not exchange.is_connected():
                print("Failed to connect")
                return
            
            # Create market feed
            feed = MarketFeed(exchange)
            
            # Fetch recent data
            print("\n1. Fetching recent BTC/USDT 1h data (limit 10)...")
            df = await feed.fetch('BTC/USDT', '1h', limit=10)
            
            print(f"\nDataFrame shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            print(f"\nData info:")
            print(df.info())
            
            print(f"\nBasic statistics:")
            print(df[['open', 'high', 'low', 'close', 'volume']].describe())
            
            # Fetch date range
            print("\n2. Fetching 24h range of ETH/USDT 15m data...")
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)
            
            df_range = await feed.fetch_range(
                'ETH/USDT',
                '15m',
                start_time,
                end_time
            )
            
            print(f"\nFetched {len(df_range)} candles")
            print(f"Time range: {df_range['timestamp'].min()} to {df_range['timestamp'].max()}")
            print(f"\nLast 5 rows:")
            print(df_range.tail())
            
            # Resample example
            print("\n3. Resampling 15m data to 1h...")
            df_resampled = MarketFeed.resample(df_range, '1h')
            print(f"Resampled to {len(df_resampled)} candles")
            print(df_resampled.head())
            
            # Add indicators
            print("\n4. Adding basic indicators...")
            df_with_indicators = MarketFeed.add_indicators(df)
            print(f"\nColumns after adding indicators:")
            print(list(df_with_indicators.columns))
            print(df_with_indicators[['timestamp', 'close', 'typical_price', 'price_change', 'price_change_pct']].tail())
            
            # Test empty result handling
            print("\n5. Testing with very old date (should return empty)...")
            old_date = datetime(2010, 1, 1, tzinfo=timezone.utc)
            df_empty = await feed.fetch('BTC/USDT', '1d', since=old_date, limit=1)
            print(f"Empty DataFrame shape: {df_empty.shape}")
            print(f"Is empty: {df_empty.empty}")
            
            # Disconnect
            await exchange.disconnect()
            print("\n✓ Example completed successfully")
            
        except ImportError as e:
            print(f"Error: {e}")
            print("Please install required packages:")
            print("  pip install ccxt pandas numpy")
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Exception in example")
    
    # Run async example
    asyncio.run(main())