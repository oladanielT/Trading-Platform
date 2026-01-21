"""
candles.py - OHLCV Data Transformation and Indicators

Provides comprehensive utilities for transforming raw OHLCV data into
clean DataFrames with optional technical indicators. Pure data processing
with no trading decision logic.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Union, Callable
from datetime import datetime

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CandleProcessor:
    """
    Stateless processor for OHLCV candle data.
    
    Provides transformation, cleaning, and technical indicator calculations
    without any trading logic or decision-making.
    """
    
    @staticmethod
    def from_list(
        raw_data: List[List],
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Convert raw OHLCV list data to DataFrame.
        
        Args:
            raw_data: List of lists containing OHLCV data
                     [[timestamp, open, high, low, close, volume], ...]
            columns: Column names (defaults to standard OHLCV)
            
        Returns:
            DataFrame with standardized columns
        """
        if pd is None:
            raise ImportError("pandas required. Install with: pip install pandas")
        
        if not raw_data:
            return pd.DataFrame(columns=columns or ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        if columns is None:
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        df = pd.DataFrame(raw_data, columns=columns)
        
        # Convert timestamp to datetime if it's numeric
        if pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    @staticmethod
    def clean(
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        remove_nulls: bool = True,
        validate_ohlc: bool = True,
        sort_by_time: bool = True
    ) -> pd.DataFrame:
        """
        Clean OHLCV DataFrame.
        
        Args:
            df: Input DataFrame
            remove_duplicates: Remove duplicate timestamps
            remove_nulls: Remove rows with null values
            validate_ohlc: Validate OHLC price relationships
            sort_by_time: Sort by timestamp ascending
            
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            return df
        
        df = df.copy()
        original_len = len(df)
        
        # Sort by timestamp
        if sort_by_time and 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Remove duplicates
        if remove_duplicates and 'timestamp' in df.columns:
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Remove nulls
        if remove_nulls:
            df = df.dropna()
        
        # Validate OHLC relationships
        if validate_ohlc:
            price_cols = ['open', 'high', 'low', 'close']
            if all(col in df.columns for col in price_cols):
                # high should be >= all others
                # low should be <= all others
                valid_mask = (
                    (df['high'] >= df['low']) &
                    (df['high'] >= df['open']) &
                    (df['high'] >= df['close']) &
                    (df['low'] <= df['open']) &
                    (df['low'] <= df['close'])
                )
                df = df[valid_mask]
        
        removed = original_len - len(df)
        if removed > 0:
            logger.debug(f"Cleaned {removed} rows from DataFrame")
        
        return df.reset_index(drop=True)
    
    @staticmethod
    def fill_gaps(
        df: pd.DataFrame,
        method: str = 'ffill',
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fill missing values in DataFrame.
        
        Args:
            df: Input DataFrame
            method: Fill method ('ffill', 'bfill', 'interpolate')
            limit: Maximum number of consecutive NaNs to fill
            
        Returns:
            DataFrame with filled values
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        if method == 'ffill':
            df = df.fillna(method='ffill', limit=limit)
        elif method == 'bfill':
            df = df.fillna(method='bfill', limit=limit)
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=limit)
        else:
            raise ValueError(f"Unknown fill method: {method}")
        
        return df


# Technical Indicators

class Indicators:
    """
    Collection of technical indicator calculations.
    
    All methods are static and stateless - pure mathematical transformations
    of price/volume data with no trading logic.
    """
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average.
        
        Args:
            series: Price series
            period: Number of periods
            
        Returns:
            SMA series
        """
        return series.rolling(window=period, min_periods=1).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average.
        
        Args:
            series: Price series
            period: Number of periods
            
        Returns:
            EMA series
        """
        return series.ewm(span=period, adjust=False, min_periods=1).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index.
        
        Args:
            series: Price series
            period: RSI period (default 14)
            
        Returns:
            RSI series (0-100)
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        series: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence.
        
        Args:
            series: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            DataFrame with macd, signal, and histogram columns
        """
        ema_fast = Indicators.ema(series, fast)
        ema_slow = Indicators.ema(series, slow)
        
        macd_line = ema_fast - ema_slow
        signal_line = Indicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> pd.DataFrame:
        """
        Bollinger Bands.
        
        Args:
            series: Price series
            period: MA period
            std_dev: Number of standard deviations
            
        Returns:
            DataFrame with upper, middle, lower bands
        """
        middle = Indicators.sma(series, period)
        std = series.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower
        })
    
    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range.
        
        Args:
            df: DataFrame with high, low, close columns
            period: ATR period
            
        Returns:
            ATR series
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        period: int = 14,
        smooth_k: int = 3,
        smooth_d: int = 3
    ) -> pd.DataFrame:
        """
        Stochastic Oscillator.
        
        Args:
            df: DataFrame with high, low, close columns
            period: Look-back period
            smooth_k: %K smoothing period
            smooth_d: %D smoothing period
            
        Returns:
            DataFrame with %K and %D columns
        """
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        k = 100 * (df['close'] - low_min) / (high_max - low_min)
        k_smooth = k.rolling(window=smooth_k).mean()
        d = k_smooth.rolling(window=smooth_d).mean()
        
        return pd.DataFrame({
            'stoch_k': k_smooth,
            'stoch_d': d
        })
    
    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price.
        
        Args:
            df: DataFrame with high, low, close, volume columns
            
        Returns:
            VWAP series
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    @staticmethod
    def obv(df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume.
        
        Args:
            df: DataFrame with close and volume columns
            
        Returns:
            OBV series
        """
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Average Directional Index.
        
        Args:
            df: DataFrame with high, low, close columns
            period: ADX period
            
        Returns:
            DataFrame with adx, +di, -di columns
        """
        # Calculate +DM and -DM
        high_diff = df['high'].diff()
        low_diff = df['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        # Calculate ATR
        atr = Indicators.atr(df, period)
        
        # Calculate directional indicators
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })


# Convenience functions for adding indicators to DataFrame

def add_moving_averages(
    df: pd.DataFrame,
    periods: List[int] = [20, 50, 200],
    ma_type: str = 'sma',
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Add multiple moving averages to DataFrame.
    
    Args:
        df: Input DataFrame
        periods: List of MA periods
        ma_type: Type of MA ('sma' or 'ema')
        price_col: Column to calculate MA on
        
    Returns:
        DataFrame with added MA columns
    """
    df = df.copy()
    
    for period in periods:
        col_name = f"{ma_type}{period}"
        if ma_type == 'sma':
            df[col_name] = Indicators.sma(df[price_col], period)
        elif ma_type == 'ema':
            df[col_name] = Indicators.ema(df[price_col], period)
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")
    
    return df


def add_rsi(df: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.DataFrame:
    """
    Add RSI indicator to DataFrame.
    
    Args:
        df: Input DataFrame
        period: RSI period
        price_col: Column to calculate RSI on
        
    Returns:
        DataFrame with rsi column
    """
    df = df.copy()
    df['rsi'] = Indicators.rsi(df[price_col], period)
    return df


def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Add MACD indicator to DataFrame.
    
    Args:
        df: Input DataFrame
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        price_col: Column to calculate MACD on
        
    Returns:
        DataFrame with macd, macd_signal, macd_histogram columns
    """
    df = df.copy()
    macd_df = Indicators.macd(df[price_col], fast, slow, signal)
    df['macd'] = macd_df['macd']
    df['macd_signal'] = macd_df['signal']
    df['macd_histogram'] = macd_df['histogram']
    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    Add Bollinger Bands to DataFrame.
    
    Args:
        df: Input DataFrame
        period: MA period
        std_dev: Number of standard deviations
        price_col: Column to calculate BB on
        
    Returns:
        DataFrame with bb_upper, bb_middle, bb_lower columns
    """
    df = df.copy()
    bb_df = Indicators.bollinger_bands(df[price_col], period, std_dev)
    df = pd.concat([df, bb_df], axis=1)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Add ATR indicator to DataFrame.
    
    Args:
        df: Input DataFrame with high, low, close columns
        period: ATR period
        
    Returns:
        DataFrame with atr column
    """
    df = df.copy()
    df['atr'] = Indicators.atr(df, period)
    return df


def add_stochastic(
    df: pd.DataFrame,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> pd.DataFrame:
    """
    Add Stochastic Oscillator to DataFrame.
    
    Args:
        df: Input DataFrame with high, low, close columns
        period: Look-back period
        smooth_k: %K smoothing
        smooth_d: %D smoothing
        
    Returns:
        DataFrame with stoch_k and stoch_d columns
    """
    df = df.copy()
    stoch_df = Indicators.stochastic(df, period, smooth_k, smooth_d)
    df = pd.concat([df, stoch_df], axis=1)
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to DataFrame.
    
    Adds: SMA(20, 50), EMA(12, 26), RSI(14), MACD, Bollinger Bands, ATR, Stochastic
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with all indicators added
    """
    df = df.copy()
    
    # Moving averages
    df = add_moving_averages(df, [20, 50], 'sma')
    df = add_moving_averages(df, [12, 26], 'ema')
    
    # Oscillators
    df = add_rsi(df)
    df = add_macd(df)
    df = add_stochastic(df)
    
    # Volatility
    df = add_bollinger_bands(df)
    df = add_atr(df)
    
    # Volume
    df['vwap'] = Indicators.vwap(df)
    df['obv'] = Indicators.obv(df)
    
    return df


# Example usage
if __name__ == "__main__":
    print("=== Candle Processor and Indicators Example ===\n")
    
    # Sample raw OHLCV data
    raw_data = [
        [1640000000000, 47000, 47500, 46800, 47200, 1000],
        [1640003600000, 47200, 47800, 47100, 47600, 1200],
        [1640007200000, 47600, 48000, 47400, 47800, 1100],
        [1640010800000, 47800, 48200, 47500, 47900, 900],
        [1640014400000, 47900, 48500, 47800, 48300, 1300],
        [1640018000000, 48300, 48700, 48000, 48400, 1150],
        [1640021600000, 48400, 48900, 48200, 48600, 1250],
        [1640025200000, 48600, 49000, 48400, 48800, 1400],
        [1640028800000, 48800, 49200, 48600, 49000, 1350],
        [1640032400000, 49000, 49500, 48900, 49200, 1450],
    ]
    
    try:
        # Convert to DataFrame
        print("1. Converting raw data to DataFrame...")
        df = CandleProcessor.from_list(raw_data)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}\n")
        print(df.head())
        
        # Clean data
        print("\n2. Cleaning data...")
        df_clean = CandleProcessor.clean(df)
        print(f"Cleaned shape: {df_clean.shape}")
        
        # Add moving averages
        print("\n3. Adding moving averages (SMA 5, 10)...")
        df = add_moving_averages(df, [5, 10], 'sma')
        print(df[['timestamp', 'close', 'sma5', 'sma10']].tail())
        
        # Add RSI
        print("\n4. Adding RSI...")
        df = add_rsi(df, period=5)  # Using shorter period for demo
        print(df[['timestamp', 'close', 'rsi']].tail())
        
        # Add MACD (using shorter periods for demo)
        print("\n5. Adding MACD...")
        df = add_macd(df, fast=5, slow=8, signal=3)
        print(df[['timestamp', 'close', 'macd', 'macd_signal', 'macd_histogram']].tail())
        
        # Add Bollinger Bands
        print("\n6. Adding Bollinger Bands...")
        df = add_bollinger_bands(df, period=5, std_dev=2.0)
        print(df[['timestamp', 'close', 'bb_upper', 'bb_middle', 'bb_lower']].tail())
        
        # Add ATR
        print("\n7. Adding ATR...")
        df = add_atr(df, period=5)
        print(df[['timestamp', 'close', 'atr']].tail())
        
        # Add Stochastic
        print("\n8. Adding Stochastic Oscillator...")
        df = add_stochastic(df, period=5, smooth_k=3, smooth_d=3)
        print(df[['timestamp', 'close', 'stoch_k', 'stoch_d']].tail())
        
        # Add volume indicators
        print("\n9. Adding volume indicators...")
        df['vwap'] = Indicators.vwap(df)
        df['obv'] = Indicators.obv(df)
        print(df[['timestamp', 'close', 'volume', 'vwap', 'obv']].tail())
        
        # Show all columns
        print("\n10. Final DataFrame columns:")
        print(list(df.columns))
        
        print("\n11. Sample of all data:")
        print(df.tail(3))
        
        print("\n✓ All transformations completed successfully")
        print("\nNote: This module provides only data transformation.")
        print("No trading decisions are made by these functions.")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages:")
        print("  pip install pandas numpy")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Exception in example")