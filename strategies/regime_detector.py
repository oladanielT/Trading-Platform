"""
regime_detector.py - Market Regime Classification

Stateless and deterministic regime detection module that classifies market
conditions based on price behavior, volatility, and trend characteristics.
Used by meta-strategies to adapt trading behavior to market conditions.

Market Regimes:
- TRENDING: Clear directional movement with strong trend
- RANGING: Price oscillating within a range with low trend strength
- VOLATILE: High volatility with unpredictable movements
- QUIET: Low volatility with minimal price movement

No machine learning, no future data leakage, fully explainable.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple
from enum import Enum

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


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"
    
    def __str__(self):
        return self.value


class RegimeDetector:
    """
    Stateless market regime detector.
    
    Analyzes price data to classify market conditions using multiple metrics:
    - Realized volatility (standard deviation of returns)
    - Trend strength (price slope normalized by volatility)
    - Price efficiency (net movement / total movement)
    - Price range expansion
    
    All computations are deterministic and use only historical data
    (no look-ahead bias).
    """
    
    def __init__(
        self,
        lookback_period: int = 100,
        volatility_threshold_high: float = 0.03,
        volatility_threshold_low: float = 0.01,
        trend_threshold_strong: float = 2.0,
        trend_threshold_weak: float = 0.5,
        efficiency_threshold: float = 0.5
    ):
        """
        Initialize regime detector.
        
        Args:
            lookback_period: Number of candles to analyze (default: 100)
            volatility_threshold_high: High volatility threshold (default: 3% daily)
            volatility_threshold_low: Low volatility threshold (default: 1% daily)
            trend_threshold_strong: Strong trend threshold (default: 2.0)
            trend_threshold_weak: Weak trend threshold (default: 0.5)
            efficiency_threshold: Efficiency ratio threshold (default: 0.5)
            
        Raises:
            ImportError: If pandas or numpy not installed
        """
        if pd is None or np is None:
            raise ImportError(
                "pandas and numpy required. Install with: pip install pandas numpy"
            )
        
        self.lookback_period = lookback_period
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.trend_threshold_strong = trend_threshold_strong
        self.trend_threshold_weak = trend_threshold_weak
        self.efficiency_threshold = efficiency_threshold
        
        logger.info(
            f"RegimeDetector initialized with lookback={lookback_period}, "
            f"vol_thresholds=({volatility_threshold_low:.3f}, {volatility_threshold_high:.3f})"
        )
    
    def detect(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect current market regime based on price data.
        
        Args:
            market_data: DataFrame with OHLCV data
                        Must have columns: timestamp, open, high, low, close, volume
                        
        Returns:
            Dictionary with:
                - regime: MarketRegime enum
                - metrics: Dict of computed metrics
                - explanation: Human-readable explanation
                - confidence: Regime classification confidence (0.0-1.0)
                
        Raises:
            ValueError: If market data is invalid or insufficient
        """
        # Validate input
        self._validate_market_data(market_data)
        
        # Extract recent data for analysis
        analysis_window = market_data.tail(self.lookback_period).copy()
        
        if len(analysis_window) < 20:
            raise ValueError(
                f"Insufficient data for regime detection. "
                f"Need at least 20 candles, got {len(analysis_window)}"
            )
        
        # Compute metrics
        metrics = self._compute_metrics(analysis_window)
        
        # Classify regime based on metrics
        regime, confidence, explanation = self._classify_regime(metrics)
        
        logger.debug(
            f"Regime detected: {regime.value} "
            f"(vol={metrics['volatility']:.4f}, "
            f"trend={metrics['trend_strength']:.2f}, "
            f"eff={metrics['price_efficiency']:.2f})"
        )
        
        return {
            'regime': regime,
            'metrics': metrics,
            'explanation': explanation,
            'confidence': confidence
        }
    
    def _validate_market_data(self, market_data: pd.DataFrame) -> None:
        """
        Validate market data has required structure.
        
        Args:
            market_data: DataFrame to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(market_data, pd.DataFrame):
            raise ValueError(f"market_data must be pandas DataFrame, got {type(market_data)}")
        
        if market_data.empty:
            raise ValueError("market_data is empty")
        
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in market_data.columns]
        
        if missing_columns:
            raise ValueError(f"market_data missing required columns: {missing_columns}")
    
    def _compute_metrics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Compute regime classification metrics.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary of computed metrics
        """
        close_prices = data['close'].values
        high_prices = data['high'].values
        low_prices = data['low'].values
        
        # 1. Log Returns
        log_returns = np.log(close_prices[1:] / close_prices[:-1])
        
        # 2. Realized Volatility (annualized, assuming daily data)
        # For hourly data, multiply by sqrt(24*365) ≈ 93
        # For daily data, multiply by sqrt(365) ≈ 19
        volatility = np.std(log_returns) * np.sqrt(252)  # Assuming daily-like periods
        
        # 3. Trend Strength (slope normalized by volatility)
        # Linear regression slope of close prices
        x = np.arange(len(close_prices))
        slope = self._compute_slope(x, close_prices)
        
        # Normalize by price level and volatility
        avg_price = np.mean(close_prices)
        price_slope_pct = (slope / avg_price) * 100 if avg_price > 0 else 0.0
        
        # Trend strength: slope relative to volatility
        trend_strength = abs(price_slope_pct) / (volatility + 1e-8)
        
        # 4. Price Efficiency (net movement / total movement)
        # Net movement: abs(close[-1] - close[0])
        # Total movement: sum of abs(close[i] - close[i-1])
        net_movement = abs(close_prices[-1] - close_prices[0])
        total_movement = np.sum(np.abs(np.diff(close_prices)))
        price_efficiency = net_movement / (total_movement + 1e-8)
        
        # 5. ATR (Average True Range) for volatility context
        atr = self._compute_atr(high_prices, low_prices, close_prices)
        atr_pct = (atr / avg_price) * 100 if avg_price > 0 else 0.0
        
        # 6. Recent vs Historical Volatility Ratio
        recent_vol = np.std(log_returns[-20:]) if len(log_returns) >= 20 else volatility
        vol_ratio = recent_vol / (volatility + 1e-8)
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'price_efficiency': price_efficiency,
            'price_slope_pct': price_slope_pct,
            'atr_pct': atr_pct,
            'vol_ratio': vol_ratio,
            'mean_return': np.mean(log_returns),
            'lookback_candles': len(data)
        }
    
    def _compute_slope(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute linear regression slope.
        
        Args:
            x: X values (usually time indices)
            y: Y values (usually prices)
            
        Returns:
            Slope of best-fit line
        """
        n = len(x)
        if n < 2:
            return 0.0
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _compute_atr(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> float:
        """
        Compute Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: 14)
            
        Returns:
            Average True Range
        """
        if len(high) < 2:
            return 0.0
        
        # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Average True Range
        if len(true_range) >= period:
            atr = np.mean(true_range[-period:])
        else:
            atr = np.mean(true_range)
        
        return atr
    
    def _classify_regime(
        self,
        metrics: Dict[str, float]
    ) -> Tuple[MarketRegime, float, str]:
        """
        Classify market regime based on computed metrics.
        
        Classification logic:
        1. VOLATILE: High volatility (vol > high_threshold) OR high vol_ratio
        2. QUIET: Low volatility (vol < low_threshold) AND low trend strength
        3. TRENDING: Strong trend (trend_strength > strong_threshold) AND good efficiency
        4. RANGING: Everything else (weak trend, moderate volatility)
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Tuple of (regime, confidence, explanation)
        """
        vol = metrics['volatility']
        trend_strength = metrics['trend_strength']
        efficiency = metrics['price_efficiency']
        vol_ratio = metrics['vol_ratio']
        
        # Initialize confidence scores for each regime
        scores = {
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.QUIET: 0.0,
            MarketRegime.TRENDING: 0.0,
            MarketRegime.RANGING: 0.0
        }
        
        # Score VOLATILE regime
        if vol > self.volatility_threshold_high:
            scores[MarketRegime.VOLATILE] += (vol - self.volatility_threshold_high) * 10
        if vol_ratio > 1.5:
            scores[MarketRegime.VOLATILE] += (vol_ratio - 1.0) * 0.3
        
        # Score QUIET regime
        if vol < self.volatility_threshold_low:
            scores[MarketRegime.QUIET] += (self.volatility_threshold_low - vol) * 10
        if trend_strength < self.trend_threshold_weak:
            scores[MarketRegime.QUIET] += (self.trend_threshold_weak - trend_strength) * 0.3
        
        # Score TRENDING regime
        if trend_strength > self.trend_threshold_strong:
            scores[MarketRegime.TRENDING] += (trend_strength - self.trend_threshold_strong) * 0.5
        if efficiency > self.efficiency_threshold:
            scores[MarketRegime.TRENDING] += (efficiency - self.efficiency_threshold) * 1.0
        
        # Score RANGING regime (default/baseline)
        # Ranging is moderate: not too volatile, not too quiet, not strongly trending
        if (self.volatility_threshold_low <= vol <= self.volatility_threshold_high and
            self.trend_threshold_weak <= trend_strength <= self.trend_threshold_strong):
            scores[MarketRegime.RANGING] += 0.5
        
        # Ensure ranging has a baseline score (it's the "catch-all" regime)
        scores[MarketRegime.RANGING] += 0.2
        
        # Select regime with highest score
        regime = max(scores, key=scores.get)
        max_score = scores[regime]
        total_score = sum(scores.values())
        
        # Confidence is the proportion of the winning score
        confidence = min(max_score / (total_score + 1e-8), 1.0)
        
        # Generate explanation
        explanation = self._generate_explanation(regime, metrics)
        
        return regime, confidence, explanation
    
    def _generate_explanation(
        self,
        regime: MarketRegime,
        metrics: Dict[str, float]
    ) -> str:
        """
        Generate human-readable explanation for regime classification.
        
        Args:
            regime: Classified regime
            metrics: Computed metrics
            
        Returns:
            Explanation string
        """
        vol = metrics['volatility']
        trend = metrics['trend_strength']
        eff = metrics['price_efficiency']
        
        explanations = {
            MarketRegime.TRENDING: (
                f"Strong directional trend detected with trend_strength={trend:.2f} "
                f"and price_efficiency={eff:.2f}. Market showing clear direction."
            ),
            MarketRegime.RANGING: (
                f"Range-bound market with moderate volatility={vol:.3f} "
                f"and weak trend_strength={trend:.2f}. Price oscillating without clear direction."
            ),
            MarketRegime.VOLATILE: (
                f"High volatility regime detected with vol={vol:.3f} "
                f"(threshold={self.volatility_threshold_high:.3f}). "
                f"Market showing unpredictable movements."
            ),
            MarketRegime.QUIET: (
                f"Low volatility regime with vol={vol:.3f} "
                f"(threshold={self.volatility_threshold_low:.3f}) "
                f"and weak trend={trend:.2f}. Market relatively inactive."
            )
        }
        
        return explanations.get(regime, f"Market classified as {regime.value}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get detector parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'lookback_period': self.lookback_period,
            'volatility_threshold_high': self.volatility_threshold_high,
            'volatility_threshold_low': self.volatility_threshold_low,
            'trend_threshold_strong': self.trend_threshold_strong,
            'trend_threshold_weak': self.trend_threshold_weak,
            'efficiency_threshold': self.efficiency_threshold
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"RegimeDetector(lookback={self.lookback_period}, "
                f"vol_range=[{self.volatility_threshold_low:.3f}, "
                f"{self.volatility_threshold_high:.3f}])")
