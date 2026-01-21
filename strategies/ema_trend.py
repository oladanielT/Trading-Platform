"""
ema_trend.py - EMA Trend Following Strategy

Combines multiple exponential moving averages to determine trend direction
and generate buy/sell signals based on crossovers and momentum filters.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from strategies.base import BaseStrategy, SignalOutput, PositionState, Signal
from data.candles import Indicators


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EMATrendStrategy(BaseStrategy):
    """
    EMA crossover trend-following strategy.
    
    Generates trading signals based on the crossover of two exponential
    moving averages (EMAs). A bullish crossover (fast EMA crossing above
    slow EMA) generates a BUY signal, while a bearish crossover generates
    a SELL signal.
    
    Default Parameters:
        fast_period: 50 (EMA50)
        slow_period: 200 (EMA200)
        signal_confidence: 0.7
        hold_confidence: 0.0
        require_trend_confirmation: False (optional additional filter)
    """
    
    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        signal_confidence: float = 0.7,
        hold_confidence: float = 0.0,
        require_trend_confirmation: bool = False,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize EMA trend strategy.
        
        Args:
            fast_period: Fast EMA period (default: 50)
            slow_period: Slow EMA period (default: 200)
            signal_confidence: Confidence for BUY/SELL signals (default: 0.7)
            hold_confidence: Confidence for HOLD signals (default: 0.0)
            require_trend_confirmation: Require price above/below both EMAs (default: False)
            name: Strategy name (default: "EMATrendStrategy")
            **kwargs: Additional parameters
        """
        if pd is None or np is None:
            raise ImportError(
                "pandas and numpy required. Install with: pip install pandas numpy"
            )
        
        super().__init__(
            name=name or "EMATrendStrategy",
            fast_period=fast_period,
            slow_period=slow_period,
            signal_confidence=signal_confidence,
            hold_confidence=hold_confidence,
            require_trend_confirmation=require_trend_confirmation,
            **kwargs
        )
        
        # Validate parameters
        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})"
            )
        
        if not 0.0 <= signal_confidence <= 1.0:
            raise ValueError(
                f"signal_confidence must be between 0.0 and 1.0, "
                f"got {signal_confidence}"
            )
        
        logger.info(
            f"EMATrendStrategy initialized: EMA{fast_period}/{slow_period} crossover"
        )
    
    def analyze(self, market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Analyze market and return signal dict for LiveTradeManager.
        
        This method provides compatibility with LiveTradeManager's expected interface.
        It returns a dictionary instead of SignalOutput for easier handling by
        async trading loops that expect JSON-serializable dicts.
        
        Args:
            market_data: DataFrame with OHLCV data
        
        Returns:
            Dictionary with:
            - action: 'buy', 'sell', or 'hold'
            - confidence: float (0.0-1.0)
            - regime: always 'trending' (EMA strategy assumes trending market)
            Or None if analysis fails
        """
        try:
            slow_period = self.parameters.get('slow_period', 200)
            
            if market_data is None or len(market_data) < slow_period:
                logger.warning(
                    f"Insufficient data for EMA analysis "
                    f"(need {slow_period}, have {len(market_data) if market_data is not None else 0})"
                )
                return None
            
            # Use default position state for analysis
            position_state = PositionState()
            signal_output = self.generate_signal(market_data, position_state)
            
            # Convert SignalOutput to dictionary format
            action = signal_output.signal.value.lower()
            
            return {
                'action': action,
                'confidence': signal_output.confidence,
                'regime': 'trending',  # EMA strategy is trend-following
                'signal_confidence': signal_output.confidence,
                'metadata': signal_output.metadata
            }
        
        except Exception as e:
            logger.error(f"[EMATrendStrategy.analyze()] Error: {e}")
            return None
    
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        position_state: PositionState
    ) -> SignalOutput:
        """
        Generate trading signal based on EMA crossover.
        
        Logic:
        1. Calculate fast and slow EMAs
        2. Detect crossover events:
           - Bullish: fast EMA crosses above slow EMA → BUY
           - Bearish: fast EMA crosses below slow EMA → SELL
        3. No crossover → HOLD
        4. Optional: Confirm trend with price position relative to EMAs
        
        Args:
            market_data: DataFrame with OHLCV data
                        Must have columns: timestamp, open, high, low, close, volume
            position_state: Current position state
            
        Returns:
            SignalOutput with signal, confidence, and metadata
        """
        # DEBUG: Show input schema and EMA parameters
        try:
            market_data.columns = [str(c).lower() for c in market_data.columns]
        except Exception:
            pass
        print(f"\n[EMA DEBUG] Data shape: {market_data.shape}, columns: {list(market_data.columns)}")
        print(f"[EMA DEBUG] Computing EMA{self.get_parameter('fast_period')} and EMA{self.get_parameter('slow_period')}")

        # Validate market data
        self.validate_market_data(market_data)
        
        # Get parameters
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        signal_confidence = self.get_parameter('signal_confidence')
        hold_confidence = self.get_parameter('hold_confidence')
        require_confirmation = self.get_parameter('require_trend_confirmation')
        
        # Check if we have enough data
        if len(market_data) < slow_period + 1:
            logger.warning(
                f"Insufficient data: {len(market_data)} candles, "
                f"need at least {slow_period + 1}"
            )
            return self.create_signal(
                Signal.HOLD,
                hold_confidence,
                reason="Insufficient data for EMA calculation",
                required_candles=slow_period + 1,
                available_candles=len(market_data)
            )
        
        # Calculate EMAs
        fast_ema = Indicators.ema(market_data['close'], fast_period)
        slow_ema = Indicators.ema(market_data['close'], slow_period)
        
        # Get current and previous values
        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        prev_fast = fast_ema.iloc[-2]
        prev_slow = slow_ema.iloc[-2]
        current_price = market_data['close'].iloc[-1]
        
        # Detect crossover
        bullish_crossover = prev_fast <= prev_slow and current_fast > current_slow
        bearish_crossover = prev_fast >= prev_slow and current_fast < current_slow
        
        # DEBUG: Show crossover detection details
        print(f"[EMA DEBUG] Previous: fast={prev_fast:.2f}, slow={prev_slow:.2f}")
        print(f"[EMA DEBUG] Current:  fast={current_fast:.2f}, slow={current_slow:.2f}")
        print(f"[EMA DEBUG] Crossover detection:")
        print(f"  - Bullish: {bullish_crossover} (prev_fast <= prev_slow: {prev_fast <= prev_slow}, current_fast > current_slow: {current_fast > current_slow})")
        print(f"  - Bearish: {bearish_crossover} (prev_fast >= prev_slow: {prev_fast >= prev_slow}, current_fast < current_slow: {current_fast < current_slow})")
        
        # Calculate EMA separation (used for metadata)
        ema_separation = abs(current_fast - current_slow)
        ema_separation_pct = (ema_separation / current_slow) * 100
        
        # Optional: Trend confirmation
        # For bullish signal, price should be above both EMAs
        # For bearish signal, price should be below both EMAs
        trend_confirmed = True
        if require_confirmation:
            if bullish_crossover:
                trend_confirmed = current_price > current_fast and current_price > current_slow
            elif bearish_crossover:
                trend_confirmed = current_price < current_fast and current_price < current_slow
        
        # Calculate ATR for stop-loss placement
        atr = Indicators.atr(market_data, period=14)
        current_atr = atr.iloc[-1] if len(atr) > 0 and not pd.isna(atr.iloc[-1]) else (current_price * 0.02)
        atr_multiplier = self.get_parameter('atr_multiplier') if 'atr_multiplier' in self.parameters else 2.0
        
        # Generate signal
        if bullish_crossover and trend_confirmed:
            # Bullish crossover - BUY signal
            entry_price = float(current_price)
            stop_loss = entry_price - (atr_multiplier * current_atr)
            
            return self.create_signal(
                Signal.BUY,
                signal_confidence,
                reason="Bullish EMA crossover",
                fast_ema=float(current_fast),
                slow_ema=float(current_slow),
                ema_separation_pct=float(ema_separation_pct),
                price=float(current_price),
                trend_confirmed=trend_confirmed,
                crossover_type="bullish",
                entry_price=entry_price,
                stop_loss=float(stop_loss),
                atr=float(current_atr)
            )
        
        elif bearish_crossover and trend_confirmed:
            # Bearish crossover - SELL signal
            entry_price = float(current_price)
            stop_loss = entry_price + (atr_multiplier * current_atr)
            
            return self.create_signal(
                Signal.SELL,
                signal_confidence,
                reason="Bearish EMA crossover",
                fast_ema=float(current_fast),
                slow_ema=float(current_slow),
                ema_separation_pct=float(ema_separation_pct),
                price=float(current_price),
                trend_confirmed=trend_confirmed,
                crossover_type="bearish",
                entry_price=entry_price,
                stop_loss=float(stop_loss),
                atr=float(current_atr)
            )
        
        elif (bullish_crossover or bearish_crossover) and not trend_confirmed:
            # Crossover detected but trend not confirmed
            crossover_type = "bullish" if bullish_crossover else "bearish"
            return self.create_signal(
                Signal.HOLD,
                hold_confidence,
                reason=f"{crossover_type.capitalize()} crossover but trend not confirmed",
                fast_ema=float(current_fast),
                slow_ema=float(current_slow),
                ema_separation_pct=float(ema_separation_pct),
                price=float(current_price),
                trend_confirmed=False,
                crossover_type=crossover_type
            )
        
        else:
            # No crossover - HOLD
            # Determine current trend based on EMA relationship
            if current_fast > current_slow:
                trend = "uptrend"
            elif current_fast < current_slow:
                trend = "downtrend"
            else:
                trend = "neutral"
            
            # DEBUG: Show why HOLD
            print(f"[EMA DEBUG] ❌ HOLD - No crossover detected")
            print(f"[EMA DEBUG]   Current trend: {trend} (fast {current_fast:.2f} vs slow {current_slow:.2f})")
            print(f"[EMA DEBUG]   Separation: {ema_separation_pct:.3f}%")
            print(f"[EMA DEBUG]   To trigger BUY: Need fast EMA to cross ABOVE slow EMA")
            print(f"[EMA DEBUG]   To trigger SELL: Need fast EMA to cross BELOW slow EMA")
            print(f"[EMA DEBUG]   In ranging markets, EMAs flatten and crossovers are rare")
            
            return self.create_signal(
                Signal.HOLD,
                hold_confidence,
                reason="No EMA crossover",
                fast_ema=float(current_fast),
                slow_ema=float(current_slow),
                ema_separation_pct=float(ema_separation_pct),
                price=float(current_price),
                current_trend=trend
            )
    
    def get_ema_values(self, market_data: pd.DataFrame) -> dict:
        """
        Get current EMA values for analysis.
        
        Args:
            market_data: DataFrame with price data
            
        Returns:
            Dictionary with current EMA values
        """
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        
        if len(market_data) < slow_period:
            return {
                'fast_ema': None,
                'slow_ema': None,
                'error': 'Insufficient data'
            }
        
        fast_ema = Indicators.ema(market_data['close'], fast_period)
        slow_ema = Indicators.ema(market_data['close'], slow_period)
        
        return {
            'fast_ema': float(fast_ema.iloc[-1]),
            'slow_ema': float(slow_ema.iloc[-1]),
            'separation': float(abs(fast_ema.iloc[-1] - slow_ema.iloc[-1])),
            'separation_pct': float((abs(fast_ema.iloc[-1] - slow_ema.iloc[-1]) / slow_ema.iloc[-1]) * 100)
        }
    
    def is_in_position_compatible_trend(
        self,
        market_data: pd.DataFrame,
        position_type: str
    ) -> bool:
        """
        Check if current trend is compatible with existing position.
        
        Useful for position management logic in higher-level systems.
        
        Args:
            market_data: DataFrame with price data
            position_type: 'long' or 'short'
            
        Returns:
            True if trend supports holding the position
        """
        fast_period = self.get_parameter('fast_period')
        slow_period = self.get_parameter('slow_period')
        
        if len(market_data) < slow_period:
            return False
        
        fast_ema = Indicators.ema(market_data['close'], fast_period)
        slow_ema = Indicators.ema(market_data['close'], slow_period)
        
        current_fast = fast_ema.iloc[-1]
        current_slow = slow_ema.iloc[-1]
        
        if position_type == 'long':
            # Long position compatible with uptrend (fast > slow)
            return current_fast > current_slow
        elif position_type == 'short':
            # Short position compatible with downtrend (fast < slow)
            return current_fast < current_slow
        else:
            return False


# Example usage
if __name__ == "__main__":
    import numpy as np
    from datetime import datetime, timedelta
    
    print("=== EMA Trend Strategy Example ===\n")
    
    try:
        # Create sample market data with a trend
        print("1. Creating sample market data...")
        n_candles = 300
        dates = pd.date_range(start='2024-01-01', periods=n_candles, freq='1h')
        
        # Create price series with trend changes
        np.random.seed(42)
        base_price = 100
        
        # First half: uptrend
        uptrend = np.cumsum(np.random.randn(n_candles // 2) * 0.3 + 0.1)
        # Second half: downtrend
        downtrend = np.cumsum(np.random.randn(n_candles // 2) * 0.3 - 0.1)
        
        prices = np.concatenate([uptrend, downtrend]) + base_price
        
        market_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(n_candles) * 0.1,
            'high': prices + np.abs(np.random.randn(n_candles) * 0.3),
            'low': prices - np.abs(np.random.randn(n_candles) * 0.3),
            'close': prices,
            'volume': np.random.randint(1000, 5000, n_candles)
        })
        
        print(f"Created {len(market_data)} candles")
        print(f"Price range: ${market_data['close'].min():.2f} - ${market_data['close'].max():.2f}\n")
        
        # Create strategy
        print("2. Initializing EMA Trend Strategy (50/200)...")
        strategy = EMATrendStrategy(
            fast_period=50,
            slow_period=200,
            signal_confidence=0.7,
            hold_confidence=0.0
        )
        print(f"Strategy: {strategy}\n")
        
        # Create position state
        position = PositionState(has_position=False)
        
        # Generate signal
        print("3. Generating signal...")
        signal = strategy.generate_signal(market_data, position)
        print(f"Signal: {signal}")
        print(f"\nSignal Details:")
        print(f"  Type: {signal.signal.value}")
        print(f"  Confidence: {signal.confidence}")
        print(f"  Reason: {signal.metadata.get('reason')}")
        print(f"  Fast EMA: ${signal.metadata.get('fast_ema', 0):.2f}")
        print(f"  Slow EMA: ${signal.metadata.get('slow_ema', 0):.2f}")
        print(f"  EMA Separation: {signal.metadata.get('ema_separation_pct', 0):.3f}%")
        print(f"  Current Price: ${signal.metadata.get('price', 0):.2f}")
        
        # Get EMA values
        print("\n4. Current EMA values...")
        ema_values = strategy.get_ema_values(market_data)
        print(f"Fast EMA (50): ${ema_values['fast_ema']:.2f}")
        print(f"Slow EMA (200): ${ema_values['slow_ema']:.2f}")
        print(f"Separation: ${ema_values['separation']:.2f} ({ema_values['separation_pct']:.3f}%)")
        
        # Test with trend confirmation
        print("\n5. Testing with trend confirmation enabled...")
        strategy_confirmed = EMATrendStrategy(
            fast_period=50,
            slow_period=200,
            signal_confidence=0.7,
            require_trend_confirmation=True
        )
        
        signal_confirmed = strategy_confirmed.generate_signal(market_data, position)
        print(f"Signal with confirmation: {signal_confirmed.signal.value}")
        print(f"Trend confirmed: {signal_confirmed.metadata.get('trend_confirmed', 'N/A')}")
        
        # Simulate signals over time
        print("\n6. Simulating signals over last 50 candles...")
        signals_history = []
        
        for i in range(250, 300):
            data_slice = market_data.iloc[:i+1]
            sig = strategy.generate_signal(data_slice, position)
            
            if sig.signal != Signal.HOLD:
                signals_history.append({
                    'timestamp': data_slice['timestamp'].iloc[-1],
                    'price': data_slice['close'].iloc[-1],
                    'signal': sig.signal.value,
                    'confidence': sig.confidence,
                    'reason': sig.metadata.get('reason')
                })
        
        if signals_history:
            print(f"\nFound {len(signals_history)} actionable signals:")
            for sig_info in signals_history:
                print(f"  {sig_info['timestamp']}: {sig_info['signal']} @ ${sig_info['price']:.2f} "
                      f"(confidence: {sig_info['confidence']}) - {sig_info['reason']}")
        else:
            print("\nNo actionable signals in the period")
        
        # Test position compatibility
        print("\n7. Testing position compatibility...")
        is_compatible_long = strategy.is_in_position_compatible_trend(market_data, 'long')
        is_compatible_short = strategy.is_in_position_compatible_trend(market_data, 'short')
        
        print(f"Current trend compatible with long position: {is_compatible_long}")
        print(f"Current trend compatible with short position: {is_compatible_short}")
        
        # Test with insufficient data
        print("\n8. Testing with insufficient data...")
        small_data = market_data.iloc[:100].copy()
        signal_small = strategy.generate_signal(small_data, position)
        print(f"Signal: {signal_small.signal.value}")
        print(f"Reason: {signal_small.metadata.get('reason')}")
        
        print("\n✓ All examples completed successfully")
        print("\nNote: This strategy only generates signals - no trades are executed.")
        
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install required packages:")
        print("  pip install pandas numpy")
    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Exception in example")