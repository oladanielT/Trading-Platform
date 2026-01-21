"""
range_trading.py - Mean Reversion Strategy for Ranging Markets

Implements RSI-based mean reversion logic optimized for sideways markets.
Buys oversold conditions, sells overbought conditions, with support/resistance
confirmation and volatility filters.
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None

from strategies.base import BaseStrategy, SignalOutput, Signal, PositionState
from data.candles import Indicators

logger = logging.getLogger(__name__)

# MODULE LOAD CONFIRMATION
print("\n" + "="*70)
print("[MODULE LOAD] range_trading.py loaded successfully!")
print("[MODULE LOAD] RangeTradingStrategy class available")
print("="*70 + "\n")
import sys
sys.stdout.flush()


class RangeTradingStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy for ranging markets.
    
    Logic:
    1. Detect range-bound conditions (low volatility, weak trend)
    2. Use RSI to identify oversold (BUY) and overbought (SELL) extremes
    3. Confirm with support/resistance proximity
    4. Optional volume and price action filters
    
    Parameters:
        rsi_period: RSI calculation period (default: 14)
        rsi_oversold: RSI threshold for BUY signals (default: 30)
        rsi_overbought: RSI threshold for SELL signals (default: 70)
        lookback: Candles for support/resistance (default: 100)
        support_resistance_pct: % proximity to S/R zones (default: 1.0%)
        require_volume_confirmation: Require volume spike (default: False)
        min_volatility: Min volatility to trade (default: 0.01)
        max_volatility: Max volatility for ranging (default: 0.08)
        signal_confidence: Base confidence for signals (default: 0.65)
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 45.0,
        rsi_overbought: float = 55.0,
        rsi_strong_oversold: float = 35.0,
        rsi_strong_overbought: float = 65.0,
        lookback: int = 100,
        support_resistance_pct: float = 0.5,
        require_volume_confirmation: bool = False,
        min_volatility: float = 0.001,
        max_volatility: float = 0.15,
        signal_confidence: float = 0.65,
        name: Optional[str] = None,
        **kwargs
    ):
        if pd is None or np is None:
            raise ImportError("pandas and numpy required")
        
        super().__init__(
            name=name or "RangeTradingStrategy",
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            rsi_strong_oversold=rsi_strong_oversold,
            rsi_strong_overbought=rsi_strong_overbought,
            lookback=lookback,
            support_resistance_pct=support_resistance_pct,
            require_volume_confirmation=require_volume_confirmation,
            min_volatility=min_volatility,
            max_volatility=max_volatility,
            signal_confidence=signal_confidence,
            **kwargs
        )
        
        logger.info(
            f"RangeTradingStrategy initialized: RSI({rsi_period}) "
            f"normal: oversold<{rsi_oversold}, overbought>{rsi_overbought} | "
            f"strong: oversold<{rsi_strong_oversold}, overbought>{rsi_strong_overbought}"
        )
    
    def generate_signal(
        self,
        market_data: pd.DataFrame,
        position_state: PositionState
    ) -> SignalOutput:
        """Generate mean reversion signal for ranging markets"""
        # IMMEDIATE DEBUG - BEFORE ANY OTHER CODE
        import sys
        print("\n" + "="*70, flush=True)
        print("[RANGE DEBUG] ========== RangeTradingStrategy.generate_signal() CALLED ==========", flush=True)
        print("="*70 + "\n", flush=True)
        logger.info("[RANGE DEBUG] ========== RangeTradingStrategy.generate_signal() CALLED ==========")
        sys.stdout.flush()
        
        try:
            # Normalize columns
            try:
                market_data.columns = [str(c).lower() for c in market_data.columns]
            except Exception:
                pass
            
            print("\n" + "="*60, flush=True)
            print("[RANGE DEBUG] Starting analysis", flush=True)
            logger.info("[RANGE DEBUG] Starting analysis")
            print(f"[RANGE DEBUG] Data shape: {market_data.shape}", flush=True)
            print(f"[RANGE DEBUG] Price data available: {len(market_data)} candles", flush=True)
            logger.info(f"[RANGE DEBUG] Data shape: {market_data.shape}")
            
            self.validate_market_data(market_data)
            
            lookback = self.get_parameter('lookback')
            rsi_period = self.get_parameter('rsi_period')
            
            if len(market_data) < lookback:
                print(f"[RANGE DEBUG] ❌ Insufficient data: {len(market_data)} < {lookback}", flush=True)
                logger.warning(f"[RANGE DEBUG] Insufficient data: {len(market_data)} < {lookback}")
                return self.create_signal(Signal.HOLD, 0.0, reason="Insufficient data")
            
            # ====== VOLATILITY CALCULATION ======
            print(f"\n[RANGE DEBUG] === VOLATILITY CHECK ===", flush=True)
            logger.info("[RANGE DEBUG] === VOLATILITY CHECK ===")
            volatility = self._calculate_volatility(market_data)
            min_vol = self.get_parameter('min_volatility')
            max_vol = self.get_parameter('max_volatility')
            
            print(f"[RANGE DEBUG] Calculation method: std(log_returns) * sqrt(252)", flush=True)
            print(f"[RANGE DEBUG] Calculated volatility: {volatility:.6f} ({volatility*100:.2f}%)", flush=True)
            print(f"[RANGE DEBUG] Acceptable range: {min_vol:.6f} - {max_vol:.6f}", flush=True)
            logger.info(f"[RANGE DEBUG] Volatility: {volatility:.6f}, range: {min_vol:.6f}-{max_vol:.6f}")
            
            volatility_in_range = (min_vol <= volatility <= max_vol)
            if volatility_in_range:
                print(f"[RANGE DEBUG] ✅ Volatility in range", flush=True)
            else:
                print(f"[RANGE DEBUG] ⚠️  Volatility OUTSIDE range (but continuing analysis)", flush=True)
            
            # ====== RSI CALCULATION ======
            print(f"\n[RANGE DEBUG] === RSI CHECK ===", flush=True)
            logger.info("[RANGE DEBUG] === RSI CHECK ===")
            rsi = Indicators.rsi(market_data['close'], rsi_period)
            current_rsi = rsi.iloc[-1]
            
            rsi_oversold = self.get_parameter('rsi_oversold')
            rsi_overbought = self.get_parameter('rsi_overbought')
            rsi_strong_oversold = self.get_parameter('rsi_strong_oversold')
            rsi_strong_overbought = self.get_parameter('rsi_strong_overbought')
            
            print(f"[RANGE DEBUG] RSI({rsi_period}): {current_rsi:.2f}", flush=True)
            print(f"[RANGE DEBUG] Normal thresholds: oversold <{rsi_oversold}, overbought >{rsi_overbought}", flush=True)
            print(f"[RANGE DEBUG] Strong thresholds: oversold <{rsi_strong_oversold}, overbought >{rsi_strong_overbought}", flush=True)
            logger.info(f"[RANGE DEBUG] RSI: {current_rsi:.2f}")
            
            # Check BOTH normal and strong thresholds
            is_oversold_normal = current_rsi < rsi_oversold
            is_oversold_strong = current_rsi < rsi_strong_oversold
            is_overbought_normal = current_rsi > rsi_overbought
            is_overbought_strong = current_rsi > rsi_strong_overbought
            
            print(f"[RANGE DEBUG]   - Oversold normal (<{rsi_oversold}): {is_oversold_normal}", flush=True)
            print(f"[RANGE DEBUG]   - Oversold strong (<{rsi_strong_oversold}): {is_oversold_strong}", flush=True)
            print(f"[RANGE DEBUG]   - Overbought normal (>{rsi_overbought}): {is_overbought_normal}", flush=True)
            print(f"[RANGE DEBUG]   - Overbought strong (>{rsi_strong_overbought}): {is_overbought_strong}", flush=True)
            
            # ====== SUPPORT/RESISTANCE DETECTION ======
            print(f"\n[RANGE DEBUG] === SUPPORT/RESISTANCE CHECK ===", flush=True)
            logger.info("[RANGE DEBUG] === SUPPORT/RESISTANCE CHECK ===")
            support, resistance = self._find_support_resistance(market_data)
            current_price = market_data['close'].iloc[-1]
            
            print(f"[RANGE DEBUG] Current price: ${current_price:.2f}", flush=True)
            print(f"[RANGE DEBUG] Support level: ${support:.2f}", flush=True)
            print(f"[RANGE DEBUG] Resistance level: ${resistance:.2f}", flush=True)
            print(f"[RANGE DEBUG] Range width: ${resistance - support:.2f} ({((resistance-support)/support)*100:.2f}%)", flush=True)
            logger.info(f"[RANGE DEBUG] Price: {current_price:.2f}, S/R: {support:.2f}/{resistance:.2f}")
            
            # Distance from S/R
            dist_to_support = ((current_price - support) / support) * 100
            dist_to_resistance = ((resistance - current_price) / resistance) * 100
            
            print(f"[RANGE DEBUG] Distance from support: {dist_to_support:.3f}%", flush=True)
            print(f"[RANGE DEBUG] Distance from resistance: {dist_to_resistance:.3f}%", flush=True)
            
            # Check proximity to S/R
            sr_pct = self.get_parameter('support_resistance_pct')
            near_support = dist_to_support <= sr_pct
            near_resistance = dist_to_resistance <= sr_pct
            
            print(f"[RANGE DEBUG] Proximity threshold: {sr_pct}%", flush=True)
            print(f"[RANGE DEBUG]   - Near support ({sr_pct}%): {near_support}", flush=True)
            print(f"[RANGE DEBUG]   - Near resistance ({sr_pct}%): {near_resistance}", flush=True)
            
            # ====== VOLUME CONFIRMATION (OPTIONAL) ======
            volume_confirmed = True
            if self.get_parameter('require_volume_confirmation'):
                print(f"\n[RANGE DEBUG] === VOLUME CHECK ===", flush=True)
                logger.info("[RANGE DEBUG] === VOLUME CHECK ===")
                volume_confirmed = self._check_volume(market_data)
                print(f"[RANGE DEBUG] Volume spike confirmed: {volume_confirmed}", flush=True)
            else:
                print(f"\n[RANGE DEBUG] === VOLUME CHECK ===", flush=True)
                print(f"[RANGE DEBUG] Volume confirmation disabled (skipped)", flush=True)
            
            # ====== SIGNAL GENERATION ======
            print(f"\n[RANGE DEBUG] === SIGNAL DECISION ===", flush=True)
            logger.info("[RANGE DEBUG] === SIGNAL DECISION ===")
            signal_conf = self.get_parameter('signal_confidence')
            
            # ====== BUY SIGNALS (Tiered) ======
            # STRONG BUY: Oversold + near support
            if is_oversold_strong and near_support and volume_confirmed:
                confidence = self._calculate_confidence(
                    current_rsi, rsi_strong_oversold, volatility, near_support, near_resistance, "buy", strength="strong"
                )
                
                print(f"[RANGE DEBUG] ✅✅ STRONG BUY SIGNAL GENERATED", flush=True)
                print(f"[RANGE DEBUG]   Condition: RSI STRONG oversold ({current_rsi:.2f} < {rsi_strong_oversold}) + near support", flush=True)
                print(f"[RANGE DEBUG]   Confidence: {confidence:.3f} (STRONG)", flush=True)
                print("="*60 + "\n", flush=True)
                logger.info(f"[RANGE DEBUG] STRONG BUY signal, confidence: {confidence:.3f}")
                import sys
                sys.stdout.flush()
                
                stop_loss = support * 0.995
                return self.create_signal(
                    Signal.BUY,
                    confidence,
                    reason="RSI strong oversold near support",
                    rsi=float(current_rsi),
                    support=float(support),
                    resistance=float(resistance),
                    volatility=float(volatility),
                    entry_price=float(current_price),
                    stop_loss=float(stop_loss),
                    target_price=float(resistance * 0.99),
                    signal_strength="strong"
                )
            
            # NORMAL BUY: Oversold (normal) + near support
            if is_oversold_normal and near_support and volume_confirmed:
                confidence = self._calculate_confidence(
                    current_rsi, rsi_oversold, volatility, near_support, near_resistance, "buy", strength="normal"
                )
                
                print(f"[RANGE DEBUG] ✅ BUY SIGNAL GENERATED", flush=True)
                print(f"[RANGE DEBUG]   Condition: RSI oversold ({current_rsi:.2f} < {rsi_oversold}) + near support", flush=True)
                print(f"[RANGE DEBUG]   Confidence: {confidence:.3f}", flush=True)
                print("="*60 + "\n", flush=True)
                logger.info(f"[RANGE DEBUG] BUY signal, confidence: {confidence:.3f}")
                import sys
                sys.stdout.flush()
                
                stop_loss = support * 0.995
                return self.create_signal(
                    Signal.BUY,
                    confidence,
                    reason="RSI oversold near support",
                    rsi=float(current_rsi),
                    support=float(support),
                    resistance=float(resistance),
                    volatility=float(volatility),
                    entry_price=float(current_price),
                    stop_loss=float(stop_loss),
                    target_price=float(resistance * 0.99),
                    signal_strength="normal"
                )
            
            # ====== SELL SIGNALS (Tiered) ======
            # STRONG SELL: Overbought + near resistance
            if is_overbought_strong and near_resistance and volume_confirmed:
                confidence = self._calculate_confidence(
                    current_rsi, rsi_strong_overbought, volatility, near_support, near_resistance, "sell", strength="strong"
                )
                
                print(f"[RANGE DEBUG] ✅✅ STRONG SELL SIGNAL GENERATED", flush=True)
                print(f"[RANGE DEBUG]   Condition: RSI STRONG overbought ({current_rsi:.2f} > {rsi_strong_overbought}) + near resistance", flush=True)
                print(f"[RANGE DEBUG]   Confidence: {confidence:.3f} (STRONG)", flush=True)
                print("="*60 + "\n", flush=True)
                logger.info(f"[RANGE DEBUG] STRONG SELL signal, confidence: {confidence:.3f}")
                import sys
                sys.stdout.flush()
                
                stop_loss = resistance * 1.005
                return self.create_signal(
                    Signal.SELL,
                    confidence,
                    reason="RSI strong overbought near resistance",
                    rsi=float(current_rsi),
                    support=float(support),
                    resistance=float(resistance),
                    volatility=float(volatility),
                    entry_price=float(current_price),
                    stop_loss=float(stop_loss),
                    target_price=float(support * 1.01),
                    signal_strength="strong"
                )
            
            # NORMAL SELL: Overbought (normal) + near resistance
            if is_overbought_normal and near_resistance and volume_confirmed:
                confidence = self._calculate_confidence(
                    current_rsi, rsi_overbought, volatility, near_support, near_resistance, "sell", strength="normal"
                )
                
                print(f"[RANGE DEBUG] ✅ SELL SIGNAL GENERATED", flush=True)
                print(f"[RANGE DEBUG]   Condition: RSI overbought ({current_rsi:.2f} > {rsi_overbought}) + near resistance", flush=True)
                print(f"[RANGE DEBUG]   Confidence: {confidence:.3f}", flush=True)
                print("="*60 + "\n", flush=True)
                logger.info(f"[RANGE DEBUG] SELL signal, confidence: {confidence:.3f}")
                import sys
                sys.stdout.flush()
                
                stop_loss = resistance * 1.005
                return self.create_signal(
                    Signal.SELL,
                    confidence,
                    reason="RSI overbought near resistance",
                    rsi=float(current_rsi),
                    support=float(support),
                    resistance=float(resistance),
                    volatility=float(volatility),
                    entry_price=float(current_price),
                    stop_loss=float(stop_loss),
                    target_price=float(support * 1.01),
                    signal_strength="normal"
                )
            
            # HOLD: Conditions not met
            print(f"[RANGE DEBUG] ❌ HOLD - Conditions not met:", flush=True)
            print(f"[RANGE DEBUG]   Oversold check (normal): {is_oversold_normal} (RSI {current_rsi:.2f} < {rsi_oversold})", flush=True)
            print(f"[RANGE DEBUG]   Oversold check (strong): {is_oversold_strong} (RSI {current_rsi:.2f} < {rsi_strong_oversold})", flush=True)
            print(f"[RANGE DEBUG]   Overbought check (normal): {is_overbought_normal} (RSI {current_rsi:.2f} > {rsi_overbought})", flush=True)
            print(f"[RANGE DEBUG]   Overbought check (strong): {is_overbought_strong} (RSI {current_rsi:.2f} > {rsi_strong_overbought})", flush=True)
            print(f"[RANGE DEBUG]   Near support: {near_support} (within {sr_pct}%)", flush=True)
            print(f"[RANGE DEBUG]   Near resistance: {near_resistance} (within {sr_pct}%)", flush=True)
            print(f"[RANGE DEBUG]   Volume confirmed: {volume_confirmed}", flush=True)
            print(f"[RANGE DEBUG]", flush=True)
            print(f"[RANGE DEBUG]   For NORMAL BUY: Need RSI < {rsi_oversold} AND price within {sr_pct}% of support", flush=True)
            print(f"[RANGE DEBUG]   For STRONG BUY: Need RSI < {rsi_strong_oversold} AND price within {sr_pct}% of support", flush=True)
            print(f"[RANGE DEBUG]   For NORMAL SELL: Need RSI > {rsi_overbought} AND price within {sr_pct}% of resistance", flush=True)
            print(f"[RANGE DEBUG]   For STRONG SELL: Need RSI > {rsi_strong_overbought} AND price within {sr_pct}% of resistance", flush=True)
            print("="*60 + "\n", flush=True)
            logger.info(f"[RANGE DEBUG] HOLD - RSI: {current_rsi:.2f}, near_support: {near_support}, near_resistance: {near_resistance}")
            import sys
            sys.stdout.flush()
            
            return self.create_signal(
                Signal.HOLD, 0.0,
                reason="No mean reversion setup",
                rsi=float(current_rsi),
                support=float(support),
                resistance=float(resistance),
                volatility=float(volatility),
                near_support=near_support,
                near_resistance=near_resistance,
                is_oversold_normal=is_oversold_normal,
                is_oversold_strong=is_oversold_strong,
                is_overbought_normal=is_overbought_normal,
                is_overbought_strong=is_overbought_strong
            )
            
        except Exception as e:
            import sys
            import traceback
            error_msg = f"[RANGE DEBUG] ❌❌❌ EXCEPTION CAUGHT: {type(e).__name__}: {e}"
            print("\n" + "="*70, flush=True)
            print(error_msg, flush=True)
            print("="*70, flush=True)
            logger.error(error_msg)
            print("\n[RANGE DEBUG] Full traceback:", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            return self.create_signal(Signal.HOLD, 0.0, error=str(e))
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """
        Calculate annualized volatility using log returns (matches regime detector).
        
        Uses: std(log_returns) * sqrt(252) to annualize
        This matches the MarketRegime.detect() calculation method.
        """
        closes = data['close'].values[-50:]  # Use more recent data for stability
        
        if len(closes) < 2:
            return 0.0
        
        # Log returns (same as regime detector)
        log_returns = np.log(closes[1:] / closes[:-1])
        
        # Annualized volatility (assuming ~252 trading periods per year)
        # For hourly data on 1h timeframe, this is conservative but consistent
        annualized_vol = float(np.std(log_returns) * np.sqrt(252))
        
        return annualized_vol
    
    def _find_support_resistance(self, data: pd.DataFrame) -> tuple[float, float]:
        """Find support and resistance levels using swing highs/lows"""
        lookback = self.get_parameter('lookback')
        recent_data = data.iloc[-lookback:]
        
        # Support: recent swing lows
        lows = recent_data['low'].values
        support_candidates = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_candidates.append(lows[i])
        
        support = min(support_candidates) if support_candidates else recent_data['low'].min()
        
        # Resistance: recent swing highs
        highs = recent_data['high'].values
        resistance_candidates = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_candidates.append(highs[i])
        
        resistance = max(resistance_candidates) if resistance_candidates else recent_data['high'].max()
        
        return float(support), float(resistance)
    
    def _check_volume(self, data: pd.DataFrame) -> bool:
        """Check for volume spike (1.2x average)"""
        avg_volume = data['volume'].iloc[-20:-1].mean()
        current_volume = data.iloc[-1]['volume']
        return current_volume > avg_volume * 1.2
    
    def _calculate_confidence(
        self,
        current_rsi: float,
        threshold: float,
        volatility: float,
        near_support: bool,
        near_resistance: bool,
        signal_type: str,
        strength: str = "normal"
    ) -> float:
        """Calculate signal confidence based on multiple factors and signal strength"""
        base_conf = self.get_parameter('signal_confidence')
        
        # Strength multiplier
        strength_multiplier = 1.2 if strength == "strong" else 1.0
        
        # RSI extremity bonus (more extreme = higher confidence)
        if signal_type == "buy":
            rsi_strength = max(0, (threshold - current_rsi) / threshold)  # 0-1
        else:
            rsi_strength = max(0, (current_rsi - threshold) / (100 - threshold))  # 0-1
        
        # Volatility in middle of range = higher confidence
        min_vol = self.get_parameter('min_volatility')
        max_vol = self.get_parameter('max_volatility')
        vol_center = (min_vol + max_vol) / 2
        vol_score = 1.0 - abs(volatility - vol_center) / (max_vol - min_vol)
        
        # S/R proximity bonus
        sr_bonus = 0.1 if (near_support and signal_type == "buy") or (near_resistance and signal_type == "sell") else 0.0
        
        # Combine factors with strength multiplier
        confidence = min(0.95, (base_conf + (rsi_strength * 0.15) + (vol_score * 0.1) + sr_bonus) * strength_multiplier)
        
        return float(confidence)
