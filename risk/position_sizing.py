"""
position_sizing.py - Position Size Calculator

Calculates appropriate position sizes based on account equity, risk management
parameters, and signal characteristics. Does not generate trading signals -
only determines sizing for existing signals.
"""

import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

try:
    import pandas as pd
except ImportError:
    pd = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing calculation methods."""
    FIXED_RISK = "fixed_risk"  # Risk fixed % of equity per trade
    KELLY_CRITERION = "kelly"  # Kelly criterion sizing
    FIXED_FRACTIONAL = "fixed_fractional"  # Fixed % of equity
    VOLATILITY_BASED = "volatility"  # Based on instrument volatility
    CONFIDENCE_WEIGHTED = "confidence_weighted"  # Scaled by signal confidence


@dataclass
class SizingResult:
    """
    Result of position sizing calculation.
    
    Attributes:
        approved: Whether position is approved for trading
        position_size: Calculated position size (units)
        position_value: Total value of position
        risk_amount: Dollar amount at risk
        risk_percentage: Percentage of equity at risk
        confidence_multiplier: Applied confidence adjustment
        rejection_reason: Reason if position was rejected
        metadata: Additional calculation details
    """
    approved: bool
    position_size: float
    position_value: float
    risk_amount: float
    risk_percentage: float
    confidence_multiplier: float = 1.0
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'approved': self.approved,
            'position_size': self.position_size,
            'position_value': self.position_value,
            'risk_amount': self.risk_amount,
            'risk_percentage': self.risk_percentage,
            'confidence_multiplier': self.confidence_multiplier,
            'rejection_reason': self.rejection_reason,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        if not self.approved:
            return f"SizingResult(approved=False, reason={self.rejection_reason})"
        return (f"SizingResult(approved=True, size={self.position_size:.4f}, "
                f"value=${self.position_value:.2f}, risk=${self.risk_amount:.2f})")


class PositionSizer:
    """
    Position size calculator with multiple sizing methods.
    
    Calculates appropriate position sizes based on account equity,
    risk parameters, and signal characteristics. Does not generate
    signals - only processes existing ones.
    """
    
    def __init__(
        self,
        account_equity: float,
        risk_per_trade: float = 0.01,
        max_position_size: Optional[float] = None,
        min_position_size: Optional[float] = None,
        sizing_method: SizingMethod = SizingMethod.FIXED_RISK,
        use_confidence_scaling: bool = False,
        min_confidence: float = 0.0,
        max_leverage: float = 1.0
    ):
        """
        Initialize position sizer.
        
        Args:
            account_equity: Total account equity/balance
            risk_per_trade: Risk percentage per trade (0.01 = 1%)
            max_position_size: Maximum position size in base currency
            min_position_size: Minimum position size in base currency
            sizing_method: Method for calculating position size
            use_confidence_scaling: Scale size by signal confidence
            min_confidence: Minimum confidence to approve position
            max_leverage: Maximum leverage to use (1.0 = no leverage)
        """
        if account_equity <= 0:
            raise ValueError(f"account_equity must be positive, got {account_equity}")
        
        if not 0 < risk_per_trade <= 1:
            raise ValueError(f"risk_per_trade must be between 0 and 1, got {risk_per_trade}")
        
        if not 0 <= min_confidence <= 1:
            raise ValueError(f"min_confidence must be between 0 and 1, got {min_confidence}")
        
        if max_leverage < 1.0:
            raise ValueError(f"max_leverage must be >= 1.0, got {max_leverage}")
        
        self.account_equity = account_equity
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.sizing_method = sizing_method
        self.use_confidence_scaling = use_confidence_scaling
        self.min_confidence = min_confidence
        self.max_leverage = max_leverage
        
        logger.info(
            f"PositionSizer initialized: equity=${account_equity:.2f}, "
            f"risk={risk_per_trade*100:.2f}%, method={sizing_method.value}"
        )
    
    def calculate_size(
        self,
        signal: Dict[str, Any],
        current_price: float,
        stop_loss_price: Optional[float] = None,
        volatility: Optional[float] = None
    ) -> SizingResult:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Signal dictionary with 'signal', 'confidence', 'metadata'
            current_price: Current market price
            stop_loss_price: Stop loss price (required for FIXED_RISK method)
            volatility: Price volatility (required for VOLATILITY_BASED method)
            
        Returns:
            SizingResult with approval status and position size
        """
        # Validate inputs
        if current_price <= 0:
            return self._reject("Invalid current price")
        
        # Extract signal information
        signal_type = signal.get('signal', 'HOLD')
        confidence = signal.get('confidence', 0.0)
        
        # Check if signal is actionable (not HOLD)
        if signal_type == 'HOLD':
            return self._reject("Signal is HOLD - no position sizing needed")
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return self._reject(
                f"Signal confidence {confidence:.2f} below minimum {self.min_confidence:.2f}"
            )
        
        # Calculate confidence multiplier
        confidence_multiplier = confidence if self.use_confidence_scaling else 1.0
        
        # Calculate base position size based on method
        if self.sizing_method == SizingMethod.FIXED_RISK:
            position_size = self._calculate_fixed_risk(
                current_price, stop_loss_price, confidence_multiplier
            )
        
        elif self.sizing_method == SizingMethod.FIXED_FRACTIONAL:
            position_size = self._calculate_fixed_fractional(
                current_price, confidence_multiplier
            )
        
        elif self.sizing_method == SizingMethod.VOLATILITY_BASED:
            position_size = self._calculate_volatility_based(
                current_price, volatility, confidence_multiplier
            )
        
        elif self.sizing_method == SizingMethod.CONFIDENCE_WEIGHTED:
            position_size = self._calculate_confidence_weighted(
                current_price, confidence
            )
        
        elif self.sizing_method == SizingMethod.KELLY_CRITERION:
            # Requires additional statistics from signal
            win_rate = signal.get('metadata', {}).get('win_rate', 0.5)
            avg_win_loss_ratio = signal.get('metadata', {}).get('avg_win_loss_ratio', 1.5)
            position_size = self._calculate_kelly(
                current_price, win_rate, avg_win_loss_ratio, confidence_multiplier
            )
        
        else:
            return self._reject(f"Unknown sizing method: {self.sizing_method}")
        
        # Check if sizing calculation failed
        if position_size is None:
            return self._reject("Position size calculation failed")
        
        # Apply min/max constraints
        if self.min_position_size and position_size < self.min_position_size:
            return self._reject(
                f"Position size {position_size:.4f} below minimum {self.min_position_size:.4f}"
            )
        
        if self.max_position_size and position_size > self.max_position_size:
            position_size = self.max_position_size
            logger.info(f"Position size capped at maximum: {self.max_position_size:.4f}")
        
        # Calculate position value
        position_value = position_size * current_price
        
        # Check leverage constraint
        max_position_value = self.account_equity * self.max_leverage
        if position_value > max_position_value:
            position_size = max_position_value / current_price
            position_value = max_position_value
            logger.info(f"Position size reduced due to leverage constraint")
        
        # Calculate risk metrics
        if stop_loss_price:
            risk_per_unit = abs(current_price - stop_loss_price)
            risk_amount = position_size * risk_per_unit
        else:
            # Estimate risk at 2% price move if no stop loss provided
            risk_amount = position_value * 0.02
        
        risk_percentage = (risk_amount / self.account_equity) * 100
        
        # Create result
        return SizingResult(
            approved=True,
            position_size=position_size,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percentage=risk_percentage,
            confidence_multiplier=confidence_multiplier,
            metadata={
                'signal_type': signal_type,
                'signal_confidence': confidence,
                'current_price': current_price,
                'stop_loss_price': stop_loss_price,
                'sizing_method': self.sizing_method.value,
                'account_equity': self.account_equity
            }
        )
    
    def _calculate_fixed_risk(
        self,
        current_price: float,
        stop_loss_price: Optional[float],
        confidence_multiplier: float
    ) -> Optional[float]:
        """Calculate position size using fixed risk method."""
        if stop_loss_price is None:
            logger.error("Stop loss price required for FIXED_RISK method")
            return None
        
        if stop_loss_price <= 0:
            logger.error("Invalid stop loss price")
            return None
        
        # Calculate risk per unit
        risk_per_unit = abs(current_price - stop_loss_price)
        
        if risk_per_unit == 0:
            logger.error("Stop loss equals current price - no risk defined")
            return None
        
        # Calculate position size to risk specified % of equity
        risk_amount = self.account_equity * self.risk_per_trade * confidence_multiplier
        position_size = risk_amount / risk_per_unit
        
        return position_size
    
    def _calculate_fixed_fractional(
        self,
        current_price: float,
        confidence_multiplier: float
    ) -> float:
        """Calculate position size as fixed fraction of equity."""
        position_value = self.account_equity * self.risk_per_trade * confidence_multiplier
        position_size = position_value / current_price
        return position_size
    
    def _calculate_volatility_based(
        self,
        current_price: float,
        volatility: Optional[float],
        confidence_multiplier: float
    ) -> Optional[float]:
        """Calculate position size based on volatility."""
        if volatility is None:
            logger.error("Volatility required for VOLATILITY_BASED method")
            return None
        
        if volatility <= 0:
            logger.error("Invalid volatility")
            return None
        
        # Inverse volatility sizing: higher volatility = smaller position
        # Normalize by assuming 2% volatility as baseline
        baseline_volatility = 0.02
        volatility_ratio = baseline_volatility / volatility
        
        # Calculate size
        base_position_value = self.account_equity * self.risk_per_trade
        adjusted_value = base_position_value * volatility_ratio * confidence_multiplier
        position_size = adjusted_value / current_price
        
        return position_size
    
    def _calculate_confidence_weighted(
        self,
        current_price: float,
        confidence: float
    ) -> float:
        """Calculate position size weighted by confidence."""
        # Linear scaling: confidence directly scales position size
        base_position_value = self.account_equity * self.risk_per_trade
        weighted_value = base_position_value * confidence
        position_size = weighted_value / current_price
        return position_size
    
    def _calculate_kelly(
        self,
        current_price: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        confidence_multiplier: float
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        # Kelly formula: f = (p * b - q) / b
        # where: p = win rate, q = 1 - p, b = avg win/loss ratio
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio
        
        # Calculate Kelly percentage
        kelly_pct = (p * b - q) / b
        
        # Kelly can be negative (don't bet) or very large - apply constraints
        kelly_pct = max(0, min(kelly_pct, 0.25))  # Cap at 25%
        
        # Use fractional Kelly (typically 1/4 to 1/2 Kelly for safety)
        fractional_kelly = kelly_pct * 0.5 * confidence_multiplier
        
        # Calculate position size
        position_value = self.account_equity * fractional_kelly
        position_size = position_value / current_price
        
        return position_size
    
    def _reject(self, reason: str) -> SizingResult:
        """Create a rejection result."""
        logger.debug(f"Position rejected: {reason}")
        return SizingResult(
            approved=False,
            position_size=0.0,
            position_value=0.0,
            risk_amount=0.0,
            risk_percentage=0.0,
            rejection_reason=reason
        )
    
    def update_equity(self, new_equity: float) -> None:
        """
        Update account equity.
        
        Args:
            new_equity: New account equity value
        """
        if new_equity <= 0:
            raise ValueError(f"new_equity must be positive, got {new_equity}")
        
        old_equity = self.account_equity
        self.account_equity = new_equity
        
        logger.info(f"Equity updated: ${old_equity:.2f} -> ${new_equity:.2f}")
    
    def get_max_position_value(self) -> float:
        """
        Get maximum position value based on equity and leverage.
        
        Returns:
            Maximum position value
        """
        return self.account_equity * self.max_leverage
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Dictionary with current settings
        """
        return {
            'account_equity': self.account_equity,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'sizing_method': self.sizing_method.value,
            'use_confidence_scaling': self.use_confidence_scaling,
            'min_confidence': self.min_confidence,
            'max_leverage': self.max_leverage
        }


# Convenience functions

def calculate_position_size(
    signal: Dict[str, Any],
    account_equity: float,
    current_price: float,
    risk_per_trade: float = 0.01,
    stop_loss_price: Optional[float] = None
) -> SizingResult:
    """
    Convenience function for quick position sizing calculation.
    
    Args:
        signal: Signal dictionary
        account_equity: Account equity
        current_price: Current market price
        risk_per_trade: Risk percentage per trade
        stop_loss_price: Stop loss price
        
    Returns:
        SizingResult
    """
    sizer = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=risk_per_trade
    )
    
    return sizer.calculate_size(signal, current_price, stop_loss_price)


# Example usage
if __name__ == "__main__":
    print("=== Position Sizing Example ===\n")
    
    # Account parameters
    account_equity = 10000.0
    current_price = 50000.0
    stop_loss_price = 49000.0
    
    print(f"Account Equity: ${account_equity:,.2f}")
    print(f"Current Price: ${current_price:,.2f}")
    print(f"Stop Loss: ${stop_loss_price:,.2f}\n")
    
    # Create sample signals
    strong_buy_signal = {
        'signal': 'BUY',
        'confidence': 0.8,
        'metadata': {
            'strategy': 'EMA_Crossover',
            'win_rate': 0.6,
            'avg_win_loss_ratio': 2.0
        }
    }
    
    weak_buy_signal = {
        'signal': 'BUY',
        'confidence': 0.3,
        'metadata': {}
    }
    
    hold_signal = {
        'signal': 'HOLD',
        'confidence': 0.0,
        'metadata': {}
    }
    
    # Test 1: Fixed Risk Method
    print("1. Fixed Risk Method (1% risk per trade)")
    print("-" * 50)
    sizer = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.01,
        sizing_method=SizingMethod.FIXED_RISK
    )
    
    result = sizer.calculate_size(strong_buy_signal, current_price, stop_loss_price)
    print(f"Result: {result}")
    print(f"Position Size: {result.position_size:.4f} BTC")
    print(f"Position Value: ${result.position_value:,.2f}")
    print(f"Risk Amount: ${result.risk_amount:.2f} ({result.risk_percentage:.2f}%)\n")
    
    # Test 2: With Confidence Scaling
    print("2. Fixed Risk with Confidence Scaling")
    print("-" * 50)
    sizer_scaled = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.01,
        sizing_method=SizingMethod.FIXED_RISK,
        use_confidence_scaling=True
    )
    
    result_scaled = sizer_scaled.calculate_size(strong_buy_signal, current_price, stop_loss_price)
    print(f"Strong Signal (0.8 confidence): {result_scaled.position_size:.4f} BTC")
    
    result_weak = sizer_scaled.calculate_size(weak_buy_signal, current_price, stop_loss_price)
    print(f"Weak Signal (0.3 confidence): {result_weak.position_size:.4f} BTC")
    print(f"Scaling Effect: {(result_scaled.position_size / result.position_size):.2f}x\n")
    
    # Test 3: Fixed Fractional Method
    print("3. Fixed Fractional Method (2% of equity)")
    print("-" * 50)
    sizer_fractional = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.02,
        sizing_method=SizingMethod.FIXED_FRACTIONAL
    )
    
    result_fractional = sizer_fractional.calculate_size(strong_buy_signal, current_price)
    print(f"Result: {result_fractional}")
    print(f"Position Size: {result_fractional.position_size:.4f} BTC")
    print(f"Position Value: ${result_fractional.position_value:,.2f}\n")
    
    # Test 4: Kelly Criterion
    print("4. Kelly Criterion Method")
    print("-" * 50)
    sizer_kelly = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.01,
        sizing_method=SizingMethod.KELLY_CRITERION
    )
    
    result_kelly = sizer_kelly.calculate_size(strong_buy_signal, current_price)
    print(f"Result: {result_kelly}")
    print(f"Position Size: {result_kelly.position_size:.4f} BTC")
    print(f"Based on: Win Rate = 60%, Avg W/L = 2.0\n")
    
    # Test 5: Volatility-Based
    print("5. Volatility-Based Method")
    print("-" * 50)
    sizer_vol = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.01,
        sizing_method=SizingMethod.VOLATILITY_BASED
    )
    
    high_vol = 0.05  # 5% volatility
    low_vol = 0.01   # 1% volatility
    
    result_high_vol = sizer_vol.calculate_size(strong_buy_signal, current_price, volatility=high_vol)
    result_low_vol = sizer_vol.calculate_size(strong_buy_signal, current_price, volatility=low_vol)
    
    print(f"High Volatility (5%): {result_high_vol.position_size:.4f} BTC")
    print(f"Low Volatility (1%): {result_low_vol.position_size:.4f} BTC")
    print(f"Ratio: {(result_low_vol.position_size / result_high_vol.position_size):.2f}x\n")
    
    # Test 6: Rejections
    print("6. Testing Rejection Cases")
    print("-" * 50)
    
    # HOLD signal
    result_hold = sizer.calculate_size(hold_signal, current_price, stop_loss_price)
    print(f"HOLD Signal: {result_hold}")
    
    # Low confidence
    sizer_strict = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.01,
        min_confidence=0.5
    )
    result_low_conf = sizer_strict.calculate_size(weak_buy_signal, current_price, stop_loss_price)
    print(f"Low Confidence: {result_low_conf}\n")
    
    # Test 7: Leverage
    print("7. Testing Leverage (2x)")
    print("-" * 50)
    sizer_leverage = PositionSizer(
        account_equity=account_equity,
        risk_per_trade=0.02,
        max_leverage=2.0,
        sizing_method=SizingMethod.FIXED_FRACTIONAL
    )
    
    result_leverage = sizer_leverage.calculate_size(strong_buy_signal, current_price)
    print(f"Max Position Value: ${sizer_leverage.get_max_position_value():,.2f}")
    print(f"Position Size: {result_leverage.position_size:.4f} BTC")
    print(f"Position Value: ${result_leverage.position_value:,.2f}\n")
    
    # Test 8: Equity Update
    print("8. Testing Equity Update")
    print("-" * 50)
    print(f"Initial Equity: ${sizer.account_equity:,.2f}")
    sizer.update_equity(12000.0)
    print(f"Updated Equity: ${sizer.account_equity:,.2f}")
    
    result_updated = sizer.calculate_size(strong_buy_signal, current_price, stop_loss_price)
    print(f"New Position Size: {result_updated.position_size:.4f} BTC")
    print(f"Increase: {((result_updated.position_size / result.position_size) - 1) * 100:.1f}%\n")
    
    print("✓ All examples completed successfully")
    print("\nNote: This module only calculates position sizes - it does not generate signals or execute trades.")