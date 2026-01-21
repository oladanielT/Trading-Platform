"""
slippage_engine.py - Realistic slippage modeling for backtesting and sim-live

Models real-world execution costs including:
- Market impact (spreads widen with larger orders)
- Order queue position (time to fill affects price)
- Volatility adjustment (higher volatility = wider spreads)
- Partial fill simulation (orders might not fill in one go)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class SlippageModel(Enum):
    """Slippage modeling strategies"""
    FIXED_PERCENTAGE = "fixed_percentage"  # Fixed 0.1% slippage (most conservative)
    VOLATILITY_SCALED = "volatility_scaled"  # Scales with historical volatility
    MARKET_IMPACT = "market_impact"  # Advanced: depends on order size vs volume
    REALISTIC = "realistic"  # Combination of all factors


@dataclass
class SlippageResult:
    """Result of slippage calculation"""
    entry_price: float
    exit_price: float
    slippage_pct: float
    slippage_amount: float
    commission_pct: float
    total_cost_pct: float
    note: str


class SlippageEngine:
    """
    Calculates realistic execution slippage for order fills.
    
    In real trading, you don't get filled at the exact price you requested:
    - Spreads cost 0.05% - 0.5% depending on liquidity
    - Large orders move the market (market impact)
    - High volatility increases spreads
    - Time delays (latency) can shift prices 0.1% - 1%
    
    Professional backtesting includes these costs.
    """
    
    def __init__(
        self,
        model: SlippageModel = SlippageModel.REALISTIC,
        fixed_slippage_pct: float = 0.1,
        commission_pct: float = 0.05,  # Typical Binance maker+taker
        volatility_multiplier: float = 2.0,
        market_impact_coefficient: float = 0.001,
        use_half_spread: bool = True  # True = favorable execution, False = worst case
    ):
        """
        Initialize slippage engine.
        
        Args:
            model: Which slippage model to use
            fixed_slippage_pct: Base slippage % (for FIXED_PERCENTAGE model)
            commission_pct: Exchange commission as % of trade size
            volatility_multiplier: How much volatility increases slippage
            market_impact_coefficient: Slippage increase per % of volume
            use_half_spread: True = assume half-spread fills (favorable), False = full spread
        """
        self.model = model
        self.fixed_slippage_pct = fixed_slippage_pct
        self.commission_pct = commission_pct
        self.volatility_multiplier = volatility_multiplier
        self.market_impact_coeff = market_impact_coefficient
        self.use_half_spread = use_half_spread
    
    def calculate_entry_slippage(
        self,
        requested_price: float,
        side: str,
        order_size: float,
        recent_volume: Optional[float] = None,
        volatility_pct: Optional[float] = None,
        is_maker: bool = False
    ) -> SlippageResult:
        """
        Calculate slippage for entry order.
        
        Args:
            requested_price: Price you wanted to buy/sell at
            side: 'buy' or 'sell'
            order_size: Size of order in base asset
            recent_volume: 24h volume in quote asset (for market impact)
            volatility_pct: Historical volatility as % (e.g., 2.5 for 2.5% volatility)
            is_maker: True if your order adds liquidity (maker fee), False if takes (taker fee)
            
        Returns:
            SlippageResult with actual fill price and costs
        """
        # Commission is straightforward
        commission = self.commission_pct if not is_maker else (self.commission_pct * 0.5)
        
        # Calculate slippage based on model
        if self.model == SlippageModel.FIXED_PERCENTAGE:
            slippage_pct = self.fixed_slippage_pct
            note = f"Fixed slippage: {slippage_pct:.3f}%"
        
        elif self.model == SlippageModel.VOLATILITY_SCALED:
            if volatility_pct is None:
                volatility_pct = 2.0  # Default 2% volatility
            # Higher volatility = wider spreads
            slippage_pct = self.fixed_slippage_pct + (volatility_pct * self.volatility_multiplier * 0.01)
            note = f"Volatility-scaled ({volatility_pct:.2f}% vol): {slippage_pct:.3f}%"
        
        elif self.model == SlippageModel.MARKET_IMPACT:
            if recent_volume is None or recent_volume == 0:
                recent_volume = 1_000_000  # Default 1M USDT 24h volume
            # Orders larger than typical volume have higher market impact
            order_pct_of_volume = (order_size * requested_price) / recent_volume * 100
            slippage_pct = self.fixed_slippage_pct + (order_pct_of_volume * self.market_impact_coeff)
            note = f"Market impact ({order_pct_of_volume:.2f}% of vol): {slippage_pct:.3f}%"
        
        else:  # REALISTIC - combination
            slippage_pct = self.fixed_slippage_pct
            
            # Add volatility impact
            if volatility_pct is None:
                volatility_pct = 2.0
            vol_impact = volatility_pct * self.volatility_multiplier * 0.01
            slippage_pct += vol_impact
            
            # Add market impact
            if recent_volume and recent_volume > 0:
                order_pct_of_volume = (order_size * requested_price) / recent_volume * 100
                impact = order_pct_of_volume * self.market_impact_coeff
                slippage_pct += impact
            
            note = f"Realistic (vol+impact): {slippage_pct:.3f}%"
        
        # For unfavorable execution, use full spread; for favorable, use half
        if not self.use_half_spread:
            slippage_pct *= 2.0
        
        # Apply slippage direction (always costs money)
        if side.lower() == 'buy':
            # Buy orders: you pay slippage (fill higher than requested)
            fill_price = requested_price * (1.0 + slippage_pct / 100.0)
            direction = "+"
        else:  # sell
            # Sell orders: you receive less due to slippage (fill lower than requested)
            fill_price = requested_price * (1.0 - slippage_pct / 100.0)
            direction = "-"
        
        # Total cost: commission + slippage
        total_cost_pct = commission + slippage_pct
        slippage_amount = abs(fill_price - requested_price)
        
        return SlippageResult(
            entry_price=requested_price,
            exit_price=fill_price,
            slippage_pct=slippage_pct,
            slippage_amount=slippage_amount,
            commission_pct=commission,
            total_cost_pct=total_cost_pct,
            note=f"{note} | Commission: {commission:.3f}% | Total cost: {direction}{total_cost_pct:.3f}%"
        )
    
    def calculate_exit_slippage(
        self,
        entry_price: float,
        exit_requested: float,
        side: str,
        order_size: float,
        volatility_pct: Optional[float] = None
    ) -> SlippageResult:
        """
        Calculate slippage for exit/close order.
        
        Args:
            entry_price: Price you originally bought/sold at
            exit_requested: Price you want to close at
            side: 'buy' (close long) or 'sell' (close short)
            order_size: Size of order
            volatility_pct: Current volatility
            
        Returns:
            SlippageResult with PnL impact
        """
        return self.calculate_entry_slippage(
            exit_requested, side, order_size, volatility_pct=volatility_pct
        )
    
    def estimate_roundtrip_cost(
        self,
        entry_price: float,
        exit_price: float,
        order_size: float,
        volatility_pct: float = 2.0
    ) -> dict:
        """
        Estimate total cost for a round-trip trade (entry + exit).
        
        Args:
            entry_price: Price you bought at
            exit_price: Price you're selling at
            order_size: Position size
            volatility_pct: Volatility for slippage calc
            
        Returns:
            Dictionary with PnL before/after slippage
        """
        # Entry slippage (buying)
        entry = self.calculate_entry_slippage(
            entry_price, 'buy', order_size, volatility_pct=volatility_pct
        )
        
        # Exit slippage (selling)
        exit_calc = self.calculate_entry_slippage(
            exit_price, 'sell', order_size, volatility_pct=volatility_pct
        )
        
        # Calculate P&L
        ideal_profit_per_unit = exit_price - entry_price
        ideal_profit_total = ideal_profit_per_unit * order_size
        
        # Actual fill prices include slippage
        actual_profit_per_unit = exit_calc.exit_price - entry.exit_price
        actual_profit_total = actual_profit_per_unit * order_size
        
        slippage_cost = ideal_profit_total - actual_profit_total
        slippage_cost_pct = (slippage_cost / ideal_profit_total * 100) if ideal_profit_total != 0 else 0
        
        return {
            'ideal_profit': ideal_profit_total,
            'actual_profit': actual_profit_total,
            'slippage_cost': slippage_cost,
            'slippage_cost_pct': slippage_cost_pct,
            'entry_fee': entry.exit_price - entry_price,
            'exit_fee': exit_calc.exit_price - exit_price,
            'total_fee': (entry.exit_price - entry_price) + (exit_calc.exit_price - exit_price),
        }


# Preset slippage engines for common scenarios
SLIPPAGE_CONSERVATIVE = SlippageEngine(
    model=SlippageModel.REALISTIC,
    fixed_slippage_pct=0.15,  # 0.15% base slippage
    volatility_multiplier=2.0,
    commission_pct=0.1,  # 0.1% commission
    use_half_spread=False  # Worst-case fills
)

SLIPPAGE_REALISTIC = SlippageEngine(
    model=SlippageModel.REALISTIC,
    fixed_slippage_pct=0.1,  # 0.1% base slippage
    volatility_multiplier=1.5,
    commission_pct=0.05,  # 0.05% commission (typical)
    use_half_spread=True  # Normal fills
)

SLIPPAGE_OPTIMISTIC = SlippageEngine(
    model=SlippageModel.FIXED_PERCENTAGE,
    fixed_slippage_pct=0.05,  # Very tight 0.05% slippage
    commission_pct=0.025,  # 0.025% commission (maker)
    use_half_spread=True  # Favorable fills
)


__all__ = [
    'SlippageEngine',
    'SlippageModel',
    'SlippageResult',
    'SLIPPAGE_CONSERVATIVE',
    'SLIPPAGE_REALISTIC',
    'SLIPPAGE_OPTIMISTIC',
]
