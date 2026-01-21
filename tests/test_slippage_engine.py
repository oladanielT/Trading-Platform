"""
test_slippage_engine.py - Tests for realistic slippage modeling
"""

import pytest
from execution.slippage_engine import (
    SlippageEngine,
    SlippageModel,
    SLIPPAGE_CONSERVATIVE,
    SLIPPAGE_REALISTIC,
    SLIPPAGE_OPTIMISTIC,
)


def test_slippage_fixed_percentage():
    """Test fixed percentage slippage model"""
    engine = SlippageEngine(
        model=SlippageModel.FIXED_PERCENTAGE,
        fixed_slippage_pct=0.1,
        commission_pct=0.05
    )
    
    # Buy order: pays slippage (fill price higher)
    result = engine.calculate_entry_slippage(
        requested_price=1000.0,
        side='buy',
        order_size=1.0
    )
    
    assert result.entry_price == 1000.0
    assert result.exit_price > result.entry_price  # Filled higher
    assert abs(result.slippage_pct - 0.1) < 0.01
    assert result.total_cost_pct == pytest.approx(0.15, abs=0.01)  # 0.1 slippage + 0.05 commission


def test_slippage_volatility_scaled():
    """Test volatility-scaled slippage"""
    engine = SlippageEngine(
        model=SlippageModel.VOLATILITY_SCALED,
        fixed_slippage_pct=0.1,
        commission_pct=0.05,
        volatility_multiplier=2.0
    )
    
    # Low volatility
    result_low_vol = engine.calculate_entry_slippage(
        1000.0, 'buy', 1.0, volatility_pct=1.0
    )
    
    # High volatility
    result_high_vol = engine.calculate_entry_slippage(
        1000.0, 'buy', 1.0, volatility_pct=5.0
    )
    
    # Higher volatility should mean higher slippage
    assert result_high_vol.slippage_pct > result_low_vol.slippage_pct


def test_slippage_market_impact():
    """Test market impact slippage (large orders)"""
    engine = SlippageEngine(
        model=SlippageModel.MARKET_IMPACT,
        fixed_slippage_pct=0.1,
        commission_pct=0.05,
        market_impact_coefficient=0.001
    )
    
    # Small order (0.1% of 24h volume)
    result_small = engine.calculate_entry_slippage(
        1000.0, 'buy', 1.0,
        recent_volume=1_000_000  # 1M in 24h
    )
    
    # Large order (5% of 24h volume)
    result_large = engine.calculate_entry_slippage(
        1000.0, 'buy', 50.0,
        recent_volume=1_000_000  # 1M in 24h
    )
    
    # Larger order should have higher slippage
    assert result_large.slippage_pct > result_small.slippage_pct


def test_slippage_sell_order():
    """Test slippage on sell orders (should fill lower)"""
    engine = SlippageEngine(
        model=SlippageModel.FIXED_PERCENTAGE,
        fixed_slippage_pct=0.1,
        commission_pct=0.05
    )
    
    result = engine.calculate_entry_slippage(
        requested_price=1000.0,
        side='sell',
        order_size=1.0
    )
    
    assert result.exit_price < result.entry_price  # Filled lower on sell
    # 1000 * (1 - 0.001) = 999.0 (0.1% slippage on sell)
    assert result.exit_price == pytest.approx(999.0, abs=0.5)


def test_roundtrip_cost():
    """Test total cost calculation for round-trip trade"""
    engine = SLIPPAGE_REALISTIC
    
    result = engine.estimate_roundtrip_cost(
        entry_price=100.0,
        exit_price=105.0,
        order_size=10.0,
        volatility_pct=2.0
    )
    
    # Ideal profit: (105 - 100) * 10 = 50
    assert result['ideal_profit'] == pytest.approx(50.0, abs=0.1)
    
    # Actual profit should be less due to slippage/commission
    assert result['actual_profit'] < result['ideal_profit']
    
    # Slippage cost should be positive
    assert result['slippage_cost'] > 0
    
    # Slippage cost as % should be reasonable (< 10% for profitable trade)
    assert result['slippage_cost_pct'] < 10.0


def test_preset_engines():
    """Test that preset engines have correct configurations"""
    # Conservative: wider slippage, higher commission, worst-case fills
    assert SLIPPAGE_CONSERVATIVE.fixed_slippage_pct == 0.15
    assert SLIPPAGE_CONSERVATIVE.commission_pct == 0.1
    assert not SLIPPAGE_CONSERVATIVE.use_half_spread
    
    # Realistic: normal slippage, normal commission, normal fills
    assert SLIPPAGE_REALISTIC.fixed_slippage_pct == 0.1
    assert SLIPPAGE_REALISTIC.commission_pct == 0.05
    assert SLIPPAGE_REALISTIC.use_half_spread
    
    # Optimistic: tight slippage, maker commission, favorable fills
    assert SLIPPAGE_OPTIMISTIC.fixed_slippage_pct == 0.05
    assert SLIPPAGE_OPTIMISTIC.commission_pct == 0.025


def test_slippage_realistic_vs_conservative():
    """Compare realistic vs conservative slippage"""
    # Same trade with two different engines
    entry_price = 100.0
    exit_price = 105.0
    order_size = 10.0
    
    realistic = SLIPPAGE_REALISTIC.estimate_roundtrip_cost(
        entry_price, exit_price, order_size
    )
    
    conservative = SLIPPAGE_CONSERVATIVE.estimate_roundtrip_cost(
        entry_price, exit_price, order_size
    )
    
    # Conservative should have higher costs
    assert conservative['slippage_cost'] > realistic['slippage_cost']
    assert conservative['slippage_cost_pct'] > realistic['slippage_cost_pct']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
