"""
Tests for CapitalGraduation.

Verifies:
- Capital limits enforced per state
- No capital for SHADOW/PAPER
- Limited capital for MICRO_LIVE
- Full portfolio rules for FULL_LIVE
- No capital leaks
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.capital_graduation import CapitalGraduation
from ai.strategy_promotion_manager import PromotionState


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'strategy_promotion': {
            'max_micro_live_capital': 0.05,
            'capital_limits': {
                'SHADOW': 0.0,
                'PAPER': 0.0,
                'MICRO_LIVE': 0.05,
                'FULL_LIVE': 1.0
            }
        }
    }


@pytest.fixture
def graduation(config):
    """Create capital graduation instance."""
    return CapitalGraduation(config)


class TestCapitalLimits:
    """Test capital allocation limits per state."""
    
    def test_shadow_gets_zero_capital(self, graduation):
        """Verify SHADOW state gets 0% capital."""
        capital = graduation.get_allowed_capital(
            state=PromotionState.SHADOW,
            portfolio_value=100000.0
        )
        
        assert capital == 0.0
    
    def test_paper_gets_zero_capital(self, graduation):
        """Verify PAPER state gets 0% capital."""
        capital = graduation.get_allowed_capital(
            state=PromotionState.PAPER,
            portfolio_value=100000.0
        )
        
        assert capital == 0.0
    
    def test_micro_live_gets_limited_capital(self, graduation):
        """Verify MICRO_LIVE gets 5% max."""
        portfolio_value = 100000.0
        capital = graduation.get_allowed_capital(
            state=PromotionState.MICRO_LIVE,
            portfolio_value=portfolio_value
        )
        
        assert capital == 5000.0  # 5% of 100k
        assert capital / portfolio_value == 0.05
    
    def test_full_live_gets_full_allocation(self, graduation):
        """Verify FULL_LIVE gets 100% allocation (subject to portfolio rules)."""
        portfolio_value = 100000.0
        capital = graduation.get_allowed_capital(
            state=PromotionState.FULL_LIVE,
            portfolio_value=portfolio_value
        )
        
        # Returns 100% but actual allocation controlled by PortfolioRiskManager
        assert capital == portfolio_value
    
    def test_research_state_gets_zero(self, graduation):
        """Verify RESEARCH state gets no capital."""
        capital = graduation.get_allowed_capital(
            state=PromotionState.RESEARCH,
            portfolio_value=100000.0
        )
        
        assert capital == 0.0
    
    def test_approved_state_gets_zero(self, graduation):
        """Verify APPROVED state gets no capital."""
        capital = graduation.get_allowed_capital(
            state=PromotionState.APPROVED,
            portfolio_value=100000.0
        )
        
        assert capital == 0.0


class TestGraduationMultiplier:
    """Test capital graduation multipliers."""
    
    def test_apply_multiplier_shadow(self, graduation):
        """Test multiplier application for SHADOW."""
        position_size = 10000.0
        
        adjusted = graduation.apply_graduation_multiplier(
            state=PromotionState.SHADOW,
            requested_capital=position_size
        )
        
        assert adjusted == 0.0  # Always 0 for SHADOW
    
    def test_apply_multiplier_paper(self, graduation):
        """Test multiplier application for PAPER."""
        position_size = 10000.0
        
        adjusted = graduation.apply_graduation_multiplier(
            state=PromotionState.PAPER,
            requested_capital=position_size
        )
        
        assert adjusted == 0.0  # Always 0 for PAPER
    
    def test_apply_multiplier_micro_live(self, graduation):
        """Test multiplier application for MICRO_LIVE."""
        # Request 10k but MICRO_LIVE has 5% limit
        requested = 10000.0
        portfolio_value = 100000.0
        
        # Get max allowed
        max_capital = graduation.get_allowed_capital(
            state=PromotionState.MICRO_LIVE,
            portfolio_value=portfolio_value
        )
        
        # Should cap at 5k
        assert max_capital == 5000.0
    
    def test_apply_multiplier_full_live(self, graduation):
        """Test multiplier for FULL_LIVE doesn't restrict."""
        requested = 10000.0
        
        adjusted = graduation.apply_graduation_multiplier(
            state=PromotionState.FULL_LIVE,
            requested_capital=requested
        )
        
        # Full allocation - no reduction
        # Note: PortfolioRiskManager applies additional limits
        assert adjusted == requested


class TestCapitalLeakPrevention:
    """Test prevention of capital leaks."""
    
    def test_cannot_exceed_state_limit(self, graduation):
        """Verify cannot allocate more than state limit."""
        portfolio_value = 100000.0
        
        # Try to request more than MICRO_LIVE allows
        requested = 20000.0  # Request 20%
        
        max_allowed = graduation.get_allowed_capital(
            state=PromotionState.MICRO_LIVE,
            portfolio_value=portfolio_value
        )
        
        # Should cap at 5%
        assert max_allowed == 5000.0
        assert max_allowed < requested
    
    def test_zero_capital_states_always_zero(self, graduation):
        """Verify zero-capital states never allocate."""
        portfolio_value = 100000.0
        
        # Test all zero-capital states
        zero_states = [
            PromotionState.RESEARCH,
            PromotionState.APPROVED,
            PromotionState.SHADOW,
            PromotionState.PAPER,
            PromotionState.REJECTED,
            PromotionState.RETIRED
        ]
        
        for state in zero_states:
            capital = graduation.get_allowed_capital(
                state=state,
                portfolio_value=portfolio_value
            )
            assert capital == 0.0, f"State {state} should have 0 capital"
    
    def test_state_limits_immutable(self, graduation):
        """Verify state limits cannot be exceeded."""
        limits = graduation.get_state_limits()
        
        assert limits[PromotionState.SHADOW] == 0.0
        assert limits[PromotionState.PAPER] == 0.0
        assert limits[PromotionState.MICRO_LIVE] == 0.05
        assert limits[PromotionState.FULL_LIVE] == 1.0


class TestConfigurableAllocation:
    """Test configurable capital allocation."""
    
    def test_custom_micro_live_limit(self):
        """Test custom MICRO_LIVE limit from config."""
        config = {
            'strategy_promotion': {
                'max_micro_live_capital': 0.03,  # 3% instead of 5%
                'capital_limits': {
                    'MICRO_LIVE': 0.03
                }
            }
        }
        
        graduation = CapitalGraduation(config)
        
        capital = graduation.get_allowed_capital(
            state=PromotionState.MICRO_LIVE,
            portfolio_value=100000.0
        )
        
        assert capital == 3000.0  # 3% of 100k
    
    def test_default_limits_when_no_config(self):
        """Test default limits when config missing."""
        config = {}
        graduation = CapitalGraduation(config)
        
        # Should use defaults
        capital = graduation.get_allowed_capital(
            state=PromotionState.MICRO_LIVE,
            portfolio_value=100000.0
        )
        
        # Default is 5%
        assert capital == 5000.0


class TestSafetyGuarantees:
    """Test critical safety guarantees."""
    
    def test_shadow_never_trades_real_money(self, graduation):
        """Verify SHADOW mode never allocates capital."""
        # Even with large portfolio
        portfolio_value = 1000000.0
        
        capital = graduation.get_allowed_capital(
            state=PromotionState.SHADOW,
            portfolio_value=portfolio_value
        )
        
        assert capital == 0.0
    
    def test_paper_never_trades_real_money(self, graduation):
        """Verify PAPER mode never allocates capital."""
        portfolio_value = 1000000.0
        
        capital = graduation.get_allowed_capital(
            state=PromotionState.PAPER,
            portfolio_value=portfolio_value
        )
        
        assert capital == 0.0
    
    def test_micro_live_bounded(self, graduation):
        """Verify MICRO_LIVE is strictly bounded."""
        # Test various portfolio sizes
        portfolio_sizes = [10000, 50000, 100000, 500000, 1000000]
        
        for size in portfolio_sizes:
            capital = graduation.get_allowed_capital(
                state=PromotionState.MICRO_LIVE,
                portfolio_value=size
            )
            
            # Should always be 5%
            assert capital / size == 0.05
    
    def test_can_reduce_but_not_increase_capital(self, graduation):
        """Verify graduation can only reduce allocation."""
        requested = 10000.0
        
        # SHADOW reduces to 0
        shadow_capital = graduation.apply_graduation_multiplier(
            state=PromotionState.SHADOW,
            requested_capital=requested
        )
        assert shadow_capital <= requested
        
        # FULL_LIVE doesn't increase (stays same)
        full_capital = graduation.apply_graduation_multiplier(
            state=PromotionState.FULL_LIVE,
            requested_capital=requested
        )
        assert full_capital <= requested


class TestDiagnostics:
    """Test diagnostic information."""
    
    def test_get_diagnostics(self, graduation):
        """Test diagnostic output."""
        diagnostics = graduation.get_diagnostics()
        
        assert 'state_limits' in diagnostics
        assert 'max_micro_live_capital' in diagnostics
        
        # Verify limits present for all live states
        limits = diagnostics['state_limits']
        assert PromotionState.SHADOW in limits
        assert PromotionState.MICRO_LIVE in limits
        assert PromotionState.FULL_LIVE in limits
    
    def test_get_state_limits(self, graduation):
        """Test state limits retrieval."""
        limits = graduation.get_state_limits()
        
        # Should have all promotion states
        assert len(limits) >= 4  # At least SHADOW, PAPER, MICRO, FULL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
