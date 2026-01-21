"""
Quick verification script for Phase 11 modules.
Tests basic functionality of all four core modules.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.strategy_promotion_manager import StrategyPromotionManager, PromotionState
from ai.shadow_execution import ShadowExecutor
from ai.capital_graduation import CapitalGraduation
from ai.promotion_gatekeeper import PromotionGatekeeper

def test_strategy_promotion_manager():
    """Test StrategyPromotionManager basics."""
    print("\n=== Testing StrategyPromotionManager ===")
    
    config = {'strategy_promotion': {'enabled': True, 'require_human_approval': True}}
    manager = StrategyPromotionManager(config)
    
    # Submit candidate
    manager.submit_candidate("test_strategy", validation_results={'qualified': True})
    state = manager.get_state("test_strategy")
    print(f"✓ Submitted strategy, state: {state}")
    
    # Approve
    manager.approve("test_strategy", approved_by="test_user")
    state = manager.get_state("test_strategy")
    print(f"✓ Approved strategy, state: {state}")
    
    # Promote to shadow
    manager.promote_to_shadow("test_strategy")
    state = manager.get_state("test_strategy")
    print(f"✓ Promoted to shadow, state: {state}")
    
    print("✅ StrategyPromotionManager works")


def test_shadow_executor():
    """Test ShadowExecutor basics."""
    print("\n=== Testing ShadowExecutor ===")
    
    executor = ShadowExecutor()
    
    # Simple test - evaluate returns signal
    result = executor.evaluate(
        strategy_id="test_strategy",
        strategy_func=lambda data: {'action': 'open', 'side': 'long', 'symbol': 'BTC/USDT', 'size': 1.0},
        market_data={'close': 50000},
        current_price=50000.0
    )
    
    print(f"✓ Evaluate returned: {result}")
    
    # Get metrics
    metrics = executor.calculate_metrics("test_strategy")
    print(f"✓ Metrics: {metrics}")
    
    print("✅ ShadowExecutor works")


def test_capital_graduation():
    """Test CapitalGraduation basics."""
    print("\n=== Testing CapitalGraduation ===")
    
    config = {'strategy_promotion': {'max_micro_live_capital': 0.05}}
    graduation = CapitalGraduation(config)
    
    # Test capital allocation
    total_equity = 100000.0
    proposed_size = 10000.0
    
    # SHADOW should get 0
    allowed = graduation.get_allowed_capital(
        state=PromotionState.SHADOW,
        total_equity=total_equity,
        proposed_size=proposed_size
    )
    print(f"✓ SHADOW capital: {allowed} (should be 0)")
    assert allowed == 0.0
    
    # MICRO_LIVE should get 5% max
    allowed = graduation.get_allowed_capital(
        state=PromotionState.MICRO_LIVE,
        total_equity=total_equity,
        proposed_size=proposed_size
    )
    print(f"✓ MICRO_LIVE capital: {allowed} (should be {total_equity * 0.05})")
    assert allowed == 5000.0
    
    # FULL_LIVE should get full proposed
    allowed = graduation.get_allowed_capital(
        state=PromotionState.FULL_LIVE,
        total_equity=total_equity,
        proposed_size=proposed_size
    )
    print(f"✓ FULL_LIVE capital: {allowed} (should be {proposed_size})")
    assert allowed == proposed_size
    
    print("✅ CapitalGraduation works")


def test_promotion_gatekeeper():
    """Test PromotionGatekeeper basics."""
    print("\n=== Testing PromotionGatekeeper ===")
    
    config = {
        'strategy_promotion': {
            'auto_demote': True,
            'demotion_criteria': {
                'max_drawdown': 0.15,
                'min_expectancy': 0.0,
                'min_win_rate': 0.40
            }
        }
    }
    gatekeeper = PromotionGatekeeper(config)
    
    # Test healthy strategy
    healthy_metrics = {
        'total_trades': 50,
        'max_drawdown': 0.05,
        'expectancy': 1.0,
        'win_rate': 0.55,
        'consecutive_losses': 2
    }
    decision = gatekeeper.evaluate("healthy_strategy", healthy_metrics)
    print(f"✓ Healthy strategy: should_demote={decision.should_demote}")
    assert decision.should_demote is False
    
    # Test unhealthy strategy
    unhealthy_metrics = {
        'total_trades': 50,
        'max_drawdown': 0.25,  # Exceeds 15%
        'expectancy': 0.5,
        'win_rate': 0.50,
        'consecutive_losses': 2
    }
    decision = gatekeeper.evaluate("unhealthy_strategy", unhealthy_metrics)
    print(f"✓ Unhealthy strategy: should_demote={decision.should_demote}, reason={decision.reason}")
    assert decision.should_demote is True
    
    print("✅ PromotionGatekeeper works")


if __name__ == "__main__":
    try:
        test_strategy_promotion_manager()
        test_shadow_executor()
        test_capital_graduation()
        test_promotion_gatekeeper()
        
        print("\n" + "="*60)
        print("🎉 ALL PHASE 11 MODULES VERIFIED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
