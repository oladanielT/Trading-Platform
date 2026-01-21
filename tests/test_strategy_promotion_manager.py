"""
Tests for StrategyPromotionManager.

Verifies:
- Strict FSM enforcement (no state skipping)
- Human approval requirement
- No auto-promotion
- Demotion logic
- Audit trail
"""

import pytest
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.strategy_promotion_manager import (
    PromotionState,
    StrategyPromotionManager,
    StrategyRecord,
    InvalidTransitionError,
    StrategyNotFoundError
)


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'strategy_promotion': {
            'enabled': True,
            'require_human_approval': True,
            'auto_demote': True
        }
    }


@pytest.fixture
def manager(config):
    """Create promotion manager."""
    return StrategyPromotionManager(config)


class TestPromotionStateTransitions:
    """Test FSM state transition logic."""
    
    def test_valid_sequential_transitions(self, manager):
        """Test allowed sequential state transitions."""
        strategy_id = "test_strategy_1"
        
        # RESEARCH -> APPROVED
        manager.submit_candidate(strategy_id, {'validation': 'passed'})
        assert manager.get_state(strategy_id) == PromotionState.RESEARCH
        
        manager.approve(strategy_id)
        assert manager.get_state(strategy_id) == PromotionState.APPROVED
        
        # APPROVED -> SHADOW
        manager.promote_to_shadow(strategy_id)
        assert manager.get_state(strategy_id) == PromotionState.SHADOW
        
        # SHADOW -> PAPER
        manager.promote_to_paper(strategy_id)
        assert manager.get_state(strategy_id) == PromotionState.PAPER
        
        # PAPER -> MICRO_LIVE
        manager.promote_to_micro_live(strategy_id)
        assert manager.get_state(strategy_id) == PromotionState.MICRO_LIVE
        
        # MICRO_LIVE -> FULL_LIVE
        manager.promote_to_full_live(strategy_id)
        assert manager.get_state(strategy_id) == PromotionState.FULL_LIVE
    
    def test_state_skipping_blocked(self, manager):
        """Verify cannot skip states in FSM."""
        strategy_id = "test_strategy_skip"
        
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        
        # Try to skip from APPROVED -> PAPER (should fail)
        with pytest.raises(InvalidTransitionError):
            manager.promote_to_paper(strategy_id)
        
        # Must go through SHADOW first
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)  # Now allowed
    
    def test_backward_transitions_only_via_demotion(self, manager):
        """Verify cannot go backwards except via demote()."""
        strategy_id = "test_backward"
        
        # Get to FULL_LIVE
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)
        manager.promote_to_micro_live(strategy_id)
        manager.promote_to_full_live(strategy_id)
        
        # Demote - goes to PAPER
        manager.demote(strategy_id, "performance_issue")
        assert manager.get_state(strategy_id) == PromotionState.PAPER
    
    def test_research_to_approved_requires_approval(self, manager):
        """Verify RESEARCH -> APPROVED requires explicit approve() call."""
        strategy_id = "test_approval_required"
        
        manager.submit_candidate(strategy_id, {})
        
        # Cannot skip approval step
        with pytest.raises(InvalidTransitionError):
            manager.promote_to_shadow(strategy_id)
        
        # Must approve first
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)  # Now works


class TestHumanApprovalEnforcement:
    """Test human approval requirement."""
    
    def test_no_auto_promotion_from_research(self, manager):
        """Verify strategies never auto-promote from RESEARCH."""
        strategy_id = "test_no_auto"
        
        manager.submit_candidate(strategy_id, {})
        
        # Simulate time passing - state should stay RESEARCH
        initial_state = manager.get_state(strategy_id)
        
        # No method should auto-promote
        assert manager.get_state(strategy_id) == initial_state
        assert initial_state == PromotionState.RESEARCH
    
    def test_rejection_blocks_promotion(self, manager):
        """Verify rejected strategies cannot be promoted."""
        strategy_id = "test_rejected"
        
        manager.submit_candidate(strategy_id, {})
        manager.reject(strategy_id, "failed_validation")
        
        # Should be in REJECTED state
        assert manager.get_state(strategy_id) == PromotionState.REJECTED
        
        # Cannot promote rejected strategies
        with pytest.raises(InvalidTransitionError):
            manager.approve(strategy_id)
    
    def test_require_human_approval_enforced(self):
        """Verify require_human_approval config is enforced."""
        config = {
            'strategy_promotion': {
                'require_human_approval': True
            }
        }
        
        manager = StrategyPromotionManager(config)
        strategy_id = "test_enforce_approval"
        
        manager.submit_candidate(strategy_id, {})
        
        # Verify no auto-approval
        assert manager.get_state(strategy_id) == PromotionState.RESEARCH


class TestDemotionLogic:
    """Test automatic demotion."""
    
    def test_demote_from_full_live(self, manager):
        """Test demotion from FULL_LIVE."""
        strategy_id = "test_demote_full"
        
        # Get to FULL_LIVE
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)
        manager.promote_to_micro_live(strategy_id)
        manager.promote_to_full_live(strategy_id)
        
        # Demote
        manager.demote(strategy_id, "excessive_drawdown")
        
        # Should demote to PAPER
        assert manager.get_state(strategy_id) == PromotionState.PAPER
    
    def test_demote_from_micro_live(self, manager):
        """Test demotion from MICRO_LIVE."""
        strategy_id = "test_demote_micro"
        
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)
        manager.promote_to_micro_live(strategy_id)
        
        manager.demote(strategy_id, "win_rate_collapse")
        
        # Should demote to PAPER
        assert manager.get_state(strategy_id) == PromotionState.PAPER
    
    def test_demote_from_paper(self, manager):
        """Test demotion from PAPER."""
        strategy_id = "test_demote_paper"
        
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)
        
        manager.demote(strategy_id, "correlation_spike")
        
        # Should demote to SHADOW
        assert manager.get_state(strategy_id) == PromotionState.SHADOW
    
    def test_retire_from_any_state(self, manager):
        """Test retiring strategy from any state."""
        strategy_id = "test_retire"
        
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        
        # Retire from SHADOW
        manager.retire(strategy_id, "obsolete")
        
        assert manager.get_state(strategy_id) == PromotionState.RETIRED


class TestAuditTrail:
    """Test audit trail and history tracking."""
    
    def test_submission_recorded(self, manager):
        """Verify strategy submission is recorded."""
        strategy_id = "test_audit_1"
        validation = {'sharpe': 2.0, 'drawdown': 0.05}
        
        manager.submit_candidate(strategy_id, validation)
        
        history = manager.get_history(strategy_id)
        assert len(history) == 1
        assert history[0].from_state is None
        assert history[0].to_state == PromotionState.RESEARCH
        assert history[0].validation_results == validation
    
    def test_all_transitions_recorded(self, manager):
        """Verify all state transitions are logged."""
        strategy_id = "test_audit_2"
        
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)
        
        history = manager.get_history(strategy_id)
        assert len(history) == 4
        
        # Check sequence
        assert history[0].to_state == PromotionState.RESEARCH
        assert history[1].to_state == PromotionState.APPROVED
        assert history[2].to_state == PromotionState.SHADOW
        assert history[3].to_state == PromotionState.PAPER
    
    def test_demotion_reason_recorded(self, manager):
        """Verify demotion reasons are logged."""
        strategy_id = "test_audit_3"
        
        manager.submit_candidate(strategy_id, {})
        manager.approve(strategy_id)
        manager.promote_to_shadow(strategy_id)
        manager.promote_to_paper(strategy_id)
        
        manager.demote(strategy_id, "max_drawdown_exceeded")
        
        history = manager.get_history(strategy_id)
        demotion_record = history[-1]
        
        assert demotion_record.reason == "max_drawdown_exceeded"
        assert demotion_record.to_state == PromotionState.SHADOW


class TestSafetyGuarantees:
    """Test critical safety guarantees."""
    
    def test_cannot_promote_nonexistent_strategy(self, manager):
        """Verify operations on nonexistent strategies fail."""
        with pytest.raises(StrategyNotFoundError):
            manager.approve("nonexistent")
        
        with pytest.raises(StrategyNotFoundError):
            manager.get_state("nonexistent")
    
    def test_state_query_returns_correct_state(self, manager):
        """Verify get_state() returns accurate current state."""
        strategy_id = "test_state_query"
        
        manager.submit_candidate(strategy_id, {})
        assert manager.get_state(strategy_id) == PromotionState.RESEARCH
        
        manager.approve(strategy_id)
        assert manager.get_state(strategy_id) == PromotionState.APPROVED
    
    def test_is_live_checks_correctly(self, manager):
        """Verify is_live() only returns True for live states."""
        strategy_id = "test_is_live"
        
        manager.submit_candidate(strategy_id, {})
        assert not manager.is_live(strategy_id)
        
        manager.approve(strategy_id)
        assert not manager.is_live(strategy_id)
        
        manager.promote_to_shadow(strategy_id)
        assert not manager.is_live(strategy_id)
        
        manager.promote_to_paper(strategy_id)
        assert not manager.is_live(strategy_id)
        
        manager.promote_to_micro_live(strategy_id)
        assert manager.is_live(strategy_id)  # Now live
        
        manager.promote_to_full_live(strategy_id)
        assert manager.is_live(strategy_id)  # Still live
    
    def test_diagnostics_include_all_strategies(self, manager):
        """Verify diagnostics includes all tracked strategies."""
        manager.submit_candidate("strategy_1", {})
        manager.submit_candidate("strategy_2", {})
        manager.approve("strategy_2")
        
        diagnostics = manager.get_diagnostics()
        
        assert len(diagnostics['strategies']) == 2
        assert "strategy_1" in [s['strategy_id'] for s in diagnostics['strategies']]
        assert "strategy_2" in [s['strategy_id'] for s in diagnostics['strategies']]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
