"""
Tests for PromotionGatekeeper.

Verifies:
- Automatic demotion triggers
- Demotion criteria enforcement
- No false positives
- Audit trail
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.promotion_gatekeeper import (
    PromotionGatekeeper,
    DemotionReason,
    PromotionDecision,
    DemotionCriteria
)


@pytest.fixture
def config():
    """Test configuration."""
    return {
        'strategy_promotion': {
            'auto_demote': True,
            'demotion_criteria': {
                'max_drawdown': 0.15,
                'min_expectancy': 0.0,
                'min_win_rate': 0.40,
                'max_consecutive_losses': 5,
                'min_trades_before_eval': 20,
                'max_correlation': 0.85,
                'regime_mismatch_threshold': 10
            }
        }
    }


@pytest.fixture
def gatekeeper(config):
    """Create gatekeeper instance."""
    return PromotionGatekeeper(config)


class TestDemotionTriggers:
    """Test automatic demotion triggers."""
    
    def test_excessive_drawdown_triggers_demotion(self, gatekeeper):
        """Test drawdown exceeding threshold."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.20,  # 20% > 15% threshold
            'expectancy': 1.0,
            'win_rate': 0.50,
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason == DemotionReason.EXCESSIVE_DRAWDOWN
    
    def test_negative_expectancy_triggers_demotion(self, gatekeeper):
        """Test negative expectancy."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.05,
            'expectancy': -0.5,  # Negative
            'win_rate': 0.40,
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason == DemotionReason.NEGATIVE_EXPECTANCY
    
    def test_win_rate_collapse_triggers_demotion(self, gatekeeper):
        """Test win rate below threshold."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.05,
            'expectancy': 0.5,
            'win_rate': 0.30,  # 30% < 40% threshold
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason == DemotionReason.WIN_RATE_COLLAPSE
    
    def test_consecutive_losses_triggers_demotion(self, gatekeeper):
        """Test circuit breaker on consecutive losses."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.05,
            'expectancy': 0.5,
            'win_rate': 0.50,
            'consecutive_losses': 6  # 6 >= 5 threshold
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason == DemotionReason.CONSECUTIVE_LOSSES
    
    def test_high_correlation_triggers_demotion(self, gatekeeper):
        """Test correlation spike."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.05,
            'expectancy': 0.5,
            'win_rate': 0.50,
            'consecutive_losses': 2,
            'correlation': 0.90  # 0.90 > 0.85 threshold
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason == DemotionReason.HIGH_CORRELATION
    
    def test_regime_mismatch_triggers_demotion(self, gatekeeper):
        """Test persistent regime mismatch."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.05,
            'expectancy': 0.5,
            'win_rate': 0.50,
            'consecutive_losses': 2,
            'regime_mismatches': 12  # 12 >= 10 threshold
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason == DemotionReason.REGIME_MISMATCH


class TestNonDemotionCases:
    """Test cases where demotion should NOT occur."""
    
    def test_healthy_strategy_not_demoted(self, gatekeeper):
        """Test strategy meeting all criteria."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.08,  # Below 15%
            'expectancy': 1.2,     # Positive
            'win_rate': 0.55,      # Above 40%
            'consecutive_losses': 3 # Below 5
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is False
        assert decision.reason is None
    
    def test_insufficient_trades_defers_evaluation(self, gatekeeper):
        """Test evaluation deferred when insufficient data."""
        metrics = {
            'total_trades': 10,  # < 20 minimum
            'max_drawdown': 0.50,  # Would trigger demotion
            'expectancy': -1.0,
            'win_rate': 0.20,
            'consecutive_losses': 10
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        # Should not demote yet - insufficient data
        assert decision.should_demote is False
        assert "Insufficient trades" in decision.details
    
    def test_borderline_metrics_not_demoted(self, gatekeeper):
        """Test metrics at exact thresholds."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.15,  # Exactly at threshold
            'expectancy': 0.0,      # Exactly at threshold
            'win_rate': 0.40,       # Exactly at threshold
            'consecutive_losses': 4 # Just below threshold
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        # At threshold should not demote (only exceed triggers)
        assert decision.should_demote is False


class TestBatchEvaluation:
    """Test batch evaluation of multiple strategies."""
    
    def test_evaluate_batch(self, gatekeeper):
        """Test evaluating multiple strategies at once."""
        strategies_metrics = {
            'strategy_1': {
                'total_trades': 50,
                'max_drawdown': 0.05,
                'expectancy': 1.0,
                'win_rate': 0.55,
                'consecutive_losses': 2
            },
            'strategy_2': {
                'total_trades': 50,
                'max_drawdown': 0.25,  # Exceeds threshold
                'expectancy': 0.5,
                'win_rate': 0.45,
                'consecutive_losses': 3
            },
            'strategy_3': {
                'total_trades': 50,
                'max_drawdown': 0.10,
                'expectancy': -0.2,  # Negative
                'win_rate': 0.50,
                'consecutive_losses': 2
            }
        }
        
        decisions = gatekeeper.evaluate_batch(strategies_metrics)
        
        assert len(decisions) == 3
        assert decisions['strategy_1'].should_demote is False  # Healthy
        assert decisions['strategy_2'].should_demote is True   # Drawdown
        assert decisions['strategy_3'].should_demote is True   # Expectancy
    
    def test_batch_evaluation_independence(self, gatekeeper):
        """Verify each strategy evaluated independently."""
        strategies_metrics = {
            'good': {
                'total_trades': 50,
                'max_drawdown': 0.05,
                'expectancy': 1.0,
                'win_rate': 0.55,
                'consecutive_losses': 2
            },
            'bad': {
                'total_trades': 50,
                'max_drawdown': 0.50,
                'expectancy': -1.0,
                'win_rate': 0.20,
                'consecutive_losses': 10
            }
        }
        
        decisions = gatekeeper.evaluate_batch(strategies_metrics)
        
        # Bad strategy doesn't affect good strategy
        assert decisions['good'].should_demote is False
        assert decisions['bad'].should_demote is True


class TestConfigurableCriteria:
    """Test configurable demotion criteria."""
    
    def test_custom_drawdown_threshold(self):
        """Test custom drawdown threshold."""
        config = {
            'strategy_promotion': {
                'demotion_criteria': {
                    'max_drawdown': 0.10  # Stricter than default 15%
                }
            }
        }
        
        gatekeeper = PromotionGatekeeper(config)
        
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.12,  # Would pass default, fails strict
            'expectancy': 1.0,
            'win_rate': 0.50,
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
    
    def test_custom_win_rate_threshold(self):
        """Test custom win rate threshold."""
        config = {
            'strategy_promotion': {
                'demotion_criteria': {
                    'min_win_rate': 0.50  # Require 50% instead of 40%
                }
            }
        }
        
        gatekeeper = PromotionGatekeeper(config)
        
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.05,
            'expectancy': 1.0,
            'win_rate': 0.45,  # Would pass default, fails strict
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
    
    def test_default_criteria_when_missing(self):
        """Test default criteria when config missing."""
        config = {}
        gatekeeper = PromotionGatekeeper(config)
        
        criteria = gatekeeper.get_criteria()
        
        # Should have defaults
        assert criteria.max_drawdown == 0.15
        assert criteria.min_expectancy == 0.0
        assert criteria.min_win_rate == 0.40


class TestSafetyGuarantees:
    """Test critical safety guarantees."""
    
    def test_gatekeeper_demotes_only(self, gatekeeper):
        """Verify gatekeeper can only demote, never promote."""
        # All methods should return demotion recommendations only
        # No promote() method should exist
        
        assert hasattr(gatekeeper, 'evaluate')
        assert not hasattr(gatekeeper, 'promote')
        assert not hasattr(gatekeeper, 'approve')
    
    def test_auto_demote_flag(self, gatekeeper):
        """Test auto-demotion can be disabled."""
        assert gatekeeper.should_auto_demote() is True
        
        # Test with auto_demote disabled
        config = {'strategy_promotion': {'auto_demote': False}}
        gatekeeper_disabled = PromotionGatekeeper(config)
        
        assert gatekeeper_disabled.should_auto_demote() is False
    
    def test_metrics_snapshot_captured(self, gatekeeper):
        """Verify metrics are captured in decision for audit."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.20,
            'expectancy': 0.5,
            'win_rate': 0.50,
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.metrics_snapshot is not None
        assert decision.metrics_snapshot == metrics
    
    def test_decision_includes_reason_details(self, gatekeeper):
        """Verify demotion includes detailed reason."""
        metrics = {
            'total_trades': 50,
            'max_drawdown': 0.20,
            'expectancy': 0.5,
            'win_rate': 0.50,
            'consecutive_losses': 2
        }
        
        decision = gatekeeper.evaluate("test_strategy", metrics)
        
        assert decision.should_demote is True
        assert decision.reason is not None
        assert decision.details != ""
        assert "20" in decision.details  # Should mention 20% drawdown


class TestDiagnostics:
    """Test diagnostic information."""
    
    def test_get_diagnostics(self, gatekeeper):
        """Test diagnostic output."""
        diagnostics = gatekeeper.get_diagnostics()
        
        assert 'auto_demote_enabled' in diagnostics
        assert 'criteria' in diagnostics
        
        criteria = diagnostics['criteria']
        assert 'max_drawdown' in criteria
        assert 'min_expectancy' in criteria
        assert 'min_win_rate' in criteria
    
    def test_get_criteria(self, gatekeeper):
        """Test criteria retrieval."""
        criteria = gatekeeper.get_criteria()
        
        assert isinstance(criteria, DemotionCriteria)
        assert criteria.max_drawdown == 0.15
        assert criteria.min_expectancy == 0.0
        assert criteria.min_win_rate == 0.40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
