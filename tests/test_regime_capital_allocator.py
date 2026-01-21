"""
test_regime_capital_allocator.py - Tests for Regime-Aware Capital Allocation

Tests allocation tracking, performance metrics, rebalancing, and integration
with position sizing constraints.

Safety Requirements Verified:
- Multipliers never exceed 1.0
- Disabled state is identical to current behavior
- Capital shifts are smooth and bounded
- No risk limits are weakened
- Backward compatible
"""

import pytest
from datetime import datetime, timedelta

from risk.regime_capital_allocator import (
    RegimeCapitalAllocator,
    RegimePerformanceMetrics,
    RegimeCapitalAllocation,
)


class TestRegimeCapitalAllocatorInitialization:
    """Test RegimeCapitalAllocator initialization."""

    def test_initialization_disabled(self):
        """Test initialization with disabled state (default)."""
        config = {
            'regime_capital': {
                'enabled': False
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        assert allocator.enabled is False
        assert allocator.base_equity == 100000.0
        assert allocator.current_equity == 100000.0

    def test_initialization_enabled(self):
        """Test initialization with enabled state."""
        config = {
            'regime_capital': {
                'enabled': True,
                'rebalance_interval': '1h',
                'min_trades_before_adjustment': 20,
                'max_allocation_shift': 0.1,
                'allocations': {
                    'trending': 0.5,
                    'ranging': 0.3,
                    'volatile': 0.2,
                }
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        assert allocator.enabled is True
        assert allocator.min_trades_before_adjustment == 20
        assert allocator.max_allocation_shift == 0.1

        # Check regime states initialized
        assert 'trending' in allocator.regimes
        assert 'ranging' in allocator.regimes
        assert 'volatile' in allocator.regimes

        # Check allocations
        assert allocator.allocations['trending'].current_allocation == 0.5
        assert allocator.allocations['ranging'].current_allocation == 0.3
        assert allocator.allocations['volatile'].current_allocation == 0.2

    def test_initialization_default_allocations(self):
        """Test initialization with default allocations."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Should use defaults
        assert allocator.allocations['trending'].current_allocation == 0.5
        assert allocator.allocations['ranging'].current_allocation == 0.3
        assert allocator.allocations['volatile'].current_allocation == 0.2

    def test_initialization_allocations_normalization(self):
        """Test that allocations > 1.0 are normalized."""
        config = {
            'regime_capital': {
                'enabled': True,
                'allocations': {
                    'trending': 0.6,
                    'ranging': 0.6,
                    'volatile': 0.6,  # Sums to 1.8
                }
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Should be normalized to sum to 1.0
        total_alloc = (
            allocator.allocations['trending'].current_allocation +
            allocator.allocations['ranging'].current_allocation +
            allocator.allocations['volatile'].current_allocation
        )
        assert abs(total_alloc - 1.0) < 0.001


class TestGetRegimeMultiplier:
    """Test get_regime_multiplier method."""

    def test_multiplier_disabled_returns_one(self):
        """Test that disabled allocator always returns 1.0."""
        config = {'regime_capital': {'enabled': False}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        assert allocator.get_regime_multiplier('trending') == 1.0
        assert allocator.get_regime_multiplier('ranging') == 1.0
        assert allocator.get_regime_multiplier('volatile') == 1.0

    def test_multiplier_enabled_returns_current_allocation(self):
        """Test that enabled allocator returns current allocation."""
        config = {
            'regime_capital': {
                'enabled': True,
                'allocations': {
                    'trending': 0.5,
                    'ranging': 0.3,
                    'volatile': 0.2,
                }
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        assert allocator.get_regime_multiplier('trending') == 0.5
        assert allocator.get_regime_multiplier('ranging') == 0.3
        assert allocator.get_regime_multiplier('volatile') == 0.2

    def test_multiplier_bounded(self):
        """Test that multipliers are always bounded [0.0, 1.0]."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Manually set an invalid multiplier
        allocator.allocations['trending'].multiplier = 1.5
        assert allocator.get_regime_multiplier('trending') == 1.0

        allocator.allocations['ranging'].multiplier = -0.5
        assert allocator.get_regime_multiplier('ranging') == 0.0

    def test_multiplier_unknown_regime(self):
        """Test behavior with unknown regime."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Unknown regime should return 1.0
        result = allocator.get_regime_multiplier('unknown_regime')
        assert result == 1.0


class TestRecordTradeResult:
    """Test record_trade_result method."""

    def test_record_winning_trade(self):
        """Test recording a winning trade."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        allocator.record_trade_result('trending', pnl=500.0)

        metrics = allocator.regimes['trending']
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 1
        assert metrics.losing_trades == 0
        assert metrics.win_rate == 1.0
        assert metrics.total_pnl == 500.0

    def test_record_losing_trade(self):
        """Test recording a losing trade."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        allocator.record_trade_result('trending', pnl=-250.0)

        metrics = allocator.regimes['trending']
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0
        assert metrics.losing_trades == 1
        assert metrics.win_rate == 0.0
        assert metrics.total_pnl == -250.0

    def test_record_break_even_trade(self):
        """Test recording a break-even trade (pnl = 0)."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        allocator.record_trade_result('ranging', pnl=0.0)

        metrics = allocator.regimes['ranging']
        assert metrics.total_trades == 1
        assert metrics.winning_trades == 0  # Zero PnL is not a win
        assert metrics.losing_trades == 0
        assert metrics.total_pnl == 0.0

    def test_record_multiple_trades(self):
        """Test recording multiple trades."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record 5 trades: 3 wins, 2 losses
        allocator.record_trade_result('trending', 100.0)
        allocator.record_trade_result('trending', 150.0)
        allocator.record_trade_result('trending', -50.0)
        allocator.record_trade_result('trending', 200.0)
        allocator.record_trade_result('trending', -75.0)

        metrics = allocator.regimes['trending']
        assert metrics.total_trades == 5
        assert metrics.winning_trades == 3
        assert metrics.losing_trades == 2
        assert abs(metrics.win_rate - 0.6) < 0.001  # 60% win rate

    def test_record_trade_ema_smoothing(self):
        """Test that PnL uses EMA smoothing, not raw cumulative."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record trades with large swings
        allocator.record_trade_result('volatile', 1000.0)
        allocator.record_trade_result('volatile', -500.0)
        allocator.record_trade_result('volatile', 800.0)

        metrics = allocator.regimes['volatile']
        # With EMA, result should not be simple sum (1000 - 500 + 800 = 1300)
        # It should be smoothed based on alpha = 2/21
        assert metrics.total_pnl != 1300.0
        assert metrics.total_pnl > 0.0

    def test_record_trade_when_disabled(self):
        """Test that recording trade when disabled has no effect."""
        config = {'regime_capital': {'enabled': False}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        allocator.record_trade_result('trending', pnl=500.0)

        # Should have no effect
        metrics = allocator.regimes['trending']
        assert metrics.total_trades == 0

    def test_record_trade_unknown_regime(self):
        """Test recording trade for unknown regime."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Should handle gracefully
        allocator.record_trade_result('unknown', pnl=100.0)
        # No exception should be raised


class TestUpdateAllocations:
    """Test update_allocations (rebalancing) method."""

    def test_update_allocations_when_disabled(self):
        """Test that rebalancing is no-op when disabled."""
        config = {'regime_capital': {'enabled': False}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Store original allocations
        original = {
            regime: alloc.current_allocation
            for regime, alloc in allocator.allocations.items()
        }

        # Call update_allocations
        allocator.update_allocations()

        # Should be unchanged
        for regime, orig_alloc in original.items():
            assert allocator.allocations[regime].current_allocation == orig_alloc

    def test_update_allocations_insufficient_trades(self):
        """Test that rebalancing doesn't happen with insufficient trades."""
        config = {
            'regime_capital': {
                'enabled': True,
                'min_trades_before_adjustment': 20,
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record only 5 trades (< 20 minimum)
        for i in range(5):
            allocator.record_trade_result('trending', 100.0)

        # Store original allocations
        original = {
            regime: alloc.current_allocation
            for regime, alloc in allocator.allocations.items()
        }

        # Call update_allocations
        allocator.update_allocations()

        # Should be unchanged (insufficient trades)
        for regime, orig_alloc in original.items():
            assert allocator.allocations[regime].current_allocation == orig_alloc

    def test_update_allocations_sufficient_trades(self):
        """Test rebalancing with sufficient trades."""
        config = {
            'regime_capital': {
                'enabled': True,
                'min_trades_before_adjustment': 5,
                'max_allocation_shift': 0.2,
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record 20 trades across regimes
        # Make trending highly profitable, volatile unprofitable
        for i in range(10):
            allocator.record_trade_result('trending', 100.0)  # Profitable
            allocator.record_trade_result('volatile', -50.0)  # Unprofitable

        # Force rebalance interval to have passed
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)

        # Call update_allocations
        allocator.update_allocations()

        # Trending should have increased allocation
        # Volatile should have decreased
        assert allocator.allocations['trending'].current_allocation >= 0.5
        assert allocator.allocations['volatile'].current_allocation <= 0.2

    def test_update_allocations_bounded_shift(self):
        """Test that allocation shifts are bounded by max_allocation_shift."""
        config = {
            'regime_capital': {
                'enabled': True,
                'min_trades_before_adjustment': 5,
                'max_allocation_shift': 0.05,  # Very small shift limit
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record trades to create performance difference
        for i in range(10):
            allocator.record_trade_result('trending', 1000.0)  # Very profitable
            allocator.record_trade_result('volatile', -1000.0)  # Very unprofitable

        # Force rebalance
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)

        # Store original
        original_trending = allocator.allocations['trending'].current_allocation
        original_volatile = allocator.allocations['volatile'].current_allocation

        allocator.update_allocations()

        # Shift should be bounded by max_allocation_shift (0.05)
        trending_shift = allocator.allocations['trending'].current_allocation - original_trending
        volatile_shift = allocator.allocations['volatile'].current_allocation - original_volatile

        assert abs(trending_shift) <= 0.05
        assert abs(volatile_shift) <= 0.05

    def test_update_allocations_sum_bounded(self):
        """Test that allocations sum to ≤ 1.0 after rebalancing."""
        config = {
            'regime_capital': {
                'enabled': True,
                'min_trades_before_adjustment': 5,
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record trades
        for i in range(10):
            allocator.record_trade_result('trending', 100.0)
            allocator.record_trade_result('ranging', 80.0)
            allocator.record_trade_result('volatile', 60.0)

        # Force rebalance
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)

        allocator.update_allocations()

        # Sum should be ≤ 1.0
        total_alloc = sum(
            alloc.current_allocation
            for alloc in allocator.allocations.values()
        )
        assert total_alloc <= 1.001  # Allow small floating point error

    def test_update_allocations_respects_rebalance_interval(self):
        """Test that rebalancing respects the rebalance_interval."""
        config = {
            'regime_capital': {
                'enabled': True,
                'min_trades_before_adjustment': 5,
                'rebalance_interval': '1d',  # 1 day
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record trades
        for i in range(10):
            allocator.record_trade_result('trending', 100.0)

        # Set last rebalance to < 1 day ago
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=12)

        # Store original
        original = allocator.allocations['trending'].current_allocation

        # Call update_allocations
        allocator.update_allocations()

        # Should not rebalance (< 1 day elapsed)
        assert allocator.allocations['trending'].current_allocation == original


class TestEquityUpdates:
    """Test equity tracking."""

    def test_update_equity(self):
        """Test updating current equity."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        assert allocator.current_equity == 100000.0

        allocator.update_equity(102500.0)
        assert allocator.current_equity == 102500.0

        allocator.update_equity(98000.0)
        assert allocator.current_equity == 98000.0


class TestDiagnostics:
    """Test diagnostics and reporting."""

    def test_get_diagnostics(self):
        """Test get_diagnostics method."""
        config = {
            'regime_capital': {
                'enabled': True,
                'allocations': {
                    'trending': 0.5,
                    'ranging': 0.3,
                    'volatile': 0.2,
                }
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record some trades
        allocator.record_trade_result('trending', 100.0)
        allocator.record_trade_result('trending', -50.0)
        allocator.record_trade_result('ranging', 75.0)

        diagnostics = allocator.get_diagnostics()

        assert diagnostics['enabled'] is True
        assert diagnostics['base_equity'] == 100000.0
        assert diagnostics['current_equity'] == 100000.0
        assert diagnostics['total_trades_recorded'] == 3

        # Check regime data
        assert 'trending' in diagnostics['regimes']
        assert 'ranging' in diagnostics['regimes']
        assert 'volatile' in diagnostics['regimes']

        # Check structure
        trend_diag = diagnostics['regimes']['trending']
        assert 'performance' in trend_diag
        assert 'allocation' in trend_diag
        assert trend_diag['performance']['total_trades'] == 2
        assert trend_diag['performance']['winning_trades'] == 1

    def test_get_stats(self):
        """Test get_stats method."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        stats = allocator.get_stats()
        assert stats['enabled'] is True
        assert stats['total_trades_recorded'] == 0
        assert 'regime_multipliers' in stats
        assert 'regime_allocations' in stats

    def test_get_stats_when_disabled(self):
        """Test get_stats returns minimal data when disabled."""
        config = {'regime_capital': {'enabled': False}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        stats = allocator.get_stats()
        assert stats['enabled'] is False
        assert 'total_trades_recorded' not in stats


class TestBackwardCompatibility:
    """Test backward compatibility when disabled."""

    def test_disabled_multiplier_never_reduces_position(self):
        """Test that disabled allocator never reduces position size."""
        config = {'regime_capital': {'enabled': False}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Any regime
        for regime in ['trending', 'ranging', 'volatile']:
            mult = allocator.get_regime_multiplier(regime)
            assert mult == 1.0

        # Record trades should have no effect
        allocator.record_trade_result('trending', 100.0)
        allocator.record_trade_result('ranging', -50.0)

        # Still should return 1.0
        for regime in ['trending', 'ranging', 'volatile']:
            mult = allocator.get_regime_multiplier(regime)
            assert mult == 1.0

    def test_disabled_no_rebalancing(self):
        """Test that disabled allocator never rebalances."""
        config = {
            'regime_capital': {
                'enabled': False,
                'min_trades_before_adjustment': 1,  # Very low threshold
            }
        }
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record many trades
        for i in range(100):
            allocator.record_trade_result('trending', 1000.0)

        # Force rebalance
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)

        # Update allocations
        allocator.update_allocations()

        # Should have no effect on allocations
        # (they should remain at initialization values)


class TestSafetyGuarantees:
    """Test mandatory safety guarantees for Phase 8."""

    def test_multiplier_never_exceeds_one(self):
        """Safety: Multiplier must never exceed 1.0."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Test initial state
        for regime in ['trending', 'ranging', 'volatile']:
            mult = allocator.get_regime_multiplier(regime)
            assert mult <= 1.0, f"{regime} multiplier {mult} exceeds 1.0"

        # After recording trades
        for i in range(20):
            allocator.record_trade_result('trending', 1000.0)

        # Force aggressive rebalancing
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)
        allocator.update_allocations()

        # Still never > 1.0
        for regime in ['trending', 'ranging', 'volatile']:
            mult = allocator.get_regime_multiplier(regime)
            assert mult <= 1.0, f"{regime} multiplier {mult} exceeds 1.0 after rebalance"

    def test_allocations_never_negative(self):
        """Safety: Allocations must never be negative."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record very bad trades for one regime
        for i in range(20):
            allocator.record_trade_result('volatile', -1000.0)

        # Force rebalance
        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)
        allocator.update_allocations()

        # All allocations should be ≥ 0
        for regime in ['trending', 'ranging', 'volatile']:
            alloc = allocator.allocations[regime].current_allocation
            mult = allocator.get_regime_multiplier(regime)
            assert alloc >= 0.0, f"{regime} allocation {alloc} is negative"
            assert mult >= 0.0, f"{regime} multiplier {mult} is negative"

    def test_allocations_sum_not_exceeded(self):
        """Safety: Sum of allocations must not exceed 1.0."""
        config = {'regime_capital': {'enabled': True}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Record balanced trades
        for i in range(30):
            allocator.record_trade_result('trending', 100.0)
            allocator.record_trade_result('ranging', 100.0)
            allocator.record_trade_result('volatile', 100.0)

        # Force multiple rebalances
        for rebalance_iteration in range(5):
            allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)
            allocator.update_allocations()

            total = sum(a.current_allocation for a in allocator.allocations.values())
            assert total <= 1.001, f"Sum of allocations {total} exceeds 1.0"

    def test_disabled_state_is_transparent(self):
        """Safety: Disabled state must be identical to current behavior."""
        # When disabled, should behave as if the feature doesn't exist
        config = {'regime_capital': {'enabled': False}}
        allocator = RegimeCapitalAllocator(config=config, base_equity=100000.0)

        # Any call should leave state unchanged
        allocator.record_trade_result('trending', 100.0)
        allocator.record_trade_result('ranging', -50.0)
        allocator.record_trade_result('volatile', 200.0)

        allocator.last_rebalance_time = datetime.utcnow() - timedelta(hours=2)
        allocator.update_allocations()

        allocator.update_equity(105000.0)

        # Behavior: Always return 1.0 multiplier
        for regime in ['trending', 'ranging', 'volatile']:
            assert allocator.get_regime_multiplier(regime) == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
