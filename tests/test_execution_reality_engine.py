"""
test_execution_reality_engine.py - Unit tests for ExecutionRealityEngine

Tests verify:
- Size never increases
- Zero-size orders are rejected safely
- Slippage grows with size and volatility
- Disabled state produces no behavioral change
- Fill probability estimation is reasonable
- Execution risk assessment is accurate
"""

import pytest
import math
from execution.execution_reality_engine import ExecutionRealityEngine, ExecutionMetrics


class TestExecutionRealityEngineInitialization:
    """Tests for engine initialization and configuration."""
    
    def test_engine_disabled_by_default(self):
        """Verify engine is disabled by default."""
        config = {'enabled': False}
        engine = ExecutionRealityEngine(config)
        assert not engine.enabled
    
    def test_engine_can_be_enabled(self):
        """Verify engine can be enabled via config."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        assert engine.enabled
    
    def test_default_parameters(self):
        """Verify default configuration parameters."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        assert engine.max_size_vs_volume == 0.02
        assert engine.min_fill_probability == 0.6
        assert engine.slippage_model == 'conservative'
        assert engine.latency_penalty_ms == 500
    
    def test_custom_parameters(self):
        """Verify custom configuration parameters."""
        config = {
            'enabled': True,
            'max_size_vs_volume': 0.05,
            'min_fill_probability': 0.7,
            'slippage_model': 'moderate',
            'latency_penalty_ms': 1000
        }
        engine = ExecutionRealityEngine(config)
        assert engine.max_size_vs_volume == 0.05
        assert engine.min_fill_probability == 0.7
        assert engine.slippage_model == 'moderate'
        assert engine.latency_penalty_ms == 1000


class TestExecutionRealityEngineDisabledMode:
    """Tests for disabled mode behavior."""
    
    def test_disabled_mode_passes_through_unchanged(self):
        """Verify disabled mode produces no size reductions."""
        config = {'enabled': False}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'high': 105.0,
            'low': 95.0,
            'volume': 1000.0,
        }
        
        result = engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=10.0,
            market_snapshot=market_snapshot
        )
        
        # In disabled mode: full size approved, no slippage, fill probability = 1.0
        assert result['approved_size'] == 10.0
        assert result['expected_slippage_bps'] == 0.0
        assert result['fill_probability'] == 1.0
        assert result['execution_risk'] == 'low'
        assert result['rejection_reason'] is None
        assert result['constraints']['disabled'] is True
    
    def test_disabled_mode_metrics_unchanged(self):
        """Verify disabled mode doesn't track metrics."""
        config = {'enabled': False}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 1000.0,
        }
        
        # Multiple evaluations
        for _ in range(5):
            engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=1.0,
                market_snapshot=market_snapshot
            )
        
        # No metric change in disabled mode
        assert engine.metrics.total_orders_evaluated == 0


class TestExecutionRealityEngineBasicFunctionality:
    """Tests for basic order evaluation."""
    
    def test_small_order_approved_in_liquid_market(self):
        """Verify small orders approved in liquid markets."""
        config = {'enabled': True, 'max_size_vs_volume': 0.02}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'high': 100.5,
            'low': 99.5,
            'volume': 10000.0,
            'atr_percent': 0.5,
            'bid_ask_spread_pct': 0.01,
        }
        
        # Small order: 1% of volume
        result = engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=1.0,  # 100 / 10000 = 1%
            market_snapshot=market_snapshot
        )
        
        # Should be approved with minimal size reduction
        assert result['approved_size'] > 0.9  # At least 90% approved
        assert result['fill_probability'] > 0.8
        assert result['execution_risk'] == 'low'
    
    def test_oversized_order_rejected_or_reduced(self):
        """Verify oversized orders are reduced."""
        config = {'enabled': True, 'max_size_vs_volume': 0.02}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 1000.0,
            'atr_percent': 0.5,
        }
        
        # Oversized order: 3% of volume (exceeds 2% limit)
        result = engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=3.0,  # 300 / 10000 = 3%
            market_snapshot=market_snapshot
        )
        
        # Should be reduced to stay within 2% of volume
        approved = result['approved_size']
        approved_vs_volume = (approved * market_snapshot['close']) / market_snapshot['volume']
        assert approved_vs_volume <= 0.02 * 1.001  # Allow 0.1% rounding
    
    def test_size_never_increases(self):
        """Verify size never increases (CRITICAL SAFETY TEST)."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshots = [
            {
                'close': 100.0,
                'high': 110.0,
                'low': 90.0,
                'volume': 100000.0,
                'atr_percent': 2.0,
            },
            {
                'close': 50.0,
                'high': 60.0,
                'low': 40.0,
                'volume': 500.0,
                'atr_percent': 0.1,
            },
            {
                'close': 1000.0,
                'high': 1500.0,
                'low': 500.0,
                'volume': 10.0,
                'atr_percent': 50.0,
            },
        ]
        
        requested_sizes = [1.0, 10.0, 0.01, 100.0]
        
        for snapshot in market_snapshots:
            for requested_size in requested_sizes:
                result = engine.evaluate_order(
                    symbol='BTC/USDT',
                    side='buy',
                    requested_size=requested_size,
                    market_snapshot=snapshot
                )
                
                # CRITICAL: Approved size must be <= requested size
                assert result['approved_size'] <= requested_size * 1.0001, \
                    f"Size increased! requested={requested_size}, approved={result['approved_size']}"
                assert result['approved_size'] >= 0, \
                    f"Size is negative! {result['approved_size']}"


class TestSlippageEstimation:
    """Tests for slippage estimation."""
    
    def test_larger_orders_have_more_slippage(self):
        """Verify larger orders incur more slippage."""
        config = {'enabled': True, 'slippage_model': 'conservative'}
        engine = ExecutionRealityEngine(config)
        
        # Fixed market conditions
        market_snapshot_base = {
            'close': 100.0,
            'high': 100.5,
            'low': 99.5,
            'volume': 100000.0,
            'atr_percent': 1.0,
            'bid_ask_spread_pct': 0.01,
        }
        
        slippages = []
        for size_vs_volume in [0.001, 0.005, 0.01, 0.02]:
            market_snapshot = market_snapshot_base.copy()
            
            result = engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=(size_vs_volume * market_snapshot['volume']) / market_snapshot['close'],
                market_snapshot=market_snapshot
            )
            
            slippages.append(result['expected_slippage_bps'])
        
        # Slippage should increase with size
        for i in range(len(slippages) - 1):
            assert slippages[i] < slippages[i + 1], \
                f"Slippage did not increase: {slippages[i]} >= {slippages[i+1]}"
    
    def test_higher_volatility_increases_slippage(self):
        """Verify higher volatility increases slippage."""
        config = {'enabled': True, 'slippage_model': 'conservative'}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot_base = {
            'close': 100.0,
            'volume': 10000.0,
            'bid_ask_spread_pct': 0.01,
        }
        
        slippages = []
        for atr_percent in [0.1, 0.5, 1.0, 2.0, 5.0]:
            market_snapshot = market_snapshot_base.copy()
            market_snapshot['atr_percent'] = atr_percent
            market_snapshot['high'] = 100.0 + (atr_percent * 100 / 100)
            market_snapshot['low'] = 100.0 - (atr_percent * 100 / 100)
            
            result = engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=0.1,
                market_snapshot=market_snapshot
            )
            
            slippages.append(result['expected_slippage_bps'])
        
        # Slippage should increase with volatility
        for i in range(len(slippages) - 1):
            assert slippages[i] < slippages[i + 1], \
                f"Slippage did not increase with volatility: {slippages[i]} >= {slippages[i+1]}"
    
    def test_slippage_models_differ(self):
        """Verify different slippage models produce different results."""
        models = ['conservative', 'moderate', 'aggressive']
        engines = {m: ExecutionRealityEngine({'enabled': True, 'slippage_model': m}) for m in models}
        
        market_snapshot = {
            'close': 100.0,
            'volume': 1000.0,
            'atr_percent': 1.0,
            'bid_ask_spread_pct': 0.01,
        }
        
        slippages = {}
        for model, engine in engines.items():
            result = engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=0.2,
                market_snapshot=market_snapshot
            )
            slippages[model] = result['expected_slippage_bps']
        
        # Conservative should have highest slippage
        assert slippages['conservative'] > slippages['moderate']
        assert slippages['moderate'] > slippages['aggressive']


class TestFillProbability:
    """Tests for fill probability estimation."""
    
    def test_fill_probability_decreases_with_size(self):
        """Verify fill probability decreases with order size."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot_base = {
            'close': 100.0,
            'volume': 100000.0,
            'atr_percent': 1.0,
        }
        
        fill_probs = []
        for size_vs_volume in [0.001, 0.005, 0.01, 0.015, 0.02]:
            market_snapshot = market_snapshot_base.copy()
            
            result = engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=(size_vs_volume * market_snapshot['volume']) / market_snapshot['close'],
                market_snapshot=market_snapshot
            )
            
            fill_probs.append(result['fill_probability'])
        
        # Fill probability should decrease with size
        for i in range(len(fill_probs) - 1):
            assert fill_probs[i] > fill_probs[i + 1], \
                f"Fill prob did not decrease: {fill_probs[i]} <= {fill_probs[i+1]}"
    
    def test_fill_probability_bounded(self):
        """Verify fill probability is always between 0 and 1."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshots = [
            {
                'close': 100.0,
                'volume': 10.0,  # Very low volume
                'atr_percent': 50.0,  # Very high volatility
            },
            {
                'close': 100.0,
                'volume': 100000.0,
                'atr_percent': 0.01,
            },
        ]
        
        for snapshot in market_snapshots:
            result = engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=1.0,
                market_snapshot=snapshot
            )
            
            assert 0.0 <= result['fill_probability'] <= 1.0, \
                f"Fill probability out of bounds: {result['fill_probability']}"


class TestExecutionRisk:
    """Tests for execution risk assessment."""
    
    def test_low_risk_in_favorable_conditions(self):
        """Verify low risk rating in favorable conditions."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 100000.0,
            'atr_percent': 0.5,  # Low volatility
            'bid_ask_spread_pct': 0.01,
        }
        
        result = engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=0.1,  # Small order
            market_snapshot=market_snapshot
        )
        
        assert result['execution_risk'] == 'low'
    
    def test_high_risk_in_adverse_conditions(self):
        """Verify high risk rating in adverse conditions."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 100.0,  # Very low volume
            'atr_percent': 15.0,  # High volatility
            'bid_ask_spread_pct': 0.1,
        }
        
        result = engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=1.0,  # Large order relative to volume
            market_snapshot=market_snapshot
        )
        
        assert result['execution_risk'] == 'high'


class TestMetricsTracking:
    """Tests for execution metrics tracking."""
    
    def test_metrics_initialized(self):
        """Verify metrics are initialized."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        assert engine.metrics.total_orders_evaluated == 0
        assert engine.metrics.total_orders_approved == 0
        assert engine.metrics.total_orders_rejected == 0
    
    def test_metrics_updated_on_evaluation(self):
        """Verify metrics are updated during evaluation."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 10000.0,
        }
        
        engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=1.0,
            market_snapshot=market_snapshot
        )
        
        assert engine.metrics.total_orders_evaluated == 1
        assert engine.metrics.total_orders_approved >= 0
        assert engine.metrics.total_orders_rejected >= 0
    
    def test_approval_rate_calculation(self):
        """Verify approval rate metrics."""
        config = {'enabled': True, 'min_fill_probability': 0.99}  # High threshold
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 100.0,  # Very small volume
            'atr_percent': 10.0,
        }
        
        # This should have low fill probability
        for _ in range(10):
            engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=1.0,  # Oversized
                market_snapshot=market_snapshot
            )
        
        diagnostics = engine.get_diagnostics()
        assert diagnostics['enabled'] is True
        assert 'metrics' in diagnostics
        assert diagnostics['metrics']['total_orders_evaluated'] == 10


class TestDiagnostics:
    """Tests for diagnostics output."""
    
    def test_diagnostics_disabled_mode(self):
        """Verify diagnostics show disabled state."""
        config = {'enabled': False}
        engine = ExecutionRealityEngine(config)
        
        diagnostics = engine.get_diagnostics()
        assert diagnostics['enabled'] is False
        assert diagnostics['status'] == 'disabled'
    
    def test_diagnostics_enabled_mode(self):
        """Verify diagnostics show enabled state."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 10000.0,
        }
        
        engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=1.0,
            market_snapshot=market_snapshot
        )
        
        diagnostics = engine.get_diagnostics()
        assert diagnostics['enabled'] is True
        assert diagnostics['status'] == 'active'
        assert diagnostics['metrics']['total_orders_evaluated'] == 1
    
    def test_diagnostics_includes_symbol_health(self):
        """Verify diagnostics include per-symbol stats."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        market_snapshot = {
            'close': 100.0,
            'volume': 10000.0,
        }
        
        engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=1.0,
            market_snapshot=market_snapshot
        )
        
        engine.evaluate_order(
            symbol='ETH/USDT',
            side='buy',
            requested_size=10.0,
            market_snapshot=market_snapshot
        )
        
        diagnostics = engine.get_diagnostics()
        assert 'symbol_health' in diagnostics
        assert 'BTC/USDT' in diagnostics['symbol_health']
        assert 'ETH/USDT' in diagnostics['symbol_health']


class TestExecutionResultRecording:
    """Tests for recording actual execution results."""
    
    def test_record_execution_result_disabled_mode(self):
        """Verify recording is no-op in disabled mode."""
        config = {'enabled': False}
        engine = ExecutionRealityEngine(config)
        
        # Should not raise or do anything
        engine.record_execution_result(
            symbol='BTC/USDT',
            requested_size=1.0,
            filled_size=1.0,
            slippage_bps=5.0,
            latency_ms=100
        )
    
    def test_record_execution_result_enabled_mode(self):
        """Verify recording updates stats in enabled mode."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        # Pre-populate stats
        engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=1.0,
            market_snapshot={'close': 100.0, 'volume': 10000.0}
        )
        
        # Record actual execution
        engine.record_execution_result(
            symbol='BTC/USDT',
            requested_size=1.0,
            filled_size=0.99,  # Partial fill
            slippage_bps=8.0,
            latency_ms=150
        )
        
        # Verify stats exist
        assert 'BTC/USDT' in engine.metrics.symbol_stats


class TestIntegrationSafety:
    """Integration tests to verify safety guarantees."""
    
    def test_no_leverage_increase(self):
        """Verify engine never increases leverage."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        for _ in range(100):
            requested_size = 10.0
            result = engine.evaluate_order(
                symbol='BTC/USDT',
                side='buy',
                requested_size=requested_size,
                market_snapshot={
                    'close': 100.0,
                    'volume': 10000.0,
                }
            )
            
            assert result['approved_size'] <= requested_size * 1.0001
    
    def test_zero_size_skips_safely(self):
        """Verify zero-size orders are safely skipped."""
        config = {'enabled': True}
        engine = ExecutionRealityEngine(config)
        
        result = engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=0.0,
            market_snapshot={
                'close': 100.0,
                'volume': 10000.0,
            }
        )
        
        assert result['approved_size'] == 0.0
    
    def test_disabled_identical_to_passthrough(self):
        """Verify disabled mode is truly transparent."""
        disabled_config = {'enabled': False}
        disabled_engine = ExecutionRealityEngine(disabled_config)
        
        market_snapshot = {
            'close': 100.0,
            'high': 105.0,
            'low': 95.0,
            'volume': 10000.0,
            'atr_percent': 2.0,
            'bid_ask_spread_pct': 0.05,
        }
        
        result = disabled_engine.evaluate_order(
            symbol='BTC/USDT',
            side='buy',
            requested_size=5.0,
            market_snapshot=market_snapshot
        )
        
        # Disabled mode should be 100% passthrough
        assert result['approved_size'] == 5.0
        assert result['expected_slippage_bps'] == 0.0
        assert result['fill_probability'] == 1.0
        assert result['execution_risk'] == 'low'
        assert result['rejection_reason'] is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
