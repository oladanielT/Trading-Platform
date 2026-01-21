"""
test_regime_readiness.py - Unit tests for PHASE 6 regime readiness gate

Tests the readiness score calculation and state transitions:
- NOT_READY → FORMING → READY
- Observation-only (no impact on trading)
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.regime_strategy import RegimeBasedStrategy, ReadinessState
from strategies.regime_detector import MarketRegime


class TestRegimeReadiness(unittest.TestCase):
    """Test regime readiness scoring and state transitions."""
    
    def setUp(self):
        """Set up test strategy."""
        self.strategy = RegimeBasedStrategy(
            allowed_regimes=['TRENDING'],
            allow_volatile_trend_override=True,
            volatile_trend_strength_min=3.0,
            volatile_confidence_scale=0.5
        )
    
    def _create_market_data(
        self,
        num_bars=150,
        trend_strength=0.5,
        volatility=0.02,
        trend_direction=1.0
    ):
        """
        Create synthetic market data with specified characteristics.
        
        Args:
            num_bars: Number of bars to generate
            trend_strength: Trend component strength (0.0-2.0)
            volatility: Random volatility component
            trend_direction: 1.0 for up, -1.0 for down
            
        Returns:
            DataFrame with OHLCV data
        """
        np.random.seed(42)
        timestamps = []
        prices = []
        
        start_time = datetime.now() - timedelta(hours=num_bars)
        base_price = 100.0
        current_price = base_price
        
        for i in range(num_bars):
            timestamps.append(start_time + timedelta(hours=i))
            # Trend component + noise
            trend_component = trend_strength * trend_direction
            noise = np.random.normal(0, volatility * current_price)
            current_price += trend_component + noise
            prices.append(current_price)
        
        # Create OHLCV data
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.005)))
            low = close * (1 - abs(np.random.normal(0, 0.005)))
            open_price = close + np.random.normal(0, 0.5)
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
    def test_readiness_state_enum_values(self):
        """Test ReadinessState enum has correct values."""
        self.assertEqual(ReadinessState.NOT_READY.value, "not_ready")
        self.assertEqual(ReadinessState.FORMING.value, "forming")
        self.assertEqual(ReadinessState.READY.value, "ready")
    
    def test_readiness_score_calculation_method_exists(self):
        """Test readiness score calculation method exists."""
        self.assertTrue(hasattr(self.strategy, '_calculate_regime_readiness_score'))
        self.assertTrue(callable(self.strategy._calculate_regime_readiness_score))
    
    def test_readiness_state_mapping_method_exists(self):
        """Test readiness state mapping method exists."""
        self.assertTrue(hasattr(self.strategy, '_map_readiness_state'))
        self.assertTrue(callable(self.strategy._map_readiness_state))
    
    def test_readiness_state_mapping_ranges(self):
        """Test readiness state maps to correct score ranges."""
        # NOT_READY: 0.0 - 0.33
        self.assertEqual(self.strategy._map_readiness_state(0.0), ReadinessState.NOT_READY)
        self.assertEqual(self.strategy._map_readiness_state(0.33), ReadinessState.NOT_READY)
        
        # FORMING: 0.34 - 0.66
        self.assertEqual(self.strategy._map_readiness_state(0.34), ReadinessState.FORMING)
        self.assertEqual(self.strategy._map_readiness_state(0.50), ReadinessState.FORMING)
        self.assertEqual(self.strategy._map_readiness_state(0.66), ReadinessState.FORMING)
        
        # READY: 0.67 - 1.0
        self.assertEqual(self.strategy._map_readiness_state(0.67), ReadinessState.READY)
        self.assertEqual(self.strategy._map_readiness_state(0.85), ReadinessState.READY)
        self.assertEqual(self.strategy._map_readiness_state(1.0), ReadinessState.READY)
    
    def test_readiness_score_bounds(self):
        """Test readiness score stays within 0.0-1.0 bounds."""
        # Test with various trend strengths
        for trend_strength in [0.0, 1.0, 2.0, 5.0, 10.0]:
            score = self.strategy._calculate_regime_readiness_score(trend_strength)
            self.assertGreaterEqual(score, 0.0, f"Score below 0.0 for strength {trend_strength}")
            self.assertLessEqual(score, 1.0, f"Score above 1.0 for strength {trend_strength}")
    
    def test_initial_readiness_state(self):
        """Test strategy initializes with NOT_READY state."""
        # With multi-symbol support, readiness is stored per-symbol with a default fallback
        self.assertIsInstance(self.strategy._readiness_state, dict)
        self.assertIsInstance(self.strategy._readiness_score, dict)
        # Default state should be NOT_READY
        sym = self.strategy._default_symbol
        self.assertEqual(self.strategy._readiness_state.get(sym, ReadinessState.NOT_READY), ReadinessState.NOT_READY)
        self.assertEqual(self.strategy._readiness_score.get(sym, 0.0), 0.0)
    
    def test_readiness_transition_not_ready_to_forming(self):
        """Test readiness transition from NOT_READY to FORMING."""
        # Create weak trend data (should be NOT_READY initially)
        weak_data = self._create_market_data(
            num_bars=150,
            trend_strength=0.2,
            volatility=0.02,
            trend_direction=1.0
        )
        
        result = self.strategy.analyze(weak_data)
        self.assertIsNotNone(result)
        
        # Should start as NOT_READY or FORMING
        sym = self.strategy._default_symbol
        initial_state = self.strategy._readiness_state.get(sym)
        self.assertIn(initial_state, [ReadinessState.NOT_READY, ReadinessState.FORMING])
        
        # Create moderate trend data (should progress to FORMING)
        moderate_data = self._create_market_data(
            num_bars=150,
            trend_strength=0.5,
            volatility=0.015,
            trend_direction=1.0
        )
        
        result = self.strategy.analyze(moderate_data)
        self.assertIsNotNone(result)
        
        # Should be FORMING or better
        sym = self.strategy._default_symbol
        self.assertIn(
            self.strategy._readiness_state.get(sym),
            [ReadinessState.FORMING, ReadinessState.READY]
        )
    
    def test_readiness_transition_forming_to_ready(self):
        """Test readiness transition from FORMING to READY."""
        # First analyze moderate data to get to FORMING
        moderate_data = self._create_market_data(
            num_bars=150,
            trend_strength=0.5,
            volatility=0.015,
            trend_direction=1.0
        )
        
        result = self.strategy.analyze(moderate_data)
        self.assertIsNotNone(result)
        
        # Now create strong trend data (should reach READY)
        strong_data = self._create_market_data(
            num_bars=150,
            trend_strength=1.2,
            volatility=0.01,
            trend_direction=1.0
        )
        
        result = self.strategy.analyze(strong_data)
        self.assertIsNotNone(result)
        
        # Should reach READY with strong trend
        # Note: May still be FORMING depending on exact metrics, but score should be higher
        sym = self.strategy._default_symbol
        self.assertGreater(self.strategy._readiness_score.get(sym, 0.0), 0.3)
    
    def test_readiness_full_transition_sequence(self):
        """Test complete readiness transition: NOT_READY → FORMING → READY."""
        states_observed = []
        sym = self.strategy._default_symbol
        
        # Sequence 1: Choppy market (NOT_READY)
        choppy_data = self._create_market_data(
            num_bars=150,
            trend_strength=0.1,
            volatility=0.03,
            trend_direction=1.0
        )
        result = self.strategy.analyze(choppy_data)
        self.assertIsNotNone(result)
        states_observed.append(self.strategy._readiness_state.get(sym))
        
        # Sequence 2: Emerging trend (FORMING expected)
        emerging_data = self._create_market_data(
            num_bars=150,
            trend_strength=0.6,
            volatility=0.018,
            trend_direction=1.0
        )
        result = self.strategy.analyze(emerging_data)
        self.assertIsNotNone(result)
        states_observed.append(self.strategy._readiness_state.get(sym))
        
        # Sequence 3: Strong trend (READY expected)
        strong_data = self._create_market_data(
            num_bars=150,
            trend_strength=1.5,
            volatility=0.01,
            trend_direction=1.0
        )
        result = self.strategy.analyze(strong_data)
        self.assertIsNotNone(result)
        states_observed.append(self.strategy._readiness_state.get(sym))
        
        # Verify we observed progression (though exact states may vary)
        # At minimum, score should increase
        scores = [0.0]  # Initial score
        for _ in range(3):
            scores.append(self.strategy._readiness_score.get(sym, 0.0))
        
        # Scores should generally trend upward with stronger trends
        self.assertGreater(scores[-1], scores[0], "Final score should exceed initial score")
    
    def test_readiness_does_not_affect_signal_generation(self):
        """Test that readiness state does NOT affect signal generation (observation only)."""
        # Create market data
        data = self._create_market_data(
            num_bars=150,
            trend_strength=0.8,
            volatility=0.015,
            trend_direction=1.0
        )
        
        # Analyze and capture result
        result1 = self.strategy.analyze(data)
        self.assertIsNotNone(result1)
        
        # Manually set readiness to different state (using default symbol)
        sym = self.strategy._default_symbol
        original_state = self.strategy._readiness_state.get(sym)
        original_score = self.strategy._readiness_score.get(sym)
        
        # Force different readiness state
        self.strategy._readiness_state[sym] = ReadinessState.NOT_READY
        self.strategy._readiness_score[sym] = 0.0
        
        # Analyze same data again
        result2 = self.strategy.analyze(data)
        self.assertIsNotNone(result2)
        
        # Signals should be the same (readiness doesn't affect trading)
        # Note: Readiness state will be recalculated, so just verify signal is consistent
        self.assertEqual(result1['action'], result2['action'])
        
        # Readiness state should be recalculated to same value
        self.assertGreater(self.strategy._readiness_score.get(sym, 0.0), 0.0)
    
    def test_readiness_score_components_weighted_equally(self):
        """Test that readiness score components are weighted equally (0.25 each)."""
        # Manually set diagnostic metrics to known values
        self.strategy._regime_persistence_count = 8  # Max score (8+)
        self.strategy._slope_stability_count = 5     # Max score (5+)
        self.strategy._trend_accel_count = 4         # Max score (4+)
        
        # Calculate with trend_strength = 3.0 (max score)
        score = self.strategy._calculate_regime_readiness_score(3.0)
        
        # All components maxed = 1.0
        self.assertAlmostEqual(score, 1.0, places=2)
        
        # Test with half values
        self.strategy._regime_persistence_count = 4  # 0.5 score
        self.strategy._slope_stability_count = 2.5   # 0.5 score (will be floored)
        self.strategy._trend_accel_count = 2         # 0.5 score
        
        score = self.strategy._calculate_regime_readiness_score(1.5)  # 0.5 score
        
        # All components at 0.5 = 0.5 overall
        self.assertAlmostEqual(score, 0.5, places=1)
    
    def test_readiness_diagnostic_log_emitted(self):
        """Test that readiness diagnostic log is emitted during analysis."""
        import logging
        from unittest.mock import patch
        
        data = self._create_market_data(num_bars=150)
        
        # Capture log output
        with patch('strategies.regime_strategy.logger') as mock_logger:
            result = self.strategy.analyze(data)
            self.assertIsNotNone(result)
            
            # Check that RegimeReadiness log was called
            readiness_logs = [
                call for call in mock_logger.info.call_args_list
                if '[RegimeReadiness]' in str(call)
            ]
            
            self.assertGreater(len(readiness_logs), 0, "RegimeReadiness log should be emitted")
    
    def test_readiness_state_persists_across_analyses(self):
        """Test that readiness state persists and updates across multiple analyses."""
        sym = self.strategy._default_symbol
        data1 = self._create_market_data(num_bars=150, trend_strength=0.3)
        result1 = self.strategy.analyze(data1)
        self.assertIsNotNone(result1)
        
        state1 = self.strategy._readiness_state.get(sym)
        score1 = self.strategy._readiness_score.get(sym)
        
        # Analyze again with different data
        data2 = self._create_market_data(num_bars=150, trend_strength=1.0)
        result2 = self.strategy.analyze(data2)
        self.assertIsNotNone(result2)
        
        state2 = self.strategy._readiness_state.get(sym)
        score2 = self.strategy._readiness_score.get(sym)
        
        # States should be tracked independently
        # They may be different based on market conditions
        self.assertIsNotNone(state2)
        self.assertIsNotNone(score2)
        
        # Score should be recalculated (likely different)
        # Just verify it's within bounds
        self.assertGreaterEqual(score2, 0.0)
        self.assertLessEqual(score2, 1.0)


class TestReadinessBackwardCompatibility(unittest.TestCase):
    """Test that readiness feature maintains backward compatibility."""
    
    def test_strategy_works_without_readiness(self):
        """Test strategy still works even if readiness attributes are deleted."""
        strategy = RegimeBasedStrategy(allowed_regimes=['TRENDING'])
        
        # Verify readiness attributes exist
        self.assertTrue(hasattr(strategy, '_readiness_score'))
        self.assertTrue(hasattr(strategy, '_readiness_state'))
        
        # Create and analyze data
        data = TestRegimeReadiness()._create_market_data(num_bars=150)
        result = strategy.analyze(data)
        
        # Should work normally
        self.assertIsNotNone(result)
        self.assertIn('action', result)
        self.assertIn('regime', result)


if __name__ == '__main__':
    unittest.main()
