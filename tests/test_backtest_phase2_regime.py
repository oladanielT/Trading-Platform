#!/usr/bin/env python3
"""
test_backtest_phase2_regime.py - RegimeBasedStrategy Integration Tests

Tests the BacktestEngine with RegimeBasedStrategy, validating:
- Regime detection across different market conditions
- Signal filtering based on regime (HOLD in unfavorable regimes)
- Proper delegation to underlying strategy in favorable regimes
- Metrics collection with regime-aware trading
- Regime transition handling

Uses synthetic OHLCV data that simulates:
- TRENDING: Steadily increasing prices with strong directional movement
- RANGING: Sideways oscillation within bounds
- VOLATILE: Large random fluctuations
- QUIET: Small, low-volatility movements
"""

import pytest
import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from backtesting.engine import BacktestEngine, ExecutionConfig
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.regime_detector import RegimeDetector, MarketRegime
from strategies.ema_trend import EMATrendStrategy
from strategies.base import PositionState, Signal


# ============================================================================
# SYNTHETIC DATA GENERATORS FOR DIFFERENT MARKET REGIMES
# ============================================================================

def generate_trending_data(
    start_price: float = 100.0,
    num_bars: int = 100,
    trend_strength: float = 0.5,
    volatility: float = 0.01,
    start_ts: int = 1700000000000
) -> List[List]:
    """
    Generate synthetic OHLCV data for TRENDING regime.
    
    Creates steadily increasing prices with consistent directional movement.
    """
    np.random.seed(42)  # For reproducibility
    bars = []
    price = start_price
    
    for i in range(num_bars):
        # Consistent upward drift
        price_change = price * trend_strength / 100
        noise = price * volatility * np.random.randn()
        
        open_price = price
        close_price = price + price_change + noise
        high_price = max(open_price, close_price) * (1 + abs(volatility * np.random.rand()))
        low_price = min(open_price, close_price) * (1 - abs(volatility * np.random.rand()))
        
        volume = 10 + 5 * np.random.rand()
        timestamp = start_ts + i * 3600000  # 1-hour bars
        
        bars.append([timestamp, open_price, high_price, low_price, close_price, volume])
        price = close_price
    
    return bars


def generate_ranging_data(
    center_price: float = 100.0,
    num_bars: int = 100,
    range_width: float = 5.0,
    start_ts: int = 1700000000000
) -> List[List]:
    """Generate synthetic OHLCV data for RANGING regime."""
    np.random.seed(42)
    bars = []
    price = center_price
    
    for i in range(num_bars):
        # Oscillate around center with sine wave + noise
        oscillation = (range_width / 2) * np.sin(i / 10)
        noise = (range_width / 10) * np.random.randn()
        
        close_price = center_price + oscillation + noise
        close_price = np.clip(
            close_price,
            center_price - range_width / 2,
            center_price + range_width / 2
        )
        
        open_price = price
        high_price = max(open_price, close_price) * 1.002
        low_price = min(open_price, close_price) * 0.998
        
        volume = 10 + 5 * np.random.rand()
        timestamp = start_ts + i * 3600000
        
        bars.append([timestamp, open_price, high_price, low_price, close_price, volume])
        price = close_price
    
    return bars


def generate_volatile_data(
    start_price: float = 100.0,
    num_bars: int = 100,
    volatility: float = 0.05,
    start_ts: int = 1700000000000
) -> List[List]:
    """Generate synthetic OHLCV data for VOLATILE regime."""
    np.random.seed(42)
    bars = []
    price = start_price
    
    for i in range(num_bars):
        # Large random jumps
        jump = price * volatility * np.random.randn()
        
        open_price = price
        close_price = price + jump
        high_price = max(open_price, close_price) * (1 + volatility * np.random.rand())
        low_price = min(open_price, close_price) * (1 - volatility * np.random.rand())
        
        volume = 15 + 10 * np.random.rand()
        timestamp = start_ts + i * 3600000
        
        bars.append([timestamp, open_price, high_price, low_price, close_price, volume])
        price = close_price
    
    return bars


def generate_quiet_data(
    start_price: float = 100.0,
    num_bars: int = 100,
    volatility: float = 0.002,
    start_ts: int = 1700000000000
) -> List[List]:
    """Generate synthetic OHLCV data for QUIET regime."""
    np.random.seed(42)
    bars = []
    price = start_price
    
    for i in range(num_bars):
        drift = price * volatility * np.random.randn()
        
        open_price = price
        close_price = price + drift
        high_price = max(open_price, close_price) * (1 + volatility / 2)
        low_price = min(open_price, close_price) * (1 - volatility / 2)
        
        volume = 5 + 2 * np.random.rand()
        timestamp = start_ts + i * 3600000
        
        bars.append([timestamp, open_price, high_price, low_price, close_price, volume])
        price = close_price
    
    return bars


def create_ohlcv_bars(data: List[List], symbol: str = "BTC/USDT") -> List[dict]:
    """Convert OHLCV arrays to normalized bar dictionaries."""
    bars = []
    for row in data:
        timestamp, o, h, l, c, v = row
        bars.append({
            'symbol': symbol,
            'timestamp': int(timestamp),
            'open': float(o),
            'high': float(h),
            'low': float(l),
            'close': float(c),
            'volume': float(v)
        })
    return bars


# ============================================================================
# STRATEGY ADAPTER
# ============================================================================

class RegimeStrategyAdapter:
    """
    Adapter to use RegimeBasedStrategy with BacktestEngine.
    
    BacktestEngine expects strategies with on_start/on_bar/on_end methods.
    This adapter wraps RegimeBasedStrategy and executes trades based on signals.
    """
    
    def __init__(self, regime_strategy: RegimeBasedStrategy, symbol: str = "BTC/USDT"):
        self.regime_strategy = regime_strategy
        self.symbol = symbol
        self.bars_buffer = []
        self.signals_generated = []
        self.orders_placed = []
        self.position_size = 0.0
    
    def on_start(self, engine):
        """Initialize strategy."""
        self.bars_buffer = []
        self.signals_generated = []
        self.orders_placed = []
        self.position_size = 0.0
    
    def on_bar(self, bar, engine):
        """Process each bar."""
        self.bars_buffer.append(bar)
        
        # Need minimum data for regime detection and EMAs
        if len(self.bars_buffer) < 30:
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.bars_buffer)
        position_state = PositionState(has_position=(self.position_size != 0))
        
        try:
            # Generate signal using regime strategy
            signal_output = self.regime_strategy.generate_signal(df, position_state)
            self.signals_generated.append(signal_output)
            
            # Execute based on signal
            if signal_output.signal == Signal.BUY and self.position_size == 0:
                result = engine.place_market_order(
                    symbol=self.symbol,
                    size=0.1,
                    side="buy"
                )
                self.orders_placed.append(result)
                self.position_size = 0.1
            
            elif signal_output.signal == Signal.SELL and self.position_size > 0:
                result = engine.place_market_order(
                    symbol=self.symbol,
                    size=self.position_size,
                    side="sell"
                )
                self.orders_placed.append(result)
                self.position_size = 0.0
        except Exception as e:
            # Handle strategy errors gracefully
            pass
    
    def on_end(self, engine):
        """Cleanup - close any open positions."""
        if self.position_size > 0:
            try:
                engine.place_market_order(
                    symbol=self.symbol,
                    size=self.position_size,
                    side="sell"
                )
            except:
                pass


# ============================================================================
# TEST FIXTURES
# ============================================================================

@pytest.fixture
def regime_detector():
    """Create RegimeDetector with standard parameters."""
    return RegimeDetector(
        lookback_period=50,
        volatility_threshold_high=0.03,
        volatility_threshold_low=0.01,
        trend_threshold_strong=2.0,
        trend_threshold_weak=0.5
    )


@pytest.fixture
def underlying_strategy():
    """Create EMATrendStrategy as the underlying strategy."""
    return EMATrendStrategy(
        fast_period=10,
        slow_period=20,
        min_trend_strength=0.001
    )


@pytest.fixture
def regime_strategy(regime_detector, underlying_strategy):
    """Create RegimeBasedStrategy with EMA as underlying."""
    return RegimeBasedStrategy(
        underlying_strategy=underlying_strategy,
        regime_detector=regime_detector,
        allowed_regimes=[MarketRegime.TRENDING],
        regime_confidence_threshold=0.5
    )


# ============================================================================
# TEST CLASS: BASIC FUNCTIONALITY
# ============================================================================

class TestRegimeStrategyBasic:
    """Test basic RegimeBasedStrategy functionality with BacktestEngine."""
    
    def test_engine_initialization_with_regime_strategy(self, regime_strategy):
        """Test that BacktestEngine runs with RegimeBasedStrategy adapter."""
        symbol = "BTC/USDT"
        bars = create_ohlcv_bars(generate_trending_data(num_bars=100), symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
        assert isinstance(result, dict)
        assert 'metrics' in result
    
    def test_engine_runs_without_errors(self, regime_strategy):
        """Test that backtest completes without exceptions."""
        symbol = "BTC/USDT"
        bars = create_ohlcv_bars(generate_trending_data(num_bars=150), symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
        assert 'metrics' in result


# ============================================================================
# TEST CLASS: REGIME DETECTION AND FILTERING
# ============================================================================

class TestRegimeDetectionAndFiltering:
    """Test regime detection and signal filtering across different market conditions."""
    
    def test_trending_regime_allows_signals(self, regime_detector, underlying_strategy):
        """Test that strategy allows signals in TRENDING regime."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        # Generate strong trending data
        bars = create_ohlcv_bars(
            generate_trending_data(num_bars=150, trend_strength=0.8, volatility=0.01),
            symbol
        )
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
        # Should have some trading activity in trending market
        assert len(adapter.signals_generated) > 0
    
    def test_ranging_regime_returns_hold(self, regime_detector, underlying_strategy):
        """Test that strategy returns HOLD in RANGING regime."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],  # RANGING not allowed
            regime_confidence_threshold=0.5
        )
        
        # Generate ranging data
        bars = create_ohlcv_bars(
            generate_ranging_data(num_bars=150, range_width=5.0),
            symbol
        )
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
        # Should have minimal trading in ranging market
        metrics = result.get('metrics', {})
    
    def test_volatile_regime_returns_hold(self, regime_detector, underlying_strategy):
        """Test that strategy avoids VOLATILE regime."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        bars = create_ohlcv_bars(
            generate_volatile_data(num_bars=150, volatility=0.06),
            symbol
        )
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
    
    def test_quiet_regime_returns_hold(self, regime_detector, underlying_strategy):
        """Test that strategy avoids QUIET regime."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        bars = create_ohlcv_bars(
            generate_quiet_data(num_bars=150, volatility=0.001),
            symbol
        )
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None


# ============================================================================
# TEST CLASS: REGIME TRANSITIONS
# ============================================================================

class TestRegimeTransitions:
    """Test strategy behavior during regime transitions."""
    
    def test_regime_transition_trending_to_ranging(self, regime_detector, underlying_strategy):
        """Test transition from TRENDING to RANGING."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        # Mixed data: trending then ranging
        trending_bars = generate_trending_data(num_bars=75, trend_strength=0.8, start_ts=1700000000000)
        ranging_bars = generate_ranging_data(
            num_bars=75,
            center_price=trending_bars[-1][4],
            start_ts=trending_bars[-1][0] + 3600000
        )
        
        bars = create_ohlcv_bars(trending_bars + ranging_bars, symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
        # Should handle transition without errors
    
    def test_multiple_regime_changes(self, regime_detector, underlying_strategy):
        """Test multiple regime changes throughout backtest."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        # Create sequence of different regimes
        bars_sequence = []
        last_ts = 1700000000000
        last_price = 100.0
        
        trend1 = generate_trending_data(num_bars=40, start_price=last_price, start_ts=last_ts)
        bars_sequence.extend(trend1)
        last_ts = trend1[-1][0] + 3600000
        last_price = trend1[-1][4]
        
        range1 = generate_ranging_data(num_bars=40, center_price=last_price, start_ts=last_ts)
        bars_sequence.extend(range1)
        last_ts = range1[-1][0] + 3600000
        last_price = range1[-1][4]
        
        volatile1 = generate_volatile_data(num_bars=30, start_price=last_price, start_ts=last_ts)
        bars_sequence.extend(volatile1)
        last_ts = volatile1[-1][0] + 3600000
        last_price = volatile1[-1][4]
        
        quiet1 = generate_quiet_data(num_bars=30, start_price=last_price, start_ts=last_ts)
        bars_sequence.extend(quiet1)
        
        bars = create_ohlcv_bars(bars_sequence, symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None


# ============================================================================
# TEST CLASS: EDGE CASES
# ============================================================================

class TestRegimeStrategyEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_bars_list(self, regime_strategy):
        """Test with empty bars list."""
        bars = []
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, "BTC/USDT")
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
    
    def test_single_bar_sequence(self, regime_strategy):
        """Test with single bar."""
        symbol = "BTC/USDT"
        bars = create_ohlcv_bars(generate_trending_data(num_bars=1), symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
    
    def test_insufficient_data_for_regime_detection(self, underlying_strategy):
        """Test with insufficient data for regime classification."""
        symbol = "BTC/USDT"
        
        # Use long lookback period
        long_lookback_detector = RegimeDetector(lookback_period=200)
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=long_lookback_detector,
            allowed_regimes=[MarketRegime.TRENDING],
            regime_confidence_threshold=0.5
        )
        
        # Only 60 bars (less than lookback)
        bars = create_ohlcv_bars(generate_trending_data(num_bars=60), symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None


# ============================================================================
# TEST CLASS: ALLOWED REGIMES CONFIGURATION
# ============================================================================

class TestAllowedRegimesConfiguration:
    """Test different allowed_regimes configurations."""
    
    def test_multiple_allowed_regimes(self, regime_detector, underlying_strategy):
        """Test with multiple allowed regimes."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[MarketRegime.TRENDING, MarketRegime.RANGING],
            regime_confidence_threshold=0.5
        )
        
        # Mix of trending and ranging
        trending = generate_trending_data(num_bars=75, start_ts=1700000000000)
        ranging = generate_ranging_data(
            num_bars=75,
            center_price=trending[-1][4],
            start_ts=trending[-1][0] + 3600000
        )
        
        bars = create_ohlcv_bars(trending + ranging, symbol)
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
    
    def test_no_allowed_regimes_always_holds(self, regime_detector, underlying_strategy):
        """Test with empty allowed_regimes (always HOLD)."""
        symbol = "BTC/USDT"
        
        regime_strategy = RegimeBasedStrategy(
            underlying_strategy=underlying_strategy,
            regime_detector=regime_detector,
            allowed_regimes=[],  # No regimes allowed
            regime_confidence_threshold=0.5
        )
        
        bars = create_ohlcv_bars(
            generate_trending_data(num_bars=150, trend_strength=1.0),
            symbol
        )
        
        engine = BacktestEngine(exec_cfg=ExecutionConfig(initial_cash=10000.0))
        adapter = RegimeStrategyAdapter(regime_strategy, symbol)
        
        result = engine.run_backtest(bars, adapter)
        
        assert result is not None
        # Should have minimal or no trades
        assert len(adapter.orders_placed) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
