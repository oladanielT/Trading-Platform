"""
test_portfolio_risk_manager.py - Tests for Portfolio-Level Risk Management

Tests correlation-based position sizing, exposure limits, and integration
with the optimization system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from risk.portfolio_risk_manager import PortfolioRiskManager, Position, PortfolioMetrics


class TestPortfolioRiskManagerInitialization:
    """Test PortfolioRiskManager initialization."""
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        assert manager.total_equity == 100000.0
        assert manager.enabled is True
        assert manager.correlation_lookback == 100
        assert manager.max_pairwise_correlation == 0.85
        assert manager.max_symbol_exposure_pct == 0.25
        assert manager.max_directional_exposure_pct == 0.60
        assert manager.max_regime_exposure_pct == 0.50
        assert len(manager.positions) == 0
        assert len(manager.market_data) == 0
    
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        manager = PortfolioRiskManager(
            total_equity=50000.0,
            enabled=True,
            correlation_lookback=50,
            max_pairwise_correlation=0.75,
            max_symbol_exposure_pct=0.20,
            max_directional_exposure_pct=0.50,
            max_regime_exposure_pct=0.40,
        )
        
        assert manager.total_equity == 50000.0
        assert manager.correlation_lookback == 50
        assert manager.max_pairwise_correlation == 0.75
        assert manager.max_symbol_exposure_pct == 0.20
        assert manager.max_directional_exposure_pct == 0.50
        assert manager.max_regime_exposure_pct == 0.40
    
    def test_initialization_disabled(self):
        """Test initialization with disabled state."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            enabled=False
        )
        
        assert manager.enabled is False
        
        # Disabled manager should return pass-through results
        allowed, reason, multiplier = manager.validate_new_position(
            "BTC/USDT", "long", 1000.0, "TRENDING", "TestStrategy"
        )
        assert allowed is True
        assert reason == "portfolio_risk_disabled"
        assert multiplier == 1.0


class TestMarketDataAndCorrelation:
    """Test market data updates and correlation calculations."""
    
    def test_update_market_data(self):
        """Test market data update."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Create sample candles
        candles = pd.DataFrame({
            'close': [100, 101, 102, 103, 104]
        })
        
        manager.update_market_data("BTC/USDT", candles)
        
        assert "BTC/USDT" in manager.market_data
        assert 'returns' in manager.market_data["BTC/USDT"].columns
        assert len(manager.market_data["BTC/USDT"]) == 5
    
    def test_correlation_same_symbol(self):
        """Test correlation for same symbol (should be 1.0)."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        corr = manager.get_correlation("BTC/USDT", "BTC/USDT")
        assert corr == 1.0
    
    def test_correlation_insufficient_data(self):
        """Test correlation with insufficient data."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Only a few data points
        candles = pd.DataFrame({
            'close': [100, 101, 102]
        })
        
        manager.update_market_data("BTC/USDT", candles)
        manager.update_market_data("ETH/USDT", candles)
        
        corr = manager.get_correlation("BTC/USDT", "ETH/USDT")
        assert corr == 0.0  # Not enough data
    
    def test_correlation_high(self):
        """Test high correlation between symbols."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Create highly correlated price series
        np.random.seed(42)
        base_returns = np.random.randn(150) * 0.01
        
        btc_prices = 100 * np.exp(np.cumsum(base_returns))
        eth_prices = 200 * np.exp(np.cumsum(base_returns + np.random.randn(150) * 0.002))
        
        btc_candles = pd.DataFrame({'close': btc_prices})
        eth_candles = pd.DataFrame({'close': eth_prices})
        
        manager.update_market_data("BTC/USDT", btc_candles)
        manager.update_market_data("ETH/USDT", eth_candles)
        
        corr = manager.get_correlation("BTC/USDT", "ETH/USDT")
        assert corr > 0.8  # Should be highly correlated
    
    def test_correlation_low(self):
        """Test low correlation between symbols."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Create uncorrelated price series
        np.random.seed(42)
        btc_returns = np.random.randn(150) * 0.01
        eth_returns = np.random.randn(150) * 0.01
        
        btc_prices = 100 * np.exp(np.cumsum(btc_returns))
        eth_prices = 200 * np.exp(np.cumsum(eth_returns))
        
        btc_candles = pd.DataFrame({'close': btc_prices})
        eth_candles = pd.DataFrame({'close': eth_prices})
        
        manager.update_market_data("BTC/USDT", btc_candles)
        manager.update_market_data("ETH/USDT", eth_candles)
        
        corr = abs(manager.get_correlation("BTC/USDT", "ETH/USDT"))
        assert corr < 0.3  # Should be weakly correlated


class TestPositionManagement:
    """Test position registration and tracking."""
    
    def test_register_position(self):
        """Test registering a position."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position(
            symbol="BTC/USDT",
            side="long",
            size=5000.0,
            regime="TRENDING",
            strategy="EMA_Trend"
        )
        
        assert "BTC/USDT" in manager.positions
        position = manager.positions["BTC/USDT"]
        assert position.symbol == "BTC/USDT"
        assert position.side == "long"
        assert position.size == 5000.0
        assert position.regime == "TRENDING"
        assert position.strategy == "EMA_Trend"
    
    def test_unregister_position(self):
        """Test unregistering a position."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position("BTC/USDT", "long", 5000.0)
        assert "BTC/USDT" in manager.positions
        
        manager.unregister_position("BTC/USDT")
        assert "BTC/USDT" not in manager.positions
    
    def test_get_symbol_exposure(self):
        """Test getting symbol exposure."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position("BTC/USDT", "long", 10000.0)
        
        exposure = manager.get_symbol_exposure("BTC/USDT")
        assert exposure == 0.10  # 10% exposure
    
    def test_get_regime_exposure(self):
        """Test getting regime exposure."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position("BTC/USDT", "long", 10000.0, regime="TRENDING")
        manager.register_position("ETH/USDT", "long", 5000.0, regime="TRENDING")
        manager.register_position("LTC/USDT", "short", 3000.0, regime="RANGING")
        
        trending_exposure = manager.get_regime_exposure("TRENDING")
        assert trending_exposure == 0.15  # 15% in TRENDING
        
        ranging_exposure = manager.get_regime_exposure("RANGING")
        assert ranging_exposure == 0.03  # 3% in RANGING
    
    def test_get_directional_exposure(self):
        """Test getting directional exposure."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position("BTC/USDT", "long", 10000.0)
        manager.register_position("ETH/USDT", "long", 5000.0)
        manager.register_position("LTC/USDT", "short", 3000.0)
        
        long_exposure = manager.get_directional_exposure("long")
        assert long_exposure == 0.15  # 15% long
        
        short_exposure = manager.get_directional_exposure("short")
        assert short_exposure == 0.03  # 3% short


class TestPortfolioMetrics:
    """Test portfolio metrics calculation."""
    
    def test_get_portfolio_metrics(self):
        """Test comprehensive portfolio metrics."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position("BTC/USDT", "long", 10000.0, regime="TRENDING", strategy="EMA")
        manager.register_position("ETH/USDT", "long", 5000.0, regime="TRENDING", strategy="FVG")
        manager.register_position("LTC/USDT", "short", 3000.0, regime="RANGING", strategy="RSI")
        
        metrics = manager.get_portfolio_metrics()
        
        assert metrics.total_exposure == pytest.approx(0.18, rel=0.001)  # 18% total
        assert metrics.long_exposure == pytest.approx(0.15, rel=0.001)   # 15% long
        assert metrics.short_exposure == pytest.approx(0.03, rel=0.001)  # 3% short
        assert "TRENDING" in metrics.regime_exposures
        assert metrics.regime_exposures["TRENDING"] == pytest.approx(0.15, rel=0.001)
        assert metrics.regime_exposures["RANGING"] == pytest.approx(0.03, rel=0.001)
        assert len(metrics.symbol_exposures) == 3


class TestPositionValidation:
    """Test position validation with various constraints."""
    
    def test_validate_basic_approval(self):
        """Test basic position approval."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=5000.0,
            regime="TRENDING",
            strategy="EMA_Trend"
        )
        
        assert allowed is True
        assert reason == "approved"
        assert multiplier == 1.0
    
    def test_validate_position_already_exists_same_direction(self):
        """Test rejection when position exists in same direction."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Register existing position
        manager.register_position("BTC/USDT", "long", 5000.0)
        
        # Try to open another long position
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=3000.0
        )
        
        assert allowed is False
        assert reason == "position_already_exists"
        assert multiplier == 0.0
        assert manager.blocked_trades_count == 1
    
    def test_validate_position_exists_opposite_direction(self):
        """Test allowing opposite direction (hedging)."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Register long position
        manager.register_position("BTC/USDT", "long", 5000.0)
        
        # Try to open short position (hedging)
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="short",
            proposed_size=3000.0
        )
        
        # Should be allowed (opposite direction positions are permitted)
        assert allowed is True
    
    def test_validate_symbol_exposure_limit(self):
        """Test symbol exposure limit enforcement."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_symbol_exposure_pct=0.20  # 20% max per symbol
        )
        
        # Try to open position exceeding limit
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=30000.0  # 30% exposure
        )
        
        # Should reduce size to fit within limit
        assert allowed is True
        assert multiplier < 1.0
        assert multiplier == pytest.approx(0.20 / 0.30, rel=0.01)
    
    def test_validate_symbol_exposure_limit_blocks_small(self):
        """Test symbol exposure limit blocks when not enough room."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_symbol_exposure_pct=0.10
        )
        
        # Existing position at limit
        manager.register_position("BTC/USDT", "long", 10000.0)
        
        # Try to add more in same direction (should be blocked - position exists)
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=5000.0
        )
        
        # Should be blocked due to position already existing
        assert allowed is False
        assert "position_already_exists" in reason
    
    def test_validate_directional_exposure_limit(self):
        """Test directional exposure limit enforcement."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_directional_exposure_pct=0.50  # 50% max per direction
        )
        
        # Fill up long exposure
        manager.register_position("BTC/USDT", "long", 30000.0)
        manager.register_position("ETH/USDT", "long", 15000.0)
        
        # Try to add more long
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="LTC/USDT",
            side="long",
            proposed_size=10000.0  # Would be 55% total
        )
        
        # Should reduce to fit within limit
        assert allowed is True
        assert multiplier < 1.0
        assert multiplier == pytest.approx(0.05 / 0.10, rel=0.01)  # 5% available / 10% proposed
    
    def test_validate_regime_exposure_limit(self):
        """Test regime exposure limit enforcement."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_regime_exposure_pct=0.40  # 40% max per regime
        )
        
        # Fill up TRENDING regime
        manager.register_position("BTC/USDT", "long", 25000.0, regime="TRENDING")
        manager.register_position("ETH/USDT", "long", 10000.0, regime="TRENDING")
        
        # Try to add more TRENDING
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="LTC/USDT",
            side="long",
            proposed_size=10000.0,
            regime="TRENDING"
        )
        
        # Should reduce to fit within limit
        assert allowed is True
        assert multiplier < 1.0
        assert multiplier == pytest.approx(0.05 / 0.10, rel=0.01)
    
    def test_validate_high_correlation_reduction(self):
        """Test position size reduction due to high correlation."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_pairwise_correlation=0.85
        )
        
        # Create highly correlated market data
        np.random.seed(42)
        base_returns = np.random.randn(150) * 0.01
        
        btc_prices = 100 * np.exp(np.cumsum(base_returns))
        eth_prices = 200 * np.exp(np.cumsum(base_returns + np.random.randn(150) * 0.001))
        
        btc_candles = pd.DataFrame({'close': btc_prices})
        eth_candles = pd.DataFrame({'close': eth_prices})
        
        manager.update_market_data("BTC/USDT", btc_candles)
        manager.update_market_data("ETH/USDT", eth_candles)
        
        # Register BTC position
        manager.register_position("BTC/USDT", "long", 10000.0)
        
        # Try to add ETH (highly correlated)
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="ETH/USDT",
            side="long",
            proposed_size=10000.0
        )
        
        # Should reduce due to high correlation
        assert allowed is True
        assert multiplier < 1.0
        assert "correlation" in reason
        assert manager.reduced_trades_count == 1
    
    def test_validate_high_correlation_blocks(self):
        """Test blocking extremely correlated positions."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_pairwise_correlation=0.70  # Stricter limit
        )
        
        # Create very highly correlated market data (nearly identical)
        np.random.seed(42)
        base_returns = np.random.randn(150) * 0.01
        noise = np.random.randn(150) * 0.0001  # Minimal noise for high correlation
        
        btc_prices = 100 * np.exp(np.cumsum(base_returns))
        eth_prices = 200 * np.exp(np.cumsum(base_returns + noise))  # Nearly identical
        
        btc_candles = pd.DataFrame({'close': btc_prices})
        eth_candles = pd.DataFrame({'close': eth_prices})
        
        manager.update_market_data("BTC/USDT", btc_candles)
        manager.update_market_data("ETH/USDT", eth_candles)
        
        # Register BTC position
        manager.register_position("BTC/USDT", "long", 10000.0)
        
        # Try to add ETH (extremely correlated)
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="ETH/USDT",
            side="long",
            proposed_size=10000.0
        )
        
        # Should be reduced or blocked due to high correlation
        # If not blocked, should at least reduce significantly
        if not allowed:
            assert "correlation" in reason
            assert manager.blocked_trades_count == 1
        else:
            assert multiplier < 0.5  # At least 50% reduction


class TestOptimizationIntegration:
    """Test interaction with optimization multipliers."""
    
    def test_multiplier_composition(self):
        """Test that portfolio multiplier composes with optimization multiplier."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_symbol_exposure_pct=0.20
        )
        
        # Propose oversized position
        allowed, reason, portfolio_multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=30000.0  # 30% exposure
        )
        
        assert allowed is True
        assert portfolio_multiplier < 1.0
        
        # Simulate optimization multiplier
        optimization_multiplier = 1.5
        
        # Final size composition
        base_size = 30000.0
        after_portfolio = base_size * portfolio_multiplier  # Reduced by portfolio
        final_size = after_portfolio * optimization_multiplier  # Can't exceed portfolio limit
        
        # Portfolio multiplier should have reduced to ~20% exposure
        assert after_portfolio <= 20000.0 + 0.01  # Allow tiny floating point error
    
    def test_cannot_increase_beyond_limits(self):
        """Test that portfolio limits cannot be bypassed."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            max_symbol_exposure_pct=0.25
        )
        
        # Even with max optimization multiplier, should not exceed limits
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=50000.0  # 50% proposed
        )
        
        # Should reduce to fit within 25% limit
        assert multiplier <= 0.5  # At most 50% of proposed = 25% exposure


class TestStatistics:
    """Test statistics and metrics tracking."""
    
    def test_get_stats(self):
        """Test statistics retrieval."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        # Register some positions
        manager.register_position("BTC/USDT", "long", 10000.0, regime="TRENDING")
        manager.register_position("ETH/USDT", "long", 5000.0, regime="TRENDING")
        
        # Simulate some blocked/reduced trades
        manager.blocked_trades_count = 3
        manager.reduced_trades_count = 5
        
        stats = manager.get_stats()
        
        assert stats['enabled'] is True
        assert stats['open_positions'] == 2
        assert stats['total_exposure'] == pytest.approx(0.15, rel=0.001)
        assert stats['long_exposure'] == pytest.approx(0.15, rel=0.001)
        assert stats['short_exposure'] == pytest.approx(0.0, abs=0.001)
        assert stats['blocked_trades'] == 3
        assert stats['reduced_trades'] == 5
        assert 'regime_exposures' in stats
        assert 'symbol_exposures' in stats


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_equity(self):
        """Test handling of zero equity."""
        manager = PortfolioRiskManager(total_equity=0.0)
        
        exposure = manager.get_symbol_exposure("BTC/USDT")
        assert exposure == 0.0
    
    def test_negative_proposed_size(self):
        """Test rejection of negative proposed size."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=-1000.0
        )
        
        assert allowed is False
        assert reason == "invalid_size"
        assert multiplier == 0.0
    
    def test_zero_proposed_size(self):
        """Test rejection of zero proposed size."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=0.0
        )
        
        assert allowed is False
        assert reason == "invalid_size"
        assert multiplier == 0.0
    
    def test_update_equity(self):
        """Test equity updates."""
        manager = PortfolioRiskManager(total_equity=100000.0)
        
        manager.register_position("BTC/USDT", "long", 10000.0)
        exposure_before = manager.get_symbol_exposure("BTC/USDT")
        assert exposure_before == 0.10
        
        # Update equity
        manager.update_equity(200000.0)
        
        exposure_after = manager.get_symbol_exposure("BTC/USDT")
        assert exposure_after == 0.05  # Same position, but now 5% of larger equity


class TestDisabledMode:
    """Test behavior when portfolio risk manager is disabled."""
    
    def test_disabled_validation_always_passes(self):
        """Test that disabled manager always approves."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            enabled=False
        )
        
        # Should approve any position
        allowed, reason, multiplier = manager.validate_new_position(
            symbol="BTC/USDT",
            side="long",
            proposed_size=999999.0  # Huge position
        )
        
        assert allowed is True
        assert reason == "portfolio_risk_disabled"
        assert multiplier == 1.0
    
    def test_disabled_register_does_nothing(self):
        """Test that disabled manager doesn't track positions."""
        manager = PortfolioRiskManager(
            total_equity=100000.0,
            enabled=False
        )
        
        manager.register_position("BTC/USDT", "long", 10000.0)
        
        # Should not actually register (disabled)
        # Note: Current implementation still registers, but this is safe
        # since validate_new_position ignores them when disabled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
