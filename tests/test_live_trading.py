"""
Integration tests for Binance live/paper trading system.
Tests multi-pair signal generation, trade execution, and portfolio metrics.
"""

import asyncio
import pytest
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from binance_regime_trader import (
    Trade, PairMetrics, PortfolioManager, LiveTradeManager
)
from data.binance_client import BinanceClient
from strategies.regime_strategy import RegimeBasedStrategy
from strategies.ema_trend import EMATrendStrategy


class TestTradeDataClass:
    """Test Trade data class."""
    
    def test_trade_creation(self):
        """Test creating a trade record."""
        trade = Trade(
            symbol='BTC/USDT',
            timestamp=datetime.now(),
            side='buy',
            price=43000.0,
            amount=0.001,
            regime='trending',
            regime_confidence=0.85,
            signal_confidence=0.75,
            order_id='order123'
        )
        
        assert trade.symbol == 'BTC/USDT'
        assert trade.side == 'buy'
        assert trade.price == 43000.0
        assert trade.regime == 'trending'
        assert trade.regime_confidence == 0.85
        
    def test_trade_to_dict(self):
        """Test converting trade to dictionary."""
        trade = Trade(
            symbol='ETH/USDT',
            timestamp=datetime(2025, 12, 30, 10, 30, 0),
            side='sell',
            price=2300.0,
            amount=0.01,
            regime='ranging',
            regime_confidence=0.60,
            signal_confidence=0.70
        )
        
        trade_dict = trade.to_dict()
        assert trade_dict['symbol'] == 'ETH/USDT'
        assert isinstance(trade_dict['timestamp'], str)
        assert '2025-12-30' in trade_dict['timestamp']


class TestPairMetrics:
    """Test PairMetrics data class."""
    
    def test_pair_metrics_creation(self):
        """Test creating pair metrics."""
        metrics = PairMetrics(
            symbol='BTC/USDT',
            total_return=0.15,
            total_trades=10,
            winning_trades=7
        )
        
        assert metrics.symbol == 'BTC/USDT'
        assert metrics.total_trades == 10
        assert abs(metrics.win_rate - 0.7) < 0.01
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        metrics = PairMetrics(
            symbol='ETH/USDT',
            total_trades=20,
            winning_trades=12
        )
        
        assert abs(metrics.win_rate - 0.6) < 0.01
    
    def test_win_rate_zero_trades(self):
        """Test win rate with zero trades."""
        metrics = PairMetrics(symbol='BNB/USDT')
        assert metrics.win_rate == 0.0


class TestPortfolioManager:
    """Test PortfolioManager."""
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        portfolio = PortfolioManager(initial_capital=100000.0)
        
        assert portfolio.initial_capital == 100000.0
        assert portfolio.current_equity == 100000.0
        assert portfolio.peak_equity == 100000.0
        assert len(portfolio.trades) == 0
    
    def test_add_trade(self):
        """Test adding a trade."""
        portfolio = PortfolioManager(100000.0)
        
        trade = Trade(
            symbol='BTC/USDT',
            timestamp=datetime.now(),
            side='buy',
            price=43000.0,
            amount=0.001,
            regime='trending',
            regime_confidence=0.85,
            signal_confidence=0.75
        )
        
        portfolio.add_trade(trade)
        
        assert len(portfolio.trades) == 1
        assert 'BTC/USDT' in portfolio.pair_metrics
        assert portfolio.pair_metrics['BTC/USDT'].total_trades == 1
    
    def test_update_position(self):
        """Test updating positions."""
        portfolio = PortfolioManager(100000.0)
        
        portfolio.update_position('BTC/USDT', 43000.0, 0.001)
        assert 'BTC/USDT' in portfolio.positions
        assert portfolio.positions['BTC/USDT']['amount'] == 0.001
        
        portfolio.update_position('BTC/USDT', 43000.0, 0)
        assert 'BTC/USDT' not in portfolio.positions
    
    def test_update_equity_with_positions(self):
        """Test updating equity with open positions."""
        portfolio = PortfolioManager(100000.0)
        
        portfolio.update_position('BTC/USDT', 43000.0, 0.001)
        
        prices = {'BTC/USDT': 44000.0}
        equity = portfolio.update_equity(prices)
        
        expected_equity = 100000.0 + (44000.0 - 43000.0) * 0.001
        assert abs(equity - expected_equity) < 1.0
    
    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        portfolio = PortfolioManager(100000.0)
        
        portfolio.peak_equity = 110000.0
        portfolio.current_equity = 99000.0
        
        drawdown = portfolio.max_drawdown
        expected_drawdown = (99000.0 - 110000.0) / 110000.0
        assert abs(drawdown - expected_drawdown) < 0.001
    
    def test_get_metrics(self):
        """Test getting aggregated metrics."""
        portfolio = PortfolioManager(100000.0)
        
        portfolio.current_equity = 115000.0
        
        trade = Trade(
            symbol='BTC/USDT',
            timestamp=datetime.now(),
            side='buy',
            price=43000.0,
            amount=0.001,
            regime='trending',
            regime_confidence=0.85,
            signal_confidence=0.75
        )
        
        portfolio.add_trade(trade)
        
        metrics = portfolio.get_metrics()
        
        assert metrics['initial_capital'] == 100000.0
        assert metrics['current_equity'] == 115000.0
        assert abs(metrics['total_return'] - 0.15) < 0.01
        assert metrics['total_trades'] == 1


class TestLiveTradeManager:
    """Test LiveTradeManager."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self):
        """Test manager initialization."""
        config_path = 'core/config.yaml'
        
        manager = LiveTradeManager(config_path=config_path)
        
        assert manager.config_path == config_path
        assert len(manager.strategies) == 0
        assert manager.portfolio is None
        assert manager.logger is not None
    
    @pytest.mark.asyncio
    async def test_initialize_creates_client_and_strategies(self):
        """Test that initialize creates client and strategies."""
        manager = LiveTradeManager()
        
        with patch.object(BinanceClient, 'initialize', new_callable=AsyncMock):
            await manager.initialize()
            
            assert manager.client is not None
            assert manager.portfolio is not None
            assert len(manager.strategies) > 0
    
    @pytest.mark.asyncio
    async def test_fetch_prices(self):
        """Test fetching prices."""
        manager = LiveTradeManager()
        manager.client = AsyncMock()
        manager.strategies = {
            'BTC/USDT': MagicMock(),
            'ETH/USDT': MagicMock()
        }
        
        mock_tickers = {
            'BTC/USDT': {'last': 43000.0},
            'ETH/USDT': {'last': 2300.0}
        }
        
        manager.client.fetch_multiple_tickers.return_value = mock_tickers
        
        prices = await manager.fetch_prices()
        
        assert prices['BTC/USDT'] == 43000.0
        assert prices['ETH/USDT'] == 2300.0
    
    @pytest.mark.asyncio
    async def test_generate_signals(self):
        """Test signal generation."""
        manager = LiveTradeManager()
        
        # Create mock strategy
        mock_strategy = MagicMock()
        mock_signal = {
            'action': 'buy',
            'confidence': 0.75,
            'regime': 'trending',
            'regime_confidence': 0.85
        }
        mock_strategy.analyze.return_value = mock_signal
        
        manager.strategies = {'BTC/USDT': mock_strategy}
        
        # Create mock OHLCV data
        df = pd.DataFrame({
            'open': [43000.0],
            'high': [43500.0],
            'low': [42500.0],
            'close': [43250.0],
            'volume': [100.0]
        })
        
        prices = {'BTC/USDT': 43250.0}
        historical_data = {'BTC/USDT': df}
        
        signals = await manager.generate_signals(prices, historical_data)
        
        assert 'BTC/USDT' in signals
        assert signals['BTC/USDT']['action'] == 'buy'
        assert signals['BTC/USDT']['symbol'] == 'BTC/USDT'
    
    @pytest.mark.asyncio
    async def test_execute_trade(self):
        """Test trade execution."""
        manager = LiveTradeManager()
        manager.client = AsyncMock()
        manager.portfolio = PortfolioManager(100000.0)
        manager.config = {
            'risk': {'max_position_size': 0.1},
            'trading': {
                'initial_capital': 100000.0,
                'allocation_per_symbol': 0.2
            }
        }
        
        mock_order = {'id': 'order123'}
        manager.client.place_order.return_value = mock_order
        
        signal = {
            'action': 'buy',
            'confidence': 0.75,
            'regime': 'trending',
            'regime_confidence': 0.85
        }
        
        trade = await manager.execute_trade('BTC/USDT', signal, 43000.0)
        
        assert trade is not None
        assert trade.symbol == 'BTC/USDT'
        assert trade.side == 'buy'
        assert trade.regime == 'trending'
        assert len(manager.portfolio.trades) == 1
    
    @pytest.mark.asyncio
    async def test_execute_trade_hold_signal(self):
        """Test that hold signals don't execute trades."""
        manager = LiveTradeManager()
        manager.portfolio = PortfolioManager(100000.0)
        
        signal = {'action': 'hold'}
        
        trade = await manager.execute_trade('BTC/USDT', signal, 43000.0)
        
        assert trade is None
        assert len(manager.portfolio.trades) == 0
    
    def test_generate_report(self):
        """Test report generation."""
        manager = LiveTradeManager()
        manager.portfolio = PortfolioManager(100000.0)
        manager.portfolio.current_equity = 115000.0
        
        trade = Trade(
            symbol='BTC/USDT',
            timestamp=datetime.now(),
            side='buy',
            price=43000.0,
            amount=0.001,
            regime='trending',
            regime_confidence=0.85,
            signal_confidence=0.75
        )
        manager.portfolio.add_trade(trade)
        
        report = manager.generate_report()
        
        assert 'TRADING SUMMARY REPORT' in report
        assert 'Current Equity' in report
        assert 'BTC/USDT' in report
        assert '115,000' in report or '115000' in report


class TestMultiPairTrading:
    """Integration tests for multi-pair trading."""
    
    @pytest.mark.asyncio
    async def test_multi_pair_signal_generation(self):
        """Test signal generation across multiple pairs."""
        manager = LiveTradeManager()
        
        # Create mock strategies
        symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        manager.strategies = {}
        
        for symbol in symbols:
            mock_strategy = MagicMock()
            mock_signal = {
                'action': 'buy' if symbol != 'BNB/USDT' else 'hold',
                'confidence': 0.75,
                'regime': 'trending',
                'regime_confidence': 0.85
            }
            mock_strategy.analyze.return_value = mock_signal
            manager.strategies[symbol] = mock_strategy
        
        # Create mock OHLCV data
        df = pd.DataFrame({
            'open': [1000.0] * 100,
            'high': [1010.0] * 100,
            'low': [990.0] * 100,
            'close': [1005.0] * 100,
            'volume': [100.0] * 100
        })
        
        prices = {
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2300.0,
            'BNB/USDT': 310.0
        }
        
        historical_data = {symbol: df for symbol in symbols}
        
        signals = await manager.generate_signals(prices, historical_data)
        
        assert len(signals) == 2  # BTC and ETH (BNB is hold)
        assert 'BTC/USDT' in signals
        assert 'ETH/USDT' in signals
        assert 'BNB/USDT' not in signals
    
    @pytest.mark.asyncio
    async def test_portfolio_metrics_aggregation(self):
        """Test portfolio metrics aggregation across pairs."""
        portfolio = PortfolioManager(100000.0)
        
        # Add trades for multiple symbols
        for i, symbol in enumerate(['BTC/USDT', 'ETH/USDT', 'BNB/USDT']):
            for j in range(3):
                trade = Trade(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    side='buy' if j % 2 == 0 else 'sell',
                    price=1000.0 + i * 100,
                    amount=0.01,
                    regime='trending',
                    regime_confidence=0.85,
                    signal_confidence=0.75
                )
                portfolio.add_trade(trade)
        
        # Update equity
        prices = {
            'BTC/USDT': 43000.0,
            'ETH/USDT': 2300.0,
            'BNB/USDT': 310.0
        }
        portfolio.update_equity(prices)
        
        metrics = portfolio.get_metrics()
        
        assert metrics['total_trades'] == 9
        assert len(portfolio.pair_metrics) == 3
        for symbol in ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']:
            assert symbol in portfolio.pair_metrics
            assert portfolio.pair_metrics[symbol].total_trades == 3


class TestRegimeSafetyFiltering:
    """Test regime-based safety filtering."""
    
    @pytest.mark.asyncio
    async def test_only_allowed_regimes_generate_signals(self):
        """Test that only allowed regimes generate trading signals."""
        manager = LiveTradeManager()
        manager.config = {
            'strategy': {
                'allowed_regimes': ['trending'],
                'regime_confidence_threshold': 0.5
            }
        }
        
        mock_strategy = MagicMock()
        
        # Signal in trending regime (allowed)
        allowed_signal = {
            'action': 'buy',
            'confidence': 0.75,
            'regime': 'trending',
            'regime_confidence': 0.85
        }
        
        # Signal in ranging regime (not allowed)
        disallowed_signal = {
            'action': 'buy',
            'confidence': 0.75,
            'regime': 'ranging',
            'regime_confidence': 0.85
        }
        
        mock_strategy.analyze.side_effect = [allowed_signal, disallowed_signal]
        manager.strategies = {'BTC/USDT': mock_strategy}
        
        df = pd.DataFrame({
            'close': [1000.0] * 100,
            'volume': [100.0] * 100
        })
        
        prices = {'BTC/USDT': 43000.0}
        historical_data = {'BTC/USDT': df}
        
        # First call should return signal (trending)
        signals1 = await manager.generate_signals(prices, historical_data)
        assert len(signals1) == 1
        
        # Second call should return no signal (ranging not allowed)
        signals2 = await manager.generate_signals(prices, historical_data)
        assert len(signals2) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
