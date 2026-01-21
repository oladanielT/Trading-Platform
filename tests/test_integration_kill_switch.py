import asyncio
import pytest

from main import TradingPlatform

@pytest.mark.asyncio
async def test_integration_kill_switch_stops_trading(monkeypatch):
    # Create platform with config that enables live_readiness and a low max_daily_loss_pct
    platform = TradingPlatform()
    # Provide a custom config
    platform.config = platform._default_config()
    platform.config['live_readiness'] = {
        'enabled': True,
        'sim_live': True,
    }
    # Set kill switch thresholds via config
    platform.config['kill_switch'] = {
        'enabled': True,
        'max_daily_loss_pct': 0.01,  # 0.01% to trigger quickly
        'max_consecutive_losses': 1,
        'close_positions_on_trigger': False,
    }
    # Use small initial capital to make percentages easier
    platform.config['trading']['initial_capital'] = 1000.0

    # Monkeypatch setup steps to avoid network/exchange calls
    async def noop_setup():
        # call subsets of setup to initialize components used in test
        await platform._setup_core()
        await platform._setup_monitoring()
        # minimal data/execution setup
        platform.exchange = None
        platform.market_feed = None
        await platform._setup_risk()
        await platform._setup_execution()

    await noop_setup()

    # Ensure kill_switch initialized
    assert platform.kill_switch is not None

    # Trigger kill switch via trade loss
    platform.kill_switch.register_trade_result(-20.0)  # on 1000 equity -> 2% > 0.01
    assert platform.kill_switch.is_blocked()

    # Attempt to place an order via sim_executor (should be blocked)
    if platform.sim_executor:
        res = await platform.sim_executor.place_market_order('BTC/USDT', 'buy', 0.1)
        assert res.get('status') == 'blocked'
    else:
        pytest.skip('Sim executor not initialized')
