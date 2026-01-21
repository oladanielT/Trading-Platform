import asyncio
import pytest

from execution.sim_live_executor import SimLiveExecutor


class DummyBroker:
    def __init__(self, response=None):
        self.response = response or {'status': 'filled', 'fill_price': 100.0, 'filled_quantity': 1.0}

    async def place_market_order(self, symbol, side, quantity):
        await asyncio.sleep(0)  # yield
        return self.response


class DummyShadow:
    async def evaluate(self, *args, **kwargs):
        return {'status': 'sim', 'score': 0.9}


class DummyKillSwitch:
    def __init__(self, blocked=False):
        self._blocked = blocked

    def is_blocked(self):
        return self._blocked


class DummyHealth:
    def __init__(self):
        self.records = []

    def register_trade(self, pnl, closed=True):
        self.records.append((pnl, closed))


@pytest.mark.asyncio
async def test_sim_live_exec_routes_to_broker_and_health():
    broker = DummyBroker()
    shadow = DummyShadow()
    health = DummyHealth()
    kill = DummyKillSwitch(blocked=False)
    cfg = {'live_readiness': {'enabled': True, 'sim_live': True, 'shadow_only': False}}

    exe = SimLiveExecutor(broker=broker, shadow_executor=shadow, kill_switch=kill, health_monitor=health, config={'live_readiness': cfg['live_readiness']})

    res = await exe.place_market_order('BTC/USDT', 'buy', 0.1)
    assert res['status'] == 'filled'
    # health should have recorded a trade (realized 0.0 on entry per sim_live executor design)
    assert len(health.records) == 1


@pytest.mark.asyncio
async def test_sim_live_exec_blocks_when_kill_switch():
    broker = DummyBroker()
    health = DummyHealth()
    kill = DummyKillSwitch(blocked=True)
    exe = SimLiveExecutor(broker=broker, shadow_executor=None, kill_switch=kill, health_monitor=health, config={'live_readiness': {'enabled': True}})

    res = await exe.place_market_order('BTC/USDT', 'buy', 0.1)
    assert res['status'] == 'blocked'


@pytest.mark.asyncio
async def test_sim_live_exec_shadow_only():
    broker = DummyBroker()
    shadow = DummyShadow()
    health = DummyHealth()
    kill = DummyKillSwitch(blocked=False)
    exe = SimLiveExecutor(broker=broker, shadow_executor=shadow, kill_switch=kill, health_monitor=health, config={'live_readiness': {'enabled': True, 'sim_live': True, 'shadow_only': True}})

    res = await exe.place_market_order('BTC/USDT', 'buy', 0.5)
    assert res['status'] == 'sim_filled'
    assert len(health.records) == 1
