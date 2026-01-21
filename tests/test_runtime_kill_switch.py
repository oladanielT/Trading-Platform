import pytest
from risk.runtime_kill_switch import RuntimeKillSwitch


def test_kill_switch_max_daily_loss():
    cfg = {'kill_switch': {'enabled': True, 'max_daily_loss_pct': 0.1, 'max_consecutive_losses': 5}}
    ks = RuntimeKillSwitch(cfg, initial_equity=1000.0)
    # register a loss that exceeds 0.1%? The config is percent; max_daily_loss_pct used as percent
    # max_daily_loss_pct: 0.1 means 0.1%? In implementation, compares loss_pct >= config.max_daily_loss_pct
    # So to trigger, set pnl such that abs(todays_pnl)/equity*100 >= 0.1
    ks.register_trade_result(-2.0)  # -2 on 1000 => 0.2% should be >= 0.1
    assert ks.is_blocked()
    state = ks.get_state()
    assert state['triggered']
    assert state['reason'] in ('max_daily_loss', 'max_consecutive_losses', 'critical_error')


def test_kill_switch_consecutive_losses():
    cfg = {'kill_switch': {'enabled': True, 'max_daily_loss_pct': 100.0, 'max_consecutive_losses': 3}}
    ks = RuntimeKillSwitch(cfg, initial_equity=1000.0)
    ks.register_trade_result(-10.0)
    assert not ks.is_blocked()
    ks.register_trade_result(-5.0)
    assert not ks.is_blocked()
    ks.register_trade_result(-1.0)
    # After 3 consecutive losses, should be triggered
    assert ks.is_blocked()
    s = ks.get_state()
    assert s['consecutive_losses'] >= 3
