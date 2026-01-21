import pytest
from monitoring.runtime_health_monitor import RuntimeHealthMonitor


def test_health_monitor_aggregation(tmp_path):
    cfg = {'live_readiness': {'enabled': True, 'csv_snapshot': False}}
    hm = RuntimeHealthMonitor(cfg)
    # register a few trades
    hm.register_trade(pnl=10.0, closed=True)
    hm.register_trade(pnl=-5.0, closed=True)
    hm.update_unrealized(2.5)
    hm.set_regime('TRENDING')
    hm.set_active_strategies(['EMA', 'RSI'])

    metrics = hm.get_metrics()
    assert metrics['trades_today'] == 2
    assert metrics['wins'] == 1
    assert metrics['losses'] == 1
    assert metrics['realized_pnl'] == 5.0
    assert metrics['unrealized_pnl'] == 2.5
    assert metrics['current_regime'] == 'TRENDING'
    assert 'EMA' in metrics['active_strategies']
