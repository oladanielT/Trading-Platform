from edge_validation.strategy_elimination_engine import StrategyEliminationEngine, EliminationDecision


class DummyPromotionManager:
    def __init__(self):
        self.retired = []
        self.demoted = []

    def retire(self, strategy, reason=None):
        self.retired.append((strategy, reason))

    def demote(self, strategy, reason=None, demoted_by=None, force_retire=False):
        self.demoted.append((strategy, reason, demoted_by, force_retire))


def test_elimination_engine_retire_and_demote():
    pm = DummyPromotionManager()
    engine = StrategyEliminationEngine(promotion_manager=pm)

    metrics = {
        's1': {'expectancy': -0.01, 'sharpe': 1.0},
        's2': {'expectancy': 0.02, 'sharpe': 0.3},
        's3': {'expectancy': 0.03, 'sharpe': 0.8, 'by_regime': {'bull': -0.02, 'bear': 0.01}},
    }

    decisions = engine.evaluate(metrics)
    actions = { (d.strategy, d.action) for d in decisions }

    assert ('s1', 'RETIRE') in actions
    assert ('s2', 'SHADOW') in actions
    assert any(d.strategy == 's3' and d.action == 'DISABLE_REGIME' for d in decisions)

    # promotion manager called appropriately
    assert any(r[0] == 's1' for r in pm.retired)
    assert any(d[0] == 's2' for d in pm.demoted)
