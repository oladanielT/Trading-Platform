from edge_validation.capital_readiness_evaluator import CapitalReadinessEvaluator


def test_capital_evaluator_no_go_blocking():
    evaluator = CapitalReadinessEvaluator()
    metrics = {
        'A': {'expectancy': 0.04, 'sharpe': 1.0},
        'B': {'expectancy': -0.01, 'sharpe': 0.5},
    }
    report = evaluator.evaluate(metrics)
    assert report.verdict == 'NO_GO'
    assert any('B' in r for r in report.blocking_reasons)


def test_capital_evaluator_micro_only():
    evaluator = CapitalReadinessEvaluator()
    metrics = {
        'C': {'expectancy': 0.005, 'sharpe': 0.5},
    }
    report = evaluator.evaluate(metrics)
    assert report.verdict == 'MICRO_ONLY'


def test_capital_evaluator_limited_live():
    evaluator = CapitalReadinessEvaluator()
    metrics = {
        'D': {'expectancy': 0.04, 'sharpe': 1.0},
    }
    report = evaluator.evaluate(metrics)
    assert report.verdict == 'LIMITED_LIVE'
    assert 0.0 < report.score <= 1.0
