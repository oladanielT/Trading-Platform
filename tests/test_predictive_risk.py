from risk.predictive_risk import PredictiveRisk


def test_predictive_risk_disabled():
    p = PredictiveRisk(enabled=False)
    r = p.simulate([], n_sims=10)
    assert r.get("simulated") is False


def test_predictive_risk_enabled():
    p = PredictiveRisk(enabled=True, seed=3)
    r = p.simulate([0.01, -0.02], n_sims=10)
    assert r.get("simulated") is True
