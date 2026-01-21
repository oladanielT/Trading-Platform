from stress.stress_engine import StressEngine


def test_stress_engine_disabled():
    s = StressEngine(enabled=False, seed=1)
    r = s.run_scenario("volatility_spike", {})
    assert r.get("applied") is False


def test_stress_engine_enabled():
    s = StressEngine(enabled=True, seed=1)
    r = s.run_scenario("volatility_spike", {"severity": 2.0})
    assert r.get("applied") is True
