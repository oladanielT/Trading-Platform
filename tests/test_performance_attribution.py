from analytics.performance_attribution import attribute


def test_performance_attribution_disabled():
    r = attribute({"s1": 0.0, "s2": 0.0})
    assert r.get("attributed") is False


def test_performance_attribution_enabled_monotonic():
    # enable via config override
    from config.phase12_config import DEFAULTS
    DEFAULTS["performance_attribution"]["enabled"] = True
    r = attribute({"s1": 10.0, "s2": -5.0})
    assert r.get("attributed") is True
