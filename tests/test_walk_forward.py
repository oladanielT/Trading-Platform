from validation.walk_forward_engine import WalkForwardEngine


def test_walk_forward_disabled():
    w = WalkForwardEngine(enabled=False)
    r = w.run(None, None)
    assert r.get("ran") is False


def test_walk_forward_enabled():
    w = WalkForwardEngine(enabled=True, seed=2)
    r = w.run(None, None)
    assert r.get("ran") is True
