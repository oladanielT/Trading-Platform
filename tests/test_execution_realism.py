import pytest
from execution.execution_realism import ExecutionRealismWrapper


class DummyBroker:
    def execute_order(self, order):
        return {"filled_volume": order.get("volume"), "avg_price": order.get("price"), "fees": 0.0, "status": "ok"}


def test_execution_realism_disabled():
    b = DummyBroker()
    w = ExecutionRealismWrapper(broker=b, enabled=False)
    res = w.execute_order({"price": 100, "volume": 1, "side": "buy"})
    assert res["filled_volume"] == 1


def test_execution_realism_enabled_seeded():
    b = DummyBroker()
    w = ExecutionRealismWrapper(broker=b, enabled=True, seed=123)
    res = w.execute_order({"price": 100, "volume": 1, "side": "buy", "volatility": 0.1, "recent_volume": 10})
    assert "status" in res
