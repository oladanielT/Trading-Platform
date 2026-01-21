"""
Tests for runtime summary aggregation and logging output format.
"""

import time

from monitoring.runtime_summary import RuntimeSummaryAggregator


def test_emit_now_payload_structure():
    emitted = []

    def fake_logger(event, payload):
        emitted.append((event, payload))

    agg = RuntimeSummaryAggregator(interval_seconds=60, logger_fn=fake_logger)
    agg.record_cycle("ready", "trending", "buy", ts=time.time())
    agg.record_cycle("forming", "ranging", "hold", ts=time.time())

    agg.emit_now()

    assert len(emitted) == 1
    event, payload = emitted[0]
    assert event == "runtime.summary"
    assert payload["window_seconds"] == 60
    assert payload["samples"] == 2
    assert payload["readiness_distribution"] == {"ready": 1, "forming": 1}
    assert payload["regime_distribution"] == {"trending": 1, "ranging": 1}
    assert payload["trades_by_readiness"] == {"ready": 1}


def test_periodic_reset_between_emits():
    emitted = []

    def fake_logger(event, payload):
        emitted.append((event, payload))

    agg = RuntimeSummaryAggregator(interval_seconds=999, logger_fn=fake_logger)

    agg.record_cycle("ready", "trending", "buy", ts=time.time())
    agg.emit_now()

    agg.record_cycle("forming", "volatile", "sell", ts=time.time())
    agg.emit_now()

    assert len(emitted) == 2
    first = emitted[0][1]
    second = emitted[1][1]

    assert first["samples"] == 1
    assert first["readiness_distribution"] == {"ready": 1}
    assert first["trades_by_readiness"] == {"ready": 1}

    assert second["samples"] == 1
    assert second["readiness_distribution"] == {"forming": 1}
    assert second["trades_by_readiness"] == {"forming": 1}
