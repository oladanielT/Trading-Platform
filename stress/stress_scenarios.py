"""Predefined stress scenarios (vol spike, gaps, correlated crashes)."""

def volatility_spike(context):
    return {"type": "volatility_spike", "severity": context.get("severity", 1.0)}


def gap_event(context):
    return {"type": "gap_event", "gap_size": context.get("gap_size", 0.05)}


def correlated_crash(context):
    return {"type": "correlated_crash", "symbols": context.get("symbols", []), "drop": context.get("drop", 0.2)}


__all__ = ["volatility_spike", "gap_event", "correlated_crash"]
