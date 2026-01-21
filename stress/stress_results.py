"""Helpers for formatting stress results."""

def summarize(result: dict) -> str:
    applied = result.get("applied", False)
    return f"Scenario {result.get('scenario')} applied={applied} impact={result.get('impact', 0.0)}"


__all__ = ["summarize"]
