"""Validation report helpers."""

def format_report(report: dict) -> str:
    return f"WalkForward ran={report.get('ran', False)} windows={report.get('windows', 0)}"


__all__ = ["format_report"]
