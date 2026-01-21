# Phase12 Quick Reference

- All Phase12 modules live under: `execution/`, `stress/`, `validation/`, `risk/`, `analytics/`.
- Default flags: `config/phase12_config.py` (all disabled).
- Tests added under `tests/` to validate disabled vs enabled behavior.

How to enable a module (example)

1. Open `config/phase12_config.py`
2. Set `DEFAULTS['execution_realism']['enabled'] = True`

Run tests:

```bash
pytest -q
```
