Phase 13 — Edge Validation & Capital Justification

Purpose

- Analysis-only validation to decide whether a strategy or strategy set is ready for capital deployment.
- Produces defensible metrics and a capital-readiness verdict: NO_GO, MICRO_ONLY, or LIMITED_LIVE.

Safety

- Default feature toggle: `edge_validation.enabled` = False
- Analysis-only: modules must not modify live trading, execution, risk math, or optimizer behavior.
- Any retire/demotion calls must go through `StrategyPromotionManager` and follow governance; no automatic promotion.

Files

- `edge_validation/walk_forward_engine.py` — rolling walk-forward split engine.
- `edge_validation/expectancy_analyzer.py` — computes expectancy, sharpe, sortino, drawdown metrics.
- `edge_validation/regime_performance_matrix.py` — per-strategy/per-regime summaries.
- `edge_validation/strategy_elimination_engine.py` — (analysis-first) applies hard rules and records decisions.
- `edge_validation/capital_readiness_evaluator.py` — computes readiness score & verdict.

Usage

- Run analyses on historical trade data only. Use outputs to inform governance decisions.
- To enable automated runs, toggle `edge_validation.enabled` in config (default False).

Testing

- Unit tests added under `tests/test_edge_validation_*.py` to validate logic and safety.

Notes

- Ensure trade PnL includes costs (commissions & slippage) before feeding to expectancy computations.
- Per-regime disables are advisory and must be reviewed by human governance before enforcement.
