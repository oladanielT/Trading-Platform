**Week 1 — Sim-Live Playbook (operational runbook)**

Purpose: Execute a controlled sim-live run to gather evidence required for any micro-live decision.

Duration: 7 calendar days (minimum)

Daily Checklist (morning pre-market)

- Verify process is running and `main.py` started in sim-live mode (`live_readiness.sim_live: true`).
- Confirm on-call operator present and reachable.
- Confirm monitoring/logging pipeline is collecting logs to `logs/sim_live.log` and metrics collector is live.
- Confirm API connectivity to exchange testnet (place a small test order & cancel).

Throughout the Day (continuous)

- Monitor PnL, position counts, and outstanding orders.
- Watch for kill-switch conditions and ensure auto-stop engages when triggered.
- Log any partial fills, rejections, or unexpected errors.

End of Day (post-market)

- Generate a daily reconciliation: expected fill prices vs simulated/observed fills, slippage quantiles (p50/p90), fill success rate.
- Summarize kill-switch events, drawdown statistics, and any rule vetoes from `PortfolioRiskManager`.
- Archive logs and snapshot the `StrategyPromotionManager` state.

Metrics to Monitor (daily & real-time)

- Equity curve and drawdown (absolute and %)
- Number of trades and average trade PnL
- Slippage distribution (expected vs actual fill price)
- Partial fill rate and average fill latency
- Kill-switch activation count and reason
- Portfolio exposures (symbol, direction, regime)

Kill-switch Triggers (operational thresholds — tune to MINIMAL_LIVE_CONFIG.yaml)

- Absolute loss threshold: stop trading immediately if equity drops by configured `absolute_loss_limit`.
- Max daily loss pct: stop trading if cumulative loss in the day exceeds `max_daily_loss_pct`.
- Consecutive loss count: pause trading if `consecutive_loss_limit` is reached.

Escalation and Actions

- If kill-switch trips: operator must acknowledge within 15 minutes and run post-mortem. No auto-restart.
- If multiple partial fills or unexpected cancellations occur (>5% of orders), pause trading and investigate.
- If slippage p90 > acceptable bound (to be defined from sim-live baseline), pause and investigate.

Criteria for Continuing After Week 1

- Pass: 7-day sim-live with no unrecoverable kill-switch trips, stable slippage profile (p50/p90 within expected), monitoring validated, and promotion manager confirms strategy state.
- Next: allow a timeboxed micro-live with strict capital cap and same runbook.

When to Stop Permanently (hard stop)

- Any persistent inability to connect to exchange or recurrent partial-fill anomalies.
- Unexpected production errors that cannot be reproduced in testnet/sim-live.

Required Deliverables After Week 1

- 7-day reconciliation report (CSV + summary) with fill-level data.
- Kill-switch stress test report and logs.
- Monitoring & alerting verification (screenshots/log evidence of alerts sent to the on-call destination).
- Strategy promotion state export from `StrategyPromotionManager` showing approvals.

Evidence & Reporting

- Complete the 7-day sim-live reconciliation template at [7_DAY_SIM_LIVE_RECONCILIATION_TEMPLATE.md](7_DAY_SIM_LIVE_RECONCILIATION_TEMPLATE.md) at the end of the 7-day run, before any Go/No-Go decision.

Notes

- This playbook is intentionally conservative: sim-live first, then micro-live only with evidence. All steps assume no code changes; only configuration and operational verification are permitted during Phase14.
