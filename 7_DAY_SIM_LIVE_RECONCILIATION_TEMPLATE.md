# 7-Day Sim-Live Reconciliation Report

> Purpose: Provide a concise, zero-ambiguity reconciliation between expected behavior (Phase13 edge validation) and actual sim-live results across seven consecutive trading days.

## Report Metadata

- Period: [YYYY-MM-DD → YYYY-MM-DD]
- Strategy: [Name]
- Symbols: [e.g., BTCUSDT, ETHUSDT]
- Runtime Config: [config-live-trading.yaml or MINIMAL_LIVE_CONFIG.yaml]
- Code Version: [git commit or tag]
- Environment: [Sim-Live environment details]
- Report Owner: [Name]
- Reviewer(s): [Name(s)]

---

## Daily Reconciliation (7 days)

| Day | Date | Primary Regime | Daily PnL | Cum PnL | Kill-Switch                  | Risk Violations | Notes |
| --- | ---- | -------------- | --------- | ------- | ---------------------------- | --------------- | ----- |
| 1   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |
| 2   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |
| 3   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |
| 4   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |
| 5   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |
| 6   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |
| 7   | [ ]  | [ ]            | [ ]       | [ ]     | [Armed/Disarmed, Triggered?] | [Y/N + brief]   | [ ]   |

- Regime notes per day (optional):
  - Day 1: [transition(s), dwell time, anomaly]
  - Day 2: [ ]
  - Day 3: [ ]
  - Day 4: [ ]
  - Day 5: [ ]
  - Day 6: [ ]
  - Day 7: [ ]

---

## Strategy-Level PnL

> Summarize at the strategy level across the full 7-day period.

| Metric                        | Value |
| ----------------------------- | ----- |
| Total Gross PnL               | [ ]   |
| Total Net PnL (fees/slippage) | [ ]   |
| Max Drawdown                  | [ ]   |
| Sharpe (7-day)                | [ ]   |
| Win Rate (orders/trades)      | [ ]   |
| Avg Slippage vs Model         | [ ]   |

- Notes: [aggregation method, outliers, fee model]

---

## Regime Breakdown

> Distribution and performance by regime during the 7-day window.

| Regime     | Share of Time | PnL Contribution | Trade Count | Notes |
| ---------- | ------------- | ---------------- | ----------- | ----- |
| [Regime A] | [ ]           | [ ]              | [ ]         | [ ]   |
| [Regime B] | [ ]           | [ ]              | [ ]         | [ ]   |
| [Regime C] | [ ]           | [ ]              | [ ]         | [ ]   |

- Transitions: [Number of transitions, notable shifts]
- Dwell Times: [Median/mean]
- Alignment: [Matches Phase13 expected regime mix?]

---

## Kill-Switch Status

> Document configuration, arming status, and any triggers with precise timestamps.

- Config: [thresholds, conditions]
- Status: [Armed/Disarmed per day]
- Triggers:
  - [YYYY-MM-DD HH:MM:SS] Condition: [ ] Action: [ ] Duration: [ ]
  - [ ]
- Downtime/Pauses: [ ]
- Rationale: [why armed/disarmed]

---

## Phase13 Verdict Comparison (Expected vs Actual)

> Compare sim-live outcomes to Phase13 expected edge validation. Use PHASE13_EDGE_VALIDATION.md and related artifacts.

| Dimension                     | Expected (Phase13) | Actual (7-day) | Delta | Verdict      |
| ----------------------------- | ------------------ | -------------- | ----- | ------------ |
| Edge Metrics (PnL per regime) | [ ]                | [ ]            | [ ]   | [Align/Miss] |
| Drawdown Constraints          | [ ]                | [ ]            | [ ]   | [Align/Miss] |
| Risk Events (0 tolerance)     | [None]             | [ ]            | [ ]   | [Align/Miss] |
| Kill-Switch Behavior          | [ ]                | [ ]            | [ ]   | [Align/Miss] |
| Slippage/Fees Model           | [ ]                | [ ]            | [ ]   | [Align/Miss] |

- Notes: [explain any discrepancies, environmental differences]

---

## Risk Violations (Zero Tolerance)

> Enumerate any violations. If any occur, justification is not acceptable; flag for Stop.

| Timestamp | Violation Type | Severity | Impact | Action Taken |
| --------- | -------------- | -------- | ------ | ------------ |
| [ ]       | [ ]            | [ ]      | [ ]    | [ ]          |

- Summary: [None OR details]

---

## Final Recommendation

> Choose one and justify clearly.

- Recommendation: [Go / Hold / Stop]
- Justification: [concise]
- Preconditions (if Go/Hold):
  - [ ] Confirm no risk violations
  - [ ] Kill-switch validated
  - [ ] Regime behavior matches Phase13 expectations
  - [ ] Acceptable slippage/fees
  - [ ] Drawdown within bounds

---

## Attachments / References

- Logs: [paths]
- Configs: [config-live-trading.yaml, MINIMAL_LIVE_CONFIG.yaml]
- Code refs: [commit/tag]
- Validation docs: [PHASE13_EDGE_VALIDATION.md]
- Tests: [links to test executions]

---

## Sign-off

- Prepared by: [ ] Date: [ ]
- Reviewed by: [ ] Date: [ ]
- Approval: [ ]
