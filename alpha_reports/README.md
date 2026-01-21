# Alpha Lab Reports

This directory contains offline strategy validation reports from Phase 10.

**CRITICAL: These are research artifacts, NOT live trading configuration.**

## Directory Structure

```
alpha_reports/
├── strategy_scorecards/   # Detailed scorecard for each tested strategy
├── validated/             # Strategies that passed all qualification criteria
├── rejected/              # Strategies that failed qualification
├── walk_forward_plots/    # Visualization of walk-forward results
└── summary_*.json         # Aggregate summaries of experiment runs
```

## Report Contents

Each strategy report includes:

- **Performance metrics**: Sharpe, expectancy, drawdown, win rate
- **Walk-forward results**: OOS consistency, degradation analysis
- **Robustness scores**: Parameter sensitivity, noise resilience
- **Regime analysis**: Performance breakdown by market regime
- **Rejection reasons**: Why strategies failed (if applicable)
- **Recommendations**: Next steps for human decision-maker

## Usage

1. **Run Experiments**: Use `OfflineAlphaLab.run_experiments()` to test strategies
2. **Review Reports**: Examine validated/rejected reports in JSON format
3. **Human Decision**: Only humans can promote validated strategies to paper trading
4. **NO AUTO-DEPLOY**: Reports never modify live trading configuration

## Safety

- Reports are **read-only** for live systems
- Alpha Lab is **completely isolated** from execution pipeline
- Qualification != automatic activation
- Human review required for all promotions
