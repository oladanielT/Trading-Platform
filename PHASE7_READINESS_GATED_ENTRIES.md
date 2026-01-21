# PHASE 7: Readiness-Gated Trend Entries (Paper/Backtest Only)

## Objective

Use `ReadinessState.READY` to softly gate trend-based entries **only** in non-live modes (paper/backtest). Live trading remains unchanged. No changes to sizing, risk, execution, or thresholds.

## Feature Overview

- **Feature flag:** `enable_readiness_gate` (default: `False`)
- **Mode awareness:** `trading_mode` (`'live' | 'paper' | 'backtest'`), default `'live'`
- **Gate condition:** Applies **only when** `enable_readiness_gate=True` **and** `trading_mode` is `paper` or `backtest`.
- **Gating rule:** If regime is trend-aligned (TRENDING or VOLATILE with override), signal is BUY/SELL, and `readiness_state != READY` → return HOLD.
- **Logging:** INFO logs for ALLOWED/BLOCKED decisions with regime, readiness state, score, signal, mode.

## Implementation

- **Flag + mode** added to `RegimeBasedStrategy.__init__`
- **Helper:** `_should_apply_readiness_gate()` determines applicability (flag on + non-live mode)
- **Gating location:** `analyze()` final decision path; gates BUY/SELL in trend regimes when readiness is not READY; sets reason `readiness_gate_blocked` and logs BLOCKED. ALLOWED path logs as well.
- **Live unaffected:** `_should_apply_readiness_gate()` returns False in live mode, so no gating/logging in live.

## Logging Examples

- Blocked: `[ReadinessGate] BLOCKED: signal=buy | regime=trending | readiness=forming (score=0.52) | mode=paper | RETURNING HOLD`
- Allowed: `[ReadinessGate] ALLOWED: signal=buy | regime=trending | readiness=ready (score=0.78) | mode=paper`

## Safety Constraints

- ❌ No changes to live trading behavior
- ❌ No sizing/risk/execution changes
- ❌ No threshold modifications
- ❌ No readiness-based confidence scaling
- ✅ Gated HOLD logs explicit reason/detail

## Tests Added (tests/test_readiness_gating.py)

- **Flag behavior:** default False; mode default live; gate applies only in paper/backtest when enabled
- **Live parity:** Live mode identical with gate on/off; never blocked by readiness
- **Paper/backtest gating:** Blocks when readiness != READY; allows when READY; logs BLOCKED/ALLOWED
- **Non-trend regimes:** Unaffected by gate
- **Legacy behavior:** Flag off preserves legacy behavior

## Test Results

- New tests: **17 passed**
- Full suite: **93 passed, 0 failures**

## Usage

```python
strategy = RegimeBasedStrategy(
    allowed_regimes=['TRENDING'],
    enable_readiness_gate=True,
    trading_mode='paper'  # or 'backtest'; 'live' leaves gate inactive
)
result = strategy.analyze(market_data)
# If readiness != READY and regime is trend-aligned, BUY/SELL will be gated to HOLD.
```

## Notes

- Applies only in `analyze()` path (dictionary output used by live/paper/backtest controllers). Live remains unaffected by design. Further gating of `generate_signal()` can be added in future if required.
