from dataclasses import asdict
import json
from typing import Dict, Any, Optional
from datetime import datetime

from .strategy_elimination_engine import StrategyEliminationEngine
from .capital_readiness_evaluator import CapitalReadinessEvaluator

DEFAULT_METRICS = {
    's1': {'expectancy': -0.02, 'sharpe': 0.9},
    's2': {'expectancy': 0.03, 'sharpe': 0.4, 'by_regime': {'bull': 0.04, 'bear': -0.01}},
    's3': {'expectancy': 0.045, 'sharpe': 1.2},
}


def _serialize_decision(d):
    # dataclass -> dict
    try:
        return asdict(d)
    except Exception:
        return {
            'strategy': getattr(d, 'strategy', None),
            'action': getattr(d, 'action', None),
            'reason': getattr(d, 'reason', None),
            'details': getattr(d, 'details', {}),
        }


def run_phase13(metrics: Optional[Dict[str, Dict[str, Any]]] = None, save_json_path: Optional[str] = None, console: bool = True) -> Dict[str, Any]:
    """Run Phase13 analysis in read-only mode and return a structured result.

    - metrics: per-strategy metrics used by the elimination engine and capital evaluator.
    - save_json_path: optional path to write a JSON report.
    - console: if True, print a human-readable summary to stdout.

    This runner is explicitly read-only: it never supplies a `promotion_manager` to
    `StrategyEliminationEngine`, so no retire/demote calls are made.
    """
    metrics = metrics or DEFAULT_METRICS

    # analysis-only engines
    elim = StrategyEliminationEngine(promotion_manager=None)
    cap = CapitalReadinessEvaluator()

    decisions = elim.evaluate(metrics)
    decisions_serial = sorted([_serialize_decision(d) for d in decisions], key=lambda x: (x.get('strategy') or '', x.get('action') or ''))

    cap_report = cap.evaluate(metrics)
    try:
        cap_report_serial = asdict(cap_report)
    except Exception:
        cap_report_serial = {
            'score': getattr(cap_report, 'score', None),
            'verdict': getattr(cap_report, 'verdict', None),
            'blocking_reasons': getattr(cap_report, 'blocking_reasons', []),
            'details': getattr(cap_report, 'details', {}),
        }

    result = {
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'decisions': decisions_serial,
        'capital_report': cap_report_serial,
        'metrics_snapshot': {k: metrics[k] for k in sorted(metrics.keys())},
    }

    if save_json_path:
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, sort_keys=True, indent=2)

    if console:
        print('Phase13 Report —', result['generated_at'])
        print('\nElimination Decisions:')
        if decisions_serial:
            for d in decisions_serial:
                print('-', d['strategy'], d['action'], '-', d['reason'])
        else:
            print('- none')
        print('\nCapital Readiness:')
        print('-', cap_report_serial.get('verdict'), f"(score: {cap_report_serial.get('score'):.3f})")
        if cap_report_serial.get('blocking_reasons'):
            print('\nBlocking reasons:')
            for b in cap_report_serial.get('blocking_reasons'):
                print('-', b)

    return result


if __name__ == '__main__':
    # simple CLI demo
    run_phase13()
