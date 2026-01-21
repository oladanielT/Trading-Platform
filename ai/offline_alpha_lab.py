"""
Offline Alpha Lab

Primary orchestrator for strategy discovery and validation.

This module:
- Runs walk-forward validation
- Tests robustness
- Qualifies strategies against strict criteria
- Produces read-only reports for human review

CRITICAL: This is OFFLINE ONLY. No live trading integration.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
import numpy as np

from ai.walk_forward_engine import WalkForwardEngine, WindowConfig
from ai.robustness_tests import RobustnessTestSuite, RobustnessConfig
from ai.performance_decomposer import PerformanceDecomposer


@dataclass
class AlphaQualificationCriteria:
    """Strict criteria for strategy validation."""
    min_expectancy: float = 0.0  # Must be positive
    max_drawdown: float = 0.20  # 20% max
    min_sharpe: float = 1.2
    min_robustness_score: float = 0.7
    min_oos_consistency: float = 0.70  # 70%
    min_trades: int = 30  # Minimum sample size


@dataclass
class StrategyReport:
    """Complete validation report for a strategy."""
    strategy_name: str
    timestamp: str
    qualified: bool
    rejection_reasons: List[str]
    
    # Performance
    expectancy: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    
    # Walk-Forward
    oos_consistency: float
    oos_degradation: float
    
    # Robustness
    robustness_score: float
    fragility_flags: List[str]
    
    # Regime Analysis
    regime_performance: Dict[str, Dict[str, float]]
    
    # Recommendations
    recommendations: List[str]


class OfflineAlphaLab:
    """
    Offline strategy research and validation laboratory.
    
    SAFETY GUARANTEES:
    - NO live trading hooks
    - NO runtime config changes
    - NO auto-weight updates
    - NO direct optimizer calls
    - NO execution integration
    
    Purpose: Discover robust alpha, reject overfit strategies.
    Output: Read-only reports for human decision-making.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Alpha Lab.
        
        Args:
            config: Configuration dict with 'alpha_lab' section
        """
        self.config = config.get('alpha_lab', {})
        
        if not self.config.get('enabled', False):
            raise RuntimeError(
                "Alpha Lab is disabled in config. "
                "Set alpha_lab.enabled=true to use."
            )
        
        # Initialize sub-engines
        wf_config = WindowConfig(
            train_pct=self.config.get('walk_forward', {}).get('train_pct', 0.6),
            test_pct=self.config.get('walk_forward', {}).get('test_pct', 0.4),
            step_size=self.config.get('walk_forward', {}).get('step_size', 0.1)
        )
        
        rob_config = RobustnessConfig(
            noise_level=self.config.get('robustness', {}).get('noise_level', 0.01),
            monte_carlo_runs=self.config.get('robustness', {}).get('monte_carlo_runs', 500),
            min_stability_score=0.7
        )
        
        self.walk_forward_engine = WalkForwardEngine(wf_config)
        self.robustness_suite = RobustnessTestSuite(rob_config)
        self.performance_decomposer = PerformanceDecomposer()
        
        # Qualification criteria
        self.criteria = AlphaQualificationCriteria()
        
        # Output directory
        self.reports_dir = self.config.get('reports_dir', 'alpha_reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(f"{self.reports_dir}/strategy_scorecards", exist_ok=True)
        os.makedirs(f"{self.reports_dir}/rejected", exist_ok=True)
        os.makedirs(f"{self.reports_dir}/validated", exist_ok=True)
        
        print(f"[ALPHA_LAB] Initialized (reports: {self.reports_dir})")
    
    def run_experiments(
        self,
        strategies: List[Dict[str, Any]],
        historical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run experiments on candidate strategies.
        
        Args:
            strategies: List of strategy configs:
                {
                    'name': str,
                    'func': Callable,  # (data, params) -> trades
                    'params': dict
                }
            historical_data: OHLCV + indicators + regime labels
            
        Returns:
            {
                'validated_strategies': [...],
                'rejected_strategies': [...],
                'performance_summary': {...}
            }
        """
        if not strategies:
            print("[ALPHA_LAB] No strategies to test")
            return {
                'validated_strategies': [],
                'rejected_strategies': [],
                'performance_summary': {}
            }
        
        print(f"[ALPHA_LAB] Running experiments on {len(strategies)} strategies")
        
        validated = []
        rejected = []
        
        for strategy in strategies:
            print(f"[ALPHA_LAB] Testing: {strategy['name']}")
            
            report = self._test_strategy(
                strategy_name=strategy['name'],
                strategy_func=strategy['func'],
                strategy_params=strategy['params'],
                data=historical_data
            )
            
            # Save report
            self._save_report(report)
            
            if report.qualified:
                validated.append(report)
                print(f"[ALPHA_LAB] ✅ {strategy['name']} VALIDATED")
            else:
                rejected.append(report)
                print(f"[ALPHA_LAB] ❌ {strategy['name']} REJECTED: {report.rejection_reasons}")
        
        # Summary
        summary = self._create_summary(validated, rejected)
        self._save_summary(summary)
        
        return {
            'validated_strategies': validated,
            'rejected_strategies': rejected,
            'performance_summary': summary
        }
    
    def _test_strategy(
        self,
        strategy_name: str,
        strategy_func: Callable,
        strategy_params: Dict[str, float],
        data: Dict[str, Any]
    ) -> StrategyReport:
        """
        Comprehensive testing of a single strategy.
        
        Returns:
            StrategyReport with all validation results
        """
        rejection_reasons = []
        recommendations = []
        
        # 1. Baseline Performance
        print(f"[ALPHA_LAB]   → Running baseline...")
        baseline_trades = strategy_func(data, strategy_params)
        baseline_metrics = self._calculate_metrics(baseline_trades)
        
        # 2. Walk-Forward Validation
        print(f"[ALPHA_LAB]   → Walk-forward validation...")
        regime_labels = data.get('regime_labels', None)
        wf_results = self.walk_forward_engine.run(
            strategy_func=lambda d: strategy_func(d, strategy_params),
            data=data,
            regime_labels=regime_labels
        )
        
        # 3. Robustness Testing
        print(f"[ALPHA_LAB]   → Robustness testing...")
        rob_results = self.robustness_suite.run_full_suite(
            strategy_func=strategy_func,
            data=data,
            strategy_params=strategy_params
        )
        
        # 4. Regime Performance Decomposition
        print(f"[ALPHA_LAB]   → Regime decomposition...")
        regime_perf = self._analyze_regime_performance(baseline_trades, regime_labels)
        
        # 5. Qualification Checks
        qualified = True
        
        # Check: Expectancy
        if baseline_metrics['expectancy'] <= self.criteria.min_expectancy:
            qualified = False
            rejection_reasons.append(
                f"Expectancy {baseline_metrics['expectancy']:.4f} <= {self.criteria.min_expectancy}"
            )
        
        # Check: Sharpe
        if baseline_metrics['sharpe'] < self.criteria.min_sharpe:
            qualified = False
            rejection_reasons.append(
                f"Sharpe {baseline_metrics['sharpe']:.2f} < {self.criteria.min_sharpe}"
            )
        
        # Check: Drawdown
        if baseline_metrics['max_dd'] > self.criteria.max_drawdown:
            qualified = False
            rejection_reasons.append(
                f"Max DD {baseline_metrics['max_dd']:.2%} > {self.criteria.max_drawdown:.2%}"
            )
        
        # Check: Robustness
        if rob_results['overall_stability'] < self.criteria.min_robustness_score:
            qualified = False
            rejection_reasons.append(
                f"Robustness {rob_results['overall_stability']:.2f} < {self.criteria.min_robustness_score}"
            )
        
        # Check: OOS Consistency
        if wf_results['overall_consistency'] < self.criteria.min_oos_consistency:
            qualified = False
            rejection_reasons.append(
                f"OOS consistency {wf_results['overall_consistency']:.2%} < {self.criteria.min_oos_consistency:.2%}"
            )
        
        # Check: Sample Size
        if len(baseline_trades) < self.criteria.min_trades:
            qualified = False
            rejection_reasons.append(
                f"Only {len(baseline_trades)} trades < {self.criteria.min_trades} minimum"
            )
        
        # Check: Fragility Flags
        if rob_results['fragility_flags']:
            recommendations.append(
                f"Fragility detected: {len(rob_results['fragility_flags'])} issues"
            )
        
        # 6. Generate Recommendations
        if qualified:
            recommendations.append("Strategy meets all criteria - eligible for paper trading")
            
            # Regime-specific advice
            for regime, perf in regime_perf.items():
                if perf['sharpe'] < 0.5:
                    recommendations.append(
                        f"Consider disabling in {regime} regime (low Sharpe: {perf['sharpe']:.2f})"
                    )
        
        # 7. Create Report
        return StrategyReport(
            strategy_name=strategy_name,
            timestamp=datetime.now().isoformat(),
            qualified=qualified,
            rejection_reasons=rejection_reasons,
            expectancy=baseline_metrics['expectancy'],
            sharpe=baseline_metrics['sharpe'],
            max_drawdown=baseline_metrics['max_dd'],
            win_rate=baseline_metrics['win_rate'],
            total_trades=len(baseline_trades),
            oos_consistency=wf_results['overall_consistency'],
            oos_degradation=wf_results['oos_degradation'],
            robustness_score=rob_results['overall_stability'],
            fragility_flags=rob_results['fragility_flags'],
            regime_performance=regime_perf,
            recommendations=recommendations
        )
    
    def _calculate_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not trades:
            return {
                'sharpe': 0.0,
                'expectancy': 0.0,
                'max_dd': 0.0,
                'win_rate': 0.0
            }
        
        pnls = [t.get('pnl', 0.0) for t in trades]
        
        # Sharpe
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Expectancy
        expectancy = np.mean(pnls)
        
        # Max Drawdown
        cumulative = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_dd = np.max(drawdown) / max(running_max[-1], 1.0) if len(drawdown) > 0 else 0.0
        
        # Win Rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        
        return {
            'sharpe': sharpe,
            'expectancy': expectancy,
            'max_dd': max_dd,
            'win_rate': win_rate
        }
    
    def _analyze_regime_performance(
        self,
        trades: List[Dict],
        regime_labels: Optional[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze performance by regime."""
        regime_perf = {}
        
        if not regime_labels or not trades:
            return regime_perf
        
        # Group trades by regime
        regime_trades = {}
        for trade in trades:
            regime = trade.get('regime', 'UNKNOWN')
            if regime not in regime_trades:
                regime_trades[regime] = []
            regime_trades[regime].append(trade)
        
        # Calculate metrics per regime
        for regime, regime_trade_list in regime_trades.items():
            metrics = self._calculate_metrics(regime_trade_list)
            regime_perf[regime] = metrics
        
        return regime_perf
    
    def _save_report(self, report: StrategyReport):
        """Save strategy report to disk."""
        # Choose directory
        if report.qualified:
            subdir = 'validated'
        else:
            subdir = 'rejected'
        
        # Filename
        safe_name = report.strategy_name.replace('/', '_').replace(' ', '_')
        filename = f"{self.reports_dir}/{subdir}/{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert to dict
        report_dict = {
            'strategy_name': report.strategy_name,
            'timestamp': report.timestamp,
            'qualified': report.qualified,
            'rejection_reasons': report.rejection_reasons,
            'performance': {
                'expectancy': report.expectancy,
                'sharpe': report.sharpe,
                'max_drawdown': report.max_drawdown,
                'win_rate': report.win_rate,
                'total_trades': report.total_trades
            },
            'validation': {
                'oos_consistency': report.oos_consistency,
                'oos_degradation': report.oos_degradation,
                'robustness_score': report.robustness_score
            },
            'fragility_flags': report.fragility_flags,
            'regime_performance': report.regime_performance,
            'recommendations': report.recommendations
        }
        
        # Write
        with open(filename, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"[ALPHA_LAB] Report saved: {filename}")
    
    def _create_summary(
        self,
        validated: List[StrategyReport],
        rejected: List[StrategyReport]
    ) -> Dict[str, Any]:
        """Create summary of all experiments."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_tested': len(validated) + len(rejected),
            'validated': len(validated),
            'rejected': len(rejected),
            'validation_rate': len(validated) / max(len(validated) + len(rejected), 1),
            'validated_strategies': [
                {
                    'name': r.strategy_name,
                    'sharpe': r.sharpe,
                    'robustness': r.robustness_score
                }
                for r in validated
            ],
            'rejection_summary': {}  # Could add rejection reason counts
        }
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary."""
        filename = f"{self.reports_dir}/summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[ALPHA_LAB] Summary saved: {filename}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information."""
        return {
            'enabled': self.config.get('enabled', False),
            'reports_dir': self.reports_dir,
            'criteria': {
                'min_expectancy': self.criteria.min_expectancy,
                'min_sharpe': self.criteria.min_sharpe,
                'max_drawdown': self.criteria.max_drawdown,
                'min_robustness': self.criteria.min_robustness_score,
                'min_oos_consistency': self.criteria.min_oos_consistency
            },
            'walk_forward': self.walk_forward_engine.get_diagnostics(),
            'robustness': self.robustness_suite.get_diagnostics()
        }
