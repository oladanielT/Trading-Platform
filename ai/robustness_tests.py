"""
Robustness Testing Suite

Tests strategy stability under:
- Parameter perturbation
- Noise injection
- Regime shuffling
- Monte Carlo trade reordering

SAFETY: This module is OFFLINE ONLY - no live trading integration.
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import numpy as np
import copy


@dataclass
class RobustnessConfig:
    """Configuration for robustness tests."""
    param_perturbation_range: float = 0.2  # ±20%
    noise_level: float = 0.01  # 1% price jitter
    monte_carlo_runs: int = 500
    min_stability_score: float = 0.7  # 0-1 scale


@dataclass
class RobustnessResult:
    """Results from robustness testing."""
    test_type: str
    baseline_sharpe: float
    perturbed_sharpes: List[float]
    stability_score: float  # 0-1, higher = more stable
    fragility_flags: List[str]
    passed: bool


class RobustnessTestSuite:
    """
    Comprehensive robustness testing for strategy validation.
    
    CRITICAL: This is an OFFLINE research tool. It does NOT:
    - Modify live strategies
    - Affect runtime parameters
    - Change position sizes
    - Influence execution
    
    Purpose: Detect overfitting and fragile strategies before human review.
    """
    
    def __init__(self, config: Optional[RobustnessConfig] = None):
        """
        Initialize robustness test suite.
        
        Args:
            config: Robustness testing configuration
        """
        self.config = config or RobustnessConfig()
        self.results: List[RobustnessResult] = []
    
    def run_full_suite(
        self,
        strategy_func: Callable,
        data: Dict[str, Any],
        strategy_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run complete robustness test suite.
        
        Args:
            strategy_func: Strategy function(data, params) -> trades
            data: Historical market data
            strategy_params: Strategy parameters to perturb
            
        Returns:
            {
                'overall_stability': float,  # 0-1
                'test_results': List[RobustnessResult],
                'fragility_flags': List[str],
                'passed': bool
            }
        """
        self.results = []
        
        # Baseline performance
        baseline_trades = strategy_func(data, strategy_params)
        baseline_metrics = self._calculate_metrics(baseline_trades)
        
        # Test 1: Parameter Perturbation
        param_result = self._test_parameter_perturbation(
            strategy_func, data, strategy_params, baseline_metrics
        )
        self.results.append(param_result)
        
        # Test 2: Noise Injection
        noise_result = self._test_noise_injection(
            strategy_func, data, strategy_params, baseline_metrics
        )
        self.results.append(noise_result)
        
        # Test 3: Regime Shuffling
        regime_result = self._test_regime_shuffle(
            strategy_func, data, strategy_params, baseline_metrics
        )
        self.results.append(regime_result)
        
        # Test 4: Monte Carlo Trade Reordering
        mc_result = self._test_monte_carlo_reorder(
            baseline_trades, baseline_metrics
        )
        self.results.append(mc_result)
        
        # Aggregate results
        return self._aggregate_results()
    
    def _test_parameter_perturbation(
        self,
        strategy_func: Callable,
        data: Dict[str, Any],
        params: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> RobustnessResult:
        """
        Test strategy stability under parameter variations.
        
        Perturbs each parameter by ±10%, ±20%, ±30%.
        """
        perturbed_sharpes = []
        fragility_flags = []
        
        perturbation_levels = [-0.30, -0.20, -0.10, 0.10, 0.20, 0.30]
        
        for param_name, param_value in params.items():
            if not isinstance(param_value, (int, float)):
                continue
                
            for delta in perturbation_levels:
                # Create perturbed params
                perturbed_params = copy.deepcopy(params)
                perturbed_params[param_name] = param_value * (1 + delta)
                
                # Run strategy
                try:
                    trades = strategy_func(data, perturbed_params)
                    metrics = self._calculate_metrics(trades)
                    perturbed_sharpes.append(metrics['sharpe'])
                except Exception as e:
                    # Strategy failed under perturbation
                    fragility_flags.append(
                        f"Failed on {param_name} {delta*100:+.0f}%: {str(e)[:50]}"
                    )
                    perturbed_sharpes.append(-999)  # Severe penalty
        
        # Calculate stability
        if perturbed_sharpes:
            sharpe_std = np.std(perturbed_sharpes)
            sharpe_range = max(perturbed_sharpes) - min(perturbed_sharpes)
            
            # Stability = how close perturbed performance is to baseline
            stability_score = 1.0 - min(sharpe_std / max(abs(baseline_metrics['sharpe']), 1.0), 1.0)
            
            # Flag severe degradation
            if sharpe_range > 2.0:
                fragility_flags.append(f"High parameter sensitivity: Sharpe range {sharpe_range:.2f}")
        else:
            stability_score = 0.0
            fragility_flags.append("No valid parameter perturbations")
        
        passed = stability_score >= self.config.min_stability_score
        
        return RobustnessResult(
            test_type='parameter_perturbation',
            baseline_sharpe=baseline_metrics['sharpe'],
            perturbed_sharpes=perturbed_sharpes,
            stability_score=stability_score,
            fragility_flags=fragility_flags,
            passed=passed
        )
    
    def _test_noise_injection(
        self,
        strategy_func: Callable,
        data: Dict[str, Any],
        params: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> RobustnessResult:
        """
        Test strategy stability under price noise.
        
        Adds random noise to OHLCV data to simulate measurement error.
        """
        perturbed_sharpes = []
        fragility_flags = []
        
        n_runs = 50  # Noise injection runs
        
        for run in range(n_runs):
            # Create noisy data
            noisy_data = self._add_price_noise(data, self.config.noise_level)
            
            try:
                trades = strategy_func(noisy_data, params)
                metrics = self._calculate_metrics(trades)
                perturbed_sharpes.append(metrics['sharpe'])
            except Exception as e:
                fragility_flags.append(f"Failed on noise run {run}: {str(e)[:50]}")
                perturbed_sharpes.append(-999)
        
        # Calculate stability
        if perturbed_sharpes:
            sharpe_std = np.std(perturbed_sharpes)
            stability_score = 1.0 - min(sharpe_std / max(abs(baseline_metrics['sharpe']), 1.0), 1.0)
            
            # Flag if noise causes severe issues
            failed_runs = sum(1 for s in perturbed_sharpes if s < -10)
            if failed_runs > n_runs * 0.1:  # >10% failures
                fragility_flags.append(f"Noise-sensitive: {failed_runs}/{n_runs} runs failed")
        else:
            stability_score = 0.0
            fragility_flags.append("No valid noise runs")
        
        passed = stability_score >= self.config.min_stability_score
        
        return RobustnessResult(
            test_type='noise_injection',
            baseline_sharpe=baseline_metrics['sharpe'],
            perturbed_sharpes=perturbed_sharpes,
            stability_score=stability_score,
            fragility_flags=fragility_flags,
            passed=passed
        )
    
    def _test_regime_shuffle(
        self,
        strategy_func: Callable,
        data: Dict[str, Any],
        params: Dict[str, float],
        baseline_metrics: Dict[str, float]
    ) -> RobustnessResult:
        """
        Test strategy by shuffling market regime periods.
        
        Ensures strategy doesn't depend on specific temporal order.
        """
        perturbed_sharpes = []
        fragility_flags = []
        
        n_shuffles = 30
        chunk_size = max(len(data.get('close', [])) // 10, 50)  # ~10 chunks
        
        for shuffle_run in range(n_shuffles):
            # Shuffle data in chunks to preserve local structure
            shuffled_data = self._shuffle_data_chunks(data, chunk_size)
            
            try:
                trades = strategy_func(shuffled_data, params)
                metrics = self._calculate_metrics(trades)
                perturbed_sharpes.append(metrics['sharpe'])
            except Exception as e:
                fragility_flags.append(f"Failed on shuffle {shuffle_run}: {str(e)[:50]}")
                perturbed_sharpes.append(-999)
        
        # Calculate stability
        if perturbed_sharpes:
            sharpe_std = np.std(perturbed_sharpes)
            stability_score = 1.0 - min(sharpe_std / max(abs(baseline_metrics['sharpe']), 1.0), 1.0)
            
            # Flag order dependency
            if sharpe_std > 1.0:
                fragility_flags.append(f"Order-dependent: Sharpe std {sharpe_std:.2f}")
        else:
            stability_score = 0.0
            fragility_flags.append("No valid shuffles")
        
        passed = stability_score >= self.config.min_stability_score
        
        return RobustnessResult(
            test_type='regime_shuffle',
            baseline_sharpe=baseline_metrics['sharpe'],
            perturbed_sharpes=perturbed_sharpes,
            stability_score=stability_score,
            fragility_flags=fragility_flags,
            passed=passed
        )
    
    def _test_monte_carlo_reorder(
        self,
        baseline_trades: List[Dict],
        baseline_metrics: Dict[str, float]
    ) -> RobustnessResult:
        """
        Test by reordering trades randomly (Monte Carlo).
        
        Ensures good Sharpe isn't due to lucky trade sequence.
        """
        perturbed_sharpes = []
        fragility_flags = []
        
        if not baseline_trades:
            return RobustnessResult(
                test_type='monte_carlo_reorder',
                baseline_sharpe=0.0,
                perturbed_sharpes=[],
                stability_score=0.0,
                fragility_flags=['No trades to reorder'],
                passed=False
            )
        
        pnls = [t.get('pnl', 0.0) for t in baseline_trades]
        
        for _ in range(self.config.monte_carlo_runs):
            # Randomly reorder PnLs
            shuffled_pnls = np.random.permutation(pnls)
            
            # Recalculate Sharpe
            if len(shuffled_pnls) > 1 and np.std(shuffled_pnls) > 0:
                sharpe = np.mean(shuffled_pnls) / np.std(shuffled_pnls) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            perturbed_sharpes.append(sharpe)
        
        # Calculate stability
        sharpe_std = np.std(perturbed_sharpes)
        sharpe_mean = np.mean(perturbed_sharpes)
        
        # Stability = consistency across reorderings
        stability_score = 1.0 - min(sharpe_std / max(abs(sharpe_mean), 1.0), 1.0)
        
        # Flag if baseline is an outlier (lucky sequence)
        sharpe_95th = np.percentile(perturbed_sharpes, 95)
        if baseline_metrics['sharpe'] > sharpe_95th + 0.5:
            fragility_flags.append(
                f"Baseline Sharpe {baseline_metrics['sharpe']:.2f} > 95th percentile {sharpe_95th:.2f}"
            )
        
        passed = stability_score >= self.config.min_stability_score
        
        return RobustnessResult(
            test_type='monte_carlo_reorder',
            baseline_sharpe=baseline_metrics['sharpe'],
            perturbed_sharpes=perturbed_sharpes,
            stability_score=stability_score,
            fragility_flags=fragility_flags,
            passed=passed
        )
    
    def _add_price_noise(self, data: Dict[str, Any], noise_level: float) -> Dict[str, Any]:
        """Add random noise to price data."""
        noisy_data = copy.deepcopy(data)
        
        price_fields = ['open', 'high', 'low', 'close']
        
        for field in price_fields:
            if field in noisy_data and isinstance(noisy_data[field], (list, np.ndarray)):
                prices = np.array(noisy_data[field])
                noise = np.random.normal(0, noise_level, len(prices))
                noisy_prices = prices * (1 + noise)
                noisy_data[field] = noisy_prices.tolist()
        
        return noisy_data
    
    def _shuffle_data_chunks(self, data: Dict[str, Any], chunk_size: int) -> Dict[str, Any]:
        """Shuffle data in chunks to preserve local structure."""
        shuffled_data = copy.deepcopy(data)
        
        for key, values in shuffled_data.items():
            if not isinstance(values, (list, np.ndarray)):
                continue
            
            n = len(values)
            n_chunks = max(n // chunk_size, 1)
            
            # Split into chunks
            chunks = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, n)
                chunks.append(values[start:end])
            
            # Shuffle chunks
            np.random.shuffle(chunks)
            
            # Reassemble
            shuffled_data[key] = [item for chunk in chunks for item in chunk]
        
        return shuffled_data
    
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
        max_dd = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Win Rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls)
        
        return {
            'sharpe': sharpe,
            'expectancy': expectancy,
            'max_dd': max_dd,
            'win_rate': win_rate
        }
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate robustness test results."""
        if not self.results:
            return {
                'overall_stability': 0.0,
                'test_results': [],
                'fragility_flags': [],
                'passed': False
            }
        
        # Overall stability = average of all tests
        stability_scores = [r.stability_score for r in self.results]
        overall_stability = np.mean(stability_scores)
        
        # Collect all fragility flags
        all_flags = []
        for result in self.results:
            all_flags.extend(result.fragility_flags)
        
        # Pass only if ALL tests pass
        all_passed = all(r.passed for r in self.results)
        
        return {
            'overall_stability': overall_stability,
            'test_results': self.results,
            'fragility_flags': all_flags,
            'passed': all_passed,
            'summary': {
                'parameter_perturbation': next(
                    (r.stability_score for r in self.results if r.test_type == 'parameter_perturbation'),
                    0.0
                ),
                'noise_injection': next(
                    (r.stability_score for r in self.results if r.test_type == 'noise_injection'),
                    0.0
                ),
                'regime_shuffle': next(
                    (r.stability_score for r in self.results if r.test_type == 'regime_shuffle'),
                    0.0
                ),
                'monte_carlo': next(
                    (r.stability_score for r in self.results if r.test_type == 'monte_carlo_reorder'),
                    0.0
                )
            }
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about robustness tests."""
        if not self.results:
            return {'status': 'no_tests', 'tests_run': 0}
        
        return {
            'status': 'completed',
            'tests_run': len(self.results),
            'all_passed': all(r.passed for r in self.results),
            'failed_tests': [r.test_type for r in self.results if not r.passed],
            'total_flags': sum(len(r.fragility_flags) for r in self.results)
        }
