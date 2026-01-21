"""
optimization_monitor.py - Monitor and visualize optimization metrics

Tracks and displays:
- Strategy weights over time
- Performance decomposition
- Regime effectiveness
- Signal quality distribution
- Strategy health status

Designed for integration with existing monitoring scripts.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


class OptimizationMonitor:
    """
    Integrates all optimization metrics for monitoring and reporting.
    
    Combines data from:
    - PerformanceDecomposer
    - AdaptiveWeightingOptimizer
    - SignalQualityScorer
    - StrategyRetirementManager
    """
    
    def __init__(self):
        """Initialize monitor."""
        self._import_status = "ready"
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get overall optimization status."""
        try:
            from ai.performance_decomposer import get_decomposer
            from ai.adaptive_weighting_optimizer import get_optimizer
            from ai.signal_quality_scorer import get_scorer
            from ai.strategy_retirement import get_manager
            
            decomposer = get_decomposer()
            optimizer = get_optimizer()
            scorer = get_scorer()
            manager = get_manager()
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': 'active',
                'components': {
                    'performance_decomposer': 'initialized',
                    'adaptive_optimizer': 'initialized',
                    'quality_scorer': 'initialized',
                    'retirement_manager': 'initialized',
                },
                'metrics': {
                    'strategies_tracked': len(decomposer.get_all_performance()),
                    'weight_updates': len(optimizer.get_weight_history()),
                    'signals_scored': len(scorer.get_signal_history()),
                    'health_assessments': len(manager.get_all_health()),
                }
            }
        except Exception as e:
            logger.error(f"Error getting optimization status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def get_strategy_summary(self, strategy_name: str) -> Dict[str, Any]:
        """Get comprehensive summary for a strategy."""
        try:
            from ai.performance_decomposer import get_decomposer
            from ai.adaptive_weighting_optimizer import get_optimizer
            from ai.signal_quality_scorer import get_scorer
            from ai.strategy_retirement import get_manager
            
            decomposer = get_decomposer()
            optimizer = get_optimizer()
            scorer = get_scorer()
            manager = get_manager()
            
            # Get performance
            perf = decomposer.get_strategy_performance(strategy_name)
            perf_dict = perf.to_dict() if perf else {}
            
            # Get weights
            current_weights = optimizer.get_current_weights()
            weight_history = optimizer.get_weight_history(strategy_name)
            
            # Get quality
            quality = scorer.get_quality_summary(strategy_name, 'ALL')
            
            # Get health
            health = manager.get_strategy_health(strategy_name)
            health_dict = health.to_dict() if health else {}
            
            # Get regime breakdown
            regime_perf = decomposer.get_regime_effectiveness(strategy_name)
            regime_breakdown = {
                regime: perf.to_dict()
                for regime, perf in regime_perf.items()
            }
            
            # Get symbol breakdown
            symbol_perf = decomposer.get_symbol_breakdown(strategy_name)
            symbol_breakdown = {
                symbol: perf.to_dict()
                for symbol, perf in symbol_perf.items()
            }
            
            return {
                'strategy_name': strategy_name,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'performance': perf_dict,
                'current_weight': current_weights.get(strategy_name, 'N/A'),
                'recent_weight_updates': [
                    {
                        'timestamp': u.timestamp.isoformat(),
                        'previous': u.previous_weight,
                        'current': u.new_weight,
                        'reason': u.change_reason,
                    }
                    for u in weight_history[-10:]
                ],
                'signal_quality': quality,
                'health_status': health_dict,
                'regime_effectiveness': regime_breakdown,
                'symbol_breakdown': symbol_breakdown,
            }
        except Exception as e:
            logger.error(f"Error getting strategy summary for {strategy_name}: {e}")
            return {'error': str(e)}
    
    def get_allocation_report(self) -> Dict[str, Any]:
        """Get current allocation report."""
        try:
            from ai.adaptive_weighting_optimizer import get_optimizer
            
            optimizer = get_optimizer()
            summary = optimizer.get_allocation_summary()
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'allocation_summary': summary,
                'normalized_weights': optimizer.normalize_weights(),
                'raw_weights': optimizer.get_current_weights(),
            }
        except Exception as e:
            logger.error(f"Error getting allocation report: {e}")
            return {'error': str(e)}
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get strategy health report."""
        try:
            from ai.strategy_retirement import get_manager, StrategyStatus
            
            manager = get_manager()
            all_health = manager.get_all_health()
            
            # Categorize by status
            by_status = defaultdict(list)
            for name, health in all_health.items():
                by_status[health.status.value].append({
                    'name': name,
                    'reason': health.status_reason,
                    'confidence': health.confidence_level,
                    'expectancy': health.recent_expectancy,
                    'profit_factor': health.recent_profit_factor,
                    'win_rate': health.recent_win_rate,
                })
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'total_strategies': len(all_health),
                'by_status': dict(by_status),
                'suspended': manager.get_suspended_strategies(),
                'degraded': manager.get_degraded_strategies(),
                'active': manager.get_active_strategies(),
            }
        except Exception as e:
            logger.error(f"Error getting health report: {e}")
            return {'error': str(e)}
    
    def get_regime_effectiveness_report(self) -> Dict[str, Any]:
        """Get regime-specific performance report."""
        try:
            from ai.performance_decomposer import get_decomposer
            
            decomposer = get_decomposer()
            all_perf = decomposer.get_all_performance()
            
            # Group by regime
            by_regime = defaultdict(list)
            for key, perf in all_perf.items():
                if perf.regime:
                    by_regime[perf.regime].append({
                        'strategy': perf.strategy_name,
                        'symbol': perf.symbol,
                        'expectancy': perf.expectancy,
                        'profit_factor': perf.profit_factor,
                        'win_rate': perf.win_rate,
                        'trades': perf.total_trades,
                        'sharpe': perf.sharpe_ratio,
                    })
            
            return {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'by_regime': dict(by_regime),
            }
        except Exception as e:
            logger.error(f"Error getting regime effectiveness report: {e}")
            return {'error': str(e)}
    
    def export_json(self, filename: str) -> bool:
        """Export all optimization data to JSON."""
        try:
            data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'optimization_status': self.get_optimization_status(),
                'allocation_report': self.get_allocation_report(),
                'health_report': self.get_health_report(),
                'regime_effectiveness': self.get_regime_effectiveness_report(),
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Exported optimization data to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting to {filename}: {e}")
            return False
    
    def get_dataframe_export(self, data_type: str = 'performance') -> Optional[Any]:
        """Export data as pandas DataFrame."""
        try:
            if pd is None:
                logger.warning("pandas not available")
                return None
            
            from ai.performance_decomposer import get_decomposer
            
            if data_type == 'performance':
                decomposer = get_decomposer()
                all_perf = decomposer.get_all_performance()
                
                rows = []
                for key, perf in all_perf.items():
                    rows.append({
                        'strategy': perf.strategy_name,
                        'symbol': perf.symbol,
                        'regime': perf.regime,
                        'total_trades': perf.total_trades,
                        'win_rate': perf.win_rate,
                        'expectancy': perf.expectancy,
                        'profit_factor': perf.profit_factor,
                        'max_drawdown': perf.max_drawdown,
                        'sharpe_ratio': perf.sharpe_ratio,
                        'total_pnl': perf.total_pnl,
                    })
                
                return pd.DataFrame(rows)
            
            else:
                logger.warning(f"Unknown data type: {data_type}")
                return None
        
        except Exception as e:
            logger.error(f"Error creating DataFrame: {e}")
            return None


# Module-level convenience instance
_default_monitor = OptimizationMonitor()


def get_monitor() -> OptimizationMonitor:
    """Get the default monitor instance."""
    return _default_monitor


def get_optimization_status() -> Dict[str, Any]:
    """Convenience function."""
    return _default_monitor.get_optimization_status()


def get_strategy_summary(strategy_name: str) -> Dict[str, Any]:
    """Convenience function."""
    return _default_monitor.get_strategy_summary(strategy_name)


def get_health_report() -> Dict[str, Any]:
    """Convenience function."""
    return _default_monitor.get_health_report()
