"""
strategy_retirement.py - Automatic Strategy Retirement & Suspension

Detects underperforming strategies and automatically:
- Reduces allocation
- Suspends temporarily
- Flags for review
- Re-enables when performance improves

Design:
- Non-invasive health checks
- Gradual de-weighting before suspension
- Automatic recovery when conditions improve
- Detailed audit trail
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import threading

logger = logging.getLogger(__name__)


class StrategyStatus(Enum):
    """Strategy operational status."""
    ACTIVE = "active"  # Normal operation
    DEGRADED = "degraded"  # Warning stage
    SUSPENDED = "suspended"  # Temporarily disabled
    RETIRED = "retired"  # Permanently disabled


@dataclass
class StrategyHealthMetrics:
    """Health assessment for a strategy."""
    strategy_name: str
    status: StrategyStatus
    timestamp: datetime
    
    # Performance metrics
    recent_expectancy: float  # Recent trades expectancy
    recent_profit_factor: float
    recent_win_rate: float
    recent_trades_count: int
    
    # Health indicators
    max_drawdown_recent: float
    consecutive_losses: int
    max_consecutive_losses: int
    
    # Status reasons
    status_reason: str = ""
    confidence_level: float = 1.0  # 0.0-1.0
    
    # History
    degraded_since: Optional[datetime] = None
    suspended_since: Optional[datetime] = None
    last_recovered: Optional[datetime] = None
    recovery_attempts: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'strategy_name': self.strategy_name,
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'recent_expectancy': self.recent_expectancy,
            'recent_profit_factor': self.recent_profit_factor,
            'recent_win_rate': self.recent_win_rate,
            'recent_trades_count': self.recent_trades_count,
            'max_drawdown_recent': self.max_drawdown_recent,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_losses': self.max_consecutive_losses,
            'status_reason': self.status_reason,
            'confidence_level': self.confidence_level,
        }


class StrategyRetirementManager:
    """
    Manages strategy health and automatically retires underperformers.
    
    Operates in stages:
    1. ACTIVE: Normal operation
    2. DEGRADED: Warning stage, reduced allocation
    3. SUSPENDED: Temporarily disabled, waiting for recovery
    4. RETIRED: Permanently disabled
    """
    
    def __init__(
        self,
        # Performance thresholds
        min_expectancy: float = 0.1,  # Min expectancy to stay active
        min_profit_factor: float = 0.8,  # Min PF to stay active
        min_win_rate: float = 0.35,  # Min win rate to stay active
        max_drawdown: float = 0.15,  # Max DD before degradation
        max_consecutive_losses: int = 10,  # Max consecutive losses
        # Timing
        min_trades_for_assessment: int = 20,  # Don't assess until N trades
        degraded_window_hours: int = 24,  # Time in degraded before suspend
        suspended_window_hours: int = 72,  # Time in suspended before retire
        recovery_window_hours: int = 48,  # Time in degraded to recover
        # Safety
        recovery_improvement_percent: float = 0.15,  # 15% improvement needed
    ):
        """
        Initialize manager.
        
        Args:
            min_expectancy: Minimum expectancy threshold
            min_profit_factor: Minimum profit factor threshold
            min_win_rate: Minimum win rate threshold
            max_drawdown: Maximum acceptable drawdown
            max_consecutive_losses: Maximum consecutive losses before alert
            min_trades_for_assessment: Minimum trades to assess
            degraded_window_hours: Hours before moving degraded -> suspended
            suspended_window_hours: Hours before retiring suspended strategy
            recovery_window_hours: Hours for strategy to recover from degraded
            recovery_improvement_percent: Improvement required to recover
        """
        self._lock = threading.RLock()
        
        # Configuration
        self.min_expectancy = min_expectancy
        self.min_profit_factor = min_profit_factor
        self.min_win_rate = min_win_rate
        self.max_drawdown = max_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.min_trades_for_assessment = min_trades_for_assessment
        self.degraded_window_hours = degraded_window_hours
        self.suspended_window_hours = suspended_window_hours
        self.recovery_window_hours = recovery_window_hours
        self.recovery_improvement_percent = recovery_improvement_percent
        
        # Status tracking (per strategy)
        self._health: Dict[str, StrategyHealthMetrics] = {}
        self._audit_log: List[Dict[str, Any]] = []
        
        logger.info("StrategyRetirementManager initialized")
    
    def assess_strategy(
        self,
        strategy_name: str,
        recent_expectancy: float,
        recent_profit_factor: float,
        recent_win_rate: float,
        recent_trades_count: int,
        max_drawdown_recent: float = 0.0,
        consecutive_losses: int = 0,
        max_consecutive_losses: int = 0,
    ) -> StrategyHealthMetrics:
        """
        Assess strategy health and update status.
        
        Args:
            strategy_name: Name of strategy
            recent_expectancy: Recent expectancy metric
            recent_profit_factor: Recent profit factor
            recent_win_rate: Recent win rate (0.0-1.0)
            recent_trades_count: Number of recent trades
            max_drawdown_recent: Recent max drawdown
            consecutive_losses: Current consecutive losses
            max_consecutive_losses: Maximum consecutive losses seen
            
        Returns:
            Updated StrategyHealthMetrics
        """
        with self._lock:
            # Get current health if exists
            current_health = self._health.get(strategy_name)
            
            # Insufficient data
            if recent_trades_count < self.min_trades_for_assessment:
                status = StrategyStatus.ACTIVE
                reason = f"Insufficient trades ({recent_trades_count}/{self.min_trades_for_assessment})"
                confidence = 0.3
            else:
                # Assess performance
                violations = self._check_thresholds(
                    recent_expectancy,
                    recent_profit_factor,
                    recent_win_rate,
                    max_drawdown_recent,
                    consecutive_losses,
                )
                
                # Determine status
                status, reason, confidence = self._determine_status(
                    violations,
                    current_health,
                )
            
            # Create health metrics
            metrics = StrategyHealthMetrics(
                strategy_name=strategy_name,
                status=status,
                timestamp=datetime.now(timezone.utc),
                recent_expectancy=recent_expectancy,
                recent_profit_factor=recent_profit_factor,
                recent_win_rate=recent_win_rate,
                recent_trades_count=recent_trades_count,
                max_drawdown_recent=max_drawdown_recent,
                consecutive_losses=consecutive_losses,
                max_consecutive_losses=max_consecutive_losses,
                status_reason=reason,
                confidence_level=confidence,
            )
            
            # Preserve timing info from previous health
            if current_health:
                metrics.degraded_since = current_health.degraded_since
                metrics.suspended_since = current_health.suspended_since
                metrics.last_recovered = current_health.last_recovered
                metrics.recovery_attempts = current_health.recovery_attempts
                
                # Update timing if status changed
                if status != current_health.status:
                    if status == StrategyStatus.DEGRADED:
                        metrics.degraded_since = datetime.now(timezone.utc)
                    elif status == StrategyStatus.SUSPENDED:
                        metrics.suspended_since = datetime.now(timezone.utc)
                    elif status == StrategyStatus.ACTIVE:
                        metrics.last_recovered = datetime.now(timezone.utc)
                        metrics.recovery_attempts += 1
            else:
                if status == StrategyStatus.DEGRADED:
                    metrics.degraded_since = datetime.now(timezone.utc)
            
            # Update cache
            self._health[strategy_name] = metrics
            
            # Log if status changed
            if not current_health or status != current_health.status:
                self._log_status_change(strategy_name, current_health, metrics)
            
            logger.info(
                f"Strategy health check: {strategy_name} = {status.value} "
                f"(exp={recent_expectancy:.2f}, pf={recent_profit_factor:.2f}, "
                f"wr={recent_win_rate:.2f}) - {reason}"
            )
            
            return metrics
    
    def _check_thresholds(
        self,
        expectancy: float,
        profit_factor: float,
        win_rate: float,
        max_drawdown: float,
        consecutive_losses: int,
    ) -> List[str]:
        """Check which thresholds are violated."""
        violations = []
        
        if expectancy < self.min_expectancy:
            violations.append(f"low_expectancy ({expectancy:.2f} < {self.min_expectancy})")
        if profit_factor < self.min_profit_factor:
            violations.append(f"low_pf ({profit_factor:.2f} < {self.min_profit_factor})")
        if win_rate < self.min_win_rate:
            violations.append(f"low_wr ({win_rate:.2f} < {self.min_win_rate})")
        if max_drawdown > self.max_drawdown:
            violations.append(f"high_dd ({max_drawdown:.2f} > {self.max_drawdown})")
        if consecutive_losses > self.max_consecutive_losses:
            violations.append(f"consecutive_losses ({consecutive_losses} > {self.max_consecutive_losses})")
        
        return violations
    
    def _determine_status(
        self,
        violations: List[str],
        current_health: Optional[StrategyHealthMetrics],
    ) -> Tuple[StrategyStatus, str, float]:
        """Determine strategy status based on violations."""
        
        # No violations -> Active
        if not violations:
            return StrategyStatus.ACTIVE, "All metrics healthy", 1.0
        
        # Has violations
        reason = ", ".join(violations)
        
        # Check current status for state transitions
        if not current_health:
            # New strategy with violations
            return StrategyStatus.DEGRADED, reason, 0.6
        
        current_status = current_health.status
        
        if current_status == StrategyStatus.ACTIVE:
            # Transition to degraded
            return StrategyStatus.DEGRADED, reason, 0.6
        
        elif current_status == StrategyStatus.DEGRADED:
            # Check if should suspend
            if current_health.degraded_since:
                time_degraded = datetime.now(timezone.utc) - current_health.degraded_since
                if time_degraded > timedelta(hours=self.degraded_window_hours):
                    return StrategyStatus.SUSPENDED, f"Degraded too long: {reason}", 0.8
            
            return StrategyStatus.DEGRADED, reason, 0.7
        
        elif current_status == StrategyStatus.SUSPENDED:
            # Check if should retire
            if current_health.suspended_since:
                time_suspended = datetime.now(timezone.utc) - current_health.suspended_since
                if time_suspended > timedelta(hours=self.suspended_window_hours):
                    return StrategyStatus.RETIRED, f"Suspended too long: {reason}", 0.9
            
            return StrategyStatus.SUSPENDED, reason, 0.8
        
        elif current_status == StrategyStatus.RETIRED:
            return StrategyStatus.RETIRED, reason, 1.0
        
        return StrategyStatus.ACTIVE, "Unknown state", 0.0
    
    def _log_status_change(
        self,
        strategy_name: str,
        old_health: Optional[StrategyHealthMetrics],
        new_health: StrategyHealthMetrics,
    ) -> None:
        """Log a status change."""
        old_status = old_health.status.value if old_health else "unknown"
        new_status = new_health.status.value
        
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'strategy': strategy_name,
            'old_status': old_status,
            'new_status': new_status,
            'reason': new_health.status_reason,
            'metrics': {
                'expectancy': new_health.recent_expectancy,
                'profit_factor': new_health.recent_profit_factor,
                'win_rate': new_health.recent_win_rate,
                'trades': new_health.recent_trades_count,
            }
        }
        
        self._audit_log.append(log_entry)
        
        logger.warning(
            f"Strategy status changed: {strategy_name} "
            f"{old_status} -> {new_status} ({new_health.status_reason})"
        )
    
    def get_strategy_health(self, strategy_name: str) -> Optional[StrategyHealthMetrics]:
        """Get current health metrics for a strategy."""
        with self._lock:
            return self._health.get(strategy_name)
    
    def get_all_health(self) -> Dict[str, StrategyHealthMetrics]:
        """Get health for all strategies."""
        with self._lock:
            return dict(self._health)
    
    def get_suspended_strategies(self) -> List[str]:
        """Get list of suspended strategies."""
        with self._lock:
            return [
                name for name, health in self._health.items()
                if health.status == StrategyStatus.SUSPENDED
            ]
    
    def get_degraded_strategies(self) -> List[str]:
        """Get list of degraded strategies."""
        with self._lock:
            return [
                name for name, health in self._health.items()
                if health.status == StrategyStatus.DEGRADED
            ]
    
    def get_active_strategies(self) -> List[str]:
        """Get list of active strategies."""
        with self._lock:
            return [
                name for name, health in self._health.items()
                if health.status == StrategyStatus.ACTIVE
            ]
    
    def get_audit_log(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get audit log of status changes."""
        with self._lock:
            if strategy_name:
                return [e for e in self._audit_log if e['strategy'] == strategy_name]
            return list(self._audit_log)
    
    def force_status(self, strategy_name: str, status: StrategyStatus, reason: str) -> None:
        """Force a strategy to a specific status (admin override)."""
        with self._lock:
            current = self._health.get(strategy_name)
            if not current:
                # Create a baseline health record if none exists
                current = StrategyHealthMetrics(
                    strategy_name=strategy_name,
                    status=status,
                    timestamp=datetime.now(timezone.utc),
                    recent_expectancy=0.0,
                    recent_profit_factor=0.0,
                    recent_win_rate=0.0,
                    recent_trades_count=0,
                    max_drawdown_recent=0.0,
                    consecutive_losses=0,
                    max_consecutive_losses=0,
                    status_reason=reason,
                )
                self._health[strategy_name] = current

            # Apply forced status
            old = StrategyHealthMetrics(
                strategy_name=current.strategy_name,
                status=current.status,
                timestamp=current.timestamp,
                recent_expectancy=current.recent_expectancy,
                recent_profit_factor=current.recent_profit_factor,
                recent_win_rate=current.recent_win_rate,
                recent_trades_count=current.recent_trades_count,
                max_drawdown_recent=current.max_drawdown_recent,
                consecutive_losses=current.consecutive_losses,
                max_consecutive_losses=current.max_consecutive_losses,
                status_reason=current.status_reason,
            )

            current.status = status
            current.status_reason = reason
            current.timestamp = datetime.now(timezone.utc)
            self._log_status_change(strategy_name, old, current)
            logger.warning(f"Forced status for {strategy_name}: {status.value} ({reason})")


# Module-level convenience instance
_default_manager: Optional[StrategyRetirementManager] = None


def initialize_manager(**kwargs) -> StrategyRetirementManager:
    """Initialize default manager instance."""
    global _default_manager
    _default_manager = StrategyRetirementManager(**kwargs)
    return _default_manager


def get_manager() -> StrategyRetirementManager:
    """Get the default manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = StrategyRetirementManager()
    return _default_manager


def assess_strategy(strategy_name: str, **metrics) -> StrategyHealthMetrics:
    """Convenience function using default manager."""
    return get_manager().assess_strategy(strategy_name, **metrics)
