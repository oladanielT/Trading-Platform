"""
drawdown_guard.py - Drawdown Monitor and Trade Veto System

Monitors account drawdown (daily and total) and vetoes trades when drawdown
limits are exceeded. Provides protective circuit breaker functionality for
risk management without generating trading signals.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VetoReason(Enum):
    """Reasons for trade veto."""
    DAILY_DRAWDOWN_EXCEEDED = "daily_drawdown_exceeded"
    TOTAL_DRAWDOWN_EXCEEDED = "total_drawdown_exceeded"
    CONSECUTIVE_LOSSES_EXCEEDED = "consecutive_losses_exceeded"
    TRADING_HALTED = "trading_halted"
    COOLDOWN_PERIOD = "cooldown_period"


@dataclass
class DrawdownMetrics:
    """
    Current drawdown metrics.
    
    Attributes:
        current_equity: Current account equity
        peak_equity: All-time peak equity
        daily_start_equity: Equity at start of current day
        total_drawdown: Total drawdown from peak
        total_drawdown_pct: Total drawdown percentage
        daily_drawdown: Drawdown for current day
        daily_drawdown_pct: Daily drawdown percentage
        consecutive_losses: Number of consecutive losing trades
        last_updated: Timestamp of last update
    """
    current_equity: float
    peak_equity: float
    daily_start_equity: float
    total_drawdown: float
    total_drawdown_pct: float
    daily_drawdown: float
    daily_drawdown_pct: float
    consecutive_losses: int
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'daily_start_equity': self.daily_start_equity,
            'total_drawdown': self.total_drawdown,
            'total_drawdown_pct': self.total_drawdown_pct,
            'daily_drawdown': self.daily_drawdown,
            'daily_drawdown_pct': self.daily_drawdown_pct,
            'consecutive_losses': self.consecutive_losses,
            'last_updated': self.last_updated.isoformat()
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"DrawdownMetrics(equity=${self.current_equity:.2f}, "
                f"daily_dd={self.daily_drawdown_pct:.2f}%, "
                f"total_dd={self.total_drawdown_pct:.2f}%)")


@dataclass
class VetoDecision:
    """
    Trade veto decision.
    
    Attributes:
        approved: Whether trade is approved (not vetoed)
        veto_reason: Reason for veto if applicable
        current_metrics: Current drawdown metrics
        limits: Applied drawdown limits
        warnings: List of warning messages
        metadata: Additional decision context
    """
    approved: bool
    veto_reason: Optional[VetoReason] = None
    current_metrics: Optional[DrawdownMetrics] = None
    limits: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'approved': self.approved,
            'veto_reason': self.veto_reason.value if self.veto_reason else None,
            'current_metrics': self.current_metrics.to_dict() if self.current_metrics else None,
            'limits': self.limits,
            'warnings': self.warnings,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        if self.approved:
            return "VetoDecision(approved=True)"
        return f"VetoDecision(approved=False, reason={self.veto_reason.value})"


class DrawdownGuard:
    """
    Drawdown monitor and trade veto system.
    
    Tracks account equity, calculates drawdown metrics, and vetoes trades
    when drawdown limits are exceeded. Provides circuit breaker protection
    without generating trading signals.
    """
    
    def __init__(
        self,
        initial_equity: float,
        max_daily_drawdown: float = 0.02,
        max_total_drawdown: float = 0.10,
        max_consecutive_losses: Optional[int] = None,
        cooldown_hours: int = 24,
        enable_auto_halt: bool = True
    ):
        """
        Initialize drawdown guard.
        
        Args:
            initial_equity: Starting account equity
            max_daily_drawdown: Maximum daily drawdown (0.02 = 2%)
            max_total_drawdown: Maximum total drawdown (0.10 = 10%)
            max_consecutive_losses: Max consecutive losses before halt (None = disabled)
            cooldown_hours: Hours to wait after drawdown breach before resuming
            enable_auto_halt: Automatically halt trading on limit breach
        """
        if initial_equity <= 0:
            raise ValueError(f"initial_equity must be positive, got {initial_equity}")
        
        if not 0 < max_daily_drawdown <= 1:
            raise ValueError(f"max_daily_drawdown must be between 0 and 1, got {max_daily_drawdown}")
        
        if not 0 < max_total_drawdown <= 1:
            raise ValueError(f"max_total_drawdown must be between 0 and 1, got {max_total_drawdown}")
        
        # State
        self.current_equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.current_date = date.today()
        self.consecutive_losses = 0
        
        # Limits
        self.max_daily_drawdown = max_daily_drawdown
        self.max_total_drawdown = max_total_drawdown
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_hours = cooldown_hours
        self.enable_auto_halt = enable_auto_halt
        
        # Halting state
        self.is_halted = False
        self.halt_timestamp: Optional[datetime] = None
        self.halt_reason: Optional[VetoReason] = None
        
        # History
        self.equity_history: List[Tuple[datetime, float]] = [(datetime.now(), initial_equity)]
        self.drawdown_breaches: List[Dict[str, Any]] = []
        
        logger.info(
            f"DrawdownGuard initialized: equity=${initial_equity:.2f}, "
            f"max_daily_dd={max_daily_drawdown*100:.1f}%, "
            f"max_total_dd={max_total_drawdown*100:.1f}%"
        )
    
    def check_trade(self) -> VetoDecision:
        """
        Check if a trade should be allowed based on current drawdown.
        
        Returns:
            VetoDecision indicating if trade is approved or vetoed
        """
        # Update daily reset if needed
        self._check_daily_reset()
        
        # Get current metrics
        metrics = self.get_current_metrics()
        
        # Build limits dict for response
        limits = {
            'max_daily_drawdown_pct': self.max_daily_drawdown * 100,
            'max_total_drawdown_pct': self.max_total_drawdown * 100,
            'max_consecutive_losses': self.max_consecutive_losses
        }
        
        # Check if in cooldown period
        if self.is_halted and self.halt_timestamp:
            cooldown_end = self.halt_timestamp + timedelta(hours=self.cooldown_hours)
            if datetime.now() < cooldown_end:
                remaining = (cooldown_end - datetime.now()).total_seconds() / 3600
                logger.warning(
                    f"Trading halted - cooldown period active. "
                    f"Remaining: {remaining:.1f} hours"
                )
                return VetoDecision(
                    approved=False,
                    veto_reason=VetoReason.COOLDOWN_PERIOD,
                    current_metrics=metrics,
                    limits=limits,
                    warnings=[f"In cooldown period. {remaining:.1f} hours remaining"],
                    metadata={'cooldown_end': cooldown_end.isoformat()}
                )
            else:
                # Cooldown period ended
                self._resume_trading()
        
        # Check if manually halted
        if self.is_halted:
            logger.warning("Trading manually halted")
            return VetoDecision(
                approved=False,
                veto_reason=VetoReason.TRADING_HALTED,
                current_metrics=metrics,
                limits=limits,
                warnings=["Trading manually halted"]
            )
        
        warnings = []
        
        # Check daily drawdown
        if metrics.daily_drawdown_pct > self.max_daily_drawdown * 100:
            logger.error(
                f"Daily drawdown limit exceeded: "
                f"{metrics.daily_drawdown_pct:.2f}% > {self.max_daily_drawdown*100:.1f}%"
            )
            
            if self.enable_auto_halt:
                self._halt_trading(VetoReason.DAILY_DRAWDOWN_EXCEEDED)
            
            self._log_breach('daily', metrics)
            
            return VetoDecision(
                approved=False,
                veto_reason=VetoReason.DAILY_DRAWDOWN_EXCEEDED,
                current_metrics=metrics,
                limits=limits,
                warnings=[
                    f"Daily drawdown {metrics.daily_drawdown_pct:.2f}% "
                    f"exceeds limit {self.max_daily_drawdown*100:.1f}%"
                ]
            )
        
        # Check total drawdown
        if metrics.total_drawdown_pct > self.max_total_drawdown * 100:
            logger.error(
                f"Total drawdown limit exceeded: "
                f"{metrics.total_drawdown_pct:.2f}% > {self.max_total_drawdown*100:.1f}%"
            )
            
            if self.enable_auto_halt:
                self._halt_trading(VetoReason.TOTAL_DRAWDOWN_EXCEEDED)
            
            self._log_breach('total', metrics)
            
            return VetoDecision(
                approved=False,
                veto_reason=VetoReason.TOTAL_DRAWDOWN_EXCEEDED,
                current_metrics=metrics,
                limits=limits,
                warnings=[
                    f"Total drawdown {metrics.total_drawdown_pct:.2f}% "
                    f"exceeds limit {self.max_total_drawdown*100:.1f}%"
                ]
            )
        
        # Check consecutive losses
        if self.max_consecutive_losses and metrics.consecutive_losses >= self.max_consecutive_losses:
            logger.error(
                f"Consecutive loss limit exceeded: "
                f"{metrics.consecutive_losses} >= {self.max_consecutive_losses}"
            )
            
            if self.enable_auto_halt:
                self._halt_trading(VetoReason.CONSECUTIVE_LOSSES_EXCEEDED)
            
            self._log_breach('consecutive_losses', metrics)
            
            return VetoDecision(
                approved=False,
                veto_reason=VetoReason.CONSECUTIVE_LOSSES_EXCEEDED,
                current_metrics=metrics,
                limits=limits,
                warnings=[
                    f"Consecutive losses {metrics.consecutive_losses} "
                    f"exceeds limit {self.max_consecutive_losses}"
                ]
            )
        
        # Warning thresholds (80% of limit)
        daily_warning_threshold = self.max_daily_drawdown * 0.8 * 100
        total_warning_threshold = self.max_total_drawdown * 0.8 * 100
        
        if metrics.daily_drawdown_pct > daily_warning_threshold:
            warning = (f"⚠️  Daily drawdown approaching limit: "
                      f"{metrics.daily_drawdown_pct:.2f}% "
                      f"(limit: {self.max_daily_drawdown*100:.1f}%)")
            warnings.append(warning)
            logger.warning(warning)
        
        if metrics.total_drawdown_pct > total_warning_threshold:
            warning = (f"⚠️  Total drawdown approaching limit: "
                      f"{metrics.total_drawdown_pct:.2f}% "
                      f"(limit: {self.max_total_drawdown*100:.1f}%)")
            warnings.append(warning)
            logger.warning(warning)
        
        if self.max_consecutive_losses and metrics.consecutive_losses >= self.max_consecutive_losses * 0.8:
            warning = (f"⚠️  Approaching consecutive loss limit: "
                      f"{metrics.consecutive_losses} "
                      f"(limit: {self.max_consecutive_losses})")
            warnings.append(warning)
            logger.warning(warning)
        
        # Trade approved
        return VetoDecision(
            approved=True,
            current_metrics=metrics,
            limits=limits,
            warnings=warnings
        )
    
    def update_equity(self, new_equity: float, is_loss: Optional[bool] = None) -> None:
        """
        Update current equity and recalculate drawdown metrics.
        
        Args:
            new_equity: New account equity value
            is_loss: Whether last trade was a loss (for consecutive loss tracking)
        """
        if new_equity <= 0:
            raise ValueError(f"new_equity must be positive, got {new_equity}")
        
        old_equity = self.current_equity
        self.current_equity = new_equity
        
        # Update peak
        if new_equity > self.peak_equity:
            self.peak_equity = new_equity
            self.consecutive_losses = 0  # Reset on new peak
            logger.info(f"New equity peak: ${new_equity:.2f}")
        
        # Update consecutive losses
        if is_loss is not None:
            if is_loss:
                self.consecutive_losses += 1
                if self.consecutive_losses > 1:
                    logger.warning(f"Consecutive losses: {self.consecutive_losses}")
            else:
                self.consecutive_losses = 0
        
        # Record in history
        self.equity_history.append((datetime.now(), new_equity))
        
        # Check daily reset
        self._check_daily_reset()
        
        logger.info(
            f"Equity updated: ${old_equity:.2f} -> ${new_equity:.2f} "
            f"({((new_equity/old_equity - 1) * 100):+.2f}%)"
        )
    
    def get_current_metrics(self) -> DrawdownMetrics:
        """
        Get current drawdown metrics.
        
        Returns:
            DrawdownMetrics with current state
        """
        # Calculate total drawdown
        total_drawdown = self.peak_equity - self.current_equity
        total_drawdown_pct = (total_drawdown / self.peak_equity) * 100 if self.peak_equity > 0 else 0
        
        # Calculate daily drawdown
        daily_drawdown = self.daily_start_equity - self.current_equity
        daily_drawdown_pct = (daily_drawdown / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0
        
        return DrawdownMetrics(
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            daily_start_equity=self.daily_start_equity,
            total_drawdown=total_drawdown,
            total_drawdown_pct=total_drawdown_pct,
            daily_drawdown=daily_drawdown,
            daily_drawdown_pct=daily_drawdown_pct,
            consecutive_losses=self.consecutive_losses,
            last_updated=datetime.now()
        )
    
    def halt_trading(self, reason: Optional[str] = None) -> None:
        """
        Manually halt trading.
        
        Args:
            reason: Optional reason for halt
        """
        self.is_halted = True
        self.halt_timestamp = datetime.now()
        self.halt_reason = VetoReason.TRADING_HALTED
        
        logger.warning(f"Trading manually halted. Reason: {reason or 'Manual halt'}")
    
    def resume_trading(self) -> None:
        """Manually resume trading."""
        self._resume_trading()
        logger.info("Trading manually resumed")
    
    def reset_daily_metrics(self) -> None:
        """Reset daily drawdown tracking."""
        self.daily_start_equity = self.current_equity
        self.current_date = date.today()
        logger.info(f"Daily metrics reset. Starting equity: ${self.current_equity:.2f}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status report.
        
        Returns:
            Dictionary with current status
        """
        metrics = self.get_current_metrics()
        
        return {
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason.value if self.halt_reason else None,
            'halt_timestamp': self.halt_timestamp.isoformat() if self.halt_timestamp else None,
            'metrics': metrics.to_dict(),
            'limits': {
                'max_daily_drawdown_pct': self.max_daily_drawdown * 100,
                'max_total_drawdown_pct': self.max_total_drawdown * 100,
                'max_consecutive_losses': self.max_consecutive_losses
            },
            'drawdown_breaches': len(self.drawdown_breaches),
            'equity_history_length': len(self.equity_history)
        }
    
    def get_breach_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of drawdown breaches.
        
        Args:
            limit: Maximum number of recent breaches to return
            
        Returns:
            List of breach records
        """
        if limit:
            return self.drawdown_breaches[-limit:]
        return self.drawdown_breaches.copy()
    
    def _check_daily_reset(self) -> None:
        """Check if we need to reset daily metrics."""
        today = date.today()
        if today > self.current_date:
            self.reset_daily_metrics()
    
    def _halt_trading(self, reason: VetoReason) -> None:
        """Halt trading with specified reason."""
        self.is_halted = True
        self.halt_timestamp = datetime.now()
        self.halt_reason = reason
        
        logger.error(
            f"Trading automatically halted: {reason.value}. "
            f"Cooldown: {self.cooldown_hours} hours"
        )
    
    def _resume_trading(self) -> None:
        """Resume trading after cooldown."""
        self.is_halted = False
        self.halt_timestamp = None
        self.halt_reason = None
        
        logger.info("Trading resumed after cooldown period")
    
    def _log_breach(self, breach_type: str, metrics: DrawdownMetrics) -> None:
        """Log a drawdown breach event."""
        breach_record = {
            'timestamp': datetime.now().isoformat(),
            'type': breach_type,
            'metrics': metrics.to_dict()
        }
        
        self.drawdown_breaches.append(breach_record)
        
        logger.error(f"Drawdown breach logged: {breach_type}")
    
    def __repr__(self) -> str:
        """String representation."""
        metrics = self.get_current_metrics()
        return (f"DrawdownGuard(equity=${metrics.current_equity:.2f}, "
                f"daily_dd={metrics.daily_drawdown_pct:.2f}%, "
                f"total_dd={metrics.total_drawdown_pct:.2f}%, "
                f"halted={self.is_halted})")


# Example usage
if __name__ == "__main__":
    print("=== Drawdown Guard Example ===\n")
    
    # Initialize guard
    initial_equity = 10000.0
    guard = DrawdownGuard(
        initial_equity=initial_equity,
        max_daily_drawdown=0.02,  # 2%
        max_total_drawdown=0.10,  # 10%
        max_consecutive_losses=5,
        cooldown_hours=24,
        enable_auto_halt=True
    )
    
    print(f"Initial state: {guard}\n")
    
    # Test 1: Normal trading
    print("1. Normal Trading Scenario")
    print("-" * 50)
    
    decision = guard.check_trade()
    print(f"Trade check: {decision}")
    print(f"Approved: {decision.approved}")
    print(f"Metrics: {decision.current_metrics}\n")
    
    # Test 2: Small loss
    print("2. Small Loss (within limits)")
    print("-" * 50)
    guard.update_equity(9900, is_loss=True)  # 1% loss
    
    decision = guard.check_trade()
    print(f"Trade check: {decision}")
    print(f"Warnings: {decision.warnings}\n")
    
    # Test 3: Approaching daily limit
    print("3. Approaching Daily Drawdown Limit")
    print("-" * 50)
    guard.update_equity(9850, is_loss=True)  # 1.5% daily loss
    
    decision = guard.check_trade()
    print(f"Trade check: {decision}")
    print(f"Warnings: {decision.warnings}\n")
    
    # Test 4: Exceed daily drawdown
    print("4. Exceeding Daily Drawdown Limit")
    print("-" * 50)
    guard.update_equity(9750, is_loss=True)  # 2.5% daily loss (exceeds 2% limit)
    
    decision = guard.check_trade()
    print(f"Trade check: {decision}")
    print(f"Approved: {decision.approved}")
    print(f"Veto reason: {decision.veto_reason.value if decision.veto_reason else None}")
    print(f"Is halted: {guard.is_halted}\n")
    
    # Test 5: Cooldown period
    print("5. During Cooldown Period")
    print("-" * 50)
    
    decision = guard.check_trade()
    print(f"Trade check: {decision}")
    print(f"Approved: {decision.approved}")
    print(f"Warnings: {decision.warnings}\n")
    
    # Test 6: Manual resume
    print("6. Manual Resume")
    print("-" * 50)
    guard.resume_trading()
    guard.update_equity(9900)  # Recover slightly
    
    decision = guard.check_trade()
    print(f"Trade check: {decision}")
    print(f"Approved: {decision.approved}")
    print(f"Is halted: {guard.is_halted}\n")
    
    # Test 7: Consecutive losses
    print("7. Consecutive Losses")
    print("-" * 50)
    
    for i in range(5):
        guard.update_equity(guard.current_equity - 50, is_loss=True)
        print(f"Loss {i+1}: Consecutive losses = {guard.consecutive_losses}")
    
    decision = guard.check_trade()
    print(f"\nTrade check after 5 losses: {decision}")
    print(f"Approved: {decision.approved}")
    print(f"Veto reason: {decision.veto_reason.value if decision.veto_reason else None}\n")
    
    # Test 8: Recovery and new peak
    print("8. Recovery to New Peak")
    print("-" * 50)
    guard.resume_trading()
    guard.update_equity(10100, is_loss=False)  # New peak
    
    metrics = guard.get_current_metrics()
    print(f"New peak: ${metrics.peak_equity:.2f}")
    print(f"Total drawdown: {metrics.total_drawdown_pct:.2f}%")
    print(f"Consecutive losses reset to: {metrics.consecutive_losses}\n")
    
    # Test 9: Status report
    print("9. Status Report")
    print("-" * 50)
    status = guard.get_status()
    print(f"Is halted: {status['is_halted']}")
    print(f"Current equity: ${status['metrics']['current_equity']:.2f}")
    print(f"Peak equity: ${status['metrics']['peak_equity']:.2f}")
    print(f"Daily drawdown: {status['metrics']['daily_drawdown_pct']:.2f}%")
    print(f"Total drawdown: {status['metrics']['total_drawdown_pct']:.2f}%")
    print(f"Total breaches: {status['drawdown_breaches']}\n")
    
    # Test 10: Breach history
    print("10. Breach History")
    print("-" * 50)
    breaches = guard.get_breach_history()
    print(f"Total breaches recorded: {len(breaches)}")
    for i, breach in enumerate(breaches, 1):
        print(f"  Breach {i}: {breach['type']} at {breach['timestamp']}")
    
    print("\n✓ All examples completed successfully")
    print("\nNote: This module only monitors drawdown and vetoes trades - it does not generate signals or execute trades.")