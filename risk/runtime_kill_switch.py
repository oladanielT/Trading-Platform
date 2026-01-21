"""
runtime_kill_switch.py - Hard runtime kill switches for live readiness

Implements deterministic, config-driven kill switches that immediately block
new trades when triggered. Triggers include max daily loss, max consecutive
losses, and critical system errors.

This module is advisory to the orchestration layer and MUST NOT modify
position sizing or optimization math. It only blocks new trade placements
and optionally requests open positions to be closed.
"""
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchConfig:
    enabled: bool = False
    max_daily_loss_pct: float = 1.0
    max_consecutive_losses: int = 5
    close_positions_on_trigger: bool = True


@dataclass
class KillSwitchState:
    enabled: bool = False
    triggered: bool = False
    trigger_reason: Optional[str] = None
    trigger_time: Optional[datetime] = None
    todays_pnl: float = 0.0
    consecutive_losses: int = 0
    history: List[dict] = field(default_factory=list)


class RuntimeKillSwitch:
    def __init__(self, config: Optional[dict] = None, initial_equity: float = 100000.0):
        cfg = config or {}
        ks_cfg = cfg.get('kill_switch', {}) if isinstance(cfg, dict) else {}

        self.config = KillSwitchConfig(
            enabled=bool(ks_cfg.get('enabled', False)),
            max_daily_loss_pct=float(ks_cfg.get('max_daily_loss_pct', 1.0)),
            max_consecutive_losses=int(ks_cfg.get('max_consecutive_losses', 5)),
            close_positions_on_trigger=bool(ks_cfg.get('close_positions_on_trigger', True)),
        )

        self.state = KillSwitchState(enabled=self.config.enabled)
        self.initial_equity = float(initial_equity)
        self.daily_start = datetime.utcnow()

        logger.info(f"[KILL_SWITCH] Initialized (enabled={self.config.enabled})")

    def reset_daily(self):
        self.daily_start = datetime.utcnow()
        self.state.todays_pnl = 0.0
        self.state.consecutive_losses = 0

    def register_trade_result(self, pnl: float):
        """Register an executed trade's PnL. Triggers kill switch if thresholds exceeded."""
        if not self.config.enabled:
            return

        now = datetime.utcnow()
        # reset daily window if day changed
        if now.date() != self.daily_start.date():
            self.reset_daily()

        self.state.todays_pnl += pnl
        self.state.history.append({'timestamp': now.isoformat(), 'pnl': pnl})

        # Consecutive loss tracking
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # Compute loss percent relative to initial equity
        loss_pct = abs(self.state.todays_pnl) / max(1.0, self.initial_equity) * 100.0

        # Trigger conditions
        if loss_pct >= self.config.max_daily_loss_pct:
            self._trigger('max_daily_loss', f"daily_loss_pct={loss_pct:.4f}")
            return

        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            self._trigger('max_consecutive_losses', f"consecutive_losses={self.state.consecutive_losses}")
            return

    def register_critical_error(self, message: str):
        """Register a critical system error (data feed, executor failure)."""
        if not self.config.enabled:
            return
        self._trigger('critical_error', message)

    def _trigger(self, reason: str, details: Optional[str] = None):
        if self.state.triggered:
            return
        self.state.triggered = True
        self.state.trigger_reason = reason
        self.state.trigger_time = datetime.utcnow()
        logger.critical(f"[KILL_SWITCH] TRIGGERED: {reason} ({details})")

    def is_blocked(self) -> bool:
        return bool(self.state.triggered)

    def should_close_positions(self) -> bool:
        return bool(self.config.close_positions_on_trigger and self.state.triggered)

    def get_state(self) -> dict:
        return {
            'enabled': self.config.enabled,
            'triggered': self.state.triggered,
            'reason': self.state.trigger_reason,
            'trigger_time': self.state.trigger_time.isoformat() if self.state.trigger_time else None,
            'todays_pnl': self.state.todays_pnl,
            'consecutive_losses': self.state.consecutive_losses,
        }
