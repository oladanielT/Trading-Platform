"""
runtime_health_monitor.py - Lightweight runtime monitoring for live readiness

Collects simple runtime metrics (trades today, win rate, realized/unrealized PnL,
drawdown, active strategies, current regime, active risk blocks) and emits
structured logs and optional CSV snapshots.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, List
import csv
import logging
import os

logger = logging.getLogger(__name__)


@dataclass
class HealthMetrics:
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    current_drawdown: float = 0.0
    active_strategies: List[str] = field(default_factory=list)
    current_regime: Optional[str] = None
    active_risk_blocks: List[str] = field(default_factory=list)


class RuntimeHealthMonitor:
    def __init__(self, config: Optional[dict] = None):
        cfg = config or {}
        rm_cfg = cfg.get('live_readiness', {}) if isinstance(cfg, dict) else {}

        self.enabled = bool(rm_cfg.get('enabled', False))
        self.csv_snapshot = bool(rm_cfg.get('csv_snapshot', False))
        self.snapshot_interval_minutes = int(rm_cfg.get('snapshot_interval_minutes', 5))
        self.snapshot_path = rm_cfg.get('snapshot_path', 'logs/health_snapshot.csv')

        self.metrics = HealthMetrics()
        self._last_snapshot = None

        if self.csv_snapshot:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.snapshot_path) or '.', exist_ok=True)
            # Create CSV header if not exists
            if not os.path.exists(self.snapshot_path):
                with open(self.snapshot_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp','trades_today','wins','losses','realized_pnl','unrealized_pnl','drawdown','active_strategies','current_regime','active_risk_blocks'])

        logger.info(f"[HEALTH] Initialized (enabled={self.enabled}, csv_snapshot={self.csv_snapshot})")

    def register_trade(self, pnl: float, closed: bool = True):
        if not self.enabled:
            return
        self.metrics.trades_today += 1
        if pnl >= 0:
            self.metrics.wins += 1
        else:
            self.metrics.losses += 1
        if closed:
            self.metrics.realized_pnl += pnl

    def update_unrealized(self, unrealized_pnl: float):
        if not self.enabled:
            return
        self.metrics.unrealized_pnl = unrealized_pnl

    def set_regime(self, regime: Optional[str]):
        if not self.enabled:
            return
        self.metrics.current_regime = regime

    def set_active_strategies(self, strategies: List[str]):
        if not self.enabled:
            return
        self.metrics.active_strategies = strategies

    def set_active_risk_blocks(self, blocks: List[str]):
        if not self.enabled:
            return
        self.metrics.active_risk_blocks = blocks

    def get_metrics(self) -> Dict:
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'trades_today': self.metrics.trades_today,
            'wins': self.metrics.wins,
            'losses': self.metrics.losses,
            'realized_pnl': self.metrics.realized_pnl,
            'unrealized_pnl': self.metrics.unrealized_pnl,
            'drawdown': self.metrics.current_drawdown,
            'active_strategies': self.metrics.active_strategies,
            'current_regime': self.metrics.current_regime,
            'active_risk_blocks': self.metrics.active_risk_blocks,
        }

    def snapshot(self):
        if not self.enabled or not self.csv_snapshot:
            return
        data = self.get_metrics()
        with open(self.snapshot_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                data['timestamp'], data['trades_today'], data['wins'], data['losses'],
                data['realized_pnl'], data['unrealized_pnl'], data['drawdown'],
                ';'.join(data['active_strategies']), data['current_regime'] or '', ';'.join(data['active_risk_blocks'])
            ])
        self._last_snapshot = datetime.utcnow()

    def log_metrics(self):
        if not self.enabled:
            return
        m = self.get_metrics()
        logger.info(f"[HEALTH] {m['timestamp']} trades={m['trades_today']} wins={m['wins']} losses={m['losses']} realized_pnl={m['realized_pnl']:.2f} unrealized_pnl={m['unrealized_pnl']:.2f} drawdown={m['drawdown']:.2f}")
