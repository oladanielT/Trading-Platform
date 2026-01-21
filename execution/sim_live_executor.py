"""
sim_live_executor.py - Simulated live executor for running in sim_live mode

Routes execution requests to either the real `Broker` (paper/testnet) or the
`ShadowExecutor` for evaluation. Preserves timing and slippage behavior by
delegating to the existing `Broker.place_market_order` and `ExecutionRealityEngine`.

This wrapper enforces the runtime kill switch and reports executions to the
`RuntimeHealthMonitor`.
"""
from typing import Optional, Dict, Any
import logging
import asyncio

logger = logging.getLogger(__name__)


class SimLiveExecutor:
    def __init__(self, broker, shadow_executor=None, execution_reality=None, kill_switch=None, health_monitor=None, config=None):
        self.broker = broker
        self.shadow = shadow_executor
        self.exec_reality = execution_reality
        self.kill_switch = kill_switch
        self.health = health_monitor
        self.config = config or {}

    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """Place a market order respecting kill switch and routing rules.

        If the kill switch is triggered, new trades are blocked and an
        immediate rejection is returned. Otherwise, route to broker for
        actual placement. If `config` requests shadow-only evaluation, delegate
        to shadow executor instead (no real fills).
        """
        # Check kill switch
        if self.kill_switch and self.kill_switch.is_blocked():
            logger.warning(f"[SIM_LIVE] Kill switch active - blocking new trade: {symbol} {side} {quantity}")
            return {'status': 'blocked', 'reason': 'kill_switch_triggered'}

        sim_cfg = self.config.get('live_readiness', {})
        # If sim_live configured to shadow-only, record with shadow and return simulated fill
        if sim_cfg.get('sim_live', True) and sim_cfg.get('shadow_only', False):
            if self.shadow:
                # Shadow evaluate: create a fake filled result without touching broker
                logger.info(f"[SIM_LIVE] Shadow-only execution: {symbol} {side} size={quantity}")
                # Record trade in health monitor with 0.0 PnL (entry has no realized gain)
                if self.health:
                    self.health.register_trade(pnl=0.0, closed=False)
                return {'status': 'sim_filled', 'fill_price': None, 'filled_quantity': quantity}
            else:
                return {'status': 'error', 'reason': 'shadow_executor_unavailable'}

        # Otherwise, execute via broker (paper/testnet). Preserve timing by awaiting broker call.
        try:
            result = await self.broker.place_market_order(symbol=symbol, side=side, quantity=quantity)

            # Record trade execution in health monitor (0.0 PnL at entry)
            if self.health and result.get('status') in ['filled', 'sim_filled']:
                self.health.register_trade(pnl=0.0, closed=False)

            return result
        except Exception as e:
            # Treat broker failure as critical error to the kill switch
            logger.error(f"[SIM_LIVE] Broker execution error: {e}")
            if self.kill_switch:
                self.kill_switch.register_critical_error(str(e))
            return {'status': 'error', 'reason': str(e)}
