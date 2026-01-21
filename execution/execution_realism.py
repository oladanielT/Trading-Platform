"""
Execution Realism wrapper for broker interfaces.

Disabled by default. Wraps an existing broker and simulates
slippage, partial fills, rejections, retry-decay, latency, and fees.
This module is intentionally conservative and must never improve PnL.
"""
import logging
import random
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ExecutionRealismWrapper:
    """Wraps a broker object and degrades executions when enabled.

    Parameters
    - broker: underlying broker implementing `execute_order(order)`
    - enabled: disabled by default
    - seed: optional deterministic seed
    - config: dict of parameters (latency_bounds, slippage_coeff, retry)
    """

    def __init__(self, broker: Any, enabled: bool = False, seed: Optional[int] = None, config: Optional[Dict] = None):
        self.broker = broker
        self.enabled = enabled
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        default = {
            "latency_bounds": (0.0, 0.05),
            "slippage_coeff": 0.001,
            "partial_fill_prob": 0.02,
            "reject_prob": 0.005,
            "retry_attempts": 2,
            "retry_decay": 0.5,
            "fee_rate": 0.00075,
        }
        self.config = {**default, **(config or {})}

    def simulate_latency(self):
        lo, hi = self.config["latency_bounds"]
        latency = random.uniform(lo, hi)
        time.sleep(0 if latency <= 0 else min(latency, 0.1))
        return latency

    def simulate_slippage(self, price: float, side: str, volume: float, volatility: float = 0.0, recent_volume: float = 1.0) -> float:
        coeff = self.config["slippage_coeff"]
        size_impact = min(1.0, volume / max(1e-9, recent_volume))
        slippage = price * coeff * (1 + volatility) * size_impact
        # always adverse: buy -> price + slippage, sell -> price - slippage
        if side.lower() == "buy":
            return price + slippage
        return price - slippage

    def execute_order(self, order: Dict) -> Dict:
        """Execute an order dict through the realism wrapper.

        Expected order keys: `price`, `side`, `volume`, optional `volatility`, `recent_volume`.
        Returns a result dict with fields: `filled_volume`, `avg_price`, `fees`, `status`.
        """
        if not self.enabled:
            # pass-through when disabled
            if hasattr(self.broker, "execute_order"):
                return self.broker.execute_order(order)
            return {"filled_volume": order.get("volume", 0), "avg_price": order.get("price"), "fees": 0.0, "status": "passed"}

        attempt = 0
        attempts = max(1, int(self.config.get("retry_attempts", 0)))
        base_retry_decay = float(self.config.get("retry_decay", 0.5))

        while attempt < attempts:
            attempt += 1
            self.simulate_latency()

            # rejection
            if random.random() < float(self.config.get("reject_prob", 0.0)):
                logger.warning("Order rejected by realism layer (attempt=%d)", attempt)
                if attempt < attempts:
                    # wait a decayed time and retry
                    time.sleep(base_retry_decay * attempt)
                    continue
                return {"filled_volume": 0.0, "avg_price": None, "fees": 0.0, "status": "rejected"}

            price = order.get("price", 0.0)
            side = order.get("side", "buy")
            volume = float(order.get("volume", 0.0))
            volatility = float(order.get("volatility", 0.0))
            recent_volume = float(order.get("recent_volume", max(volume, 1.0)))

            sim_price = self.simulate_slippage(price, side, volume, volatility, recent_volume)

            # partial fills
            filled = volume
            if random.random() < float(self.config.get("partial_fill_prob", 0.0)):
                # partial fill fraction between 10%-90%
                frac = random.uniform(0.1, 0.9)
                filled = round(volume * frac, 8)

            fees = abs(sim_price * filled) * float(self.config.get("fee_rate", 0.0))

            result = {"filled_volume": filled, "avg_price": sim_price, "fees": fees, "status": "filled"}

            # call underlying broker to record trade if possible, but never reduce costs
            if hasattr(self.broker, "record_execution"):
                try:
                    self.broker.record_execution({**order, **result})
                except Exception:
                    logger.exception("broker.record_execution failed; continuing")

            return result


__all__ = ["ExecutionRealismWrapper"]
