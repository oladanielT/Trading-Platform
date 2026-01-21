"""
execution_reality_engine.py - Execution Reality & Market Impact Engine

Introduces realistic market constraints to protect capital:
- Slippage modeling based on order size and volatility
- Partial fill probability estimation
- Order rejection simulation
- Latency impact assessment
- Liquidity constraints

This engine applies ONLY downward pressure on position sizes - never increases them.
Disabled by default. When enabled, makes execution more conservative and realistic.

Phase 9: Execution Realism Layer
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class ExecutionMetrics:
    """Tracks execution quality metrics."""
    total_orders_evaluated: int = 0
    total_orders_approved: int = 0
    total_orders_rejected: int = 0
    total_size_reduction_bps: float = 0.0  # Basis points of size reduction
    total_estimated_slippage_bps: float = 0.0  # Basis points of slippage
    low_fill_probability_rejections: int = 0
    high_volatility_rejections: int = 0
    low_liquidity_rejections: int = 0
    symbol_stats: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))
    execution_history: list = field(default_factory=list)


class ExecutionRealityEngine:
    """
    Models execution friction and market impact.
    
    Responsibilities:
    - Estimate fill probability based on liquidity and volatility
    - Calculate slippage impact using order size and volatility
    - Reduce position sizes conservatively
    - Track execution quality metrics
    - Provide diagnostics and monitoring
    
    Design Principle: Conservative by default, realistic assumptions.
    Never increases size. Only applies downward pressure.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ExecutionRealityEngine.
        
        Args:
            config: Configuration dict with keys:
                - enabled: bool (default False)
                - max_size_vs_volume: float, max order size as % of recent volume (0.02 = 2%)
                - min_fill_probability: float, minimum acceptable fill probability (0.6 = 60%)
                - slippage_model: str, one of ['conservative', 'moderate', 'aggressive']
                - latency_penalty_ms: int, estimated latency impact in milliseconds
        """
        self.config = config
        self.enabled = config.get('enabled', False)
        self.max_size_vs_volume = config.get('max_size_vs_volume', 0.02)
        self.min_fill_probability = config.get('min_fill_probability', 0.6)
        self.slippage_model = config.get('slippage_model', 'conservative')
        self.latency_penalty_ms = config.get('latency_penalty_ms', 500)
        
        # Slippage model parameters (bps per unit of metric)
        self._slippage_params = {
            'conservative': {
                'base_slippage': 5.0,  # 5 bps baseline
                'size_multiplier': 100.0,  # 100 bps per 1% of volume
                'volatility_multiplier': 2.0,  # 2x bps per ATR%
                'spread_multiplier': 1.5,  # 1.5x bps per spread%
            },
            'moderate': {
                'base_slippage': 3.0,
                'size_multiplier': 50.0,
                'volatility_multiplier': 1.5,
                'spread_multiplier': 1.0,
            },
            'aggressive': {
                'base_slippage': 1.0,
                'size_multiplier': 25.0,
                'volatility_multiplier': 1.0,
                'spread_multiplier': 0.5,
            }
        }
        
        # Metrics tracking
        self.metrics = ExecutionMetrics()
        
        logger.info(f"[EXECUTION_REALITY] Engine initialized: enabled={self.enabled}, "
                   f"model={self.slippage_model}, min_fill_prob={self.min_fill_probability:.1%}")
    
    def evaluate_order(
        self,
        symbol: str,
        side: str,
        requested_size: float,
        market_snapshot: dict
    ) -> dict:
        """
        Evaluate order execution feasibility and constraints.
        
        Returns realistic execution parameters that account for:
        - Liquidity constraints (size vs volume)
        - Volatility impact on slippage and fill probability
        - Spread impact
        - Order rejection probability
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            side: 'buy' or 'sell'
            requested_size: Requested order size in base currency
            market_snapshot: Dict with keys:
                - 'close': float, current close price
                - 'high': float, candle high
                - 'low': float, candle low
                - 'volume': float, candle volume in base currency
                - Optional 'atr_percent': float, ATR as % of close (for volatility)
                - Optional 'bid_ask_spread_pct': float, bid-ask spread as % of close
        
        Returns:
            {
                'approved_size': float,  # Size approved for execution (≤ requested)
                'expected_slippage_bps': float,  # Expected slippage in basis points
                'fill_probability': float,  # Probability of fill (0-1)
                'execution_risk': str,  # 'low' | 'medium' | 'high'
                'rejection_reason': Optional[str],  # If size becomes 0
                'constraints': dict,  # Detailed constraint analysis
            }
        """
        if not self.enabled:
            # Disabled mode: pass through unchanged
            return {
                'approved_size': requested_size,
                'expected_slippage_bps': 0.0,
                'fill_probability': 1.0,
                'execution_risk': 'low',
                'rejection_reason': None,
                'constraints': {'disabled': True},
            }
        
        self.metrics.total_orders_evaluated += 1
        
        # Extract market data with defaults
        close = market_snapshot.get('close', 1.0)
        high = market_snapshot.get('high', close)
        low = market_snapshot.get('low', close)
        volume = market_snapshot.get('volume', 1e10)  # Default to large volume
        atr_percent = market_snapshot.get('atr_percent', (high - low) / close * 100 if close > 0 else 1.0)
        spread_pct = market_snapshot.get('bid_ask_spread_pct', 0.05)  # Default 5 bps
        
        constraints = {
            'close': close,
            'volume': volume,
            'atr_percent': atr_percent,
            'spread_pct': spread_pct,
        }
        
        # Calculate liquidity constraint: size vs volume
        size_vs_volume = requested_size * close / volume if volume > 0 else 1.0
        liquidity_approved_size = requested_size
        liquidity_reason = None
        
        if size_vs_volume > self.max_size_vs_volume:
            # Size is too large relative to recent volume
            liquidity_approved_size = (self.max_size_vs_volume * volume / close) if close > 0 else 0
            liquidity_reason = f"exceeds max size vs volume ({size_vs_volume:.2%} > {self.max_size_vs_volume:.2%})"
            constraints['liquidity_constraint'] = {
                'requested_vs_volume': size_vs_volume,
                'max_allowed_vs_volume': self.max_size_vs_volume,
                'approved_size': liquidity_approved_size,
            }
        
        # Calculate slippage
        slippage_bps = self._estimate_slippage(
            size_vs_volume=size_vs_volume,
            atr_percent=atr_percent,
            spread_pct=spread_pct
        )
        constraints['slippage_bps'] = slippage_bps
        
        # Calculate fill probability
        fill_probability = self._estimate_fill_probability(
            size_vs_volume=size_vs_volume,
            atr_percent=atr_percent,
            slippage_bps=slippage_bps
        )
        constraints['fill_probability'] = fill_probability
        
        # Determine execution risk
        execution_risk = self._assess_execution_risk(
            size_vs_volume=size_vs_volume,
            atr_percent=atr_percent,
            fill_probability=fill_probability
        )
        constraints['execution_risk'] = execution_risk
        
        # Final size decision: apply liquidity constraint
        approved_size = liquidity_approved_size
        rejection_reason = None
        
        # Check if fill probability is too low
        if fill_probability < self.min_fill_probability:
            # Size too risky - reduce further or reject
            if approved_size > 0:
                # Reduce size proportionally based on fill probability deficit
                approved_size = 0  # Conservative: reject if fill probability too low
                rejection_reason = f"fill probability too low ({fill_probability:.1%} < {self.min_fill_probability:.1%})"
                self.metrics.low_fill_probability_rejections += 1
        
        # Calculate size reduction
        if approved_size > 0:
            size_reduction_bps = (1 - approved_size / max(requested_size, 1e-10)) * 10000
            self.metrics.total_size_reduction_bps += size_reduction_bps
            self.metrics.total_orders_approved += 1
        else:
            self.metrics.total_orders_rejected += 1
            if not rejection_reason:
                if liquidity_reason:
                    rejection_reason = f"liquidity: {liquidity_reason}"
                else:
                    rejection_reason = "execution risk too high"
        
        # Track per-symbol statistics
        if symbol not in self.metrics.symbol_stats:
            self.metrics.symbol_stats[symbol] = {
                'total_evaluated': 0,
                'total_approved': 0,
                'total_slippage_bps': 0,
                'avg_fill_probability': 0,
            }
        
        self.metrics.symbol_stats[symbol]['total_evaluated'] += 1
        if approved_size > 0:
            self.metrics.symbol_stats[symbol]['total_approved'] += 1
        self.metrics.symbol_stats[symbol]['total_slippage_bps'] += slippage_bps
        
        # Record execution event
        self.metrics.execution_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'symbol': symbol,
            'side': side,
            'requested_size': requested_size,
            'approved_size': approved_size,
            'slippage_bps': slippage_bps,
            'fill_probability': fill_probability,
            'execution_risk': execution_risk,
            'rejection_reason': rejection_reason,
        })
        
        return {
            'approved_size': approved_size,
            'expected_slippage_bps': slippage_bps,
            'fill_probability': fill_probability,
            'execution_risk': execution_risk,
            'rejection_reason': rejection_reason,
            'constraints': constraints,
        }
    
    def _estimate_slippage(
        self,
        size_vs_volume: float,
        atr_percent: float,
        spread_pct: float
    ) -> float:
        """
        Estimate slippage in basis points.
        
        Larger orders, higher volatility, and wider spreads all increase slippage.
        Uses a conservative additive model.
        
        Args:
            size_vs_volume: Order size as % of recent candle volume
            atr_percent: Volatility as ATR percent of close
            spread_pct: Bid-ask spread as % of close
        
        Returns:
            Slippage estimate in basis points
        """
        params = self._slippage_params.get(self.slippage_model, self._slippage_params['conservative'])
        
        # Base slippage
        slippage_bps = params['base_slippage']
        
        # Size impact: larger orders worse slippage
        size_impact = size_vs_volume * params['size_multiplier']
        
        # Volatility impact: higher volatility worse slippage
        atr_impact = (atr_percent / 100.0) * params['volatility_multiplier']
        
        # Spread impact: wider spreads worse slippage
        spread_impact = (spread_pct * 100.0) * params['spread_multiplier']  # spread_pct is %, convert to bps
        
        total_slippage = slippage_bps + size_impact + atr_impact + spread_impact
        
        return total_slippage
    
    def _estimate_fill_probability(
        self,
        size_vs_volume: float,
        atr_percent: float,
        slippage_bps: float
    ) -> float:
        """
        Estimate probability of full fill.
        
        Factors:
        - Larger orders have lower fill probability
        - Higher volatility reduces fill probability
        - Higher slippage reduces fill probability
        
        Args:
            size_vs_volume: Order size as % of recent volume
            atr_percent: Volatility as ATR percent
            slippage_bps: Expected slippage in basis points
        
        Returns:
            Fill probability (0-1)
        """
        # Base fill probability
        base_prob = 0.95
        
        # Size penalty: each 1% of volume = -3% fill probability
        size_penalty = size_vs_volume * 300.0  # 300 bps = 3%
        
        # Volatility penalty: each 1% ATR = -1% fill probability
        volatility_penalty = atr_percent * 10.0  # 100 bps per 1%
        
        # Slippage penalty: higher slippage = lower probability
        # Each 10 bps of slippage = -1% probability
        slippage_penalty = slippage_bps * 10.0
        
        # Apply penalties (in bps, divide by 10000)
        total_penalty = (size_penalty + volatility_penalty + slippage_penalty) / 10000.0
        
        fill_probability = max(0.0, min(1.0, base_prob - total_penalty))
        
        return fill_probability
    
    def _assess_execution_risk(
        self,
        size_vs_volume: float,
        atr_percent: float,
        fill_probability: float
    ) -> str:
        """
        Assess execution risk level.
        
        Returns one of: 'low', 'medium', 'high'
        """
        risk_score = 0
        
        # Size risk: >2% of volume is medium, >5% is high
        if size_vs_volume > 0.05:
            risk_score += 3
        elif size_vs_volume > 0.02:
            risk_score += 1
        
        # Volatility risk: >5% ATR is medium, >10% is high
        if atr_percent > 10:
            risk_score += 3
        elif atr_percent > 5:
            risk_score += 1
        
        # Fill probability risk
        if fill_probability < 0.5:
            risk_score += 3
        elif fill_probability < 0.7:
            risk_score += 1
        
        if risk_score >= 3:
            return 'high'
        elif risk_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def record_execution_result(
        self,
        symbol: str,
        requested_size: float,
        filled_size: float,
        slippage_bps: float,
        latency_ms: int
    ):
        """
        Record actual execution result for diagnostics and learning.
        
        Args:
            symbol: Trading pair
            requested_size: Size requested
            filled_size: Size actually filled
            slippage_bps: Actual slippage observed
            latency_ms: Actual latency in milliseconds
        """
        if not self.enabled:
            return
        
        # Update tracking
        if symbol in self.metrics.symbol_stats:
            stats = self.metrics.symbol_stats[symbol]
            
            # Update average fill probability (simplified)
            actual_fill_rate = filled_size / max(requested_size, 1e-10)
            stats['avg_fill_probability'] = (
                stats.get('avg_fill_probability', 0) * 0.9 +
                actual_fill_rate * 0.1
            )
    
    def get_diagnostics(self) -> dict:
        """
        Get comprehensive execution engine diagnostics.
        
        Returns:
            Dict with execution metrics and health status
        """
        if not self.enabled:
            return {
                'enabled': False,
                'status': 'disabled',
            }
        
        # Calculate approval rate
        approval_rate = (
            self.metrics.total_orders_approved / max(self.metrics.total_orders_evaluated, 1)
        )
        
        # Calculate average slippage
        avg_slippage = (
            self.metrics.total_estimated_slippage_bps / max(self.metrics.total_orders_evaluated, 1)
        )
        
        # Per-symbol health
        symbol_health = {}
        for symbol, stats in self.metrics.symbol_stats.items():
            evaluated = stats['total_evaluated']
            if evaluated > 0:
                approved = stats['total_approved']
                symbol_health[symbol] = {
                    'orders_evaluated': evaluated,
                    'approval_rate': approved / evaluated,
                    'avg_slippage_bps': stats['total_slippage_bps'] / evaluated,
                    'avg_fill_probability': stats.get('avg_fill_probability', 0),
                }
        
        return {
            'enabled': True,
            'status': 'active',
            'config': {
                'slippage_model': self.slippage_model,
                'max_size_vs_volume': self.max_size_vs_volume,
                'min_fill_probability': self.min_fill_probability,
                'latency_penalty_ms': self.latency_penalty_ms,
            },
            'metrics': {
                'total_orders_evaluated': self.metrics.total_orders_evaluated,
                'total_orders_approved': self.metrics.total_orders_approved,
                'total_orders_rejected': self.metrics.total_orders_rejected,
                'approval_rate': approval_rate,
                'avg_estimated_slippage_bps': avg_slippage,
                'avg_size_reduction_bps': (
                    self.metrics.total_size_reduction_bps / max(self.metrics.total_orders_evaluated, 1)
                ),
                'low_fill_probability_rejections': self.metrics.low_fill_probability_rejections,
                'high_volatility_rejections': self.metrics.high_volatility_rejections,
                'low_liquidity_rejections': self.metrics.low_liquidity_rejections,
            },
            'symbol_health': symbol_health,
            'last_100_executions': self.metrics.execution_history[-100:] if self.metrics.execution_history else [],
        }
    
    def reset_metrics(self):
        """Reset execution metrics (useful for periodic reporting)."""
        self.metrics = ExecutionMetrics()
