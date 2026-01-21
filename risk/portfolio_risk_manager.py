"""
portfolio_risk_manager.py - Portfolio-Level Correlation & Exposure Control

Prevents correlated drawdowns and overexposure by:
- Tracking correlations across symbols
- Monitoring regime and directional exposure
- Applying portfolio-level constraints

This is a risk-reduction system that can only reduce or block trades,
never increase exposure. Safe by default (disabled unless configured).
"""

import logging
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Open position information."""
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    regime: Optional[str] = None
    strategy: Optional[str] = None
    entry_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioMetrics:
    """Portfolio-level exposure metrics."""
    total_exposure: float = 0.0
    long_exposure: float = 0.0
    short_exposure: float = 0.0
    regime_exposures: Dict[str, float] = field(default_factory=dict)
    symbol_exposures: Dict[str, float] = field(default_factory=dict)
    max_correlation: float = 0.0


class PortfolioRiskManager:
    """
    Portfolio-level risk management with correlation and exposure control.
    
    Core responsibilities:
    - Track open positions across all symbols
    - Compute rolling correlation matrix (returns-based)
    - Monitor exposure by symbol, direction, and regime
    - Enforce portfolio-level limits
    - Provide position validation with size multipliers
    
    This module CANNOT place trades or modify execution logic.
    It only validates and potentially reduces position sizes.
    """
    
    def __init__(
        self,
        total_equity: float,
        enabled: bool = True,
        correlation_lookback: int = 100,
        max_pairwise_correlation: float = 0.85,
        max_symbol_exposure_pct: float = 0.25,
        max_directional_exposure_pct: float = 0.60,
        max_regime_exposure_pct: float = 0.50,
    ):
        """
        Initialize portfolio risk manager.
        
        Args:
            total_equity: Total portfolio equity for exposure calculations
            enabled: Enable portfolio risk checks (default: True)
            correlation_lookback: Lookback period for correlation (default: 100 bars)
            max_pairwise_correlation: Max allowed correlation between positions (default: 0.85)
            max_symbol_exposure_pct: Max exposure per symbol (default: 0.25 = 25%)
            max_directional_exposure_pct: Max long or short exposure (default: 0.60 = 60%)
            max_regime_exposure_pct: Max exposure per regime (default: 0.50 = 50%)
        """
        self.total_equity = total_equity
        self.enabled = enabled
        self.correlation_lookback = correlation_lookback
        self.max_pairwise_correlation = max_pairwise_correlation
        self.max_symbol_exposure_pct = max_symbol_exposure_pct
        self.max_directional_exposure_pct = max_directional_exposure_pct
        self.max_regime_exposure_pct = max_regime_exposure_pct
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        
        # Market data storage for correlation calculation
        # Format: {symbol: DataFrame with columns ['timestamp', 'close', 'returns']}
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # Correlation cache (symbol_a, symbol_b) -> correlation
        self._correlation_cache: Dict[Tuple[str, str], float] = {}
        self._cache_valid = False
        
        # Metrics tracking
        self.blocked_trades_count = 0
        self.reduced_trades_count = 0
        
        logger.info(
            f"PortfolioRiskManager initialized: "
            f"enabled={enabled}, correlation_lookback={correlation_lookback}, "
            f"max_correlation={max_pairwise_correlation:.2f}"
        )
    
    def update_equity(self, new_equity: float) -> None:
        """
        Update total portfolio equity.
        
        Args:
            new_equity: New total equity value
        """
        self.total_equity = new_equity
        logger.debug(f"Portfolio equity updated: ${new_equity:.2f}")
    
    def update_market_data(self, symbol: str, candles: pd.DataFrame) -> None:
        """
        Update market data for correlation calculation.
        
        Args:
            symbol: Trading pair symbol
            candles: DataFrame with OHLCV data (must have 'close' column)
        """
        if not self.enabled:
            return
        
        if candles is None or len(candles) == 0:
            logger.warning(f"Empty candles provided for {symbol}")
            return
        
        # Ensure we have a 'close' column
        if 'close' not in candles.columns:
            logger.error(f"Missing 'close' column in candles for {symbol}")
            return
        
        # Calculate log returns
        df = candles[['close']].copy()
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Store only the lookback period + buffer
        lookback = self.correlation_lookback + 50
        if len(df) > lookback:
            df = df.tail(lookback)
        
        self.market_data[symbol] = df
        
        # Invalidate correlation cache
        self._cache_valid = False
        
        logger.debug(f"Market data updated for {symbol}: {len(df)} bars")
    
    def register_position(
        self,
        symbol: str,
        side: str,
        size: float,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> None:
        """
        Register an open position.
        
        Args:
            symbol: Trading pair symbol
            side: 'long' or 'short'
            size: Position size (in base currency)
            regime: Market regime when position was opened
            strategy: Strategy that opened the position
        """
        if not self.enabled:
            return
        
        position = Position(
            symbol=symbol,
            side=side.lower(),
            size=size,
            regime=regime,
            strategy=strategy,
        )
        
        self.positions[symbol] = position
        
        logger.info(
            f"[PORTFOLIO_RISK] Position registered: {symbol} {side} "
            f"size={size:.6f} regime={regime} strategy={strategy}"
        )
    
    def unregister_position(self, symbol: str) -> None:
        """
        Unregister a closed position.
        
        Args:
            symbol: Trading pair symbol
        """
        if symbol in self.positions:
            del self.positions[symbol]
            logger.info(f"[PORTFOLIO_RISK] Position unregistered: {symbol}")
    
    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """
        Get correlation between two symbols.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
        
        Returns:
            Pearson correlation coefficient (-1.0 to 1.0), or 0.0 if insufficient data
        """
        if not self.enabled:
            return 0.0
        
        if symbol_a == symbol_b:
            return 1.0
        
        # Check cache
        cache_key = tuple(sorted([symbol_a, symbol_b]))
        if self._cache_valid and cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]
        
        # Need data for both symbols
        if symbol_a not in self.market_data or symbol_b not in self.market_data:
            return 0.0
        
        df_a = self.market_data[symbol_a]
        df_b = self.market_data[symbol_b]
        
        # Need sufficient data
        if len(df_a) < 30 or len(df_b) < 30:
            return 0.0
        
        # Get returns
        returns_a = df_a['returns'].dropna().tail(self.correlation_lookback)
        returns_b = df_b['returns'].dropna().tail(self.correlation_lookback)
        
        # Align by index and compute correlation
        try:
            aligned_a = returns_a.reindex(returns_b.index).dropna()
            aligned_b = returns_b.reindex(returns_a.index).dropna()
            
            if len(aligned_a) < 30 or len(aligned_b) < 30:
                return 0.0
            
            correlation = aligned_a.corr(aligned_b)
            
            # Cache result
            self._correlation_cache[cache_key] = correlation
            
            return correlation if not np.isnan(correlation) else 0.0
        
        except Exception as e:
            logger.warning(f"Failed to compute correlation {symbol_a}/{symbol_b}: {e}")
            return 0.0
    
    def get_symbol_exposure(self, symbol: str) -> float:
        """
        Get current exposure for a specific symbol.
        
        Args:
            symbol: Trading pair symbol
        
        Returns:
            Exposure as percentage of total equity (0.0 to 1.0)
        """
        if symbol not in self.positions:
            return 0.0
        
        position = self.positions[symbol]
        return position.size / self.total_equity if self.total_equity > 0 else 0.0
    
    def get_regime_exposure(self, regime: str) -> float:
        """
        Get total exposure for a specific market regime.
        
        Args:
            regime: Market regime name
        
        Returns:
            Total exposure as percentage of total equity (0.0 to 1.0)
        """
        if not regime:
            return 0.0
        
        total = sum(
            pos.size for pos in self.positions.values()
            if pos.regime and pos.regime.upper() == regime.upper()
        )
        
        return total / self.total_equity if self.total_equity > 0 else 0.0
    
    def get_directional_exposure(self, side: str) -> float:
        """
        Get total exposure for a direction (long or short).
        
        Args:
            side: 'long' or 'short'
        
        Returns:
            Total exposure as percentage of total equity (0.0 to 1.0)
        """
        total = sum(
            pos.size for pos in self.positions.values()
            if pos.side.lower() == side.lower()
        )
        
        return total / self.total_equity if self.total_equity > 0 else 0.0
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """
        Get comprehensive portfolio exposure metrics.
        
        Returns:
            PortfolioMetrics with current exposure breakdown
        """
        metrics = PortfolioMetrics()
        
        # Calculate exposures
        for pos in self.positions.values():
            exposure = pos.size / self.total_equity if self.total_equity > 0 else 0.0
            
            metrics.total_exposure += exposure
            
            if pos.side == 'long':
                metrics.long_exposure += exposure
            elif pos.side == 'short':
                metrics.short_exposure += exposure
            
            # Regime exposure
            if pos.regime:
                regime_key = pos.regime.upper()
                metrics.regime_exposures[regime_key] = metrics.regime_exposures.get(regime_key, 0.0) + exposure
            
            # Symbol exposure
            metrics.symbol_exposures[pos.symbol] = exposure
        
        # Calculate max correlation
        symbols = list(self.positions.keys())
        if len(symbols) > 1:
            correlations = []
            for i, sym_a in enumerate(symbols):
                for sym_b in symbols[i+1:]:
                    corr = abs(self.get_correlation(sym_a, sym_b))
                    correlations.append(corr)
            
            if correlations:
                metrics.max_correlation = max(correlations)
        
        return metrics
    
    def validate_new_position(
        self,
        symbol: str,
        side: str,
        proposed_size: float,
        regime: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Tuple[bool, str, float]:
        """
        Validate a new position and calculate size multiplier.
        
        This is the PRIMARY integration point. Called before executing a trade.
        
        Args:
            symbol: Trading pair symbol
            side: 'long' or 'short'
            proposed_size: Proposed position size
            regime: Market regime
            strategy: Strategy name
        
        Returns:
            Tuple of:
                - allowed: True if position can be opened (possibly with reduced size)
                - reason: Explanation for rejection or reduction
                - size_multiplier: Multiplier to apply (0.0 = block, 1.0 = full size)
        """
        if not self.enabled:
            return (True, "portfolio_risk_disabled", 1.0)
        
        if proposed_size <= 0:
            return (False, "invalid_size", 0.0)
        
        # Initialize multiplier at full size
        multiplier = 1.0
        reasons = []
        
        # Calculate proposed exposure
        proposed_exposure = proposed_size / self.total_equity if self.total_equity > 0 else 0.0
        
        # Check 1: Symbol already has position?
        if symbol in self.positions:
            existing = self.positions[symbol]
            
            # Same direction - reject (no stacking)
            if existing.side.lower() == side.lower():
                self.blocked_trades_count += 1
                logger.warning(
                    f"[PORTFOLIO_RISK] Trade blocked: {symbol} {side} "
                    f"reason=position_exists"
                )
                return (False, "position_already_exists", 0.0)
            
            # Opposite direction - allow but monitor exposure
            reasons.append("hedging_existing_position")
        
        # Check 2: Symbol exposure limit
        existing_symbol_exposure = self.get_symbol_exposure(symbol)
        total_symbol_exposure = existing_symbol_exposure + proposed_exposure
        
        if total_symbol_exposure > self.max_symbol_exposure_pct:
            # Reduce size to fit within limit
            available = max(0.0, self.max_symbol_exposure_pct - existing_symbol_exposure)
            reduction = available / proposed_exposure if proposed_exposure > 0 else 0.0
            
            if reduction < 0.1:  # Less than 10% of proposed size - block
                self.blocked_trades_count += 1
                logger.warning(
                    f"[PORTFOLIO_RISK] Trade blocked: {symbol} {side} "
                    f"reason=symbol_exposure_limit "
                    f"current={existing_symbol_exposure:.2%} "
                    f"proposed={proposed_exposure:.2%} "
                    f"limit={self.max_symbol_exposure_pct:.2%}"
                )
                return (False, "symbol_exposure_limit_exceeded", 0.0)
            
            multiplier = min(multiplier, reduction)
            reasons.append(f"symbol_exposure_reduced_{reduction:.2f}")
        
        # Check 3: Directional exposure limit
        directional_exposure = self.get_directional_exposure(side)
        total_directional = directional_exposure + proposed_exposure
        
        if total_directional > self.max_directional_exposure_pct:
            # Reduce size to fit within limit
            available = max(0.0, self.max_directional_exposure_pct - directional_exposure)
            reduction = available / proposed_exposure if proposed_exposure > 0 else 0.0
            
            if reduction < 0.1:  # Less than 10% - block
                self.blocked_trades_count += 1
                logger.warning(
                    f"[PORTFOLIO_RISK] Trade blocked: {symbol} {side} "
                    f"reason=directional_exposure_limit "
                    f"current={directional_exposure:.2%} "
                    f"limit={self.max_directional_exposure_pct:.2%}"
                )
                return (False, "directional_exposure_limit_exceeded", 0.0)
            
            multiplier = min(multiplier, reduction)
            reasons.append(f"directional_exposure_reduced_{reduction:.2f}")
        
        # Check 4: Regime exposure limit
        if regime:
            regime_exposure = self.get_regime_exposure(regime)
            total_regime = regime_exposure + proposed_exposure
            
            if total_regime > self.max_regime_exposure_pct:
                # Reduce size to fit within limit
                available = max(0.0, self.max_regime_exposure_pct - regime_exposure)
                reduction = available / proposed_exposure if proposed_exposure > 0 else 0.0
                
                if reduction < 0.1:  # Less than 10% - block
                    self.blocked_trades_count += 1
                    logger.warning(
                        f"[PORTFOLIO_RISK] Trade blocked: {symbol} {side} "
                        f"reason=regime_exposure_limit "
                        f"regime={regime} "
                        f"current={regime_exposure:.2%} "
                        f"limit={self.max_regime_exposure_pct:.2%}"
                    )
                    return (False, f"regime_exposure_limit_exceeded_{regime}", 0.0)
                
                multiplier = min(multiplier, reduction)
                reasons.append(f"regime_exposure_reduced_{reduction:.2f}")
        
        # Check 5: Correlation with existing positions
        max_correlation = 0.0
        for existing_symbol, existing_pos in self.positions.items():
            if existing_symbol == symbol:
                continue
            
            # Only check correlation for same-direction positions
            if existing_pos.side.lower() != side.lower():
                continue
            
            corr = abs(self.get_correlation(symbol, existing_symbol))
            max_correlation = max(max_correlation, corr)
            
            if corr > self.max_pairwise_correlation:
                # High correlation - reduce size proportionally
                # correlation 0.85 → 50% reduction
                # correlation 0.95 → 75% reduction
                excess = (corr - self.max_pairwise_correlation) / (1.0 - self.max_pairwise_correlation)
                reduction = 1.0 - min(0.75, excess)  # Max 75% reduction
                
                if reduction < 0.1:  # Less than 10% - block
                    self.blocked_trades_count += 1
                    logger.warning(
                        f"[PORTFOLIO_RISK] Trade blocked: {symbol} {side} "
                        f"reason=high_correlation "
                        f"with={existing_symbol} "
                        f"correlation={corr:.2f} "
                        f"limit={self.max_pairwise_correlation:.2f}"
                    )
                    return (False, f"high_correlation_with_{existing_symbol}", 0.0)
                
                multiplier = min(multiplier, reduction)
                reasons.append(f"correlation_{existing_symbol}_{corr:.2f}_reduced_{reduction:.2f}")
        
        # Determine final result
        if multiplier < 1.0:
            self.reduced_trades_count += 1
            reason = "_".join(reasons) if reasons else "portfolio_constraints"
            
            logger.info(
                f"[PORTFOLIO_RISK] Trade size reduced: {symbol} {side} "
                f"multiplier={multiplier:.2f} "
                f"reason={reason} "
                f"max_correlation={max_correlation:.2f}"
            )
            
            return (True, reason, multiplier)
        
        # Full size approved
        logger.debug(
            f"[PORTFOLIO_RISK] Trade approved: {symbol} {side} "
            f"size={proposed_size:.6f}"
        )
        
        return (True, "approved", 1.0)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get portfolio risk manager statistics.
        
        Returns:
            Dictionary with stats and metrics
        """
        metrics = self.get_portfolio_metrics()
        
        return {
            'enabled': self.enabled,
            'open_positions': len(self.positions),
            'total_exposure': metrics.total_exposure,
            'long_exposure': metrics.long_exposure,
            'short_exposure': metrics.short_exposure,
            'max_correlation': metrics.max_correlation,
            'blocked_trades': self.blocked_trades_count,
            'reduced_trades': self.reduced_trades_count,
            'regime_exposures': metrics.regime_exposures,
            'symbol_exposures': metrics.symbol_exposures,
        }
