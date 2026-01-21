"""
Capital Graduation System

Controls capital allocation based on promotion state.

Rules (STRICT):
    SHADOW      → 0% capital (no execution)
    PAPER       → 0% capital (simulated fills only)
    MICRO_LIVE  → 1-5% of total capital
    FULL_LIVE   → Full allocation per portfolio rules

SAFETY:
- Capital limits NEVER exceeded
- PortfolioRiskManager has final authority
- DrawdownGuard overrides everything
- No bypassing of existing risk controls
"""

from enum import Enum
from typing import Dict, Optional
import logging

from ai.strategy_promotion_manager import PromotionState

logger = logging.getLogger(__name__)


class CapitalGraduation:
    """
    Manages capital allocation based on strategy promotion state.
    
    CRITICAL SAFETY:
    - Acts as REDUCTION ONLY (never increases exposure)
    - PortfolioRiskManager limits are absolute
    - DrawdownGuard has veto power
    - Cannot bypass existing risk controls
    """
    
    # Capital allocation limits by state (% of total equity)
    STATE_CAPITAL_LIMITS = {
        PromotionState.RESEARCH: 0.0,      # No trading
        PromotionState.APPROVED: 0.0,      # No trading yet
        PromotionState.SHADOW: 0.0,        # No execution
        PromotionState.PAPER: 0.0,         # Simulated only
        PromotionState.MICRO_LIVE: 0.05,   # 5% max
        PromotionState.FULL_LIVE: 1.0,     # Full allocation (subject to portfolio rules)
        PromotionState.RETIRED: 0.0        # No trading
    }
    
    def __init__(self, config: Dict):
        """
        Initialize capital graduation system.
        
        Args:
            config: Configuration dict with 'strategy_promotion' section
        """
        self.config = config.get('strategy_promotion', {})
        
        # Override max micro_live capital if configured
        self.max_micro_capital = self.config.get('max_micro_live_capital', 0.05)
        
        # Ensure micro capital doesn't exceed 5%
        if self.max_micro_capital > 0.05:
            logger.warning(
                f"[CAPITAL] max_micro_live_capital {self.max_micro_capital} "
                f"exceeds safe limit. Capping at 0.05"
            )
            self.max_micro_capital = 0.05

        # Load state limits overrides from config, respecting safety caps
        self.state_limits = dict(self.STATE_CAPITAL_LIMITS)
        overrides = self.config.get('capital_limits', {})
        if overrides:
            for key, value in overrides.items():
                try:
                    state = self._map_state(key)
                    limit = float(value)
                    if state == PromotionState.MICRO_LIVE and limit > self.max_micro_capital:
                        logger.warning(
                            f"[CAPITAL] MICRO_LIVE override {limit} exceeds max_micro_live_capital {self.max_micro_capital}; capping"
                        )
                        limit = self.max_micro_capital
                    self.state_limits[state] = limit
                except Exception as e:
                    logger.error(f"[CAPITAL] Invalid capital_limits override {key}: {e}")
        
        logger.info(
            f"[CAPITAL] Initialized (micro_live_max: {self.max_micro_capital:.1%})"
        )
    
    def get_allowed_capital(
        self,
        state: PromotionState,
        portfolio_value: float,
        proposed_size: Optional[float] = None
    ) -> float:
        """
        Calculate allowed capital for a strategy given its promotion state.
        
        Args:
            state: Current promotion state
            total_equity: Total account equity
            proposed_size: Proposed position size
            
        Returns:
            Allowed capital (may be 0, reduced, or full proposed_size)
        """
        # If proposed_size not provided, assume full portfolio allocation
        if proposed_size is None:
            proposed_size = portfolio_value

        # Get state limit
        state_limit_pct = self.state_limits.get(state, self.STATE_CAPITAL_LIMITS.get(state, 0.0))
        
        # Override micro_live with config and safety cap
        if state == PromotionState.MICRO_LIVE:
            state_limit_pct = min(state_limit_pct, self.max_micro_capital)
        
        # Calculate maximum allowed capital
        max_allowed = portfolio_value * state_limit_pct

        # Return minimum of proposed and allowed
        allowed = min(proposed_size, max_allowed)
        
        if allowed < proposed_size:
            logger.debug(
                f"[CAPITAL] Reduced: {proposed_size:.2f} → {allowed:.2f} "
                f"(state={state.value}, limit={state_limit_pct:.1%})"
            )
        
        return allowed
    
    def get_capital_multiplier(
        self,
        state: PromotionState
    ) -> float:
        """
        Get capital multiplier for a promotion state.
        
        This multiplier is applied to proposed position sizes:
            final_size = proposed_size * multiplier
        
        Args:
            state: Current promotion state
            
        Returns:
            Multiplier in range [0.0, 1.0]
        """
        # For MICRO_LIVE, use configured limit
        if state == PromotionState.MICRO_LIVE:
            return min(self.state_limits.get(state, self.max_micro_capital), self.max_micro_capital)
        # For other states, use state limits
        return self.state_limits.get(state, self.STATE_CAPITAL_LIMITS.get(state, 0.0))

    def apply_graduation_multiplier(self, state: PromotionState, requested_capital: float) -> float:
        """Apply graduation multiplier to a requested capital amount.

        Kept as instance method for backward compatibility with tests.
        """
        if not self.is_execution_allowed(state):
            return 0.0
        mult = self.get_capital_multiplier(state)
        return requested_capital * mult

    def _map_state(self, name):
        """Map string or enum-like inputs to PromotionState."""
        normalized = str(name).lower()
        for st in PromotionState:
            if normalized == st.value or normalized == st.name.lower():
                return st
        raise ValueError(f"Unknown promotion state: {name}")

    def get_state_limits(self) -> Dict[PromotionState, float]:
        """Expose current state limits for diagnostics."""
        return dict(self.state_limits)
    
    def is_execution_allowed(self, state: PromotionState) -> bool:
        """
        Check if real execution is allowed in this state.
        
        Args:
            state: Current promotion state
            
        Returns:
            True if real trades can be executed
        """
        return state in [PromotionState.MICRO_LIVE, PromotionState.FULL_LIVE]
    
    def is_shadow_only(self, state: PromotionState) -> bool:
        """
        Check if strategy should run in shadow mode only.
        
        Args:
            state: Current promotion state
            
        Returns:
            True if only shadow execution allowed
        """
        return state == PromotionState.SHADOW
    
    def is_paper_only(self, state: PromotionState) -> bool:
        """
        Check if strategy should run in paper trading only.
        
        Args:
            state: Current promotion state
            
        Returns:
            True if only paper trading allowed
        """
        return state == PromotionState.PAPER
    
    def validate_capital_request(
        self,
        state: PromotionState,
        requested_capital: float,
        total_equity: float
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a capital allocation request.
        
        Args:
            state: Current promotion state
            requested_capital: Requested capital amount
            total_equity: Total account equity
            
        Returns:
            (is_valid, error_message)
        """
        # Check if execution is allowed at all
        if not self.is_execution_allowed(state):
            return False, f"Execution not allowed in {state.value} state"
        
        # Get allowed capital
        allowed = self.get_allowed_capital(state, total_equity, requested_capital)
        
        if requested_capital > allowed:
            return False, (
                f"Requested capital {requested_capital:.2f} exceeds "
                f"allowed {allowed:.2f} for {state.value} state"
            )
        
        return True, None
    
    def get_state_description(self, state: PromotionState) -> str:
        """Get human-readable description of state capital rules."""
        limit_pct = self.state_limits.get(state, self.STATE_CAPITAL_LIMITS.get(state, 0.0))
        
        if state == PromotionState.MICRO_LIVE:
            limit_pct = min(limit_pct, self.max_micro_capital)
        
        descriptions = {
            PromotionState.RESEARCH: "No trading - research only",
            PromotionState.APPROVED: "No trading - awaiting shadow deployment",
            PromotionState.SHADOW: "No execution - signal tracking only",
            PromotionState.PAPER: "Paper trading - simulated fills only",
            PromotionState.MICRO_LIVE: f"Live trading - max {limit_pct:.1%} of equity",
            PromotionState.FULL_LIVE: f"Full allocation - subject to portfolio limits",
            PromotionState.RETIRED: "No trading - strategy retired"
        }
        
        return descriptions.get(state, "Unknown state")
    
    def get_diagnostics(self) -> Dict:
        """Get diagnostic information."""
        return {
            'max_micro_live_capital': self.max_micro_capital,
            'state_limits': {
                state: limit 
                for state, limit in self.state_limits.items()
            },
            'micro_live_override': min(
                self.state_limits.get(PromotionState.MICRO_LIVE, self.max_micro_capital),
                self.max_micro_capital
            )
        }


def apply_graduation_multiplier(
    proposed_size: float,
    promotion_state: PromotionState,
    graduation: CapitalGraduation
) -> float:
    """
    Helper function to apply capital graduation to a proposed position size.
    
    This should be called AFTER all other risk checks (portfolio risk, drawdown, etc.)
    and acts as a FINAL REDUCTION based on promotion state.
    
    Args:
        proposed_size: Position size after all other risk checks
        promotion_state: Current strategy promotion state
        graduation: CapitalGraduation instance
        
    Returns:
        Final position size (may be 0 if state doesn't allow execution)
    """
    # If execution not allowed, return 0
    if not graduation.is_execution_allowed(promotion_state):
        logger.debug(
            f"[CAPITAL] Execution not allowed in {promotion_state.value}, "
            f"size={proposed_size:.2f} → 0.0"
        )
        return 0.0
    
    # Get multiplier and apply
    multiplier = graduation.get_capital_multiplier(promotion_state)
    final_size = proposed_size * multiplier
    
    if final_size < proposed_size:
        logger.debug(
            f"[CAPITAL] Graduation reduction: {proposed_size:.2f} → {final_size:.2f} "
            f"(state={promotion_state.value}, mult={multiplier:.2f})"
        )
    
    return final_size
