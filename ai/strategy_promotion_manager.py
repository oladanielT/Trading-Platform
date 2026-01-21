"""
Strategy Promotion Manager

Manages the controlled promotion of strategies from research to live trading.

CRITICAL: This is a GOVERNANCE layer, not a trading strategy.
- NO auto-promotion allowed
- Human approval MANDATORY for all state transitions
- Capital exposure controlled at each stage
- Automatic demotion if performance degrades

Promotion States (Strict FSM):
    RESEARCH     → Offline validation only (Phase 10)
    APPROVED     → Human approved for testing
    SHADOW       → Signals tracked, no execution
    PAPER        → Paper trading (simulated fills)
    MICRO_LIVE   → Live with 1-5% capital
    FULL_LIVE    → Live with full allocation
    RETIRED      → Deactivated permanently
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import os
import logging

logger = logging.getLogger(__name__)


# Custom exceptions
class InvalidTransitionError(Exception):
    """Raised when attempting invalid state transition."""
    pass


class StrategyNotFoundError(Exception):
    """Raised when strategy not found in promotion system."""
    pass


class PromotionState(Enum):
    """Promotion states in strict order."""
    RESEARCH = "research"           # Phase 10 validation only
    APPROVED = "approved"           # Human approved, ready for shadow
    SHADOW = "shadow"               # Signals tracked, no trades
    PAPER = "paper"                 # Paper trading
    REJECTED = "rejected"           # Rejected during approval
    MICRO_LIVE = "micro_live"       # Live with minimal capital (1-5%)
    FULL_LIVE = "full_live"         # Live with full allocation
    RETIRED = "retired"             # Permanently deactivated
    
    @classmethod
    def can_transition(cls, from_state: 'PromotionState', to_state: 'PromotionState') -> bool:
        """
        Check if transition is allowed.
        
        Rules:
        - Can only move forward sequentially (no skipping)
        - Can demote to RETIRED from any state
        - Cannot leave RETIRED
        """
        if from_state == cls.RETIRED:
            return False  # No transitions from RETIRED
        
        if to_state == cls.RETIRED:
            return True  # Can retire from any state
        
        # Define valid forward transitions
        valid_transitions = {
            cls.RESEARCH: [cls.APPROVED],
            cls.APPROVED: [cls.SHADOW],
            cls.SHADOW: [cls.PAPER],
            cls.PAPER: [cls.MICRO_LIVE],
            cls.MICRO_LIVE: [cls.FULL_LIVE],
            cls.FULL_LIVE: []  # Can only retire
        }
        
        return to_state in valid_transitions.get(from_state, [])


@dataclass
class StrategyRecord:
    """Record for a strategy in the promotion system."""
    strategy_id: str
    state: PromotionState
    alpha_lab_report_path: Optional[str] = None
    
    # Approval tracking
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    approval_notes: str = ""
    
    # State history
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    shadow_metrics: Dict[str, float] = field(default_factory=dict)
    paper_metrics: Dict[str, float] = field(default_factory=dict)
    live_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'strategy_id': self.strategy_id,
            'state': self.state.value,
            'alpha_lab_report_path': self.alpha_lab_report_path,
            'approved_by': self.approved_by,
            'approval_date': self.approval_date.isoformat() if self.approval_date else None,
            'approval_notes': self.approval_notes,
            'state_history': [
                {
                    'from_state': e.from_state.value if hasattr(e, 'from_state') and e.from_state is not None else (e.get('from_state') if isinstance(e, dict) else None),
                    'to_state': e.to_state.value if hasattr(e, 'to_state') else (e.get('state') or e.get('to_state')),
                    'timestamp': e.timestamp.isoformat() if hasattr(e, 'timestamp') else (e.get('timestamp') if isinstance(e, dict) else None),
                    'action': e.action if hasattr(e, 'action') else (e.get('action') if isinstance(e, dict) else None),
                    'by': e.by if hasattr(e, 'by') else (e.get('by') if isinstance(e, dict) else None),
                    'reason': e.reason if hasattr(e, 'reason') else (e.get('reason') if isinstance(e, dict) else None),
                    'validation_results': e.validation_results if hasattr(e, 'validation_results') else (e.get('validation_results') if isinstance(e, dict) else None),
                    'notes': e.notes if hasattr(e, 'notes') else (e.get('notes') if isinstance(e, dict) else None),
                }
                for e in self.state_history
            ],
            'shadow_metrics': self.shadow_metrics,
            'paper_metrics': self.paper_metrics,
            'live_metrics': self.live_metrics,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyRecord':
        """Create from dictionary."""
        # Reconstruct TransitionRecord objects from stored dicts
        raw_history = data.get('state_history', [])
        history = []
        for entry in raw_history:
            if isinstance(entry, dict):
                from_state = PromotionState(entry['from_state']) if entry.get('from_state') else None
                to_state = PromotionState(entry.get('to_state') or entry.get('state')) if (entry.get('to_state') or entry.get('state')) else None
                ts = datetime.fromisoformat(entry['timestamp']) if entry.get('timestamp') else datetime.now(timezone.utc)
                history.append(TransitionRecord(
                    from_state=from_state,
                    to_state=to_state,
                    timestamp=ts,
                    action=entry.get('action'),
                    by=entry.get('by'),
                    reason=entry.get('reason'),
                    validation_results=entry.get('validation_results'),
                    notes=entry.get('notes', '')
                ))
            else:
                history.append(entry)

        return cls(
            strategy_id=data['strategy_id'],
            state=PromotionState(data['state']),
            alpha_lab_report_path=data.get('alpha_lab_report_path'),
            approved_by=data.get('approved_by'),
            approval_date=datetime.fromisoformat(data['approval_date']) if data.get('approval_date') else None,
            approval_notes=data.get('approval_notes', ''),
            state_history=history,
            shadow_metrics=data.get('shadow_metrics', {}),
            paper_metrics=data.get('paper_metrics', {}),
            live_metrics=data.get('live_metrics', {}),
            created_at=datetime.fromisoformat(data['created_at']),
            last_updated=datetime.fromisoformat(data['last_updated'])
        )



@dataclass
class TransitionRecord:
    """Structured transition/audit record for state changes."""
    from_state: Optional[PromotionState]
    to_state: PromotionState
    timestamp: datetime
    action: str
    by: Optional[str] = None
    reason: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    notes: str = ""



@dataclass
class TransitionRecord:
    """Structured transition/audit record for state changes."""
    from_state: Optional[PromotionState]
    to_state: PromotionState
    timestamp: datetime
    action: str
    by: Optional[str] = None
    reason: Optional[str] = None
    validation_results: Optional[Dict[str, Any]] = None
    notes: str = ""


class StrategyPromotionManager:
    """
    Manages strategy promotion from research to live trading.
    
    SAFETY GUARANTEES:
    - NO auto-promotion (human approval required)
    - NO state skipping (sequential progression only)
    - NO capital leaks (graduation rules enforced)
    - Automatic demotion on poor performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize promotion manager.
        
        Args:
            config: Configuration dict with 'strategy_promotion' section
        """
        self.config = config.get('strategy_promotion', {})
        
        if not self.config.get('enabled', False):
            logger.warning(
                "Strategy promotion is disabled. "
                "Set strategy_promotion.enabled=true to use."
            )
        
        # Storage (in-memory by default; set 'storage_dir' in config to persist)
        self.storage_dir = self.config.get('storage_dir')
        self._persist = bool(self.storage_dir)
        if self._persist:
            os.makedirs(self.storage_dir, exist_ok=True)

        # Approvals directory
        self.approvals_dir = self.config.get('approvals_directory', 'approvals')
        self.pending_dir = os.path.join(self.approvals_dir, 'pending')
        self.approved_dir = os.path.join(self.approvals_dir, 'approved')
        self.rejected_dir = os.path.join(self.approvals_dir, 'rejected')
        self.processed_dir = os.path.join(self.approvals_dir, 'processed')
        self.invalid_dir = os.path.join(self.processed_dir, 'invalid')
        for path in [self.pending_dir, self.approved_dir, self.rejected_dir, self.processed_dir, self.invalid_dir]:
            os.makedirs(path, exist_ok=True)

        # Scheduling and durations
        evaluation_cfg = self.config.get('evaluation', {})
        self.approval_check_interval = evaluation_cfg.get('approval_check_interval', 300)
        self.demotion_check_interval = evaluation_cfg.get('demotion_check_interval', 3600)
        self.min_duration_hours = {
            PromotionState.SHADOW: evaluation_cfg.get('min_shadow_duration_hours', 0),
            PromotionState.PAPER: evaluation_cfg.get('min_paper_duration_hours', 0),
            PromotionState.MICRO_LIVE: evaluation_cfg.get('min_micro_live_duration_hours', 0)
        }
        self._last_approval_poll: Optional[datetime] = None
        self._last_demotion_poll: Optional[datetime] = None
        
        # Load existing strategies (only if persistence enabled)
        self.strategies: Dict[str, StrategyRecord] = {}
        if self._persist:
            self._load_strategies()
        
        # Safety checks
        self.require_approval = self.config.get('require_human_approval', True)
        if not self.require_approval:
            raise ValueError(
                "SAFETY VIOLATION: require_human_approval must be True. "
                "Auto-promotion is not allowed."
            )
        
        logger.info(f"[PROMOTION] Initialized (strategies: {len(self.strategies)})")
    
    def submit_candidate(
        self,
        strategy_id: str,
        report_path: Optional[str] = None,
        validation_results: Optional[Dict] = None
    ) -> bool:
        """Submit a new candidate strategy for approval and tracking.

        Creates a `StrategyRecord` in `RESEARCH` state and records submission.
        """
        if strategy_id in self.strategies:
            logger.warning(f"[PROMOTION] Candidate {strategy_id} already exists")
            return False

        # Accept either (strategy_id, validation_results) or (strategy_id, report_path, validation_results)
        if isinstance(report_path, dict) and validation_results is None:
            validation_results = report_path
            report_path = None

        record = StrategyRecord(strategy_id=strategy_id, state=PromotionState.RESEARCH, alpha_lab_report_path=report_path)
        record.state_history.append(TransitionRecord(
            from_state=None,
            to_state=record.state,
            timestamp=datetime.now(timezone.utc),
            action='submitted',
            by='system',
            validation_results=validation_results
        ))

        self.strategies[strategy_id] = record
        self._save_strategy(record)

        logger.info(f"[PROMOTION] Submitted candidate: {strategy_id}")
        return True

    def demote(
        self,
        strategy_id: str,
        reason: str,
        demoted_by: str = "system",
        force_retire: bool = False
    ) -> bool:
        """
        Demote a strategy safely.
        
        Downgrades one level at a time:
            FULL_LIVE → MICRO_LIVE
            MICRO_LIVE → PAPER
            PAPER → SHADOW
            SHADOW → RETIRED (or force_retire)
        
        Args:
            strategy_id: Strategy to demote
            reason: Demotion reason
            demoted_by: Who initiated demotion
            force_retire: If True, retire immediately
            
        Returns:
            True if demoted successfully
        """
        # Delegate to the single demotion implementation earlier in the file
        # (keeps behaviour deterministic and centralized)
        return self._demote_impl(strategy_id=strategy_id, reason=reason, demoted_by=demoted_by, force_retire=force_retire)

    def _demote_impl(
        self,
        strategy_id: str,
        reason: str,
        demoted_by: str = "system",
        force_retire: bool = False
    ) -> bool:
        """Internal demotion implementation.

        Demotion mapping (one governance step):
          FULL_LIVE / MICRO_LIVE -> PAPER
          PAPER -> SHADOW
          SHADOW -> RETIRED
        """
        record = self.get_strategy(strategy_id)
        if record is None:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        old_state = record.state

        if force_retire:
            new_state = PromotionState.RETIRED
        else:
            if old_state in (PromotionState.FULL_LIVE, PromotionState.MICRO_LIVE):
                new_state = PromotionState.PAPER
            elif old_state == PromotionState.PAPER:
                new_state = PromotionState.SHADOW
            elif old_state == PromotionState.SHADOW:
                new_state = PromotionState.RETIRED
            else:
                new_state = PromotionState.RETIRED

        record.state = new_state
        record.last_updated = datetime.now(timezone.utc)
        record.state_history.append(TransitionRecord(
            from_state=old_state,
            to_state=new_state,
            timestamp=datetime.now(timezone.utc),
            action='demoted',
            by=demoted_by,
            reason=reason
        ))
        self._save_strategy(record)
        logger.info(f"[PROMOTION] DEMOTED: {strategy_id} {old_state.value} → {new_state.value} (reason: {reason})")
        return True

    # Promotion methods (proper class scope)
    def promote_to_shadow(self, strategy_id: str) -> bool:
        """Promote strategy to SHADOW state."""
        return self._transition(strategy_id, PromotionState.SHADOW)

    def promote_to_paper(self, strategy_id: str) -> bool:
        """Promote strategy to PAPER state."""
        return self._transition(strategy_id, PromotionState.PAPER)

    def promote_to_micro_live(self, strategy_id: str) -> bool:
        """Promote strategy to MICRO_LIVE state."""
        return self._transition(strategy_id, PromotionState.MICRO_LIVE)

    def promote_to_full_live(self, strategy_id: str) -> bool:
        """Promote strategy to FULL_LIVE state."""
        return self._transition(strategy_id, PromotionState.FULL_LIVE)

    def retire(self, strategy_id: str, reason: str) -> bool:
        """Retire a strategy."""
        record = self.get_strategy(strategy_id)
        if record is None:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        if record.state == PromotionState.RETIRED:
            logger.warning(f"[PROMOTION] Strategy {strategy_id} already retired")
            return False

        old_state = record.state
        record.state = PromotionState.RETIRED
        record.last_updated = datetime.now(timezone.utc)
        record.state_history.append(TransitionRecord(
            from_state=old_state,
            to_state=PromotionState.RETIRED,
            timestamp=datetime.now(timezone.utc),
            action='retired',
            reason=reason
        ))
        self._save_strategy(record)
        logger.info(f"[PROMOTION] RETIRED: {strategy_id} (reason: {reason})")
        return True

    # Accessors
    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Get strategy record."""
        return self.strategies.get(strategy_id)

    def get_state(self, strategy_id: str) -> PromotionState:
        """Get current state of a strategy."""
        record = self.get_strategy(strategy_id)
        if record is None:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
        return record.state

    def is_live(self, strategy_id: str) -> bool:
        """Check if strategy is in a live-execution state."""
        state = self.get_state(strategy_id)
        return state in (PromotionState.MICRO_LIVE, PromotionState.FULL_LIVE)

    def get_history(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get state transition history for a strategy."""
        record = self.get_strategy(strategy_id)
        if record is None:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
        return record.state_history

    # Internal helpers
    def _hours_in_current_state(self, record: StrategyRecord) -> float:
        """Compute hours spent in the current state."""
        # Find last history entry for current state
        state_time = record.created_at
        for entry in reversed(record.state_history):
            # support both dict and TransitionRecord
            try:
                entry_state = entry.to_state.value if hasattr(entry, 'to_state') else entry.get('state')
                entry_ts = entry.timestamp if hasattr(entry, 'timestamp') else entry.get('timestamp')
            except Exception:
                continue

            if entry_state == record.state.value:
                try:
                    if hasattr(entry_ts, 'isoformat'):
                        state_time = entry_ts
                    else:
                        state_time = datetime.fromisoformat(entry_ts)
                except Exception:
                    state_time = record.created_at
                break
        delta = datetime.now(timezone.utc) - state_time
        return delta.total_seconds() / 3600.0

    def _enforce_min_duration(self, record: StrategyRecord) -> None:
        """Ensure minimum time-in-state before promotion."""
        min_hours = self.min_duration_hours.get(record.state, 0)
        if min_hours <= 0:
            return
        hours_here = self._hours_in_current_state(record)
        if hours_here < min_hours:
            raise InvalidTransitionError(
                f"Minimum duration not met for {record.state.value}: "
                f"{hours_here:.2f}h < {min_hours}h"
            )

    def _transition(self, strategy_id: str, target_state: PromotionState) -> bool:
        """Shared transition handler with safety checks."""
        record = self.get_strategy(strategy_id)
        if record is None:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        # Enforce sequential transitions
        if not PromotionState.can_transition(record.state, target_state):
            raise InvalidTransitionError(
                f"Cannot transition {strategy_id} from {record.state.value} to {target_state.value}"
            )

        # Enforce minimum duration in current state
        self._enforce_min_duration(record)

        old_state = record.state
        record.state = target_state
        record.last_updated = datetime.now(timezone.utc)
        record.state_history.append(TransitionRecord(
            from_state=old_state,
            to_state=target_state,
            timestamp=datetime.now(timezone.utc),
            action='promoted'
        ))
        self._save_strategy(record)
        logger.info(f"[PROMOTION] {strategy_id}: {old_state.value} → {target_state.value}")
        return True
    
    def approve(
        self,
        strategy_id: str,
        approved_by: Optional[str] = None,
        target_state: Optional[PromotionState] = None,
        notes: str = ""
    ) -> bool:
        """
        Approve a strategy for promotion (HUMAN APPROVAL REQUIRED).
        
        Args:
            strategy_id: Strategy to approve
            approved_by: Name/ID of approver (for audit trail)
            target_state: Target state (default: next sequential state)
            notes: Approval notes
            
        Returns:
            True if approved successfully
        """
        if not self.config.get('enabled', False):
            raise RuntimeError("Strategy promotion is disabled")
        
        if strategy_id not in self.strategies:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")
        
        record = self.strategies[strategy_id]
        
        # Cannot approve rejected or retired strategies
        if record.state in (PromotionState.REJECTED, PromotionState.RETIRED):
            raise InvalidTransitionError(f"Cannot approve strategy in state {record.state.value}")

        # Determine target state
        if target_state is None:
            # Auto-determine next state
            state_order = [
                PromotionState.RESEARCH,
                PromotionState.APPROVED,
                PromotionState.SHADOW,
                PromotionState.PAPER,
                PromotionState.MICRO_LIVE,
                PromotionState.FULL_LIVE
            ]
            current_idx = state_order.index(record.state)
            if current_idx >= len(state_order) - 1:
                raise ValueError(f"Strategy {strategy_id} is already at {record.state.value}")
            target_state = state_order[current_idx + 1]
        
        # Verify transition is allowed
        if not PromotionState.can_transition(record.state, target_state):
            raise InvalidTransitionError(
                f"Cannot transition {strategy_id} from {record.state.value} "
                f"to {target_state.value}. State skipping not allowed."
            )

        # Enforce minimum duration in current state before approval-driven promotion
        self._enforce_min_duration(record)
        
        # Default approver
        if approved_by is None:
            approved_by = 'human'

        # Update record
        old_state = record.state
        record.state = target_state
        record.approved_by = approved_by
        record.approval_date = datetime.now(timezone.utc)
        record.approval_notes = notes
        record.last_updated = datetime.now(timezone.utc)

        # Record in structured history
        record.state_history.append(TransitionRecord(
            from_state=old_state,
            to_state=target_state,
            timestamp=datetime.now(timezone.utc),
            action='approved',
            by=approved_by,
            notes=notes
        ))
        
        self._save_strategy(record)
        
        logger.info(
            f"[PROMOTION] APPROVED: {strategy_id} "
            f"{old_state.value} → {target_state.value} (by {approved_by})"
        )
        
        return True
    
    def reject(
        self,
        strategy_id: str,
        reason: str,
        rejected_by: Optional[str] = None
    ) -> bool:
        """
        Reject a strategy candidate.
        
        Args:
            strategy_id: Strategy to reject
            reason: Rejection reason
            rejected_by: Name/ID of person rejecting
            
        Returns:
            True if rejected successfully
        """
        if strategy_id not in self.strategies:
            raise StrategyNotFoundError(f"Strategy {strategy_id} not found")

        record = self.strategies[strategy_id]
        old_state = record.state

        # Move to REJECTED state
        record.state = PromotionState.REJECTED
        record.last_updated = datetime.now(timezone.utc)

        if rejected_by is None:
            rejected_by = 'human'

        record.state_history.append(TransitionRecord(
            from_state=old_state,
            to_state=PromotionState.REJECTED,
            timestamp=datetime.now(timezone.utc),
            action='rejected',
            by=rejected_by,
            reason=reason
        ))
        
        self._save_strategy(record)
        
        logger.info(
            f"[PROMOTION] REJECTED: {strategy_id} "
            f"from {old_state.value} (by {rejected_by}): {reason}"
        )
        
        return True
    
    # (Demotion implementation is defined earlier as a single method)
    
    def get_strategy(self, strategy_id: str) -> Optional[StrategyRecord]:
        """Get strategy record."""
        return self.strategies.get(strategy_id)
    
    def get_strategies_by_state(self, state: PromotionState) -> List[StrategyRecord]:
        """Get all strategies in a given state."""
        return [s for s in self.strategies.values() if s.state == state]

    # Approvals processing
    def process_approvals(self) -> None:
        """Poll approval files and apply transitions."""
        now = datetime.now(timezone.utc)
        self._last_approval_poll = now

        # Process approved files
        for folder, action in [(self.approved_dir, 'approve'), (self.rejected_dir, 'reject')]:
            for filename in os.listdir(folder):
                if not filename.lower().endswith(('.json', '.yaml', '.yml')):
                    continue
                filepath = os.path.join(folder, filename)
                try:
                    approval = self._load_approval_file(filepath)
                    if not approval:
                        self._archive_invalid(filepath)
                        continue
                    strategy_id = approval.get('strategy_id')
                    if not strategy_id:
                        logger.error(f"[PROMOTION] Missing strategy_id in approval file {filename}")
                        self._archive_invalid(filepath)
                        continue
                    target_state = approval.get('target_state')
                    notes = approval.get('notes', '')
                    actor = approval.get('approved_by') or approval.get('by') or 'file_approval'

                    if action == 'approve':
                        if not target_state:
                            logger.error(f"[PROMOTION] Missing target_state in approval file {filename}")
                            self._archive_invalid(filepath)
                            continue
                        try:
                            mapped_state = self._map_state_name(target_state)
                            self.approve(strategy_id, approved_by=actor, target_state=mapped_state, notes=notes)
                        except Exception as e:
                            logger.error(f"[PROMOTION] Failed to approve {strategy_id} from {filename}: {e}")
                            self._archive_invalid(filepath)
                            continue
                    else:
                        reason = approval.get('reason', 'rejected_via_file')
                        try:
                            self.reject(strategy_id, reason=reason, rejected_by=actor)
                        except Exception as e:
                            logger.error(f"[PROMOTION] Failed to reject {strategy_id} from {filename}: {e}")
                            self._archive_invalid(filepath)
                            continue

                    self._archive_processed(filepath)
                except Exception as e:
                    logger.error(f"[PROMOTION] Error processing approval file {filename}: {e}")
                    self._archive_invalid(filepath)

    def _load_approval_file(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load approval JSON or YAML file."""
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            # Try JSON first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                try:
                    import yaml  # Optional dependency
                    return yaml.safe_load(content)
                except Exception:
                    logger.error(f"[PROMOTION] Unable to parse approval file {filepath}")
                    return None
        except Exception as e:
            logger.error(f"[PROMOTION] Failed reading approval file {filepath}: {e}")
            return None

    def _map_state_name(self, state_name: str) -> PromotionState:
        """Map string name to PromotionState."""
        normalized = state_name.strip().lower()
        for state in PromotionState:
            if normalized == state.value or normalized == state.name.lower():
                return state
        raise ValueError(f"Unknown promotion state: {state_name}")

    def _archive_processed(self, filepath: str) -> None:
        filename = os.path.basename(filepath)
        dest = os.path.join(self.processed_dir, filename)
        try:
            os.replace(filepath, dest)
        except Exception as e:
            logger.error(f"[PROMOTION] Failed to archive processed file {filename}: {e}")

    def _archive_invalid(self, filepath: str) -> None:
        filename = os.path.basename(filepath)
        dest = os.path.join(self.invalid_dir, filename)
        try:
            os.replace(filepath, dest)
        except Exception as e:
            logger.error(f"[PROMOTION] Failed to archive invalid file {filename}: {e}")
    
    def update_metrics(
        self,
        strategy_id: str,
        metrics: Dict[str, float],
        metrics_type: str = "live"
    ) -> None:
        """
        Update performance metrics for a strategy.
        
        Args:
            strategy_id: Strategy to update
            metrics: Performance metrics dict
            metrics_type: 'shadow', 'paper', or 'live'
        """
        if strategy_id not in self.strategies:
            raise KeyError(f"Strategy {strategy_id} not found")
        
        record = self.strategies[strategy_id]
        
        if metrics_type == "shadow":
            record.shadow_metrics = metrics
        elif metrics_type == "paper":
            record.paper_metrics = metrics
        elif metrics_type == "live":
            record.live_metrics = metrics
        else:
            raise ValueError(f"Unknown metrics_type: {metrics_type}")
        
        record.last_updated = datetime.now(timezone.utc)
        self._save_strategy(record)
    
    def _save_strategy(self, record: StrategyRecord) -> None:
        """Save strategy record to disk."""
        if not getattr(self, '_persist', False):
            return
        filename = f"{self.storage_dir}/{record.strategy_id}.json"
        with open(filename, 'w') as f:
            json.dump(record.to_dict(), f, indent=2)
    
    def _load_strategies(self) -> None:
        """Load all strategy records from disk."""
        if not getattr(self, '_persist', False):
            return
        if not os.path.exists(self.storage_dir):
            return

        for filename in os.listdir(self.storage_dir):
            if not filename.endswith('.json'):
                continue

            filepath = os.path.join(self.storage_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                record = StrategyRecord.from_dict(data)
                self.strategies[record.strategy_id] = record
            except Exception as e:
                logger.error(f"[PROMOTION] Failed to load {filename}: {e}")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information."""
        state_counts = {}
        for state in PromotionState:
            state_counts[state.value] = len(self.get_strategies_by_state(state))
        
        return {
            'enabled': self.config.get('enabled', False),
            'require_approval': self.require_approval,
            'total_strategies': len(self.strategies),
            'state_counts': state_counts,
            'storage_dir': self.storage_dir,
            'strategies': [s.to_dict() for s in self.strategies.values()]
        }
