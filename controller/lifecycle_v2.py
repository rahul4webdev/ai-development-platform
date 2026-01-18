"""
Phase 15.2: Continuous Change Cycles

This module implements a deterministic lifecycle state machine for managing
project and change lifecycles with explicit states, validated transitions,
and immutable audit logging.

Key features:
- Deterministic state machine (no state skipping)
- PROJECT_MODE and CHANGE_MODE support
- Multi-aspect isolation
- Event-driven transitions (Claude completion, test results, feedback, approval)
- Persistent state with crash recovery
- Safety guarantees (no implicit approvals, no cross-aspect side effects)
- Continuous change cycles (DEPLOYED → AWAITING_FEEDBACK → FIXING → ...)
- Change history and lineage tracking
- Deployment summaries

States:
    CREATED → PLANNING → DEVELOPMENT → TESTING → AWAITING_FEEDBACK →
    FIXING → READY_FOR_PRODUCTION → DEPLOYED → ARCHIVED
    (REJECTED can occur from PLANNING, DEVELOPMENT, TESTING, AWAITING_FEEDBACK)

Continuous Loops (Phase 15.2):
    DEPLOYED → AWAITING_FEEDBACK → FIXING → READY_FOR_PRODUCTION → DEPLOYED

IMPORTANT: This module does NOT replace Phase 12 lifecycle_engine.py.
It ADDS a new deterministic lifecycle layer for Phase 15.1+.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

logger = logging.getLogger("lifecycle_v2")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LIFECYCLE_STATE_DIR = Path("/home/aitesting.mybd.in/jobs/lifecycle")
LIFECYCLE_AUDIT_LOG = LIFECYCLE_STATE_DIR / "lifecycle_audit.log"

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class LifecycleState(str, Enum):
    """
    Deterministic lifecycle states.

    State machine enforces explicit, validated transitions only.
    No state skipping is allowed.

    Phase 15.2: DEPLOYED is no longer terminal - can loop back to AWAITING_FEEDBACK.
    """
    CREATED = "created"
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    AWAITING_FEEDBACK = "awaiting_feedback"
    FIXING = "fixing"
    READY_FOR_PRODUCTION = "ready_for_production"
    DEPLOYED = "deployed"
    REJECTED = "rejected"
    ARCHIVED = "archived"

    @classmethod
    def terminal_states(cls) -> Set["LifecycleState"]:
        """States that indicate lifecycle completion (no further work)."""
        # Phase 15.2: DEPLOYED is no longer terminal, only REJECTED and ARCHIVED
        return {cls.REJECTED, cls.ARCHIVED}

    @classmethod
    def deployed_states(cls) -> Set["LifecycleState"]:
        """States indicating deployment (may still receive changes)."""
        return {cls.DEPLOYED}

    @classmethod
    def active_states(cls) -> Set["LifecycleState"]:
        """States where work is actively being performed."""
        return {cls.PLANNING, cls.DEVELOPMENT, cls.TESTING, cls.FIXING}


class LifecycleMode(str, Enum):
    """
    Mode of work for a lifecycle instance.

    PROJECT_MODE: New project development
    CHANGE_MODE: Feature, bug, improvement on existing project
    """
    PROJECT_MODE = "project_mode"
    CHANGE_MODE = "change_mode"


class ChangeType(str, Enum):
    """
    Type of change for CHANGE_MODE lifecycles.

    Phase 15.2: Added SECURITY for security-related fixes.
    """
    BUG = "bug"
    FEATURE = "feature"
    IMPROVEMENT = "improvement"
    REFACTOR = "refactor"
    SECURITY = "security"  # Phase 15.2: Security fixes


class ProjectAspect(str, Enum):
    """
    Project aspects for multi-aspect enforcement.

    Each aspect has its own lifecycle, Claude jobs, and approvals.
    Aspects cannot affect each other implicitly.
    """
    CORE = "core"
    BACKEND = "backend"
    FRONTEND_WEB = "frontend_web"
    FRONTEND_MOBILE = "frontend_mobile"
    ADMIN = "admin"
    CUSTOM = "custom"


class TransitionTrigger(str, Enum):
    """
    Valid triggers for lifecycle transitions.

    Transitions may ONLY be triggered by these events.

    Phase 15.2: Added CONTINUOUS_CHANGE_REQUEST for post-deployment loops.
    """
    CLAUDE_JOB_COMPLETED = "claude_job_completed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    TELEGRAM_FEEDBACK = "telegram_feedback"
    HUMAN_APPROVAL = "human_approval"
    HUMAN_REJECTION = "human_rejection"
    SYSTEM_INIT = "system_init"
    MANUAL_ARCHIVE = "manual_archive"
    # Phase 15.2: Continuous change cycle triggers
    CONTINUOUS_CHANGE_REQUEST = "continuous_change_request"  # DEPLOYED -> AWAITING_FEEDBACK


class UserRole(str, Enum):
    """
    User roles for permission validation.
    """
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    TESTER = "tester"
    VIEWER = "viewer"


# -----------------------------------------------------------------------------
# Valid Transitions
# -----------------------------------------------------------------------------

# Map of current_state -> {trigger: target_state}
# Phase 15.2: Added DEPLOYED -> AWAITING_FEEDBACK loop for continuous changes
VALID_TRANSITIONS: Dict[LifecycleState, Dict[TransitionTrigger, LifecycleState]] = {
    LifecycleState.CREATED: {
        TransitionTrigger.SYSTEM_INIT: LifecycleState.PLANNING,
    },
    LifecycleState.PLANNING: {
        TransitionTrigger.CLAUDE_JOB_COMPLETED: LifecycleState.DEVELOPMENT,
        TransitionTrigger.HUMAN_REJECTION: LifecycleState.REJECTED,
    },
    LifecycleState.DEVELOPMENT: {
        TransitionTrigger.CLAUDE_JOB_COMPLETED: LifecycleState.TESTING,
        TransitionTrigger.HUMAN_REJECTION: LifecycleState.REJECTED,
    },
    LifecycleState.TESTING: {
        TransitionTrigger.TEST_PASSED: LifecycleState.AWAITING_FEEDBACK,
        TransitionTrigger.TEST_FAILED: LifecycleState.FIXING,
        TransitionTrigger.HUMAN_REJECTION: LifecycleState.REJECTED,
    },
    LifecycleState.AWAITING_FEEDBACK: {
        TransitionTrigger.HUMAN_APPROVAL: LifecycleState.READY_FOR_PRODUCTION,
        TransitionTrigger.TELEGRAM_FEEDBACK: LifecycleState.FIXING,
        TransitionTrigger.HUMAN_REJECTION: LifecycleState.REJECTED,
    },
    LifecycleState.FIXING: {
        TransitionTrigger.CLAUDE_JOB_COMPLETED: LifecycleState.TESTING,
        TransitionTrigger.HUMAN_REJECTION: LifecycleState.REJECTED,
    },
    LifecycleState.READY_FOR_PRODUCTION: {
        TransitionTrigger.HUMAN_APPROVAL: LifecycleState.DEPLOYED,
        TransitionTrigger.HUMAN_REJECTION: LifecycleState.REJECTED,
    },
    LifecycleState.DEPLOYED: {
        # Phase 15.2: Continuous change cycle - DEPLOYED can loop back to AWAITING_FEEDBACK
        TransitionTrigger.CONTINUOUS_CHANGE_REQUEST: LifecycleState.AWAITING_FEEDBACK,
        TransitionTrigger.MANUAL_ARCHIVE: LifecycleState.ARCHIVED,
    },
    LifecycleState.REJECTED: {
        TransitionTrigger.MANUAL_ARCHIVE: LifecycleState.ARCHIVED,
    },
    LifecycleState.ARCHIVED: {
        # Terminal state - no transitions out
    },
}

# Roles that can trigger each transition
TRANSITION_PERMISSIONS: Dict[TransitionTrigger, Set[UserRole]] = {
    TransitionTrigger.CLAUDE_JOB_COMPLETED: {UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER},
    TransitionTrigger.TEST_PASSED: {UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER},
    TransitionTrigger.TEST_FAILED: {UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER},
    TransitionTrigger.TELEGRAM_FEEDBACK: {UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER},
    TransitionTrigger.HUMAN_APPROVAL: {UserRole.OWNER, UserRole.ADMIN},
    TransitionTrigger.HUMAN_REJECTION: {UserRole.OWNER, UserRole.ADMIN},
    TransitionTrigger.SYSTEM_INIT: {UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER},
    TransitionTrigger.MANUAL_ARCHIVE: {UserRole.OWNER, UserRole.ADMIN},
    # Phase 15.2: Continuous change triggers
    TransitionTrigger.CONTINUOUS_CHANGE_REQUEST: {UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER},
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class ChangeReference:
    """
    Reference information for CHANGE_MODE lifecycles.
    """
    project_id: str
    aspect: ProjectAspect
    change_type: ChangeType
    parent_lifecycle_id: Optional[str] = None  # For tracking related changes
    description: str = ""


@dataclass
class CycleRecord:
    """
    Phase 15.2: Record of a deployment cycle iteration.

    Each time a lifecycle loops from DEPLOYED -> AWAITING_FEEDBACK -> ... -> DEPLOYED,
    a new cycle record is created.
    """
    cycle_number: int
    started_at: datetime
    completed_at: Optional[datetime] = None
    change_type: Optional[ChangeType] = None
    change_summary: str = ""
    previous_deployment_id: Optional[str] = None  # Claude job ID of previous deployment
    deployment_job_id: Optional[str] = None  # Claude job ID of this deployment
    files_changed: List[str] = field(default_factory=list)
    commit_hash: Optional[str] = None
    feedback_summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_number": self.cycle_number,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "change_type": self.change_type.value if self.change_type else None,
            "change_summary": self.change_summary,
            "previous_deployment_id": self.previous_deployment_id,
            "deployment_job_id": self.deployment_job_id,
            "files_changed": self.files_changed,
            "commit_hash": self.commit_hash,
            "feedback_summary": self.feedback_summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CycleRecord":
        return cls(
            cycle_number=data["cycle_number"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            change_type=ChangeType(data["change_type"]) if data.get("change_type") else None,
            change_summary=data.get("change_summary", ""),
            previous_deployment_id=data.get("previous_deployment_id"),
            deployment_job_id=data.get("deployment_job_id"),
            files_changed=data.get("files_changed", []),
            commit_hash=data.get("commit_hash"),
            feedback_summary=data.get("feedback_summary", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TransitionRecord:
    """
    Immutable record of a state transition.
    """
    record_id: str
    lifecycle_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    trigger: TransitionTrigger
    triggered_by: str
    triggered_at: datetime
    role: UserRole
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "lifecycle_id": self.lifecycle_id,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "trigger": self.trigger.value,
            "triggered_by": self.triggered_by,
            "triggered_at": self.triggered_at.isoformat(),
            "role": self.role.value,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass
class LifecycleInstance:
    """
    A single lifecycle instance for a project or change.

    Each instance tracks:
    - Identity and mode
    - Current state
    - Associated Claude jobs
    - Transition history
    - Aspect isolation
    - Phase 15.2: Cycle history for continuous change loops
    """
    lifecycle_id: str
    project_name: str
    mode: LifecycleMode
    aspect: ProjectAspect
    state: LifecycleState
    created_at: datetime
    created_by: str
    # CHANGE_MODE reference (None for PROJECT_MODE)
    change_reference: Optional[ChangeReference] = None
    # Timestamps
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    # Associated Claude jobs
    claude_job_ids: List[str] = field(default_factory=list)
    current_claude_job_id: Optional[str] = None
    # Test results
    test_run_ids: List[str] = field(default_factory=list)
    last_test_passed: Optional[bool] = None
    # Feedback
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    # Transition history
    transition_count: int = 0
    last_transition_at: Optional[datetime] = None
    # Phase 15.2: Continuous change cycle tracking
    cycle_number: int = 1  # Current cycle (starts at 1, increments on each DEPLOYED -> AWAITING_FEEDBACK loop)
    cycle_history: List[Dict[str, Any]] = field(default_factory=list)  # History of all cycles
    current_change_summary: str = ""  # Summary of changes in current cycle
    previous_deployment_id: Optional[str] = None  # Claude job ID of previous deployment
    # Metadata
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lifecycle_id": self.lifecycle_id,
            "project_name": self.project_name,
            "mode": self.mode.value,
            "aspect": self.aspect.value,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "change_reference": {
                "project_id": self.change_reference.project_id,
                "aspect": self.change_reference.aspect.value,
                "change_type": self.change_reference.change_type.value,
                "parent_lifecycle_id": self.change_reference.parent_lifecycle_id,
                "description": self.change_reference.description,
            } if self.change_reference else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "claude_job_ids": self.claude_job_ids,
            "current_claude_job_id": self.current_claude_job_id,
            "test_run_ids": self.test_run_ids,
            "last_test_passed": self.last_test_passed,
            "feedback_history": self.feedback_history,
            "transition_count": self.transition_count,
            "last_transition_at": self.last_transition_at.isoformat() if self.last_transition_at else None,
            # Phase 15.2: Cycle tracking
            "cycle_number": self.cycle_number,
            "cycle_history": self.cycle_history,
            "current_change_summary": self.current_change_summary,
            "previous_deployment_id": self.previous_deployment_id,
            "description": self.description,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifecycleInstance":
        """Deserialize from dictionary."""
        change_ref = None
        if data.get("change_reference"):
            cr = data["change_reference"]
            change_ref = ChangeReference(
                project_id=cr["project_id"],
                aspect=ProjectAspect(cr["aspect"]),
                change_type=ChangeType(cr["change_type"]),
                parent_lifecycle_id=cr.get("parent_lifecycle_id"),
                description=cr.get("description", ""),
            )

        return cls(
            lifecycle_id=data["lifecycle_id"],
            project_name=data["project_name"],
            mode=LifecycleMode(data["mode"]),
            aspect=ProjectAspect(data["aspect"]),
            state=LifecycleState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            change_reference=change_ref,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            claude_job_ids=data.get("claude_job_ids", []),
            current_claude_job_id=data.get("current_claude_job_id"),
            test_run_ids=data.get("test_run_ids", []),
            last_test_passed=data.get("last_test_passed"),
            feedback_history=data.get("feedback_history", []),
            transition_count=data.get("transition_count", 0),
            last_transition_at=datetime.fromisoformat(data["last_transition_at"]) if data.get("last_transition_at") else None,
            # Phase 15.2: Cycle tracking
            cycle_number=data.get("cycle_number", 1),
            cycle_history=data.get("cycle_history", []),
            current_change_summary=data.get("current_change_summary", ""),
            previous_deployment_id=data.get("previous_deployment_id"),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Lifecycle State Machine
# -----------------------------------------------------------------------------

class LifecycleStateMachine:
    """
    Deterministic lifecycle state machine.

    Enforces:
    - No state skipping
    - Explicit, validated transitions only
    - Role-based permissions
    - Immutable audit logging
    """

    def __init__(self, state_dir: Path = LIFECYCLE_STATE_DIR):
        self._state_dir = state_dir
        self._audit_log = state_dir / "lifecycle_audit.log"
        self._lock = asyncio.Lock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure state directories exist."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.debug(f"Could not create state directory: {e}")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def can_transition(
        self,
        current_state: LifecycleState,
        trigger: TransitionTrigger,
        user_role: UserRole,
    ) -> Tuple[bool, str, Optional[LifecycleState]]:
        """
        Check if a transition is valid.

        Returns (allowed, reason, target_state)
        """
        # Check if trigger is valid for current state
        valid_triggers = VALID_TRANSITIONS.get(current_state, {})
        if trigger not in valid_triggers:
            valid_list = list(valid_triggers.keys())
            return False, f"Invalid trigger '{trigger.value}' for state '{current_state.value}'. Valid triggers: {[t.value for t in valid_list]}", None

        # Check role permission
        allowed_roles = TRANSITION_PERMISSIONS.get(trigger, set())
        if user_role not in allowed_roles:
            return False, f"Role '{user_role.value}' cannot trigger '{trigger.value}'. Required: {[r.value for r in allowed_roles]}", None

        target_state = valid_triggers[trigger]
        return True, f"Transition allowed: {current_state.value} -> {target_state.value}", target_state

    def get_valid_transitions(self, current_state: LifecycleState) -> Dict[TransitionTrigger, LifecycleState]:
        """Get all valid transitions from current state."""
        return VALID_TRANSITIONS.get(current_state, {})

    def get_next_guidance(self, lifecycle: LifecycleInstance) -> Dict[str, Any]:
        """
        Get guidance for what happens next in the lifecycle.

        Used by Telegram bot to show users what actions are available.
        """
        state = lifecycle.state
        valid_transitions = self.get_valid_transitions(state)

        guidance = {
            "current_state": state.value,
            "available_actions": [],
            "waiting_for": None,
            "next_step": None,
        }

        # State-specific guidance
        if state == LifecycleState.CREATED:
            guidance["next_step"] = "Initialize lifecycle to begin planning"
            guidance["waiting_for"] = "system initialization"
            guidance["available_actions"] = ["start_planning"]

        elif state == LifecycleState.PLANNING:
            guidance["next_step"] = "Claude is generating implementation plan"
            guidance["waiting_for"] = "Claude job completion"
            guidance["available_actions"] = ["reject"]

        elif state == LifecycleState.DEVELOPMENT:
            guidance["next_step"] = "Claude is implementing the feature/fix"
            guidance["waiting_for"] = "Claude job completion"
            guidance["available_actions"] = ["reject"]

        elif state == LifecycleState.TESTING:
            guidance["next_step"] = "Automated tests are running"
            guidance["waiting_for"] = "test results"
            guidance["available_actions"] = ["reject"]

        elif state == LifecycleState.AWAITING_FEEDBACK:
            guidance["next_step"] = "Human review and feedback required"
            guidance["waiting_for"] = "human approval or feedback"
            guidance["available_actions"] = ["approve", "provide_feedback", "reject"]

        elif state == LifecycleState.FIXING:
            guidance["next_step"] = "Claude is fixing reported issues"
            guidance["waiting_for"] = "Claude job completion"
            guidance["available_actions"] = ["reject"]

        elif state == LifecycleState.READY_FOR_PRODUCTION:
            guidance["next_step"] = "Final approval required for production deployment"
            guidance["waiting_for"] = "production approval"
            guidance["available_actions"] = ["approve_production", "reject"]

        elif state == LifecycleState.DEPLOYED:
            # Phase 15.2: DEPLOYED is no longer terminal - can request continuous changes
            guidance["next_step"] = "Deployment complete - can request changes or archive"
            guidance["waiting_for"] = None
            guidance["available_actions"] = ["new_feature", "report_bug", "improve", "refactor", "security_fix", "archive"]
            guidance["cycle_number"] = lifecycle.cycle_number  # Phase 15.2: Include cycle info

        elif state == LifecycleState.REJECTED:
            guidance["next_step"] = "Lifecycle was rejected - can archive"
            guidance["waiting_for"] = None
            guidance["available_actions"] = ["archive"]

        elif state == LifecycleState.ARCHIVED:
            guidance["next_step"] = "Lifecycle is archived (terminal state)"
            guidance["waiting_for"] = None
            guidance["available_actions"] = []

        return guidance

    # -------------------------------------------------------------------------
    # State Transitions
    # -------------------------------------------------------------------------

    async def transition(
        self,
        lifecycle: LifecycleInstance,
        trigger: TransitionTrigger,
        triggered_by: str,
        user_role: UserRole,
        reason: str = "",
        metadata: Dict[str, Any] = None,
    ) -> Tuple[bool, str, Optional[LifecycleState]]:
        """
        Perform a state transition.

        Returns (success, message, new_state)
        """
        async with self._lock:
            # Validate transition
            can_do, message, target_state = self.can_transition(
                lifecycle.state, trigger, user_role
            )

            if not can_do:
                # Log rejected transition attempt
                await self._log_audit(
                    lifecycle_id=lifecycle.lifecycle_id,
                    event="transition_rejected",
                    from_state=lifecycle.state.value,
                    to_state=None,
                    trigger=trigger.value,
                    triggered_by=triggered_by,
                    role=user_role.value,
                    reason=message,
                    metadata=metadata or {},
                )
                return False, message, None

            # Perform transition
            old_state = lifecycle.state
            lifecycle.state = target_state
            lifecycle.updated_at = datetime.utcnow()
            lifecycle.transition_count += 1
            lifecycle.last_transition_at = datetime.utcnow()

            # Phase 15.2: Handle cycle tracking for continuous change loops
            if (old_state == LifecycleState.DEPLOYED and
                trigger == TransitionTrigger.CONTINUOUS_CHANGE_REQUEST):
                # Starting a new change cycle
                # Save current cycle to history before incrementing
                current_cycle_record = {
                    "cycle_number": lifecycle.cycle_number,
                    "completed_at": datetime.utcnow().isoformat(),
                    "change_summary": lifecycle.current_change_summary,
                    "deployment_job_id": lifecycle.current_claude_job_id,
                }
                lifecycle.cycle_history.append(current_cycle_record)

                # Increment cycle and store previous deployment reference
                lifecycle.previous_deployment_id = lifecycle.current_claude_job_id
                lifecycle.cycle_number += 1
                lifecycle.current_change_summary = reason  # Store the change request as summary

                logger.info(
                    f"Lifecycle {lifecycle.lifecycle_id}: Starting cycle {lifecycle.cycle_number} "
                    f"(previous deployment: {lifecycle.previous_deployment_id})"
                )

            # Phase 15.2: Track when a cycle completes deployment
            if (target_state == LifecycleState.DEPLOYED and
                old_state == LifecycleState.READY_FOR_PRODUCTION):
                # Cycle completed successfully
                logger.info(
                    f"Lifecycle {lifecycle.lifecycle_id}: Cycle {lifecycle.cycle_number} deployed "
                    f"(job: {lifecycle.current_claude_job_id})"
                )

            # Check if terminal state
            if target_state in LifecycleState.terminal_states():
                lifecycle.completed_at = datetime.utcnow()

            # Create transition record
            record = TransitionRecord(
                record_id=str(uuid.uuid4()),
                lifecycle_id=lifecycle.lifecycle_id,
                from_state=old_state,
                to_state=target_state,
                trigger=trigger,
                triggered_by=triggered_by,
                triggered_at=datetime.utcnow(),
                role=user_role,
                reason=reason,
                metadata=metadata or {},
            )

            # Log to audit trail (enhanced for Phase 15.2)
            audit_metadata = metadata or {}
            audit_metadata["cycle_number"] = lifecycle.cycle_number
            if lifecycle.previous_deployment_id:
                audit_metadata["previous_deployment_id"] = lifecycle.previous_deployment_id

            await self._log_audit(
                lifecycle_id=lifecycle.lifecycle_id,
                event="transition_completed",
                from_state=old_state.value,
                to_state=target_state.value,
                trigger=trigger.value,
                triggered_by=triggered_by,
                role=user_role.value,
                reason=reason,
                metadata=audit_metadata,
            )

            logger.info(
                f"Lifecycle {lifecycle.lifecycle_id}: {old_state.value} -> {target_state.value} "
                f"(trigger: {trigger.value}, by: {triggered_by}, cycle: {lifecycle.cycle_number})"
            )

            return True, f"Transitioned to {target_state.value}", target_state

    # -------------------------------------------------------------------------
    # Audit Logging
    # -------------------------------------------------------------------------

    async def _log_audit(
        self,
        lifecycle_id: str,
        event: str,
        from_state: Optional[str],
        to_state: Optional[str],
        trigger: str,
        triggered_by: str,
        role: str,
        reason: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Log an entry to the immutable audit trail.
        """
        try:
            self._audit_log.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "lifecycle_id": lifecycle_id,
                "event": event,
                "from_state": from_state,
                "to_state": to_state,
                "trigger": trigger,
                "triggered_by": triggered_by,
                "role": role,
                "reason": reason,
                "metadata": metadata,
            }

            with open(self._audit_log, "a") as f:
                f.write(json.dumps(entry) + "\n")

        except IOError as e:
            logger.warning(f"Failed to write audit log: {e}")


# -----------------------------------------------------------------------------
# Lifecycle Manager
# -----------------------------------------------------------------------------

class LifecycleManager:
    """
    Manager for lifecycle instances.

    Handles:
    - Creating new lifecycles
    - Persisting and loading state
    - Multi-aspect isolation
    - Claude CLI coordination
    """

    def __init__(self, state_dir: Path = LIFECYCLE_STATE_DIR):
        self._state_dir = state_dir
        self._state_file = state_dir / "lifecycles.json"
        self._state_machine = LifecycleStateMachine(state_dir)
        self._lock = asyncio.Lock()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure state directories exist."""
        try:
            self._state_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.debug(f"Could not create state directory: {e}")

    # -------------------------------------------------------------------------
    # Lifecycle Creation
    # -------------------------------------------------------------------------

    async def create_lifecycle(
        self,
        project_name: str,
        mode: LifecycleMode,
        aspect: ProjectAspect,
        created_by: str,
        description: str = "",
        change_reference: Optional[ChangeReference] = None,
    ) -> Tuple[bool, str, Optional[LifecycleInstance]]:
        """
        Create a new lifecycle instance.

        CHANGE_MODE requires change_reference with project_id, aspect, change_type.
        """
        # Validate CHANGE_MODE requirements
        if mode == LifecycleMode.CHANGE_MODE:
            if not change_reference:
                return False, "CHANGE_MODE requires change_reference", None
            if not change_reference.project_id:
                return False, "change_reference.project_id is required", None

        lifecycle = LifecycleInstance(
            lifecycle_id=str(uuid.uuid4()),
            project_name=project_name,
            mode=mode,
            aspect=aspect,
            state=LifecycleState.CREATED,
            created_at=datetime.utcnow(),
            created_by=created_by,
            change_reference=change_reference,
            description=description,
        )

        # Persist
        await self._save_lifecycle(lifecycle)

        # Log creation
        await self._state_machine._log_audit(
            lifecycle_id=lifecycle.lifecycle_id,
            event="lifecycle_created",
            from_state=None,
            to_state=LifecycleState.CREATED.value,
            trigger="creation",
            triggered_by=created_by,
            role="creator",
            reason=f"Created {mode.value} lifecycle for {project_name}/{aspect.value}",
            metadata={"description": description},
        )

        logger.info(f"Created lifecycle {lifecycle.lifecycle_id} for {project_name}/{aspect.value}")
        return True, f"Lifecycle created: {lifecycle.lifecycle_id}", lifecycle

    # -------------------------------------------------------------------------
    # Lifecycle Transitions
    # -------------------------------------------------------------------------

    async def transition_lifecycle(
        self,
        lifecycle_id: str,
        trigger: TransitionTrigger,
        triggered_by: str,
        user_role: UserRole,
        reason: str = "",
        metadata: Dict[str, Any] = None,
    ) -> Tuple[bool, str, Optional[LifecycleState]]:
        """
        Transition a lifecycle to a new state.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found", None

        success, message, new_state = await self._state_machine.transition(
            lifecycle=lifecycle,
            trigger=trigger,
            triggered_by=triggered_by,
            user_role=user_role,
            reason=reason,
            metadata=metadata,
        )

        if success:
            await self._save_lifecycle(lifecycle)

        return success, message, new_state

    # -------------------------------------------------------------------------
    # Multi-Aspect Enforcement
    # -------------------------------------------------------------------------

    async def get_lifecycles_for_aspect(
        self,
        project_name: str,
        aspect: ProjectAspect,
    ) -> List[LifecycleInstance]:
        """
        Get all lifecycles for a specific project aspect.

        Ensures aspect isolation - only returns lifecycles for the specified aspect.
        """
        all_lifecycles = await self._load_all_lifecycles()
        return [
            lc for lc in all_lifecycles
            if lc.project_name == project_name and lc.aspect == aspect
        ]

    async def get_active_lifecycle_for_aspect(
        self,
        project_name: str,
        aspect: ProjectAspect,
    ) -> Optional[LifecycleInstance]:
        """
        Get the active (non-terminal) lifecycle for an aspect.

        Returns None if no active lifecycle exists.
        """
        lifecycles = await self.get_lifecycles_for_aspect(project_name, aspect)
        for lc in lifecycles:
            if lc.state not in LifecycleState.terminal_states():
                return lc
        return None

    async def check_aspect_isolation(
        self,
        lifecycle_id: str,
        target_aspect: ProjectAspect,
    ) -> Tuple[bool, str]:
        """
        Verify that a lifecycle operation doesn't violate aspect isolation.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found"

        if lifecycle.aspect != target_aspect:
            return False, f"Lifecycle {lifecycle_id} belongs to aspect {lifecycle.aspect.value}, not {target_aspect.value}"

        return True, "Aspect isolation verified"

    # -------------------------------------------------------------------------
    # Claude CLI Integration
    # -------------------------------------------------------------------------

    async def associate_claude_job(
        self,
        lifecycle_id: str,
        claude_job_id: str,
    ) -> Tuple[bool, str]:
        """
        Associate a Claude CLI job with a lifecycle.

        Claude CLI operates ONLY inside its job workspace.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found"

        lifecycle.claude_job_ids.append(claude_job_id)
        lifecycle.current_claude_job_id = claude_job_id
        lifecycle.updated_at = datetime.utcnow()

        await self._save_lifecycle(lifecycle)

        logger.info(f"Associated Claude job {claude_job_id} with lifecycle {lifecycle_id}")
        return True, f"Claude job {claude_job_id} associated"

    async def record_test_result(
        self,
        lifecycle_id: str,
        test_run_id: str,
        passed: bool,
    ) -> Tuple[bool, str]:
        """
        Record test results for a lifecycle.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found"

        lifecycle.test_run_ids.append(test_run_id)
        lifecycle.last_test_passed = passed
        lifecycle.updated_at = datetime.utcnow()

        await self._save_lifecycle(lifecycle)

        logger.info(f"Recorded test result for lifecycle {lifecycle_id}: {'passed' if passed else 'failed'}")
        return True, f"Test result recorded: {'passed' if passed else 'failed'}"

    async def add_feedback(
        self,
        lifecycle_id: str,
        feedback_by: str,
        feedback_type: str,
        feedback_text: str,
    ) -> Tuple[bool, str]:
        """
        Add feedback to a lifecycle.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found"

        feedback_entry = {
            "feedback_id": str(uuid.uuid4()),
            "feedback_by": feedback_by,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
            "submitted_at": datetime.utcnow().isoformat(),
        }

        lifecycle.feedback_history.append(feedback_entry)
        lifecycle.updated_at = datetime.utcnow()

        await self._save_lifecycle(lifecycle)

        logger.info(f"Added feedback to lifecycle {lifecycle_id}: {feedback_type}")
        return True, "Feedback recorded"

    # -------------------------------------------------------------------------
    # Phase 15.2: Continuous Change Cycles
    # -------------------------------------------------------------------------

    async def request_continuous_change(
        self,
        lifecycle_id: str,
        change_type: ChangeType,
        change_summary: str,
        requested_by: str,
        user_role: UserRole,
    ) -> Tuple[bool, str, Optional[int]]:
        """
        Phase 15.2: Request a continuous change on a deployed lifecycle.

        This triggers the DEPLOYED -> AWAITING_FEEDBACK transition and starts
        a new cycle.

        Returns (success, message, new_cycle_number)
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found", None

        if lifecycle.state != LifecycleState.DEPLOYED:
            return False, f"Cannot request change: lifecycle is in state '{lifecycle.state.value}', must be 'deployed'", None

        # Build metadata for the change request
        metadata = {
            "change_type": change_type.value,
            "change_summary": change_summary,
            "previous_cycle": lifecycle.cycle_number,
        }

        # Perform the transition
        success, message, new_state = await self._state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by=requested_by,
            user_role=user_role,
            reason=change_summary,
            metadata=metadata,
        )

        if success:
            # Update change reference for tracking
            if lifecycle.change_reference:
                lifecycle.change_reference.change_type = change_type
                lifecycle.change_reference.description = change_summary
            await self._save_lifecycle(lifecycle)

            # Log the change request
            await self._state_machine._log_audit(
                lifecycle_id=lifecycle_id,
                event="continuous_change_requested",
                from_state=LifecycleState.DEPLOYED.value,
                to_state=LifecycleState.AWAITING_FEEDBACK.value,
                trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST.value,
                triggered_by=requested_by,
                role=user_role.value,
                reason=change_summary,
                metadata={
                    "change_type": change_type.value,
                    "cycle_number": lifecycle.cycle_number,
                },
            )

            return True, f"Change request accepted. Starting cycle {lifecycle.cycle_number}", lifecycle.cycle_number

        return False, message, None

    async def get_cycle_history(
        self,
        lifecycle_id: str,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """
        Phase 15.2: Get the complete cycle history for a lifecycle.

        Returns (success, message, cycle_history)
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found", []

        # Build history including current cycle
        history = list(lifecycle.cycle_history)

        # Add current cycle as in-progress
        current_cycle = {
            "cycle_number": lifecycle.cycle_number,
            "started_at": lifecycle.last_transition_at.isoformat() if lifecycle.last_transition_at else lifecycle.created_at.isoformat(),
            "completed_at": None,
            "change_summary": lifecycle.current_change_summary,
            "state": lifecycle.state.value,
            "deployment_job_id": lifecycle.current_claude_job_id,
            "is_current": True,
        }
        history.append(current_cycle)

        return True, f"Found {len(history)} cycles", history

    async def get_change_lineage(
        self,
        lifecycle_id: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Phase 15.2: Get the lineage of changes for a lifecycle.

        Returns the chain of deployments and their relationships.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found", {}

        lineage = {
            "lifecycle_id": lifecycle_id,
            "project_name": lifecycle.project_name,
            "aspect": lifecycle.aspect.value,
            "total_cycles": lifecycle.cycle_number,
            "total_deployments": len([c for c in lifecycle.cycle_history if c.get("deployment_job_id")]),
            "current_state": lifecycle.state.value,
            "deployments": [],
        }

        # Build deployment chain
        for cycle in lifecycle.cycle_history:
            if cycle.get("deployment_job_id"):
                lineage["deployments"].append({
                    "cycle": cycle["cycle_number"],
                    "job_id": cycle["deployment_job_id"],
                    "completed_at": cycle.get("completed_at"),
                    "change_summary": cycle.get("change_summary", ""),
                })

        # Add current deployment if deployed
        if lifecycle.state == LifecycleState.DEPLOYED and lifecycle.current_claude_job_id:
            lineage["deployments"].append({
                "cycle": lifecycle.cycle_number,
                "job_id": lifecycle.current_claude_job_id,
                "completed_at": lifecycle.updated_at.isoformat() if lifecycle.updated_at else None,
                "change_summary": lifecycle.current_change_summary,
                "is_current": True,
            })

        return True, "Lineage retrieved", lineage

    async def generate_deployment_summary(
        self,
        lifecycle_id: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Phase 15.2: Generate a "What Changed" deployment summary.

        Returns a summary of what changed across cycles.
        """
        lifecycle = await self.get_lifecycle(lifecycle_id)
        if not lifecycle:
            return False, f"Lifecycle {lifecycle_id} not found", {}

        summary = {
            "lifecycle_id": lifecycle_id,
            "project_name": lifecycle.project_name,
            "aspect": lifecycle.aspect.value,
            "mode": lifecycle.mode.value,
            "current_cycle": lifecycle.cycle_number,
            "current_state": lifecycle.state.value,
            "created_at": lifecycle.created_at.isoformat(),
            "last_updated": lifecycle.updated_at.isoformat() if lifecycle.updated_at else None,
            "total_transitions": lifecycle.transition_count,
            "total_feedback_items": len(lifecycle.feedback_history),
            "total_claude_jobs": len(lifecycle.claude_job_ids),
            "total_test_runs": len(lifecycle.test_run_ids),
            "changes": [],
        }

        # Build change summary from cycle history
        for cycle in lifecycle.cycle_history:
            if cycle.get("change_summary"):
                summary["changes"].append({
                    "cycle": cycle["cycle_number"],
                    "summary": cycle["change_summary"],
                    "completed_at": cycle.get("completed_at"),
                })

        # Add current cycle change if any
        if lifecycle.current_change_summary:
            summary["changes"].append({
                "cycle": lifecycle.cycle_number,
                "summary": lifecycle.current_change_summary,
                "in_progress": True,
            })

        # Latest feedback summary
        if lifecycle.feedback_history:
            latest_feedback = lifecycle.feedback_history[-1]
            summary["latest_feedback"] = {
                "type": latest_feedback.get("feedback_type"),
                "text": latest_feedback.get("feedback_text"),
                "submitted_at": latest_feedback.get("submitted_at"),
            }

        return True, "Summary generated", summary

    async def get_changes_for_project(
        self,
        project_name: str,
        aspect: Optional[ProjectAspect] = None,
    ) -> List[Dict[str, Any]]:
        """
        Phase 15.2: Get all changes (CHANGE_MODE lifecycles) for a project.

        Optionally filtered by aspect.
        """
        lifecycles = await self._load_all_lifecycles()

        # Filter to CHANGE_MODE only
        changes = [
            lc for lc in lifecycles
            if lc.project_name == project_name and lc.mode == LifecycleMode.CHANGE_MODE
        ]

        # Filter by aspect if specified
        if aspect:
            changes = [lc for lc in changes if lc.aspect == aspect]

        # Sort by created_at descending
        changes.sort(key=lambda lc: lc.created_at, reverse=True)

        # Build response
        return [
            {
                "lifecycle_id": lc.lifecycle_id,
                "aspect": lc.aspect.value,
                "state": lc.state.value,
                "change_type": lc.change_reference.change_type.value if lc.change_reference else None,
                "description": lc.change_reference.description if lc.change_reference else lc.description,
                "cycle_number": lc.cycle_number,
                "created_at": lc.created_at.isoformat(),
                "updated_at": lc.updated_at.isoformat() if lc.updated_at else None,
            }
            for lc in changes
        ]

    async def get_aspect_history(
        self,
        project_name: str,
        aspect: ProjectAspect,
    ) -> Dict[str, Any]:
        """
        Phase 15.2: Get the complete history for a project aspect.

        Includes all lifecycles (project and change mode) for the aspect.
        """
        lifecycles = await self.get_lifecycles_for_aspect(project_name, aspect)

        # Sort by created_at
        lifecycles.sort(key=lambda lc: lc.created_at)

        history = {
            "project_name": project_name,
            "aspect": aspect.value,
            "total_lifecycles": len(lifecycles),
            "lifecycles": [],
        }

        total_cycles = 0
        total_deployments = 0

        for lc in lifecycles:
            total_cycles += lc.cycle_number
            if lc.state == LifecycleState.DEPLOYED:
                total_deployments += 1
            total_deployments += len([c for c in lc.cycle_history if c.get("deployment_job_id")])

            history["lifecycles"].append({
                "lifecycle_id": lc.lifecycle_id,
                "mode": lc.mode.value,
                "state": lc.state.value,
                "cycles": lc.cycle_number,
                "created_at": lc.created_at.isoformat(),
                "description": lc.description or (lc.change_reference.description if lc.change_reference else ""),
            })

        history["total_cycles"] = total_cycles
        history["total_deployments"] = total_deployments

        return history

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    async def _save_lifecycle(self, lifecycle: LifecycleInstance) -> None:
        """Save a lifecycle to persistent storage."""
        async with self._lock:
            state = await self._load_state()
            state["lifecycles"][lifecycle.lifecycle_id] = lifecycle.to_dict()
            state["last_updated"] = datetime.utcnow().isoformat()
            await self._save_state(state)

    async def get_lifecycle(self, lifecycle_id: str) -> Optional[LifecycleInstance]:
        """Load a specific lifecycle from storage."""
        state = await self._load_state()
        lc_data = state.get("lifecycles", {}).get(lifecycle_id)
        if lc_data:
            return LifecycleInstance.from_dict(lc_data)
        return None

    async def _load_all_lifecycles(self) -> List[LifecycleInstance]:
        """Load all lifecycles from storage."""
        state = await self._load_state()
        lifecycles = []
        for lc_data in state.get("lifecycles", {}).values():
            try:
                lifecycles.append(LifecycleInstance.from_dict(lc_data))
            except Exception as e:
                logger.warning(f"Failed to deserialize lifecycle: {e}")
        return lifecycles

    async def list_lifecycles(
        self,
        project_name: Optional[str] = None,
        state_filter: Optional[LifecycleState] = None,
        mode_filter: Optional[LifecycleMode] = None,
        limit: int = 50,
    ) -> List[LifecycleInstance]:
        """
        List lifecycles with optional filtering.
        """
        lifecycles = await self._load_all_lifecycles()

        # Apply filters
        if project_name:
            lifecycles = [lc for lc in lifecycles if lc.project_name == project_name]
        if state_filter:
            lifecycles = [lc for lc in lifecycles if lc.state == state_filter]
        if mode_filter:
            lifecycles = [lc for lc in lifecycles if lc.mode == mode_filter]

        # Sort by created_at descending
        lifecycles.sort(key=lambda lc: lc.created_at, reverse=True)
        return lifecycles[:limit]

    async def _load_state(self) -> Dict[str, Any]:
        """Load state from file."""
        if not self._state_file.exists():
            return {"lifecycles": {}, "created_at": datetime.utcnow().isoformat()}
        try:
            return json.loads(self._state_file.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load state file: {e}")
            return {"lifecycles": {}, "created_at": datetime.utcnow().isoformat()}

    async def _save_state(self, state: Dict[str, Any]) -> None:
        """Save state to file atomically."""
        temp_file = self._state_file.with_suffix(".tmp")
        try:
            temp_file.write_text(json.dumps(state, indent=2, default=str))
            temp_file.replace(self._state_file)
        except IOError as e:
            logger.error(f"Failed to save state file: {e}")
            if temp_file.exists():
                temp_file.unlink()

    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    async def recover_state(self) -> Dict[str, Any]:
        """
        Recover state after restart.

        Returns summary of recovered state.
        """
        lifecycles = await self._load_all_lifecycles()

        summary = {
            "total": len(lifecycles),
            "by_state": {},
            "by_mode": {},
            "active": [],
        }

        for lc in lifecycles:
            # Count by state
            state_val = lc.state.value
            summary["by_state"][state_val] = summary["by_state"].get(state_val, 0) + 1

            # Count by mode
            mode_val = lc.mode.value
            summary["by_mode"][mode_val] = summary["by_mode"].get(mode_val, 0) + 1

            # Track active lifecycles
            if lc.state not in LifecycleState.terminal_states():
                summary["active"].append({
                    "id": lc.lifecycle_id,
                    "project": lc.project_name,
                    "aspect": lc.aspect.value,
                    "state": lc.state.value,
                })

        logger.info(f"Recovered {summary['total']} lifecycles ({len(summary['active'])} active)")
        return summary

    # -------------------------------------------------------------------------
    # Status and Guidance
    # -------------------------------------------------------------------------

    def get_guidance(self, lifecycle: LifecycleInstance) -> Dict[str, Any]:
        """Get guidance for what happens next."""
        return self._state_machine.get_next_guidance(lifecycle)


# -----------------------------------------------------------------------------
# Global Manager Instance
# -----------------------------------------------------------------------------

_manager_instance: Optional[LifecycleManager] = None


def get_lifecycle_manager(state_dir: Path = LIFECYCLE_STATE_DIR) -> LifecycleManager:
    """Get or create the lifecycle manager singleton."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LifecycleManager(state_dir)
    return _manager_instance


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------

async def create_project_lifecycle(
    project_name: str,
    aspect: ProjectAspect,
    created_by: str,
    description: str = "",
) -> Tuple[bool, str, Optional[LifecycleInstance]]:
    """
    Create a new PROJECT_MODE lifecycle.
    """
    manager = get_lifecycle_manager()
    return await manager.create_lifecycle(
        project_name=project_name,
        mode=LifecycleMode.PROJECT_MODE,
        aspect=aspect,
        created_by=created_by,
        description=description,
    )


async def create_change_lifecycle(
    project_name: str,
    project_id: str,
    aspect: ProjectAspect,
    change_type: ChangeType,
    created_by: str,
    description: str = "",
) -> Tuple[bool, str, Optional[LifecycleInstance]]:
    """
    Create a new CHANGE_MODE lifecycle.
    """
    change_ref = ChangeReference(
        project_id=project_id,
        aspect=aspect,
        change_type=change_type,
        description=description,
    )

    manager = get_lifecycle_manager()
    return await manager.create_lifecycle(
        project_name=project_name,
        mode=LifecycleMode.CHANGE_MODE,
        aspect=aspect,
        created_by=created_by,
        description=description,
        change_reference=change_ref,
    )


async def get_lifecycle(lifecycle_id: str) -> Optional[LifecycleInstance]:
    """Get a lifecycle by ID."""
    manager = get_lifecycle_manager()
    return await manager.get_lifecycle(lifecycle_id)


async def list_lifecycles(
    project_name: Optional[str] = None,
    state: Optional[str] = None,
    mode: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """
    List lifecycles with optional filtering.
    """
    manager = get_lifecycle_manager()

    state_filter = LifecycleState(state) if state else None
    mode_filter = LifecycleMode(mode) if mode else None

    lifecycles = await manager.list_lifecycles(
        project_name=project_name,
        state_filter=state_filter,
        mode_filter=mode_filter,
        limit=limit,
    )

    return [lc.to_dict() for lc in lifecycles]


async def transition_lifecycle(
    lifecycle_id: str,
    trigger: str,
    triggered_by: str,
    role: str,
    reason: str = "",
    metadata: Dict[str, Any] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    Transition a lifecycle state.
    """
    manager = get_lifecycle_manager()

    success, message, new_state = await manager.transition_lifecycle(
        lifecycle_id=lifecycle_id,
        trigger=TransitionTrigger(trigger),
        triggered_by=triggered_by,
        user_role=UserRole(role),
        reason=reason,
        metadata=metadata,
    )

    return success, message, new_state.value if new_state else None


async def get_lifecycle_guidance(lifecycle_id: str) -> Optional[Dict[str, Any]]:
    """
    Get guidance for what happens next in a lifecycle.
    """
    manager = get_lifecycle_manager()
    lifecycle = await manager.get_lifecycle(lifecycle_id)
    if not lifecycle:
        return None
    return manager.get_guidance(lifecycle)


# -----------------------------------------------------------------------------
# Phase 15.2: Continuous Change Cycle Public API
# -----------------------------------------------------------------------------

async def request_continuous_change(
    lifecycle_id: str,
    change_type: str,
    change_summary: str,
    requested_by: str,
    role: str,
) -> Tuple[bool, str, Optional[int]]:
    """
    Phase 15.2: Request a continuous change on a deployed lifecycle.

    Args:
        lifecycle_id: ID of the deployed lifecycle
        change_type: Type of change (bug, feature, improvement, refactor, security)
        change_summary: Description of the requested change
        requested_by: User requesting the change
        role: User's role (owner, admin, developer, tester)

    Returns:
        (success, message, new_cycle_number)
    """
    manager = get_lifecycle_manager()
    return await manager.request_continuous_change(
        lifecycle_id=lifecycle_id,
        change_type=ChangeType(change_type),
        change_summary=change_summary,
        requested_by=requested_by,
        user_role=UserRole(role),
    )


async def get_cycle_history(lifecycle_id: str) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """
    Phase 15.2: Get the complete cycle history for a lifecycle.
    """
    manager = get_lifecycle_manager()
    return await manager.get_cycle_history(lifecycle_id)


async def get_change_lineage(lifecycle_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Phase 15.2: Get the lineage of changes for a lifecycle.
    """
    manager = get_lifecycle_manager()
    return await manager.get_change_lineage(lifecycle_id)


async def get_deployment_summary(lifecycle_id: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Phase 15.2: Generate a "What Changed" deployment summary.
    """
    manager = get_lifecycle_manager()
    return await manager.generate_deployment_summary(lifecycle_id)


async def get_project_changes(
    project_name: str,
    aspect: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Phase 15.2: Get all changes for a project.
    """
    manager = get_lifecycle_manager()
    aspect_filter = ProjectAspect(aspect) if aspect else None
    return await manager.get_changes_for_project(project_name, aspect_filter)


async def get_aspect_history(
    project_name: str,
    aspect: str,
) -> Dict[str, Any]:
    """
    Phase 15.2: Get the complete history for a project aspect.
    """
    manager = get_lifecycle_manager()
    return await manager.get_aspect_history(project_name, ProjectAspect(aspect))


logger.info("Lifecycle V2 module loaded successfully (Phase 15.2)")
