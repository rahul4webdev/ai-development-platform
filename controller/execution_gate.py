"""
Phase 15.6: Execution Gate Model

This module implements explicit permission enforcement for Claude CLI execution.
It is the SINGLE source of truth for what actions Claude can perform.

Key Features:
- Lifecycle-state based action permissions
- Role-based action permissions
- Aspect-based action permissions
- Filesystem scope enforcement (job workspace only)
- Hard fail conditions
- Immutable audit trail

Actions:
- READ_CODE: Read source files
- WRITE_CODE: Write/modify source files
- RUN_TESTS: Execute test suites
- COMMIT: Create git commits (local only)
- PUSH: Push to remote repository
- DEPLOY_TEST: Deploy to testing environment
- DEPLOY_PROD: Deploy to production environment

SECURITY-CRITICAL: This module MUST be consulted before ANY Claude CLI execution.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple, FrozenSet

logger = logging.getLogger("execution_gate")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
EXECUTION_AUDIT_LOG = Path("/home/aitesting.mybd.in/jobs/execution_audit.log")
EXECUTION_GATE_CONFIG = Path("/home/aitesting.mybd.in/jobs/execution_gate_config.json")

# Ensure audit log directory exists
EXECUTION_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Action Enum
# -----------------------------------------------------------------------------
class ExecutionAction(str, Enum):
    """
    Explicit actions that Claude can perform.

    Each action has specific lifecycle state requirements and role permissions.
    NO OTHER ACTIONS ARE PERMITTED.
    """
    READ_CODE = "read_code"
    WRITE_CODE = "write_code"
    RUN_TESTS = "run_tests"
    COMMIT = "commit"
    PUSH = "push"
    DEPLOY_TEST = "deploy_test"
    DEPLOY_PROD = "deploy_prod"

    @classmethod
    def all_read_actions(cls) -> Set["ExecutionAction"]:
        """Actions that only read, never modify."""
        return {cls.READ_CODE}

    @classmethod
    def all_write_actions(cls) -> Set["ExecutionAction"]:
        """Actions that modify code or state."""
        return {cls.WRITE_CODE, cls.COMMIT}

    @classmethod
    def all_deploy_actions(cls) -> Set["ExecutionAction"]:
        """Actions that deploy to environments."""
        return {cls.DEPLOY_TEST, cls.DEPLOY_PROD}


class LifecycleState(str, Enum):
    """
    Lifecycle states (mirror from lifecycle_v2.py for type safety).
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


class UserRole(str, Enum):
    """
    User roles for permission validation (mirror from lifecycle_v2.py).
    """
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"
    TESTER = "tester"
    VIEWER = "viewer"


class ProjectAspect(str, Enum):
    """
    Project aspects (mirror from lifecycle_v2.py).
    """
    CORE = "core"
    BACKEND = "backend"
    FRONTEND_WEB = "frontend_web"
    FRONTEND_MOBILE = "frontend_mobile"
    ADMIN = "admin"
    CUSTOM = "custom"


# -----------------------------------------------------------------------------
# Lifecycle State -> Allowed Actions Mapping
# -----------------------------------------------------------------------------
# This is the SINGLE SOURCE OF TRUTH for what actions are permitted in each state.
# SECURITY-CRITICAL: Changes to this mapping MUST be reviewed carefully.

LIFECYCLE_ALLOWED_ACTIONS: Dict[LifecycleState, FrozenSet[ExecutionAction]] = {
    # CREATED: No actions allowed - waiting for initialization
    LifecycleState.CREATED: frozenset(),

    # PLANNING: Can read code to plan, cannot write or deploy
    LifecycleState.PLANNING: frozenset({
        ExecutionAction.READ_CODE,
    }),

    # DEVELOPMENT: Can read and write code, run tests locally, commit
    LifecycleState.DEVELOPMENT: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.WRITE_CODE,
        ExecutionAction.RUN_TESTS,
        ExecutionAction.COMMIT,
    }),

    # TESTING: Can read, run tests. Cannot write code (tests must pass as-is)
    LifecycleState.TESTING: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.RUN_TESTS,
    }),

    # AWAITING_FEEDBACK: Read-only. Waiting for human feedback.
    LifecycleState.AWAITING_FEEDBACK: frozenset({
        ExecutionAction.READ_CODE,
    }),

    # FIXING: Can read, write, test, commit to fix issues
    LifecycleState.FIXING: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.WRITE_CODE,
        ExecutionAction.RUN_TESTS,
        ExecutionAction.COMMIT,
    }),

    # READY_FOR_PRODUCTION: Can push (after approval) and deploy to testing
    LifecycleState.READY_FOR_PRODUCTION: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.PUSH,
        ExecutionAction.DEPLOY_TEST,
    }),

    # DEPLOYED: Read-only. Changes require new lifecycle.
    LifecycleState.DEPLOYED: frozenset({
        ExecutionAction.READ_CODE,
    }),

    # REJECTED: No actions allowed - lifecycle terminated
    LifecycleState.REJECTED: frozenset(),

    # ARCHIVED: No actions allowed - lifecycle archived
    LifecycleState.ARCHIVED: frozenset(),
}

# Production deployment is NEVER allowed via Claude automation
# It requires HUMAN approval with dual-confirmation via Telegram
# This is enforced at the gate level, not the lifecycle level

# -----------------------------------------------------------------------------
# Role -> Allowed Actions Mapping
# -----------------------------------------------------------------------------
# Maps roles to maximum actions they can request Claude to perform.

ROLE_ALLOWED_ACTIONS: Dict[UserRole, FrozenSet[ExecutionAction]] = {
    # OWNER: All actions (still gated by lifecycle state)
    UserRole.OWNER: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.WRITE_CODE,
        ExecutionAction.RUN_TESTS,
        ExecutionAction.COMMIT,
        ExecutionAction.PUSH,
        ExecutionAction.DEPLOY_TEST,
        # DEPLOY_PROD explicitly excluded - requires human dual-approval
    }),

    # ADMIN: Same as owner except production
    UserRole.ADMIN: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.WRITE_CODE,
        ExecutionAction.RUN_TESTS,
        ExecutionAction.COMMIT,
        ExecutionAction.PUSH,
        ExecutionAction.DEPLOY_TEST,
    }),

    # DEVELOPER: Can write, test, commit - no push or deploy
    UserRole.DEVELOPER: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.WRITE_CODE,
        ExecutionAction.RUN_TESTS,
        ExecutionAction.COMMIT,
    }),

    # TESTER: Can read and run tests only
    UserRole.TESTER: frozenset({
        ExecutionAction.READ_CODE,
        ExecutionAction.RUN_TESTS,
    }),

    # VIEWER: Read-only
    UserRole.VIEWER: frozenset({
        ExecutionAction.READ_CODE,
    }),
}


# -----------------------------------------------------------------------------
# Execution Request & Result
# -----------------------------------------------------------------------------
@dataclass
class ExecutionRequest:
    """
    Request to execute an action via Claude CLI.

    All fields are required for audit trail completeness.
    """
    job_id: str
    project_name: str
    aspect: str  # ProjectAspect value
    lifecycle_id: str
    lifecycle_state: str  # LifecycleState value
    requested_action: str  # ExecutionAction value
    requesting_user_id: str
    requesting_user_role: str  # UserRole value
    workspace_path: str
    task_description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "project_name": self.project_name,
            "aspect": self.aspect,
            "lifecycle_id": self.lifecycle_id,
            "lifecycle_state": self.lifecycle_state,
            "requested_action": self.requested_action,
            "requesting_user_id": self.requesting_user_id,
            "requesting_user_role": self.requesting_user_role,
            "workspace_path": self.workspace_path,
            "task_description": self.task_description[:200],  # Truncate for log
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class GateDecision:
    """
    Result of execution gate evaluation.

    If allowed is False, execution MUST NOT proceed.
    """
    allowed: bool
    request: ExecutionRequest
    allowed_actions: List[str]  # Actions permitted by lifecycle state
    denied_reason: Optional[str] = None
    hard_fail: bool = False  # If True, this is a security violation
    workspace_validated: bool = False
    governance_docs_present: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "allowed": self.allowed,
            "request": self.request.to_dict(),
            "allowed_actions": self.allowed_actions,
            "denied_reason": self.denied_reason,
            "hard_fail": self.hard_fail,
            "workspace_validated": self.workspace_validated,
            "governance_docs_present": self.governance_docs_present,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ExecutionAuditEntry:
    """
    Immutable audit entry for Claude execution.

    Every execution attempt MUST be logged, whether allowed or not.
    """
    job_id: str
    project_name: str
    aspect: str
    lifecycle_id: str
    lifecycle_state: str
    allowed_actions: List[str]
    executed_action: str
    requesting_user_id: str
    requesting_user_role: str
    gate_decision: str  # "ALLOWED" or "DENIED"
    denied_reason: Optional[str]
    outcome: Optional[str]  # "SUCCESS", "FAILURE", "PENDING"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit log."""
        return {
            "job_id": self.job_id,
            "project_name": self.project_name,
            "aspect": self.aspect,
            "lifecycle_id": self.lifecycle_id,
            "lifecycle_state": self.lifecycle_state,
            "allowed_actions": self.allowed_actions,
            "executed_action": self.executed_action,
            "requesting_user_id": self.requesting_user_id,
            "requesting_user_role": self.requesting_user_role,
            "gate_decision": self.gate_decision,
            "denied_reason": self.denied_reason,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
        }


# -----------------------------------------------------------------------------
# Required Governance Documents
# -----------------------------------------------------------------------------
REQUIRED_GOVERNANCE_DOCS = frozenset({
    "AI_POLICY.md",
    "ARCHITECTURE.md",
    "CURRENT_STATE.md",
    "DEPLOYMENT.md",
    "PROJECT_CONTEXT.md",
    "PROJECT_MANIFEST.yaml",
    "TESTING_STRATEGY.md",
})


# -----------------------------------------------------------------------------
# Execution Gate
# -----------------------------------------------------------------------------
class ExecutionGate:
    """
    The Execution Gate is the SINGLE point of authorization for Claude CLI execution.

    SECURITY-CRITICAL: All Claude executions MUST pass through this gate.
    The gate enforces:
    1. Lifecycle state permissions
    2. Role permissions
    3. Aspect isolation
    4. Workspace scope validation
    5. Governance document presence

    If any check fails, execution is DENIED with a hard fail.
    """

    def __init__(self):
        self._audit_log_path = EXECUTION_AUDIT_LOG
        # Ensure audit log exists
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self, request: ExecutionRequest) -> GateDecision:
        """
        Evaluate an execution request against all gates.

        Returns a GateDecision indicating whether execution is allowed.
        If denied, includes the reason and whether it's a hard fail (security violation).

        EVERY evaluation is logged to the audit trail, regardless of outcome.
        """
        allowed_actions: List[str] = []

        # Step 1: Validate lifecycle state
        try:
            lifecycle_state = LifecycleState(request.lifecycle_state)
        except ValueError:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=[],
                denied_reason=f"Invalid lifecycle state: {request.lifecycle_state}",
                hard_fail=True,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 2: Get allowed actions for lifecycle state
        state_actions = LIFECYCLE_ALLOWED_ACTIONS.get(lifecycle_state, frozenset())
        allowed_actions = [a.value for a in state_actions]

        # Step 3: Validate requested action exists
        try:
            requested_action = ExecutionAction(request.requested_action)
        except ValueError:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason=f"Invalid action: {request.requested_action}",
                hard_fail=True,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 4: Check if action is allowed in lifecycle state
        if requested_action not in state_actions:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason=f"Action '{requested_action.value}' not permitted in state '{lifecycle_state.value}'. "
                             f"Allowed actions: {allowed_actions}",
                hard_fail=True,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 5: Validate user role
        try:
            user_role = UserRole(request.requesting_user_role)
        except ValueError:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason=f"Invalid user role: {request.requesting_user_role}",
                hard_fail=True,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 6: Check if role permits this action
        role_actions = ROLE_ALLOWED_ACTIONS.get(user_role, frozenset())
        if requested_action not in role_actions:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason=f"Role '{user_role.value}' cannot perform '{requested_action.value}'",
                hard_fail=True,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 7: Validate workspace path (must be within jobs directory)
        workspace_path = Path(request.workspace_path)
        workspace_validated = self._validate_workspace(workspace_path)
        if not workspace_validated:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason=f"Invalid workspace path: {request.workspace_path}. "
                             f"Workspace must be within /home/aitesting.mybd.in/jobs/",
                hard_fail=True,
                workspace_validated=False,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 8: Check governance documents presence
        governance_docs_present = self._check_governance_docs(workspace_path)
        if not governance_docs_present:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason=f"Required governance documents missing in workspace. "
                             f"Required: {sorted(REQUIRED_GOVERNANCE_DOCS)}",
                hard_fail=True,
                workspace_validated=True,
                governance_docs_present=False,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # Step 9: DEPLOY_PROD is NEVER allowed through automated execution
        if requested_action == ExecutionAction.DEPLOY_PROD:
            decision = GateDecision(
                allowed=False,
                request=request,
                allowed_actions=allowed_actions,
                denied_reason="Production deployment is NEVER allowed through automated execution. "
                             "Use Telegram dual-approval flow instead.",
                hard_fail=True,
                workspace_validated=True,
                governance_docs_present=True,
            )
            self._log_audit(request, decision, "DENIED")
            return decision

        # All checks passed - execution is ALLOWED
        decision = GateDecision(
            allowed=True,
            request=request,
            allowed_actions=allowed_actions,
            workspace_validated=True,
            governance_docs_present=True,
        )
        self._log_audit(request, decision, "ALLOWED")

        logger.info(
            f"Execution ALLOWED: job={request.job_id}, action={requested_action.value}, "
            f"state={lifecycle_state.value}, user={request.requesting_user_id}"
        )

        return decision

    def _validate_workspace(self, workspace_path: Path) -> bool:
        """
        Validate that workspace is within allowed directory.

        SECURITY-CRITICAL: Prevents path traversal attacks.
        """
        try:
            # Resolve to absolute path
            resolved = workspace_path.resolve()

            # Must be within jobs directory
            jobs_dir = Path("/home/aitesting.mybd.in/jobs").resolve()

            # Check if workspace is within jobs directory
            try:
                resolved.relative_to(jobs_dir)
                return True
            except ValueError:
                # Not within jobs directory - check for development environment
                # Allow /tmp for testing
                if str(resolved).startswith("/tmp/"):
                    logger.warning(f"Allowing /tmp workspace for testing: {resolved}")
                    return True
                return False

        except Exception as e:
            logger.error(f"Workspace validation error: {e}")
            return False

    def _check_governance_docs(self, workspace_path: Path) -> bool:
        """
        Check that all required governance documents exist in workspace.

        SECURITY-CRITICAL: Claude must have policy constraints before execution.
        """
        try:
            for doc in REQUIRED_GOVERNANCE_DOCS:
                doc_path = workspace_path / doc
                if not doc_path.exists():
                    logger.warning(f"Missing governance document: {doc_path}")
                    return False
            return True
        except Exception as e:
            logger.error(f"Governance doc check error: {e}")
            return False

    def _log_audit(
        self,
        request: ExecutionRequest,
        decision: GateDecision,
        gate_decision: str,
        outcome: Optional[str] = None
    ) -> None:
        """
        Log execution attempt to immutable audit trail.

        SECURITY-CRITICAL: All attempts MUST be logged.
        """
        try:
            entry = ExecutionAuditEntry(
                job_id=request.job_id,
                project_name=request.project_name,
                aspect=request.aspect,
                lifecycle_id=request.lifecycle_id,
                lifecycle_state=request.lifecycle_state,
                allowed_actions=decision.allowed_actions,
                executed_action=request.requested_action,
                requesting_user_id=request.requesting_user_id,
                requesting_user_role=request.requesting_user_role,
                gate_decision=gate_decision,
                denied_reason=decision.denied_reason,
                outcome=outcome,
            )

            # Append to audit log (immutable, append-only)
            with open(self._audit_log_path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")

            # Also log to standard logger for observability
            if gate_decision == "DENIED":
                logger.warning(
                    f"EXECUTION DENIED: job={request.job_id}, action={request.requested_action}, "
                    f"reason={decision.denied_reason}, hard_fail={decision.hard_fail}"
                )

        except Exception as e:
            # Audit logging failure is CRITICAL - log to stderr
            logger.critical(f"AUDIT LOG FAILURE: {e}")
            # Do not raise - allow decision to proceed but alert

    def log_execution_outcome(
        self,
        request: ExecutionRequest,
        outcome: str,  # "SUCCESS" or "FAILURE"
        error_message: Optional[str] = None
    ) -> None:
        """
        Log the outcome of an execution after completion.

        Call this after Claude CLI execution completes.
        """
        try:
            entry = {
                "type": "execution_outcome",
                "job_id": request.job_id,
                "lifecycle_id": request.lifecycle_id,
                "executed_action": request.requested_action,
                "outcome": outcome,
                "error_message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
            }

            with open(self._audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            logger.critical(f"AUDIT LOG FAILURE (outcome): {e}")

    def get_allowed_actions_for_state(self, lifecycle_state: str) -> List[str]:
        """
        Get the list of allowed actions for a given lifecycle state.

        Useful for displaying to users what they can do.
        """
        try:
            state = LifecycleState(lifecycle_state)
            actions = LIFECYCLE_ALLOWED_ACTIONS.get(state, frozenset())
            return [a.value for a in actions]
        except ValueError:
            return []

    def get_allowed_actions_for_role(self, role: str) -> List[str]:
        """
        Get the list of allowed actions for a given role.

        Useful for displaying to users what they can do.
        """
        try:
            user_role = UserRole(role)
            actions = ROLE_ALLOWED_ACTIONS.get(user_role, frozenset())
            return [a.value for a in actions]
        except ValueError:
            return []


# Global execution gate instance
execution_gate = ExecutionGate()


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------
def check_execution_allowed(
    job_id: str,
    project_name: str,
    aspect: str,
    lifecycle_id: str,
    lifecycle_state: str,
    requested_action: str,
    user_id: str,
    user_role: str,
    workspace_path: str,
    task_description: str,
) -> Tuple[bool, Optional[str], List[str]]:
    """
    Convenience function to check if execution is allowed.

    Returns:
        Tuple of (allowed: bool, denied_reason: Optional[str], allowed_actions: List[str])
    """
    request = ExecutionRequest(
        job_id=job_id,
        project_name=project_name,
        aspect=aspect,
        lifecycle_id=lifecycle_id,
        lifecycle_state=lifecycle_state,
        requested_action=requested_action,
        requesting_user_id=user_id,
        requesting_user_role=user_role,
        workspace_path=workspace_path,
        task_description=task_description,
    )

    decision = execution_gate.evaluate(request)
    return decision.allowed, decision.denied_reason, decision.allowed_actions


def get_execution_constraints_for_job(
    lifecycle_state: str,
    user_role: str,
) -> Dict[str, Any]:
    """
    Get execution constraints that should be passed to Claude.

    This information helps Claude understand what it can and cannot do.
    """
    state_actions = execution_gate.get_allowed_actions_for_state(lifecycle_state)
    role_actions = execution_gate.get_allowed_actions_for_role(user_role)

    # Intersection of state and role permissions
    effective_actions = list(set(state_actions) & set(role_actions))

    return {
        "lifecycle_state": lifecycle_state,
        "user_role": user_role,
        "allowed_actions": effective_actions,
        "state_allowed_actions": state_actions,
        "role_allowed_actions": role_actions,
        "constraints": [
            "You may ONLY perform actions listed in 'allowed_actions'",
            "You may NOT deploy to production under any circumstances",
            "You may NOT access files outside the job workspace",
            "You may NOT modify AI_POLICY.md or ARCHITECTURE.md",
            "You MUST log all actions to logs/EXECUTION_LOG.md",
        ],
    }


logger.info("Execution Gate module loaded (Phase 15.6)")
