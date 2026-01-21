"""
Phase 18B: Human Approval Orchestration

DECISION-ONLY orchestrator that answers: "What is the approval status?"

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- DECISION-ONLY: Returns approval status, NOTHING ELSE
- NO EXECUTION: Never executes, schedules, triggers, or mutates
- NO AUTOMATION: Does not trigger any automated actions
- NO NOTIFICATIONS: Does not send alerts, emails, or messages
- NO PLANNING: Does not plan what to do next
- NO SIDE EFFECTS: Only side effect is append-only audit writes
- DETERMINISTIC: Same inputs ALWAYS produce same output (no ML, no heuristics)
- HUMAN-GOVERNED: Humans approve, system tracks

This orchestrator sits AFTER eligibility (Phase 18A) and governs human approval.
It is a STATUS TRACKER, not an ACTOR.

If ANY input is missing → APPROVAL_DENIED
If audit write fails → APPROVAL_DENIED
If eligibility is FORBIDDEN → APPROVAL_DENIED (immediate)
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Import from Phase 18A
from .automation_eligibility import (
    EligibilityDecision,
    EligibilityResult,
    RecommendationInput,
    LifecycleStateInput,
    ExecutionGateInput,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ORCHESTRATOR_VERSION = "18B.1.0"

# Audit storage (append-only)
AUDIT_DIR = Path(os.getenv("AUDIT_DIR", "audit"))
APPROVAL_AUDIT_FILE = AUDIT_DIR / "approval_decisions.jsonl"

# Approval expiration (24 hours default)
DEFAULT_APPROVAL_EXPIRY_HOURS = 24


# -----------------------------------------------------------------------------
# Approval Status Enum (LOCKED - EXACTLY 3 VALUES)
# -----------------------------------------------------------------------------
class ApprovalStatus(str, Enum):
    """
    Human approval status.

    This enum is LOCKED - do not add values without explicit approval.
    EXACTLY 3 values, no more, no less.
    """
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_PENDING = "approval_pending"


# -----------------------------------------------------------------------------
# Denial Reason Enum (LOCKED)
# -----------------------------------------------------------------------------
class DenialReason(str, Enum):
    """
    Reasons for approval denial.

    These are the ONLY valid denial reasons.
    """
    # Input validation failures
    MISSING_ELIGIBILITY = "missing_eligibility"
    MISSING_RECOMMENDATION = "missing_recommendation"
    MISSING_LIFECYCLE_STATE = "missing_lifecycle_state"
    MISSING_EXECUTION_GATE = "missing_execution_gate"
    MISSING_REQUESTER = "missing_requester"

    # Eligibility failures
    ELIGIBILITY_FORBIDDEN = "eligibility_forbidden"

    # Approval process failures
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_REVOKED = "approval_revoked"
    APPROVER_SAME_AS_REQUESTER = "approver_same_as_requester"
    APPROVER_UNAUTHORIZED = "approver_unauthorized"
    INSUFFICIENT_APPROVERS = "insufficient_approvers"

    # Gate failures
    EXECUTION_GATE_DENIED = "execution_gate_denied"

    # Audit failures
    AUDIT_WRITE_FAILED = "audit_write_failed"


# -----------------------------------------------------------------------------
# Pending Reason Enum (LOCKED)
# -----------------------------------------------------------------------------
class PendingReason(str, Enum):
    """
    Reasons for approval being pending.

    These are the ONLY valid pending reasons.
    """
    AWAITING_APPROVAL = "awaiting_approval"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    AWAITING_DUAL_APPROVAL = "awaiting_dual_approval"


# -----------------------------------------------------------------------------
# Approval Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class ApprovalType(str, Enum):
    """
    Types of approval required.

    Determined by recommendation's approval_required field.
    """
    NONE_REQUIRED = "none_required"
    CONFIRMATION_REQUIRED = "confirmation_required"
    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"
    DUAL_APPROVAL_REQUIRED = "dual_approval_required"


# -----------------------------------------------------------------------------
# Input Data Classes (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ApprovalRequesterInput:
    """
    Input identifying the approval requester.

    Immutable snapshot of requester data.
    """
    requester_id: str
    requester_role: str
    request_timestamp: str  # ISO format
    request_reason: Optional[str]


@dataclass(frozen=True)
class ApproverInput:
    """
    Input identifying an approver.

    Immutable snapshot of approver data.
    """
    approver_id: str
    approver_role: str
    approval_timestamp: str  # ISO format
    approval_reason: Optional[str]


@dataclass(frozen=True)
class ApprovalStateInput:
    """
    Current state of approvals for a request.

    Immutable snapshot of approval state.
    """
    approval_request_id: str
    approvers: Tuple[ApproverInput, ...]  # Immutable tuple of approvers
    required_approver_count: int
    approval_type: str  # ApprovalType value
    created_at: str  # ISO format
    expires_at: str  # ISO format

    def __post_init__(self):
        if self.required_approver_count < 0:
            raise ValueError(f"required_approver_count cannot be negative: {self.required_approver_count}")
        if not isinstance(self.approvers, tuple):
            raise ValueError("approvers must be a tuple for immutability")


# -----------------------------------------------------------------------------
# Orchestration Input (Frozen - All Inputs Combined)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class OrchestrationInput:
    """
    Complete input for approval orchestration.

    ALL fields are MANDATORY. If any is None → APPROVAL_DENIED.
    """
    eligibility_result: Optional[EligibilityResult]
    recommendation: Optional[RecommendationInput]
    lifecycle_state: Optional[LifecycleStateInput]
    execution_gate: Optional[ExecutionGateInput]
    requester: Optional[ApprovalRequesterInput]
    approval_state: Optional[ApprovalStateInput]
    current_timestamp: str  # ISO format for expiry checking

    def compute_hash(self) -> str:
        """Compute deterministic hash of all inputs."""
        data = {
            "eligibility_result": self._serialize_eligibility(),
            "recommendation": self._serialize_recommendation(),
            "lifecycle_state": self._serialize_lifecycle(),
            "execution_gate": self._serialize_gate(),
            "requester": self._serialize_requester(),
            "approval_state": self._serialize_approval_state(),
            "current_timestamp": self.current_timestamp,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _serialize_eligibility(self) -> Optional[Dict[str, Any]]:
        if self.eligibility_result is None:
            return None
        return self.eligibility_result.to_dict()

    def _serialize_recommendation(self) -> Optional[Dict[str, Any]]:
        if self.recommendation is None:
            return None
        return {
            "recommendation_id": self.recommendation.recommendation_id,
            "recommendation_type": self.recommendation.recommendation_type,
            "severity": self.recommendation.severity,
            "approval_required": self.recommendation.approval_required,
            "status": self.recommendation.status,
            "project_id": self.recommendation.project_id,
            "confidence": self.recommendation.confidence,
        }

    def _serialize_lifecycle(self) -> Optional[Dict[str, Any]]:
        if self.lifecycle_state is None:
            return None
        return {
            "state": self.lifecycle_state.state,
            "project_id": self.lifecycle_state.project_id,
            "is_active": self.lifecycle_state.is_active,
        }

    def _serialize_gate(self) -> Optional[Dict[str, Any]]:
        if self.execution_gate is None:
            return None
        return {
            "gate_allows_action": self.execution_gate.gate_allows_action,
            "required_role": self.execution_gate.required_role,
            "gate_denial_reason": self.execution_gate.gate_denial_reason,
        }

    def _serialize_requester(self) -> Optional[Dict[str, Any]]:
        if self.requester is None:
            return None
        return {
            "requester_id": self.requester.requester_id,
            "requester_role": self.requester.requester_role,
            "request_timestamp": self.requester.request_timestamp,
            "request_reason": self.requester.request_reason,
        }

    def _serialize_approval_state(self) -> Optional[Dict[str, Any]]:
        if self.approval_state is None:
            return None
        return {
            "approval_request_id": self.approval_state.approval_request_id,
            "approvers": [
                {
                    "approver_id": a.approver_id,
                    "approver_role": a.approver_role,
                    "approval_timestamp": a.approval_timestamp,
                    "approval_reason": a.approval_reason,
                }
                for a in self.approval_state.approvers
            ],
            "required_approver_count": self.approval_state.required_approver_count,
            "approval_type": self.approval_state.approval_type,
            "created_at": self.approval_state.created_at,
            "expires_at": self.approval_state.expires_at,
        }


# -----------------------------------------------------------------------------
# Orchestration Result (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class OrchestrationResult:
    """
    Result of approval orchestration.

    Immutable record of the status determination.
    """
    status: str  # ApprovalStatus value
    reason: Optional[str]  # DenialReason or PendingReason value
    input_hash: str
    timestamp: str
    orchestrator_version: str
    approval_request_id: Optional[str]
    approver_count: int
    required_approver_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "input_hash": self.input_hash,
            "timestamp": self.timestamp,
            "orchestrator_version": self.orchestrator_version,
            "approval_request_id": self.approval_request_id,
            "approver_count": self.approver_count,
            "required_approver_count": self.required_approver_count,
        }


# -----------------------------------------------------------------------------
# Audit Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ApprovalAuditRecord:
    """
    Immutable audit record for approval decisions.

    Append-only. No updates. No deletes.
    """
    audit_id: str
    input_hash: str
    status: str
    reason: Optional[str]
    timestamp: str
    orchestrator_version: str
    approval_request_id: Optional[str]
    requester_id: Optional[str]
    project_id: Optional[str]
    recommendation_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "input_hash": self.input_hash,
            "status": self.status,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "orchestrator_version": self.orchestrator_version,
            "approval_request_id": self.approval_request_id,
            "requester_id": self.requester_id,
            "project_id": self.project_id,
            "recommendation_id": self.recommendation_id,
        }


# -----------------------------------------------------------------------------
# Human Approval Orchestrator (DECISION-ONLY)
# -----------------------------------------------------------------------------
class HumanApprovalOrchestrator:
    """
    Phase 18B: Human Approval Orchestrator.

    DECISION-ONLY orchestrator that determines approval status.

    CRITICAL CONSTRAINTS:
    - Returns approval status, NOTHING ELSE
    - NO execution, scheduling, triggering, or mutation
    - NO notifications or alerts
    - DETERMINISTIC: Same inputs = same output
    - AUDIT REQUIRED: Every evaluation emits immutable audit record

    If ANY input is missing → APPROVAL_DENIED
    If audit write fails → APPROVAL_DENIED
    If eligibility is FORBIDDEN → APPROVAL_DENIED
    """

    def __init__(self, audit_file: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            audit_file: Path to audit file (optional, for testing)
        """
        self._audit_file = audit_file or APPROVAL_AUDIT_FILE
        self._version = ORCHESTRATOR_VERSION

    def evaluate(self, orchestration_input: OrchestrationInput) -> OrchestrationResult:
        """
        Evaluate approval status.

        This is the ONLY public method. It:
        1. Validates all inputs
        2. Checks immediate denial rules
        3. Checks approval grant rules
        4. Determines pending state if neither
        5. Writes audit record
        6. Returns result

        If audit write fails → APPROVAL_DENIED

        Args:
            orchestration_input: Complete input for evaluation

        Returns:
            OrchestrationResult with status and reason
        """
        timestamp = datetime.utcnow().isoformat()
        input_hash = orchestration_input.compute_hash()

        # Phase 1: Check for missing inputs (ALL are mandatory)
        missing_reason = self._check_missing_inputs(orchestration_input)
        if missing_reason:
            return self._create_denied_result(
                reason=missing_reason,
                input_hash=input_hash,
                timestamp=timestamp,
                orchestration_input=orchestration_input,
            )

        # Phase 2: Check immediate denial rules
        denial_reason = self._check_immediate_denial(orchestration_input)
        if denial_reason:
            return self._create_denied_result(
                reason=denial_reason,
                input_hash=input_hash,
                timestamp=timestamp,
                orchestration_input=orchestration_input,
            )

        # Phase 3: Check approval grant rules
        grant_result = self._check_approval_grant(orchestration_input, input_hash, timestamp)
        if grant_result:
            result = grant_result
        else:
            # Phase 4: Determine pending state
            result = self._determine_pending_state(orchestration_input, input_hash, timestamp)

        # Phase 5: Write audit record (MANDATORY)
        audit_success = self._write_audit_record(result, orchestration_input)

        # If audit fails, override to DENIED
        if not audit_success:
            return self._create_denied_result(
                reason=DenialReason.AUDIT_WRITE_FAILED.value,
                input_hash=input_hash,
                timestamp=timestamp,
                orchestration_input=orchestration_input,
            )

        return result

    def _check_missing_inputs(self, orchestration_input: OrchestrationInput) -> Optional[str]:
        """Check for missing mandatory inputs."""
        if orchestration_input.eligibility_result is None:
            return DenialReason.MISSING_ELIGIBILITY.value

        if orchestration_input.recommendation is None:
            return DenialReason.MISSING_RECOMMENDATION.value

        if orchestration_input.lifecycle_state is None:
            return DenialReason.MISSING_LIFECYCLE_STATE.value

        if orchestration_input.execution_gate is None:
            return DenialReason.MISSING_EXECUTION_GATE.value

        if orchestration_input.requester is None:
            return DenialReason.MISSING_REQUESTER.value

        # approval_state can be None for fresh requests (will be PENDING)

        return None

    def _check_immediate_denial(self, orchestration_input: OrchestrationInput) -> Optional[str]:
        """
        Check for immediate denial conditions.

        If ANY condition matches → APPROVAL_DENIED
        """
        # Rule 1: Eligibility is FORBIDDEN
        eligibility = orchestration_input.eligibility_result
        if eligibility and eligibility.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value:
            return DenialReason.ELIGIBILITY_FORBIDDEN.value

        # Rule 2: Execution gate denies
        gate = orchestration_input.execution_gate
        if gate and not gate.gate_allows_action:
            return DenialReason.EXECUTION_GATE_DENIED.value

        # Rule 3: Approval expired
        approval_state = orchestration_input.approval_state
        if approval_state:
            if self._is_expired(approval_state.expires_at, orchestration_input.current_timestamp):
                return DenialReason.APPROVAL_EXPIRED.value

        # Rule 4: Self-approval attempt (approver same as requester)
        if approval_state and orchestration_input.requester:
            for approver in approval_state.approvers:
                if approver.approver_id == orchestration_input.requester.requester_id:
                    return DenialReason.APPROVER_SAME_AS_REQUESTER.value

        return None

    def _check_approval_grant(
        self,
        orchestration_input: OrchestrationInput,
        input_hash: str,
        timestamp: str,
    ) -> Optional[OrchestrationResult]:
        """
        Check if approval should be granted.

        Returns OrchestrationResult if granted, None otherwise.
        """
        recommendation = orchestration_input.recommendation
        approval_state = orchestration_input.approval_state

        # Case 1: No approval required (info only)
        if recommendation.approval_required == "none_required":
            return self._create_granted_result(
                input_hash=input_hash,
                timestamp=timestamp,
                orchestration_input=orchestration_input,
            )

        # Case 2: Check if required approvals are met
        if approval_state is None:
            # No approvals yet - cannot grant
            return None

        actual_approver_count = len(approval_state.approvers)
        required_count = approval_state.required_approver_count

        # For confirmation_required: 1 approval needed
        if recommendation.approval_required == "confirmation_required":
            if actual_approver_count >= 1:
                return self._create_granted_result(
                    input_hash=input_hash,
                    timestamp=timestamp,
                    orchestration_input=orchestration_input,
                )

        # For explicit_approval_required: meet required count
        if recommendation.approval_required == "explicit_approval_required":
            if actual_approver_count >= required_count:
                return self._create_granted_result(
                    input_hash=input_hash,
                    timestamp=timestamp,
                    orchestration_input=orchestration_input,
                )

        return None

    def _determine_pending_state(
        self,
        orchestration_input: OrchestrationInput,
        input_hash: str,
        timestamp: str,
    ) -> OrchestrationResult:
        """
        Determine the pending reason.

        Called when neither denied nor granted.
        """
        recommendation = orchestration_input.recommendation
        approval_state = orchestration_input.approval_state

        # Determine pending reason based on approval type
        if recommendation.approval_required == "confirmation_required":
            pending_reason = PendingReason.AWAITING_CONFIRMATION.value
        elif recommendation.approval_required == "explicit_approval_required":
            if approval_state and approval_state.required_approver_count > 1:
                pending_reason = PendingReason.AWAITING_DUAL_APPROVAL.value
            else:
                pending_reason = PendingReason.AWAITING_APPROVAL.value
        else:
            pending_reason = PendingReason.AWAITING_APPROVAL.value

        # Get counts
        approver_count = 0
        required_count = 1
        approval_request_id = None

        if approval_state:
            approver_count = len(approval_state.approvers)
            required_count = approval_state.required_approver_count
            approval_request_id = approval_state.approval_request_id

        return OrchestrationResult(
            status=ApprovalStatus.APPROVAL_PENDING.value,
            reason=pending_reason,
            input_hash=input_hash,
            timestamp=timestamp,
            orchestrator_version=self._version,
            approval_request_id=approval_request_id,
            approver_count=approver_count,
            required_approver_count=required_count,
        )

    def _create_denied_result(
        self,
        reason: str,
        input_hash: str,
        timestamp: str,
        orchestration_input: OrchestrationInput,
    ) -> OrchestrationResult:
        """Create APPROVAL_DENIED result."""
        approval_request_id = None
        approver_count = 0
        required_count = 0

        if orchestration_input.approval_state:
            approval_request_id = orchestration_input.approval_state.approval_request_id
            approver_count = len(orchestration_input.approval_state.approvers)
            required_count = orchestration_input.approval_state.required_approver_count

        return OrchestrationResult(
            status=ApprovalStatus.APPROVAL_DENIED.value,
            reason=reason,
            input_hash=input_hash,
            timestamp=timestamp,
            orchestrator_version=self._version,
            approval_request_id=approval_request_id,
            approver_count=approver_count,
            required_approver_count=required_count,
        )

    def _create_granted_result(
        self,
        input_hash: str,
        timestamp: str,
        orchestration_input: OrchestrationInput,
    ) -> OrchestrationResult:
        """Create APPROVAL_GRANTED result."""
        approval_request_id = None
        approver_count = 0
        required_count = 0

        if orchestration_input.approval_state:
            approval_request_id = orchestration_input.approval_state.approval_request_id
            approver_count = len(orchestration_input.approval_state.approvers)
            required_count = orchestration_input.approval_state.required_approver_count

        return OrchestrationResult(
            status=ApprovalStatus.APPROVAL_GRANTED.value,
            reason=None,
            input_hash=input_hash,
            timestamp=timestamp,
            orchestrator_version=self._version,
            approval_request_id=approval_request_id,
            approver_count=approver_count,
            required_approver_count=required_count,
        )

    def _is_expired(self, expires_at: str, current_timestamp: str) -> bool:
        """Check if approval has expired."""
        try:
            expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            current = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
            return current > expiry
        except (ValueError, TypeError):
            # If we can't parse dates, assume expired (fail closed)
            return True

    def _write_audit_record(
        self,
        result: OrchestrationResult,
        orchestration_input: OrchestrationInput,
    ) -> bool:
        """
        Write immutable audit record.

        Append-only. No updates. No deletes.
        If write fails → return False (triggers APPROVAL_DENIED).
        """
        try:
            # Ensure audit directory exists
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate audit ID
            audit_id = f"appr-{result.timestamp.replace(':', '-').replace('.', '-')}-{result.input_hash[:8]}"

            # Extract IDs if available
            requester_id = None
            project_id = None
            recommendation_id = None

            if orchestration_input.requester:
                requester_id = orchestration_input.requester.requester_id

            if orchestration_input.recommendation:
                recommendation_id = orchestration_input.recommendation.recommendation_id
                project_id = orchestration_input.recommendation.project_id

            if not project_id and orchestration_input.lifecycle_state:
                project_id = orchestration_input.lifecycle_state.project_id

            # Create audit record
            audit_record = ApprovalAuditRecord(
                audit_id=audit_id,
                input_hash=result.input_hash,
                status=result.status,
                reason=result.reason,
                timestamp=result.timestamp,
                orchestrator_version=result.orchestrator_version,
                approval_request_id=result.approval_request_id,
                requester_id=requester_id,
                project_id=project_id,
                recommendation_id=recommendation_id,
            )

            # Append to audit file (append-only)
            with open(self._audit_file, 'a') as f:
                f.write(json.dumps(audit_record.to_dict()) + '\n')
                f.flush()
                os.fsync(f.fileno())

            return True

        except Exception:
            # Any failure in audit write → return False
            return False


# -----------------------------------------------------------------------------
# Module-Level Functions (Read-Only Access)
# -----------------------------------------------------------------------------

# Singleton instance
_orchestrator: Optional[HumanApprovalOrchestrator] = None


def get_approval_orchestrator(audit_file: Optional[Path] = None) -> HumanApprovalOrchestrator:
    """Get the approval orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = HumanApprovalOrchestrator(audit_file=audit_file)
    return _orchestrator


def evaluate_approval(orchestration_input: OrchestrationInput) -> OrchestrationResult:
    """
    Evaluate approval status.

    Convenience function that uses singleton orchestrator.

    Args:
        orchestration_input: Complete input for evaluation

    Returns:
        OrchestrationResult with status and reason
    """
    orchestrator = get_approval_orchestrator()
    return orchestrator.evaluate(orchestration_input)


def create_orchestration_input(
    eligibility_result: Optional[Dict[str, Any]] = None,
    recommendation: Optional[Dict[str, Any]] = None,
    lifecycle_state: Optional[Dict[str, Any]] = None,
    execution_gate: Optional[Dict[str, Any]] = None,
    requester: Optional[Dict[str, Any]] = None,
    approval_state: Optional[Dict[str, Any]] = None,
    current_timestamp: Optional[str] = None,
) -> OrchestrationInput:
    """
    Create OrchestrationInput from dictionary data.

    Convenience function for creating input from API/external data.
    """
    # Convert eligibility result
    elig_result = None
    if eligibility_result:
        elig_result = EligibilityResult(
            decision=eligibility_result.get("decision", ""),
            matched_rules=tuple(eligibility_result.get("matched_rules", [])),
            input_hash=eligibility_result.get("input_hash", ""),
            timestamp=eligibility_result.get("timestamp", ""),
            engine_version=eligibility_result.get("engine_version", ""),
            allowed_actions=tuple(eligibility_result.get("allowed_actions", [])),
        )

    # Convert recommendation
    rec_input = None
    if recommendation:
        rec_input = RecommendationInput(
            recommendation_id=recommendation.get("recommendation_id", ""),
            recommendation_type=recommendation.get("recommendation_type", ""),
            severity=recommendation.get("severity", ""),
            approval_required=recommendation.get("approval_required", ""),
            status=recommendation.get("status", ""),
            project_id=recommendation.get("project_id"),
            confidence=recommendation.get("confidence", 0.0),
        )

    # Convert lifecycle state
    lifecycle_input = None
    if lifecycle_state:
        lifecycle_input = LifecycleStateInput(
            state=lifecycle_state.get("state", ""),
            project_id=lifecycle_state.get("project_id", ""),
            is_active=lifecycle_state.get("is_active", False),
        )

    # Convert execution gate
    gate_input = None
    if execution_gate:
        gate_input = ExecutionGateInput(
            gate_allows_action=execution_gate.get("gate_allows_action", False),
            required_role=execution_gate.get("required_role"),
            gate_denial_reason=execution_gate.get("gate_denial_reason"),
        )

    # Convert requester
    requester_input = None
    if requester:
        requester_input = ApprovalRequesterInput(
            requester_id=requester.get("requester_id", ""),
            requester_role=requester.get("requester_role", ""),
            request_timestamp=requester.get("request_timestamp", ""),
            request_reason=requester.get("request_reason"),
        )

    # Convert approval state
    approval_state_input = None
    if approval_state:
        approvers_data = approval_state.get("approvers", [])
        approvers = tuple(
            ApproverInput(
                approver_id=a.get("approver_id", ""),
                approver_role=a.get("approver_role", ""),
                approval_timestamp=a.get("approval_timestamp", ""),
                approval_reason=a.get("approval_reason"),
            )
            for a in approvers_data
        )
        approval_state_input = ApprovalStateInput(
            approval_request_id=approval_state.get("approval_request_id", ""),
            approvers=approvers,
            required_approver_count=approval_state.get("required_approver_count", 1),
            approval_type=approval_state.get("approval_type", ""),
            created_at=approval_state.get("created_at", ""),
            expires_at=approval_state.get("expires_at", ""),
        )

    # Use current time if not provided
    if current_timestamp is None:
        current_timestamp = datetime.utcnow().isoformat()

    return OrchestrationInput(
        eligibility_result=elig_result,
        recommendation=rec_input,
        lifecycle_state=lifecycle_input,
        execution_gate=gate_input,
        requester=requester_input,
        approval_state=approval_state_input,
        current_timestamp=current_timestamp,
    )


# -----------------------------------------------------------------------------
# Symmetry Guarantee Table (for documentation/validation)
# -----------------------------------------------------------------------------
"""
SYMMETRY GUARANTEE TABLE
========================

This table documents the symmetric relationships between inputs and outputs.
Any change to this table requires explicit approval.

+---------------------------+-------------------+--------------------------------+
| Input Condition           | Output Status     | Output Reason                  |
+---------------------------+-------------------+--------------------------------+
| eligibility_result=None   | APPROVAL_DENIED   | MISSING_ELIGIBILITY            |
| recommendation=None       | APPROVAL_DENIED   | MISSING_RECOMMENDATION         |
| lifecycle_state=None      | APPROVAL_DENIED   | MISSING_LIFECYCLE_STATE        |
| execution_gate=None       | APPROVAL_DENIED   | MISSING_EXECUTION_GATE         |
| requester=None            | APPROVAL_DENIED   | MISSING_REQUESTER              |
+---------------------------+-------------------+--------------------------------+
| eligibility=FORBIDDEN     | APPROVAL_DENIED   | ELIGIBILITY_FORBIDDEN          |
| gate_allows_action=False  | APPROVAL_DENIED   | EXECUTION_GATE_DENIED          |
| expires_at < current      | APPROVAL_DENIED   | APPROVAL_EXPIRED               |
| approver=requester        | APPROVAL_DENIED   | APPROVER_SAME_AS_REQUESTER     |
+---------------------------+-------------------+--------------------------------+
| approval_required=none    | APPROVAL_GRANTED  | None                           |
| approvers >= required     | APPROVAL_GRANTED  | None                           |
+---------------------------+-------------------+--------------------------------+
| confirmation, 0 approvers | APPROVAL_PENDING  | AWAITING_CONFIRMATION          |
| explicit, 0 approvers     | APPROVAL_PENDING  | AWAITING_APPROVAL              |
| dual, < required          | APPROVAL_PENDING  | AWAITING_DUAL_APPROVAL         |
+---------------------------+-------------------+--------------------------------+
| audit write fails         | APPROVAL_DENIED   | AUDIT_WRITE_FAILED             |
+---------------------------+-------------------+--------------------------------+

INVARIANTS:
- Same inputs ALWAYS produce same output (deterministic)
- Missing input → DENIED (never PENDING or GRANTED)
- Eligibility FORBIDDEN → DENIED (never PENDING or GRANTED)
- Audit failure → DENIED (fail closed)
- Self-approval → DENIED (anti-corruption)
- Expiry → DENIED (time-bounded)
"""
