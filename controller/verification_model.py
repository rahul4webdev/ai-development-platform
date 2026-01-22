"""
Phase 18D: Post-Execution Verification - Data Models

Frozen dataclasses and LOCKED enums for post-execution verification.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- NO EXECUTION: Never executes, retries, or triggers anything
- NO MUTATION: Never modifies external state
- NO ROLLBACK: Never undoes any changes
- NO RECOMMENDATIONS: Never suggests fixes or actions
- NO NOTIFICATIONS: Never sends alerts or messages
- 100% DETERMINISTIC: Same inputs = same output
- FAIL CLOSED: If data missing → verification UNKNOWN

This module defines the data models for verification.
It is a DATA MODEL, not an ACTOR.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


# -----------------------------------------------------------------------------
# Verification Status Enum (LOCKED - EXACTLY 3 VALUES)
# -----------------------------------------------------------------------------
class VerificationStatus(str, Enum):
    """
    Verification status.

    This enum is LOCKED - do not add values without explicit approval.
    EXACTLY 3 values, no more, no less.
    """
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Violation Severity Enum (LOCKED - EXACTLY 4 VALUES)
# -----------------------------------------------------------------------------
class ViolationSeverity(str, Enum):
    """
    Severity of an invariant violation.

    This enum is LOCKED - do not add values without explicit approval.
    EXACTLY 4 values, no more, no less.
    """
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# -----------------------------------------------------------------------------
# Violation Type Enum (LOCKED - EXACTLY 6 DOMAINS)
# -----------------------------------------------------------------------------
class ViolationType(str, Enum):
    """
    Types of invariant violations.

    This enum is LOCKED to EXACTLY 6 verification domains.
    No other domains allowed.
    """
    # Domain 1: Scope Compliance
    SCOPE_VIOLATION = "scope_violation"  # Touched files/modules outside approved scope

    # Domain 2: Action Compliance
    ACTION_VIOLATION = "action_violation"  # Executed action type not approved

    # Domain 3: Boundary Compliance
    BOUNDARY_VIOLATION = "boundary_violation"  # Production deploy or external access

    # Domain 4: Intent Compliance
    INTENT_VIOLATION = "intent_violation"  # Intent drift introduced

    # Domain 5: Invariant Compliance
    INVARIANT_VIOLATION = "invariant_violation"  # Audit or approval chain broken

    # Domain 6: Outcome Consistency
    OUTCOME_VIOLATION = "outcome_violation"  # SUCCESS/FAILURE misaligns with logs


# -----------------------------------------------------------------------------
# Unknown Reason Enum (LOCKED)
# -----------------------------------------------------------------------------
class UnknownReason(str, Enum):
    """
    Reasons for UNKNOWN verification status.

    These are the ONLY valid reasons for UNKNOWN status.
    """
    MISSING_EXECUTION_RESULT = "missing_execution_result"
    MISSING_EXECUTION_INTENT = "missing_execution_intent"
    MISSING_EXECUTION_AUDIT = "missing_execution_audit"
    MISSING_LOGS = "missing_logs"
    MISSING_LIFECYCLE_SNAPSHOT = "missing_lifecycle_snapshot"
    MISSING_INTENT_BASELINE = "missing_intent_baseline"
    LOGS_UNREADABLE = "logs_unreadable"
    DATA_CORRUPTION = "data_corruption"


# -----------------------------------------------------------------------------
# Input Data Classes (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionResultSnapshot:
    """
    Immutable snapshot of execution result for verification.

    Read-only copy of execution result from Phase 18C.
    """
    execution_id: str
    intent_id: str
    status: str  # ExecutionStatus value
    block_reason: Optional[str]
    failure_reason: Optional[str]
    timestamp: str
    gate_decision_allowed: Optional[bool]
    execution_output: Optional[str]
    rollback_performed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "status": self.status,
            "block_reason": self.block_reason,
            "failure_reason": self.failure_reason,
            "timestamp": self.timestamp,
            "gate_decision_allowed": self.gate_decision_allowed,
            "execution_output": self.execution_output,
            "rollback_performed": self.rollback_performed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResultSnapshot":
        return cls(
            execution_id=data["execution_id"],
            intent_id=data["intent_id"],
            status=data["status"],
            block_reason=data.get("block_reason"),
            failure_reason=data.get("failure_reason"),
            timestamp=data["timestamp"],
            gate_decision_allowed=data.get("gate_decision_allowed"),
            execution_output=data.get("execution_output"),
            rollback_performed=data.get("rollback_performed", False),
        )


@dataclass(frozen=True)
class ExecutionIntentSnapshot:
    """
    Immutable snapshot of execution intent for verification.

    Read-only copy of execution intent from Phase 18C.
    """
    intent_id: str
    project_id: str
    project_name: str
    action_type: str
    action_description: str
    requester_id: str
    requester_role: str
    target_workspace: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent_id": self.intent_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "requester_id": self.requester_id,
            "requester_role": self.requester_role,
            "target_workspace": self.target_workspace,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionIntentSnapshot":
        return cls(
            intent_id=data["intent_id"],
            project_id=data["project_id"],
            project_name=data["project_name"],
            action_type=data["action_type"],
            action_description=data["action_description"],
            requester_id=data["requester_id"],
            requester_role=data["requester_role"],
            target_workspace=data["target_workspace"],
            created_at=data["created_at"],
        )


@dataclass(frozen=True)
class ExecutionAuditSnapshot:
    """
    Immutable snapshot of execution audit record for verification.

    Read-only copy from Phase 18C audit.
    """
    audit_id: str
    execution_id: str
    intent_id: str
    input_hash: str
    status: str
    project_id: Optional[str]
    action_type: Optional[str]
    requester_id: Optional[str]
    eligibility_decision: Optional[str]
    approval_status: Optional[str]
    gate_allowed: Optional[bool]
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "input_hash": self.input_hash,
            "status": self.status,
            "project_id": self.project_id,
            "action_type": self.action_type,
            "requester_id": self.requester_id,
            "eligibility_decision": self.eligibility_decision,
            "approval_status": self.approval_status,
            "gate_allowed": self.gate_allowed,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionAuditSnapshot":
        return cls(
            audit_id=data["audit_id"],
            execution_id=data["execution_id"],
            intent_id=data["intent_id"],
            input_hash=data["input_hash"],
            status=data["status"],
            project_id=data.get("project_id"),
            action_type=data.get("action_type"),
            requester_id=data.get("requester_id"),
            eligibility_decision=data.get("eligibility_decision"),
            approval_status=data.get("approval_status"),
            gate_allowed=data.get("gate_allowed"),
            timestamp=data["timestamp"],
        )


@dataclass(frozen=True)
class LifecycleSnapshot:
    """
    Immutable snapshot of lifecycle state post-execution.

    Read-only copy for verification.
    """
    project_id: str
    lifecycle_id: str
    state: str
    is_active: bool
    last_transition: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "lifecycle_id": self.lifecycle_id,
            "state": self.state,
            "is_active": self.is_active,
            "last_transition": self.last_transition,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifecycleSnapshot":
        return cls(
            project_id=data["project_id"],
            lifecycle_id=data["lifecycle_id"],
            state=data["state"],
            is_active=data["is_active"],
            last_transition=data.get("last_transition"),
        )


@dataclass(frozen=True)
class IntentBaselineSnapshot:
    """
    Immutable snapshot of intent baseline for verification.

    Read-only copy for drift checking.
    """
    project_id: str
    baseline_version: str
    baseline_valid: bool
    approved_scope: Tuple[str, ...]  # Approved files/modules
    approved_actions: Tuple[str, ...]  # Approved action types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "baseline_version": self.baseline_version,
            "baseline_valid": self.baseline_valid,
            "approved_scope": list(self.approved_scope),
            "approved_actions": list(self.approved_actions),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentBaselineSnapshot":
        return cls(
            project_id=data["project_id"],
            baseline_version=data["baseline_version"],
            baseline_valid=data["baseline_valid"],
            approved_scope=tuple(data.get("approved_scope", [])),
            approved_actions=tuple(data.get("approved_actions", [])),
        )


@dataclass(frozen=True)
class ExecutionConstraints:
    """
    Constraints that were in effect during execution.

    Read-only copy for verification.
    """
    allowed_actions: Tuple[str, ...]
    forbidden_paths: Tuple[str, ...]  # Paths that must not be touched
    production_deploy_allowed: bool
    external_network_allowed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed_actions": list(self.allowed_actions),
            "forbidden_paths": list(self.forbidden_paths),
            "production_deploy_allowed": self.production_deploy_allowed,
            "external_network_allowed": self.external_network_allowed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionConstraints":
        return cls(
            allowed_actions=tuple(data.get("allowed_actions", [])),
            forbidden_paths=tuple(data.get("forbidden_paths", [])),
            production_deploy_allowed=data.get("production_deploy_allowed", False),
            external_network_allowed=data.get("external_network_allowed", False),
        )


@dataclass(frozen=True)
class ExecutionLogs:
    """
    Immutable snapshot of execution logs for verification.

    Read-only access to log content.
    """
    logs_path: Optional[str]
    logs_content: Optional[str]  # Truncated if too long
    logs_readable: bool
    exit_code: Optional[int]
    files_touched: Tuple[str, ...]  # Files modified during execution

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logs_path": self.logs_path,
            "logs_content": self.logs_content,
            "logs_readable": self.logs_readable,
            "exit_code": self.exit_code,
            "files_touched": list(self.files_touched),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionLogs":
        return cls(
            logs_path=data.get("logs_path"),
            logs_content=data.get("logs_content"),
            logs_readable=data.get("logs_readable", False),
            exit_code=data.get("exit_code"),
            files_touched=tuple(data.get("files_touched", [])),
        )


# -----------------------------------------------------------------------------
# Verification Input (Frozen - All Inputs Combined)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VerificationInput:
    """
    Complete input for post-execution verification.

    All fields are read-only. Missing fields → UNKNOWN status.
    """
    execution_result: Optional[ExecutionResultSnapshot]
    execution_intent: Optional[ExecutionIntentSnapshot]
    execution_audit: Optional[ExecutionAuditSnapshot]
    lifecycle_snapshot: Optional[LifecycleSnapshot]
    intent_baseline: Optional[IntentBaselineSnapshot]
    execution_constraints: Optional[ExecutionConstraints]
    execution_logs: Optional[ExecutionLogs]

    def compute_hash(self) -> str:
        """Compute deterministic hash of all inputs."""
        data = {
            "execution_result": self.execution_result.to_dict() if self.execution_result else None,
            "execution_intent": self.execution_intent.to_dict() if self.execution_intent else None,
            "execution_audit": self.execution_audit.to_dict() if self.execution_audit else None,
            "lifecycle_snapshot": self.lifecycle_snapshot.to_dict() if self.lifecycle_snapshot else None,
            "intent_baseline": self.intent_baseline.to_dict() if self.intent_baseline else None,
            "execution_constraints": self.execution_constraints.to_dict() if self.execution_constraints else None,
            "execution_logs": self.execution_logs.to_dict() if self.execution_logs else None,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Invariant Violation (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class InvariantViolation:
    """
    A single invariant violation detected during verification.

    Immutable record of a constraint violation.
    """
    violation_id: str
    violation_type: str  # ViolationType value
    severity: str  # ViolationSeverity value
    description: str
    evidence_path: Optional[str]  # Path to evidence (file, log line, etc.)
    evidence_snippet: Optional[str]  # Brief excerpt of evidence
    detected_at: str  # ISO format

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type,
            "severity": self.severity,
            "description": self.description,
            "evidence_path": self.evidence_path,
            "evidence_snippet": self.evidence_snippet,
            "detected_at": self.detected_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvariantViolation":
        return cls(
            violation_id=data["violation_id"],
            violation_type=data["violation_type"],
            severity=data["severity"],
            description=data["description"],
            evidence_path=data.get("evidence_path"),
            evidence_snippet=data.get("evidence_snippet"),
            detected_at=data["detected_at"],
        )


# -----------------------------------------------------------------------------
# Verification Result (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionVerificationResult:
    """
    Result of post-execution verification.

    Immutable record of verification outcome.
    """
    verification_id: str
    execution_id: str
    verification_status: str  # VerificationStatus value
    unknown_reason: Optional[str]  # UnknownReason value if UNKNOWN
    violations: Tuple[InvariantViolation, ...]  # Immutable tuple
    input_hash: str
    checked_at: str  # ISO format
    verifier_version: str
    # Summary fields
    violation_count: int
    high_severity_count: int
    domains_checked: Tuple[str, ...]  # Which domains were verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "execution_id": self.execution_id,
            "verification_status": self.verification_status,
            "unknown_reason": self.unknown_reason,
            "violations": [v.to_dict() for v in self.violations],
            "input_hash": self.input_hash,
            "checked_at": self.checked_at,
            "verifier_version": self.verifier_version,
            "violation_count": self.violation_count,
            "high_severity_count": self.high_severity_count,
            "domains_checked": list(self.domains_checked),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionVerificationResult":
        violations = tuple(
            InvariantViolation.from_dict(v)
            for v in data.get("violations", [])
        )
        return cls(
            verification_id=data["verification_id"],
            execution_id=data["execution_id"],
            verification_status=data["verification_status"],
            unknown_reason=data.get("unknown_reason"),
            violations=violations,
            input_hash=data["input_hash"],
            checked_at=data["checked_at"],
            verifier_version=data["verifier_version"],
            violation_count=data["violation_count"],
            high_severity_count=data["high_severity_count"],
            domains_checked=tuple(data.get("domains_checked", [])),
        )


# -----------------------------------------------------------------------------
# Verification Audit Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class VerificationAuditRecord:
    """
    Immutable audit record for verification operations.

    Append-only. No updates. No deletes.
    """
    audit_id: str
    verification_id: str
    execution_id: str
    verification_status: str
    unknown_reason: Optional[str]
    violation_count: int
    high_severity_count: int
    input_hash: str
    checked_at: str
    verifier_version: str
    project_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "verification_id": self.verification_id,
            "execution_id": self.execution_id,
            "verification_status": self.verification_status,
            "unknown_reason": self.unknown_reason,
            "violation_count": self.violation_count,
            "high_severity_count": self.high_severity_count,
            "input_hash": self.input_hash,
            "checked_at": self.checked_at,
            "verifier_version": self.verifier_version,
            "project_id": self.project_id,
        }


# -----------------------------------------------------------------------------
# Symmetry Guarantee Table (for documentation/validation)
# -----------------------------------------------------------------------------
"""
SYMMETRY GUARANTEE TABLE
========================

This table documents the verification domains and their constraints.
Any change to this table requires explicit approval.

+---------------------------+-------------------+----------------------------------+
| Verification Domain       | Violation Type    | What It Checks                   |
+---------------------------+-------------------+----------------------------------+
| Scope Compliance          | SCOPE_VIOLATION   | Only approved files touched      |
| Action Compliance         | ACTION_VIOLATION  | Only approved action type        |
| Boundary Compliance       | BOUNDARY_VIOLATION| No prod deploy, no ext network   |
| Intent Compliance         | INTENT_VIOLATION  | No intent drift introduced       |
| Invariant Compliance      | INVARIANT_VIOLATION| Audit/approval chain intact     |
| Outcome Consistency       | OUTCOME_VIOLATION | SUCCESS/FAILURE matches logs     |
+---------------------------+-------------------+----------------------------------+

VERIFICATION STATUS RULES:
- All domains pass, no violations → PASSED
- Any violation detected → FAILED
- Missing required data → UNKNOWN (fail closed)

INVARIANTS:
- Same inputs ALWAYS produce same output (deterministic)
- Missing data → UNKNOWN (never guess, never assume)
- Violations are RECORDED, not ACTED UPON
- NO execution, NO retries, NO rollback, NO mutations
- NO recommendations, NO alerts, NO notifications
"""
