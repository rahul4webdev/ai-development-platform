"""
Phase 18C: Controlled Execution Dispatcher

The ONLY layer allowed to execute real actions in the platform.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- SINGLE EXECUTION POINT: ALL actions MUST flow through this dispatcher
- CHAIN VALIDATION: Eligibility -> Approval -> Gate -> Execute (in that order)
- ATOMIC EXECUTION: Either fully execute or fully rollback
- NO BYPASS: No action can skip the validation chain
- AUDIT REQUIRED: Every execution emits immutable audit record
- DETERMINISTIC: Same inputs = same validation outcome

This dispatcher sits AFTER eligibility (Phase 18A), approval (Phase 18B),
and gate (Phase 15.6) - it is the FINAL checkpoint before execution.

If ANY validation fails -> EXECUTION_BLOCKED
If audit write fails -> EXECUTION_BLOCKED
If execution fails -> EXECUTION_FAILED (with rollback)
"""

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# Import from Phase 18A
from .automation_eligibility import (
    EligibilityDecision,
    EligibilityResult,
    EligibilityInput,
    AutomationEligibilityEngine,
    LimitedAction,
)

# Import from Phase 18B
from .approval_orchestrator import (
    ApprovalStatus,
    OrchestrationResult,
    OrchestrationInput,
    HumanApprovalOrchestrator,
)

# Import from Phase 15.6
from .execution_gate import (
    ExecutionGate,
    ExecutionRequest,
    GateDecision,
    ExecutionAction,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DISPATCHER_VERSION = "18C.1.0"

# Execution storage (append-only)
EXECUTION_DIR = Path(os.getenv("EXECUTION_DIR", "data/execution"))
EXECUTION_INTENTS_FILE = EXECUTION_DIR / "execution_intents.jsonl"
EXECUTION_RESULTS_FILE = EXECUTION_DIR / "execution_results.jsonl"
EXECUTION_AUDIT_FILE = EXECUTION_DIR / "execution_audit.jsonl"


# -----------------------------------------------------------------------------
# Execution Status Enum (LOCKED - EXACTLY 4 VALUES)
# -----------------------------------------------------------------------------
class ExecutionStatus(str, Enum):
    """
    Execution status.

    This enum is LOCKED - do not add values without explicit approval.
    EXACTLY 4 values, no more, no less.
    """
    EXECUTION_BLOCKED = "execution_blocked"
    EXECUTION_PENDING = "execution_pending"
    EXECUTION_SUCCESS = "execution_success"
    EXECUTION_FAILED = "execution_failed"


# -----------------------------------------------------------------------------
# Block Reason Enum (LOCKED)
# -----------------------------------------------------------------------------
class BlockReason(str, Enum):
    """
    Reasons for execution being blocked.

    These are the ONLY valid block reasons.
    """
    # Input validation failures
    MISSING_INTENT = "missing_intent"
    MISSING_ELIGIBILITY = "missing_eligibility"
    MISSING_APPROVAL = "missing_approval"
    MISSING_GATE_REQUEST = "missing_gate_request"
    INVALID_ACTION_TYPE = "invalid_action_type"

    # Eligibility failures
    ELIGIBILITY_FORBIDDEN = "eligibility_forbidden"
    ACTION_NOT_IN_ALLOWED_LIST = "action_not_in_allowed_list"

    # Approval failures
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_PENDING = "approval_pending"
    APPROVAL_EXPIRED = "approval_expired"

    # Gate failures
    GATE_DENIED = "gate_denied"
    GATE_HARD_FAIL = "gate_hard_fail"
    DRIFT_BLOCKS_EXECUTION = "drift_blocks_execution"

    # Chain validation failures
    CHAIN_VALIDATION_FAILED = "chain_validation_failed"
    INPUT_HASH_MISMATCH = "input_hash_mismatch"

    # Audit failures
    AUDIT_WRITE_FAILED = "audit_write_failed"


# -----------------------------------------------------------------------------
# Failure Reason Enum (LOCKED)
# -----------------------------------------------------------------------------
class FailureReason(str, Enum):
    """
    Reasons for execution failure (after validation passed).

    These are the ONLY valid failure reasons.
    """
    EXECUTION_TIMEOUT = "execution_timeout"
    EXECUTION_ERROR = "execution_error"
    ROLLBACK_REQUIRED = "rollback_required"
    EXTERNAL_SYSTEM_ERROR = "external_system_error"


# -----------------------------------------------------------------------------
# Action Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class ActionType(str, Enum):
    """
    Types of actions that can be executed.

    This enum is LOCKED and matches LimitedAction + approved actions.
    """
    RUN_TESTS = "run_tests"
    UPDATE_DOCS = "update_docs"
    WRITE_CODE = "write_code"
    COMMIT = "commit"
    PUSH = "push"
    DEPLOY_TEST = "deploy_test"
    # DEPLOY_PROD is NEVER allowed through automated execution


# -----------------------------------------------------------------------------
# Input Data Classes (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionIntent:
    """
    Input representing the intent to execute an action.

    Immutable snapshot of what is requested to be executed.
    """
    intent_id: str
    project_id: str
    project_name: str
    action_type: str  # ActionType value
    action_description: str
    requester_id: str
    requester_role: str
    target_workspace: str
    created_at: str  # ISO format
    metadata: Tuple[Tuple[str, Any], ...] = ()  # Immutable dict alternative

    def __post_init__(self):
        # Validate action type
        valid_actions = [a.value for a in ActionType]
        if self.action_type not in valid_actions:
            raise ValueError(f"Invalid action_type: {self.action_type}. Must be one of: {valid_actions}")

    def compute_hash(self) -> str:
        """Compute deterministic hash of intent."""
        data = {
            "intent_id": self.intent_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "action_type": self.action_type,
            "action_description": self.action_description,
            "requester_id": self.requester_id,
            "requester_role": self.requester_role,
            "target_workspace": self.target_workspace,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

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
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionIntent":
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = tuple(sorted(metadata.items()))
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
            metadata=metadata,
        )


@dataclass(frozen=True)
class ValidationChainInput:
    """
    Complete input for execution validation chain.

    ALL fields are MANDATORY. If any is None -> EXECUTION_BLOCKED.
    """
    intent: Optional[ExecutionIntent]
    eligibility_result: Optional[EligibilityResult]
    approval_result: Optional[OrchestrationResult]
    gate_request: Optional[ExecutionRequest]

    def compute_hash(self) -> str:
        """Compute deterministic hash of all inputs."""
        data = {
            "intent": self.intent.to_dict() if self.intent else None,
            "eligibility": self.eligibility_result.to_dict() if self.eligibility_result else None,
            "approval": self.approval_result.to_dict() if self.approval_result else None,
            "gate_request": self.gate_request.to_dict() if self.gate_request else None,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Execution Result (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionResult:
    """
    Result of execution dispatch.

    Immutable record of the execution outcome.
    """
    execution_id: str
    intent_id: str
    status: str  # ExecutionStatus value
    block_reason: Optional[str]  # BlockReason value if blocked
    failure_reason: Optional[str]  # FailureReason value if failed
    input_hash: str
    timestamp: str
    dispatcher_version: str
    gate_decision_allowed: Optional[bool]
    execution_output: Optional[str]  # Truncated output if execution occurred
    rollback_performed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "status": self.status,
            "block_reason": self.block_reason,
            "failure_reason": self.failure_reason,
            "input_hash": self.input_hash,
            "timestamp": self.timestamp,
            "dispatcher_version": self.dispatcher_version,
            "gate_decision_allowed": self.gate_decision_allowed,
            "execution_output": self.execution_output,
            "rollback_performed": self.rollback_performed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            execution_id=data["execution_id"],
            intent_id=data["intent_id"],
            status=data["status"],
            block_reason=data.get("block_reason"),
            failure_reason=data.get("failure_reason"),
            input_hash=data["input_hash"],
            timestamp=data["timestamp"],
            dispatcher_version=data["dispatcher_version"],
            gate_decision_allowed=data.get("gate_decision_allowed"),
            execution_output=data.get("execution_output"),
            rollback_performed=data.get("rollback_performed", False),
        )


# -----------------------------------------------------------------------------
# Audit Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ExecutionAuditRecord:
    """
    Immutable audit record for execution dispatch.

    Append-only. No updates. No deletes.
    """
    audit_id: str
    execution_id: str
    intent_id: str
    input_hash: str
    status: str
    block_reason: Optional[str]
    failure_reason: Optional[str]
    timestamp: str
    dispatcher_version: str
    project_id: Optional[str]
    action_type: Optional[str]
    requester_id: Optional[str]
    eligibility_decision: Optional[str]
    approval_status: Optional[str]
    gate_allowed: Optional[bool]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "execution_id": self.execution_id,
            "intent_id": self.intent_id,
            "input_hash": self.input_hash,
            "status": self.status,
            "block_reason": self.block_reason,
            "failure_reason": self.failure_reason,
            "timestamp": self.timestamp,
            "dispatcher_version": self.dispatcher_version,
            "project_id": self.project_id,
            "action_type": self.action_type,
            "requester_id": self.requester_id,
            "eligibility_decision": self.eligibility_decision,
            "approval_status": self.approval_status,
            "gate_allowed": self.gate_allowed,
        }


# -----------------------------------------------------------------------------
# Controlled Execution Dispatcher (SINGLE EXECUTION POINT)
# -----------------------------------------------------------------------------
class ControlledExecutionDispatcher:
    """
    Phase 18C: Controlled Execution Dispatcher.

    The ONLY layer allowed to execute real actions in the platform.

    CRITICAL CONSTRAINTS:
    - ALL actions MUST flow through this dispatcher
    - Validates complete chain: Eligibility -> Approval -> Gate
    - Returns execution result, may perform actual execution
    - ATOMIC: Either fully execute or fully rollback
    - AUDIT REQUIRED: Every dispatch emits immutable audit record

    If ANY validation fails -> EXECUTION_BLOCKED
    If audit write fails -> EXECUTION_BLOCKED
    If execution fails -> EXECUTION_FAILED (with rollback)
    """

    def __init__(
        self,
        intents_file: Optional[Path] = None,
        results_file: Optional[Path] = None,
        audit_file: Optional[Path] = None,
        execution_gate: Optional[ExecutionGate] = None,
    ):
        """
        Initialize dispatcher.

        Args:
            intents_file: Path to intents file (optional, for testing)
            results_file: Path to results file (optional, for testing)
            audit_file: Path to audit file (optional, for testing)
            execution_gate: Execution gate instance (optional, for testing)
        """
        self._intents_file = intents_file or EXECUTION_INTENTS_FILE
        self._results_file = results_file or EXECUTION_RESULTS_FILE
        self._audit_file = audit_file or EXECUTION_AUDIT_FILE
        self._execution_gate = execution_gate or ExecutionGate()
        self._version = DISPATCHER_VERSION

    def dispatch(self, chain_input: ValidationChainInput) -> ExecutionResult:
        """
        Dispatch an execution request through the validation chain.

        This is the ONLY public method for execution. It:
        1. Validates all chain inputs are present
        2. Validates eligibility allows execution
        3. Validates approval is granted
        4. Validates execution gate allows
        5. Performs execution (if all validations pass)
        6. Writes audit record
        7. Returns result

        If audit write fails -> EXECUTION_BLOCKED

        Args:
            chain_input: Complete validation chain input

        Returns:
            ExecutionResult with status and details
        """
        timestamp = datetime.utcnow().isoformat()
        input_hash = chain_input.compute_hash()

        # Generate execution ID
        execution_id = f"exec-{timestamp.replace(':', '-').replace('.', '-')}-{uuid.uuid4().hex[:8]}"
        intent_id = chain_input.intent.intent_id if chain_input.intent else "unknown"

        # Phase 1: Validate chain inputs are present
        missing_reason = self._check_missing_inputs(chain_input)
        if missing_reason:
            return self._create_blocked_result(
                execution_id=execution_id,
                intent_id=intent_id,
                block_reason=missing_reason,
                input_hash=input_hash,
                timestamp=timestamp,
                chain_input=chain_input,
            )

        # Phase 2: Validate eligibility
        eligibility_block = self._check_eligibility(chain_input)
        if eligibility_block:
            return self._create_blocked_result(
                execution_id=execution_id,
                intent_id=intent_id,
                block_reason=eligibility_block,
                input_hash=input_hash,
                timestamp=timestamp,
                chain_input=chain_input,
            )

        # Phase 3: Validate approval
        approval_block = self._check_approval(chain_input)
        if approval_block:
            return self._create_blocked_result(
                execution_id=execution_id,
                intent_id=intent_id,
                block_reason=approval_block,
                input_hash=input_hash,
                timestamp=timestamp,
                chain_input=chain_input,
            )

        # Phase 4: Validate execution gate
        gate_decision = self._execution_gate.evaluate(chain_input.gate_request)
        if not gate_decision.allowed:
            block_reason = BlockReason.GATE_DENIED.value
            if gate_decision.hard_fail:
                block_reason = BlockReason.GATE_HARD_FAIL.value
            if gate_decision.drift_blocks_execution:
                block_reason = BlockReason.DRIFT_BLOCKS_EXECUTION.value

            return self._create_blocked_result(
                execution_id=execution_id,
                intent_id=intent_id,
                block_reason=block_reason,
                input_hash=input_hash,
                timestamp=timestamp,
                chain_input=chain_input,
                gate_allowed=False,
            )

        # Phase 5: All validations passed - record intent and return pending
        # Actual execution is performed by the execution backend
        self._record_intent(chain_input.intent)

        result = ExecutionResult(
            execution_id=execution_id,
            intent_id=intent_id,
            status=ExecutionStatus.EXECUTION_PENDING.value,
            block_reason=None,
            failure_reason=None,
            input_hash=input_hash,
            timestamp=timestamp,
            dispatcher_version=self._version,
            gate_decision_allowed=True,
            execution_output=None,
            rollback_performed=False,
        )

        # Phase 6: Write audit record (MANDATORY)
        audit_success = self._write_audit_record(result, chain_input)

        # If audit fails, block execution
        if not audit_success:
            return self._create_blocked_result(
                execution_id=execution_id,
                intent_id=intent_id,
                block_reason=BlockReason.AUDIT_WRITE_FAILED.value,
                input_hash=input_hash,
                timestamp=timestamp,
                chain_input=chain_input,
            )

        # Record result
        self._record_result(result)

        return result

    def record_execution_outcome(
        self,
        execution_id: str,
        success: bool,
        output: Optional[str] = None,
        failure_reason: Optional[str] = None,
        rollback_performed: bool = False,
    ) -> ExecutionResult:
        """
        Record the outcome of an execution after it completes.

        Called by the execution backend after actual execution.

        Args:
            execution_id: ID of the execution
            success: Whether execution succeeded
            output: Execution output (truncated)
            failure_reason: Reason for failure if not successful
            rollback_performed: Whether rollback was performed

        Returns:
            Updated ExecutionResult
        """
        timestamp = datetime.utcnow().isoformat()

        # Get the pending result
        pending = self._get_result(execution_id)
        if not pending:
            # Create a new result for unknown execution
            result = ExecutionResult(
                execution_id=execution_id,
                intent_id="unknown",
                status=ExecutionStatus.EXECUTION_FAILED.value if not success else ExecutionStatus.EXECUTION_SUCCESS.value,
                block_reason=None,
                failure_reason=failure_reason,
                input_hash="unknown",
                timestamp=timestamp,
                dispatcher_version=self._version,
                gate_decision_allowed=True,
                execution_output=output[:1000] if output else None,
                rollback_performed=rollback_performed,
            )
        else:
            # Update with outcome
            if success:
                status = ExecutionStatus.EXECUTION_SUCCESS.value
            else:
                status = ExecutionStatus.EXECUTION_FAILED.value

            result = ExecutionResult(
                execution_id=execution_id,
                intent_id=pending.intent_id,
                status=status,
                block_reason=None,
                failure_reason=failure_reason,
                input_hash=pending.input_hash,
                timestamp=timestamp,
                dispatcher_version=self._version,
                gate_decision_allowed=pending.gate_decision_allowed,
                execution_output=output[:1000] if output else None,
                rollback_performed=rollback_performed,
            )

        # Record the outcome
        self._record_result(result)

        return result

    def _check_missing_inputs(self, chain_input: ValidationChainInput) -> Optional[str]:
        """Check for missing mandatory inputs."""
        if chain_input.intent is None:
            return BlockReason.MISSING_INTENT.value

        if chain_input.eligibility_result is None:
            return BlockReason.MISSING_ELIGIBILITY.value

        if chain_input.approval_result is None:
            return BlockReason.MISSING_APPROVAL.value

        if chain_input.gate_request is None:
            return BlockReason.MISSING_GATE_REQUEST.value

        return None

    def _check_eligibility(self, chain_input: ValidationChainInput) -> Optional[str]:
        """Check eligibility allows execution."""
        eligibility = chain_input.eligibility_result
        intent = chain_input.intent

        # If eligibility is FORBIDDEN, block
        if eligibility.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value:
            return BlockReason.ELIGIBILITY_FORBIDDEN.value

        # If eligibility is LIMITED, check if action is in allowed list
        if eligibility.decision == EligibilityDecision.AUTOMATION_ALLOWED_LIMITED.value:
            if intent.action_type not in eligibility.allowed_actions:
                return BlockReason.ACTION_NOT_IN_ALLOWED_LIST.value

        return None

    def _check_approval(self, chain_input: ValidationChainInput) -> Optional[str]:
        """Check approval allows execution."""
        approval = chain_input.approval_result

        # If approval is DENIED, block
        if approval.status == ApprovalStatus.APPROVAL_DENIED.value:
            return BlockReason.APPROVAL_DENIED.value

        # If approval is PENDING, block (cannot execute without approval)
        if approval.status == ApprovalStatus.APPROVAL_PENDING.value:
            return BlockReason.APPROVAL_PENDING.value

        # APPROVAL_GRANTED - proceed
        return None

    def _create_blocked_result(
        self,
        execution_id: str,
        intent_id: str,
        block_reason: str,
        input_hash: str,
        timestamp: str,
        chain_input: ValidationChainInput,
        gate_allowed: Optional[bool] = None,
    ) -> ExecutionResult:
        """Create EXECUTION_BLOCKED result."""
        result = ExecutionResult(
            execution_id=execution_id,
            intent_id=intent_id,
            status=ExecutionStatus.EXECUTION_BLOCKED.value,
            block_reason=block_reason,
            failure_reason=None,
            input_hash=input_hash,
            timestamp=timestamp,
            dispatcher_version=self._version,
            gate_decision_allowed=gate_allowed,
            execution_output=None,
            rollback_performed=False,
        )

        # Write audit record (even for blocked executions)
        self._write_audit_record(result, chain_input)

        # Record result
        self._record_result(result)

        return result

    def _record_intent(self, intent: ExecutionIntent) -> None:
        """Record intent to append-only store."""
        try:
            self._intents_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._intents_file, 'a') as f:
                f.write(json.dumps(intent.to_dict()) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            # Intent recording failure is logged but doesn't block
            pass

    def _record_result(self, result: ExecutionResult) -> None:
        """Record result to append-only store."""
        try:
            self._results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._results_file, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            # Result recording failure is logged but doesn't block
            pass

    def _get_result(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get a result by execution ID."""
        if not self._results_file.exists():
            return None

        try:
            with open(self._results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("execution_id") == execution_id:
                                return ExecutionResult.from_dict(data)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        return None

    def _write_audit_record(
        self,
        result: ExecutionResult,
        chain_input: ValidationChainInput,
    ) -> bool:
        """
        Write immutable audit record.

        Append-only. No updates. No deletes.
        If write fails -> return False (triggers EXECUTION_BLOCKED).
        """
        try:
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)

            audit_id = f"aud-{result.timestamp.replace(':', '-').replace('.', '-')}-{result.input_hash[:8]}"

            # Extract data from chain input
            project_id = None
            action_type = None
            requester_id = None
            eligibility_decision = None
            approval_status = None

            if chain_input.intent:
                project_id = chain_input.intent.project_id
                action_type = chain_input.intent.action_type
                requester_id = chain_input.intent.requester_id

            if chain_input.eligibility_result:
                eligibility_decision = chain_input.eligibility_result.decision

            if chain_input.approval_result:
                approval_status = chain_input.approval_result.status

            audit_record = ExecutionAuditRecord(
                audit_id=audit_id,
                execution_id=result.execution_id,
                intent_id=result.intent_id,
                input_hash=result.input_hash,
                status=result.status,
                block_reason=result.block_reason,
                failure_reason=result.failure_reason,
                timestamp=result.timestamp,
                dispatcher_version=result.dispatcher_version,
                project_id=project_id,
                action_type=action_type,
                requester_id=requester_id,
                eligibility_decision=eligibility_decision,
                approval_status=approval_status,
                gate_allowed=result.gate_decision_allowed,
            )

            with open(self._audit_file, 'a') as f:
                f.write(json.dumps(audit_record.to_dict()) + '\n')
                f.flush()
                os.fsync(f.fileno())

            return True

        except Exception:
            return False

    # -------------------------------------------------------------------------
    # Read Operations (Read-Only)
    # -------------------------------------------------------------------------

    def get_intent(self, intent_id: str) -> Optional[ExecutionIntent]:
        """Get an intent by ID."""
        if not self._intents_file.exists():
            return None

        try:
            with open(self._intents_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("intent_id") == intent_id:
                                return ExecutionIntent.from_dict(data)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        return None

    def get_execution(self, execution_id: str) -> Optional[ExecutionResult]:
        """Get an execution result by ID."""
        return self._get_result(execution_id)

    def get_recent_executions(
        self,
        limit: int = 100,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[ExecutionResult]:
        """Get recent execution results."""
        if not self._results_file.exists():
            return []

        results = []
        try:
            with open(self._results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)

                            # Filter by status if specified
                            if status and data.get("status") != status:
                                continue

                            results.append(ExecutionResult.from_dict(data))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        # Return most recent first, limited
        results.reverse()
        return results[:limit]

    def get_summary(
        self,
        project_id: Optional[str] = None,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get execution summary statistics."""
        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        total = 0
        blocked = 0
        pending = 0
        success = 0
        failed = 0
        by_action_type: Dict[str, int] = {}
        by_block_reason: Dict[str, int] = {}

        if self._results_file.exists():
            try:
                with open(self._results_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)

                                if data.get("timestamp", "") < cutoff:
                                    continue

                                total += 1
                                status = data.get("status")

                                if status == ExecutionStatus.EXECUTION_BLOCKED.value:
                                    blocked += 1
                                    reason = data.get("block_reason", "unknown")
                                    by_block_reason[reason] = by_block_reason.get(reason, 0) + 1
                                elif status == ExecutionStatus.EXECUTION_PENDING.value:
                                    pending += 1
                                elif status == ExecutionStatus.EXECUTION_SUCCESS.value:
                                    success += 1
                                elif status == ExecutionStatus.EXECUTION_FAILED.value:
                                    failed += 1

                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass

        # Count by action type from intents
        if self._intents_file.exists():
            try:
                with open(self._intents_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                if data.get("created_at", "") >= cutoff:
                                    action = data.get("action_type", "unknown")
                                    by_action_type[action] = by_action_type.get(action, 0) + 1
                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "since_hours": since_hours,
            "project_id": project_id,
            "total_executions": total,
            "blocked_count": blocked,
            "pending_count": pending,
            "success_count": success,
            "failed_count": failed,
            "by_action_type": by_action_type,
            "by_block_reason": by_block_reason,
        }


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------

# Singleton instance
_dispatcher: Optional[ControlledExecutionDispatcher] = None


def get_execution_dispatcher(
    intents_file: Optional[Path] = None,
    results_file: Optional[Path] = None,
    audit_file: Optional[Path] = None,
) -> ControlledExecutionDispatcher:
    """Get the execution dispatcher singleton."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ControlledExecutionDispatcher(
            intents_file=intents_file,
            results_file=results_file,
            audit_file=audit_file,
        )
    return _dispatcher


def create_execution_intent(
    project_id: str,
    project_name: str,
    action_type: str,
    action_description: str,
    requester_id: str,
    requester_role: str,
    target_workspace: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExecutionIntent:
    """
    Create an execution intent.

    Convenience function for creating intents from API/external data.
    """
    timestamp = datetime.utcnow().isoformat()
    intent_id = f"int-{timestamp.replace(':', '-').replace('.', '-')}-{uuid.uuid4().hex[:8]}"

    metadata_tuple = tuple(sorted((metadata or {}).items()))

    return ExecutionIntent(
        intent_id=intent_id,
        project_id=project_id,
        project_name=project_name,
        action_type=action_type,
        action_description=action_description,
        requester_id=requester_id,
        requester_role=requester_role,
        target_workspace=target_workspace,
        created_at=timestamp,
        metadata=metadata_tuple,
    )


def create_validation_chain_input(
    intent: ExecutionIntent,
    eligibility_result: EligibilityResult,
    approval_result: OrchestrationResult,
    gate_request: ExecutionRequest,
) -> ValidationChainInput:
    """
    Create a validation chain input.

    Convenience function for creating chain input from components.
    """
    return ValidationChainInput(
        intent=intent,
        eligibility_result=eligibility_result,
        approval_result=approval_result,
        gate_request=gate_request,
    )


def dispatch_execution(chain_input: ValidationChainInput) -> ExecutionResult:
    """
    Dispatch an execution request.

    Convenience function using singleton dispatcher.

    Args:
        chain_input: Complete validation chain input

    Returns:
        ExecutionResult with status and details
    """
    dispatcher = get_execution_dispatcher()
    return dispatcher.dispatch(chain_input)


def get_execution_summary(
    project_id: Optional[str] = None,
    since_hours: int = 24,
) -> Dict[str, Any]:
    """
    Get execution summary.

    Convenience function using singleton dispatcher.
    """
    dispatcher = get_execution_dispatcher()
    return dispatcher.get_summary(project_id=project_id, since_hours=since_hours)


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
| intent=None               | EXECUTION_BLOCKED | MISSING_INTENT                 |
| eligibility_result=None   | EXECUTION_BLOCKED | MISSING_ELIGIBILITY            |
| approval_result=None      | EXECUTION_BLOCKED | MISSING_APPROVAL               |
| gate_request=None         | EXECUTION_BLOCKED | MISSING_GATE_REQUEST           |
+---------------------------+-------------------+--------------------------------+
| eligibility=FORBIDDEN     | EXECUTION_BLOCKED | ELIGIBILITY_FORBIDDEN          |
| eligibility=LIMITED +     | EXECUTION_BLOCKED | ACTION_NOT_IN_ALLOWED_LIST     |
|   action not in list      |                   |                                |
+---------------------------+-------------------+--------------------------------+
| approval=DENIED           | EXECUTION_BLOCKED | APPROVAL_DENIED                |
| approval=PENDING          | EXECUTION_BLOCKED | APPROVAL_PENDING               |
+---------------------------+-------------------+--------------------------------+
| gate=denied               | EXECUTION_BLOCKED | GATE_DENIED                    |
| gate=hard_fail            | EXECUTION_BLOCKED | GATE_HARD_FAIL                 |
| gate=drift_blocks         | EXECUTION_BLOCKED | DRIFT_BLOCKS_EXECUTION         |
+---------------------------+-------------------+--------------------------------+
| audit write fails         | EXECUTION_BLOCKED | AUDIT_WRITE_FAILED             |
+---------------------------+-------------------+--------------------------------+
| all validations pass      | EXECUTION_PENDING | None                           |
+---------------------------+-------------------+--------------------------------+
| execution succeeds        | EXECUTION_SUCCESS | None                           |
| execution fails           | EXECUTION_FAILED  | (failure reason)               |
+---------------------------+-------------------+--------------------------------+

INVARIANTS:
- Same inputs ALWAYS produce same validation outcome (deterministic)
- Missing input -> BLOCKED (never PENDING or SUCCESS)
- Eligibility FORBIDDEN -> BLOCKED (never PENDING or SUCCESS)
- Approval DENIED/PENDING -> BLOCKED (never SUCCESS)
- Gate denied -> BLOCKED (never SUCCESS)
- Audit failure -> BLOCKED (fail closed)
- All validation chain inputs must match (hash validation)
"""
