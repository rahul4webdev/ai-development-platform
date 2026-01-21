"""
Phase 16F: Intent Contract Enforcement

Human-AI Contract model for enforcing intent boundaries.

This module defines the contract between humans and Claude about what
Claude is ALLOWED to do within a project's scope. It enforces hard limits
on project evolution without explicit approval.

CONTRACT TYPES:
- SOFT: Claude may proceed, warning logged
- CONFIRMATION_REQUIRED: Claude must stop and ask user
- HARD_BLOCK: Claude is forbidden from proceeding

CLAUDE RESTRICTIONS (ENFORCED):
- NEVER change architecture class without approval
- NEVER add new production domains silently
- NEVER introduce new database types without consent
- NEVER expand project purpose beyond baseline
- NEVER bypass drift checks during execution
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from .intent_drift_engine import (
    DriftLevel,
    DriftDimension,
    DriftAnalysisResult,
    analyze_drift,
)
from .intent_baseline import (
    get_active_baseline,
    IntentBaseline,
)

logger = logging.getLogger("intent_contract")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CONTRACT_DIR = Path(os.getenv(
    "INTENT_CONTRACT_DIR",
    "/home/aitesting.mybd.in/jobs/intent_contracts"
))

# Fallback for local development
if not CONTRACT_DIR.exists():
    CONTRACT_DIR = Path("/tmp/intent_contracts")


# -----------------------------------------------------------------------------
# Contract Types
# -----------------------------------------------------------------------------
class ContractType(str, Enum):
    """
    Contract enforcement types.

    These determine how violations are handled.
    """
    SOFT = "soft"                       # Log and proceed
    CONFIRMATION_REQUIRED = "confirmation_required"  # Stop and ask
    HARD_BLOCK = "hard_block"           # Forbidden


class ViolationType(str, Enum):
    """Types of contract violations."""
    ARCHITECTURE_CHANGE = "architecture_change"
    DATABASE_CHANGE = "database_change"
    PURPOSE_EXPANSION = "purpose_expansion"
    DOMAIN_ADDITION = "domain_addition"
    MODULE_ADDITION = "module_addition"
    USER_TYPE_CHANGE = "user_type_change"
    DRIFT_THRESHOLD_EXCEEDED = "drift_threshold_exceeded"


class EnforcementAction(str, Enum):
    """Actions taken on contract evaluation."""
    ALLOW = "allow"                     # Proceed normally
    WARN = "warn"                       # Log warning, proceed
    CONFIRM = "confirm"                 # Require user confirmation
    BLOCK = "block"                     # Block execution
    FREEZE = "freeze"                   # Freeze project entirely


# -----------------------------------------------------------------------------
# Violation Rules Configuration
# -----------------------------------------------------------------------------
# Maps violation types to their default contract type
VIOLATION_CONTRACT_MAP = {
    ViolationType.ARCHITECTURE_CHANGE: ContractType.HARD_BLOCK,
    ViolationType.DATABASE_CHANGE: ContractType.HARD_BLOCK,
    ViolationType.PURPOSE_EXPANSION: ContractType.CONFIRMATION_REQUIRED,
    ViolationType.DOMAIN_ADDITION: ContractType.CONFIRMATION_REQUIRED,
    ViolationType.MODULE_ADDITION: ContractType.SOFT,
    ViolationType.USER_TYPE_CHANGE: ContractType.SOFT,
    ViolationType.DRIFT_THRESHOLD_EXCEEDED: ContractType.HARD_BLOCK,
}

# Drift level to contract type mapping
DRIFT_LEVEL_CONTRACT_MAP = {
    DriftLevel.NONE: ContractType.SOFT,
    DriftLevel.LOW: ContractType.SOFT,
    DriftLevel.MEDIUM: ContractType.CONFIRMATION_REQUIRED,
    DriftLevel.HIGH: ContractType.HARD_BLOCK,
    DriftLevel.CRITICAL: ContractType.HARD_BLOCK,
}


# -----------------------------------------------------------------------------
# Contract Violation Record
# -----------------------------------------------------------------------------
@dataclass
class ContractViolation:
    """A detected contract violation."""
    violation_id: str
    project_id: str
    project_name: str
    violation_type: str  # ViolationType value
    contract_type: str   # ContractType value
    dimension: str       # DriftDimension value (if applicable)
    description: str
    baseline_value: Any
    current_value: Any
    severity: str        # DriftLevel value
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContractViolation":
        return cls(**data)


@dataclass
class ContractEvaluationResult:
    """
    Result of contract evaluation.

    This is the final verdict on whether execution can proceed.
    """
    project_id: str
    project_name: str
    baseline_id: str
    evaluation_timestamp: str

    # Overall verdict
    action: str  # EnforcementAction value
    can_proceed: bool
    requires_confirmation: bool

    # Drift analysis reference
    drift_analysis: Optional[Dict[str, Any]]  # DriftAnalysisResult as dict

    # Violations found
    violations: List[ContractViolation]

    # Messages
    summary: str
    user_prompt: Optional[str]  # What to ask user if confirmation needed
    block_reason: Optional[str]  # Why blocked if blocked

    # Confirmation state (if applicable)
    confirmation_id: Optional[str] = None
    confirmed_at: Optional[str] = None
    confirmed_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "baseline_id": self.baseline_id,
            "evaluation_timestamp": self.evaluation_timestamp,
            "action": self.action,
            "can_proceed": self.can_proceed,
            "requires_confirmation": self.requires_confirmation,
            "drift_analysis": self.drift_analysis,
            "violations": [v.to_dict() for v in self.violations],
            "summary": self.summary,
            "user_prompt": self.user_prompt,
            "block_reason": self.block_reason,
            "confirmation_id": self.confirmation_id,
            "confirmed_at": self.confirmed_at,
            "confirmed_by": self.confirmed_by,
        }
        return result

    def get_blocking_violations(self) -> List[ContractViolation]:
        """Get violations that are blocking execution."""
        return [
            v for v in self.violations
            if v.contract_type == ContractType.HARD_BLOCK.value
        ]

    def get_confirmation_violations(self) -> List[ContractViolation]:
        """Get violations that require confirmation."""
        return [
            v for v in self.violations
            if v.contract_type == ContractType.CONFIRMATION_REQUIRED.value
        ]


# -----------------------------------------------------------------------------
# Pending Confirmation Tracking
# -----------------------------------------------------------------------------
@dataclass
class PendingConfirmation:
    """A pending user confirmation request."""
    confirmation_id: str
    project_id: str
    project_name: str
    evaluation_result: Dict[str, Any]  # ContractEvaluationResult as dict
    prompt: str
    created_at: str
    expires_at: str  # Confirmation expires after timeout
    status: str = "pending"  # pending, confirmed, rejected, expired
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PendingConfirmation":
        return cls(**data)


# -----------------------------------------------------------------------------
# Intent Contract Enforcer
# -----------------------------------------------------------------------------
class IntentContractEnforcer:
    """
    Enforces intent contracts between humans and Claude.

    This is the final arbiter of whether Claude can proceed with execution.
    All evaluations are logged immutably.
    """

    def __init__(self, contract_dir: Optional[Path] = None):
        self._contract_dir = contract_dir or CONTRACT_DIR
        self._evaluations_file = self._contract_dir / "evaluations.json"
        self._confirmations_file = self._contract_dir / "pending_confirmations.json"
        self._audit_file = self._contract_dir / "contract_audit.log"

        # Ensure directory exists
        self._contract_dir.mkdir(parents=True, exist_ok=True)

        # Load pending confirmations
        self._pending_confirmations: Dict[str, PendingConfirmation] = {}
        self._load_confirmations()

    def _load_confirmations(self) -> None:
        """Load pending confirmations from storage."""
        if self._confirmations_file.exists():
            try:
                with open(self._confirmations_file) as f:
                    data = json.load(f)
                    for item in data.get("confirmations", []):
                        conf = PendingConfirmation.from_dict(item)
                        self._pending_confirmations[conf.confirmation_id] = conf
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load confirmations: {e}")

    def _save_confirmations(self) -> None:
        """Persist pending confirmations."""
        data = {
            "confirmations": [c.to_dict() for c in self._pending_confirmations.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        temp_file = self._confirmations_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self._confirmations_file)

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Append to audit log (immutable)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            **details,
        }
        with open(self._audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -------------------------------------------------------------------------
    # Contract Evaluation
    # -------------------------------------------------------------------------

    def evaluate_contract(
        self,
        project_id: str,
        project_name: str,
        current_intent: Dict[str, Any],
        action_description: str = "execution",
    ) -> ContractEvaluationResult:
        """
        Evaluate if an action is allowed under the project's contract.

        This is the main entry point for contract enforcement.

        Args:
            project_id: Project identifier
            project_name: Project name
            current_intent: Current NormalizedIntent as dict
            action_description: What action is being attempted

        Returns:
            ContractEvaluationResult with verdict
        """
        now = datetime.utcnow().isoformat()

        # Get active baseline
        baseline = get_active_baseline(project_id)
        if not baseline:
            # No baseline means new project - allow
            return ContractEvaluationResult(
                project_id=project_id,
                project_name=project_name,
                baseline_id="none",
                evaluation_timestamp=now,
                action=EnforcementAction.ALLOW.value,
                can_proceed=True,
                requires_confirmation=False,
                drift_analysis=None,
                violations=[],
                summary="No baseline found. Project may proceed (initial setup).",
                user_prompt=None,
                block_reason=None,
            )

        # Perform drift analysis
        drift_result = analyze_drift(
            project_id=project_id,
            baseline_id=baseline.baseline_id,
            baseline_intent=baseline.normalized_intent,
            current_intent=current_intent,
        )

        # Detect violations
        violations = self._detect_violations(
            project_id=project_id,
            project_name=project_name,
            baseline=baseline,
            current_intent=current_intent,
            drift_result=drift_result,
        )

        # Determine action based on violations
        action, can_proceed, requires_confirmation = self._determine_action(violations, drift_result)

        # Generate messages
        summary = self._generate_summary(action, violations, drift_result)
        user_prompt = self._generate_user_prompt(violations) if requires_confirmation else None
        block_reason = self._generate_block_reason(violations) if not can_proceed else None

        result = ContractEvaluationResult(
            project_id=project_id,
            project_name=project_name,
            baseline_id=baseline.baseline_id,
            evaluation_timestamp=now,
            action=action,
            can_proceed=can_proceed,
            requires_confirmation=requires_confirmation,
            drift_analysis=drift_result.to_dict(),
            violations=violations,
            summary=summary,
            user_prompt=user_prompt,
            block_reason=block_reason,
        )

        # Log evaluation
        self._log_audit("CONTRACT_EVALUATED", {
            "project_id": project_id,
            "project_name": project_name,
            "action": action,
            "can_proceed": can_proceed,
            "requires_confirmation": requires_confirmation,
            "violation_count": len(violations),
            "drift_level": drift_result.overall_level,
        })

        return result

    def _detect_violations(
        self,
        project_id: str,
        project_name: str,
        baseline: IntentBaseline,
        current_intent: Dict[str, Any],
        drift_result: DriftAnalysisResult,
    ) -> List[ContractViolation]:
        """Detect all contract violations based on drift analysis."""
        violations = []
        now = datetime.utcnow().isoformat()
        violation_counter = 0

        for dim_drift in drift_result.dimension_drifts:
            if dim_drift.level in [DriftLevel.NONE.value, DriftLevel.LOW.value]:
                continue

            # Map dimension to violation type
            violation_type = self._dimension_to_violation_type(dim_drift.dimension)
            contract_type = VIOLATION_CONTRACT_MAP.get(
                violation_type,
                ContractType.SOFT
            )

            # Override contract type based on drift level
            drift_level = DriftLevel(dim_drift.level)
            drift_contract = DRIFT_LEVEL_CONTRACT_MAP.get(drift_level, ContractType.SOFT)

            # Use stricter of the two
            if self._contract_severity(drift_contract) > self._contract_severity(contract_type):
                contract_type = drift_contract

            violation_counter += 1
            violation = ContractViolation(
                violation_id=f"viol-{project_id[:8]}-{violation_counter}",
                project_id=project_id,
                project_name=project_name,
                violation_type=violation_type.value,
                contract_type=contract_type.value,
                dimension=dim_drift.dimension,
                description=dim_drift.explanation,
                baseline_value=dim_drift.baseline_value,
                current_value=dim_drift.current_value,
                severity=dim_drift.level,
                timestamp=now,
            )
            violations.append(violation)

        # Check for overall drift threshold violation
        if drift_result.overall_level in [DriftLevel.HIGH.value, DriftLevel.CRITICAL.value]:
            violation_counter += 1
            violations.append(ContractViolation(
                violation_id=f"viol-{project_id[:8]}-{violation_counter}",
                project_id=project_id,
                project_name=project_name,
                violation_type=ViolationType.DRIFT_THRESHOLD_EXCEEDED.value,
                contract_type=ContractType.HARD_BLOCK.value,
                dimension="overall",
                description=f"Overall drift score {drift_result.overall_score} exceeds threshold",
                baseline_value=None,
                current_value=drift_result.overall_score,
                severity=drift_result.overall_level,
                timestamp=now,
            ))

        return violations

    def _dimension_to_violation_type(self, dimension: str) -> ViolationType:
        """Map drift dimension to violation type."""
        mapping = {
            DriftDimension.ARCHITECTURE.value: ViolationType.ARCHITECTURE_CHANGE,
            DriftDimension.DATABASE.value: ViolationType.DATABASE_CHANGE,
            DriftDimension.PURPOSE.value: ViolationType.PURPOSE_EXPANSION,
            DriftDimension.SURFACE_AREA.value: ViolationType.DOMAIN_ADDITION,
            DriftDimension.MODULE.value: ViolationType.MODULE_ADDITION,
            DriftDimension.NON_FUNCTIONAL.value: ViolationType.USER_TYPE_CHANGE,
        }
        return mapping.get(dimension, ViolationType.DRIFT_THRESHOLD_EXCEEDED)

    def _contract_severity(self, contract_type: ContractType) -> int:
        """Get numeric severity for contract type comparison."""
        severity_map = {
            ContractType.SOFT: 1,
            ContractType.CONFIRMATION_REQUIRED: 2,
            ContractType.HARD_BLOCK: 3,
        }
        return severity_map.get(contract_type, 0)

    def _determine_action(
        self,
        violations: List[ContractViolation],
        drift_result: DriftAnalysisResult,
    ) -> Tuple[str, bool, bool]:
        """
        Determine the enforcement action based on violations.

        Returns: (action, can_proceed, requires_confirmation)
        """
        if not violations:
            return EnforcementAction.ALLOW.value, True, False

        # Check for hard blocks
        has_hard_block = any(
            v.contract_type == ContractType.HARD_BLOCK.value
            for v in violations
        )

        # Check for confirmation required
        has_confirmation = any(
            v.contract_type == ContractType.CONFIRMATION_REQUIRED.value
            for v in violations
        )

        # Critical drift level freezes project
        if drift_result.overall_level == DriftLevel.CRITICAL.value:
            return EnforcementAction.FREEZE.value, False, False

        if has_hard_block:
            return EnforcementAction.BLOCK.value, False, False

        if has_confirmation:
            return EnforcementAction.CONFIRM.value, False, True

        # Only soft violations
        return EnforcementAction.WARN.value, True, False

    def _generate_summary(
        self,
        action: str,
        violations: List[ContractViolation],
        drift_result: DriftAnalysisResult,
    ) -> str:
        """Generate human-readable summary."""
        if action == EnforcementAction.ALLOW.value:
            return "Contract evaluation passed. Execution allowed."

        if action == EnforcementAction.WARN.value:
            return f"Minor contract concerns detected ({len(violations)} warnings). Proceeding with logging."

        if action == EnforcementAction.CONFIRM.value:
            dims = set(v.dimension for v in violations if v.dimension != "overall")
            return f"Contract requires confirmation. Changes detected in: {', '.join(dims)}"

        if action == EnforcementAction.BLOCK.value:
            blocking = [v for v in violations if v.contract_type == ContractType.HARD_BLOCK.value]
            dims = set(v.dimension for v in blocking)
            return f"Contract BLOCKS execution. Violations in: {', '.join(dims)}"

        # FREEZE
        return "Contract FREEZES project. Critical drift detected - human review required."

    def _generate_user_prompt(self, violations: List[ContractViolation]) -> str:
        """Generate the prompt to show the user for confirmation."""
        confirm_violations = [
            v for v in violations
            if v.contract_type == ContractType.CONFIRMATION_REQUIRED.value
        ]

        lines = ["The project has evolved beyond its original scope:", ""]
        for v in confirm_violations:
            lines.append(f"- {v.description}")
        lines.append("")
        lines.append("Do you want to approve these changes and continue?")

        return "\n".join(lines)

    def _generate_block_reason(self, violations: List[ContractViolation]) -> str:
        """Generate the reason for blocking execution."""
        block_violations = [
            v for v in violations
            if v.contract_type == ContractType.HARD_BLOCK.value
        ]

        lines = ["Execution blocked due to contract violations:", ""]
        for v in block_violations:
            lines.append(f"- [{v.violation_type}] {v.description}")
        lines.append("")
        lines.append("To proceed, request a rebaseline approval to update the project's intent baseline.")

        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Confirmation Workflow
    # -------------------------------------------------------------------------

    def create_confirmation_request(
        self,
        evaluation_result: ContractEvaluationResult,
        timeout_hours: int = 24,
    ) -> PendingConfirmation:
        """
        Create a pending confirmation request.

        This is called when contract evaluation requires user confirmation.
        """
        import uuid
        now = datetime.utcnow()
        expires = datetime.utcfromtimestamp(now.timestamp() + timeout_hours * 3600)

        confirmation_id = f"confirm-{uuid.uuid4().hex[:12]}"

        confirmation = PendingConfirmation(
            confirmation_id=confirmation_id,
            project_id=evaluation_result.project_id,
            project_name=evaluation_result.project_name,
            evaluation_result=evaluation_result.to_dict(),
            prompt=evaluation_result.user_prompt or "",
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )

        self._pending_confirmations[confirmation_id] = confirmation
        self._save_confirmations()

        self._log_audit("CONFIRMATION_REQUESTED", {
            "confirmation_id": confirmation_id,
            "project_id": evaluation_result.project_id,
            "expires_at": expires.isoformat(),
        })

        return confirmation

    def approve_confirmation(
        self,
        confirmation_id: str,
        approved_by: str,
        notes: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Approve a pending confirmation.

        Returns: (success, message)
        """
        confirmation = self._pending_confirmations.get(confirmation_id)
        if not confirmation:
            return False, f"Confirmation not found: {confirmation_id}"

        if confirmation.status != "pending":
            return False, f"Confirmation is not pending (status: {confirmation.status})"

        # Check expiry
        now = datetime.utcnow()
        expires = datetime.fromisoformat(confirmation.expires_at)
        if now > expires:
            confirmation.status = "expired"
            self._save_confirmations()
            return False, "Confirmation has expired"

        confirmation.status = "confirmed"
        confirmation.resolved_at = now.isoformat()
        confirmation.resolved_by = approved_by
        confirmation.resolution_notes = notes

        self._save_confirmations()

        self._log_audit("CONFIRMATION_APPROVED", {
            "confirmation_id": confirmation_id,
            "project_id": confirmation.project_id,
            "approved_by": approved_by,
        })

        return True, f"Confirmation approved: {confirmation_id}"

    def reject_confirmation(
        self,
        confirmation_id: str,
        rejected_by: str,
        notes: str,
    ) -> Tuple[bool, str]:
        """
        Reject a pending confirmation.

        Returns: (success, message)
        """
        confirmation = self._pending_confirmations.get(confirmation_id)
        if not confirmation:
            return False, f"Confirmation not found: {confirmation_id}"

        if confirmation.status != "pending":
            return False, f"Confirmation is not pending (status: {confirmation.status})"

        now = datetime.utcnow()

        confirmation.status = "rejected"
        confirmation.resolved_at = now.isoformat()
        confirmation.resolved_by = rejected_by
        confirmation.resolution_notes = notes

        self._save_confirmations()

        self._log_audit("CONFIRMATION_REJECTED", {
            "confirmation_id": confirmation_id,
            "project_id": confirmation.project_id,
            "rejected_by": rejected_by,
        })

        return True, f"Confirmation rejected: {confirmation_id}"

    def get_pending_confirmations(
        self,
        project_id: Optional[str] = None,
    ) -> List[PendingConfirmation]:
        """Get all pending confirmations."""
        # First, expire any old ones
        now = datetime.utcnow()
        for conf in self._pending_confirmations.values():
            if conf.status == "pending":
                expires = datetime.fromisoformat(conf.expires_at)
                if now > expires:
                    conf.status = "expired"

        self._save_confirmations()

        pending = [
            c for c in self._pending_confirmations.values()
            if c.status == "pending"
        ]

        if project_id:
            pending = [c for c in pending if c.project_id == project_id]

        return pending

    def check_confirmation_approved(
        self,
        project_id: str,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if there's an approved confirmation for a project.

        Returns: (has_approval, confirmation_id)
        """
        for conf in self._pending_confirmations.values():
            if conf.project_id == project_id and conf.status == "confirmed":
                return True, conf.confirmation_id
        return False, None


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------
_enforcer: Optional[IntentContractEnforcer] = None


def get_contract_enforcer() -> IntentContractEnforcer:
    """Get the global contract enforcer instance."""
    global _enforcer
    if _enforcer is None:
        _enforcer = IntentContractEnforcer()
    return _enforcer


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
def evaluate_contract(
    project_id: str,
    project_name: str,
    current_intent: Dict[str, Any],
    action_description: str = "execution",
) -> ContractEvaluationResult:
    """Evaluate if an action is allowed under the project's contract."""
    return get_contract_enforcer().evaluate_contract(
        project_id=project_id,
        project_name=project_name,
        current_intent=current_intent,
        action_description=action_description,
    )


def check_can_execute(
    project_id: str,
    project_name: str,
    current_intent: Dict[str, Any],
) -> Tuple[bool, str, Optional[ContractEvaluationResult]]:
    """
    Quick check if execution is allowed.

    Returns: (can_proceed, reason, evaluation_result)
    """
    result = evaluate_contract(
        project_id=project_id,
        project_name=project_name,
        current_intent=current_intent,
    )

    if result.can_proceed:
        return True, result.summary, result
    else:
        reason = result.block_reason or result.user_prompt or result.summary
        return False, reason, result


def create_confirmation_request(
    evaluation_result: ContractEvaluationResult,
    timeout_hours: int = 24,
) -> PendingConfirmation:
    """Create a pending confirmation request."""
    return get_contract_enforcer().create_confirmation_request(
        evaluation_result=evaluation_result,
        timeout_hours=timeout_hours,
    )


def approve_confirmation(
    confirmation_id: str,
    approved_by: str,
    notes: Optional[str] = None,
) -> Tuple[bool, str]:
    """Approve a pending confirmation."""
    return get_contract_enforcer().approve_confirmation(
        confirmation_id=confirmation_id,
        approved_by=approved_by,
        notes=notes,
    )


def reject_confirmation(
    confirmation_id: str,
    rejected_by: str,
    notes: str,
) -> Tuple[bool, str]:
    """Reject a pending confirmation."""
    return get_contract_enforcer().reject_confirmation(
        confirmation_id=confirmation_id,
        rejected_by=rejected_by,
        notes=notes,
    )


def get_pending_confirmations(
    project_id: Optional[str] = None,
) -> List[PendingConfirmation]:
    """Get pending confirmations."""
    return get_contract_enforcer().get_pending_confirmations(project_id=project_id)
