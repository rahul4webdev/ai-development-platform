"""
Phase 18A: Automation Eligibility Engine

DECISION-ONLY engine that answers: "Is automation allowed in this situation?"

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- DECISION-ONLY: Returns eligibility decision, NOTHING ELSE
- NO EXECUTION: Never executes, schedules, triggers, or mutates
- NO RECOMMENDATIONS: Does not suggest actions
- NO PLANNING: Does not plan what to do next
- NO SIDE EFFECTS: No disk writes, no state changes, no notifications
- DETERMINISTIC: Same inputs ALWAYS produce same output (no ML, no heuristics)
- AUDIT REQUIRED: Every evaluation emits immutable audit record

This engine sits AFTER recommendations (Phase 17C) and BEFORE any automation.
It is a GATE, not an ACTOR.

If ANY hard-stop rule matches → AUTOMATION_FORBIDDEN
If audit write fails → AUTOMATION_FORBIDDEN
If ANY input is missing → AUTOMATION_FORBIDDEN
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, FrozenSet, Tuple

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ENGINE_VERSION = "18A.1.0"

# Audit storage (append-only)
AUDIT_DIR = Path(os.getenv("AUDIT_DIR", "audit"))
ELIGIBILITY_AUDIT_FILE = AUDIT_DIR / "eligibility_decisions.jsonl"


# -----------------------------------------------------------------------------
# Eligibility Decision Enum (LOCKED - EXACTLY 3 VALUES)
# -----------------------------------------------------------------------------
class EligibilityDecision(str, Enum):
    """
    Automation eligibility decision.

    This enum is LOCKED - do not add values without explicit approval.
    EXACTLY 3 values, no more, no less.
    """
    AUTOMATION_FORBIDDEN = "automation_forbidden"
    AUTOMATION_ALLOWED_WITH_APPROVAL = "automation_allowed_with_approval"
    AUTOMATION_ALLOWED_LIMITED = "automation_allowed_limited"


# -----------------------------------------------------------------------------
# Limited Actions Enum (LOCKED)
# -----------------------------------------------------------------------------
class LimitedAction(str, Enum):
    """
    Actions allowed under AUTOMATION_ALLOWED_LIMITED.

    ONLY these two actions are permitted under LIMITED automation.
    No code writes, no commits, no deployments.
    """
    RUN_TESTS = "run_tests"
    UPDATE_DOCS = "update_docs"


# -----------------------------------------------------------------------------
# Hard Stop Rule IDs (LOCKED)
# -----------------------------------------------------------------------------
class HardStopRule(str, Enum):
    """
    Hard stop rule identifiers.

    If ANY of these rules match, result is AUTOMATION_FORBIDDEN.
    """
    # Input rules
    MISSING_RECOMMENDATION = "missing_recommendation"
    MISSING_DRIFT_EVALUATION = "missing_drift_evaluation"
    MISSING_INCIDENT_SUMMARY = "missing_incident_summary"
    MISSING_LIFECYCLE_STATE = "missing_lifecycle_state"
    MISSING_EXECUTION_GATE = "missing_execution_gate"
    MISSING_ENVIRONMENT = "missing_environment"
    MISSING_INTENT_BASELINE = "missing_intent_baseline"

    # Drift rules
    DRIFT_LEVEL_HIGH = "drift_level_high"
    DRIFT_LEVEL_CRITICAL = "drift_level_critical"
    ARCHITECTURE_CHANGE_DETECTED = "architecture_change_detected"
    DATABASE_CHANGE_DETECTED = "database_change_detected"
    INTENT_BASELINE_INVALID = "intent_baseline_invalid"

    # Incident rules
    INCIDENT_SEVERITY_CRITICAL = "incident_severity_critical"
    INCIDENT_TYPE_SECURITY = "incident_type_security"
    INCIDENT_STATE_UNKNOWN = "incident_state_unknown"

    # Signal rules
    SIGNAL_SEVERITY_UNKNOWN = "signal_severity_unknown"
    MISSING_RUNTIME_INTELLIGENCE = "missing_runtime_intelligence"

    # Environment rules
    PRODUCTION_WITH_MEDIUM_DRIFT = "production_with_medium_drift"
    PRODUCTION_WITH_MEDIUM_INCIDENT = "production_with_medium_incident"

    # Governance rules
    EXECUTION_GATE_DENIED = "execution_gate_denied"
    GOVERNANCE_DOCUMENTS_MISSING = "governance_documents_missing"
    AUDIT_SUBSYSTEM_UNAVAILABLE = "audit_subsystem_unavailable"


# -----------------------------------------------------------------------------
# Environment Enum (LOCKED)
# -----------------------------------------------------------------------------
class Environment(str, Enum):
    """Environment types."""
    TEST = "test"
    PRODUCTION = "production"


# -----------------------------------------------------------------------------
# Input Data Classes (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RecommendationInput:
    """
    Input from Phase 17C Recommendation.

    Immutable snapshot of recommendation data for eligibility evaluation.
    """
    recommendation_id: str
    recommendation_type: str
    severity: str
    approval_required: str
    status: str
    project_id: Optional[str]
    confidence: float

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")


@dataclass(frozen=True)
class DriftEvaluationInput:
    """
    Input from Phase 16F Drift Detection.

    Immutable snapshot of drift evaluation data.
    """
    drift_level: str  # "none", "low", "medium", "high", "critical"
    architecture_change: bool
    database_change: bool
    drift_dimensions: FrozenSet[str]  # Which dimensions have drift


@dataclass(frozen=True)
class IncidentSummaryInput:
    """
    Input from Phase 17B Incident Classification.

    Immutable snapshot of incident summary data.
    """
    total_incidents: int
    open_count: int
    critical_count: int
    security_count: int
    unknown_count: int
    max_severity: str  # "info", "low", "medium", "high", "critical", "unknown"


@dataclass(frozen=True)
class LifecycleStateInput:
    """
    Input from Lifecycle Engine.

    Immutable snapshot of lifecycle state.
    """
    state: str
    project_id: str
    is_active: bool


@dataclass(frozen=True)
class ExecutionGateInput:
    """
    Input from Phase 15.6 Execution Gate.

    Immutable snapshot of gate constraints.
    """
    gate_allows_action: bool
    required_role: Optional[str]
    gate_denial_reason: Optional[str]


@dataclass(frozen=True)
class IntentBaselineInput:
    """
    Input from Intent Baseline Manager.

    Immutable snapshot of intent baseline presence.
    """
    baseline_exists: bool
    baseline_valid: bool
    baseline_version: Optional[str]


@dataclass(frozen=True)
class RuntimeIntelligenceInput:
    """
    Input from Phase 17A Runtime Intelligence.

    Immutable snapshot of signal health.
    """
    signals_available: bool
    unknown_severity_count: int
    intelligence_window_present: bool


# -----------------------------------------------------------------------------
# Eligibility Evaluation Input (Frozen - All Inputs Combined)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityInput:
    """
    Complete input for eligibility evaluation.

    ALL fields are MANDATORY. If any is None → AUTOMATION_FORBIDDEN.
    """
    recommendation: Optional[RecommendationInput]
    drift_evaluation: Optional[DriftEvaluationInput]
    incident_summary: Optional[IncidentSummaryInput]
    lifecycle_state: Optional[LifecycleStateInput]
    execution_gate: Optional[ExecutionGateInput]
    environment: Optional[str]  # "test" or "production"
    intent_baseline: Optional[IntentBaselineInput]
    runtime_intelligence: Optional[RuntimeIntelligenceInput]

    def compute_hash(self) -> str:
        """Compute deterministic hash of all inputs."""
        # Serialize to JSON-compatible dict
        data = {
            "recommendation": self._serialize_recommendation(),
            "drift_evaluation": self._serialize_drift(),
            "incident_summary": self._serialize_incidents(),
            "lifecycle_state": self._serialize_lifecycle(),
            "execution_gate": self._serialize_gate(),
            "environment": self.environment,
            "intent_baseline": self._serialize_baseline(),
            "runtime_intelligence": self._serialize_runtime(),
        }
        # Deterministic JSON serialization
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

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

    def _serialize_drift(self) -> Optional[Dict[str, Any]]:
        if self.drift_evaluation is None:
            return None
        return {
            "drift_level": self.drift_evaluation.drift_level,
            "architecture_change": self.drift_evaluation.architecture_change,
            "database_change": self.drift_evaluation.database_change,
            "drift_dimensions": sorted(self.drift_evaluation.drift_dimensions),
        }

    def _serialize_incidents(self) -> Optional[Dict[str, Any]]:
        if self.incident_summary is None:
            return None
        return {
            "total_incidents": self.incident_summary.total_incidents,
            "open_count": self.incident_summary.open_count,
            "critical_count": self.incident_summary.critical_count,
            "security_count": self.incident_summary.security_count,
            "unknown_count": self.incident_summary.unknown_count,
            "max_severity": self.incident_summary.max_severity,
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

    def _serialize_baseline(self) -> Optional[Dict[str, Any]]:
        if self.intent_baseline is None:
            return None
        return {
            "baseline_exists": self.intent_baseline.baseline_exists,
            "baseline_valid": self.intent_baseline.baseline_valid,
            "baseline_version": self.intent_baseline.baseline_version,
        }

    def _serialize_runtime(self) -> Optional[Dict[str, Any]]:
        if self.runtime_intelligence is None:
            return None
        return {
            "signals_available": self.runtime_intelligence.signals_available,
            "unknown_severity_count": self.runtime_intelligence.unknown_severity_count,
            "intelligence_window_present": self.runtime_intelligence.intelligence_window_present,
        }


# -----------------------------------------------------------------------------
# Eligibility Result (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityResult:
    """
    Result of eligibility evaluation.

    Immutable record of the decision.
    """
    decision: str  # EligibilityDecision value
    matched_rules: Tuple[str, ...]  # Tuple of HardStopRule values that matched
    input_hash: str  # Hash of input for audit
    timestamp: str  # ISO format
    engine_version: str
    allowed_actions: Tuple[str, ...]  # Empty unless LIMITED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "matched_rules": list(self.matched_rules),
            "input_hash": self.input_hash,
            "timestamp": self.timestamp,
            "engine_version": self.engine_version,
            "allowed_actions": list(self.allowed_actions),
        }


# -----------------------------------------------------------------------------
# Audit Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class EligibilityAuditRecord:
    """
    Immutable audit record for eligibility decisions.

    Append-only. No updates. No deletes.
    """
    audit_id: str
    input_hash: str
    decision: str
    matched_rules: Tuple[str, ...]
    timestamp: str
    engine_version: str
    environment: Optional[str]
    project_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "input_hash": self.input_hash,
            "decision": self.decision,
            "matched_rules": list(self.matched_rules),
            "timestamp": self.timestamp,
            "engine_version": self.engine_version,
            "environment": self.environment,
            "project_id": self.project_id,
        }


# -----------------------------------------------------------------------------
# Automation Eligibility Engine (DECISION-ONLY)
# -----------------------------------------------------------------------------
class AutomationEligibilityEngine:
    """
    Phase 18A: Automation Eligibility Engine.

    DECISION-ONLY engine that evaluates whether automation is allowed.

    CRITICAL CONSTRAINTS:
    - Returns eligibility decision, NOTHING ELSE
    - NO execution, scheduling, triggering, or mutation
    - NO recommendations or planning
    - DETERMINISTIC: Same inputs = same output
    - AUDIT REQUIRED: Every evaluation emits immutable audit record

    If ANY hard-stop rule matches → AUTOMATION_FORBIDDEN
    If audit write fails → AUTOMATION_FORBIDDEN
    If ANY input is missing → AUTOMATION_FORBIDDEN
    """

    def __init__(self, audit_file: Optional[Path] = None):
        """
        Initialize engine.

        Args:
            audit_file: Path to audit file (optional, for testing)
        """
        self._audit_file = audit_file or ELIGIBILITY_AUDIT_FILE
        self._version = ENGINE_VERSION

    def evaluate(self, eligibility_input: EligibilityInput) -> EligibilityResult:
        """
        Evaluate automation eligibility.

        This is the ONLY public method. It:
        1. Checks all hard-stop rules
        2. Computes decision
        3. Writes audit record
        4. Returns result

        If audit write fails → AUTOMATION_FORBIDDEN

        Args:
            eligibility_input: Complete input for evaluation

        Returns:
            EligibilityResult with decision and matched rules
        """
        timestamp = datetime.utcnow().isoformat()
        input_hash = eligibility_input.compute_hash()
        matched_rules: List[str] = []

        # Phase 1: Check for missing inputs (ALL are mandatory)
        missing_rules = self._check_missing_inputs(eligibility_input)
        matched_rules.extend(missing_rules)

        # If any input is missing, short-circuit to FORBIDDEN
        if missing_rules:
            return self._create_forbidden_result(
                matched_rules=tuple(matched_rules),
                input_hash=input_hash,
                timestamp=timestamp,
                eligibility_input=eligibility_input,
            )

        # Phase 2: Check all hard-stop rules
        hard_stop_rules = self._check_hard_stop_rules(eligibility_input)
        matched_rules.extend(hard_stop_rules)

        # Phase 3: Determine decision
        if matched_rules:
            result = self._create_forbidden_result(
                matched_rules=tuple(matched_rules),
                input_hash=input_hash,
                timestamp=timestamp,
                eligibility_input=eligibility_input,
            )
        else:
            result = self._determine_allowed_decision(
                eligibility_input=eligibility_input,
                input_hash=input_hash,
                timestamp=timestamp,
            )

        # Phase 4: Write audit record (MANDATORY)
        audit_success = self._write_audit_record(result, eligibility_input)

        # If audit fails, override to FORBIDDEN
        if not audit_success:
            return self._create_forbidden_result(
                matched_rules=(HardStopRule.AUDIT_SUBSYSTEM_UNAVAILABLE.value,),
                input_hash=input_hash,
                timestamp=timestamp,
                eligibility_input=eligibility_input,
            )

        return result

    def _check_missing_inputs(self, eligibility_input: EligibilityInput) -> List[str]:
        """Check for missing mandatory inputs."""
        missing = []

        if eligibility_input.recommendation is None:
            missing.append(HardStopRule.MISSING_RECOMMENDATION.value)

        if eligibility_input.drift_evaluation is None:
            missing.append(HardStopRule.MISSING_DRIFT_EVALUATION.value)

        if eligibility_input.incident_summary is None:
            missing.append(HardStopRule.MISSING_INCIDENT_SUMMARY.value)

        if eligibility_input.lifecycle_state is None:
            missing.append(HardStopRule.MISSING_LIFECYCLE_STATE.value)

        if eligibility_input.execution_gate is None:
            missing.append(HardStopRule.MISSING_EXECUTION_GATE.value)

        if eligibility_input.environment is None:
            missing.append(HardStopRule.MISSING_ENVIRONMENT.value)

        if eligibility_input.intent_baseline is None:
            missing.append(HardStopRule.MISSING_INTENT_BASELINE.value)

        return missing

    def _check_hard_stop_rules(self, eligibility_input: EligibilityInput) -> List[str]:
        """
        Check all hard-stop rules.

        If ANY rule matches → AUTOMATION_FORBIDDEN
        """
        matched = []

        # Drift rules
        matched.extend(self._check_drift_rules(eligibility_input.drift_evaluation))

        # Intent baseline rules
        matched.extend(self._check_baseline_rules(eligibility_input.intent_baseline))

        # Incident rules
        matched.extend(self._check_incident_rules(eligibility_input.incident_summary))

        # Signal/Runtime intelligence rules
        matched.extend(self._check_runtime_rules(eligibility_input.runtime_intelligence))

        # Environment rules (production constraints)
        matched.extend(self._check_environment_rules(
            eligibility_input.environment,
            eligibility_input.drift_evaluation,
            eligibility_input.incident_summary,
        ))

        # Governance rules
        matched.extend(self._check_governance_rules(eligibility_input.execution_gate))

        return matched

    def _check_drift_rules(self, drift: Optional[DriftEvaluationInput]) -> List[str]:
        """Check drift-related hard-stop rules."""
        if drift is None:
            return []  # Already handled by missing input check

        matched = []

        # Drift level HIGH or CRITICAL
        if drift.drift_level == "high":
            matched.append(HardStopRule.DRIFT_LEVEL_HIGH.value)
        elif drift.drift_level == "critical":
            matched.append(HardStopRule.DRIFT_LEVEL_CRITICAL.value)

        # Architecture change detected
        if drift.architecture_change:
            matched.append(HardStopRule.ARCHITECTURE_CHANGE_DETECTED.value)

        # Database change detected
        if drift.database_change:
            matched.append(HardStopRule.DATABASE_CHANGE_DETECTED.value)

        return matched

    def _check_baseline_rules(self, baseline: Optional[IntentBaselineInput]) -> List[str]:
        """Check intent baseline hard-stop rules."""
        if baseline is None:
            return []  # Already handled by missing input check

        matched = []

        # Baseline missing or invalid
        if not baseline.baseline_exists:
            matched.append(HardStopRule.MISSING_INTENT_BASELINE.value)
        elif not baseline.baseline_valid:
            matched.append(HardStopRule.INTENT_BASELINE_INVALID.value)

        return matched

    def _check_incident_rules(self, incidents: Optional[IncidentSummaryInput]) -> List[str]:
        """Check incident-related hard-stop rules."""
        if incidents is None:
            return []  # Already handled by missing input check

        matched = []

        # Critical incidents
        if incidents.critical_count > 0 or incidents.max_severity == "critical":
            matched.append(HardStopRule.INCIDENT_SEVERITY_CRITICAL.value)

        # Security incidents
        if incidents.security_count > 0:
            matched.append(HardStopRule.INCIDENT_TYPE_SECURITY.value)

        # UNKNOWN state incidents
        if incidents.unknown_count > 0 or incidents.max_severity == "unknown":
            matched.append(HardStopRule.INCIDENT_STATE_UNKNOWN.value)

        return matched

    def _check_runtime_rules(self, runtime: Optional[RuntimeIntelligenceInput]) -> List[str]:
        """Check runtime intelligence hard-stop rules."""
        if runtime is None:
            # Runtime intelligence is checked via its own input
            matched = [HardStopRule.MISSING_RUNTIME_INTELLIGENCE.value]
            return matched

        matched = []

        # Any signal with UNKNOWN severity
        if runtime.unknown_severity_count > 0:
            matched.append(HardStopRule.SIGNAL_SEVERITY_UNKNOWN.value)

        # Missing runtime intelligence window
        if not runtime.intelligence_window_present:
            matched.append(HardStopRule.MISSING_RUNTIME_INTELLIGENCE.value)

        return matched

    def _check_environment_rules(
        self,
        environment: Optional[str],
        drift: Optional[DriftEvaluationInput],
        incidents: Optional[IncidentSummaryInput],
    ) -> List[str]:
        """Check environment-specific hard-stop rules."""
        if environment is None or drift is None or incidents is None:
            return []  # Already handled by missing input checks

        matched = []

        # Production-specific constraints
        if environment == Environment.PRODUCTION.value:
            # Production + MEDIUM drift
            if drift.drift_level in ["medium", "high", "critical"]:
                matched.append(HardStopRule.PRODUCTION_WITH_MEDIUM_DRIFT.value)

            # Production + MEDIUM incident severity
            if incidents.max_severity in ["medium", "high", "critical"]:
                matched.append(HardStopRule.PRODUCTION_WITH_MEDIUM_INCIDENT.value)

        return matched

    def _check_governance_rules(self, gate: Optional[ExecutionGateInput]) -> List[str]:
        """Check governance-related hard-stop rules."""
        if gate is None:
            return []  # Already handled by missing input check

        matched = []

        # Execution gate denies action
        if not gate.gate_allows_action:
            matched.append(HardStopRule.EXECUTION_GATE_DENIED.value)

        return matched

    def _determine_allowed_decision(
        self,
        eligibility_input: EligibilityInput,
        input_hash: str,
        timestamp: str,
    ) -> EligibilityResult:
        """
        Determine which ALLOWED decision applies.

        Called ONLY when no hard-stop rules matched.
        """
        recommendation = eligibility_input.recommendation

        # If recommendation requires explicit approval → WITH_APPROVAL
        if recommendation and recommendation.approval_required == "explicit_approval_required":
            return EligibilityResult(
                decision=EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value,
                matched_rules=(),
                input_hash=input_hash,
                timestamp=timestamp,
                engine_version=self._version,
                allowed_actions=(),  # Actions require approval first
            )

        # If recommendation requires confirmation → WITH_APPROVAL
        if recommendation and recommendation.approval_required == "confirmation_required":
            return EligibilityResult(
                decision=EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value,
                matched_rules=(),
                input_hash=input_hash,
                timestamp=timestamp,
                engine_version=self._version,
                allowed_actions=(),
            )

        # If no approval required → LIMITED (only RUN_TESTS, UPDATE_DOCS)
        return EligibilityResult(
            decision=EligibilityDecision.AUTOMATION_ALLOWED_LIMITED.value,
            matched_rules=(),
            input_hash=input_hash,
            timestamp=timestamp,
            engine_version=self._version,
            allowed_actions=(
                LimitedAction.RUN_TESTS.value,
                LimitedAction.UPDATE_DOCS.value,
            ),
        )

    def _create_forbidden_result(
        self,
        matched_rules: Tuple[str, ...],
        input_hash: str,
        timestamp: str,
        eligibility_input: EligibilityInput,
    ) -> EligibilityResult:
        """Create AUTOMATION_FORBIDDEN result."""
        return EligibilityResult(
            decision=EligibilityDecision.AUTOMATION_FORBIDDEN.value,
            matched_rules=matched_rules,
            input_hash=input_hash,
            timestamp=timestamp,
            engine_version=self._version,
            allowed_actions=(),
        )

    def _write_audit_record(
        self,
        result: EligibilityResult,
        eligibility_input: EligibilityInput,
    ) -> bool:
        """
        Write immutable audit record.

        Append-only. No updates. No deletes.
        If write fails → return False (triggers AUTOMATION_FORBIDDEN).
        """
        try:
            # Ensure audit directory exists
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)

            # Generate audit ID
            audit_id = f"elig-{result.timestamp.replace(':', '-').replace('.', '-')}-{result.input_hash[:8]}"

            # Extract project_id if available
            project_id = None
            if eligibility_input.recommendation:
                project_id = eligibility_input.recommendation.project_id
            elif eligibility_input.lifecycle_state:
                project_id = eligibility_input.lifecycle_state.project_id

            # Create audit record
            audit_record = EligibilityAuditRecord(
                audit_id=audit_id,
                input_hash=result.input_hash,
                decision=result.decision,
                matched_rules=result.matched_rules,
                timestamp=result.timestamp,
                engine_version=result.engine_version,
                environment=eligibility_input.environment,
                project_id=project_id,
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
_engine: Optional[AutomationEligibilityEngine] = None


def get_eligibility_engine(audit_file: Optional[Path] = None) -> AutomationEligibilityEngine:
    """Get the eligibility engine singleton."""
    global _engine
    if _engine is None:
        _engine = AutomationEligibilityEngine(audit_file=audit_file)
    return _engine


def evaluate_eligibility(eligibility_input: EligibilityInput) -> EligibilityResult:
    """
    Evaluate automation eligibility.

    Convenience function that uses singleton engine.

    Args:
        eligibility_input: Complete input for evaluation

    Returns:
        EligibilityResult with decision and matched rules
    """
    engine = get_eligibility_engine()
    return engine.evaluate(eligibility_input)


def create_eligibility_input(
    recommendation: Optional[Dict[str, Any]] = None,
    drift_evaluation: Optional[Dict[str, Any]] = None,
    incident_summary: Optional[Dict[str, Any]] = None,
    lifecycle_state: Optional[Dict[str, Any]] = None,
    execution_gate: Optional[Dict[str, Any]] = None,
    environment: Optional[str] = None,
    intent_baseline: Optional[Dict[str, Any]] = None,
    runtime_intelligence: Optional[Dict[str, Any]] = None,
) -> EligibilityInput:
    """
    Create EligibilityInput from dictionary data.

    Convenience function for creating input from API/external data.
    """
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

    drift_input = None
    if drift_evaluation:
        drift_input = DriftEvaluationInput(
            drift_level=drift_evaluation.get("drift_level", "none"),
            architecture_change=drift_evaluation.get("architecture_change", False),
            database_change=drift_evaluation.get("database_change", False),
            drift_dimensions=frozenset(drift_evaluation.get("drift_dimensions", [])),
        )

    incident_input = None
    if incident_summary:
        incident_input = IncidentSummaryInput(
            total_incidents=incident_summary.get("total_incidents", 0),
            open_count=incident_summary.get("open_count", 0),
            critical_count=incident_summary.get("critical_count", 0),
            security_count=incident_summary.get("security_count", 0),
            unknown_count=incident_summary.get("unknown_count", 0),
            max_severity=incident_summary.get("max_severity", "info"),
        )

    lifecycle_input = None
    if lifecycle_state:
        lifecycle_input = LifecycleStateInput(
            state=lifecycle_state.get("state", ""),
            project_id=lifecycle_state.get("project_id", ""),
            is_active=lifecycle_state.get("is_active", False),
        )

    gate_input = None
    if execution_gate:
        gate_input = ExecutionGateInput(
            gate_allows_action=execution_gate.get("gate_allows_action", False),
            required_role=execution_gate.get("required_role"),
            gate_denial_reason=execution_gate.get("gate_denial_reason"),
        )

    baseline_input = None
    if intent_baseline:
        baseline_input = IntentBaselineInput(
            baseline_exists=intent_baseline.get("baseline_exists", False),
            baseline_valid=intent_baseline.get("baseline_valid", False),
            baseline_version=intent_baseline.get("baseline_version"),
        )

    runtime_input = None
    if runtime_intelligence:
        runtime_input = RuntimeIntelligenceInput(
            signals_available=runtime_intelligence.get("signals_available", False),
            unknown_severity_count=runtime_intelligence.get("unknown_severity_count", 0),
            intelligence_window_present=runtime_intelligence.get("intelligence_window_present", False),
        )

    return EligibilityInput(
        recommendation=rec_input,
        drift_evaluation=drift_input,
        incident_summary=incident_input,
        lifecycle_state=lifecycle_input,
        execution_gate=gate_input,
        environment=environment,
        intent_baseline=baseline_input,
        runtime_intelligence=runtime_input,
    )
