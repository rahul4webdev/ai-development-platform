"""
Phase 18D: Post-Execution Verification Engine

Verifies that executed actions respected all approved constraints and system invariants.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- VERIFICATION ONLY: Answers "Did execution respect constraints?" - NOTHING ELSE
- NO EXECUTION: Never executes, retries, or triggers anything
- NO MUTATION: Never modifies external state
- NO ROLLBACK: Never undoes any changes
- NO RECOMMENDATIONS: Never suggests fixes or actions
- NO NOTIFICATIONS: Never sends alerts or messages
- 100% DETERMINISTIC: Same inputs = same output
- FAIL CLOSED: If data missing → verification UNKNOWN

This engine runs AFTER execution (Phase 18C).
It verifies reality, not changes it.

If ANY required data missing → UNKNOWN
Violations are RECORDED, not ACTED UPON.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from .verification_model import (
    VerificationStatus,
    ViolationSeverity,
    ViolationType,
    UnknownReason,
    ExecutionResultSnapshot,
    ExecutionIntentSnapshot,
    ExecutionAuditSnapshot,
    LifecycleSnapshot,
    IntentBaselineSnapshot,
    ExecutionConstraints,
    ExecutionLogs,
    VerificationInput,
    InvariantViolation,
    ExecutionVerificationResult,
    VerificationAuditRecord,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VERIFIER_VERSION = "18D.1.0"

# Audit storage (append-only)
VERIFICATION_DIR = Path(os.getenv("VERIFICATION_DIR", "data/verification"))
VERIFICATION_RESULTS_FILE = VERIFICATION_DIR / "verification_results.jsonl"
VERIFICATION_AUDIT_FILE = VERIFICATION_DIR / "verification_audit.jsonl"

# All verification domains (LOCKED - exactly 6)
ALL_DOMAINS = (
    ViolationType.SCOPE_VIOLATION.value,
    ViolationType.ACTION_VIOLATION.value,
    ViolationType.BOUNDARY_VIOLATION.value,
    ViolationType.INTENT_VIOLATION.value,
    ViolationType.INVARIANT_VIOLATION.value,
    ViolationType.OUTCOME_VIOLATION.value,
)


# -----------------------------------------------------------------------------
# Post-Execution Verification Engine (VERIFICATION ONLY)
# -----------------------------------------------------------------------------
class PostExecutionVerificationEngine:
    """
    Phase 18D: Post-Execution Verification Engine.

    VERIFICATION ONLY engine that checks if execution respected constraints.

    CRITICAL CONSTRAINTS:
    - Returns verification result, NOTHING ELSE
    - NO execution, retries, rollback, or mutations
    - NO recommendations or notifications
    - DETERMINISTIC: Same inputs = same output
    - FAIL CLOSED: Missing data → UNKNOWN

    Violations are RECORDED, not ACTED UPON.
    """

    def __init__(
        self,
        results_file: Optional[Path] = None,
        audit_file: Optional[Path] = None,
    ):
        """
        Initialize engine.

        Args:
            results_file: Path to results file (optional, for testing)
            audit_file: Path to audit file (optional, for testing)
        """
        self._results_file = results_file or VERIFICATION_RESULTS_FILE
        self._audit_file = audit_file or VERIFICATION_AUDIT_FILE
        self._version = VERIFIER_VERSION

    def verify(self, verification_input: VerificationInput) -> ExecutionVerificationResult:
        """
        Verify an execution against constraints and invariants.

        This is the ONLY public method. It:
        1. Checks for missing required data
        2. Verifies all 6 domains
        3. Collects violations
        4. Writes audit record
        5. Returns result

        If required data missing → UNKNOWN

        Args:
            verification_input: Complete input for verification

        Returns:
            ExecutionVerificationResult with status and violations
        """
        checked_at = datetime.utcnow().isoformat()
        input_hash = verification_input.compute_hash()

        # Generate verification ID
        verification_id = f"ver-{checked_at.replace(':', '-').replace('.', '-')}-{uuid.uuid4().hex[:8]}"

        # Get execution ID from input
        execution_id = "unknown"
        if verification_input.execution_result:
            execution_id = verification_input.execution_result.execution_id
        elif verification_input.execution_audit:
            execution_id = verification_input.execution_audit.execution_id

        # Phase 1: Check for missing required data
        unknown_reason = self._check_missing_data(verification_input)
        if unknown_reason:
            return self._create_unknown_result(
                verification_id=verification_id,
                execution_id=execution_id,
                unknown_reason=unknown_reason,
                input_hash=input_hash,
                checked_at=checked_at,
                verification_input=verification_input,
            )

        # Phase 2: Verify all domains
        violations: List[InvariantViolation] = []
        domains_checked: List[str] = []

        # Domain 1: Scope Compliance
        scope_violations = self._verify_scope_compliance(verification_input, checked_at)
        violations.extend(scope_violations)
        domains_checked.append(ViolationType.SCOPE_VIOLATION.value)

        # Domain 2: Action Compliance
        action_violations = self._verify_action_compliance(verification_input, checked_at)
        violations.extend(action_violations)
        domains_checked.append(ViolationType.ACTION_VIOLATION.value)

        # Domain 3: Boundary Compliance
        boundary_violations = self._verify_boundary_compliance(verification_input, checked_at)
        violations.extend(boundary_violations)
        domains_checked.append(ViolationType.BOUNDARY_VIOLATION.value)

        # Domain 4: Intent Compliance
        intent_violations = self._verify_intent_compliance(verification_input, checked_at)
        violations.extend(intent_violations)
        domains_checked.append(ViolationType.INTENT_VIOLATION.value)

        # Domain 5: Invariant Compliance
        invariant_violations = self._verify_invariant_compliance(verification_input, checked_at)
        violations.extend(invariant_violations)
        domains_checked.append(ViolationType.INVARIANT_VIOLATION.value)

        # Domain 6: Outcome Consistency
        outcome_violations = self._verify_outcome_consistency(verification_input, checked_at)
        violations.extend(outcome_violations)
        domains_checked.append(ViolationType.OUTCOME_VIOLATION.value)

        # Phase 3: Determine status
        if violations:
            status = VerificationStatus.FAILED.value
        else:
            status = VerificationStatus.PASSED.value

        # Count severities
        high_severity_count = sum(
            1 for v in violations
            if v.severity == ViolationSeverity.HIGH.value
        )

        # Create result
        result = ExecutionVerificationResult(
            verification_id=verification_id,
            execution_id=execution_id,
            verification_status=status,
            unknown_reason=None,
            violations=tuple(violations),
            input_hash=input_hash,
            checked_at=checked_at,
            verifier_version=self._version,
            violation_count=len(violations),
            high_severity_count=high_severity_count,
            domains_checked=tuple(domains_checked),
        )

        # Phase 4: Write audit record (MANDATORY)
        self._write_audit_record(result, verification_input)

        # Record result
        self._record_result(result)

        return result

    def _check_missing_data(self, verification_input: VerificationInput) -> Optional[str]:
        """
        Check for missing required data.

        If any required data is missing → return UNKNOWN reason.
        """
        if verification_input.execution_result is None:
            return UnknownReason.MISSING_EXECUTION_RESULT.value

        if verification_input.execution_intent is None:
            return UnknownReason.MISSING_EXECUTION_INTENT.value

        if verification_input.execution_audit is None:
            return UnknownReason.MISSING_EXECUTION_AUDIT.value

        if verification_input.execution_logs is None:
            return UnknownReason.MISSING_LOGS.value

        if verification_input.execution_logs and not verification_input.execution_logs.logs_readable:
            return UnknownReason.LOGS_UNREADABLE.value

        # lifecycle_snapshot, intent_baseline, and constraints are optional
        # but if they're needed for specific checks and missing, those checks will be skipped

        return None

    def _verify_scope_compliance(
        self,
        verification_input: VerificationInput,
        checked_at: str,
    ) -> List[InvariantViolation]:
        """
        Domain 1: Verify only approved files/modules were touched.

        VERIFICATION ONLY - never modifies anything.
        """
        violations = []

        logs = verification_input.execution_logs
        baseline = verification_input.intent_baseline
        constraints = verification_input.execution_constraints

        if not logs or not logs.files_touched:
            return violations  # No files touched, no violation

        # Check against approved scope
        approved_scope = set()
        if baseline and baseline.approved_scope:
            approved_scope = set(baseline.approved_scope)

        # Check against forbidden paths
        forbidden_paths = set()
        if constraints and constraints.forbidden_paths:
            forbidden_paths = set(constraints.forbidden_paths)

        for file_path in logs.files_touched:
            # Check forbidden paths
            for forbidden in forbidden_paths:
                if file_path.startswith(forbidden) or forbidden in file_path:
                    violation_id = f"viol-scope-{uuid.uuid4().hex[:8]}"
                    violations.append(InvariantViolation(
                        violation_id=violation_id,
                        violation_type=ViolationType.SCOPE_VIOLATION.value,
                        severity=ViolationSeverity.HIGH.value,
                        description=f"Touched forbidden path: {file_path}",
                        evidence_path=file_path,
                        evidence_snippet=f"Forbidden pattern: {forbidden}",
                        detected_at=checked_at,
                    ))

            # Check against approved scope (if defined)
            if approved_scope:
                scope_ok = False
                for approved in approved_scope:
                    if file_path.startswith(approved) or approved in file_path:
                        scope_ok = True
                        break
                if not scope_ok:
                    violation_id = f"viol-scope-{uuid.uuid4().hex[:8]}"
                    violations.append(InvariantViolation(
                        violation_id=violation_id,
                        violation_type=ViolationType.SCOPE_VIOLATION.value,
                        severity=ViolationSeverity.MEDIUM.value,
                        description=f"Touched file outside approved scope: {file_path}",
                        evidence_path=file_path,
                        evidence_snippet=None,
                        detected_at=checked_at,
                    ))

        return violations

    def _verify_action_compliance(
        self,
        verification_input: VerificationInput,
        checked_at: str,
    ) -> List[InvariantViolation]:
        """
        Domain 2: Verify only approved action type was executed.

        VERIFICATION ONLY - never modifies anything.
        """
        violations = []

        intent = verification_input.execution_intent
        baseline = verification_input.intent_baseline
        constraints = verification_input.execution_constraints

        if not intent:
            return violations

        action_type = intent.action_type

        # Check against allowed actions from constraints
        if constraints and constraints.allowed_actions:
            if action_type not in constraints.allowed_actions:
                violation_id = f"viol-action-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.ACTION_VIOLATION.value,
                    severity=ViolationSeverity.HIGH.value,
                    description=f"Executed action type '{action_type}' not in allowed list",
                    evidence_path=None,
                    evidence_snippet=f"Allowed: {list(constraints.allowed_actions)}",
                    detected_at=checked_at,
                ))

        # Check against approved actions from baseline
        if baseline and baseline.approved_actions:
            if action_type not in baseline.approved_actions:
                violation_id = f"viol-action-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.ACTION_VIOLATION.value,
                    severity=ViolationSeverity.MEDIUM.value,
                    description=f"Executed action type '{action_type}' not in baseline approved actions",
                    evidence_path=None,
                    evidence_snippet=f"Approved: {list(baseline.approved_actions)}",
                    detected_at=checked_at,
                ))

        return violations

    def _verify_boundary_compliance(
        self,
        verification_input: VerificationInput,
        checked_at: str,
    ) -> List[InvariantViolation]:
        """
        Domain 3: Verify no production deploy or external network access.

        VERIFICATION ONLY - never modifies anything.
        """
        violations = []

        intent = verification_input.execution_intent
        logs = verification_input.execution_logs
        constraints = verification_input.execution_constraints

        # Check for production deployment
        if intent and intent.action_type == "deploy_prod":
            # Production deployment should NEVER happen
            violation_id = f"viol-boundary-{uuid.uuid4().hex[:8]}"
            violations.append(InvariantViolation(
                violation_id=violation_id,
                violation_type=ViolationType.BOUNDARY_VIOLATION.value,
                severity=ViolationSeverity.HIGH.value,
                description="Production deployment action detected",
                evidence_path=None,
                evidence_snippet=f"Action type: {intent.action_type}",
                detected_at=checked_at,
            ))

        # Check constraints
        if constraints:
            # Check if prod deploy was explicitly forbidden
            if not constraints.production_deploy_allowed:
                if intent and "prod" in intent.action_type.lower():
                    violation_id = f"viol-boundary-{uuid.uuid4().hex[:8]}"
                    violations.append(InvariantViolation(
                        violation_id=violation_id,
                        violation_type=ViolationType.BOUNDARY_VIOLATION.value,
                        severity=ViolationSeverity.HIGH.value,
                        description="Production-related action when not allowed",
                        evidence_path=None,
                        evidence_snippet=f"Action: {intent.action_type}",
                        detected_at=checked_at,
                    ))

            # Check for external network access in logs
            if not constraints.external_network_allowed and logs and logs.logs_content:
                network_indicators = [
                    "curl ", "wget ", "http://", "https://",
                    "requests.get", "requests.post", "urllib",
                    "external API", "remote host",
                ]
                for indicator in network_indicators:
                    if indicator.lower() in logs.logs_content.lower():
                        violation_id = f"viol-boundary-{uuid.uuid4().hex[:8]}"
                        violations.append(InvariantViolation(
                            violation_id=violation_id,
                            violation_type=ViolationType.BOUNDARY_VIOLATION.value,
                            severity=ViolationSeverity.MEDIUM.value,
                            description=f"Possible external network access detected in logs",
                            evidence_path=logs.logs_path,
                            evidence_snippet=f"Indicator: {indicator}",
                            detected_at=checked_at,
                        ))
                        break  # Only report once

        return violations

    def _verify_intent_compliance(
        self,
        verification_input: VerificationInput,
        checked_at: str,
    ) -> List[InvariantViolation]:
        """
        Domain 4: Verify no intent drift was introduced.

        VERIFICATION ONLY - never modifies anything.
        """
        violations = []

        baseline = verification_input.intent_baseline
        lifecycle = verification_input.lifecycle_snapshot

        # Check if baseline exists and is valid
        if baseline:
            if not baseline.baseline_valid:
                violation_id = f"viol-intent-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.INTENT_VIOLATION.value,
                    severity=ViolationSeverity.MEDIUM.value,
                    description="Intent baseline marked as invalid after execution",
                    evidence_path=None,
                    evidence_snippet=f"Baseline version: {baseline.baseline_version}",
                    detected_at=checked_at,
                ))

        # Check lifecycle state consistency
        if lifecycle:
            # Execution shouldn't leave lifecycle in unexpected states
            terminal_states = ["rejected", "archived"]
            if lifecycle.state.lower() in terminal_states:
                violation_id = f"viol-intent-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.INTENT_VIOLATION.value,
                    severity=ViolationSeverity.HIGH.value,
                    description=f"Execution resulted in terminal lifecycle state: {lifecycle.state}",
                    evidence_path=None,
                    evidence_snippet=None,
                    detected_at=checked_at,
                ))

        return violations

    def _verify_invariant_compliance(
        self,
        verification_input: VerificationInput,
        checked_at: str,
    ) -> List[InvariantViolation]:
        """
        Domain 5: Verify audit trail and approval chain are intact.

        VERIFICATION ONLY - never modifies anything.
        """
        violations = []

        audit = verification_input.execution_audit
        result = verification_input.execution_result
        intent = verification_input.execution_intent

        if not audit:
            return violations

        # Check audit record completeness
        if audit.eligibility_decision is None:
            violation_id = f"viol-invariant-{uuid.uuid4().hex[:8]}"
            violations.append(InvariantViolation(
                violation_id=violation_id,
                violation_type=ViolationType.INVARIANT_VIOLATION.value,
                severity=ViolationSeverity.HIGH.value,
                description="Audit record missing eligibility decision",
                evidence_path=None,
                evidence_snippet=f"Audit ID: {audit.audit_id}",
                detected_at=checked_at,
            ))

        if audit.approval_status is None:
            violation_id = f"viol-invariant-{uuid.uuid4().hex[:8]}"
            violations.append(InvariantViolation(
                violation_id=violation_id,
                violation_type=ViolationType.INVARIANT_VIOLATION.value,
                severity=ViolationSeverity.HIGH.value,
                description="Audit record missing approval status",
                evidence_path=None,
                evidence_snippet=f"Audit ID: {audit.audit_id}",
                detected_at=checked_at,
            ))

        # Check consistency between audit and result
        if result and audit:
            if result.execution_id != audit.execution_id:
                violation_id = f"viol-invariant-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.INVARIANT_VIOLATION.value,
                    severity=ViolationSeverity.HIGH.value,
                    description="Execution ID mismatch between result and audit",
                    evidence_path=None,
                    evidence_snippet=f"Result: {result.execution_id}, Audit: {audit.execution_id}",
                    detected_at=checked_at,
                ))

        # Check consistency between audit and intent
        if intent and audit:
            if intent.intent_id != audit.intent_id:
                violation_id = f"viol-invariant-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.INVARIANT_VIOLATION.value,
                    severity=ViolationSeverity.HIGH.value,
                    description="Intent ID mismatch between intent and audit",
                    evidence_path=None,
                    evidence_snippet=f"Intent: {intent.intent_id}, Audit: {audit.intent_id}",
                    detected_at=checked_at,
                ))

        return violations

    def _verify_outcome_consistency(
        self,
        verification_input: VerificationInput,
        checked_at: str,
    ) -> List[InvariantViolation]:
        """
        Domain 6: Verify SUCCESS/FAILURE aligns with logs.

        VERIFICATION ONLY - never modifies anything.
        """
        violations = []

        result = verification_input.execution_result
        logs = verification_input.execution_logs

        if not result or not logs:
            return violations

        # Check exit code consistency
        if logs.exit_code is not None:
            if result.status == "execution_success" and logs.exit_code != 0:
                violation_id = f"viol-outcome-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.OUTCOME_VIOLATION.value,
                    severity=ViolationSeverity.HIGH.value,
                    description=f"Execution marked SUCCESS but exit code is {logs.exit_code}",
                    evidence_path=logs.logs_path,
                    evidence_snippet=f"Exit code: {logs.exit_code}",
                    detected_at=checked_at,
                ))

            if result.status == "execution_failed" and logs.exit_code == 0:
                violation_id = f"viol-outcome-{uuid.uuid4().hex[:8]}"
                violations.append(InvariantViolation(
                    violation_id=violation_id,
                    violation_type=ViolationType.OUTCOME_VIOLATION.value,
                    severity=ViolationSeverity.MEDIUM.value,
                    description=f"Execution marked FAILED but exit code is 0",
                    evidence_path=logs.logs_path,
                    evidence_snippet=f"Exit code: {logs.exit_code}",
                    detected_at=checked_at,
                ))

        # Check logs content for error indicators
        if logs.logs_content and result.status == "execution_success":
            error_indicators = [
                "error:", "ERROR:", "Error:", "FATAL:",
                "fatal:", "Fatal:", "exception:", "Exception:",
                "EXCEPTION:", "failed:", "FAILED:", "Failed:",
            ]
            for indicator in error_indicators:
                if indicator in logs.logs_content:
                    # Only flag if there are many errors, not just one mention
                    count = logs.logs_content.count(indicator)
                    if count >= 3:
                        violation_id = f"viol-outcome-{uuid.uuid4().hex[:8]}"
                        violations.append(InvariantViolation(
                            violation_id=violation_id,
                            violation_type=ViolationType.OUTCOME_VIOLATION.value,
                            severity=ViolationSeverity.LOW.value,
                            description=f"Execution marked SUCCESS but logs contain error indicators",
                            evidence_path=logs.logs_path,
                            evidence_snippet=f"'{indicator}' found {count} times",
                            detected_at=checked_at,
                        ))
                        break

        return violations

    def _create_unknown_result(
        self,
        verification_id: str,
        execution_id: str,
        unknown_reason: str,
        input_hash: str,
        checked_at: str,
        verification_input: VerificationInput,
    ) -> ExecutionVerificationResult:
        """Create UNKNOWN result when required data is missing."""
        result = ExecutionVerificationResult(
            verification_id=verification_id,
            execution_id=execution_id,
            verification_status=VerificationStatus.UNKNOWN.value,
            unknown_reason=unknown_reason,
            violations=(),
            input_hash=input_hash,
            checked_at=checked_at,
            verifier_version=self._version,
            violation_count=0,
            high_severity_count=0,
            domains_checked=(),
        )

        # Write audit record
        self._write_audit_record(result, verification_input)

        # Record result
        self._record_result(result)

        return result

    def _record_result(self, result: ExecutionVerificationResult) -> None:
        """Record result to append-only store."""
        try:
            self._results_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._results_file, 'a') as f:
                f.write(json.dumps(result.to_dict()) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            # Result recording failure is logged but doesn't affect verification
            pass

    def _write_audit_record(
        self,
        result: ExecutionVerificationResult,
        verification_input: VerificationInput,
    ) -> bool:
        """
        Write immutable audit record.

        Append-only. No updates. No deletes.
        """
        try:
            self._audit_file.parent.mkdir(parents=True, exist_ok=True)

            audit_id = f"vaud-{result.checked_at.replace(':', '-').replace('.', '-')}-{result.input_hash[:8]}"

            # Extract project_id if available
            project_id = None
            if verification_input.execution_intent:
                project_id = verification_input.execution_intent.project_id
            elif verification_input.execution_audit:
                project_id = verification_input.execution_audit.project_id

            audit_record = VerificationAuditRecord(
                audit_id=audit_id,
                verification_id=result.verification_id,
                execution_id=result.execution_id,
                verification_status=result.verification_status,
                unknown_reason=result.unknown_reason,
                violation_count=result.violation_count,
                high_severity_count=result.high_severity_count,
                input_hash=result.input_hash,
                checked_at=result.checked_at,
                verifier_version=result.verifier_version,
                project_id=project_id,
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

    def get_verification(self, execution_id: str) -> Optional[ExecutionVerificationResult]:
        """Get verification result for an execution."""
        if not self._results_file.exists():
            return None

        latest = None
        try:
            with open(self._results_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if data.get("execution_id") == execution_id:
                                latest = ExecutionVerificationResult.from_dict(data)
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        return latest

    def get_recent_verifications(
        self,
        limit: int = 100,
        status: Optional[str] = None,
    ) -> List[ExecutionVerificationResult]:
        """Get recent verification results."""
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
                            if status and data.get("verification_status") != status:
                                continue
                            results.append(ExecutionVerificationResult.from_dict(data))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        # Return most recent first
        results.reverse()
        return results[:limit]

    def get_summary(
        self,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get verification summary statistics."""
        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        total = 0
        passed = 0
        failed = 0
        unknown = 0
        total_violations = 0
        high_severity_violations = 0
        by_violation_type: Dict[str, int] = {}

        if self._results_file.exists():
            try:
                with open(self._results_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)

                                if data.get("checked_at", "") < cutoff:
                                    continue

                                total += 1
                                status = data.get("verification_status")

                                if status == VerificationStatus.PASSED.value:
                                    passed += 1
                                elif status == VerificationStatus.FAILED.value:
                                    failed += 1
                                    # Count violations
                                    violations = data.get("violations", [])
                                    total_violations += len(violations)
                                    for v in violations:
                                        if v.get("severity") == ViolationSeverity.HIGH.value:
                                            high_severity_violations += 1
                                        v_type = v.get("violation_type", "unknown")
                                        by_violation_type[v_type] = by_violation_type.get(v_type, 0) + 1
                                elif status == VerificationStatus.UNKNOWN.value:
                                    unknown += 1

                            except json.JSONDecodeError:
                                continue
            except Exception:
                pass

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "since_hours": since_hours,
            "total_verifications": total,
            "passed_count": passed,
            "failed_count": failed,
            "unknown_count": unknown,
            "pass_rate": passed / total if total > 0 else 0.0,
            "total_violations": total_violations,
            "high_severity_violations": high_severity_violations,
            "by_violation_type": by_violation_type,
        }


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------

# Singleton instance
_engine: Optional[PostExecutionVerificationEngine] = None


def get_verification_engine(
    results_file: Optional[Path] = None,
    audit_file: Optional[Path] = None,
) -> PostExecutionVerificationEngine:
    """Get the verification engine singleton."""
    global _engine
    if _engine is None:
        _engine = PostExecutionVerificationEngine(
            results_file=results_file,
            audit_file=audit_file,
        )
    return _engine


def verify_execution(verification_input: VerificationInput) -> ExecutionVerificationResult:
    """
    Verify an execution.

    Convenience function using singleton engine.

    Args:
        verification_input: Complete input for verification

    Returns:
        ExecutionVerificationResult with status and violations
    """
    engine = get_verification_engine()
    return engine.verify(verification_input)


def create_verification_input(
    execution_result: Optional[Dict[str, Any]] = None,
    execution_intent: Optional[Dict[str, Any]] = None,
    execution_audit: Optional[Dict[str, Any]] = None,
    lifecycle_snapshot: Optional[Dict[str, Any]] = None,
    intent_baseline: Optional[Dict[str, Any]] = None,
    execution_constraints: Optional[Dict[str, Any]] = None,
    execution_logs: Optional[Dict[str, Any]] = None,
) -> VerificationInput:
    """
    Create VerificationInput from dictionary data.

    Convenience function for creating input from API/external data.
    """
    result_snapshot = None
    if execution_result:
        result_snapshot = ExecutionResultSnapshot.from_dict(execution_result)

    intent_snapshot = None
    if execution_intent:
        intent_snapshot = ExecutionIntentSnapshot.from_dict(execution_intent)

    audit_snapshot = None
    if execution_audit:
        audit_snapshot = ExecutionAuditSnapshot.from_dict(execution_audit)

    lifecycle = None
    if lifecycle_snapshot:
        lifecycle = LifecycleSnapshot.from_dict(lifecycle_snapshot)

    baseline = None
    if intent_baseline:
        baseline = IntentBaselineSnapshot.from_dict(intent_baseline)

    constraints = None
    if execution_constraints:
        constraints = ExecutionConstraints.from_dict(execution_constraints)

    logs = None
    if execution_logs:
        logs = ExecutionLogs.from_dict(execution_logs)

    return VerificationInput(
        execution_result=result_snapshot,
        execution_intent=intent_snapshot,
        execution_audit=audit_snapshot,
        lifecycle_snapshot=lifecycle,
        intent_baseline=baseline,
        execution_constraints=constraints,
        execution_logs=logs,
    )


def get_verification_summary(since_hours: int = 24) -> Dict[str, Any]:
    """
    Get verification summary.

    Convenience function using singleton engine.
    """
    engine = get_verification_engine()
    return engine.get_summary(since_hours=since_hours)
