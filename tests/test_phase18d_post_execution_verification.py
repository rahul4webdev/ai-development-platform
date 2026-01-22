"""
Phase 18D: Post-Execution Verification & Invariant Enforcement Tests

Comprehensive test suite for the post-execution verification system.
Minimum 30 tests covering all critical behaviors.

Test Categories:
1. Enum Validation Tests (5 tests)
2. Immutability Tests (5 tests)
3. Missing Data Tests (5 tests)
4. Scope Compliance Tests (4 tests)
5. Action Compliance Tests (4 tests)
6. Boundary Compliance Tests (4 tests)
7. Intent Compliance Tests (3 tests)
8. Invariant Compliance Tests (4 tests)
9. Outcome Consistency Tests (4 tests)
10. Determinism Tests (3 tests)
11. Audit Tests (3 tests)
12. Store Tests (4 tests)
13. No-Action Constraint Tests (4 tests)

Total: 52 tests
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytest

# Import verification model components
from controller.verification_model import (
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

# Import verification engine components
from controller.verification_engine import (
    PostExecutionVerificationEngine,
    get_verification_engine,
    verify_execution,
    create_verification_input,
    get_verification_summary,
    VERIFIER_VERSION,
    ALL_DOMAINS,
)

# Import verification store components
from controller.verification_store import (
    VerificationStore,
    get_verification_store,
    record_verification,
    get_verification_for_execution,
    get_violations_for_execution,
    get_verification_store_summary,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def engine(temp_dir):
    """Create a verification engine with temp files."""
    return PostExecutionVerificationEngine(
        results_file=temp_dir / "results.jsonl",
        audit_file=temp_dir / "audit.jsonl",
    )


@pytest.fixture
def store(temp_dir):
    """Create a verification store with temp files."""
    return VerificationStore(
        results_file=temp_dir / "results.jsonl",
        violations_file=temp_dir / "violations.jsonl",
    )


@pytest.fixture
def valid_execution_result():
    """Create a valid execution result snapshot."""
    return ExecutionResultSnapshot(
        execution_id="exec-test-001",
        intent_id="int-test-001",
        status="execution_success",
        block_reason=None,
        failure_reason=None,
        timestamp=datetime.utcnow().isoformat(),
        gate_decision_allowed=True,
        execution_output="Test output",
        rollback_performed=False,
    )


@pytest.fixture
def valid_execution_intent():
    """Create a valid execution intent snapshot."""
    return ExecutionIntentSnapshot(
        intent_id="int-test-001",
        project_id="proj-test-001",
        project_name="Test Project",
        action_type="run_tests",
        action_description="Run unit tests",
        requester_id="user-001",
        requester_role="developer",
        target_workspace="/tmp/test-workspace",
        created_at=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def valid_execution_audit():
    """Create a valid execution audit snapshot."""
    return ExecutionAuditSnapshot(
        audit_id="aud-test-001",
        execution_id="exec-test-001",
        intent_id="int-test-001",
        input_hash="abc123",
        status="execution_success",
        project_id="proj-test-001",
        action_type="run_tests",
        requester_id="user-001",
        eligibility_decision="automation_allowed_limited",
        approval_status="approval_granted",
        gate_allowed=True,
        timestamp=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def valid_execution_logs():
    """Create valid execution logs."""
    return ExecutionLogs(
        logs_path="/tmp/test-logs/test.log",
        logs_content="Running tests...\nAll tests passed.\n",
        logs_readable=True,
        exit_code=0,
        files_touched=("src/test.py", "tests/test_main.py"),
    )


@pytest.fixture
def valid_constraints():
    """Create valid execution constraints."""
    return ExecutionConstraints(
        allowed_actions=("run_tests", "update_docs"),
        forbidden_paths=("/etc", "/root", "/prod"),
        production_deploy_allowed=False,
        external_network_allowed=False,
    )


@pytest.fixture
def valid_intent_baseline():
    """Create a valid intent baseline."""
    return IntentBaselineSnapshot(
        project_id="proj-test-001",
        baseline_version="1.0.0",
        baseline_valid=True,
        approved_scope=("src/", "tests/"),
        approved_actions=("run_tests", "update_docs"),
    )


@pytest.fixture
def valid_lifecycle_snapshot():
    """Create a valid lifecycle snapshot."""
    return LifecycleSnapshot(
        project_id="proj-test-001",
        lifecycle_id="lc-001",
        state="DEVELOPMENT",
        is_active=True,
        last_transition=datetime.utcnow().isoformat(),
    )


@pytest.fixture
def complete_verification_input(
    valid_execution_result,
    valid_execution_intent,
    valid_execution_audit,
    valid_execution_logs,
    valid_constraints,
    valid_intent_baseline,
    valid_lifecycle_snapshot,
):
    """Create a complete verification input with all fields."""
    return VerificationInput(
        execution_result=valid_execution_result,
        execution_intent=valid_execution_intent,
        execution_audit=valid_execution_audit,
        lifecycle_snapshot=valid_lifecycle_snapshot,
        intent_baseline=valid_intent_baseline,
        execution_constraints=valid_constraints,
        execution_logs=valid_execution_logs,
    )


# =============================================================================
# 1. Enum Validation Tests (5 tests)
# =============================================================================

class TestEnumValidation:
    """Test enum constraints are enforced."""

    def test_verification_status_has_exactly_3_values(self):
        """VerificationStatus must have exactly 3 values."""
        values = list(VerificationStatus)
        assert len(values) == 3
        assert VerificationStatus.PASSED in values
        assert VerificationStatus.FAILED in values
        assert VerificationStatus.UNKNOWN in values

    def test_violation_severity_has_exactly_4_values(self):
        """ViolationSeverity must have exactly 4 values."""
        values = list(ViolationSeverity)
        assert len(values) == 4
        assert ViolationSeverity.INFO in values
        assert ViolationSeverity.LOW in values
        assert ViolationSeverity.MEDIUM in values
        assert ViolationSeverity.HIGH in values

    def test_violation_type_has_exactly_6_domains(self):
        """ViolationType must have exactly 6 domains."""
        values = list(ViolationType)
        assert len(values) == 6
        assert ViolationType.SCOPE_VIOLATION in values
        assert ViolationType.ACTION_VIOLATION in values
        assert ViolationType.BOUNDARY_VIOLATION in values
        assert ViolationType.INTENT_VIOLATION in values
        assert ViolationType.INVARIANT_VIOLATION in values
        assert ViolationType.OUTCOME_VIOLATION in values

    def test_unknown_reason_enum_values(self):
        """UnknownReason must have expected values."""
        values = list(UnknownReason)
        # Should have at least the documented reasons
        assert UnknownReason.MISSING_EXECUTION_RESULT in values
        assert UnknownReason.MISSING_EXECUTION_INTENT in values
        assert UnknownReason.MISSING_EXECUTION_AUDIT in values
        assert UnknownReason.MISSING_LOGS in values
        assert UnknownReason.LOGS_UNREADABLE in values

    def test_all_domains_constant_matches_enum(self):
        """ALL_DOMAINS constant must match ViolationType enum values."""
        assert len(ALL_DOMAINS) == 6
        enum_values = {v.value for v in ViolationType}
        assert set(ALL_DOMAINS) == enum_values


# =============================================================================
# 2. Immutability Tests (5 tests)
# =============================================================================

class TestImmutability:
    """Test frozen dataclass constraints."""

    def test_execution_result_snapshot_is_immutable(self, valid_execution_result):
        """ExecutionResultSnapshot must be frozen."""
        with pytest.raises(Exception):  # FrozenInstanceError
            valid_execution_result.execution_id = "new-id"

    def test_execution_intent_snapshot_is_immutable(self, valid_execution_intent):
        """ExecutionIntentSnapshot must be frozen."""
        with pytest.raises(Exception):
            valid_execution_intent.intent_id = "new-id"

    def test_verification_input_is_immutable(self, complete_verification_input):
        """VerificationInput must be frozen."""
        with pytest.raises(Exception):
            complete_verification_input.execution_result = None

    def test_invariant_violation_is_immutable(self):
        """InvariantViolation must be frozen."""
        violation = InvariantViolation(
            violation_id="viol-001",
            violation_type=ViolationType.SCOPE_VIOLATION.value,
            severity=ViolationSeverity.HIGH.value,
            description="Test violation",
            evidence_path="/test/path",
            evidence_snippet="Test snippet",
            detected_at=datetime.utcnow().isoformat(),
        )
        with pytest.raises(Exception):
            violation.severity = ViolationSeverity.LOW.value

    def test_verification_result_is_immutable(self):
        """ExecutionVerificationResult must be frozen."""
        result = ExecutionVerificationResult(
            verification_id="ver-001",
            execution_id="exec-001",
            verification_status=VerificationStatus.PASSED.value,
            unknown_reason=None,
            violations=(),
            input_hash="abc123",
            checked_at=datetime.utcnow().isoformat(),
            verifier_version=VERIFIER_VERSION,
            violation_count=0,
            high_severity_count=0,
            domains_checked=ALL_DOMAINS,
        )
        with pytest.raises(Exception):
            result.verification_status = VerificationStatus.FAILED.value


# =============================================================================
# 3. Missing Data Tests (5 tests)
# =============================================================================

class TestMissingData:
    """Test fail-closed behavior with missing data."""

    def test_missing_execution_result_returns_unknown(self, engine, valid_execution_intent, valid_execution_audit, valid_execution_logs):
        """Missing execution result must return UNKNOWN."""
        verification_input = VerificationInput(
            execution_result=None,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)
        assert result.verification_status == VerificationStatus.UNKNOWN.value
        assert result.unknown_reason == UnknownReason.MISSING_EXECUTION_RESULT.value

    def test_missing_execution_intent_returns_unknown(self, engine, valid_execution_result, valid_execution_audit, valid_execution_logs):
        """Missing execution intent must return UNKNOWN."""
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=None,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)
        assert result.verification_status == VerificationStatus.UNKNOWN.value
        assert result.unknown_reason == UnknownReason.MISSING_EXECUTION_INTENT.value

    def test_missing_execution_audit_returns_unknown(self, engine, valid_execution_result, valid_execution_intent, valid_execution_logs):
        """Missing execution audit must return UNKNOWN."""
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=None,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)
        assert result.verification_status == VerificationStatus.UNKNOWN.value
        assert result.unknown_reason == UnknownReason.MISSING_EXECUTION_AUDIT.value

    def test_missing_logs_returns_unknown(self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit):
        """Missing logs must return UNKNOWN."""
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=None,
        )
        result = engine.verify(verification_input)
        assert result.verification_status == VerificationStatus.UNKNOWN.value
        assert result.unknown_reason == UnknownReason.MISSING_LOGS.value

    def test_unreadable_logs_returns_unknown(self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit):
        """Unreadable logs must return UNKNOWN."""
        unreadable_logs = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content=None,
            logs_readable=False,  # Key: not readable
            exit_code=None,
            files_touched=(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=unreadable_logs,
        )
        result = engine.verify(verification_input)
        assert result.verification_status == VerificationStatus.UNKNOWN.value
        assert result.unknown_reason == UnknownReason.LOGS_UNREADABLE.value


# =============================================================================
# 4. Scope Compliance Tests (4 tests)
# =============================================================================

class TestScopeCompliance:
    """Test scope compliance verification (Domain 1)."""

    def test_scope_pass_when_files_in_approved_scope(self, engine, complete_verification_input):
        """Files in approved scope should pass."""
        result = engine.verify(complete_verification_input)
        # Should pass when files_touched ("src/test.py", "tests/test_main.py")
        # are within approved_scope ("src/", "tests/")
        scope_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.SCOPE_VIOLATION.value
        ]
        assert len(scope_violations) == 0

    def test_scope_fail_when_forbidden_path_touched(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit, valid_constraints
    ):
        """Touching forbidden paths must create HIGH severity violation."""
        logs_with_forbidden = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="Test logs",
            logs_readable=True,
            exit_code=0,
            files_touched=("/etc/passwd", "src/test.py"),  # /etc is forbidden
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=valid_constraints,
            execution_logs=logs_with_forbidden,
        )
        result = engine.verify(verification_input)
        assert result.verification_status == VerificationStatus.FAILED.value

        scope_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.SCOPE_VIOLATION.value
        ]
        assert len(scope_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in scope_violations)

    def test_scope_fail_when_outside_approved_scope(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit, valid_intent_baseline
    ):
        """Files outside approved scope must create MEDIUM severity violation."""
        logs_outside_scope = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="Test logs",
            logs_readable=True,
            exit_code=0,
            files_touched=("docs/readme.md",),  # Not in src/ or tests/
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=valid_intent_baseline,
            execution_constraints=None,
            execution_logs=logs_outside_scope,
        )
        result = engine.verify(verification_input)

        scope_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.SCOPE_VIOLATION.value
        ]
        assert len(scope_violations) >= 1
        assert any(v.severity == ViolationSeverity.MEDIUM.value for v in scope_violations)

    def test_scope_no_violation_when_no_files_touched(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit
    ):
        """No files touched should not create scope violation."""
        logs_no_files = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="Test logs",
            logs_readable=True,
            exit_code=0,
            files_touched=(),  # No files touched
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=logs_no_files,
        )
        result = engine.verify(verification_input)

        scope_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.SCOPE_VIOLATION.value
        ]
        assert len(scope_violations) == 0


# =============================================================================
# 5. Action Compliance Tests (4 tests)
# =============================================================================

class TestActionCompliance:
    """Test action compliance verification (Domain 2)."""

    def test_action_pass_when_in_allowed_list(self, engine, complete_verification_input):
        """Action in allowed list should pass."""
        result = engine.verify(complete_verification_input)
        action_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ACTION_VIOLATION.value
        ]
        assert len(action_violations) == 0

    def test_action_fail_when_not_in_allowed_list(
        self, engine, valid_execution_result, valid_execution_audit, valid_execution_logs, valid_constraints
    ):
        """Action not in allowed list must create HIGH severity violation."""
        intent_with_unapproved_action = ExecutionIntentSnapshot(
            intent_id="int-test-001",
            project_id="proj-test-001",
            project_name="Test Project",
            action_type="deploy_prod",  # Not in allowed_actions
            action_description="Deploy to production",
            requester_id="user-001",
            requester_role="developer",
            target_workspace="/tmp/test-workspace",
            created_at=datetime.utcnow().isoformat(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=intent_with_unapproved_action,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=valid_constraints,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        action_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ACTION_VIOLATION.value
        ]
        assert len(action_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in action_violations)

    def test_action_fail_when_not_in_baseline(
        self, engine, valid_execution_result, valid_execution_audit, valid_execution_logs
    ):
        """Action not in baseline approved actions must create MEDIUM severity violation."""
        intent_with_unapproved_action = ExecutionIntentSnapshot(
            intent_id="int-test-001",
            project_id="proj-test-001",
            project_name="Test Project",
            action_type="code_review",  # Not in baseline approved_actions
            action_description="Code review",
            requester_id="user-001",
            requester_role="developer",
            target_workspace="/tmp/test-workspace",
            created_at=datetime.utcnow().isoformat(),
        )
        baseline = IntentBaselineSnapshot(
            project_id="proj-test-001",
            baseline_version="1.0.0",
            baseline_valid=True,
            approved_scope=("src/",),
            approved_actions=("run_tests",),  # Does not include code_review
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=intent_with_unapproved_action,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=baseline,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        action_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ACTION_VIOLATION.value
        ]
        assert len(action_violations) >= 1
        assert any(v.severity == ViolationSeverity.MEDIUM.value for v in action_violations)

    def test_action_no_violation_without_constraints_or_baseline(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit, valid_execution_logs
    ):
        """No constraints or baseline means no action violations (can't check)."""
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        action_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.ACTION_VIOLATION.value
        ]
        assert len(action_violations) == 0


# =============================================================================
# 6. Boundary Compliance Tests (4 tests)
# =============================================================================

class TestBoundaryCompliance:
    """Test boundary compliance verification (Domain 3)."""

    def test_boundary_pass_when_no_prod_or_network(self, engine, complete_verification_input):
        """No production deploy or network access should pass."""
        result = engine.verify(complete_verification_input)
        boundary_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.BOUNDARY_VIOLATION.value
        ]
        assert len(boundary_violations) == 0

    def test_boundary_fail_on_deploy_prod_action(
        self, engine, valid_execution_result, valid_execution_audit, valid_execution_logs, valid_constraints
    ):
        """deploy_prod action must create HIGH severity boundary violation."""
        prod_intent = ExecutionIntentSnapshot(
            intent_id="int-test-001",
            project_id="proj-test-001",
            project_name="Test Project",
            action_type="deploy_prod",  # Production deploy
            action_description="Deploy to production",
            requester_id="user-001",
            requester_role="developer",
            target_workspace="/tmp/test-workspace",
            created_at=datetime.utcnow().isoformat(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=prod_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=valid_constraints,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        boundary_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.BOUNDARY_VIOLATION.value
        ]
        assert len(boundary_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in boundary_violations)

    def test_boundary_fail_on_network_access_in_logs(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit, valid_constraints
    ):
        """Network access indicators in logs must create MEDIUM severity violation."""
        logs_with_network = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="curl https://api.external.com/data\nDownloaded file",
            logs_readable=True,
            exit_code=0,
            files_touched=("src/test.py",),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=valid_constraints,
            execution_logs=logs_with_network,
        )
        result = engine.verify(verification_input)

        boundary_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.BOUNDARY_VIOLATION.value
        ]
        assert len(boundary_violations) >= 1
        assert any(v.severity == ViolationSeverity.MEDIUM.value for v in boundary_violations)

    def test_boundary_network_allowed_when_constraint_permits(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit
    ):
        """Network access allowed when constraints permit."""
        constraints_allow_network = ExecutionConstraints(
            allowed_actions=("run_tests",),
            forbidden_paths=(),
            production_deploy_allowed=False,
            external_network_allowed=True,  # Allow network
        )
        logs_with_network = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="curl https://api.external.com/data\nDownloaded file",
            logs_readable=True,
            exit_code=0,
            files_touched=("src/test.py",),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=constraints_allow_network,
            execution_logs=logs_with_network,
        )
        result = engine.verify(verification_input)

        boundary_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.BOUNDARY_VIOLATION.value
            and "network" in v.description.lower()
        ]
        assert len(boundary_violations) == 0


# =============================================================================
# 7. Intent Compliance Tests (3 tests)
# =============================================================================

class TestIntentCompliance:
    """Test intent compliance verification (Domain 4)."""

    def test_intent_pass_with_valid_baseline(self, engine, complete_verification_input):
        """Valid baseline should not create intent violations."""
        result = engine.verify(complete_verification_input)
        intent_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTENT_VIOLATION.value
        ]
        assert len(intent_violations) == 0

    def test_intent_fail_when_baseline_invalid(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit, valid_execution_logs
    ):
        """Invalid baseline must create MEDIUM severity violation."""
        invalid_baseline = IntentBaselineSnapshot(
            project_id="proj-test-001",
            baseline_version="1.0.0",
            baseline_valid=False,  # Invalid
            approved_scope=("src/",),
            approved_actions=("run_tests",),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=invalid_baseline,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        intent_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTENT_VIOLATION.value
        ]
        assert len(intent_violations) >= 1
        assert any(v.severity == ViolationSeverity.MEDIUM.value for v in intent_violations)

    def test_intent_fail_when_lifecycle_terminal(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_audit, valid_execution_logs
    ):
        """Terminal lifecycle state must create HIGH severity violation."""
        terminal_lifecycle = LifecycleSnapshot(
            project_id="proj-test-001",
            lifecycle_id="lc-001",
            state="rejected",  # Terminal state
            is_active=False,
            last_transition=datetime.utcnow().isoformat(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=terminal_lifecycle,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        intent_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INTENT_VIOLATION.value
        ]
        assert len(intent_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in intent_violations)


# =============================================================================
# 8. Invariant Compliance Tests (4 tests)
# =============================================================================

class TestInvariantCompliance:
    """Test invariant compliance verification (Domain 5)."""

    def test_invariant_pass_with_complete_audit(self, engine, complete_verification_input):
        """Complete audit record should pass."""
        result = engine.verify(complete_verification_input)
        invariant_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INVARIANT_VIOLATION.value
        ]
        assert len(invariant_violations) == 0

    def test_invariant_fail_when_missing_eligibility(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_logs
    ):
        """Missing eligibility decision must create HIGH severity violation."""
        audit_no_eligibility = ExecutionAuditSnapshot(
            audit_id="aud-test-001",
            execution_id="exec-test-001",
            intent_id="int-test-001",
            input_hash="abc123",
            status="execution_success",
            project_id="proj-test-001",
            action_type="run_tests",
            requester_id="user-001",
            eligibility_decision=None,  # Missing
            approval_status="approval_granted",
            gate_allowed=True,
            timestamp=datetime.utcnow().isoformat(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=audit_no_eligibility,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        invariant_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INVARIANT_VIOLATION.value
        ]
        assert len(invariant_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in invariant_violations)

    def test_invariant_fail_when_missing_approval(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_logs
    ):
        """Missing approval status must create HIGH severity violation."""
        audit_no_approval = ExecutionAuditSnapshot(
            audit_id="aud-test-001",
            execution_id="exec-test-001",
            intent_id="int-test-001",
            input_hash="abc123",
            status="execution_success",
            project_id="proj-test-001",
            action_type="run_tests",
            requester_id="user-001",
            eligibility_decision="automation_allowed_limited",
            approval_status=None,  # Missing
            gate_allowed=True,
            timestamp=datetime.utcnow().isoformat(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=audit_no_approval,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        invariant_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INVARIANT_VIOLATION.value
        ]
        assert len(invariant_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in invariant_violations)

    def test_invariant_fail_when_id_mismatch(
        self, engine, valid_execution_result, valid_execution_intent, valid_execution_logs
    ):
        """Execution ID mismatch must create HIGH severity violation."""
        audit_id_mismatch = ExecutionAuditSnapshot(
            audit_id="aud-test-001",
            execution_id="exec-DIFFERENT-001",  # Mismatch with result
            intent_id="int-test-001",
            input_hash="abc123",
            status="execution_success",
            project_id="proj-test-001",
            action_type="run_tests",
            requester_id="user-001",
            eligibility_decision="automation_allowed_limited",
            approval_status="approval_granted",
            gate_allowed=True,
            timestamp=datetime.utcnow().isoformat(),
        )
        verification_input = VerificationInput(
            execution_result=valid_execution_result,
            execution_intent=valid_execution_intent,
            execution_audit=audit_id_mismatch,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        invariant_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.INVARIANT_VIOLATION.value
        ]
        assert len(invariant_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in invariant_violations)


# =============================================================================
# 9. Outcome Consistency Tests (4 tests)
# =============================================================================

class TestOutcomeConsistency:
    """Test outcome consistency verification (Domain 6)."""

    def test_outcome_pass_when_success_matches_exit_code(self, engine, complete_verification_input):
        """SUCCESS with exit code 0 should pass."""
        result = engine.verify(complete_verification_input)
        outcome_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.OUTCOME_VIOLATION.value
        ]
        assert len(outcome_violations) == 0

    def test_outcome_fail_when_success_but_nonzero_exit(
        self, engine, valid_execution_intent, valid_execution_audit
    ):
        """SUCCESS with nonzero exit code must create HIGH severity violation."""
        success_result = ExecutionResultSnapshot(
            execution_id="exec-test-001",
            intent_id="int-test-001",
            status="execution_success",  # Success
            block_reason=None,
            failure_reason=None,
            timestamp=datetime.utcnow().isoformat(),
            gate_decision_allowed=True,
            execution_output="Test output",
            rollback_performed=False,
        )
        logs_nonzero = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="Test failed",
            logs_readable=True,
            exit_code=1,  # Nonzero
            files_touched=(),
        )
        verification_input = VerificationInput(
            execution_result=success_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=logs_nonzero,
        )
        result = engine.verify(verification_input)

        outcome_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.OUTCOME_VIOLATION.value
        ]
        assert len(outcome_violations) >= 1
        assert any(v.severity == ViolationSeverity.HIGH.value for v in outcome_violations)

    def test_outcome_fail_when_failed_but_zero_exit(
        self, engine, valid_execution_intent, valid_execution_audit
    ):
        """FAILED with exit code 0 must create MEDIUM severity violation."""
        failed_result = ExecutionResultSnapshot(
            execution_id="exec-test-001",
            intent_id="int-test-001",
            status="execution_failed",  # Failed
            block_reason=None,
            failure_reason="Test failure",
            timestamp=datetime.utcnow().isoformat(),
            gate_decision_allowed=True,
            execution_output="Test output",
            rollback_performed=False,
        )
        logs_zero = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="All good",
            logs_readable=True,
            exit_code=0,  # Zero
            files_touched=(),
        )
        verification_input = VerificationInput(
            execution_result=failed_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=logs_zero,
        )
        result = engine.verify(verification_input)

        outcome_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.OUTCOME_VIOLATION.value
        ]
        assert len(outcome_violations) >= 1
        assert any(v.severity == ViolationSeverity.MEDIUM.value for v in outcome_violations)

    def test_outcome_warn_when_success_but_many_errors_in_logs(
        self, engine, valid_execution_intent, valid_execution_audit
    ):
        """SUCCESS with many error indicators should create LOW severity violation."""
        success_result = ExecutionResultSnapshot(
            execution_id="exec-test-001",
            intent_id="int-test-001",
            status="execution_success",
            block_reason=None,
            failure_reason=None,
            timestamp=datetime.utcnow().isoformat(),
            gate_decision_allowed=True,
            execution_output="Test output",
            rollback_performed=False,
        )
        logs_with_errors = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="error: problem 1\nerror: problem 2\nerror: problem 3\nerror: problem 4",
            logs_readable=True,
            exit_code=0,
            files_touched=(),
        )
        verification_input = VerificationInput(
            execution_result=success_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=logs_with_errors,
        )
        result = engine.verify(verification_input)

        outcome_violations = [
            v for v in result.violations
            if v.violation_type == ViolationType.OUTCOME_VIOLATION.value
        ]
        assert len(outcome_violations) >= 1
        assert any(v.severity == ViolationSeverity.LOW.value for v in outcome_violations)


# =============================================================================
# 10. Determinism Tests (3 tests)
# =============================================================================

class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_input_produces_same_status(self, engine, complete_verification_input):
        """Same input must produce same verification status."""
        result1 = engine.verify(complete_verification_input)
        result2 = engine.verify(complete_verification_input)

        assert result1.verification_status == result2.verification_status

    def test_same_input_produces_same_violation_count(self, engine, complete_verification_input):
        """Same input must produce same violation count."""
        result1 = engine.verify(complete_verification_input)
        result2 = engine.verify(complete_verification_input)

        assert result1.violation_count == result2.violation_count

    def test_same_input_produces_same_hash(self, complete_verification_input):
        """Same input must produce same hash."""
        hash1 = complete_verification_input.compute_hash()
        hash2 = complete_verification_input.compute_hash()

        assert hash1 == hash2


# =============================================================================
# 11. Audit Tests (3 tests)
# =============================================================================

class TestAudit:
    """Test audit record writing."""

    def test_audit_record_written_on_pass(self, engine, temp_dir, complete_verification_input):
        """Audit record must be written on PASSED verification."""
        result = engine.verify(complete_verification_input)

        audit_file = temp_dir / "audit.jsonl"
        assert audit_file.exists()

        with open(audit_file, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) >= 1
        assert records[-1]["verification_status"] == VerificationStatus.PASSED.value

    def test_audit_record_written_on_fail(self, engine, temp_dir, valid_execution_intent, valid_execution_audit):
        """Audit record must be written on FAILED verification."""
        # Create input that will fail
        fail_result = ExecutionResultSnapshot(
            execution_id="exec-test-001",
            intent_id="int-test-001",
            status="execution_success",
            block_reason=None,
            failure_reason=None,
            timestamp=datetime.utcnow().isoformat(),
            gate_decision_allowed=True,
            execution_output="Test output",
            rollback_performed=False,
        )
        logs_fail = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="Test",
            logs_readable=True,
            exit_code=1,  # Will cause outcome violation
            files_touched=(),
        )
        verification_input = VerificationInput(
            execution_result=fail_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=logs_fail,
        )
        result = engine.verify(verification_input)

        audit_file = temp_dir / "audit.jsonl"
        assert audit_file.exists()

        with open(audit_file, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) >= 1
        assert records[-1]["verification_status"] == VerificationStatus.FAILED.value

    def test_audit_record_written_on_unknown(self, engine, temp_dir, valid_execution_logs):
        """Audit record must be written on UNKNOWN verification."""
        # Create input with missing data
        verification_input = VerificationInput(
            execution_result=None,
            execution_intent=None,
            execution_audit=None,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=None,
            execution_logs=valid_execution_logs,
        )
        result = engine.verify(verification_input)

        audit_file = temp_dir / "audit.jsonl"
        assert audit_file.exists()

        with open(audit_file, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) >= 1
        assert records[-1]["verification_status"] == VerificationStatus.UNKNOWN.value


# =============================================================================
# 12. Store Tests (4 tests)
# =============================================================================

class TestStore:
    """Test verification store operations."""

    def test_store_record_verification(self, store, temp_dir):
        """Store must record verification results."""
        result = ExecutionVerificationResult(
            verification_id="ver-test-001",
            execution_id="exec-001",
            verification_status=VerificationStatus.PASSED.value,
            unknown_reason=None,
            violations=(),
            input_hash="abc123",
            checked_at=datetime.utcnow().isoformat(),
            verifier_version=VERIFIER_VERSION,
            violation_count=0,
            high_severity_count=0,
            domains_checked=ALL_DOMAINS,
        )
        store.record_verification(result)

        results_file = temp_dir / "results.jsonl"
        assert results_file.exists()

        with open(results_file, 'r') as f:
            records = [json.loads(line) for line in f if line.strip()]

        assert len(records) >= 1
        assert records[0]["verification_id"] == "ver-test-001"

    def test_store_get_verification(self, store, temp_dir):
        """Store must retrieve verification by execution ID."""
        result = ExecutionVerificationResult(
            verification_id="ver-test-001",
            execution_id="exec-001",
            verification_status=VerificationStatus.PASSED.value,
            unknown_reason=None,
            violations=(),
            input_hash="abc123",
            checked_at=datetime.utcnow().isoformat(),
            verifier_version=VERIFIER_VERSION,
            violation_count=0,
            high_severity_count=0,
            domains_checked=ALL_DOMAINS,
        )
        store.record_verification(result)

        retrieved = store.get_verification("exec-001")
        assert retrieved is not None
        assert retrieved.verification_id == "ver-test-001"
        assert retrieved.execution_id == "exec-001"

    def test_store_get_recent_verifications(self, store, temp_dir):
        """Store must retrieve recent verifications."""
        for i in range(5):
            result = ExecutionVerificationResult(
                verification_id=f"ver-test-{i:03d}",
                execution_id=f"exec-{i:03d}",
                verification_status=VerificationStatus.PASSED.value,
                unknown_reason=None,
                violations=(),
                input_hash="abc123",
                checked_at=datetime.utcnow().isoformat(),
                verifier_version=VERIFIER_VERSION,
                violation_count=0,
                high_severity_count=0,
                domains_checked=ALL_DOMAINS,
            )
            store.record_verification(result)

        recent = store.get_recent_verifications(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].verification_id == "ver-test-004"

    def test_store_get_summary(self, store, temp_dir):
        """Store must provide summary statistics."""
        # Add some results
        for i in range(3):
            result = ExecutionVerificationResult(
                verification_id=f"ver-test-{i:03d}",
                execution_id=f"exec-{i:03d}",
                verification_status=VerificationStatus.PASSED.value if i < 2 else VerificationStatus.FAILED.value,
                unknown_reason=None,
                violations=(),
                input_hash="abc123",
                checked_at=datetime.utcnow().isoformat(),
                verifier_version=VERIFIER_VERSION,
                violation_count=0 if i < 2 else 1,
                high_severity_count=0,
                domains_checked=ALL_DOMAINS,
            )
            store.record_verification(result)

        summary = store.get_summary(since_hours=24)
        assert summary["total_verifications"] == 3
        assert summary["passed_count"] == 2
        assert summary["failed_count"] == 1


# =============================================================================
# 13. No-Action Constraint Tests (4 tests)
# =============================================================================

class TestNoActionConstraints:
    """Test that verification NEVER executes, recommends, or mutates."""

    def test_verification_returns_result_only(self, engine, complete_verification_input):
        """Verification must return result, nothing else."""
        result = engine.verify(complete_verification_input)

        # Result must be an ExecutionVerificationResult
        assert isinstance(result, ExecutionVerificationResult)

        # Result must have required fields
        assert result.verification_id is not None
        assert result.execution_id is not None
        assert result.verification_status in [v.value for v in VerificationStatus]

    def test_verification_does_not_modify_input(self, complete_verification_input):
        """Verification must not modify input."""
        original_hash = complete_verification_input.compute_hash()

        engine = PostExecutionVerificationEngine()
        engine.verify(complete_verification_input)

        # Hash must be unchanged
        assert complete_verification_input.compute_hash() == original_hash

    def test_violation_has_no_recommendation_field(self):
        """InvariantViolation must not have recommendation fields."""
        violation = InvariantViolation(
            violation_id="viol-001",
            violation_type=ViolationType.SCOPE_VIOLATION.value,
            severity=ViolationSeverity.HIGH.value,
            description="Test violation",
            evidence_path="/test/path",
            evidence_snippet="Test snippet",
            detected_at=datetime.utcnow().isoformat(),
        )

        # Should NOT have these fields
        assert not hasattr(violation, 'recommendation')
        assert not hasattr(violation, 'suggested_fix')
        assert not hasattr(violation, 'action_to_take')
        assert not hasattr(violation, 'auto_remediate')

    def test_result_has_no_action_fields(self):
        """ExecutionVerificationResult must not have action fields."""
        result = ExecutionVerificationResult(
            verification_id="ver-001",
            execution_id="exec-001",
            verification_status=VerificationStatus.FAILED.value,
            unknown_reason=None,
            violations=(),
            input_hash="abc123",
            checked_at=datetime.utcnow().isoformat(),
            verifier_version=VERIFIER_VERSION,
            violation_count=0,
            high_severity_count=0,
            domains_checked=ALL_DOMAINS,
        )

        # Should NOT have these fields
        assert not hasattr(result, 'rollback_triggered')
        assert not hasattr(result, 'retry_scheduled')
        assert not hasattr(result, 'notification_sent')
        assert not hasattr(result, 'auto_fix_applied')
        assert not hasattr(result, 'recommendation')


# =============================================================================
# Additional Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the verification system."""

    def test_full_verification_flow_pass(self, engine, complete_verification_input):
        """Full verification flow with passing result."""
        result = engine.verify(complete_verification_input)

        assert result.verification_status == VerificationStatus.PASSED.value
        assert result.violation_count == 0
        assert result.high_severity_count == 0
        assert len(result.domains_checked) == 6

    def test_full_verification_flow_fail(
        self, engine, valid_execution_intent, valid_execution_audit
    ):
        """Full verification flow with failing result."""
        # Create result that will have violations
        fail_result = ExecutionResultSnapshot(
            execution_id="exec-test-001",
            intent_id="int-test-001",
            status="execution_success",
            block_reason=None,
            failure_reason=None,
            timestamp=datetime.utcnow().isoformat(),
            gate_decision_allowed=True,
            execution_output="Test output",
            rollback_performed=False,
        )
        fail_logs = ExecutionLogs(
            logs_path="/tmp/test.log",
            logs_content="Test",
            logs_readable=True,
            exit_code=1,
            files_touched=("/etc/passwd",),  # Forbidden path
        )
        fail_constraints = ExecutionConstraints(
            allowed_actions=("run_tests",),
            forbidden_paths=("/etc",),
            production_deploy_allowed=False,
            external_network_allowed=False,
        )
        verification_input = VerificationInput(
            execution_result=fail_result,
            execution_intent=valid_execution_intent,
            execution_audit=valid_execution_audit,
            lifecycle_snapshot=None,
            intent_baseline=None,
            execution_constraints=fail_constraints,
            execution_logs=fail_logs,
        )
        result = engine.verify(verification_input)

        assert result.verification_status == VerificationStatus.FAILED.value
        assert result.violation_count > 0

    def test_create_verification_input_from_dict(self):
        """Test convenience function for creating input from dicts."""
        result_dict = {
            "execution_id": "exec-001",
            "intent_id": "int-001",
            "status": "execution_success",
            "timestamp": datetime.utcnow().isoformat(),
            "rollback_performed": False,
        }
        intent_dict = {
            "intent_id": "int-001",
            "project_id": "proj-001",
            "project_name": "Test",
            "action_type": "run_tests",
            "action_description": "Test",
            "requester_id": "user-001",
            "requester_role": "developer",
            "target_workspace": "/tmp",
            "created_at": datetime.utcnow().isoformat(),
        }
        audit_dict = {
            "audit_id": "aud-001",
            "execution_id": "exec-001",
            "intent_id": "int-001",
            "input_hash": "abc123",
            "status": "execution_success",
            "timestamp": datetime.utcnow().isoformat(),
        }
        logs_dict = {
            "logs_readable": True,
            "exit_code": 0,
        }

        verification_input = create_verification_input(
            execution_result=result_dict,
            execution_intent=intent_dict,
            execution_audit=audit_dict,
            execution_logs=logs_dict,
        )

        assert verification_input.execution_result is not None
        assert verification_input.execution_result.execution_id == "exec-001"
        assert verification_input.execution_intent is not None
        assert verification_input.execution_intent.intent_id == "int-001"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
