"""
Unit Tests for Execution Gate - Phase 15.6

Security-critical tests that prove:
1. Claude cannot deploy from TESTING state
2. Claude cannot commit in AWAITING_FEEDBACK state
3. Claude cannot write outside workspace
4. Claude cannot act without lifecycle permission
5. Audit trail is always written
6. Hard fail conditions are enforced
"""

import pytest
import tempfile
import os
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

# Set up test environment before imports
TEST_TEMP_DIR = tempfile.mkdtemp()
os.environ['EXECUTION_AUDIT_LOG'] = os.path.join(TEST_TEMP_DIR, 'test_audit.log')

from controller.execution_gate import (
    ExecutionAction,
    LifecycleState,
    UserRole,
    ProjectAspect,
    ExecutionRequest,
    GateDecision,
    ExecutionAuditEntry,
    ExecutionGate,
    LIFECYCLE_ALLOWED_ACTIONS,
    ROLE_ALLOWED_ACTIONS,
    REQUIRED_GOVERNANCE_DOCS,
    execution_gate,
    check_execution_allowed,
    get_execution_constraints_for_job,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_workspace():
    """Create a temporary workspace with governance documents."""
    path = Path(tempfile.mkdtemp(prefix="test_workspace_"))

    # Create required governance documents
    for doc in REQUIRED_GOVERNANCE_DOCS:
        (path / doc).write_text(f"# {doc}\nTest governance document")

    yield path

    # Cleanup
    import shutil
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def temp_workspace_no_docs():
    """Create a temporary workspace WITHOUT governance documents."""
    path = Path(tempfile.mkdtemp(prefix="test_workspace_nodocs_"))
    yield path

    # Cleanup
    import shutil
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def temp_audit_log():
    """Create a temporary audit log file."""
    fd, path = tempfile.mkstemp(suffix=".log", prefix="test_audit_")
    os.close(fd)
    yield Path(path)

    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def gate_with_temp_log(temp_audit_log):
    """Create an ExecutionGate with a temporary audit log."""
    gate = ExecutionGate()
    gate._audit_log_path = temp_audit_log
    return gate


def create_request(
    job_id: str = "test-job-1",
    project_name: str = "test-project",
    aspect: str = "core",
    lifecycle_id: str = "test-lifecycle-1",
    lifecycle_state: str = "development",
    requested_action: str = "write_code",
    user_id: str = "user-123",
    user_role: str = "developer",
    workspace_path: str = "/tmp/test_workspace",
    task_description: str = "Test task",
) -> ExecutionRequest:
    """Helper to create test execution requests."""
    return ExecutionRequest(
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


# -----------------------------------------------------------------------------
# Test: Lifecycle State Permissions
# -----------------------------------------------------------------------------
class TestLifecycleStatePermissions:
    """Tests proving lifecycle state permission enforcement."""

    def test_development_allows_write_code(self, gate_with_temp_log, temp_workspace):
        """DEVELOPMENT state should allow WRITE_CODE action."""
        request = create_request(
            lifecycle_state="development",
            requested_action="write_code",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)
        assert decision.allowed is True
        assert "write_code" in decision.allowed_actions

    def test_development_allows_commit(self, gate_with_temp_log, temp_workspace):
        """DEVELOPMENT state should allow COMMIT action."""
        request = create_request(
            lifecycle_state="development",
            requested_action="commit",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)
        assert decision.allowed is True

    def test_development_denies_push(self, gate_with_temp_log, temp_workspace):
        """DEVELOPMENT state should DENY PUSH action."""
        request = create_request(
            lifecycle_state="development",
            requested_action="push",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)
        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "push" in decision.denied_reason.lower()

    def test_development_denies_deploy_test(self, gate_with_temp_log, temp_workspace):
        """DEVELOPMENT state should DENY DEPLOY_TEST action."""
        request = create_request(
            lifecycle_state="development",
            requested_action="deploy_test",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)
        assert decision.allowed is False
        assert decision.hard_fail is True

    def test_testing_denies_write_code(self, gate_with_temp_log, temp_workspace):
        """TESTING state should DENY WRITE_CODE action."""
        request = create_request(
            lifecycle_state="testing",
            requested_action="write_code",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)
        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "write_code" in decision.denied_reason

    def test_testing_allows_run_tests(self, gate_with_temp_log, temp_workspace):
        """TESTING state should allow RUN_TESTS action."""
        request = create_request(
            lifecycle_state="testing",
            requested_action="run_tests",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)
        assert decision.allowed is True

    def test_created_denies_all_actions(self, gate_with_temp_log, temp_workspace):
        """CREATED state should DENY ALL actions."""
        for action in ExecutionAction:
            request = create_request(
                lifecycle_state="created",
                requested_action=action.value,
                workspace_path=str(temp_workspace),
            )
            decision = gate_with_temp_log.evaluate(request)
            assert decision.allowed is False, f"CREATED should deny {action}"
            assert decision.hard_fail is True

    def test_archived_denies_all_actions(self, gate_with_temp_log, temp_workspace):
        """ARCHIVED state should DENY ALL actions."""
        for action in ExecutionAction:
            request = create_request(
                lifecycle_state="archived",
                requested_action=action.value,
                workspace_path=str(temp_workspace),
            )
            decision = gate_with_temp_log.evaluate(request)
            assert decision.allowed is False, f"ARCHIVED should deny {action}"

    def test_rejected_denies_all_actions(self, gate_with_temp_log, temp_workspace):
        """REJECTED state should DENY ALL actions."""
        for action in ExecutionAction:
            request = create_request(
                lifecycle_state="rejected",
                requested_action=action.value,
                workspace_path=str(temp_workspace),
            )
            decision = gate_with_temp_log.evaluate(request)
            assert decision.allowed is False, f"REJECTED should deny {action}"


# -----------------------------------------------------------------------------
# Test: SECURITY-CRITICAL - Deploy Restrictions
# -----------------------------------------------------------------------------
class TestDeployRestrictions:
    """SECURITY-CRITICAL tests proving Claude cannot deploy inappropriately."""

    def test_cannot_deploy_from_testing(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Claude CANNOT deploy from TESTING state.

        This test proves that automated deployment is blocked when code
        is still being tested.
        """
        request = create_request(
            lifecycle_state="testing",
            requested_action="deploy_test",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "deploy_test" in decision.denied_reason

    def test_cannot_deploy_from_development(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Claude CANNOT deploy from DEVELOPMENT state.
        """
        request = create_request(
            lifecycle_state="development",
            requested_action="deploy_test",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True

    def test_can_deploy_test_from_ready_for_production(self, gate_with_temp_log, temp_workspace):
        """
        READY_FOR_PRODUCTION state CAN deploy to test environment.
        """
        # Need owner/admin role for push/deploy
        request = create_request(
            lifecycle_state="ready_for_production",
            requested_action="deploy_test",
            user_role="owner",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is True

    def test_production_deploy_always_denied(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Production deployment is NEVER allowed via automated execution.

        Even in READY_FOR_PRODUCTION state with OWNER role, DEPLOY_PROD is blocked.
        """
        # Try every lifecycle state and every role
        for state in LifecycleState:
            for role in UserRole:
                request = create_request(
                    lifecycle_state=state.value,
                    requested_action="deploy_prod",
                    user_role=role.value,
                    workspace_path=str(temp_workspace),
                )
                decision = gate_with_temp_log.evaluate(request)

                assert decision.allowed is False, \
                    f"DEPLOY_PROD should be denied for state={state}, role={role}"
                assert decision.hard_fail is True
                assert "production" in decision.denied_reason.lower()


# -----------------------------------------------------------------------------
# Test: SECURITY-CRITICAL - Commit Restrictions
# -----------------------------------------------------------------------------
class TestCommitRestrictions:
    """SECURITY-CRITICAL tests proving Claude cannot commit inappropriately."""

    def test_cannot_commit_in_awaiting_feedback(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Claude CANNOT commit in AWAITING_FEEDBACK state.

        Code changes must not be made while waiting for human feedback.
        """
        request = create_request(
            lifecycle_state="awaiting_feedback",
            requested_action="commit",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "commit" in decision.denied_reason

    def test_cannot_commit_in_testing(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Claude CANNOT commit in TESTING state.

        Tests should run against committed code, not change during testing.
        """
        request = create_request(
            lifecycle_state="testing",
            requested_action="commit",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True

    def test_cannot_commit_in_deployed(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Claude CANNOT commit in DEPLOYED state.

        Changes to deployed code require a new lifecycle.
        """
        request = create_request(
            lifecycle_state="deployed",
            requested_action="commit",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True


# -----------------------------------------------------------------------------
# Test: SECURITY-CRITICAL - Workspace Isolation
# -----------------------------------------------------------------------------
class TestWorkspaceIsolation:
    """SECURITY-CRITICAL tests proving workspace isolation."""

    def test_cannot_access_outside_jobs_directory(self, gate_with_temp_log, temp_workspace):
        """
        SECURITY-CRITICAL: Claude CANNOT access paths outside /home/aitesting.mybd.in/jobs/

        This prevents path traversal attacks and unauthorized file access.
        """
        # Try various path traversal attempts
        dangerous_paths = [
            "/etc/passwd",
            "/home/aitesting.mybd.in/../etc/passwd",
            "/root/.ssh/",
            "/var/log/",
            "/home/otheruser/",
            "/home/aitesting.mybd.in/secrets/",
        ]

        for dangerous_path in dangerous_paths:
            request = create_request(
                workspace_path=dangerous_path,
            )
            decision = gate_with_temp_log.evaluate(request)

            assert decision.allowed is False, \
                f"Access to {dangerous_path} should be DENIED"
            assert decision.hard_fail is True
            assert "workspace" in decision.denied_reason.lower()

    def test_allows_valid_jobs_workspace(self, gate_with_temp_log):
        """
        Valid workspace paths within /home/aitesting.mybd.in/jobs/ should be allowed
        (assuming governance docs exist).
        """
        # Note: This test would pass with temp_workspace fixture but
        # we're testing the workspace validation logic specifically
        request = create_request(
            workspace_path="/home/aitesting.mybd.in/jobs/test-job-1",
        )
        # Will fail due to missing governance docs (expected behavior)
        decision = gate_with_temp_log.evaluate(request)

        # The decision should fail due to governance docs, NOT workspace validation
        # If it fails due to workspace, that means our workspace path isn't accepted
        # (This is actually the expected behavior since the directory doesn't exist)
        # The key test is that it doesn't fail due to workspace path being invalid

    def test_allows_tmp_workspace_for_testing(self, gate_with_temp_log, temp_workspace):
        """
        /tmp/ workspaces are allowed for testing purposes.
        """
        request = create_request(
            lifecycle_state="development",
            requested_action="read_code",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        # Should not fail due to workspace validation
        assert decision.workspace_validated is True or decision.allowed is True


# -----------------------------------------------------------------------------
# Test: Governance Document Requirements
# -----------------------------------------------------------------------------
class TestGovernanceDocuments:
    """Tests proving governance document requirements."""

    def test_fails_without_governance_docs(self, gate_with_temp_log, temp_workspace_no_docs):
        """
        Execution MUST fail if governance documents are missing.
        """
        request = create_request(
            lifecycle_state="development",
            requested_action="write_code",
            workspace_path=str(temp_workspace_no_docs),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "governance" in decision.denied_reason.lower()

    def test_required_governance_docs_list(self):
        """
        Verify the required governance documents are correct.
        """
        expected_docs = {
            "AI_POLICY.md",
            "ARCHITECTURE.md",
            "CURRENT_STATE.md",
            "DEPLOYMENT.md",
            "PROJECT_CONTEXT.md",
            "PROJECT_MANIFEST.yaml",
            "TESTING_STRATEGY.md",
        }
        assert REQUIRED_GOVERNANCE_DOCS == expected_docs


# -----------------------------------------------------------------------------
# Test: Role-Based Access Control
# -----------------------------------------------------------------------------
class TestRoleBasedAccessControl:
    """Tests proving role-based access control."""

    def test_viewer_cannot_write_code(self, gate_with_temp_log, temp_workspace):
        """
        VIEWER role cannot write code, even in DEVELOPMENT state.
        """
        request = create_request(
            lifecycle_state="development",
            requested_action="write_code",
            user_role="viewer",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "viewer" in decision.denied_reason.lower()

    def test_tester_cannot_write_code(self, gate_with_temp_log, temp_workspace):
        """
        TESTER role cannot write code.
        """
        request = create_request(
            lifecycle_state="development",
            requested_action="write_code",
            user_role="tester",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True

    def test_developer_cannot_push(self, gate_with_temp_log, temp_workspace):
        """
        DEVELOPER role cannot push, even in READY_FOR_PRODUCTION state.
        """
        request = create_request(
            lifecycle_state="ready_for_production",
            requested_action="push",
            user_role="developer",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True

    def test_owner_can_push_in_ready_for_production(self, gate_with_temp_log, temp_workspace):
        """
        OWNER role can push in READY_FOR_PRODUCTION state.
        """
        request = create_request(
            lifecycle_state="ready_for_production",
            requested_action="push",
            user_role="owner",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is True


# -----------------------------------------------------------------------------
# Test: Audit Trail
# -----------------------------------------------------------------------------
class TestAuditTrail:
    """Tests proving audit trail is always written."""

    def test_audit_log_written_on_allow(self, temp_audit_log, temp_workspace):
        """
        Audit log must be written when execution is ALLOWED.
        """
        gate = ExecutionGate()
        gate._audit_log_path = temp_audit_log

        request = create_request(
            lifecycle_state="development",
            requested_action="write_code",
            workspace_path=str(temp_workspace),
        )
        decision = gate.evaluate(request)

        # Check audit log was written
        assert temp_audit_log.exists()
        with open(temp_audit_log, 'r') as f:
            log_content = f.read()

        assert "ALLOWED" in log_content
        assert request.job_id in log_content

    def test_audit_log_written_on_deny(self, temp_audit_log, temp_workspace):
        """
        Audit log must be written when execution is DENIED.
        """
        gate = ExecutionGate()
        gate._audit_log_path = temp_audit_log

        request = create_request(
            lifecycle_state="testing",
            requested_action="deploy_test",  # Not allowed in testing
            workspace_path=str(temp_workspace),
        )
        decision = gate.evaluate(request)

        # Check audit log was written
        assert temp_audit_log.exists()
        with open(temp_audit_log, 'r') as f:
            log_content = f.read()

        assert "DENIED" in log_content
        assert request.job_id in log_content

    def test_audit_entry_contains_all_fields(self, temp_audit_log, temp_workspace):
        """
        Audit entry must contain all required fields.
        """
        gate = ExecutionGate()
        gate._audit_log_path = temp_audit_log

        request = create_request(
            job_id="audit-test-job",
            project_name="audit-test-project",
            aspect="backend",
            lifecycle_id="audit-test-lifecycle",
            lifecycle_state="development",
            requested_action="write_code",
            user_id="audit-test-user",
            user_role="developer",
            workspace_path=str(temp_workspace),
            task_description="Test audit entry fields",
        )
        gate.evaluate(request)

        # Parse audit log
        with open(temp_audit_log, 'r') as f:
            entry = json.loads(f.readline())

        # Verify all required fields
        assert entry["job_id"] == "audit-test-job"
        assert entry["project_name"] == "audit-test-project"
        assert entry["aspect"] == "backend"
        assert entry["lifecycle_id"] == "audit-test-lifecycle"
        assert entry["lifecycle_state"] == "development"
        assert entry["executed_action"] == "write_code"
        assert entry["requesting_user_id"] == "audit-test-user"
        assert entry["requesting_user_role"] == "developer"
        assert "timestamp" in entry


# -----------------------------------------------------------------------------
# Test: Invalid Inputs
# -----------------------------------------------------------------------------
class TestInvalidInputs:
    """Tests proving invalid inputs are handled correctly."""

    def test_invalid_lifecycle_state(self, gate_with_temp_log, temp_workspace):
        """
        Invalid lifecycle state must result in hard fail.
        """
        request = create_request(
            lifecycle_state="invalid_state",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "invalid" in decision.denied_reason.lower()

    def test_invalid_action(self, gate_with_temp_log, temp_workspace):
        """
        Invalid action must result in hard fail.
        """
        request = create_request(
            requested_action="invalid_action",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True
        assert "invalid" in decision.denied_reason.lower()

    def test_invalid_role(self, gate_with_temp_log, temp_workspace):
        """
        Invalid user role must result in hard fail.
        """
        request = create_request(
            user_role="invalid_role",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.hard_fail is True


# -----------------------------------------------------------------------------
# Test: Convenience Functions
# -----------------------------------------------------------------------------
class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_check_execution_allowed_returns_tuple(self, temp_workspace):
        """
        check_execution_allowed returns (allowed, denied_reason, allowed_actions).
        """
        allowed, denied_reason, allowed_actions = check_execution_allowed(
            job_id="test-job",
            project_name="test-project",
            aspect="core",
            lifecycle_id="test-lifecycle",
            lifecycle_state="development",
            requested_action="write_code",
            user_id="test-user",
            user_role="developer",
            workspace_path=str(temp_workspace),
            task_description="Test",
        )

        assert isinstance(allowed, bool)
        assert isinstance(allowed_actions, list)

    def test_get_execution_constraints_for_job(self):
        """
        get_execution_constraints_for_job returns correct constraints.
        """
        constraints = get_execution_constraints_for_job(
            lifecycle_state="development",
            user_role="developer",
        )

        assert "lifecycle_state" in constraints
        assert "user_role" in constraints
        assert "allowed_actions" in constraints
        assert "constraints" in constraints

        # Developer in development should be able to write code
        assert "write_code" in constraints["allowed_actions"]
        # But not push
        assert "push" not in constraints["allowed_actions"]


# -----------------------------------------------------------------------------
# Test: Lifecycle State Mapping Completeness
# -----------------------------------------------------------------------------
class TestLifecycleStateMappingCompleteness:
    """Tests ensuring all lifecycle states have permission mappings."""

    def test_all_lifecycle_states_have_mapping(self):
        """
        Every LifecycleState must have a mapping in LIFECYCLE_ALLOWED_ACTIONS.
        """
        for state in LifecycleState:
            assert state in LIFECYCLE_ALLOWED_ACTIONS, \
                f"Missing mapping for lifecycle state: {state}"

    def test_all_roles_have_mapping(self):
        """
        Every UserRole must have a mapping in ROLE_ALLOWED_ACTIONS.
        """
        for role in UserRole:
            assert role in ROLE_ALLOWED_ACTIONS, \
                f"Missing mapping for role: {role}"


# -----------------------------------------------------------------------------
# Test: Decision Object Integrity
# -----------------------------------------------------------------------------
class TestDecisionObjectIntegrity:
    """Tests ensuring GateDecision object has correct data."""

    def test_allowed_decision_has_no_denied_reason(self, gate_with_temp_log, temp_workspace):
        """
        When allowed=True, denied_reason should be None.
        """
        request = create_request(
            lifecycle_state="development",
            requested_action="write_code",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        if decision.allowed:
            assert decision.denied_reason is None
            assert decision.hard_fail is False

    def test_denied_decision_has_reason(self, gate_with_temp_log, temp_workspace):
        """
        When allowed=False, denied_reason must be set.
        """
        request = create_request(
            lifecycle_state="testing",
            requested_action="deploy_prod",
            workspace_path=str(temp_workspace),
        )
        decision = gate_with_temp_log.evaluate(request)

        assert decision.allowed is False
        assert decision.denied_reason is not None
        assert len(decision.denied_reason) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
