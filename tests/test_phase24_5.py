"""
Phase 24.5: Self-Healing Micro-Remediation Layer Tests

Tests:
- PART 1: Failure classification (FailurePattern, FailureOwner enums and rules)
- PART 2: Remediation task creation
- PART 3: Claude fix prompt generation
- PART 4: ExecutionGate Step 14 blocking
- PART 5: Dashboard field propagation
"""

import sys
import os
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# PART 1: Failure Classification
# ---------------------------------------------------------------------------

class TestFailureClassification:
    """FailurePattern and FailureOwner enum values and classification rules."""

    def test_enum_values_exist(self):
        """All FailurePattern and FailureOwner values are defined."""
        from controller.micro_remediator import FailurePattern, FailureOwner

        assert FailurePattern.CORS == "cors"
        assert FailurePattern.HTTP_404 == "http_404"
        assert FailurePattern.HTTP_500 == "http_500"
        assert FailurePattern.AUTH_FAILURE == "auth_failure"
        assert FailurePattern.DATA_MISSING == "data_missing"
        assert FailurePattern.CONSOLE_ERROR == "console_error"
        assert FailurePattern.BUILD_FAILURE == "build_failure"
        assert FailurePattern.UNKNOWN == "unknown"

        assert FailureOwner.FRONTEND == "frontend"
        assert FailureOwner.BACKEND == "backend"
        assert FailureOwner.INFRA == "infra"
        assert FailureOwner.DEPLOYMENT == "deployment"
        assert FailureOwner.UNKNOWN == "unknown"

    def test_remediation_status_enum(self):
        """RemediationStatus enum has all expected values."""
        from controller.micro_remediator import RemediationStatus

        assert RemediationStatus.PENDING == "pending"
        assert RemediationStatus.FIXING == "fixing"
        assert RemediationStatus.RETESTING == "retesting"
        assert RemediationStatus.RESOLVED == "resolved"
        assert RemediationStatus.FAILED == "failed"

    def test_cors_failure_classified_as_backend(self):
        """CORS check_type -> CORS pattern, BACKEND owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "cors_browser_origin",
            "passed": False,
            "message": "No Access-Control-Allow-Origin header",
            "details": {},
        })
        assert pattern == FailurePattern.CORS
        assert owner == FailureOwner.BACKEND

    def test_frontend_404_classified_as_frontend(self):
        """/login 404 -> HTTP_404 pattern, FRONTEND owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "frontend_pages",
            "passed": False,
            "message": "/login returned 404",
            "details": {},
        })
        assert pattern == FailurePattern.HTTP_404
        assert owner == FailureOwner.FRONTEND

    def test_api_503_classified_as_backend(self):
        """API returning 503 -> HTTP_500 pattern, BACKEND owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "api_schema_valid",
            "passed": False,
            "message": "API returned 503 Service Unavailable",
            "details": {},
        })
        assert pattern == FailurePattern.HTTP_500
        assert owner == FailureOwner.BACKEND

    def test_auth_failure_classified_as_backend(self):
        """Auth E2E failure -> AUTH_FAILURE pattern, BACKEND owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "auth_login_e2e",
            "passed": False,
            "message": "Registration failed: 500 Internal Server Error",
            "details": {},
        })
        assert pattern == FailurePattern.AUTH_FAILURE
        assert owner == FailureOwner.BACKEND

    def test_business_data_missing_classified_as_backend(self):
        """Missing business data -> DATA_MISSING pattern, BACKEND owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "api_business_data",
            "passed": False,
            "message": "Missing fields: total_calories, entries_count",
            "details": {"missing_fields": ["total_calories", "entries_count"]},
        })
        assert pattern == FailurePattern.DATA_MISSING
        assert owner == FailureOwner.BACKEND

    def test_unknown_check_type_returns_unknown(self):
        """Unknown check type -> UNKNOWN pattern, UNKNOWN owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "some_future_check",
            "passed": False,
            "message": "Something went wrong",
            "details": {},
        })
        assert pattern == FailurePattern.UNKNOWN
        assert owner == FailureOwner.UNKNOWN

    def test_commit_sha_classified_as_deployment(self):
        """Commit SHA mismatch -> BUILD_FAILURE pattern, DEPLOYMENT owner."""
        from controller.micro_remediator import (
            MicroRemediator, FailurePattern, FailureOwner,
        )

        remediator = MicroRemediator()
        pattern, owner = remediator.classify_failure({
            "check_type": "commit_sha_match",
            "passed": False,
            "message": "Deployed SHA abc123 does not match GitHub SHA def456",
            "details": {},
        })
        assert pattern == FailurePattern.BUILD_FAILURE
        assert owner == FailureOwner.DEPLOYMENT


# ---------------------------------------------------------------------------
# PART 2: Remediation Task Creation
# ---------------------------------------------------------------------------

class TestRemediationTaskCreation:
    """MicroRemediationTask creation from classified failures."""

    def test_cors_failure_creates_backend_task(self):
        """CORS failure creates a backend remediation task."""
        from controller.micro_remediator import MicroRemediator, MicroRemediationTask

        remediator = MicroRemediator()
        task = remediator.create_remediation_task({
            "check_type": "cors_browser_origin",
            "passed": False,
            "message": "No Access-Control-Allow-Origin header",
            "details": {},
            "_project_name": "test-project",
        })
        assert isinstance(task, MicroRemediationTask)
        assert task.failure_pattern == "cors"
        assert task.owner == "backend"
        assert task.project == "test-project"
        assert task.requires_redeploy is True
        assert task.check_type == "cors_browser_origin"

    def test_task_is_frozen(self):
        """MicroRemediationTask is immutable."""
        from controller.micro_remediator import MicroRemediator

        remediator = MicroRemediator()
        task = remediator.create_remediation_task({
            "check_type": "cors_browser_origin",
            "passed": False,
            "message": "CORS error",
            "details": {},
            "_project_name": "test-project",
        })
        with pytest.raises(AttributeError):
            task.owner = "frontend"

    def test_task_has_suggested_fix(self):
        """Remediation task has non-empty suggested_fix."""
        from controller.micro_remediator import MicroRemediator

        remediator = MicroRemediator()
        task = remediator.create_remediation_task({
            "check_type": "auth_login_e2e",
            "passed": False,
            "message": "Login failed",
            "details": {},
            "_project_name": "my-project",
        })
        assert task.suggested_fix
        assert len(task.suggested_fix) > 10
        assert task.failure_pattern == "auth_failure"


# ---------------------------------------------------------------------------
# PART 3: Claude Fix Prompt Generation
# ---------------------------------------------------------------------------

class TestClaudeFixPrompt:
    """Claude fix prompt generation from remediation tasks."""

    def test_prompt_contains_project_name(self):
        """Fix prompt includes the project name."""
        from controller.micro_remediator import MicroRemediator

        remediator = MicroRemediator()
        task = remediator.create_remediation_task({
            "check_type": "cors_browser_origin",
            "passed": False,
            "message": "No CORS header",
            "details": {},
            "_project_name": "health-tracker",
        })
        prompt = remediator.generate_claude_fix_prompt(task)
        assert "health-tracker" in prompt

    def test_prompt_contains_failure_pattern(self):
        """Fix prompt includes the failure pattern."""
        from controller.micro_remediator import MicroRemediator

        remediator = MicroRemediator()
        task = remediator.create_remediation_task({
            "check_type": "api_business_data",
            "passed": False,
            "message": "Missing fields",
            "details": {},
            "_project_name": "my-app",
        })
        prompt = remediator.generate_claude_fix_prompt(task)
        assert "data_missing" in prompt

    def test_prompt_contains_scoped_instructions(self):
        """Fix prompt scopes to the correct owner component."""
        from controller.micro_remediator import MicroRemediator

        remediator = MicroRemediator()
        task = remediator.create_remediation_task({
            "check_type": "frontend_pages",
            "passed": False,
            "message": "/login returned 404",
            "details": {},
            "_project_name": "my-app",
        })
        prompt = remediator.generate_claude_fix_prompt(task)
        assert "Only fix the frontend component" in prompt
        assert "MICRO-REMEDIATION JOB" in prompt


# ---------------------------------------------------------------------------
# PART 4: ExecutionGate Step 14 Blocking
# ---------------------------------------------------------------------------

class TestExecutionGateStep14:
    """ExecutionGate blocks on unresolved micro-remediation."""

    def test_deploy_test_blocked_when_remediation_fixing(self, tmp_path):
        """DEPLOY_TEST blocked when remediation status is FIXING."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        with patch(
            "controller.micro_remediator.MicroRemediator.get_remediation_state",
            return_value={"status": "fixing", "attempts": 1},
        ):
            gate = ExecutionGate()
            request = ExecutionRequest(
                job_id="test-job",
                project_name="test-project",
                aspect="core",
                lifecycle_id="lc-1",
                lifecycle_state=LifecycleState.READY_FOR_PRODUCTION.value,
                requested_action=ExecutionAction.DEPLOY_TEST.value,
                requesting_user_id="user-1",
                requesting_user_role=UserRole.OWNER.value,
                workspace_path=str(workspace),
                task_description="Deploy to testing",
            )
            decision = gate.evaluate(request)
            assert decision.allowed is False
            assert decision.micro_remediation_blocks is True
            assert "micro-remediation" in decision.denied_reason.lower()

    def test_deploy_test_blocked_when_remediation_failed(self, tmp_path):
        """DEPLOY_TEST blocked with hard_fail when remediation FAILED."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        with patch(
            "controller.micro_remediator.MicroRemediator.get_remediation_state",
            return_value={"status": "failed", "attempts": 3},
        ):
            gate = ExecutionGate()
            request = ExecutionRequest(
                job_id="test-job",
                project_name="test-project",
                aspect="core",
                lifecycle_id="lc-1",
                lifecycle_state=LifecycleState.READY_FOR_PRODUCTION.value,
                requested_action=ExecutionAction.DEPLOY_TEST.value,
                requesting_user_id="user-1",
                requesting_user_role=UserRole.OWNER.value,
                workspace_path=str(workspace),
                task_description="Deploy to testing",
            )
            decision = gate.evaluate(request)
            assert decision.allowed is False
            assert decision.hard_fail is True
            assert decision.micro_remediation_blocks is True
            assert "MANUAL_INTERVENTION" in decision.denied_reason

    def test_deploy_test_allowed_when_remediation_resolved(self, tmp_path):
        """DEPLOY_TEST allowed when remediation status is RESOLVED."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        with patch(
            "controller.micro_remediator.MicroRemediator.get_remediation_state",
            return_value={"status": "resolved", "attempts": 1},
        ):
            gate = ExecutionGate()
            request = ExecutionRequest(
                job_id="test-job",
                project_name="test-project",
                aspect="core",
                lifecycle_id="lc-1",
                lifecycle_state=LifecycleState.READY_FOR_PRODUCTION.value,
                requested_action=ExecutionAction.DEPLOY_TEST.value,
                requesting_user_id="user-1",
                requesting_user_role=UserRole.OWNER.value,
                workspace_path=str(workspace),
                task_description="Deploy to testing",
            )
            decision = gate.evaluate(request)
            assert decision.allowed is True
            assert decision.micro_remediation_blocks is False

    def test_deploy_test_allowed_when_no_remediation_state(self, tmp_path):
        """DEPLOY_TEST allowed when no remediation state exists."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        with patch(
            "controller.micro_remediator.MicroRemediator.get_remediation_state",
            return_value={},
        ):
            gate = ExecutionGate()
            request = ExecutionRequest(
                job_id="test-job",
                project_name="test-project",
                aspect="core",
                lifecycle_id="lc-1",
                lifecycle_state=LifecycleState.READY_FOR_PRODUCTION.value,
                requested_action=ExecutionAction.DEPLOY_TEST.value,
                requesting_user_id="user-1",
                requesting_user_role=UserRole.OWNER.value,
                workspace_path=str(workspace),
                task_description="Deploy to testing",
            )
            decision = gate.evaluate(request)
            assert decision.allowed is True

    def test_read_code_not_affected_by_remediation(self, tmp_path):
        """READ_CODE action is never blocked by remediation status."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        # Even with remediation FIXING, READ_CODE should pass
        with patch(
            "controller.micro_remediator.MicroRemediator.get_remediation_state",
            return_value={"status": "fixing", "attempts": 1},
        ):
            gate = ExecutionGate()
            request = ExecutionRequest(
                job_id="test-job",
                project_name="test-project",
                aspect="core",
                lifecycle_id="lc-1",
                lifecycle_state=LifecycleState.DEVELOPMENT.value,
                requested_action=ExecutionAction.READ_CODE.value,
                requesting_user_id="user-1",
                requesting_user_role=UserRole.DEVELOPER.value,
                workspace_path=str(workspace),
                task_description="Read code",
            )
            decision = gate.evaluate(request)
            assert decision.allowed is True
            assert decision.micro_remediation_blocks is False


# ---------------------------------------------------------------------------
# PART 5: Dashboard Fields
# ---------------------------------------------------------------------------

class TestDashboardFields:
    """Dashboard fields for micro-remediation."""

    def test_project_overview_has_remediation_fields(self):
        """ProjectOverview includes micro-remediation fields."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="id-1",
            project_name="test",
            mode="project_mode",
            current_lifecycle_state="development",
            aspects={},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id=None,
            created_at=None,
            updated_at=None,
            last_micro_failure={"pattern": "cors", "owner": "backend"},
            remediation_status="fixing",
        )
        d = overview.to_dict()
        assert d["last_micro_failure"]["pattern"] == "cors"
        assert d["last_micro_failure"]["owner"] == "backend"
        assert d["remediation_status"] == "fixing"

    def test_project_overview_defaults_none(self):
        """Micro-remediation fields default to None (backward compat)."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="id-1",
            project_name="test",
            mode="project_mode",
            current_lifecycle_state="development",
            aspects={},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id=None,
            created_at=None,
            updated_at=None,
        )
        d = overview.to_dict()
        assert d["last_micro_failure"] is None
        assert d["remediation_status"] is None


# ---------------------------------------------------------------------------
# PART 6: State Management
# ---------------------------------------------------------------------------

class TestStateManagement:
    """Remediation state read/write via registry metadata."""

    def test_can_attempt_when_no_state(self):
        """First attempt always allowed."""
        from controller.micro_remediator import MicroRemediator

        with patch.object(
            MicroRemediator, "get_remediation_state", return_value={}
        ):
            can, reason = MicroRemediator.can_attempt_remediation("my-project")
            assert can is True
            assert "No previous" in reason

    def test_cannot_attempt_when_failed(self):
        """Cannot attempt when status is FAILED."""
        from controller.micro_remediator import MicroRemediator

        with patch.object(
            MicroRemediator, "get_remediation_state",
            return_value={"status": "failed", "attempts": 3},
        ):
            can, reason = MicroRemediator.can_attempt_remediation("my-project")
            assert can is False
            assert "manual intervention" in reason.lower()

    def test_cannot_attempt_when_already_fixing(self):
        """Cannot attempt when status is FIXING."""
        from controller.micro_remediator import MicroRemediator

        with patch.object(
            MicroRemediator, "get_remediation_state",
            return_value={"status": "fixing", "attempts": 1},
        ):
            can, reason = MicroRemediator.can_attempt_remediation("my-project")
            assert can is False
            assert "in progress" in reason.lower()

    def test_can_attempt_when_resolved(self):
        """Can attempt after previous remediation was resolved."""
        from controller.micro_remediator import MicroRemediator

        with patch.object(
            MicroRemediator, "get_remediation_state",
            return_value={"status": "resolved", "attempts": 1},
        ):
            can, reason = MicroRemediator.can_attempt_remediation("my-project")
            assert can is True

    def test_cannot_attempt_when_max_reached(self):
        """Cannot attempt when max attempts reached."""
        from controller.micro_remediator import MicroRemediator

        with patch.object(
            MicroRemediator, "get_remediation_state",
            return_value={"status": "pending", "attempts": 3},
        ):
            can, reason = MicroRemediator.can_attempt_remediation("my-project")
            assert can is False
            assert "Max" in reason


# ---------------------------------------------------------------------------
# PART 7: BlockReason Enum
# ---------------------------------------------------------------------------

class TestBlockReasonEnum:
    """Execution dispatcher BlockReason includes micro-remediation."""

    def test_unresolved_micro_failure_in_block_reason(self):
        """UNRESOLVED_MICRO_FAILURE is a valid BlockReason."""
        from controller.execution_dispatcher import BlockReason

        assert BlockReason.UNRESOLVED_MICRO_FAILURE == "unresolved_micro_failure"
        assert hasattr(BlockReason, "UNRESOLVED_MICRO_FAILURE")
