"""
Phase 26: Meta-Remediation (Rescue-the-Rescue Layer) Tests

Tests:
- PART 1: Deterministic classification of RESCUE_JOB_FAILED
- PART 2: Immutable MetaRemediationTask
- PART 3: Gate blocks DEPLOY_TEST when meta status is PENDING/FIXING/RETESTING
- PART 4: Retry limit of 3 enforced
- PART 5: Registry metadata updated correctly
- PART 6: Prompt contains all required fields
- PART 7: BlockReason enum
- PART 8: FailurePattern and FailureOwner enums
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# PART 1: Deterministic Classification
# ---------------------------------------------------------------------------

class TestClassifyRescueFailure:
    """MetaRemediator classifies rescue failures deterministically."""

    def test_exit_code_5_returns_rescue_job_failed(self):
        """Exit code 5 → RESCUE_JOB_FAILED."""
        from controller.meta_remediator import MetaRemediator
        from controller.micro_remediator import FailurePattern

        rem = MetaRemediator()
        result = rem.classify_rescue_failure("job-1", 5, "test-project")
        assert result == FailurePattern.RESCUE_JOB_FAILED.value

    def test_exit_code_1_returns_rescue_job_failed(self):
        """Any non-zero exit code → RESCUE_JOB_FAILED."""
        from controller.meta_remediator import MetaRemediator
        from controller.micro_remediator import FailurePattern

        rem = MetaRemediator()
        result = rem.classify_rescue_failure("job-2", 1, "test-project")
        assert result == FailurePattern.RESCUE_JOB_FAILED.value

    def test_exit_code_0_returns_unknown(self):
        """Exit code 0 → UNKNOWN (no failure)."""
        from controller.meta_remediator import MetaRemediator
        from controller.micro_remediator import FailurePattern

        rem = MetaRemediator()
        result = rem.classify_rescue_failure("job-3", 0, "test-project")
        assert result == FailurePattern.UNKNOWN.value

    def test_classification_with_module(self):
        """Classification works with optional module parameter."""
        from controller.meta_remediator import MetaRemediator
        from controller.micro_remediator import FailurePattern

        rem = MetaRemediator()
        result = rem.classify_rescue_failure("job-4", 5, "test-project", module="api")
        assert result == FailurePattern.RESCUE_JOB_FAILED.value


# ---------------------------------------------------------------------------
# PART 2: Immutable MetaRemediationTask
# ---------------------------------------------------------------------------

class TestMetaRemediationTask:
    """MetaRemediationTask is frozen (immutable)."""

    def test_task_is_frozen(self):
        from controller.meta_remediator import MetaRemediationTask, MetaRemediationStatus

        task = MetaRemediationTask(
            id="test-id",
            project="test-project",
            module=None,
            original_failure="http_503",
            rescue_job_id="rescue-1",
            rescue_exit_code=5,
            created_at="2026-02-08T00:00:00",
            status=MetaRemediationStatus.PENDING.value,
            attempt=1,
        )
        with pytest.raises(AttributeError):
            task.project = "changed"

    def test_task_to_dict(self):
        from controller.meta_remediator import MetaRemediationTask, MetaRemediationStatus

        task = MetaRemediationTask(
            id="test-id",
            project="test-project",
            module="api",
            original_failure="http_503",
            rescue_job_id="rescue-1",
            rescue_exit_code=5,
            created_at="2026-02-08T00:00:00",
            status=MetaRemediationStatus.PENDING.value,
            attempt=1,
        )
        d = task.to_dict()
        assert d["project"] == "test-project"
        assert d["module"] == "api"
        assert d["rescue_exit_code"] == 5
        assert d["attempt"] == 1


# ---------------------------------------------------------------------------
# PART 3: Execution Gate Step 16
# ---------------------------------------------------------------------------

class TestExecutionGateStep16:
    """Gate blocks DEPLOY_TEST when meta-remediation is in progress."""

    def _make_workspace(self, tmp_path):
        from controller.execution_gate import REQUIRED_GOVERNANCE_DOCS
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")
        return workspace

    def test_blocked_when_meta_pending(self, tmp_path):
        """DEPLOY_TEST is blocked when meta_remediation_status is PENDING."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        mock_state = {"meta_remediation_status": "pending", "attempts": 1}

        with patch("controller.meta_remediator.MetaRemediator.get_meta_state", return_value=mock_state):
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
        assert decision.meta_remediation_blocks is True
        assert "UNRESOLVED_META_FAILURE" in (decision.denied_reason or "")

    def test_blocked_when_meta_fixing(self, tmp_path):
        """DEPLOY_TEST is blocked when meta_remediation_status is FIXING."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        mock_state = {"meta_remediation_status": "fixing", "attempts": 2}

        with patch("controller.meta_remediator.MetaRemediator.get_meta_state", return_value=mock_state):
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
        assert decision.meta_remediation_blocks is True

    def test_allowed_when_meta_resolved(self, tmp_path):
        """DEPLOY_TEST is allowed when meta_remediation_status is RESOLVED."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        mock_state = {"meta_remediation_status": "resolved", "attempts": 1}

        with patch("controller.meta_remediator.MetaRemediator.get_meta_state", return_value=mock_state):
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
        assert decision.meta_remediation_blocks is False

    def test_read_code_unaffected(self, tmp_path):
        """READ_CODE action is not affected by meta-remediation."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        gate = ExecutionGate()
        request = ExecutionRequest(
            job_id="test-job",
            project_name="test-project",
            aspect="core",
            lifecycle_id="lc-1",
            lifecycle_state=LifecycleState.DEVELOPMENT.value,
            requested_action=ExecutionAction.READ_CODE.value,
            requesting_user_id="user-1",
            requesting_user_role=UserRole.OWNER.value,
            workspace_path=str(workspace),
            task_description="Read code",
        )
        decision = gate.evaluate(request)

        assert decision.allowed is True
        assert decision.meta_remediation_blocks is False

    def test_allowed_when_no_meta_state(self, tmp_path):
        """DEPLOY_TEST is allowed when meta_remediator import fails (fail-open)."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        with patch("controller.meta_remediator.MetaRemediator.get_meta_state", side_effect=ImportError("not available")):
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


# ---------------------------------------------------------------------------
# PART 4: Retry Limit Enforcement
# ---------------------------------------------------------------------------

class TestRetryLimit:
    """Meta-remediation enforces max 3 attempts."""

    def test_can_attempt_when_no_state(self):
        from controller.meta_remediator import MetaRemediator

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {}
            mock_reg.return_value.get_project.return_value = mock_project

            can, reason = MetaRemediator.can_attempt_meta_remediation("test-project")
            assert can is True

    def test_cannot_attempt_when_failed(self):
        from controller.meta_remediator import MetaRemediator

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {
                "meta_remediation": {
                    "meta_remediation_status": "failed",
                    "attempts": 3,
                }
            }
            mock_reg.return_value.get_project.return_value = mock_project

            can, reason = MetaRemediator.can_attempt_meta_remediation("test-project")
            assert can is False
            assert "manual intervention" in reason.lower()

    def test_cannot_attempt_when_max_reached(self):
        from controller.meta_remediator import MetaRemediator, META_REMEDIATION_MAX_ATTEMPTS

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {
                "meta_remediation": {
                    "meta_remediation_status": "pending",
                    "attempts": META_REMEDIATION_MAX_ATTEMPTS,
                }
            }
            mock_reg.return_value.get_project.return_value = mock_project

            can, reason = MetaRemediator.can_attempt_meta_remediation("test-project")
            assert can is False
            assert "max" in reason.lower()

    def test_cannot_attempt_when_already_fixing(self):
        from controller.meta_remediator import MetaRemediator

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {
                "meta_remediation": {
                    "meta_remediation_status": "fixing",
                    "attempts": 1,
                }
            }
            mock_reg.return_value.get_project.return_value = mock_project

            can, reason = MetaRemediator.can_attempt_meta_remediation("test-project")
            assert can is False
            assert "in progress" in reason.lower()

    def test_can_attempt_when_resolved(self):
        from controller.meta_remediator import MetaRemediator

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {
                "meta_remediation": {
                    "meta_remediation_status": "resolved",
                    "attempts": 2,
                }
            }
            mock_reg.return_value.get_project.return_value = mock_project

            can, reason = MetaRemediator.can_attempt_meta_remediation("test-project")
            assert can is True


# ---------------------------------------------------------------------------
# PART 5: Registry Metadata Updated
# ---------------------------------------------------------------------------

class TestRegistryMetadata:
    """create_meta_remediation_task updates Project.metadata correctly."""

    def test_creates_task_and_updates_metadata(self):
        from controller.meta_remediator import MetaRemediator, MetaRemediationStatus, META_STATE_FILE

        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "test_state.jsonl"

            with patch.object(MetaRemediator, "_append_record"):
                with patch("controller.project_registry.get_registry") as mock_reg:
                    mock_project = MagicMock()
                    mock_project.metadata = {}
                    mock_reg.return_value.get_project.return_value = mock_project

                    rem = MetaRemediator()
                    task = rem.create_meta_remediation_task(
                        project="test-project",
                        rescue_job_id="rescue-123",
                        rescue_exit_code=5,
                        original_failure="http_503",
                        module="api",
                    )

                    assert task.project == "test-project"
                    assert task.rescue_exit_code == 5
                    assert task.attempt == 1
                    assert task.status == MetaRemediationStatus.PENDING.value

                    # Verify metadata was updated
                    mock_reg.return_value.update_project.assert_called_once()
                    call_args = mock_reg.return_value.update_project.call_args
                    updated_metadata = call_args[0][1]["metadata"]
                    assert "meta_remediation" in updated_metadata
                    meta = updated_metadata["meta_remediation"]
                    assert meta["meta_remediation_status"] == MetaRemediationStatus.PENDING.value
                    assert meta["attempts"] == 1
                    assert meta["last_meta_failure"]["rescue_job_id"] == "rescue-123"

    def test_handle_completion_resolved(self):
        from controller.meta_remediator import MetaRemediator, MetaRemediationStatus

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {
                "meta_remediation": {
                    "meta_remediation_status": "retesting",
                    "attempts": 1,
                }
            }
            mock_reg.return_value.get_project.return_value = mock_project

            rem = MetaRemediator()
            status = rem.handle_meta_remediation_completion("test-project", "job-1", passed=True)
            assert status == MetaRemediationStatus.RESOLVED.value

    def test_handle_completion_failed_at_max(self):
        from controller.meta_remediator import MetaRemediator, MetaRemediationStatus, META_REMEDIATION_MAX_ATTEMPTS

        with patch("controller.project_registry.get_registry") as mock_reg:
            mock_project = MagicMock()
            mock_project.metadata = {
                "meta_remediation": {
                    "meta_remediation_status": "retesting",
                    "attempts": META_REMEDIATION_MAX_ATTEMPTS,
                }
            }
            mock_reg.return_value.get_project.return_value = mock_project

            rem = MetaRemediator()
            status = rem.handle_meta_remediation_completion("test-project", "job-1", passed=False)
            assert status == MetaRemediationStatus.FAILED.value


# ---------------------------------------------------------------------------
# PART 6: Prompt Contains Required Fields
# ---------------------------------------------------------------------------

class TestPromptGeneration:
    """generate_meta_fix_prompt includes all diagnostic fields."""

    def test_prompt_contains_all_fields(self):
        from controller.meta_remediator import MetaRemediator, MetaRemediationTask, MetaRemediationStatus

        task = MetaRemediationTask(
            id="test-id",
            project="health-tracker",
            module="api",
            original_failure="http_503",
            rescue_job_id="rescue-abc123",
            rescue_exit_code=5,
            created_at="2026-02-08T10:00:00",
            status=MetaRemediationStatus.PENDING.value,
            attempt=2,
        )

        context = {
            "github_actions_url": "https://github.com/user/repo/actions/runs/12345",
            "deployed_sha": "abc1234",
            "repo_sha": "def5678",
            "server_type": "CyberPanel/LiteSpeed",
            "failed_endpoints": ["https://healthapi.example.com/health"],
            "validator_results": {"api": "503"},
            "api_log_tail": "ModuleNotFoundError: No module named 'app'",
            "uvicorn_running": "No",
            "litespeed_proxy_snippet": "proxy_pass 127.0.0.1:8022",
        }

        rem = MetaRemediator()
        prompt = rem.generate_meta_fix_prompt(task, context)

        assert "rescue-abc123" in prompt
        assert "5" in prompt  # exit code
        assert "github.com/user/repo/actions/runs/12345" in prompt
        assert "abc1234" in prompt  # deployed SHA
        assert "def5678" in prompt  # repo SHA
        assert "CyberPanel/LiteSpeed" in prompt
        assert "healthapi.example.com/health" in prompt
        assert "ModuleNotFoundError" in prompt
        assert "uvicorn" in prompt.lower()
        assert "proxy_pass" in prompt
        assert "META-REMEDIATION JOB" in prompt

    def test_prompt_with_empty_context(self):
        from controller.meta_remediator import MetaRemediator, MetaRemediationTask, MetaRemediationStatus

        task = MetaRemediationTask(
            id="test-id",
            project="test-project",
            module=None,
            original_failure="unknown",
            rescue_job_id="rescue-xyz",
            rescue_exit_code=1,
            created_at="2026-02-08T10:00:00",
            status=MetaRemediationStatus.PENDING.value,
            attempt=1,
        )

        rem = MetaRemediator()
        prompt = rem.generate_meta_fix_prompt(task, {})

        assert "rescue-xyz" in prompt
        assert "META-REMEDIATION JOB" in prompt
        assert "Not available" in prompt


# ---------------------------------------------------------------------------
# PART 7: BlockReason Enum
# ---------------------------------------------------------------------------

class TestBlockReasonEnum:
    """BlockReason includes UNRESOLVED_META_FAILURE."""

    def test_unresolved_meta_failure_in_enum(self):
        from controller.execution_dispatcher import BlockReason

        assert BlockReason.UNRESOLVED_META_FAILURE == "unresolved_meta_failure"
        assert BlockReason.UNRESOLVED_META_FAILURE.value == "unresolved_meta_failure"


# ---------------------------------------------------------------------------
# PART 8: FailurePattern and FailureOwner Enums
# ---------------------------------------------------------------------------

class TestEnumExtensions:
    """FailurePattern has RESCUE_JOB_FAILED, FailureOwner has PLATFORM."""

    def test_rescue_job_failed_in_failure_pattern(self):
        from controller.micro_remediator import FailurePattern

        assert FailurePattern.RESCUE_JOB_FAILED == "rescue_job_failed"
        assert FailurePattern.RESCUE_JOB_FAILED.value == "rescue_job_failed"

    def test_platform_in_failure_owner(self):
        from controller.micro_remediator import FailureOwner

        assert FailureOwner.PLATFORM == "platform"
        assert FailureOwner.PLATFORM.value == "platform"


# ---------------------------------------------------------------------------
# PART 9: JSONL Persistence
# ---------------------------------------------------------------------------

class TestJSONLPersistence:
    """Append-only JSONL store works correctly."""

    def test_append_and_read(self):
        from controller.meta_remediator import MetaRemediator, META_STATE_FILE

        with tempfile.TemporaryDirectory() as tmp:
            test_file = Path(tmp) / "test_meta.jsonl"

            # Monkey-patch the file path temporarily
            import controller.meta_remediator as mod
            original = mod.META_STATE_FILE
            mod.META_STATE_FILE = test_file

            try:
                MetaRemediator._append_record({"project": "proj-a", "id": "1"})
                MetaRemediator._append_record({"project": "proj-b", "id": "2"})
                MetaRemediator._append_record({"project": "proj-a", "id": "3"})

                all_records = MetaRemediator.get_all_records()
                assert len(all_records) == 3

                proj_a = MetaRemediator.get_all_records("proj-a")
                assert len(proj_a) == 2

                proj_b = MetaRemediator.get_all_records("proj-b")
                assert len(proj_b) == 1
            finally:
                mod.META_STATE_FILE = original
