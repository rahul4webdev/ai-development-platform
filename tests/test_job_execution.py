"""
Unit Tests for Claude Job Execution - Phase 19

Test coverage for:
- Job creation and queuing
- Job script configuration
- Permission mode settings
- Job state transitions
- Error handling

Phase 19: Ensures job execution chain works end-to-end
"""

import pytest
import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_job_data():
    """Sample job creation data."""
    return {
        "project_name": "test-project",
        "task_description": "Test task description",
        "task_type": "planning",
        "created_by": "test-user",
        "priority": 75,
        "aspect": "core",
        "lifecycle_state": "planning",
        "requested_action": "write_code",
        "user_role": "owner",
    }


@pytest.fixture
def mock_scheduler():
    """Mock multi-worker scheduler."""
    scheduler = MagicMock()
    scheduler.running = True
    scheduler.enqueue_job = AsyncMock()
    return scheduler


# -----------------------------------------------------------------------------
# Test Cases: Script Configuration
# -----------------------------------------------------------------------------
class TestScriptConfiguration:
    """Tests for run_claude_job.sh script configuration."""

    def test_script_exists(self):
        """Verify run_claude_job.sh exists."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        assert script_path.exists(), f"Script not found: {script_path}"

    def test_script_uses_correct_permission_mode(self):
        """Verify script uses --permission-mode acceptEdits, NOT --dangerously-skip-permissions."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        content = script_path.read_text()

        # MUST use --permission-mode acceptEdits
        assert "--permission-mode acceptEdits" in content, \
            "Script MUST use --permission-mode acceptEdits for automation"

        # MUST NOT use --dangerously-skip-permissions (causes core dump)
        # Allow it only in comments
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if '--dangerously-skip-permissions' in line:
                # If it's not a comment, fail
                stripped = line.strip()
                if not stripped.startswith('#'):
                    pytest.fail(
                        f"Line {i}: Found --dangerously-skip-permissions outside comment. "
                        "This flag causes Claude CLI to crash. Use --permission-mode acceptEdits instead."
                    )

    def test_script_has_proper_timeout(self):
        """Verify script has timeout configuration."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        content = script_path.read_text()

        assert "CLAUDE_TIMEOUT" in content, "Script must have CLAUDE_TIMEOUT configuration"
        assert "timeout" in content, "Script must use timeout command"

    def test_script_has_exit_codes_documented(self):
        """Verify script documents exit codes."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        content = script_path.read_text()

        # Check for exit code documentation
        assert "Exit Codes:" in content or "exit code" in content.lower(), \
            "Script should document exit codes"

    def test_script_copies_required_docs(self):
        """Verify script copies required governance documents."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        content = script_path.read_text()

        required_docs = [
            "AI_POLICY.md",
            "ARCHITECTURE.md",
            "CURRENT_STATE.md",
        ]

        for doc in required_docs:
            assert doc in content, f"Script should reference required document: {doc}"


# -----------------------------------------------------------------------------
# Test Cases: Job Creation
# -----------------------------------------------------------------------------
class TestJobCreation:
    """Tests for job creation functionality."""

    @pytest.mark.asyncio
    async def test_create_job_returns_job_object(self, sample_job_data):
        """Test that create_job returns a proper job object."""
        with patch('controller.claude_backend.multi_scheduler') as mock_sched:
            mock_sched.running = True
            mock_sched.enqueue_job = AsyncMock()

            from controller.claude_backend import create_job

            job = await create_job(
                project_name=sample_job_data["project_name"],
                task_description=sample_job_data["task_description"],
                task_type=sample_job_data["task_type"],
            )

            assert job is not None
            assert job.job_id is not None
            assert job.project_name == sample_job_data["project_name"]
            assert job.task_description == sample_job_data["task_description"]

    @pytest.mark.asyncio
    async def test_create_job_with_high_priority(self, sample_job_data):
        """Test that HIGH priority jobs get correct priority value."""
        with patch('controller.claude_backend.multi_scheduler') as mock_sched:
            mock_sched.running = True
            mock_sched.enqueue_job = AsyncMock()

            from controller.claude_backend import create_job, JobPriority

            job = await create_job(
                project_name=sample_job_data["project_name"],
                task_description=sample_job_data["task_description"],
                task_type=sample_job_data["task_type"],
                priority=JobPriority.HIGH.value,
            )

            assert job.priority == JobPriority.HIGH.value

    @pytest.mark.asyncio
    async def test_create_job_enqueues_to_scheduler(self, sample_job_data):
        """Test that created job is enqueued to scheduler."""
        with patch('controller.claude_backend.multi_scheduler') as mock_sched:
            mock_sched.running = True
            mock_sched.enqueue_job = AsyncMock()

            from controller.claude_backend import create_job

            job = await create_job(
                project_name=sample_job_data["project_name"],
                task_description=sample_job_data["task_description"],
                task_type=sample_job_data["task_type"],
            )

            # Verify enqueue was called
            mock_sched.enqueue_job.assert_called_once()
            # Verify the job was passed to enqueue
            call_args = mock_sched.enqueue_job.call_args
            assert call_args[0][0].job_id == job.job_id


# -----------------------------------------------------------------------------
# Test Cases: Job State Transitions
# -----------------------------------------------------------------------------
class TestJobStateTransitions:
    """Tests for job state machine."""

    def test_job_initial_state_is_queued(self):
        """Test that new jobs start in QUEUED state."""
        from controller.claude_backend import ClaudeJob, JobState

        job = ClaudeJob(
            job_id="test-123",
            project_name="test-project",
            task_description="test task",
            task_type="planning",
        )

        assert job.state == JobState.QUEUED

    def test_job_state_enum_values(self):
        """Test that JobState enum has expected values."""
        from controller.claude_backend import JobState

        expected_states = ["queued", "running", "completed", "failed", "cancelled"]
        actual_states = [s.value for s in JobState]

        for state in expected_states:
            assert state in actual_states, f"Missing state: {state}"


# -----------------------------------------------------------------------------
# Test Cases: Error Handling
# -----------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_create_job_fails_when_scheduler_stopped(self, sample_job_data):
        """Test that job creation fails gracefully when scheduler is stopped."""
        with patch('controller.claude_backend.multi_scheduler') as mock_sched:
            mock_sched.running = False

            from controller.claude_backend import create_job

            with pytest.raises(Exception):
                await create_job(
                    project_name=sample_job_data["project_name"],
                    task_description=sample_job_data["task_description"],
                    task_type=sample_job_data["task_type"],
                )


# -----------------------------------------------------------------------------
# Test Cases: Integration with phase12_router
# -----------------------------------------------------------------------------
class TestPhase12Integration:
    """Tests for integration with phase12_router project creation."""

    def test_phase12_router_imports_claude_backend(self):
        """Verify phase12_router can import claude_backend for job scheduling."""
        try:
            from controller.phase12_router import CLAUDE_BACKEND_AVAILABLE
            # Import should succeed
            assert isinstance(CLAUDE_BACKEND_AVAILABLE, bool)
        except ImportError as e:
            pytest.fail(f"phase12_router should import claude_backend: {e}")

    def test_phase12_router_has_job_scheduling(self):
        """Verify phase12_router includes job scheduling code."""
        router_path = Path(__file__).parent.parent / "controller" / "phase12_router.py"
        content = router_path.read_text()

        # Check for job scheduling code
        assert "create_claude_job" in content or "schedule" in content.lower(), \
            "phase12_router should have job scheduling code"

        # Check for planning job creation
        assert "planning" in content.lower(), \
            "phase12_router should create planning jobs"


# -----------------------------------------------------------------------------
# Test Cases: Workspace Setup
# -----------------------------------------------------------------------------
class TestWorkspaceSetup:
    """Tests for job workspace setup."""

    def test_workspace_includes_task_file(self):
        """Verify job workspace is configured with TASK.md."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        content = script_path.read_text()

        assert "TASK.md" in content, "Script should reference TASK.md"

    def test_workspace_has_logs_directory(self):
        """Verify job workspace creates logs directory."""
        script_path = Path(__file__).parent.parent / "scripts" / "run_claude_job.sh"
        content = script_path.read_text()

        assert "logs" in content, "Script should create logs directory"
        assert "stdout" in content.lower(), "Script should capture stdout"
        assert "stderr" in content.lower(), "Script should capture stderr"
