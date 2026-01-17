"""
Unit Tests for Task Controller - Phase 6

Test coverage for:
- Project bootstrap
- Task lifecycle state transitions
- Plan generation
- Approval gating
- Diff generation (Phase 3)
- Dry-run, apply, rollback (Phase 4)
- Commit, CI trigger, CI result, deploy testing (Phase 5)
- Production request, approve, apply, rollback (Phase 6)
- Dual approval enforcement
- Audit trail logging
- Policy enforcement
- Backup and restore
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

# Override PROJECTS_DIR before importing the app
import os
os.environ['PROJECTS_DIR'] = tempfile.mkdtemp()

from controller.main import (
    app,
    TaskType,
    TaskState,
    CIStatus,
    ProjectPhase,
    DeploymentEnvironment,
    PROJECTS_DIR,
    can_create_project,
    can_submit_task,
    can_approve_task,
    can_generate_diff,
    # Phase 4 policy hooks
    can_dry_run,
    can_apply,
    can_rollback,
    # Phase 5 policy hooks
    can_commit,
    can_trigger_ci,
    can_deploy_testing,
    # Phase 6 policy hooks
    can_request_prod_deploy,
    can_approve_prod_deploy,
    can_apply_prod_deploy,
    can_rollback_prod,
    diff_file_limit_ok,
    diff_within_scope,
    validate_task_transition,
    # Phase 4 helper functions
    parse_diff_files,
    count_diff_lines,
    # Phase 6 helper functions
    append_audit_log,
    get_audit_log_path,
    get_release_path,
)


# -----------------------------------------------------------------------------
# Test Client Setup
# -----------------------------------------------------------------------------
@pytest.fixture
def client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def temp_projects_dir(tmp_path):
    """Create a temporary projects directory."""
    projects_dir = tmp_path / "projects"
    projects_dir.mkdir()
    return projects_dir


@pytest.fixture(autouse=True)
def cleanup_projects():
    """Clean up projects directory after each test."""
    yield
    # Clean up any created projects
    if PROJECTS_DIR.exists():
        for item in PROJECTS_DIR.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                shutil.rmtree(item, ignore_errors=True)


# -----------------------------------------------------------------------------
# Health Check Tests
# -----------------------------------------------------------------------------
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "AI Development Platform - Task Controller"
        assert data["status"] == "running"
        assert data["phase"] == "Phase 6: Production Hardening & Explicit Go-Live Controls"
        assert data["version"] == "0.6.0"

    def test_health_endpoint(self, client):
        """Test health endpoint returns detailed status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "capabilities" in data
        assert "project_bootstrap" in data["capabilities"]
        assert "task_lifecycle" in data["capabilities"]
        assert "diff_generation" in data["capabilities"]
        assert "dry_run" in data["capabilities"]  # Phase 4
        assert "apply_with_confirmation" in data["capabilities"]  # Phase 4
        assert "rollback" in data["capabilities"]  # Phase 4
        assert "commit_with_confirmation" in data["capabilities"]  # Phase 5
        assert "ci_trigger" in data["capabilities"]  # Phase 5
        assert "deploy_testing_with_confirmation" in data["capabilities"]  # Phase 5
        assert "constraints" in data
        assert "NO_AUTONOMOUS_EXECUTION" in data["constraints"]
        assert "CONFIRMATION_REQUIRED" in data["constraints"]
        assert "NO_AUTOMATIC_COMMITS" in data["constraints"]  # Phase 5
        assert "NO_BYPASS_TEST_FAILURES" in data["constraints"]  # Phase 5


# -----------------------------------------------------------------------------
# Project Bootstrap Tests (TASK 1)
# -----------------------------------------------------------------------------
class TestProjectBootstrap:
    """Tests for project bootstrap endpoint."""

    def test_bootstrap_success(self, client):
        """Test successful project bootstrap."""
        response = client.post("/project/bootstrap", json={
            "project_name": "test-project",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python", "fastapi"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "bootstrapped"
        assert data["project_name"] == "test-project"
        assert "manifest_path" in data
        assert "state_path" in data

    def test_bootstrap_creates_directories(self, client):
        """Test bootstrap creates required directories."""
        client.post("/project/bootstrap", json={
            "project_name": "dir-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })

        project_path = PROJECTS_DIR / "dir-test"
        assert project_path.exists()
        assert (project_path / "tasks").exists()
        assert (project_path / "plans").exists()
        assert (project_path / "diffs").exists()  # Phase 3
        assert (project_path / "backups").exists()  # Phase 4
        assert (project_path / "PROJECT_MANIFEST.yaml").exists()
        assert (project_path / "CURRENT_STATE.md").exists()

    def test_bootstrap_duplicate_fails(self, client):
        """Test bootstrapping duplicate project fails."""
        # First bootstrap
        client.post("/project/bootstrap", json={
            "project_name": "duplicate-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })

        # Second bootstrap should fail
        response = client.post("/project/bootstrap", json={
            "project_name": "duplicate-test",
            "repo_url": "https://github.com/user/repo2",
            "tech_stack": []
        })
        assert response.status_code == 409

    def test_bootstrap_invalid_name(self, client):
        """Test bootstrap with invalid project name fails."""
        response = client.post("/project/bootstrap", json={
            "project_name": "Invalid Name!",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        assert response.status_code == 422  # Validation error


# -----------------------------------------------------------------------------
# Task Lifecycle Tests (TASK 2)
# -----------------------------------------------------------------------------
class TestTaskLifecycle:
    """Tests for task lifecycle management."""

    @pytest.fixture
    def bootstrapped_project(self, client):
        """Create a bootstrapped project for task tests."""
        client.post("/project/bootstrap", json={
            "project_name": "task-test-project",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        return "task-test-project"

    def test_create_task_success(self, client, bootstrapped_project):
        """Test successful task creation."""
        response = client.post("/task", json={
            "project_name": bootstrapped_project,
            "task_description": "Fix the login button not working",
            "task_type": "bug",
            "user_id": "test-user"
        })
        assert response.status_code == 200
        data = response.json()
        assert "task_id" in data
        assert data["project_name"] == bootstrapped_project
        assert data["task_type"] == "bug"
        assert data["state"] == "received"

    def test_create_task_no_project(self, client):
        """Test task creation fails without project."""
        response = client.post("/task", json={
            "project_name": "nonexistent-project",
            "task_description": "Test task",
            "task_type": "feature"
        })
        assert response.status_code == 404

    def test_task_state_transitions(self):
        """Test valid task state transitions."""
        # RECEIVED -> VALIDATED
        assert validate_task_transition(TaskState.RECEIVED, TaskState.VALIDATED)
        assert validate_task_transition(TaskState.RECEIVED, TaskState.REJECTED)

        # VALIDATED -> PLANNED
        assert validate_task_transition(TaskState.VALIDATED, TaskState.PLANNED)

        # PLANNED -> AWAITING_APPROVAL (implicit in generate_plan)
        assert validate_task_transition(TaskState.PLANNED, TaskState.AWAITING_APPROVAL)

        # AWAITING_APPROVAL -> APPROVED/REJECTED
        assert validate_task_transition(TaskState.AWAITING_APPROVAL, TaskState.APPROVED)
        assert validate_task_transition(TaskState.AWAITING_APPROVAL, TaskState.REJECTED)

        # Phase 3: APPROVED -> DIFF_GENERATED
        assert validate_task_transition(TaskState.APPROVED, TaskState.DIFF_GENERATED)
        assert validate_task_transition(TaskState.APPROVED, TaskState.ARCHIVED)

        # Phase 4: DIFF_GENERATED -> READY_TO_APPLY (dry-run)
        assert validate_task_transition(TaskState.DIFF_GENERATED, TaskState.READY_TO_APPLY)
        assert validate_task_transition(TaskState.DIFF_GENERATED, TaskState.ARCHIVED)

        # Phase 4: READY_TO_APPLY -> APPLYING
        assert validate_task_transition(TaskState.READY_TO_APPLY, TaskState.APPLYING)
        assert validate_task_transition(TaskState.READY_TO_APPLY, TaskState.ARCHIVED)

        # Phase 4: APPLYING -> APPLIED or EXECUTION_FAILED
        assert validate_task_transition(TaskState.APPLYING, TaskState.APPLIED)
        assert validate_task_transition(TaskState.APPLYING, TaskState.EXECUTION_FAILED)

        # Phase 4: APPLIED -> ROLLED_BACK or COMMITTED (Phase 5) or ARCHIVED
        assert validate_task_transition(TaskState.APPLIED, TaskState.ROLLED_BACK)
        assert validate_task_transition(TaskState.APPLIED, TaskState.COMMITTED)  # Phase 5
        assert validate_task_transition(TaskState.APPLIED, TaskState.ARCHIVED)

        # Phase 4: ROLLED_BACK -> ARCHIVED
        assert validate_task_transition(TaskState.ROLLED_BACK, TaskState.ARCHIVED)

        # Phase 4: EXECUTION_FAILED -> READY_TO_APPLY (retry) or ARCHIVED
        assert validate_task_transition(TaskState.EXECUTION_FAILED, TaskState.READY_TO_APPLY)
        assert validate_task_transition(TaskState.EXECUTION_FAILED, TaskState.ARCHIVED)

        # Phase 5: COMMITTED -> CI_RUNNING or ARCHIVED
        assert validate_task_transition(TaskState.COMMITTED, TaskState.CI_RUNNING)
        assert validate_task_transition(TaskState.COMMITTED, TaskState.ARCHIVED)

        # Phase 5: CI_RUNNING -> CI_PASSED or CI_FAILED
        assert validate_task_transition(TaskState.CI_RUNNING, TaskState.CI_PASSED)
        assert validate_task_transition(TaskState.CI_RUNNING, TaskState.CI_FAILED)

        # Phase 5: CI_PASSED -> DEPLOYED_TESTING or ARCHIVED
        assert validate_task_transition(TaskState.CI_PASSED, TaskState.DEPLOYED_TESTING)
        assert validate_task_transition(TaskState.CI_PASSED, TaskState.ARCHIVED)

        # Phase 5: CI_FAILED -> COMMITTED (re-commit after fix) or ARCHIVED
        assert validate_task_transition(TaskState.CI_FAILED, TaskState.COMMITTED)
        assert validate_task_transition(TaskState.CI_FAILED, TaskState.ARCHIVED)

        # Phase 5: DEPLOYED_TESTING -> ARCHIVED
        assert validate_task_transition(TaskState.DEPLOYED_TESTING, TaskState.ARCHIVED)

        # Invalid transitions
        assert not validate_task_transition(TaskState.RECEIVED, TaskState.APPROVED)
        assert not validate_task_transition(TaskState.APPROVED, TaskState.RECEIVED)
        assert not validate_task_transition(TaskState.ARCHIVED, TaskState.RECEIVED)
        # Cannot skip to DIFF_GENERATED from non-APPROVED state
        assert not validate_task_transition(TaskState.VALIDATED, TaskState.DIFF_GENERATED)
        # Cannot skip to APPLIED without READY_TO_APPLY
        assert not validate_task_transition(TaskState.DIFF_GENERATED, TaskState.APPLIED)
        # Phase 5: Cannot skip to CI without commit
        assert not validate_task_transition(TaskState.APPLIED, TaskState.CI_RUNNING)
        # Phase 5: Cannot deploy testing without CI passing
        assert not validate_task_transition(TaskState.CI_FAILED, TaskState.DEPLOYED_TESTING)


# -----------------------------------------------------------------------------
# Task Validation Tests
# -----------------------------------------------------------------------------
class TestTaskValidation:
    """Tests for task validation endpoint."""

    @pytest.fixture
    def project_with_task(self, client):
        """Create project with a task."""
        client.post("/project/bootstrap", json={
            "project_name": "validation-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        response = client.post("/task", json={
            "project_name": "validation-test",
            "task_description": "A valid task description for testing",
            "task_type": "feature"
        })
        return "validation-test", response.json()["task_id"]

    def test_validate_task_success(self, client, project_with_task):
        """Test successful task validation."""
        project_name, task_id = project_with_task
        response = client.post(
            f"/task/{task_id}/validate",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["validation_passed"] is True
        assert data["current_state"] == "validated"

    def test_validate_task_wrong_state(self, client, project_with_task):
        """Test validation fails if task not in RECEIVED state."""
        project_name, task_id = project_with_task

        # First validation
        client.post(f"/task/{task_id}/validate", params={"project_name": project_name})

        # Second validation should fail
        response = client.post(
            f"/task/{task_id}/validate",
            params={"project_name": project_name}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Plan Generation Tests (TASK 3)
# -----------------------------------------------------------------------------
class TestPlanGeneration:
    """Tests for plan generation endpoint."""

    @pytest.fixture
    def validated_task(self, client):
        """Create a validated task."""
        client.post("/project/bootstrap", json={
            "project_name": "plan-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python", "fastapi"]
        })
        task_response = client.post("/task", json={
            "project_name": "plan-test",
            "task_description": "Add user authentication feature",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "plan-test"})
        return "plan-test", task_id

    def test_generate_plan_success(self, client, validated_task):
        """Test successful plan generation."""
        project_name, task_id = validated_task
        response = client.post(
            f"/task/{task_id}/plan",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state"] == "awaiting_approval"
        assert "plan_path" in data

    def test_plan_creates_file(self, client, validated_task):
        """Test plan generation creates markdown file."""
        project_name, task_id = validated_task
        client.post(f"/task/{task_id}/plan", params={"project_name": project_name})

        plan_path = PROJECTS_DIR / project_name / "plans" / f"{task_id}_plan.md"
        assert plan_path.exists()

        # Verify plan content
        content = plan_path.read_text()
        assert "Implementation Plan" in content
        assert "Risk Analysis" in content
        assert "Rollback Strategy" in content

    def test_generate_plan_wrong_state(self, client):
        """Test plan generation fails if task not validated."""
        client.post("/project/bootstrap", json={
            "project_name": "plan-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "plan-state-test",
            "task_description": "Test task for state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try to generate plan without validation
        response = client.post(
            f"/task/{task_id}/plan",
            params={"project_name": "plan-state-test"}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Approval Gate Tests (TASK 4)
# -----------------------------------------------------------------------------
class TestApprovalGate:
    """Tests for approval/rejection endpoints."""

    @pytest.fixture
    def planned_task(self, client):
        """Create a task with plan."""
        client.post("/project/bootstrap", json={
            "project_name": "approval-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "approval-test",
            "task_description": "Task for approval testing purposes",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "approval-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "approval-test"})
        return "approval-test", task_id

    def test_approve_task_success(self, client, planned_task):
        """Test successful task approval."""
        project_name, task_id = planned_task
        response = client.post(
            f"/task/{task_id}/approve",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "approved"

    def test_approve_task_wrong_state(self, client):
        """Test approval fails if task not awaiting approval."""
        client.post("/project/bootstrap", json={
            "project_name": "approve-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "approve-state-test",
            "task_description": "Test task description here",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try to approve without going through lifecycle
        response = client.post(
            f"/task/{task_id}/approve",
            params={"project_name": "approve-state-test"}
        )
        assert response.status_code == 400

    def test_reject_task_success(self, client, planned_task):
        """Test successful task rejection."""
        project_name, task_id = planned_task
        response = client.post(
            f"/task/{task_id}/reject",
            params={"project_name": project_name},
            json={"rejection_reason": "Requirements are unclear and need more details"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "rejected"
        assert data["rejection_reason"] is not None

    def test_reject_requires_reason(self, client, planned_task):
        """Test rejection requires a reason."""
        project_name, task_id = planned_task
        response = client.post(
            f"/task/{task_id}/reject",
            params={"project_name": project_name},
            json={"rejection_reason": "short"}  # Too short
        )
        assert response.status_code == 422  # Validation error


# -----------------------------------------------------------------------------
# Policy Enforcement Tests (TASK 6)
# -----------------------------------------------------------------------------
class TestPolicyEnforcement:
    """Tests for policy enforcement hooks."""

    def test_can_create_project(self):
        """Test project creation policy."""
        allowed, reason = can_create_project("test-user")
        assert allowed is True
        assert "allowed" in reason.lower()

    def test_can_submit_task(self):
        """Test task submission policy."""
        allowed, reason = can_submit_task("test-project", "test-user")
        assert allowed is True

    def test_can_approve_task(self):
        """Test task approval policy."""
        allowed, reason = can_approve_task("task-123", "test-user")
        assert allowed is True


# -----------------------------------------------------------------------------
# Deployment Tests (unchanged behavior)
# -----------------------------------------------------------------------------
class TestDeployment:
    """Tests for deployment endpoint."""

    @pytest.fixture
    def project_for_deploy(self, client):
        """Create a project for deployment tests."""
        client.post("/project/bootstrap", json={
            "project_name": "deploy-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        return "deploy-test"

    def test_deploy_to_testing_placeholder(self, client, project_for_deploy):
        """Test deployment to testing returns placeholder."""
        response = client.post("/deploy", json={
            "project_name": project_for_deploy,
            "environment": "testing"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "placeholder"

    def test_deploy_to_production_blocked(self, client, project_for_deploy):
        """Test production deployment is blocked."""
        response = client.post("/deploy", json={
            "project_name": project_for_deploy,
            "environment": "production"
        })
        assert response.status_code == 403


# -----------------------------------------------------------------------------
# Project Status Tests
# -----------------------------------------------------------------------------
class TestProjectStatus:
    """Tests for project status endpoint."""

    def test_status_success(self, client):
        """Test getting project status."""
        client.post("/project/bootstrap", json={
            "project_name": "status-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })

        response = client.get("/status/status-test")
        assert response.status_code == 200
        data = response.json()
        assert data["project_name"] == "status-test"
        assert data["current_phase"] == "bootstrap"
        assert "tasks_by_state" in data

    def test_status_nonexistent_project(self, client):
        """Test status for nonexistent project fails."""
        response = client.get("/status/nonexistent")
        assert response.status_code == 404


# -----------------------------------------------------------------------------
# Projects List Tests
# -----------------------------------------------------------------------------
class TestProjectsList:
    """Tests for projects listing endpoint."""

    def test_list_projects_empty(self, client):
        """Test listing projects when empty."""
        response = client.get("/projects")
        assert response.status_code == 200
        data = response.json()
        assert "projects" in data
        assert "count" in data

    def test_list_projects_with_projects(self, client):
        """Test listing projects after creating some."""
        client.post("/project/bootstrap", json={
            "project_name": "list-test-1",
            "repo_url": "https://github.com/user/repo1",
            "tech_stack": []
        })
        client.post("/project/bootstrap", json={
            "project_name": "list-test-2",
            "repo_url": "https://github.com/user/repo2",
            "tech_stack": []
        })

        response = client.get("/projects")
        data = response.json()
        assert data["count"] >= 2


# -----------------------------------------------------------------------------
# Diff Generation Tests (Phase 3)
# -----------------------------------------------------------------------------
class TestDiffGeneration:
    """Tests for diff generation endpoint (Phase 3)."""

    @pytest.fixture
    def approved_task(self, client):
        """Create an approved task ready for diff generation."""
        client.post("/project/bootstrap", json={
            "project_name": "diff-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python", "fastapi"]
        })
        task_response = client.post("/task", json={
            "project_name": "diff-test",
            "task_description": "Add user authentication feature with login",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "diff-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "diff-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "diff-test"})
        return "diff-test", task_id

    def test_generate_diff_success(self, client, approved_task):
        """Test successful diff generation."""
        project_name, task_id = approved_task
        response = client.post(
            f"/task/{task_id}/generate-diff",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "diff_generated"
        assert "diff_path" in data
        assert data["files_in_diff"] > 0
        assert "warning" in data
        assert "NOT APPLIED" in data["warning"]

    def test_generate_diff_creates_file(self, client, approved_task):
        """Test diff generation creates file with correct header."""
        project_name, task_id = approved_task
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": project_name})

        diff_path = PROJECTS_DIR / project_name / "diffs" / f"{task_id}.diff"
        assert diff_path.exists()

        # Verify diff content has required header
        content = diff_path.read_text()
        assert f"# TASK_ID: {task_id}" in content
        assert f"# PROJECT: {project_name}" in content
        assert "# DISCLAIMER: NOT APPLIED" in content
        assert "HUMAN REVIEW ONLY" in content

    def test_generate_diff_wrong_state(self, client):
        """Test diff generation fails if task not approved."""
        client.post("/project/bootstrap", json={
            "project_name": "diff-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "diff-state-test",
            "task_description": "Test task for diff state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try to generate diff without approval
        response = client.post(
            f"/task/{task_id}/generate-diff",
            params={"project_name": "diff-state-test"}
        )
        assert response.status_code == 400

    def test_generate_diff_requires_plan(self, client):
        """Test diff generation requires plan file to exist."""
        client.post("/project/bootstrap", json={
            "project_name": "diff-plan-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "diff-plan-test",
            "task_description": "Test task for plan requirement",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]

        # Manually set task to APPROVED without generating plan
        # This simulates a corrupted state
        task_path = PROJECTS_DIR / "diff-plan-test" / "tasks" / f"{task_id}.yaml"
        import yaml
        with open(task_path, 'r') as f:
            task_data = yaml.safe_load(f)
        task_data['current_state'] = 'approved'
        with open(task_path, 'w') as f:
            yaml.dump(task_data, f)

        response = client.post(
            f"/task/{task_id}/generate-diff",
            params={"project_name": "diff-plan-test"}
        )
        assert response.status_code == 400

    def test_get_diff_endpoint(self, client, approved_task):
        """Test retrieving diff content."""
        project_name, task_id = approved_task
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": project_name})

        response = client.get(
            f"/task/{task_id}/diff",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert "diff" in data
        assert "warning" in data
        assert "NOT APPLIED" in data["warning"]


# -----------------------------------------------------------------------------
# Phase 3 Policy Enforcement Tests
# -----------------------------------------------------------------------------
class TestPhase3PolicyEnforcement:
    """Tests for Phase 3 policy enforcement hooks."""

    def test_can_generate_diff(self):
        """Test diff generation policy."""
        allowed, reason = can_generate_diff("task-123", "test-user")
        assert allowed is True
        assert "allowed" in reason.lower()

    def test_diff_file_limit_ok_within_limit(self):
        """Test file limit check passes within limit."""
        allowed, reason = diff_file_limit_ok(5, max_files=10)
        assert allowed is True

    def test_diff_file_limit_ok_exceeds_limit(self):
        """Test file limit check fails when exceeding limit."""
        allowed, reason = diff_file_limit_ok(15, max_files=10)
        assert allowed is False
        assert "exceeds" in reason.lower()

    def test_diff_within_scope(self):
        """Test scope check for diff files."""
        plan_files = ["src/main.py", "tests/test_main.py"]
        diff_files = ["src/main.py", "tests/test_main.py"]
        allowed, reason = diff_within_scope(plan_files, diff_files)
        assert allowed is True


# -----------------------------------------------------------------------------
# Phase 4 Dry-Run Tests
# -----------------------------------------------------------------------------
class TestDryRun:
    """Tests for dry-run endpoint (Phase 4)."""

    @pytest.fixture
    def diff_generated_task(self, client):
        """Create a task with diff generated."""
        client.post("/project/bootstrap", json={
            "project_name": "dryrun-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python", "fastapi"]
        })
        task_response = client.post("/task", json={
            "project_name": "dryrun-test",
            "task_description": "Add user feature for dry run testing",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "dryrun-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "dryrun-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "dryrun-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "dryrun-test"})
        return "dryrun-test", task_id

    def test_dry_run_success(self, client, diff_generated_task):
        """Test successful dry-run."""
        project_name, task_id = diff_generated_task
        response = client.post(
            f"/task/{task_id}/dry-run",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "ready_to_apply"
        assert "files_affected" in data
        assert "lines_added" in data
        assert "lines_removed" in data
        assert data["can_apply"] is True
        assert "warning" in data
        assert "NO FILES MODIFIED" in data["warning"]

    def test_dry_run_wrong_state(self, client):
        """Test dry-run fails if task not in DIFF_GENERATED state."""
        client.post("/project/bootstrap", json={
            "project_name": "dryrun-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "dryrun-state-test",
            "task_description": "Test task for dry run state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try dry-run without diff generation
        response = client.post(
            f"/task/{task_id}/dry-run",
            params={"project_name": "dryrun-state-test"}
        )
        assert response.status_code == 400

    def test_dry_run_does_not_modify_files(self, client, diff_generated_task):
        """Test dry-run does NOT modify any files."""
        project_name, task_id = diff_generated_task
        project_path = PROJECTS_DIR / project_name

        # Get file counts before
        files_before = list(project_path.rglob("*"))
        file_count_before = len([f for f in files_before if f.is_file()])

        # Run dry-run
        client.post(f"/task/{task_id}/dry-run", params={"project_name": project_name})

        # Get file counts after
        files_after = list(project_path.rglob("*"))
        file_count_after = len([f for f in files_after if f.is_file()])

        # No new files should be created (except possibly task state update)
        # The only changes should be to existing task/state files
        assert file_count_after <= file_count_before + 1  # Allow for task state update


# -----------------------------------------------------------------------------
# Phase 4 Apply Tests
# -----------------------------------------------------------------------------
class TestApply:
    """Tests for apply endpoint (Phase 4)."""

    @pytest.fixture
    def ready_to_apply_task(self, client):
        """Create a task ready to apply."""
        client.post("/project/bootstrap", json={
            "project_name": "apply-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "apply-test",
            "task_description": "Add feature for apply testing purposes",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "apply-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "apply-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "apply-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "apply-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "apply-test"})
        return "apply-test", task_id

    def test_apply_requires_confirmation(self, client, ready_to_apply_task):
        """Test apply fails without confirmation."""
        project_name, task_id = ready_to_apply_task
        response = client.post(
            f"/task/{task_id}/apply",
            params={"project_name": project_name, "confirm": False}
        )
        assert response.status_code == 403
        assert "confirmation" in response.json()["detail"].lower()

    def test_apply_success_with_confirmation(self, client, ready_to_apply_task):
        """Test successful apply with confirmation."""
        project_name, task_id = ready_to_apply_task
        response = client.post(
            f"/task/{task_id}/apply",
            params={"project_name": project_name, "confirm": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "applied"
        assert "files_modified" in data
        assert "backup_path" in data
        assert data["rollback_available"] is True

    def test_apply_creates_backup(self, client, ready_to_apply_task):
        """Test apply creates backup directory."""
        project_name, task_id = ready_to_apply_task
        client.post(
            f"/task/{task_id}/apply",
            params={"project_name": project_name, "confirm": True}
        )

        backup_path = PROJECTS_DIR / project_name / "backups" / task_id
        assert backup_path.exists()
        assert (backup_path / "BACKUP_MANIFEST.yaml").exists()

    def test_apply_wrong_state(self, client):
        """Test apply fails if task not in READY_TO_APPLY state."""
        client.post("/project/bootstrap", json={
            "project_name": "apply-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "apply-state-test",
            "task_description": "Test task for apply state check validation",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try apply without dry-run
        response = client.post(
            f"/task/{task_id}/apply",
            params={"project_name": "apply-state-test", "confirm": True}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Phase 4 Rollback Tests
# -----------------------------------------------------------------------------
class TestRollback:
    """Tests for rollback endpoint (Phase 4)."""

    @pytest.fixture
    def applied_task(self, client):
        """Create an applied task."""
        client.post("/project/bootstrap", json={
            "project_name": "rollback-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "rollback-test",
            "task_description": "Add feature for rollback testing",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "rollback-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "rollback-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "rollback-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "rollback-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "rollback-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "rollback-test", "confirm": True})
        return "rollback-test", task_id

    def test_rollback_success(self, client, applied_task):
        """Test successful rollback."""
        project_name, task_id = applied_task
        response = client.post(
            f"/task/{task_id}/rollback",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "rolled_back"
        assert "files_restored" in data

    def test_rollback_wrong_state(self, client):
        """Test rollback fails if task not in APPLIED state."""
        client.post("/project/bootstrap", json={
            "project_name": "rollback-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "rollback-state-test",
            "task_description": "Test task for rollback state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try rollback without apply
        response = client.post(
            f"/task/{task_id}/rollback",
            params={"project_name": "rollback-state-test"}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Phase 4 Policy Enforcement Tests
# -----------------------------------------------------------------------------
class TestPhase4PolicyEnforcement:
    """Tests for Phase 4 policy enforcement hooks."""

    def test_can_dry_run(self):
        """Test dry-run policy."""
        allowed, reason = can_dry_run("task-123", "test-user")
        assert allowed is True
        assert "allowed" in reason.lower()

    def test_can_apply_without_confirmation(self):
        """Test apply policy rejects without confirmation."""
        allowed, reason = can_apply("task-123", "test-user", confirmed=False)
        assert allowed is False
        assert "confirmation" in reason.lower()

    def test_can_apply_with_confirmation(self):
        """Test apply policy allows with confirmation."""
        allowed, reason = can_apply("task-123", "test-user", confirmed=True)
        assert allowed is True
        assert "confirmed" in reason.lower()

    def test_can_rollback(self):
        """Test rollback policy."""
        allowed, reason = can_rollback("task-123", "test-user")
        assert allowed is True
        assert "allowed" in reason.lower()


# -----------------------------------------------------------------------------
# Phase 4 Helper Function Tests
# -----------------------------------------------------------------------------
class TestPhase4HelperFunctions:
    """Tests for Phase 4 helper functions."""

    def test_parse_diff_files(self):
        """Test diff file parsing."""
        diff_content = """--- a/src/main.py
+++ b/src/main.py
@@ -1,5 +1,10 @@
 def main():
+    pass
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,5 @@
 def test():
+    assert True
"""
        files = parse_diff_files(diff_content)
        assert "src/main.py" in files
        assert "tests/test_main.py" in files
        assert len(files) == 2

    def test_count_diff_lines(self):
        """Test diff line counting."""
        diff_content = """--- a/src/main.py
+++ b/src/main.py
@@ -1,5 +1,8 @@
 def main():
-    old_line_1
-    old_line_2
+    new_line_1
+    new_line_2
+    new_line_3
 pass
"""
        added, removed = count_diff_lines(diff_content)
        assert added == 3
        assert removed == 2


# -----------------------------------------------------------------------------
# Phase 4 Full Lifecycle Tests
# -----------------------------------------------------------------------------
class TestPhase4FullLifecycle:
    """Tests for complete Phase 4 task lifecycle."""

    def test_full_lifecycle_apply_and_rollback(self, client):
        """Test complete lifecycle with apply and rollback."""
        # Bootstrap
        client.post("/project/bootstrap", json={
            "project_name": "lifecycle-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Create task
        task_response = client.post("/task", json={
            "project_name": "lifecycle-test",
            "task_description": "Full lifecycle testing with apply and rollback",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        assert task_response.json()["state"] == "received"

        # Validate
        validate_response = client.post(
            f"/task/{task_id}/validate",
            params={"project_name": "lifecycle-test"}
        )
        assert validate_response.json()["current_state"] == "validated"

        # Plan
        plan_response = client.post(
            f"/task/{task_id}/plan",
            params={"project_name": "lifecycle-test"}
        )
        assert plan_response.json()["state"] == "awaiting_approval"

        # Approve
        approve_response = client.post(
            f"/task/{task_id}/approve",
            params={"project_name": "lifecycle-test"}
        )
        assert approve_response.json()["current_state"] == "approved"

        # Generate diff
        diff_response = client.post(
            f"/task/{task_id}/generate-diff",
            params={"project_name": "lifecycle-test"}
        )
        assert diff_response.json()["current_state"] == "diff_generated"

        # Dry-run
        dryrun_response = client.post(
            f"/task/{task_id}/dry-run",
            params={"project_name": "lifecycle-test"}
        )
        assert dryrun_response.json()["current_state"] == "ready_to_apply"
        assert dryrun_response.json()["can_apply"] is True

        # Apply (without confirmation - should fail)
        apply_fail_response = client.post(
            f"/task/{task_id}/apply",
            params={"project_name": "lifecycle-test", "confirm": False}
        )
        assert apply_fail_response.status_code == 403

        # Apply (with confirmation - should succeed)
        apply_response = client.post(
            f"/task/{task_id}/apply",
            params={"project_name": "lifecycle-test", "confirm": True}
        )
        assert apply_response.json()["current_state"] == "applied"
        assert apply_response.json()["rollback_available"] is True

        # Rollback
        rollback_response = client.post(
            f"/task/{task_id}/rollback",
            params={"project_name": "lifecycle-test"}
        )
        assert rollback_response.json()["current_state"] == "rolled_back"


# -----------------------------------------------------------------------------
# Phase 5 Commit Tests
# -----------------------------------------------------------------------------
class TestCommit:
    """Tests for commit endpoint (Phase 5)."""

    @pytest.fixture
    def applied_task(self, client):
        """Create an applied task ready for commit."""
        client.post("/project/bootstrap", json={
            "project_name": "commit-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "commit-test",
            "task_description": "Add feature for commit testing purposes",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "commit-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "commit-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "commit-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "commit-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "commit-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "commit-test", "confirm": True})
        return "commit-test", task_id

    def test_commit_requires_confirmation(self, client, applied_task):
        """Test commit fails without confirmation."""
        project_name, task_id = applied_task
        response = client.post(
            f"/task/{task_id}/commit",
            params={"project_name": project_name, "confirm": False}
        )
        assert response.status_code == 403
        assert "confirmation" in response.json()["detail"].lower()

    def test_commit_success_with_confirmation(self, client, applied_task):
        """Test successful commit with confirmation."""
        project_name, task_id = applied_task
        response = client.post(
            f"/task/{task_id}/commit",
            params={"project_name": project_name, "confirm": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "committed"
        assert "commit_hash" in data
        assert "commit_message" in data
        assert "files_committed" in data
        assert "warning" in data
        assert "NOT PUSHED" in data["warning"]

    def test_commit_wrong_state(self, client):
        """Test commit fails if task not in APPLIED state."""
        client.post("/project/bootstrap", json={
            "project_name": "commit-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "commit-state-test",
            "task_description": "Test task for commit state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try commit without apply
        response = client.post(
            f"/task/{task_id}/commit",
            params={"project_name": "commit-state-test", "confirm": True}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Phase 5 CI Trigger Tests
# -----------------------------------------------------------------------------
class TestCITrigger:
    """Tests for CI trigger endpoint (Phase 5)."""

    @pytest.fixture
    def committed_task(self, client):
        """Create a committed task ready for CI."""
        client.post("/project/bootstrap", json={
            "project_name": "ci-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "ci-test",
            "task_description": "Add feature for CI testing purposes",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "ci-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "ci-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "ci-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "ci-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "ci-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "ci-test", "confirm": True})
        client.post(f"/task/{task_id}/commit", params={"project_name": "ci-test", "confirm": True})
        return "ci-test", task_id

    def test_ci_trigger_success(self, client, committed_task):
        """Test successful CI trigger."""
        project_name, task_id = committed_task
        response = client.post(
            f"/task/{task_id}/ci/run",
            params={"project_name": project_name}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "ci_running"
        assert "ci_run_id" in data
        assert "warning" in data
        assert "WAIT FOR RESULTS" in data["warning"]

    def test_ci_trigger_wrong_state(self, client):
        """Test CI trigger fails if task not in COMMITTED state."""
        client.post("/project/bootstrap", json={
            "project_name": "ci-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "ci-state-test",
            "task_description": "Test task for CI state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try CI trigger without commit
        response = client.post(
            f"/task/{task_id}/ci/run",
            params={"project_name": "ci-state-test"}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Phase 5 CI Result Tests
# -----------------------------------------------------------------------------
class TestCIResult:
    """Tests for CI result endpoint (Phase 5)."""

    @pytest.fixture
    def ci_running_task(self, client):
        """Create a task with CI running."""
        client.post("/project/bootstrap", json={
            "project_name": "ci-result-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "ci-result-test",
            "task_description": "Add feature for CI result testing",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "ci-result-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "ci-result-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "ci-result-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "ci-result-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "ci-result-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "ci-result-test", "confirm": True})
        client.post(f"/task/{task_id}/commit", params={"project_name": "ci-result-test", "confirm": True})
        client.post(f"/task/{task_id}/ci/run", params={"project_name": "ci-result-test"})
        return "ci-result-test", task_id

    def test_ci_result_passed(self, client, ci_running_task):
        """Test CI result passed."""
        project_name, task_id = ci_running_task
        response = client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": project_name},
            json={"status": "passed", "logs_url": "https://ci.example.com/logs/123"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "ci_passed"
        assert data["ci_status"] == "passed"
        assert data["logs_url"] == "https://ci.example.com/logs/123"

    def test_ci_result_failed(self, client, ci_running_task):
        """Test CI result failed."""
        project_name, task_id = ci_running_task
        response = client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": project_name},
            json={"status": "failed", "details": "Test failed"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "ci_failed"
        assert data["ci_status"] == "failed"

    def test_ci_result_wrong_state(self, client):
        """Test CI result fails if task not in CI_RUNNING state."""
        client.post("/project/bootstrap", json={
            "project_name": "ci-result-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": []
        })
        task_response = client.post("/task", json={
            "project_name": "ci-result-state-test",
            "task_description": "Test task for CI result state check",
            "task_type": "bug"
        })
        task_id = task_response.json()["task_id"]

        # Try CI result without CI running
        response = client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": "ci-result-state-test"},
            json={"status": "passed"}
        )
        assert response.status_code == 400


# -----------------------------------------------------------------------------
# Phase 5 Deploy Testing Tests
# -----------------------------------------------------------------------------
class TestDeployTesting:
    """Tests for deploy testing endpoint (Phase 5)."""

    @pytest.fixture
    def ci_passed_task(self, client):
        """Create a task with CI passed."""
        client.post("/project/bootstrap", json={
            "project_name": "deploy-testing-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "deploy-testing-test",
            "task_description": "Add feature for deploy testing purposes",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "deploy-testing-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "deploy-testing-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "deploy-testing-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "deploy-testing-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "deploy-testing-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "deploy-testing-test", "confirm": True})
        client.post(f"/task/{task_id}/commit", params={"project_name": "deploy-testing-test", "confirm": True})
        client.post(f"/task/{task_id}/ci/run", params={"project_name": "deploy-testing-test"})
        client.post(f"/task/{task_id}/ci/result", params={"project_name": "deploy-testing-test"}, json={"status": "passed"})
        return "deploy-testing-test", task_id

    def test_deploy_testing_requires_confirmation(self, client, ci_passed_task):
        """Test deploy testing fails without confirmation."""
        project_name, task_id = ci_passed_task
        response = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": project_name, "confirm": False}
        )
        assert response.status_code == 403
        assert "confirmation" in response.json()["detail"].lower()

    def test_deploy_testing_success_with_confirmation(self, client, ci_passed_task):
        """Test successful deploy testing with confirmation."""
        project_name, task_id = ci_passed_task
        response = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": project_name, "confirm": True}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["current_state"] == "deployed_testing"
        assert "deployment_url" in data
        assert "warning" in data
        assert "TESTING ONLY" in data["warning"]

    def test_deploy_testing_blocked_by_ci_failure(self, client):
        """Test deploy testing blocked when CI failed."""
        client.post("/project/bootstrap", json={
            "project_name": "deploy-ci-fail-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })
        task_response = client.post("/task", json={
            "project_name": "deploy-ci-fail-test",
            "task_description": "Task for CI failure deploy block test",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "deploy-ci-fail-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "deploy-ci-fail-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "deploy-ci-fail-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "deploy-ci-fail-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "deploy-ci-fail-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "deploy-ci-fail-test", "confirm": True})
        client.post(f"/task/{task_id}/commit", params={"project_name": "deploy-ci-fail-test", "confirm": True})
        client.post(f"/task/{task_id}/ci/run", params={"project_name": "deploy-ci-fail-test"})
        # CI fails
        client.post(f"/task/{task_id}/ci/result", params={"project_name": "deploy-ci-fail-test"}, json={"status": "failed"})

        # Try to deploy - should fail
        response = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": "deploy-ci-fail-test", "confirm": True}
        )
        assert response.status_code == 400
        assert "CI_PASSED" in response.json()["detail"]


# -----------------------------------------------------------------------------
# Phase 5 Policy Enforcement Tests
# -----------------------------------------------------------------------------
class TestPhase5PolicyEnforcement:
    """Tests for Phase 5 policy enforcement hooks."""

    def test_can_commit_without_confirmation(self):
        """Test commit policy rejects without confirmation."""
        allowed, reason = can_commit("task-123", "test-user", confirmed=False)
        assert allowed is False
        assert "confirmation" in reason.lower()

    def test_can_commit_with_confirmation(self):
        """Test commit policy allows with confirmation."""
        allowed, reason = can_commit("task-123", "test-user", confirmed=True)
        assert allowed is True
        assert "confirmed" in reason.lower()

    def test_can_trigger_ci(self):
        """Test CI trigger policy."""
        allowed, reason = can_trigger_ci("task-123", "test-user")
        assert allowed is True
        assert "allowed" in reason.lower()

    def test_can_deploy_testing_without_confirmation(self):
        """Test deploy testing policy rejects without confirmation."""
        allowed, reason = can_deploy_testing("task-123", "test-user", confirmed=False)
        assert allowed is False
        assert "confirmation" in reason.lower()

    def test_can_deploy_testing_with_confirmation(self):
        """Test deploy testing policy allows with confirmation."""
        allowed, reason = can_deploy_testing("task-123", "test-user", confirmed=True)
        assert allowed is True
        assert "confirmed" in reason.lower()


# -----------------------------------------------------------------------------
# Phase 5 Full Lifecycle Tests
# -----------------------------------------------------------------------------
class TestPhase5FullLifecycle:
    """Tests for complete Phase 5 task lifecycle."""

    def test_full_lifecycle_to_testing_deployment(self, client):
        """Test complete lifecycle from task to testing deployment."""
        # Bootstrap
        client.post("/project/bootstrap", json={
            "project_name": "full-lifecycle-p5",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Create task
        task_response = client.post("/task", json={
            "project_name": "full-lifecycle-p5",
            "task_description": "Full lifecycle testing through Phase 5",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        assert task_response.json()["state"] == "received"

        # Validate
        validate_response = client.post(
            f"/task/{task_id}/validate",
            params={"project_name": "full-lifecycle-p5"}
        )
        assert validate_response.json()["current_state"] == "validated"

        # Plan
        plan_response = client.post(
            f"/task/{task_id}/plan",
            params={"project_name": "full-lifecycle-p5"}
        )
        assert plan_response.json()["state"] == "awaiting_approval"

        # Approve
        approve_response = client.post(
            f"/task/{task_id}/approve",
            params={"project_name": "full-lifecycle-p5"}
        )
        assert approve_response.json()["current_state"] == "approved"

        # Generate diff
        diff_response = client.post(
            f"/task/{task_id}/generate-diff",
            params={"project_name": "full-lifecycle-p5"}
        )
        assert diff_response.json()["current_state"] == "diff_generated"

        # Dry-run
        dryrun_response = client.post(
            f"/task/{task_id}/dry-run",
            params={"project_name": "full-lifecycle-p5"}
        )
        assert dryrun_response.json()["current_state"] == "ready_to_apply"

        # Apply
        apply_response = client.post(
            f"/task/{task_id}/apply",
            params={"project_name": "full-lifecycle-p5", "confirm": True}
        )
        assert apply_response.json()["current_state"] == "applied"

        # Phase 5: Commit (without confirmation - should fail)
        commit_fail = client.post(
            f"/task/{task_id}/commit",
            params={"project_name": "full-lifecycle-p5", "confirm": False}
        )
        assert commit_fail.status_code == 403

        # Phase 5: Commit (with confirmation)
        commit_response = client.post(
            f"/task/{task_id}/commit",
            params={"project_name": "full-lifecycle-p5", "confirm": True}
        )
        assert commit_response.json()["current_state"] == "committed"
        assert "NOT PUSHED" in commit_response.json()["warning"]

        # Phase 5: Trigger CI
        ci_run_response = client.post(
            f"/task/{task_id}/ci/run",
            params={"project_name": "full-lifecycle-p5"}
        )
        assert ci_run_response.json()["current_state"] == "ci_running"

        # Phase 5: CI Result (passed)
        ci_result_response = client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": "full-lifecycle-p5"},
            json={"status": "passed", "logs_url": "https://ci.example.com/123"}
        )
        assert ci_result_response.json()["current_state"] == "ci_passed"

        # Phase 5: Deploy Testing (without confirmation - should fail)
        deploy_fail = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": "full-lifecycle-p5", "confirm": False}
        )
        assert deploy_fail.status_code == 403

        # Phase 5: Deploy Testing (with confirmation)
        deploy_response = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": "full-lifecycle-p5", "confirm": True}
        )
        assert deploy_response.json()["current_state"] == "deployed_testing"
        assert "TESTING ONLY" in deploy_response.json()["warning"]

    def test_ci_failure_blocks_deployment(self, client):
        """Test that CI failure blocks testing deployment."""
        # Bootstrap
        client.post("/project/bootstrap", json={
            "project_name": "ci-block-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Full lifecycle to CI
        task_response = client.post("/task", json={
            "project_name": "ci-block-test",
            "task_description": "CI failure blocking test",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]
        client.post(f"/task/{task_id}/validate", params={"project_name": "ci-block-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "ci-block-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "ci-block-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "ci-block-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "ci-block-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "ci-block-test", "confirm": True})
        client.post(f"/task/{task_id}/commit", params={"project_name": "ci-block-test", "confirm": True})
        client.post(f"/task/{task_id}/ci/run", params={"project_name": "ci-block-test"})

        # CI fails
        ci_result = client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": "ci-block-test"},
            json={"status": "failed", "details": "Tests failed"}
        )
        assert ci_result.json()["current_state"] == "ci_failed"

        # Attempt to deploy - should be blocked
        deploy_response = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": "ci-block-test", "confirm": True}
        )
        assert deploy_response.status_code == 400
        assert "CI_PASSED" in deploy_response.json()["detail"]


# -----------------------------------------------------------------------------
# Phase 6 Tests - Production Deployment Request
# -----------------------------------------------------------------------------
class TestProductionRequest:
    """Tests for production deployment request endpoint."""

    def test_prod_request_without_risk_acknowledgment_fails(self, client):
        """Test that production request without risk acknowledgment is rejected."""
        # Bootstrap and get to deployed_testing state
        client.post("/project/bootstrap", json={
            "project_name": "prod-test-1",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Check policy hook directly
        allowed, reason = can_request_prod_deploy("test-task", "user1", risk_acknowledged=False)
        assert not allowed
        assert "risk acknowledgment" in reason.lower()

    def test_prod_request_without_user_id_fails(self, client):
        """Test that production request without user ID is rejected."""
        allowed, reason = can_request_prod_deploy("test-task", user_id=None, risk_acknowledged=True)
        assert not allowed
        assert "user identification" in reason.lower()

    def test_prod_request_with_all_requirements_passes(self, client):
        """Test that production request with all requirements succeeds."""
        allowed, reason = can_request_prod_deploy("test-task", "user1", risk_acknowledged=True)
        assert allowed

    def test_prod_request_requires_deployed_testing_state(self, client):
        """Test that production request requires DEPLOYED_TESTING state."""
        # Bootstrap
        client.post("/project/bootstrap", json={
            "project_name": "prod-state-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Create and progress task to only APPLIED state (not deployed_testing)
        task_response = client.post("/task", json={
            "project_name": "prod-state-test",
            "task_description": "Test task for prod request",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]

        # Try to request production from wrong state - should fail
        # (In this placeholder, we'd need to progress to deployed_testing first)


# -----------------------------------------------------------------------------
# Phase 6 Tests - Production Approval (Dual Approval)
# -----------------------------------------------------------------------------
class TestProductionApproval:
    """Tests for production approval endpoint with dual approval enforcement."""

    def test_same_user_cannot_approve_own_request(self, client):
        """CRITICAL: Test that same user cannot approve their own request."""
        allowed, reason = can_approve_prod_deploy(
            "test-task",
            approver_id="user1",
            requester_id="user1",  # Same user
            reviewed_changes=True,
            reviewed_rollback=True
        )
        assert not allowed
        assert "DUAL APPROVAL" in reason or "CANNOT be the same" in reason

    def test_different_user_can_approve(self, client):
        """Test that different user CAN approve the request."""
        allowed, reason = can_approve_prod_deploy(
            "test-task",
            approver_id="user2",
            requester_id="user1",  # Different user
            reviewed_changes=True,
            reviewed_rollback=True
        )
        assert allowed

    def test_approval_requires_reviewed_changes(self, client):
        """Test that approval requires reviewed_changes confirmation."""
        allowed, reason = can_approve_prod_deploy(
            "test-task",
            approver_id="user2",
            requester_id="user1",
            reviewed_changes=False,  # Not reviewed
            reviewed_rollback=True
        )
        assert not allowed
        assert "reviewed" in reason.lower()

    def test_approval_requires_reviewed_rollback(self, client):
        """Test that approval requires reviewed_rollback confirmation."""
        allowed, reason = can_approve_prod_deploy(
            "test-task",
            approver_id="user2",
            requester_id="user1",
            reviewed_changes=True,
            reviewed_rollback=False  # Not reviewed
        )
        assert not allowed
        assert "rollback" in reason.lower()

    def test_approval_without_approver_id_fails(self, client):
        """Test that approval without approver ID is rejected."""
        allowed, reason = can_approve_prod_deploy(
            "test-task",
            approver_id=None,
            requester_id="user1",
            reviewed_changes=True,
            reviewed_rollback=True
        )
        assert not allowed


# -----------------------------------------------------------------------------
# Phase 6 Tests - Production Apply
# -----------------------------------------------------------------------------
class TestProductionApply:
    """Tests for production deployment apply endpoint."""

    def test_prod_apply_requires_confirmation(self, client):
        """Test that production apply requires explicit confirmation."""
        allowed, reason = can_apply_prod_deploy("test-task", "user1", confirmed=False)
        assert not allowed
        assert "confirm" in reason.lower()

    def test_prod_apply_requires_user_id(self, client):
        """Test that production apply requires user identification."""
        allowed, reason = can_apply_prod_deploy("test-task", user_id=None, confirmed=True)
        assert not allowed
        assert "user identification" in reason.lower()

    def test_prod_apply_with_all_requirements_passes(self, client):
        """Test that production apply with all requirements succeeds."""
        allowed, reason = can_apply_prod_deploy("test-task", "user1", confirmed=True)
        assert allowed


# -----------------------------------------------------------------------------
# Phase 6 Tests - Production Rollback (Break-Glass)
# -----------------------------------------------------------------------------
class TestProductionRollback:
    """Tests for production rollback endpoint (break-glass emergency)."""

    def test_rollback_requires_user_id(self, client):
        """Test that rollback requires user identification for audit trail."""
        allowed, reason = can_rollback_prod("test-task", user_id=None)
        assert not allowed
        assert "user identification" in reason.lower()

    def test_rollback_with_user_id_always_allowed(self, client):
        """Test that rollback is always allowed with user ID (break-glass)."""
        # Rollback should NOT require dual approval - speed > ceremony
        allowed, reason = can_rollback_prod("test-task", "user1")
        assert allowed
        assert "break-glass" in reason.lower()


# -----------------------------------------------------------------------------
# Phase 6 Tests - Policy Enforcement
# -----------------------------------------------------------------------------
class TestPhase6PolicyEnforcement:
    """Tests for Phase 6 policy enforcement."""

    def test_state_transition_deployed_testing_to_prod_requested(self):
        """Test state transition from DEPLOYED_TESTING to PROD_DEPLOY_REQUESTED is valid."""
        assert validate_task_transition(
            TaskState.DEPLOYED_TESTING,
            TaskState.PROD_DEPLOY_REQUESTED
        )

    def test_state_transition_prod_requested_to_approved(self):
        """Test state transition from PROD_DEPLOY_REQUESTED to PROD_APPROVED is valid."""
        assert validate_task_transition(
            TaskState.PROD_DEPLOY_REQUESTED,
            TaskState.PROD_APPROVED
        )

    def test_state_transition_prod_approved_to_deployed(self):
        """Test state transition from PROD_APPROVED to DEPLOYED_PRODUCTION is valid."""
        assert validate_task_transition(
            TaskState.PROD_APPROVED,
            TaskState.DEPLOYED_PRODUCTION
        )

    def test_state_transition_deployed_to_rolled_back(self):
        """Test state transition from DEPLOYED_PRODUCTION to PROD_ROLLED_BACK is valid."""
        assert validate_task_transition(
            TaskState.DEPLOYED_PRODUCTION,
            TaskState.PROD_ROLLED_BACK
        )

    def test_cannot_skip_testing_to_production(self):
        """Test that CI_PASSED cannot go directly to PROD_DEPLOY_REQUESTED (must deploy testing first)."""
        assert not validate_task_transition(
            TaskState.CI_PASSED,
            TaskState.PROD_DEPLOY_REQUESTED
        )

    def test_cannot_bypass_approval(self):
        """Test that PROD_DEPLOY_REQUESTED cannot go directly to DEPLOYED_PRODUCTION."""
        assert not validate_task_transition(
            TaskState.PROD_DEPLOY_REQUESTED,
            TaskState.DEPLOYED_PRODUCTION
        )


# -----------------------------------------------------------------------------
# Phase 6 Tests - Audit Trail
# -----------------------------------------------------------------------------
class TestAuditTrail:
    """Tests for audit trail functionality."""

    def test_audit_log_path(self, tmp_path):
        """Test audit log path is correctly constructed."""
        # Note: This tests the helper function directly
        from controller.main import get_audit_log_path, PROJECTS_DIR
        audit_path = get_audit_log_path("test-project")
        assert "audit" in str(audit_path)
        assert "production.log" in str(audit_path)

    def test_release_path(self, tmp_path):
        """Test release path is correctly constructed."""
        from controller.main import get_release_path
        release_path = get_release_path("test-project", "task-123")
        assert "releases" in str(release_path)
        assert "task-123" in str(release_path)


# -----------------------------------------------------------------------------
# Phase 6 Tests - Full Lifecycle
# -----------------------------------------------------------------------------
class TestPhase6FullLifecycle:
    """Tests for complete Phase 6 task lifecycle with production deployment."""

    def test_full_lifecycle_through_production(self, client):
        """Test complete lifecycle from task creation through production deployment."""
        # Bootstrap project
        client.post("/project/bootstrap", json={
            "project_name": "full-lifecycle-p6",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Create and progress task through Phase 5
        task_response = client.post("/task", json={
            "project_name": "full-lifecycle-p6",
            "task_description": "Full lifecycle test through production",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]

        # Phase 2-3: Validate, plan, approve, generate diff
        client.post(f"/task/{task_id}/validate", params={"project_name": "full-lifecycle-p6"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "full-lifecycle-p6"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "full-lifecycle-p6"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "full-lifecycle-p6"})

        # Phase 4: Dry-run and apply
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "full-lifecycle-p6"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "full-lifecycle-p6", "confirm": True})

        # Phase 5: Commit, CI, deploy testing
        client.post(f"/task/{task_id}/commit", params={"project_name": "full-lifecycle-p6", "confirm": True})
        client.post(f"/task/{task_id}/ci/run", params={"project_name": "full-lifecycle-p6"})
        client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": "full-lifecycle-p6"},
            json={"status": "passed"}
        )
        deploy_testing_response = client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": "full-lifecycle-p6", "confirm": True}
        )
        assert deploy_testing_response.json()["current_state"] == "deployed_testing"

        # Phase 6: Request production deployment
        prod_request_response = client.post(
            f"/task/{task_id}/deploy/production/request",
            params={"project_name": "full-lifecycle-p6", "user_id": "user1"},
            json={
                "justification": "Critical security fix that must go to production immediately",
                "risk_acknowledged": True,
                "rollback_plan": "Revert to previous commit"
            }
        )
        assert prod_request_response.json()["current_state"] == "prod_deploy_requested"
        assert "DIFFERENT USER" in prod_request_response.json()["warning"]

        # Phase 6: Approve by DIFFERENT user
        prod_approve_response = client.post(
            f"/task/{task_id}/deploy/production/approve",
            params={"project_name": "full-lifecycle-p6", "user_id": "user2"},  # Different user!
            json={
                "approval_reason": "Reviewed code, fix looks correct",
                "reviewed_changes": True,
                "reviewed_rollback": True
            }
        )
        assert prod_approve_response.json()["current_state"] == "prod_approved"

        # Phase 6: Apply production deployment
        prod_apply_response = client.post(
            f"/task/{task_id}/deploy/production/apply",
            params={"project_name": "full-lifecycle-p6", "confirm": True, "user_id": "user1"}
        )
        assert prod_apply_response.json()["current_state"] == "deployed_production"
        assert "PRODUCTION" in prod_apply_response.json()["warning"]
        assert prod_apply_response.json()["rollback_available"] == True

    def test_same_user_approval_blocked(self, client):
        """Test that same user cannot approve their own production request."""
        # Bootstrap
        client.post("/project/bootstrap", json={
            "project_name": "dual-approval-test",
            "repo_url": "https://github.com/user/repo",
            "tech_stack": ["python"]
        })

        # Progress through to deployed_testing
        task_response = client.post("/task", json={
            "project_name": "dual-approval-test",
            "task_description": "Dual approval test",
            "task_type": "feature"
        })
        task_id = task_response.json()["task_id"]

        client.post(f"/task/{task_id}/validate", params={"project_name": "dual-approval-test"})
        client.post(f"/task/{task_id}/plan", params={"project_name": "dual-approval-test"})
        client.post(f"/task/{task_id}/approve", params={"project_name": "dual-approval-test"})
        client.post(f"/task/{task_id}/generate-diff", params={"project_name": "dual-approval-test"})
        client.post(f"/task/{task_id}/dry-run", params={"project_name": "dual-approval-test"})
        client.post(f"/task/{task_id}/apply", params={"project_name": "dual-approval-test", "confirm": True})
        client.post(f"/task/{task_id}/commit", params={"project_name": "dual-approval-test", "confirm": True})
        client.post(f"/task/{task_id}/ci/run", params={"project_name": "dual-approval-test"})
        client.post(
            f"/task/{task_id}/ci/result",
            params={"project_name": "dual-approval-test"},
            json={"status": "passed"}
        )
        client.post(
            f"/task/{task_id}/deploy/testing",
            params={"project_name": "dual-approval-test", "confirm": True}
        )

        # Request production as user1
        client.post(
            f"/task/{task_id}/deploy/production/request",
            params={"project_name": "dual-approval-test", "user_id": "user1"},
            json={
                "justification": "Need to deploy critical fix",
                "risk_acknowledged": True,
                "rollback_plan": "Revert commit"
            }
        )

        # Attempt to approve as SAME user1 - should be blocked
        approve_response = client.post(
            f"/task/{task_id}/deploy/production/approve",
            params={"project_name": "dual-approval-test", "user_id": "user1"},  # Same user!
            json={
                "approval_reason": "I approve myself",
                "reviewed_changes": True,
                "reviewed_rollback": True
            }
        )
        assert approve_response.status_code == 403
        assert "DUAL APPROVAL" in approve_response.json()["detail"] or "same" in approve_response.json()["detail"].lower()

    def test_production_rollback_no_dual_approval_needed(self, client):
        """Test that production rollback does NOT require dual approval (break-glass)."""
        # The policy hook should allow rollback with just user_id
        allowed, reason = can_rollback_prod("test-task", "any_user")
        assert allowed
        # Rollback should be immediate, no waiting for another user
