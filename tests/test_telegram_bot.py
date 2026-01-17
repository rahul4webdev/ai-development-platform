"""
Unit Tests for Telegram Bot - Phase 4

Test coverage for:
- Command parsing (including Phase 2, Phase 3, and Phase 4 commands)
- Session management with last_task_id
- Task lifecycle command handlers
- Diff generation command handler (Phase 3)
- Dry-run, apply, rollback command handlers (Phase 4)
- Task type inference
"""

import pytest

from bots.telegram_bot import (
    parse_command,
    infer_task_type,
    get_or_create_session,
    set_current_project,
    set_last_task_id,
    process_message,
    handle_start,
    handle_help,
    handle_bootstrap,
    handle_project,
    handle_task,
    handle_validate,
    handle_plan,
    handle_approve,
    handle_reject,
    handle_generate_diff,  # Phase 3
    # Phase 4 handlers
    handle_dry_run,
    handle_apply,
    handle_rollback,
    handle_status,
    handle_list,
    BotCommand,
    TaskType,
    user_sessions,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear user sessions before each test."""
    user_sessions.clear()
    yield
    user_sessions.clear()


# -----------------------------------------------------------------------------
# Command Parsing Tests
# -----------------------------------------------------------------------------
class TestCommandParsing:
    """Tests for command parsing functionality."""

    def test_parse_start_command(self):
        """Test parsing /start command."""
        result = parse_command("/start")
        assert result is not None
        assert result.command == BotCommand.START
        assert result.args == []

    def test_parse_bootstrap_command(self):
        """Test parsing /bootstrap command with arguments."""
        result = parse_command("/bootstrap my-project https://github.com/user/repo")
        assert result is not None
        assert result.command == BotCommand.BOOTSTRAP
        assert result.args == ["my-project", "https://github.com/user/repo"]

    def test_parse_validate_command(self):
        """Test parsing /validate command."""
        result = parse_command("/validate abc123")
        assert result is not None
        assert result.command == BotCommand.VALIDATE
        assert result.args == ["abc123"]

    def test_parse_plan_command(self):
        """Test parsing /plan command."""
        result = parse_command("/plan task-id-123")
        assert result is not None
        assert result.command == BotCommand.PLAN
        assert result.args == ["task-id-123"]

    def test_parse_approve_command(self):
        """Test parsing /approve command."""
        result = parse_command("/approve task-id")
        assert result is not None
        assert result.command == BotCommand.APPROVE
        assert result.args == ["task-id"]

    def test_parse_reject_command(self):
        """Test parsing /reject command with reason."""
        result = parse_command("/reject task-id Requirements unclear")
        assert result is not None
        assert result.command == BotCommand.REJECT
        assert result.args == ["task-id", "Requirements", "unclear"]

    def test_parse_generate_diff_command(self):
        """Test parsing /generate_diff command (Phase 3)."""
        result = parse_command("/generate_diff task-id-123")
        assert result is not None
        assert result.command == BotCommand.GENERATE_DIFF
        assert result.args == ["task-id-123"]

    def test_parse_unknown_command(self):
        """Test parsing unknown command returns None."""
        result = parse_command("/unknown")
        assert result is None

    def test_parse_non_command(self):
        """Test parsing non-command text returns None."""
        result = parse_command("hello world")
        assert result is None

    def test_parse_command_case_insensitive(self):
        """Test command parsing is case insensitive."""
        result = parse_command("/HELP")
        assert result is not None
        assert result.command == BotCommand.HELP


# -----------------------------------------------------------------------------
# Task Type Inference Tests
# -----------------------------------------------------------------------------
class TestTaskTypeInference:
    """Tests for task type inference from description."""

    def test_infer_bug(self):
        """Test inferring bug task type."""
        assert infer_task_type("Fix the login bug") == TaskType.BUG
        assert infer_task_type("There's an error in the form") == TaskType.BUG
        assert infer_task_type("The button is broken") == TaskType.BUG

    def test_infer_feature(self):
        """Test inferring feature task type."""
        assert infer_task_type("Add dark mode") == TaskType.FEATURE
        assert infer_task_type("Implement user authentication") == TaskType.FEATURE
        assert infer_task_type("Create new dashboard") == TaskType.FEATURE

    def test_infer_refactor(self):
        """Test inferring refactor task type."""
        assert infer_task_type("Refactor the database module") == TaskType.REFACTOR
        assert infer_task_type("Clean up the code") == TaskType.REFACTOR
        assert infer_task_type("Optimize performance") == TaskType.REFACTOR

    def test_infer_infra(self):
        """Test inferring infra task type."""
        assert infer_task_type("Set up CI/CD pipeline") == TaskType.INFRA
        assert infer_task_type("Configure deployment infra") == TaskType.INFRA

    def test_infer_feature_default(self):
        """Test default to feature for unclear descriptions."""
        assert infer_task_type("Update the system") == TaskType.FEATURE
        assert infer_task_type("Check the logs") == TaskType.FEATURE


# -----------------------------------------------------------------------------
# Session Management Tests
# -----------------------------------------------------------------------------
class TestSessionManagement:
    """Tests for user session management."""

    def test_create_new_session(self):
        """Test creating a new user session."""
        session = get_or_create_session("user1", "TestUser")
        assert session.user_id == "user1"
        assert session.username == "TestUser"
        assert session.current_project is None
        assert session.last_task_id is None

    def test_get_existing_session(self):
        """Test retrieving existing session."""
        get_or_create_session("user1", "TestUser")
        session = get_or_create_session("user1")
        assert session.user_id == "user1"
        assert session.username == "TestUser"

    def test_set_current_project(self):
        """Test setting current project for session."""
        get_or_create_session("user1")
        result = set_current_project("user1", "my-project")
        assert result is True
        session = get_or_create_session("user1")
        assert session.current_project == "my-project"

    def test_set_last_task_id(self):
        """Test setting last task ID for session."""
        get_or_create_session("user1")
        set_last_task_id("user1", "task-abc123")
        session = get_or_create_session("user1")
        assert session.last_task_id == "task-abc123"


# -----------------------------------------------------------------------------
# Command Handler Tests - Phase 2
# -----------------------------------------------------------------------------
class TestCommandHandlers:
    """Tests for command handler functions."""

    def test_handle_start(self):
        """Test /start command handler shows Phase 3 info."""
        response = handle_start("user1", "TestUser")
        assert "Welcome" in response
        assert "Phase 3" in response
        assert "/bootstrap" in response
        assert "/validate" in response
        assert "/plan" in response
        assert "/approve" in response
        assert "/generate_diff" in response  # Phase 3

    def test_handle_help(self):
        """Test /help command handler."""
        response = handle_help("user1")
        assert "Commands:" in response or "COMMANDS:" in response
        assert "/bootstrap" in response
        assert "/validate" in response
        assert "/plan" in response
        assert "/approve" in response
        assert "/reject" in response
        assert "/generate_diff" in response  # Phase 3

    def test_handle_bootstrap_no_args(self):
        """Test /bootstrap with no arguments shows usage."""
        response = handle_bootstrap("user1", [])
        assert "Usage:" in response
        assert "project_name" in response

    def test_handle_bootstrap_success(self):
        """Test /bootstrap with valid arguments."""
        response = handle_bootstrap("user1", ["my-project", "https://github.com/user/repo", "python,fastapi"])
        assert "bootstrapped" in response.lower()
        assert "my-project" in response
        # Verify project was selected
        session = get_or_create_session("user1")
        assert session.current_project == "my-project"

    def test_handle_project_no_args(self):
        """Test /project with no arguments."""
        response = handle_project("user1", [])
        assert "No project selected" in response

    def test_handle_project_set_project(self):
        """Test /project with project name."""
        response = handle_project("user1", ["my-webapp"])
        assert "Switched to project: my-webapp" in response

    def test_handle_task_no_project(self):
        """Test /task when no project selected."""
        response = handle_task("user1", ["Fix", "bug"])
        assert "No project selected" in response

    def test_handle_task_with_project(self):
        """Test /task with project selected."""
        set_current_project("user1", "my-webapp")
        response = handle_task("user1", ["Fix", "the", "login", "button"])
        assert "Task created" in response
        assert "my-webapp" in response
        # Verify last_task_id was set
        session = get_or_create_session("user1")
        assert session.last_task_id is not None

    def test_handle_validate_no_project(self):
        """Test /validate when no project selected."""
        response = handle_validate("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_validate_with_project(self):
        """Test /validate with project selected."""
        set_current_project("user1", "my-webapp")
        response = handle_validate("user1", ["task-id-123"])
        assert "task-id-123" in response.lower() or "validated" in response.lower()

    def test_handle_validate_last(self):
        """Test /validate last uses last_task_id."""
        set_current_project("user1", "my-webapp")
        set_last_task_id("user1", "last-task-123")
        response = handle_validate("user1", ["last"])
        assert "last-task-123" in response or "validated" in response.lower()

    def test_handle_plan_no_project(self):
        """Test /plan when no project selected."""
        response = handle_plan("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_plan_with_project(self):
        """Test /plan with project selected."""
        set_current_project("user1", "my-webapp")
        response = handle_plan("user1", ["task-id-123"])
        assert "plan" in response.lower()

    def test_handle_approve_no_project(self):
        """Test /approve when no project selected."""
        response = handle_approve("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_approve_with_project(self):
        """Test /approve with project selected."""
        set_current_project("user1", "my-webapp")
        response = handle_approve("user1", ["task-id-123"])
        assert "approved" in response.lower()
        assert "/generate_diff" in response  # Phase 3: Should mention next step

    def test_handle_reject_no_args(self):
        """Test /reject with insufficient arguments."""
        set_current_project("user1", "my-webapp")
        response = handle_reject("user1", ["task-id"])  # Missing reason
        assert "Usage:" in response or "reason" in response.lower()

    def test_handle_reject_short_reason(self):
        """Test /reject with too short reason."""
        set_current_project("user1", "my-webapp")
        response = handle_reject("user1", ["task-id", "short"])
        assert "10 characters" in response

    def test_handle_reject_success(self):
        """Test /reject with valid arguments."""
        set_current_project("user1", "my-webapp")
        response = handle_reject("user1", ["task-id", "Requirements", "are", "unclear", "and", "need", "clarification"])
        assert "rejected" in response.lower()

    def test_handle_status_no_project(self):
        """Test /status when no project selected."""
        response = handle_status("user1")
        assert "No project selected" in response

    def test_handle_status_with_project(self):
        """Test /status with project selected."""
        set_current_project("user1", "my-webapp")
        response = handle_status("user1")
        assert "my-webapp" in response
        assert "Phase:" in response or "Tasks" in response


# -----------------------------------------------------------------------------
# Message Processing Tests
# -----------------------------------------------------------------------------
class TestMessageProcessing:
    """Tests for main message processing function."""

    def test_process_start_message(self):
        """Test processing /start message."""
        response = process_message("user1", "TestUser", "/start")
        assert "Welcome" in response

    def test_process_bootstrap_message(self):
        """Test processing /bootstrap message."""
        response = process_message("user1", "TestUser", "/bootstrap my-app https://github.com/user/repo")
        assert "bootstrapped" in response.lower() or "my-app" in response

    def test_process_task_lifecycle(self):
        """Test processing full task lifecycle."""
        # Bootstrap
        process_message("user1", "TestUser", "/bootstrap lifecycle-test https://github.com/user/repo")

        # Create task
        task_response = process_message("user1", "TestUser", "/task Fix the login button not working")
        assert "Task created" in task_response

        # Validate
        validate_response = process_message("user1", "TestUser", "/validate last")
        assert "validated" in validate_response.lower() or "task" in validate_response.lower()

        # Plan
        plan_response = process_message("user1", "TestUser", "/plan last")
        assert "plan" in plan_response.lower()

        # Approve
        approve_response = process_message("user1", "TestUser", "/approve last")
        assert "approved" in approve_response.lower()

    def test_process_unknown_command(self):
        """Test processing unknown command."""
        response = process_message("user1", "TestUser", "/unknown")
        assert "Unknown command" in response

    def test_process_non_command(self):
        """Test processing non-command text."""
        response = process_message("user1", "TestUser", "hello")
        assert "Unknown command" in response


# -----------------------------------------------------------------------------
# Production Deployment Block Tests
# -----------------------------------------------------------------------------
class TestProductionDeploymentBlock:
    """Tests for production deployment blocking."""

    def test_deploy_production_blocked(self):
        """Test production deployment is blocked via chat."""
        set_current_project("user1", "my-webapp")
        from bots.telegram_bot import handle_deploy
        response = handle_deploy("user1", ["production"])
        assert "NOT available" in response or "requires" in response.lower()
        assert "AI_POLICY" in response or "approval" in response.lower()


# -----------------------------------------------------------------------------
# Phase 3 Diff Generation Tests
# -----------------------------------------------------------------------------
class TestDiffGenerationCommand:
    """Tests for /generate_diff command handler (Phase 3)."""

    def test_handle_generate_diff_no_project(self):
        """Test /generate_diff when no project selected."""
        response = handle_generate_diff("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_generate_diff_no_args(self):
        """Test /generate_diff with no arguments shows usage."""
        set_current_project("user1", "my-webapp")
        response = handle_generate_diff("user1", [])
        assert "Usage:" in response or "generate_diff" in response.lower()

    def test_handle_generate_diff_with_task_id(self):
        """Test /generate_diff with task ID."""
        set_current_project("user1", "my-webapp")
        response = handle_generate_diff("user1", ["task-id-123"])
        assert "Diff generated" in response or "diff" in response.lower()
        assert "NOT APPLIED" in response.upper() or "not applied" in response.lower()

    def test_handle_generate_diff_last(self):
        """Test /generate_diff last uses last_task_id."""
        set_current_project("user1", "my-webapp")
        set_last_task_id("user1", "last-task-abc")
        response = handle_generate_diff("user1", ["last"])
        assert "last-task-abc" in response or "diff" in response.lower()

    def test_handle_generate_diff_warning(self):
        """Test /generate_diff includes safety warning."""
        set_current_project("user1", "my-webapp")
        response = handle_generate_diff("user1", ["task-id"])
        assert "REVIEW" in response.upper() or "review" in response.lower()
        assert "NOT" in response or "not" in response

    def test_process_generate_diff_message(self):
        """Test processing /generate_diff message."""
        process_message("user1", "TestUser", "/bootstrap diff-app https://github.com/user/repo")
        response = process_message("user1", "TestUser", "/generate_diff some-task-id")
        assert "diff" in response.lower()


# -----------------------------------------------------------------------------
# Phase 3 Task Lifecycle Tests
# -----------------------------------------------------------------------------
class TestPhase3TaskLifecycle:
    """Tests for full Phase 3 task lifecycle."""

    def test_full_lifecycle_with_diff(self):
        """Test complete lifecycle including diff generation."""
        # Bootstrap
        process_message("user1", "TestUser", "/bootstrap phase3-test https://github.com/user/repo python")

        # Create task
        task_response = process_message("user1", "TestUser", "/task Add dark mode feature")
        assert "Task created" in task_response

        # Validate
        validate_response = process_message("user1", "TestUser", "/validate last")
        assert "validated" in validate_response.lower() or "task" in validate_response.lower()

        # Plan
        plan_response = process_message("user1", "TestUser", "/plan last")
        assert "plan" in plan_response.lower()

        # Approve
        approve_response = process_message("user1", "TestUser", "/approve last")
        assert "approved" in approve_response.lower()
        assert "/generate_diff" in approve_response  # Phase 3 next step

        # Generate diff
        diff_response = process_message("user1", "TestUser", "/generate_diff last")
        assert "diff" in diff_response.lower()
        assert "NOT APPLIED" in diff_response.upper() or "review" in diff_response.lower()


# -----------------------------------------------------------------------------
# Phase 4 Dry-Run Command Tests
# -----------------------------------------------------------------------------
class TestDryRunCommand:
    """Tests for /dry_run command handler (Phase 4)."""

    def test_handle_dry_run_no_project(self):
        """Test /dry_run when no project selected."""
        response = handle_dry_run("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_dry_run_no_args(self):
        """Test /dry_run with no arguments shows usage."""
        set_current_project("user1", "my-webapp")
        response = handle_dry_run("user1", [])
        assert "Usage:" in response or "dry_run" in response.lower()

    def test_handle_dry_run_with_task_id(self):
        """Test /dry_run with task ID."""
        set_current_project("user1", "my-webapp")
        response = handle_dry_run("user1", ["task-id-123"])
        assert "dry" in response.lower() or "simulation" in response.lower()
        assert "NOT" in response or "not" in response

    def test_handle_dry_run_last(self):
        """Test /dry_run last uses last_task_id."""
        set_current_project("user1", "my-webapp")
        set_last_task_id("user1", "last-task-xyz")
        response = handle_dry_run("user1", ["last"])
        assert "last-task-xyz" in response or "dry" in response.lower()

    def test_process_dry_run_message(self):
        """Test processing /dry_run message."""
        process_message("user1", "TestUser", "/bootstrap dryrun-app https://github.com/user/repo")
        response = process_message("user1", "TestUser", "/dry_run some-task-id")
        assert "dry" in response.lower() or "simulation" in response.lower()


# -----------------------------------------------------------------------------
# Phase 4 Apply Command Tests
# -----------------------------------------------------------------------------
class TestApplyCommand:
    """Tests for /apply command handler (Phase 4)."""

    def test_handle_apply_no_project(self):
        """Test /apply when no project selected."""
        response = handle_apply("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_apply_no_args(self):
        """Test /apply with no arguments shows usage."""
        set_current_project("user1", "my-webapp")
        response = handle_apply("user1", [])
        assert "Usage:" in response or "apply" in response.lower()

    def test_handle_apply_without_confirmation(self):
        """Test /apply without confirm keyword requires confirmation."""
        set_current_project("user1", "my-webapp")
        response = handle_apply("user1", ["task-id-123"])
        assert "CONFIRM" in response.upper() or "confirm" in response.lower()
        assert "REQUIRED" in response.upper() or "required" in response.lower()

    def test_handle_apply_with_confirmation(self):
        """Test /apply with confirm keyword."""
        set_current_project("user1", "my-webapp")
        response = handle_apply("user1", ["task-id-123", "confirm"])
        # May succeed or fail based on state, but should attempt apply
        assert "apply" in response.lower() or "backup" in response.lower() or "error" in response.lower()

    def test_handle_apply_last_without_confirm(self):
        """Test /apply last without confirm shows confirmation prompt."""
        set_current_project("user1", "my-webapp")
        set_last_task_id("user1", "last-task-apply")
        response = handle_apply("user1", ["last"])
        assert "CONFIRM" in response.upper() or "confirm" in response.lower()

    def test_handle_apply_confirmation_warning(self):
        """Test /apply confirmation message includes safety warning."""
        set_current_project("user1", "my-webapp")
        response = handle_apply("user1", ["task-id"])
        assert "BACKUP" in response.upper() or "backup" in response.lower() or "reversible" in response.lower()

    def test_process_apply_message_no_confirm(self):
        """Test processing /apply message without confirmation."""
        process_message("user1", "TestUser", "/bootstrap apply-app https://github.com/user/repo")
        response = process_message("user1", "TestUser", "/apply some-task-id")
        assert "confirm" in response.lower()


# -----------------------------------------------------------------------------
# Phase 4 Rollback Command Tests
# -----------------------------------------------------------------------------
class TestRollbackCommand:
    """Tests for /rollback command handler (Phase 4)."""

    def test_handle_rollback_no_project(self):
        """Test /rollback when no project selected."""
        response = handle_rollback("user1", ["task-id"])
        assert "No project selected" in response

    def test_handle_rollback_no_args(self):
        """Test /rollback with no arguments shows usage."""
        set_current_project("user1", "my-webapp")
        response = handle_rollback("user1", [])
        assert "Usage:" in response or "rollback" in response.lower()

    def test_handle_rollback_with_task_id(self):
        """Test /rollback with task ID."""
        set_current_project("user1", "my-webapp")
        response = handle_rollback("user1", ["task-id-123"])
        # May succeed or fail based on state, but should attempt rollback
        assert "rollback" in response.lower() or "restore" in response.lower() or "error" in response.lower()

    def test_handle_rollback_last(self):
        """Test /rollback last uses last_task_id."""
        set_current_project("user1", "my-webapp")
        set_last_task_id("user1", "last-task-rollback")
        response = handle_rollback("user1", ["last"])
        assert "last-task-rollback" in response or "rollback" in response.lower()

    def test_process_rollback_message(self):
        """Test processing /rollback message."""
        process_message("user1", "TestUser", "/bootstrap rollback-app https://github.com/user/repo")
        response = process_message("user1", "TestUser", "/rollback some-task-id")
        assert "rollback" in response.lower() or "restore" in response.lower() or "error" in response.lower()


# -----------------------------------------------------------------------------
# Phase 4 Full Lifecycle Tests
# -----------------------------------------------------------------------------
class TestPhase4TaskLifecycle:
    """Tests for full Phase 4 task lifecycle with execution."""

    def test_full_lifecycle_with_execution(self):
        """Test complete lifecycle including dry-run and apply."""
        # Bootstrap
        process_message("user1", "TestUser", "/bootstrap phase4-test https://github.com/user/repo python")

        # Create task
        task_response = process_message("user1", "TestUser", "/task Add new configuration option")
        assert "Task created" in task_response

        # Validate
        validate_response = process_message("user1", "TestUser", "/validate last")
        assert "validated" in validate_response.lower() or "task" in validate_response.lower()

        # Plan
        plan_response = process_message("user1", "TestUser", "/plan last")
        assert "plan" in plan_response.lower()

        # Approve
        approve_response = process_message("user1", "TestUser", "/approve last")
        assert "approved" in approve_response.lower()

        # Generate diff
        diff_response = process_message("user1", "TestUser", "/generate_diff last")
        assert "diff" in diff_response.lower()

        # Dry-run
        dry_run_response = process_message("user1", "TestUser", "/dry_run last")
        assert "dry" in dry_run_response.lower() or "simulation" in dry_run_response.lower()

        # Apply (will require confirmation)
        apply_response = process_message("user1", "TestUser", "/apply last")
        assert "confirm" in apply_response.lower()

    def test_apply_confirmation_required(self):
        """Test that apply always requires explicit confirmation."""
        set_current_project("user1", "my-webapp")

        # Try without confirm
        response1 = handle_apply("user1", ["task-id"])
        assert "CONFIRM" in response1.upper() or "confirm" in response1.lower()

        # Try with wrong confirmation word
        response2 = handle_apply("user1", ["task-id", "yes"])
        assert "CONFIRM" in response2.upper() or "confirm" in response2.lower()

        # Only "confirm" keyword should work
        response3 = handle_apply("user1", ["task-id", "CONFIRM"])
        # Case insensitive - should attempt apply
        assert "apply" in response3.lower() or "backup" in response3.lower() or "error" in response3.lower()


# -----------------------------------------------------------------------------
# Phase 4 Command Help Tests
# -----------------------------------------------------------------------------
class TestPhase4CommandHelp:
    """Tests for Phase 4 commands in help and start messages."""

    def test_start_includes_phase4_commands(self):
        """Test /start includes Phase 4 commands."""
        response = handle_start("user1", "TestUser")
        # Phase 4 commands should be mentioned
        assert "/dry_run" in response or "dry_run" in response
        assert "/apply" in response or "apply" in response
        assert "/rollback" in response or "rollback" in response

    def test_help_includes_phase4_commands(self):
        """Test /help includes Phase 4 commands."""
        response = handle_help("user1")
        assert "/dry_run" in response or "dry_run" in response
        assert "/apply" in response or "apply" in response
        assert "/rollback" in response or "rollback" in response


# -----------------------------------------------------------------------------
# Placeholder Tests
# -----------------------------------------------------------------------------
class TestPlaceholders:
    """Placeholder tests for future functionality."""

    def test_placeholder_passes(self):
        """Placeholder test to ensure test suite runs."""
        assert True

    # TODO: Add tests for:
    # - Actual HTTP calls to controller
    # - Authorization checks
    # - Rate limiting
    # - Actual Telegram API integration
    # - Phase 4: Integration tests with actual file operations
