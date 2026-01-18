"""
Unit Tests for Lifecycle Engine - Phase 15.1

Test coverage for:
- Valid state transitions
- Invalid state transitions (rejection)
- Per-aspect isolation
- PROJECT_MODE vs CHANGE_MODE
- Event-driven transitions
- Role-based permissions
- Persistence and recovery
"""

import pytest
import asyncio
import tempfile
import os
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Set up test environment before imports
os.environ['LIFECYCLE_STATE_DIR'] = tempfile.mkdtemp()

from controller.lifecycle_v2 import (
    LifecycleState,
    LifecycleMode,
    ChangeType,
    ProjectAspect,
    TransitionTrigger,
    UserRole,
    LifecycleInstance,
    ChangeReference,
    LifecycleStateMachine,
    LifecycleManager,
    VALID_TRANSITIONS,
    TRANSITION_PERMISSIONS,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_state_dir():
    """Create a temporary state directory for testing."""
    path = Path(tempfile.mkdtemp())
    yield path
    # Cleanup
    import shutil
    if path.exists():
        shutil.rmtree(path)


@pytest.fixture
def state_machine(temp_state_dir):
    """Create a LifecycleStateMachine with temp directory."""
    return LifecycleStateMachine(state_dir=temp_state_dir)


@pytest.fixture
def lifecycle_manager(temp_state_dir):
    """Create a LifecycleManager with temp directory."""
    return LifecycleManager(state_dir=temp_state_dir)


def create_test_lifecycle(
    lifecycle_id: str = "test-lifecycle-1",
    project_name: str = "test-project",
    mode: LifecycleMode = LifecycleMode.PROJECT_MODE,
    aspect: ProjectAspect = ProjectAspect.CORE,
    state: LifecycleState = LifecycleState.CREATED,
) -> LifecycleInstance:
    """Helper to create test lifecycle instances."""
    return LifecycleInstance(
        lifecycle_id=lifecycle_id,
        project_name=project_name,
        mode=mode,
        aspect=aspect,
        state=state,
        created_at=datetime.utcnow(),
        created_by="test-user",
    )


# -----------------------------------------------------------------------------
# Test 1: Valid State Transitions
# -----------------------------------------------------------------------------
class TestValidTransitions:
    """Test that valid transitions are allowed."""

    @pytest.mark.asyncio
    async def test_created_to_planning(self, state_machine):
        """Test transition from CREATED to PLANNING."""
        lifecycle = create_test_lifecycle(state=LifecycleState.CREATED)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.SYSTEM_INIT,
            triggered_by="test-user",
            user_role=UserRole.DEVELOPER,
            reason="Starting lifecycle",
        )

        assert success is True
        assert new_state == LifecycleState.PLANNING
        assert lifecycle.state == LifecycleState.PLANNING

    @pytest.mark.asyncio
    async def test_planning_to_development(self, state_machine):
        """Test transition from PLANNING to DEVELOPMENT."""
        lifecycle = create_test_lifecycle(state=LifecycleState.PLANNING)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CLAUDE_JOB_COMPLETED,
            triggered_by="claude-cli",
            user_role=UserRole.DEVELOPER,
            reason="Plan complete",
        )

        assert success is True
        assert new_state == LifecycleState.DEVELOPMENT

    @pytest.mark.asyncio
    async def test_development_to_testing(self, state_machine):
        """Test transition from DEVELOPMENT to TESTING."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEVELOPMENT)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CLAUDE_JOB_COMPLETED,
            triggered_by="claude-cli",
            user_role=UserRole.DEVELOPER,
            reason="Development complete",
        )

        assert success is True
        assert new_state == LifecycleState.TESTING

    @pytest.mark.asyncio
    async def test_testing_to_awaiting_feedback(self, state_machine):
        """Test transition from TESTING to AWAITING_FEEDBACK on test pass."""
        lifecycle = create_test_lifecycle(state=LifecycleState.TESTING)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TEST_PASSED,
            triggered_by="ci-system",
            user_role=UserRole.TESTER,
            reason="All tests passed",
        )

        assert success is True
        assert new_state == LifecycleState.AWAITING_FEEDBACK

    @pytest.mark.asyncio
    async def test_awaiting_feedback_to_ready_for_production(self, state_machine):
        """Test transition from AWAITING_FEEDBACK to READY_FOR_PRODUCTION."""
        lifecycle = create_test_lifecycle(state=LifecycleState.AWAITING_FEEDBACK)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="admin-user",
            user_role=UserRole.ADMIN,
            reason="Approved for production",
        )

        assert success is True
        assert new_state == LifecycleState.READY_FOR_PRODUCTION

    @pytest.mark.asyncio
    async def test_full_lifecycle_flow(self, state_machine):
        """Test complete lifecycle from CREATED to DEPLOYED."""
        lifecycle = create_test_lifecycle(state=LifecycleState.CREATED)

        # CREATED -> PLANNING
        success, _, _ = await state_machine.transition(
            lifecycle, TransitionTrigger.SYSTEM_INIT, "user", UserRole.DEVELOPER
        )
        assert success and lifecycle.state == LifecycleState.PLANNING

        # PLANNING -> DEVELOPMENT
        success, _, _ = await state_machine.transition(
            lifecycle, TransitionTrigger.CLAUDE_JOB_COMPLETED, "claude", UserRole.DEVELOPER
        )
        assert success and lifecycle.state == LifecycleState.DEVELOPMENT

        # DEVELOPMENT -> TESTING
        success, _, _ = await state_machine.transition(
            lifecycle, TransitionTrigger.CLAUDE_JOB_COMPLETED, "claude", UserRole.DEVELOPER
        )
        assert success and lifecycle.state == LifecycleState.TESTING

        # TESTING -> AWAITING_FEEDBACK
        success, _, _ = await state_machine.transition(
            lifecycle, TransitionTrigger.TEST_PASSED, "ci", UserRole.TESTER
        )
        assert success and lifecycle.state == LifecycleState.AWAITING_FEEDBACK

        # AWAITING_FEEDBACK -> READY_FOR_PRODUCTION
        success, _, _ = await state_machine.transition(
            lifecycle, TransitionTrigger.HUMAN_APPROVAL, "admin", UserRole.ADMIN
        )
        assert success and lifecycle.state == LifecycleState.READY_FOR_PRODUCTION

        # READY_FOR_PRODUCTION -> DEPLOYED
        success, _, _ = await state_machine.transition(
            lifecycle, TransitionTrigger.HUMAN_APPROVAL, "admin", UserRole.ADMIN
        )
        assert success and lifecycle.state == LifecycleState.DEPLOYED

        # Verify terminal state
        assert lifecycle.completed_at is not None


# -----------------------------------------------------------------------------
# Test 2: Invalid State Transitions
# -----------------------------------------------------------------------------
class TestInvalidTransitions:
    """Test that invalid transitions are rejected."""

    @pytest.mark.asyncio
    async def test_skip_planning_fails(self, state_machine):
        """Test that skipping from CREATED to DEVELOPMENT fails."""
        lifecycle = create_test_lifecycle(state=LifecycleState.CREATED)

        # Try to go directly to DEVELOPMENT (invalid)
        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CLAUDE_JOB_COMPLETED,
            triggered_by="user",
            user_role=UserRole.DEVELOPER,
        )

        assert success is False
        assert new_state is None
        assert lifecycle.state == LifecycleState.CREATED  # State unchanged
        assert "Invalid trigger" in message

    @pytest.mark.asyncio
    async def test_skip_testing_fails(self, state_machine):
        """Test that skipping from DEVELOPMENT to AWAITING_FEEDBACK fails."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEVELOPMENT)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="admin",
            user_role=UserRole.ADMIN,
        )

        assert success is False
        assert new_state is None
        assert lifecycle.state == LifecycleState.DEVELOPMENT

    @pytest.mark.asyncio
    async def test_terminal_state_no_transitions(self, state_machine):
        """Test that terminal states have no valid transitions."""
        lifecycle = create_test_lifecycle(state=LifecycleState.ARCHIVED)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.SYSTEM_INIT,
            triggered_by="user",
            user_role=UserRole.OWNER,
        )

        assert success is False
        assert new_state is None

    @pytest.mark.asyncio
    async def test_invalid_trigger_for_state(self, state_machine):
        """Test using wrong trigger for current state."""
        lifecycle = create_test_lifecycle(state=LifecycleState.TESTING)

        # Use SYSTEM_INIT which is not valid for TESTING state
        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.SYSTEM_INIT,
            triggered_by="user",
            user_role=UserRole.DEVELOPER,
        )

        assert success is False
        assert "Invalid trigger" in message


# -----------------------------------------------------------------------------
# Test 3: Per-Aspect Isolation
# -----------------------------------------------------------------------------
class TestAspectIsolation:
    """Test that aspects are properly isolated."""

    @pytest.mark.asyncio
    async def test_create_multiple_aspects(self, lifecycle_manager):
        """Test creating lifecycles for different aspects."""
        # Create CORE aspect lifecycle
        success1, _, lc1 = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )
        assert success1 and lc1.aspect == ProjectAspect.CORE

        # Create BACKEND aspect lifecycle
        success2, _, lc2 = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.BACKEND,
            created_by="user1",
        )
        assert success2 and lc2.aspect == ProjectAspect.BACKEND

        # Verify different lifecycle IDs
        assert lc1.lifecycle_id != lc2.lifecycle_id

    @pytest.mark.asyncio
    async def test_get_lifecycles_by_aspect(self, lifecycle_manager):
        """Test retrieving lifecycles filtered by aspect."""
        # Create lifecycles for different aspects
        await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )
        await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.BACKEND,
            created_by="user1",
        )
        await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )

        # Get only CORE aspect lifecycles
        core_lifecycles = await lifecycle_manager.get_lifecycles_for_aspect(
            project_name="test-project",
            aspect=ProjectAspect.CORE,
        )

        assert len(core_lifecycles) == 2
        assert all(lc.aspect == ProjectAspect.CORE for lc in core_lifecycles)

    @pytest.mark.asyncio
    async def test_aspect_isolation_check(self, lifecycle_manager):
        """Test aspect isolation validation."""
        success, _, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )

        # Check against correct aspect
        valid, _ = await lifecycle_manager.check_aspect_isolation(
            lifecycle.lifecycle_id, ProjectAspect.CORE
        )
        assert valid is True

        # Check against wrong aspect
        invalid, message = await lifecycle_manager.check_aspect_isolation(
            lifecycle.lifecycle_id, ProjectAspect.BACKEND
        )
        assert invalid is False
        assert "not" in message.lower()


# -----------------------------------------------------------------------------
# Test 4: PROJECT_MODE vs CHANGE_MODE
# -----------------------------------------------------------------------------
class TestLifecycleModes:
    """Test PROJECT_MODE and CHANGE_MODE behavior."""

    @pytest.mark.asyncio
    async def test_project_mode_no_change_reference(self, lifecycle_manager):
        """Test that PROJECT_MODE doesn't require change_reference."""
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )

        assert success is True
        assert lifecycle.mode == LifecycleMode.PROJECT_MODE
        assert lifecycle.change_reference is None

    @pytest.mark.asyncio
    async def test_change_mode_requires_reference(self, lifecycle_manager):
        """Test that CHANGE_MODE requires change_reference."""
        # Should fail without change_reference
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
            change_reference=None,
        )

        assert success is False
        assert "change_reference" in message.lower()

    @pytest.mark.asyncio
    async def test_change_mode_with_reference(self, lifecycle_manager):
        """Test CHANGE_MODE with valid change_reference."""
        change_ref = ChangeReference(
            project_id="parent-project-id",
            aspect=ProjectAspect.CORE,
            change_type=ChangeType.BUG,
            description="Fix login issue",
        )

        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
            change_reference=change_ref,
        )

        assert success is True
        assert lifecycle.mode == LifecycleMode.CHANGE_MODE
        assert lifecycle.change_reference is not None
        assert lifecycle.change_reference.change_type == ChangeType.BUG

    @pytest.mark.asyncio
    async def test_all_change_types(self, lifecycle_manager):
        """Test all change types in CHANGE_MODE."""
        for change_type in ChangeType:
            change_ref = ChangeReference(
                project_id="parent-id",
                aspect=ProjectAspect.CORE,
                change_type=change_type,
            )

            success, _, lifecycle = await lifecycle_manager.create_lifecycle(
                project_name=f"test-{change_type.value}",
                mode=LifecycleMode.CHANGE_MODE,
                aspect=ProjectAspect.CORE,
                created_by="user1",
                change_reference=change_ref,
            )

            assert success is True
            assert lifecycle.change_reference.change_type == change_type


# -----------------------------------------------------------------------------
# Test 5: Event-Driven Transitions
# -----------------------------------------------------------------------------
class TestEventDrivenTransitions:
    """Test event-driven transition triggers."""

    @pytest.mark.asyncio
    async def test_claude_job_completed_trigger(self, state_machine):
        """Test CLAUDE_JOB_COMPLETED trigger."""
        lifecycle = create_test_lifecycle(state=LifecycleState.PLANNING)

        success, _, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CLAUDE_JOB_COMPLETED,
            triggered_by="claude-cli",
            user_role=UserRole.DEVELOPER,
        )

        assert success is True
        assert new_state == LifecycleState.DEVELOPMENT

    @pytest.mark.asyncio
    async def test_test_passed_trigger(self, state_machine):
        """Test TEST_PASSED trigger."""
        lifecycle = create_test_lifecycle(state=LifecycleState.TESTING)

        success, _, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TEST_PASSED,
            triggered_by="ci-system",
            user_role=UserRole.TESTER,
        )

        assert success is True
        assert new_state == LifecycleState.AWAITING_FEEDBACK

    @pytest.mark.asyncio
    async def test_test_failed_trigger(self, state_machine):
        """Test TEST_FAILED trigger goes to FIXING."""
        lifecycle = create_test_lifecycle(state=LifecycleState.TESTING)

        success, _, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TEST_FAILED,
            triggered_by="ci-system",
            user_role=UserRole.TESTER,
        )

        assert success is True
        assert new_state == LifecycleState.FIXING

    @pytest.mark.asyncio
    async def test_telegram_feedback_trigger(self, state_machine):
        """Test TELEGRAM_FEEDBACK trigger from AWAITING_FEEDBACK."""
        lifecycle = create_test_lifecycle(state=LifecycleState.AWAITING_FEEDBACK)

        success, _, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TELEGRAM_FEEDBACK,
            triggered_by="tester-user",
            user_role=UserRole.TESTER,
        )

        assert success is True
        assert new_state == LifecycleState.FIXING

    @pytest.mark.asyncio
    async def test_rejection_from_various_states(self, state_machine):
        """Test HUMAN_REJECTION can occur from multiple states."""
        rejectable_states = [
            LifecycleState.PLANNING,
            LifecycleState.DEVELOPMENT,
            LifecycleState.TESTING,
            LifecycleState.AWAITING_FEEDBACK,
            LifecycleState.READY_FOR_PRODUCTION,
        ]

        for state in rejectable_states:
            lifecycle = create_test_lifecycle(state=state)

            success, _, new_state = await state_machine.transition(
                lifecycle=lifecycle,
                trigger=TransitionTrigger.HUMAN_REJECTION,
                triggered_by="admin",
                user_role=UserRole.ADMIN,
                reason=f"Rejected from {state.value}",
            )

            assert success is True, f"Rejection should work from {state.value}"
            assert new_state == LifecycleState.REJECTED


# -----------------------------------------------------------------------------
# Test 6: Role-Based Permissions
# -----------------------------------------------------------------------------
class TestRolePermissions:
    """Test role-based access control for transitions."""

    @pytest.mark.asyncio
    async def test_viewer_cannot_transition(self, state_machine):
        """Test that VIEWER role cannot trigger most transitions."""
        lifecycle = create_test_lifecycle(state=LifecycleState.AWAITING_FEEDBACK)

        success, message, _ = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="viewer-user",
            user_role=UserRole.VIEWER,
        )

        assert success is False
        assert "Role" in message or "cannot" in message.lower()

    @pytest.mark.asyncio
    async def test_tester_can_submit_test_results(self, state_machine):
        """Test that TESTER role can submit test results."""
        lifecycle = create_test_lifecycle(state=LifecycleState.TESTING)

        success, _, _ = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TEST_PASSED,
            triggered_by="tester",
            user_role=UserRole.TESTER,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_only_admin_owner_can_approve(self, state_machine):
        """Test that only ADMIN and OWNER can approve."""
        lifecycle = create_test_lifecycle(state=LifecycleState.AWAITING_FEEDBACK)

        # DEVELOPER should fail
        success, _, _ = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="dev",
            user_role=UserRole.DEVELOPER,
        )
        assert success is False

        # ADMIN should succeed
        lifecycle2 = create_test_lifecycle(state=LifecycleState.AWAITING_FEEDBACK)
        success, _, _ = await state_machine.transition(
            lifecycle=lifecycle2,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="admin",
            user_role=UserRole.ADMIN,
        )
        assert success is True

    @pytest.mark.asyncio
    async def test_owner_has_full_access(self, state_machine):
        """Test that OWNER role has access to all triggers."""
        for trigger in TransitionTrigger:
            allowed_roles = TRANSITION_PERMISSIONS.get(trigger, set())
            assert UserRole.OWNER in allowed_roles, f"OWNER should have access to {trigger.value}"


# -----------------------------------------------------------------------------
# Test 7: Persistence and Recovery
# -----------------------------------------------------------------------------
class TestPersistenceRecovery:
    """Test persistence and crash recovery."""

    @pytest.mark.asyncio
    async def test_lifecycle_persisted(self, lifecycle_manager, temp_state_dir):
        """Test that lifecycle is persisted to disk."""
        success, _, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )

        assert success is True

        # Check state file exists
        state_file = temp_state_dir / "lifecycles.json"
        assert state_file.exists()

        # Verify content
        with open(state_file) as f:
            state = json.load(f)
        assert lifecycle.lifecycle_id in state["lifecycles"]

    @pytest.mark.asyncio
    async def test_lifecycle_recovered(self, temp_state_dir):
        """Test that lifecycle can be recovered from disk."""
        # Create manager and lifecycle
        manager1 = LifecycleManager(state_dir=temp_state_dir)
        success, _, lifecycle = await manager1.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )

        lifecycle_id = lifecycle.lifecycle_id

        # Create new manager (simulating restart)
        manager2 = LifecycleManager(state_dir=temp_state_dir)

        # Recover and verify
        recovered = await manager2.get_lifecycle(lifecycle_id)
        assert recovered is not None
        assert recovered.lifecycle_id == lifecycle_id
        assert recovered.project_name == "test-project"

    @pytest.mark.asyncio
    async def test_state_recovery_summary(self, lifecycle_manager):
        """Test recovery summary includes correct counts."""
        # Create several lifecycles
        await lifecycle_manager.create_lifecycle(
            project_name="project1",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user1",
        )
        await lifecycle_manager.create_lifecycle(
            project_name="project2",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.BACKEND,
            created_by="user1",
            change_reference=ChangeReference(
                project_id="p1",
                aspect=ProjectAspect.BACKEND,
                change_type=ChangeType.FEATURE,
            ),
        )

        # Recover state
        summary = await lifecycle_manager.recover_state()

        assert summary["total"] == 2
        assert ProjectAspect.CORE.value in str(summary)
        assert len(summary["active"]) == 2


# -----------------------------------------------------------------------------
# Test 8: State Machine Configuration
# -----------------------------------------------------------------------------
class TestStateMachineConfig:
    """Test state machine configuration."""

    def test_all_states_have_transitions_defined(self):
        """Test that all states have transition rules."""
        for state in LifecycleState:
            assert state in VALID_TRANSITIONS, f"Missing transitions for {state.value}"

    def test_terminal_states_have_no_outgoing(self):
        """Test that terminal states have no outgoing transitions."""
        terminal_states = LifecycleState.terminal_states()
        for state in terminal_states:
            transitions = VALID_TRANSITIONS.get(state, {})
            # DEPLOYED can go to ARCHIVED, REJECTED can go to ARCHIVED
            if state == LifecycleState.ARCHIVED:
                assert len(transitions) == 0, f"{state.value} should have no outgoing transitions"

    def test_all_triggers_have_permissions(self):
        """Test that all triggers have permission rules."""
        for trigger in TransitionTrigger:
            assert trigger in TRANSITION_PERMISSIONS, f"Missing permissions for {trigger.value}"

    def test_no_cycles_in_happy_path(self):
        """Test that happy path doesn't have cycles."""
        # Happy path: CREATED -> PLANNING -> DEVELOPMENT -> TESTING ->
        #             AWAITING_FEEDBACK -> READY_FOR_PRODUCTION -> DEPLOYED -> ARCHIVED
        happy_path = [
            LifecycleState.CREATED,
            LifecycleState.PLANNING,
            LifecycleState.DEVELOPMENT,
            LifecycleState.TESTING,
            LifecycleState.AWAITING_FEEDBACK,
            LifecycleState.READY_FOR_PRODUCTION,
            LifecycleState.DEPLOYED,
            LifecycleState.ARCHIVED,
        ]

        # Each state should only appear once
        assert len(happy_path) == len(set(happy_path))


# -----------------------------------------------------------------------------
# Test 9: Guidance System
# -----------------------------------------------------------------------------
class TestGuidanceSystem:
    """Test the guidance/next steps system."""

    def test_guidance_for_created(self, state_machine):
        """Test guidance for CREATED state."""
        lifecycle = create_test_lifecycle(state=LifecycleState.CREATED)
        guidance = state_machine.get_next_guidance(lifecycle)

        assert guidance["current_state"] == "created"
        assert "start_planning" in guidance["available_actions"]
        assert guidance["waiting_for"] is not None

    def test_guidance_for_awaiting_feedback(self, state_machine):
        """Test guidance for AWAITING_FEEDBACK state."""
        lifecycle = create_test_lifecycle(state=LifecycleState.AWAITING_FEEDBACK)
        guidance = state_machine.get_next_guidance(lifecycle)

        assert guidance["current_state"] == "awaiting_feedback"
        assert "approve" in guidance["available_actions"]
        assert "provide_feedback" in guidance["available_actions"]
        assert "reject" in guidance["available_actions"]

    def test_guidance_for_terminal_state(self, state_machine):
        """Test guidance for terminal state."""
        lifecycle = create_test_lifecycle(state=LifecycleState.ARCHIVED)
        guidance = state_machine.get_next_guidance(lifecycle)

        assert guidance["current_state"] == "archived"
        assert len(guidance["available_actions"]) == 0


# -----------------------------------------------------------------------------
# Test 10: Serialization
# -----------------------------------------------------------------------------
class TestSerialization:
    """Test lifecycle serialization/deserialization."""

    def test_lifecycle_to_dict(self):
        """Test LifecycleInstance.to_dict()."""
        lifecycle = create_test_lifecycle()
        data = lifecycle.to_dict()

        assert data["lifecycle_id"] == lifecycle.lifecycle_id
        assert data["project_name"] == lifecycle.project_name
        assert data["mode"] == lifecycle.mode.value
        assert data["state"] == lifecycle.state.value

    def test_lifecycle_from_dict_roundtrip(self):
        """Test serialization roundtrip."""
        original = create_test_lifecycle()
        original.claude_job_ids = ["job1", "job2"]
        original.transition_count = 5

        data = original.to_dict()
        restored = LifecycleInstance.from_dict(data)

        assert restored.lifecycle_id == original.lifecycle_id
        assert restored.mode == original.mode
        assert restored.state == original.state
        assert restored.claude_job_ids == original.claude_job_ids
        assert restored.transition_count == original.transition_count

    def test_change_reference_serialization(self):
        """Test ChangeReference serialization."""
        change_ref = ChangeReference(
            project_id="parent-id",
            aspect=ProjectAspect.BACKEND,
            change_type=ChangeType.BUG,
            description="Test bug",
        )

        lifecycle = LifecycleInstance(
            lifecycle_id="test-id",
            project_name="test",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.BACKEND,
            state=LifecycleState.CREATED,
            created_at=datetime.utcnow(),
            created_by="user",
            change_reference=change_ref,
        )

        data = lifecycle.to_dict()
        restored = LifecycleInstance.from_dict(data)

        assert restored.change_reference is not None
        assert restored.change_reference.project_id == "parent-id"
        assert restored.change_reference.change_type == ChangeType.BUG


# -----------------------------------------------------------------------------
# Phase 15.2: Continuous Change Cycle Tests
# -----------------------------------------------------------------------------

class TestContinuousChangeCycles:
    """Test continuous change cycle support (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_deployed_to_awaiting_feedback(self, state_machine):
        """Test transition from DEPLOYED to AWAITING_FEEDBACK via continuous change."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="user-1",
            user_role=UserRole.DEVELOPER,
            reason="New feature request",
        )

        assert success is True
        assert new_state == LifecycleState.AWAITING_FEEDBACK

    @pytest.mark.asyncio
    async def test_deployed_not_terminal_state(self):
        """Test that DEPLOYED is no longer a terminal state."""
        assert LifecycleState.DEPLOYED not in LifecycleState.terminal_states()
        assert LifecycleState.DEPLOYED in LifecycleState.deployed_states()

    @pytest.mark.asyncio
    async def test_cycle_number_increments(self, state_machine):
        """Test that cycle number increments on continuous change request."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)
        lifecycle.cycle_number = 1
        lifecycle.current_claude_job_id = "job-1"

        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="user-1",
            user_role=UserRole.DEVELOPER,
            reason="New feature",
        )

        assert lifecycle.cycle_number == 2

    @pytest.mark.asyncio
    async def test_previous_deployment_id_tracked(self, state_machine):
        """Test that previous deployment ID is tracked on continuous change."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)
        lifecycle.cycle_number = 1
        lifecycle.current_claude_job_id = "job-1"

        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="user-1",
            user_role=UserRole.DEVELOPER,
            reason="Bug fix",
        )

        assert lifecycle.previous_deployment_id == "job-1"

    @pytest.mark.asyncio
    async def test_change_summary_recorded(self, state_machine):
        """Test that change summary is recorded on continuous change."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)

        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="user-1",
            user_role=UserRole.DEVELOPER,
            reason="Add login functionality",
        )

        assert lifecycle.current_change_summary == "Add login functionality"

    @pytest.mark.asyncio
    async def test_cycle_history_recorded(self, state_machine):
        """Test that cycle history is recorded."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)
        lifecycle.cycle_number = 1
        lifecycle.current_claude_job_id = "job-1"
        lifecycle.current_change_summary = "Initial deployment"

        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="user-1",
            user_role=UserRole.DEVELOPER,
            reason="Feature update",
        )

        assert len(lifecycle.cycle_history) == 1
        assert lifecycle.cycle_history[0]["cycle_number"] == 1
        assert lifecycle.cycle_history[0]["change_summary"] == "Initial deployment"

    @pytest.mark.asyncio
    async def test_full_continuous_loop(self, state_machine):
        """Test full continuous change loop: DEPLOYED -> AWAITING -> FIXING -> TESTING -> AWAITING -> READY -> DEPLOYED."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)
        lifecycle.cycle_number = 1

        # Start continuous change
        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="user-1",
            user_role=UserRole.DEVELOPER,
            reason="Bug fix",
        )
        assert lifecycle.state == LifecycleState.AWAITING_FEEDBACK
        assert lifecycle.cycle_number == 2

        # Provide feedback -> FIXING
        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TELEGRAM_FEEDBACK,
            triggered_by="user-1",
            user_role=UserRole.TESTER,
            reason="Found issue",
        )
        assert lifecycle.state == LifecycleState.FIXING

        # Fix complete -> TESTING
        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CLAUDE_JOB_COMPLETED,
            triggered_by="system",
            user_role=UserRole.DEVELOPER,
        )
        assert lifecycle.state == LifecycleState.TESTING

        # Tests pass -> AWAITING_FEEDBACK
        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.TEST_PASSED,
            triggered_by="system",
            user_role=UserRole.TESTER,
        )
        assert lifecycle.state == LifecycleState.AWAITING_FEEDBACK

        # Approve -> READY_FOR_PRODUCTION
        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="user-1",
            user_role=UserRole.ADMIN,
        )
        assert lifecycle.state == LifecycleState.READY_FOR_PRODUCTION

        # Final approval -> DEPLOYED
        await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.HUMAN_APPROVAL,
            triggered_by="user-2",
            user_role=UserRole.OWNER,
        )
        assert lifecycle.state == LifecycleState.DEPLOYED


class TestSecurityChangeType:
    """Test SECURITY change type (Phase 15.2)."""

    def test_security_change_type_exists(self):
        """Test that SECURITY change type is defined."""
        assert ChangeType.SECURITY.value == "security"

    @pytest.mark.asyncio
    async def test_security_change_lifecycle(self, lifecycle_manager):
        """Test creating a lifecycle with SECURITY change type."""
        change_ref = ChangeReference(
            project_id="parent-project",
            aspect=ProjectAspect.CORE,
            change_type=ChangeType.SECURITY,
            description="Fix SQL injection vulnerability",
        )

        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.CORE,
            created_by="security-user",
            change_reference=change_ref,
        )

        assert success is True
        assert lifecycle.change_reference.change_type == ChangeType.SECURITY


class TestCycleHistory:
    """Test cycle history tracking (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_get_cycle_history(self, lifecycle_manager):
        """Test getting cycle history for a lifecycle."""
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user-1",
        )

        # Add some cycle history manually
        lifecycle.cycle_history.append({
            "cycle_number": 1,
            "completed_at": datetime.utcnow().isoformat(),
            "change_summary": "Initial deployment",
            "deployment_job_id": "job-1",
        })
        lifecycle.cycle_number = 2
        await lifecycle_manager._save_lifecycle(lifecycle)

        success, message, history = await lifecycle_manager.get_cycle_history(lifecycle.lifecycle_id)

        assert success is True
        assert len(history) == 2  # 1 historical + 1 current
        assert history[0]["cycle_number"] == 1
        assert history[1]["is_current"] is True

    @pytest.mark.asyncio
    async def test_get_change_lineage(self, lifecycle_manager):
        """Test getting change lineage for a lifecycle."""
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user-1",
        )

        # Add cycle history with deployment
        lifecycle.cycle_history.append({
            "cycle_number": 1,
            "completed_at": datetime.utcnow().isoformat(),
            "change_summary": "Initial",
            "deployment_job_id": "deploy-job-1",
        })
        lifecycle.cycle_number = 2
        lifecycle.state = LifecycleState.DEPLOYED
        lifecycle.current_claude_job_id = "deploy-job-2"
        await lifecycle_manager._save_lifecycle(lifecycle)

        success, message, lineage = await lifecycle_manager.get_change_lineage(lifecycle.lifecycle_id)

        assert success is True
        assert lineage["total_cycles"] == 2
        assert len(lineage["deployments"]) == 2


class TestDeploymentSummary:
    """Test deployment summary generation (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_generate_deployment_summary(self, lifecycle_manager):
        """Test generating a deployment summary."""
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="summary-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.BACKEND,
            created_by="user-1",
        )

        # Add some data
        lifecycle.transition_count = 5
        lifecycle.claude_job_ids = ["job-1", "job-2"]
        lifecycle.test_run_ids = ["test-1"]
        lifecycle.current_change_summary = "Added API endpoints"
        await lifecycle_manager._save_lifecycle(lifecycle)

        success, message, summary = await lifecycle_manager.generate_deployment_summary(lifecycle.lifecycle_id)

        assert success is True
        assert summary["project_name"] == "summary-project"
        assert summary["aspect"] == "backend"
        assert summary["total_transitions"] == 5
        assert summary["total_claude_jobs"] == 2


class TestContinuousChangeRequest:
    """Test request_continuous_change method (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_request_change_success(self, lifecycle_manager):
        """Test successful continuous change request."""
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user-1",
        )

        # Move to DEPLOYED state
        lifecycle.state = LifecycleState.DEPLOYED
        lifecycle.cycle_number = 1
        await lifecycle_manager._save_lifecycle(lifecycle)

        success, message, cycle_number = await lifecycle_manager.request_continuous_change(
            lifecycle_id=lifecycle.lifecycle_id,
            change_type=ChangeType.FEATURE,
            change_summary="Add user dashboard",
            requested_by="user-1",
            user_role=UserRole.DEVELOPER,
        )

        assert success is True
        assert cycle_number == 2

        # Verify state changed
        updated = await lifecycle_manager.get_lifecycle(lifecycle.lifecycle_id)
        assert updated.state == LifecycleState.AWAITING_FEEDBACK

    @pytest.mark.asyncio
    async def test_request_change_not_deployed_fails(self, lifecycle_manager):
        """Test that change request fails if not in DEPLOYED state."""
        success, message, lifecycle = await lifecycle_manager.create_lifecycle(
            project_name="test-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user-1",
        )

        # Keep in CREATED state
        success, message, cycle_number = await lifecycle_manager.request_continuous_change(
            lifecycle_id=lifecycle.lifecycle_id,
            change_type=ChangeType.BUG,
            change_summary="Fix bug",
            requested_by="user-1",
            user_role=UserRole.DEVELOPER,
        )

        assert success is False
        assert "must be 'deployed'" in message

    @pytest.mark.asyncio
    async def test_all_change_types_accepted(self, lifecycle_manager):
        """Test all change types can be requested."""
        for change_type in [ChangeType.BUG, ChangeType.FEATURE, ChangeType.IMPROVEMENT, ChangeType.REFACTOR, ChangeType.SECURITY]:
            success, message, lifecycle = await lifecycle_manager.create_lifecycle(
                project_name=f"test-{change_type.value}",
                mode=LifecycleMode.PROJECT_MODE,
                aspect=ProjectAspect.CORE,
                created_by="user-1",
            )

            lifecycle.state = LifecycleState.DEPLOYED
            lifecycle.cycle_number = 1
            await lifecycle_manager._save_lifecycle(lifecycle)

            success, message, cycle_number = await lifecycle_manager.request_continuous_change(
                lifecycle_id=lifecycle.lifecycle_id,
                change_type=change_type,
                change_summary=f"Test {change_type.value}",
                requested_by="user-1",
                user_role=UserRole.DEVELOPER,
            )

            assert success is True, f"Failed for {change_type.value}: {message}"


class TestProjectChanges:
    """Test get_changes_for_project method (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_get_changes_for_project(self, lifecycle_manager):
        """Test getting all changes for a project."""
        # Create a project lifecycle
        await lifecycle_manager.create_lifecycle(
            project_name="my-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user-1",
        )

        # Create change lifecycles
        change_ref = ChangeReference(
            project_id="my-project-id",
            aspect=ProjectAspect.CORE,
            change_type=ChangeType.BUG,
            description="Bug fix 1",
        )
        await lifecycle_manager.create_lifecycle(
            project_name="my-project",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.CORE,
            created_by="user-1",
            change_reference=change_ref,
        )

        change_ref2 = ChangeReference(
            project_id="my-project-id",
            aspect=ProjectAspect.BACKEND,
            change_type=ChangeType.FEATURE,
            description="Feature 1",
        )
        await lifecycle_manager.create_lifecycle(
            project_name="my-project",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.BACKEND,
            created_by="user-1",
            change_reference=change_ref2,
        )

        # Get all changes
        changes = await lifecycle_manager.get_changes_for_project("my-project")
        assert len(changes) == 2

        # Get changes by aspect
        core_changes = await lifecycle_manager.get_changes_for_project("my-project", ProjectAspect.CORE)
        assert len(core_changes) == 1
        assert core_changes[0]["aspect"] == "core"


class TestAspectHistory:
    """Test get_aspect_history method (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_get_aspect_history(self, lifecycle_manager):
        """Test getting complete history for an aspect."""
        # Create multiple lifecycles for same aspect
        await lifecycle_manager.create_lifecycle(
            project_name="history-project",
            mode=LifecycleMode.PROJECT_MODE,
            aspect=ProjectAspect.FRONTEND_WEB,
            created_by="user-1",
        )

        change_ref = ChangeReference(
            project_id="history-project-id",
            aspect=ProjectAspect.FRONTEND_WEB,
            change_type=ChangeType.IMPROVEMENT,
            description="UI improvement",
        )
        await lifecycle_manager.create_lifecycle(
            project_name="history-project",
            mode=LifecycleMode.CHANGE_MODE,
            aspect=ProjectAspect.FRONTEND_WEB,
            created_by="user-1",
            change_reference=change_ref,
        )

        history = await lifecycle_manager.get_aspect_history("history-project", ProjectAspect.FRONTEND_WEB)

        assert history["project_name"] == "history-project"
        assert history["aspect"] == "frontend_web"
        assert history["total_lifecycles"] == 2


class TestContinuousChangePermissions:
    """Test permissions for continuous change requests (Phase 15.2)."""

    @pytest.mark.asyncio
    async def test_developer_can_request_change(self, state_machine):
        """Test that developer can request continuous change."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="dev-1",
            user_role=UserRole.DEVELOPER,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_tester_can_request_change(self, state_machine):
        """Test that tester can request continuous change."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="tester-1",
            user_role=UserRole.TESTER,
        )

        assert success is True

    @pytest.mark.asyncio
    async def test_viewer_cannot_request_change(self, state_machine):
        """Test that viewer cannot request continuous change."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)

        success, message, new_state = await state_machine.transition(
            lifecycle=lifecycle,
            trigger=TransitionTrigger.CONTINUOUS_CHANGE_REQUEST,
            triggered_by="viewer-1",
            user_role=UserRole.VIEWER,
        )

        assert success is False


class TestCycleDataFields:
    """Test cycle-related data fields in LifecycleInstance (Phase 15.2)."""

    def test_default_cycle_number(self):
        """Test that default cycle number is 1."""
        lifecycle = create_test_lifecycle()
        assert lifecycle.cycle_number == 1

    def test_default_cycle_history_empty(self):
        """Test that default cycle history is empty."""
        lifecycle = create_test_lifecycle()
        assert lifecycle.cycle_history == []

    def test_default_change_summary_empty(self):
        """Test that default change summary is empty."""
        lifecycle = create_test_lifecycle()
        assert lifecycle.current_change_summary == ""

    def test_default_previous_deployment_none(self):
        """Test that default previous deployment ID is None."""
        lifecycle = create_test_lifecycle()
        assert lifecycle.previous_deployment_id is None

    def test_cycle_fields_serialization(self):
        """Test that cycle fields serialize correctly."""
        lifecycle = create_test_lifecycle()
        lifecycle.cycle_number = 3
        lifecycle.cycle_history = [{"cycle_number": 1}, {"cycle_number": 2}]
        lifecycle.current_change_summary = "Test summary"
        lifecycle.previous_deployment_id = "job-123"

        data = lifecycle.to_dict()

        assert data["cycle_number"] == 3
        assert len(data["cycle_history"]) == 2
        assert data["current_change_summary"] == "Test summary"
        assert data["previous_deployment_id"] == "job-123"

    def test_cycle_fields_deserialization(self):
        """Test that cycle fields deserialize correctly."""
        lifecycle = create_test_lifecycle()
        lifecycle.cycle_number = 5
        lifecycle.cycle_history = [{"cycle_number": 1}]
        lifecycle.current_change_summary = "Deserialization test"
        lifecycle.previous_deployment_id = "prev-job"

        data = lifecycle.to_dict()
        restored = LifecycleInstance.from_dict(data)

        assert restored.cycle_number == 5
        assert len(restored.cycle_history) == 1
        assert restored.current_change_summary == "Deserialization test"
        assert restored.previous_deployment_id == "prev-job"


class TestGuidanceForDeployed:
    """Test guidance updates for DEPLOYED state (Phase 15.2)."""

    def test_deployed_guidance_includes_change_actions(self, state_machine):
        """Test that DEPLOYED state guidance includes change actions."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)
        guidance = state_machine.get_next_guidance(lifecycle)

        assert "new_feature" in guidance["available_actions"]
        assert "report_bug" in guidance["available_actions"]
        assert "improve" in guidance["available_actions"]
        assert "refactor" in guidance["available_actions"]
        assert "security_fix" in guidance["available_actions"]
        assert "archive" in guidance["available_actions"]

    def test_deployed_guidance_includes_cycle_number(self, state_machine):
        """Test that DEPLOYED guidance includes cycle number."""
        lifecycle = create_test_lifecycle(state=LifecycleState.DEPLOYED)
        lifecycle.cycle_number = 5
        guidance = state_machine.get_next_guidance(lifecycle)

        assert guidance.get("cycle_number") == 5


class TestTransitionTriggers:
    """Test Phase 15.2 transition triggers."""

    def test_continuous_change_request_trigger_exists(self):
        """Test that CONTINUOUS_CHANGE_REQUEST trigger exists."""
        assert TransitionTrigger.CONTINUOUS_CHANGE_REQUEST.value == "continuous_change_request"

    def test_continuous_change_request_permission(self):
        """Test permissions for CONTINUOUS_CHANGE_REQUEST trigger."""
        perms = TRANSITION_PERMISSIONS[TransitionTrigger.CONTINUOUS_CHANGE_REQUEST]
        assert UserRole.OWNER in perms
        assert UserRole.ADMIN in perms
        assert UserRole.DEVELOPER in perms
        assert UserRole.TESTER in perms
        assert UserRole.VIEWER not in perms

    def test_deployed_has_continuous_change_transition(self):
        """Test that DEPLOYED state can transition to AWAITING_FEEDBACK."""
        transitions = VALID_TRANSITIONS[LifecycleState.DEPLOYED]
        assert TransitionTrigger.CONTINUOUS_CHANGE_REQUEST in transitions
        assert transitions[TransitionTrigger.CONTINUOUS_CHANGE_REQUEST] == LifecycleState.AWAITING_FEEDBACK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
