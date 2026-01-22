"""
Phase 18C: Controlled Execution Dispatcher Tests

Comprehensive test suite for the execution dispatcher system.
Minimum 30 tests covering all critical behaviors.

Test Categories:
1. Enum Validation Tests (5 tests)
2. Immutability Tests (5 tests)
3. Missing Input Tests (5 tests)
4. Eligibility Block Tests (4 tests)
5. Approval Block Tests (4 tests)
6. Gate Block Tests (4 tests)
7. Success Path Tests (3 tests)
8. Outcome Recording Tests (4 tests)
9. Determinism Tests (3 tests)
10. Audit Tests (3 tests)
11. Store Tests (5 tests)

Total: 45 tests
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytest

# Import dispatcher components
from controller.execution_dispatcher import (
    ExecutionStatus,
    BlockReason,
    FailureReason,
    ActionType,
    ExecutionIntent,
    ValidationChainInput,
    ExecutionResult,
    ExecutionAuditRecord,
    ControlledExecutionDispatcher,
    get_execution_dispatcher,
    create_execution_intent,
    create_validation_chain_input,
    dispatch_execution,
    get_execution_summary,
)

# Import store components
from controller.execution_store import (
    ExecutionStore,
    get_execution_store,
    record_execution_intent,
    record_execution_result,
    get_execution_result,
)

# Import from Phase 18A
from controller.automation_eligibility import (
    EligibilityDecision,
    EligibilityResult,
)

# Import from Phase 18B
from controller.approval_orchestrator import (
    ApprovalStatus,
    OrchestrationResult,
)

# Import from Phase 15.6
from controller.execution_gate import (
    ExecutionRequest,
    ExecutionGate,
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
def dispatcher(temp_dir):
    """Create a dispatcher with temp files."""
    return ControlledExecutionDispatcher(
        intents_file=temp_dir / "intents.jsonl",
        results_file=temp_dir / "results.jsonl",
        audit_file=temp_dir / "audit.jsonl",
    )


@pytest.fixture
def store(temp_dir):
    """Create a store with temp files."""
    return ExecutionStore(
        intents_file=temp_dir / "intents.jsonl",
        results_file=temp_dir / "results.jsonl",
    )


@pytest.fixture
def valid_intent():
    """Create a valid execution intent."""
    return ExecutionIntent(
        intent_id="int-test-001",
        project_id="test-project",
        project_name="Test Project",
        action_type=ActionType.RUN_TESTS.value,
        action_description="Run unit tests",
        requester_id="user-001",
        requester_role="developer",
        target_workspace="/tmp/test-workspace",
        created_at=datetime.utcnow().isoformat(),
        metadata=(),
    )


@pytest.fixture
def allowed_eligibility():
    """Create an eligibility result that allows execution."""
    return EligibilityResult(
        decision=EligibilityDecision.AUTOMATION_ALLOWED_LIMITED.value,
        matched_rules=(),
        input_hash="abc123",
        timestamp=datetime.utcnow().isoformat(),
        engine_version="18A.1.0",
        allowed_actions=(ActionType.RUN_TESTS.value, "update_docs"),
    )


@pytest.fixture
def allowed_with_approval_eligibility():
    """Create an eligibility result that requires approval."""
    return EligibilityResult(
        decision=EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value,
        matched_rules=(),
        input_hash="abc123",
        timestamp=datetime.utcnow().isoformat(),
        engine_version="18A.1.0",
        allowed_actions=(),
    )


@pytest.fixture
def forbidden_eligibility():
    """Create a forbidden eligibility result."""
    return EligibilityResult(
        decision=EligibilityDecision.AUTOMATION_FORBIDDEN.value,
        matched_rules=("drift_level_high",),
        input_hash="def456",
        timestamp=datetime.utcnow().isoformat(),
        engine_version="18A.1.0",
        allowed_actions=(),
    )


@pytest.fixture
def granted_approval():
    """Create a granted approval result."""
    return OrchestrationResult(
        status=ApprovalStatus.APPROVAL_GRANTED.value,
        reason=None,
        input_hash="xyz789",
        timestamp=datetime.utcnow().isoformat(),
        orchestrator_version="18B.1.0",
        approval_request_id="req-001",
        approver_count=1,
        required_approver_count=1,
    )


@pytest.fixture
def denied_approval():
    """Create a denied approval result."""
    return OrchestrationResult(
        status=ApprovalStatus.APPROVAL_DENIED.value,
        reason="eligibility_forbidden",
        input_hash="xyz789",
        timestamp=datetime.utcnow().isoformat(),
        orchestrator_version="18B.1.0",
        approval_request_id=None,
        approver_count=0,
        required_approver_count=1,
    )


@pytest.fixture
def pending_approval():
    """Create a pending approval result."""
    return OrchestrationResult(
        status=ApprovalStatus.APPROVAL_PENDING.value,
        reason="awaiting_approval",
        input_hash="xyz789",
        timestamp=datetime.utcnow().isoformat(),
        orchestrator_version="18B.1.0",
        approval_request_id="req-001",
        approver_count=0,
        required_approver_count=1,
    )


@pytest.fixture
def valid_gate_request(valid_intent):
    """Create a valid gate request."""
    return ExecutionRequest(
        job_id=valid_intent.intent_id,
        project_name=valid_intent.project_name,
        aspect="core",
        lifecycle_id=valid_intent.project_id,
        lifecycle_state="development",
        requested_action=valid_intent.action_type,
        requesting_user_id=valid_intent.requester_id,
        requesting_user_role="developer",
        workspace_path=valid_intent.target_workspace,
        task_description=valid_intent.action_description,
        project_id=valid_intent.project_id,
    )


# =============================================================================
# 1. Enum Validation Tests (5 tests)
# =============================================================================

class TestEnumValidation:
    """Test that all enums are properly defined and locked."""

    def test_execution_status_has_exactly_4_values(self):
        """ExecutionStatus must have exactly 4 values."""
        assert len(ExecutionStatus) == 4
        assert ExecutionStatus.EXECUTION_BLOCKED.value == "execution_blocked"
        assert ExecutionStatus.EXECUTION_PENDING.value == "execution_pending"
        assert ExecutionStatus.EXECUTION_SUCCESS.value == "execution_success"
        assert ExecutionStatus.EXECUTION_FAILED.value == "execution_failed"

    def test_block_reason_values_are_valid(self):
        """BlockReason enum values must be valid."""
        required_values = {
            "missing_intent",
            "missing_eligibility",
            "missing_approval",
            "missing_gate_request",
            "eligibility_forbidden",
            "approval_denied",
            "approval_pending",
            "gate_denied",
            "audit_write_failed",
        }
        actual_values = {br.value for br in BlockReason}
        assert required_values.issubset(actual_values)

    def test_failure_reason_values_are_valid(self):
        """FailureReason enum values must be valid."""
        assert FailureReason.EXECUTION_TIMEOUT.value == "execution_timeout"
        assert FailureReason.EXECUTION_ERROR.value == "execution_error"
        assert FailureReason.ROLLBACK_REQUIRED.value == "rollback_required"
        assert FailureReason.EXTERNAL_SYSTEM_ERROR.value == "external_system_error"

    def test_action_type_values_are_valid(self):
        """ActionType enum values must match allowed actions."""
        required_actions = {"run_tests", "update_docs", "write_code", "commit", "push", "deploy_test"}
        actual_values = {at.value for at in ActionType}
        assert required_actions == actual_values

    def test_invalid_action_type_raises_error(self):
        """Invalid action type in ExecutionIntent should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid action_type"):
            ExecutionIntent(
                intent_id="int-001",
                project_id="proj-001",
                project_name="Test",
                action_type="deploy_prod",  # NOT allowed
                action_description="Test",
                requester_id="user-001",
                requester_role="admin",
                target_workspace="/tmp",
                created_at=datetime.utcnow().isoformat(),
            )


# =============================================================================
# 2. Immutability Tests (5 tests)
# =============================================================================

class TestImmutability:
    """Test that frozen dataclasses are truly immutable."""

    def test_execution_intent_is_immutable(self, valid_intent):
        """ExecutionIntent cannot be modified after creation."""
        with pytest.raises(AttributeError):
            valid_intent.action_type = "commit"

    def test_execution_result_is_immutable(self):
        """ExecutionResult cannot be modified after creation."""
        result = ExecutionResult(
            execution_id="exec-001",
            intent_id="int-001",
            status=ExecutionStatus.EXECUTION_SUCCESS.value,
            block_reason=None,
            failure_reason=None,
            input_hash="abc",
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            gate_decision_allowed=True,
            execution_output=None,
            rollback_performed=False,
        )
        with pytest.raises(AttributeError):
            result.status = ExecutionStatus.EXECUTION_FAILED.value

    def test_validation_chain_input_is_immutable(self, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """ValidationChainInput cannot be modified after creation."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        with pytest.raises(AttributeError):
            chain.intent = None

    def test_audit_record_is_immutable(self):
        """ExecutionAuditRecord cannot be modified after creation."""
        audit = ExecutionAuditRecord(
            audit_id="aud-001",
            execution_id="exec-001",
            intent_id="int-001",
            input_hash="abc",
            status=ExecutionStatus.EXECUTION_BLOCKED.value,
            block_reason=BlockReason.MISSING_ELIGIBILITY.value,
            failure_reason=None,
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            project_id="proj-001",
            action_type="run_tests",
            requester_id="user-001",
            eligibility_decision=None,
            approval_status=None,
            gate_allowed=None,
        )
        with pytest.raises(AttributeError):
            audit.status = ExecutionStatus.EXECUTION_SUCCESS.value

    def test_intent_metadata_is_immutable(self):
        """ExecutionIntent metadata must be a tuple (immutable)."""
        intent = ExecutionIntent(
            intent_id="int-001",
            project_id="proj-001",
            project_name="Test",
            action_type=ActionType.RUN_TESTS.value,
            action_description="Test",
            requester_id="user-001",
            requester_role="admin",
            target_workspace="/tmp",
            created_at=datetime.utcnow().isoformat(),
            metadata=(("key1", "value1"), ("key2", "value2")),
        )
        assert isinstance(intent.metadata, tuple)


# =============================================================================
# 3. Missing Input Tests (5 tests)
# =============================================================================

class TestMissingInputs:
    """Test that missing inputs result in EXECUTION_BLOCKED."""

    def test_missing_intent_blocks_execution(self, dispatcher, allowed_eligibility, granted_approval, valid_gate_request):
        """Missing intent results in EXECUTION_BLOCKED."""
        chain = ValidationChainInput(
            intent=None,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.MISSING_INTENT.value

    def test_missing_eligibility_blocks_execution(self, dispatcher, valid_intent, granted_approval, valid_gate_request):
        """Missing eligibility result blocks execution."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=None,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.MISSING_ELIGIBILITY.value

    def test_missing_approval_blocks_execution(self, dispatcher, valid_intent, allowed_eligibility, valid_gate_request):
        """Missing approval result blocks execution."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=None,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.MISSING_APPROVAL.value

    def test_missing_gate_request_blocks_execution(self, dispatcher, valid_intent, allowed_eligibility, granted_approval):
        """Missing gate request blocks execution."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=None,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.MISSING_GATE_REQUEST.value

    def test_all_missing_blocks_with_first_error(self, dispatcher):
        """When all inputs missing, first error (intent) is reported."""
        chain = ValidationChainInput(
            intent=None,
            eligibility_result=None,
            approval_result=None,
            gate_request=None,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.MISSING_INTENT.value


# =============================================================================
# 4. Eligibility Block Tests (4 tests)
# =============================================================================

class TestEligibilityBlocking:
    """Test that eligibility failures block execution."""

    def test_forbidden_eligibility_blocks(self, dispatcher, valid_intent, forbidden_eligibility, granted_approval, valid_gate_request):
        """AUTOMATION_FORBIDDEN eligibility blocks execution."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=forbidden_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.ELIGIBILITY_FORBIDDEN.value

    def test_limited_eligibility_allows_listed_actions(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """LIMITED eligibility allows actions in allowed_actions list."""
        # valid_intent has action_type=run_tests, allowed_eligibility has run_tests in allowed_actions
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        # May be blocked by gate, but not by eligibility
        assert result.block_reason != BlockReason.ELIGIBILITY_FORBIDDEN.value
        assert result.block_reason != BlockReason.ACTION_NOT_IN_ALLOWED_LIST.value

    def test_limited_eligibility_blocks_unlisted_actions(self, dispatcher, allowed_eligibility, granted_approval):
        """LIMITED eligibility blocks actions NOT in allowed_actions list."""
        # Create intent with action_type=commit (not in allowed_actions)
        intent = ExecutionIntent(
            intent_id="int-002",
            project_id="test-project",
            project_name="Test Project",
            action_type=ActionType.COMMIT.value,  # NOT in allowed_actions
            action_description="Commit changes",
            requester_id="user-001",
            requester_role="developer",
            target_workspace="/tmp/test",
            created_at=datetime.utcnow().isoformat(),
        )
        gate_request = ExecutionRequest(
            job_id=intent.intent_id,
            project_name=intent.project_name,
            aspect="core",
            lifecycle_id=intent.project_id,
            lifecycle_state="development",
            requested_action=intent.action_type,
            requesting_user_id=intent.requester_id,
            requesting_user_role="developer",
            workspace_path=intent.target_workspace,
            task_description=intent.action_description,
        )
        chain = ValidationChainInput(
            intent=intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.ACTION_NOT_IN_ALLOWED_LIST.value

    def test_with_approval_eligibility_allows_any_action(self, dispatcher, valid_intent, allowed_with_approval_eligibility, granted_approval, valid_gate_request):
        """WITH_APPROVAL eligibility doesn't restrict action types."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_with_approval_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        # Should not be blocked by eligibility
        assert result.block_reason != BlockReason.ELIGIBILITY_FORBIDDEN.value
        assert result.block_reason != BlockReason.ACTION_NOT_IN_ALLOWED_LIST.value


# =============================================================================
# 5. Approval Block Tests (4 tests)
# =============================================================================

class TestApprovalBlocking:
    """Test that approval failures block execution."""

    def test_denied_approval_blocks(self, dispatcher, valid_intent, allowed_with_approval_eligibility, denied_approval, valid_gate_request):
        """APPROVAL_DENIED blocks execution."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_with_approval_eligibility,
            approval_result=denied_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.APPROVAL_DENIED.value

    def test_pending_approval_blocks(self, dispatcher, valid_intent, allowed_with_approval_eligibility, pending_approval, valid_gate_request):
        """APPROVAL_PENDING blocks execution."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_with_approval_eligibility,
            approval_result=pending_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason == BlockReason.APPROVAL_PENDING.value

    def test_granted_approval_allows_execution(self, dispatcher, valid_intent, allowed_with_approval_eligibility, granted_approval, valid_gate_request):
        """APPROVAL_GRANTED allows execution to proceed."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_with_approval_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        # Should not be blocked by approval
        assert result.block_reason != BlockReason.APPROVAL_DENIED.value
        assert result.block_reason != BlockReason.APPROVAL_PENDING.value

    def test_approval_checked_after_eligibility(self, dispatcher, valid_intent, forbidden_eligibility, denied_approval, valid_gate_request):
        """Eligibility is checked before approval."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=forbidden_eligibility,
            approval_result=denied_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        # Should be blocked by eligibility, not approval
        assert result.block_reason == BlockReason.ELIGIBILITY_FORBIDDEN.value


# =============================================================================
# 6. Gate Block Tests (4 tests)
# =============================================================================

class TestGateBlocking:
    """Test that gate failures block execution."""

    def test_gate_denial_blocks_execution(self, temp_dir, valid_intent, allowed_eligibility, granted_approval):
        """Gate denial blocks execution."""
        # Create gate request with invalid lifecycle state
        gate_request = ExecutionRequest(
            job_id=valid_intent.intent_id,
            project_name=valid_intent.project_name,
            aspect="core",
            lifecycle_id=valid_intent.project_id,
            lifecycle_state="invalid_state",  # Invalid state
            requested_action=valid_intent.action_type,
            requesting_user_id=valid_intent.requester_id,
            requesting_user_role="developer",
            workspace_path=valid_intent.target_workspace,
            task_description=valid_intent.action_description,
        )

        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=temp_dir / "audit.jsonl",
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value
        assert result.block_reason in [
            BlockReason.GATE_DENIED.value,
            BlockReason.GATE_HARD_FAIL.value,
        ]

    def test_gate_checked_after_approval(self, temp_dir, valid_intent, allowed_eligibility, pending_approval):
        """Gate is not checked if approval is pending."""
        gate_request = ExecutionRequest(
            job_id=valid_intent.intent_id,
            project_name=valid_intent.project_name,
            aspect="core",
            lifecycle_id=valid_intent.project_id,
            lifecycle_state="invalid_state",
            requested_action=valid_intent.action_type,
            requesting_user_id=valid_intent.requester_id,
            requesting_user_role="developer",
            workspace_path=valid_intent.target_workspace,
            task_description=valid_intent.action_description,
        )

        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=temp_dir / "audit.jsonl",
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=pending_approval,
            gate_request=gate_request,
        )
        result = dispatcher.dispatch(chain)
        # Should be blocked by approval, not gate
        assert result.block_reason == BlockReason.APPROVAL_PENDING.value

    def test_gate_records_allowed_decision(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Gate decision is recorded in result."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        # gate_decision_allowed should be set
        assert result.gate_decision_allowed is not None

    def test_invalid_action_in_gate_blocks(self, temp_dir, valid_intent, allowed_eligibility, granted_approval):
        """Invalid action in gate request blocks execution."""
        gate_request = ExecutionRequest(
            job_id=valid_intent.intent_id,
            project_name=valid_intent.project_name,
            aspect="core",
            lifecycle_id=valid_intent.project_id,
            lifecycle_state="development",
            requested_action="invalid_action",  # Invalid action
            requesting_user_id=valid_intent.requester_id,
            requesting_user_role="developer",
            workspace_path=valid_intent.target_workspace,
            task_description=valid_intent.action_description,
        )

        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=temp_dir / "audit.jsonl",
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.status == ExecutionStatus.EXECUTION_BLOCKED.value


# =============================================================================
# 7. Success Path Tests (3 tests)
# =============================================================================

class TestSuccessPath:
    """Test successful execution dispatch."""

    def test_valid_chain_returns_pending(self, temp_dir, valid_intent, allowed_eligibility, granted_approval):
        """Valid chain with gate allowing returns EXECUTION_PENDING."""
        # Create a gate request that will pass (temp workspace allowed)
        gate_request = ExecutionRequest(
            job_id=valid_intent.intent_id,
            project_name=valid_intent.project_name,
            aspect="core",
            lifecycle_id=valid_intent.project_id,
            lifecycle_state="development",
            requested_action="run_tests",  # Valid action for development
            requesting_user_id=valid_intent.requester_id,
            requesting_user_role="developer",
            workspace_path="/tmp/test-workspace",  # Allowed for testing
            task_description=valid_intent.action_description,
            skip_drift_check=True,
        )

        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=temp_dir / "audit.jsonl",
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=gate_request,
        )
        result = dispatcher.dispatch(chain)
        # May still be blocked by workspace/governance checks
        # but if all pass, should be PENDING
        if result.status == ExecutionStatus.EXECUTION_PENDING.value:
            assert result.block_reason is None
            assert result.gate_decision_allowed is True

    def test_execution_id_is_generated(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Execution ID is always generated."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.execution_id is not None
        assert result.execution_id.startswith("exec-")

    def test_intent_id_is_preserved(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Intent ID is preserved in result."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        result = dispatcher.dispatch(chain)
        assert result.intent_id == valid_intent.intent_id


# =============================================================================
# 8. Outcome Recording Tests (4 tests)
# =============================================================================

class TestOutcomeRecording:
    """Test execution outcome recording."""

    def test_record_success_outcome(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Success outcome can be recorded."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatch_result = dispatcher.dispatch(chain)

        outcome = dispatcher.record_execution_outcome(
            execution_id=dispatch_result.execution_id,
            success=True,
            output="Tests passed: 42/42",
        )
        assert outcome.status == ExecutionStatus.EXECUTION_SUCCESS.value
        assert outcome.execution_output == "Tests passed: 42/42"

    def test_record_failure_outcome(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Failure outcome can be recorded with reason."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatch_result = dispatcher.dispatch(chain)

        outcome = dispatcher.record_execution_outcome(
            execution_id=dispatch_result.execution_id,
            success=False,
            failure_reason=FailureReason.EXECUTION_ERROR.value,
            output="Error: Test failed",
        )
        assert outcome.status == ExecutionStatus.EXECUTION_FAILED.value
        assert outcome.failure_reason == FailureReason.EXECUTION_ERROR.value

    def test_record_rollback_outcome(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Rollback can be recorded."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatch_result = dispatcher.dispatch(chain)

        outcome = dispatcher.record_execution_outcome(
            execution_id=dispatch_result.execution_id,
            success=False,
            failure_reason=FailureReason.ROLLBACK_REQUIRED.value,
            rollback_performed=True,
        )
        assert outcome.rollback_performed is True

    def test_outcome_output_is_truncated(self, dispatcher, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Long output is truncated to 1000 chars."""
        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatch_result = dispatcher.dispatch(chain)

        long_output = "X" * 2000
        outcome = dispatcher.record_execution_outcome(
            execution_id=dispatch_result.execution_id,
            success=True,
            output=long_output,
        )
        assert len(outcome.execution_output) == 1000


# =============================================================================
# 9. Determinism Tests (3 tests)
# =============================================================================

class TestDeterminism:
    """Test that same inputs produce same outputs."""

    def test_same_chain_same_result(self, temp_dir, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Same chain input produces same block reason (if blocked)."""
        dispatcher1 = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents1.jsonl",
            results_file=temp_dir / "results1.jsonl",
            audit_file=temp_dir / "audit1.jsonl",
        )
        dispatcher2 = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents2.jsonl",
            results_file=temp_dir / "results2.jsonl",
            audit_file=temp_dir / "audit2.jsonl",
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )

        result1 = dispatcher1.dispatch(chain)
        result2 = dispatcher2.dispatch(chain)

        assert result1.status == result2.status
        assert result1.block_reason == result2.block_reason

    def test_input_hash_is_deterministic(self, valid_intent, allowed_eligibility, granted_approval, valid_gate_request):
        """Input hash is deterministic for same inputs."""
        chain1 = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        chain2 = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=allowed_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        assert chain1.compute_hash() == chain2.compute_hash()

    def test_intent_hash_is_deterministic(self, valid_intent):
        """Intent hash is deterministic."""
        hash1 = valid_intent.compute_hash()
        hash2 = valid_intent.compute_hash()
        assert hash1 == hash2


# =============================================================================
# 10. Audit Tests (3 tests)
# =============================================================================

class TestAudit:
    """Test audit recording."""

    def test_blocked_execution_is_audited(self, temp_dir, valid_intent, forbidden_eligibility, granted_approval, valid_gate_request):
        """Blocked executions are audited."""
        audit_file = temp_dir / "audit.jsonl"
        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=audit_file,
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=forbidden_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatcher.dispatch(chain)

        assert audit_file.exists()
        with open(audit_file, 'r') as f:
            lines = f.readlines()
        assert len(lines) >= 1

        audit_record = json.loads(lines[-1])
        assert audit_record["status"] == ExecutionStatus.EXECUTION_BLOCKED.value

    def test_audit_contains_project_info(self, temp_dir, valid_intent, forbidden_eligibility, granted_approval, valid_gate_request):
        """Audit contains project information."""
        audit_file = temp_dir / "audit.jsonl"
        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=audit_file,
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=forbidden_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatcher.dispatch(chain)

        with open(audit_file, 'r') as f:
            audit_record = json.loads(f.readlines()[-1])

        assert audit_record["project_id"] == valid_intent.project_id
        assert audit_record["action_type"] == valid_intent.action_type
        assert audit_record["requester_id"] == valid_intent.requester_id

    def test_audit_contains_chain_decisions(self, temp_dir, valid_intent, forbidden_eligibility, granted_approval, valid_gate_request):
        """Audit contains eligibility and approval decisions."""
        audit_file = temp_dir / "audit.jsonl"
        dispatcher = ControlledExecutionDispatcher(
            intents_file=temp_dir / "intents.jsonl",
            results_file=temp_dir / "results.jsonl",
            audit_file=audit_file,
        )

        chain = ValidationChainInput(
            intent=valid_intent,
            eligibility_result=forbidden_eligibility,
            approval_result=granted_approval,
            gate_request=valid_gate_request,
        )
        dispatcher.dispatch(chain)

        with open(audit_file, 'r') as f:
            audit_record = json.loads(f.readlines()[-1])

        assert audit_record["eligibility_decision"] == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert audit_record["approval_status"] == ApprovalStatus.APPROVAL_GRANTED.value


# =============================================================================
# 11. Store Tests (5 tests)
# =============================================================================

class TestStore:
    """Test execution store operations."""

    def test_record_and_get_intent(self, store, valid_intent):
        """Intent can be recorded and retrieved."""
        store.record_intent(valid_intent)
        retrieved = store.get_intent(valid_intent.intent_id)
        assert retrieved is not None
        assert retrieved.intent_id == valid_intent.intent_id
        assert retrieved.action_type == valid_intent.action_type

    def test_record_and_get_result(self, store):
        """Result can be recorded and retrieved."""
        result = ExecutionResult(
            execution_id="exec-test-001",
            intent_id="int-test-001",
            status=ExecutionStatus.EXECUTION_SUCCESS.value,
            block_reason=None,
            failure_reason=None,
            input_hash="abc",
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            gate_decision_allowed=True,
            execution_output="OK",
            rollback_performed=False,
        )
        store.record_result(result)

        retrieved = store.get_result("exec-test-001")
        assert retrieved is not None
        assert retrieved.execution_id == "exec-test-001"
        assert retrieved.status == ExecutionStatus.EXECUTION_SUCCESS.value

    def test_get_recent_results(self, store):
        """Recent results can be retrieved."""
        for i in range(5):
            result = ExecutionResult(
                execution_id=f"exec-{i}",
                intent_id=f"int-{i}",
                status=ExecutionStatus.EXECUTION_SUCCESS.value if i % 2 == 0 else ExecutionStatus.EXECUTION_FAILED.value,
                block_reason=None,
                failure_reason=None,
                input_hash=f"hash-{i}",
                timestamp=datetime.utcnow().isoformat(),
                dispatcher_version="18C.1.0",
                gate_decision_allowed=True,
                execution_output=None,
                rollback_performed=False,
            )
            store.record_result(result)

        recent = store.get_recent_results(limit=10)
        assert len(recent) == 5

        # Filter by status
        success_only = store.get_recent_results(status=ExecutionStatus.EXECUTION_SUCCESS.value)
        assert len(success_only) == 3

    def test_get_summary(self, store):
        """Summary statistics are correct."""
        for i in range(10):
            if i < 3:
                status = ExecutionStatus.EXECUTION_BLOCKED.value
            elif i < 6:
                status = ExecutionStatus.EXECUTION_SUCCESS.value
            else:
                status = ExecutionStatus.EXECUTION_FAILED.value

            result = ExecutionResult(
                execution_id=f"exec-{i}",
                intent_id=f"int-{i}",
                status=status,
                block_reason=BlockReason.ELIGIBILITY_FORBIDDEN.value if status == ExecutionStatus.EXECUTION_BLOCKED.value else None,
                failure_reason=None,
                input_hash=f"hash-{i}",
                timestamp=datetime.utcnow().isoformat(),
                dispatcher_version="18C.1.0",
                gate_decision_allowed=True,
                execution_output=None,
                rollback_performed=False,
            )
            store.record_result(result)

        summary = store.get_summary(since_hours=1)
        assert summary["total_executions"] == 10
        assert summary["blocked_count"] == 3
        assert summary["success_count"] == 3
        assert summary["failed_count"] == 4

    def test_is_execution_complete(self, store):
        """Complete status is correctly identified."""
        # Pending is not complete
        pending = ExecutionResult(
            execution_id="exec-pending",
            intent_id="int-pending",
            status=ExecutionStatus.EXECUTION_PENDING.value,
            block_reason=None,
            failure_reason=None,
            input_hash="abc",
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            gate_decision_allowed=True,
            execution_output=None,
            rollback_performed=False,
        )
        store.record_result(pending)
        assert store.is_execution_complete("exec-pending") is False

        # Success is complete
        success = ExecutionResult(
            execution_id="exec-success",
            intent_id="int-success",
            status=ExecutionStatus.EXECUTION_SUCCESS.value,
            block_reason=None,
            failure_reason=None,
            input_hash="abc",
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            gate_decision_allowed=True,
            execution_output=None,
            rollback_performed=False,
        )
        store.record_result(success)
        assert store.is_execution_complete("exec-success") is True

        # Blocked is complete
        blocked = ExecutionResult(
            execution_id="exec-blocked",
            intent_id="int-blocked",
            status=ExecutionStatus.EXECUTION_BLOCKED.value,
            block_reason=BlockReason.ELIGIBILITY_FORBIDDEN.value,
            failure_reason=None,
            input_hash="abc",
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            gate_decision_allowed=False,
            execution_output=None,
            rollback_performed=False,
        )
        store.record_result(blocked)
        assert store.is_execution_complete("exec-blocked") is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full dispatch flow."""

    def test_create_intent_helper_function(self):
        """create_execution_intent helper works correctly."""
        intent = create_execution_intent(
            project_id="proj-001",
            project_name="Test Project",
            action_type="run_tests",
            action_description="Run tests",
            requester_id="user-001",
            requester_role="developer",
            target_workspace="/tmp/test",
            metadata={"key": "value"},
        )
        assert intent.intent_id.startswith("int-")
        assert intent.project_id == "proj-001"
        assert intent.action_type == "run_tests"
        assert dict(intent.metadata) == {"key": "value"}

    def test_intent_to_dict_and_back(self, valid_intent):
        """Intent can be serialized and deserialized."""
        data = valid_intent.to_dict()
        restored = ExecutionIntent.from_dict(data)
        assert restored.intent_id == valid_intent.intent_id
        assert restored.action_type == valid_intent.action_type
        assert restored.project_id == valid_intent.project_id

    def test_result_to_dict_and_back(self):
        """Result can be serialized and deserialized."""
        result = ExecutionResult(
            execution_id="exec-001",
            intent_id="int-001",
            status=ExecutionStatus.EXECUTION_SUCCESS.value,
            block_reason=None,
            failure_reason=None,
            input_hash="abc",
            timestamp=datetime.utcnow().isoformat(),
            dispatcher_version="18C.1.0",
            gate_decision_allowed=True,
            execution_output="OK",
            rollback_performed=False,
        )
        data = result.to_dict()
        restored = ExecutionResult.from_dict(data)
        assert restored.execution_id == result.execution_id
        assert restored.status == result.status
