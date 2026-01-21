"""
Phase 18B: Human Approval Orchestration Tests

Comprehensive test suite for the approval orchestration system.
Minimum 30 tests covering all critical behaviors.

Test Categories:
1. Enum Validation Tests (5 tests)
2. Immutability Tests (5 tests)
3. Missing Input Tests (6 tests)
4. Immediate Denial Tests (5 tests)
5. Approval Grant Tests (5 tests)
6. Pending State Tests (4 tests)
7. Determinism Tests (3 tests)
8. Audit Tests (2 tests)
9. Store Tests (6 tests)
10. Integration Tests (2 tests)

Total: 43 tests
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytest

# Import orchestrator components
from controller.approval_orchestrator import (
    ApprovalStatus,
    DenialReason,
    PendingReason,
    ApprovalType,
    ApprovalRequesterInput,
    ApproverInput,
    ApprovalStateInput,
    OrchestrationInput,
    OrchestrationResult,
    ApprovalAuditRecord,
    HumanApprovalOrchestrator,
    get_approval_orchestrator,
    evaluate_approval,
    create_orchestration_input,
)

# Import store components
from controller.approval_store import (
    RequestStatus,
    ApproverAction,
    ApprovalRequestRecord,
    ApproverActionRecord,
    DecisionRecord,
    ApprovalStore,
    get_approval_store,
    create_approval_request,
    record_approval,
    record_denial,
    get_approval_summary,
)

# Import from Phase 18A
from controller.automation_eligibility import (
    EligibilityDecision,
    EligibilityResult,
    RecommendationInput,
    LifecycleStateInput,
    ExecutionGateInput,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_audit_dir():
    """Create a temporary directory for audit files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def orchestrator(temp_audit_dir):
    """Create an orchestrator with temp audit file."""
    audit_file = temp_audit_dir / "approval_audit.jsonl"
    return HumanApprovalOrchestrator(audit_file=audit_file)


@pytest.fixture
def store(temp_audit_dir):
    """Create a store with temp files."""
    return ApprovalStore(
        requests_file=temp_audit_dir / "requests.jsonl",
        decisions_file=temp_audit_dir / "decisions.jsonl",
        actions_file=temp_audit_dir / "actions.jsonl",
    )


@pytest.fixture
def valid_eligibility_result():
    """Create a valid eligibility result (ALLOWED_WITH_APPROVAL)."""
    return EligibilityResult(
        decision=EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value,
        matched_rules=(),
        input_hash="abc123",
        timestamp=datetime.utcnow().isoformat(),
        engine_version="18A.1.0",
        allowed_actions=(),
    )


@pytest.fixture
def forbidden_eligibility_result():
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
def valid_recommendation():
    """Create a valid recommendation input."""
    return RecommendationInput(
        recommendation_id="rec-001",
        recommendation_type="mitigate",
        severity="medium",
        approval_required="explicit_approval_required",
        status="pending",
        project_id="test-project",
        confidence=0.8,
    )


@pytest.fixture
def no_approval_recommendation():
    """Create a recommendation that requires no approval."""
    return RecommendationInput(
        recommendation_id="rec-002",
        recommendation_type="no_action",
        severity="info",
        approval_required="none_required",
        status="pending",
        project_id="test-project",
        confidence=0.9,
    )


@pytest.fixture
def valid_lifecycle_state():
    """Create a valid lifecycle state input."""
    return LifecycleStateInput(
        state="AWAITING_FEEDBACK",
        project_id="test-project",
        is_active=True,
    )


@pytest.fixture
def valid_execution_gate():
    """Create a valid execution gate input (allows action)."""
    return ExecutionGateInput(
        gate_allows_action=True,
        required_role="developer",
        gate_denial_reason=None,
    )


@pytest.fixture
def denied_execution_gate():
    """Create a denied execution gate input."""
    return ExecutionGateInput(
        gate_allows_action=False,
        required_role="admin",
        gate_denial_reason="Insufficient permissions",
    )


@pytest.fixture
def valid_requester():
    """Create a valid requester input."""
    return ApprovalRequesterInput(
        requester_id="user-001",
        requester_role="developer",
        request_timestamp=datetime.utcnow().isoformat(),
        request_reason="Need to deploy fix",
    )


@pytest.fixture
def valid_approval_state():
    """Create a valid approval state with one approver."""
    now = datetime.utcnow()
    return ApprovalStateInput(
        approval_request_id="req-001",
        approvers=(
            ApproverInput(
                approver_id="approver-001",
                approver_role="lead",
                approval_timestamp=now.isoformat(),
                approval_reason="Looks good",
            ),
        ),
        required_approver_count=1,
        approval_type="explicit_approval_required",
        created_at=(now - timedelta(hours=1)).isoformat(),
        expires_at=(now + timedelta(hours=23)).isoformat(),
    )


@pytest.fixture
def expired_approval_state():
    """Create an expired approval state."""
    now = datetime.utcnow()
    return ApprovalStateInput(
        approval_request_id="req-002",
        approvers=(),
        required_approver_count=1,
        approval_type="explicit_approval_required",
        created_at=(now - timedelta(hours=25)).isoformat(),
        expires_at=(now - timedelta(hours=1)).isoformat(),  # Expired 1 hour ago
    )


# =============================================================================
# 1. Enum Validation Tests (5 tests)
# =============================================================================

class TestEnumValidation:
    """Test that all enums have exactly the required values."""

    def test_approval_status_exactly_3_values(self):
        """ApprovalStatus must have EXACTLY 3 values."""
        assert len(ApprovalStatus) == 3
        assert ApprovalStatus.APPROVAL_GRANTED in ApprovalStatus
        assert ApprovalStatus.APPROVAL_DENIED in ApprovalStatus
        assert ApprovalStatus.APPROVAL_PENDING in ApprovalStatus

    def test_denial_reason_values(self):
        """DenialReason must have all expected values."""
        expected = [
            "missing_eligibility",
            "missing_recommendation",
            "missing_lifecycle_state",
            "missing_execution_gate",
            "missing_requester",
            "eligibility_forbidden",
            "approval_expired",
            "approval_revoked",
            "approver_same_as_requester",
            "approver_unauthorized",
            "insufficient_approvers",
            "execution_gate_denied",
            "audit_write_failed",
        ]
        for value in expected:
            assert value in [r.value for r in DenialReason]

    def test_pending_reason_values(self):
        """PendingReason must have all expected values."""
        assert PendingReason.AWAITING_APPROVAL.value == "awaiting_approval"
        assert PendingReason.AWAITING_CONFIRMATION.value == "awaiting_confirmation"
        assert PendingReason.AWAITING_DUAL_APPROVAL.value == "awaiting_dual_approval"

    def test_approval_type_values(self):
        """ApprovalType must have all expected values."""
        assert len(ApprovalType) == 4
        assert ApprovalType.NONE_REQUIRED.value == "none_required"
        assert ApprovalType.CONFIRMATION_REQUIRED.value == "confirmation_required"
        assert ApprovalType.EXPLICIT_APPROVAL_REQUIRED.value == "explicit_approval_required"
        assert ApprovalType.DUAL_APPROVAL_REQUIRED.value == "dual_approval_required"

    def test_store_request_status_values(self):
        """RequestStatus must have all expected values."""
        assert RequestStatus.OPEN.value == "open"
        assert RequestStatus.APPROVED.value == "approved"
        assert RequestStatus.DENIED.value == "denied"
        assert RequestStatus.EXPIRED.value == "expired"
        assert RequestStatus.CANCELLED.value == "cancelled"


# =============================================================================
# 2. Immutability Tests (5 tests)
# =============================================================================

class TestImmutability:
    """Test that all dataclasses are properly immutable."""

    def test_approval_requester_input_frozen(self):
        """ApprovalRequesterInput must be frozen."""
        requester = ApprovalRequesterInput(
            requester_id="user-001",
            requester_role="developer",
            request_timestamp=datetime.utcnow().isoformat(),
            request_reason="Test",
        )
        with pytest.raises(AttributeError):
            requester.requester_id = "changed"

    def test_approver_input_frozen(self):
        """ApproverInput must be frozen."""
        approver = ApproverInput(
            approver_id="approver-001",
            approver_role="lead",
            approval_timestamp=datetime.utcnow().isoformat(),
            approval_reason="OK",
        )
        with pytest.raises(AttributeError):
            approver.approver_id = "changed"

    def test_approval_state_input_frozen(self):
        """ApprovalStateInput must be frozen."""
        state = ApprovalStateInput(
            approval_request_id="req-001",
            approvers=(),
            required_approver_count=1,
            approval_type="explicit_approval_required",
            created_at=datetime.utcnow().isoformat(),
            expires_at=(datetime.utcnow() + timedelta(hours=24)).isoformat(),
        )
        with pytest.raises(AttributeError):
            state.required_approver_count = 5

    def test_orchestration_result_frozen(self):
        """OrchestrationResult must be frozen."""
        result = OrchestrationResult(
            status=ApprovalStatus.APPROVAL_PENDING.value,
            reason=PendingReason.AWAITING_APPROVAL.value,
            input_hash="abc123",
            timestamp=datetime.utcnow().isoformat(),
            orchestrator_version="18B.1.0",
            approval_request_id="req-001",
            approver_count=0,
            required_approver_count=1,
        )
        with pytest.raises(AttributeError):
            result.status = "changed"

    def test_approval_audit_record_frozen(self):
        """ApprovalAuditRecord must be frozen."""
        record = ApprovalAuditRecord(
            audit_id="aud-001",
            input_hash="abc123",
            status=ApprovalStatus.APPROVAL_DENIED.value,
            reason=DenialReason.ELIGIBILITY_FORBIDDEN.value,
            timestamp=datetime.utcnow().isoformat(),
            orchestrator_version="18B.1.0",
            approval_request_id="req-001",
            requester_id="user-001",
            project_id="test-project",
            recommendation_id="rec-001",
        )
        with pytest.raises(AttributeError):
            record.status = "changed"


# =============================================================================
# 3. Missing Input Tests (6 tests)
# =============================================================================

class TestMissingInputs:
    """Test that missing inputs result in APPROVAL_DENIED."""

    def test_missing_eligibility_result(
        self,
        orchestrator,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Missing eligibility result must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=None,  # Missing
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.MISSING_ELIGIBILITY.value

    def test_missing_recommendation(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Missing recommendation must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=None,  # Missing
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.MISSING_RECOMMENDATION.value

    def test_missing_lifecycle_state(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_execution_gate,
        valid_requester,
    ):
        """Missing lifecycle state must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=None,  # Missing
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.MISSING_LIFECYCLE_STATE.value

    def test_missing_execution_gate(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_requester,
    ):
        """Missing execution gate must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=None,  # Missing
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.MISSING_EXECUTION_GATE.value

    def test_missing_requester(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
    ):
        """Missing requester must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=None,  # Missing
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.MISSING_REQUESTER.value

    def test_missing_approval_state_is_pending(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Missing approval state (for fresh requests) results in PENDING, not DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,  # Fresh request
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_PENDING.value


# =============================================================================
# 4. Immediate Denial Tests (5 tests)
# =============================================================================

class TestImmediateDenial:
    """Test immediate denial conditions."""

    def test_eligibility_forbidden_causes_denial(
        self,
        orchestrator,
        forbidden_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """AUTOMATION_FORBIDDEN eligibility must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=forbidden_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.ELIGIBILITY_FORBIDDEN.value

    def test_gate_denied_causes_denial(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        denied_execution_gate,
        valid_requester,
    ):
        """Execution gate denial must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=denied_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.EXECUTION_GATE_DENIED.value

    def test_expired_approval_causes_denial(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
        expired_approval_state,
    ):
        """Expired approval must result in DENIED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=expired_approval_state,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.APPROVAL_EXPIRED.value

    def test_self_approval_causes_denial(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
    ):
        """Self-approval (approver same as requester) must result in DENIED."""
        now = datetime.utcnow()
        requester = ApprovalRequesterInput(
            requester_id="user-001",  # Same ID
            requester_role="developer",
            request_timestamp=now.isoformat(),
            request_reason="Need to deploy",
        )
        approval_state = ApprovalStateInput(
            approval_request_id="req-001",
            approvers=(
                ApproverInput(
                    approver_id="user-001",  # Same ID as requester
                    approver_role="lead",
                    approval_timestamp=now.isoformat(),
                    approval_reason="LGTM",
                ),
            ),
            required_approver_count=1,
            approval_type="explicit_approval_required",
            created_at=(now - timedelta(hours=1)).isoformat(),
            expires_at=(now + timedelta(hours=23)).isoformat(),
        )
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=requester,
            approval_state=approval_state,
            current_timestamp=now.isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value
        assert result.reason == DenialReason.APPROVER_SAME_AS_REQUESTER.value

    def test_denial_priority_over_pending(
        self,
        orchestrator,
        forbidden_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Denial conditions must take priority over pending state."""
        input_data = OrchestrationInput(
            eligibility_result=forbidden_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,  # Would normally be PENDING
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        # Should be DENIED, not PENDING
        assert result.status == ApprovalStatus.APPROVAL_DENIED.value


# =============================================================================
# 5. Approval Grant Tests (5 tests)
# =============================================================================

class TestApprovalGrant:
    """Test approval grant conditions."""

    def test_no_approval_required_grants_immediately(
        self,
        orchestrator,
        valid_eligibility_result,
        no_approval_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Recommendation with no approval required should be GRANTED immediately."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=no_approval_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_GRANTED.value
        assert result.reason is None

    def test_sufficient_approvers_grants(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
        valid_approval_state,
    ):
        """Meeting required approver count should result in GRANTED."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=valid_approval_state,  # Has 1 approver, needs 1
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_GRANTED.value

    def test_confirmation_with_one_approver_grants(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Confirmation required with one approver should be GRANTED."""
        now = datetime.utcnow()
        recommendation = RecommendationInput(
            recommendation_id="rec-003",
            recommendation_type="mitigate",
            severity="low",
            approval_required="confirmation_required",
            status="pending",
            project_id="test-project",
            confidence=0.8,
        )
        approval_state = ApprovalStateInput(
            approval_request_id="req-001",
            approvers=(
                ApproverInput(
                    approver_id="approver-001",
                    approver_role="lead",
                    approval_timestamp=now.isoformat(),
                    approval_reason="Confirmed",
                ),
            ),
            required_approver_count=1,
            approval_type="confirmation_required",
            created_at=(now - timedelta(hours=1)).isoformat(),
            expires_at=(now + timedelta(hours=23)).isoformat(),
        )
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=approval_state,
            current_timestamp=now.isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_GRANTED.value

    def test_dual_approval_with_two_approvers_grants(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Dual approval with two approvers should be GRANTED."""
        now = datetime.utcnow()
        approval_state = ApprovalStateInput(
            approval_request_id="req-001",
            approvers=(
                ApproverInput(
                    approver_id="approver-001",
                    approver_role="lead",
                    approval_timestamp=now.isoformat(),
                    approval_reason="LGTM",
                ),
                ApproverInput(
                    approver_id="approver-002",
                    approver_role="senior",
                    approval_timestamp=now.isoformat(),
                    approval_reason="Approved",
                ),
            ),
            required_approver_count=2,
            approval_type="dual_approval_required",
            created_at=(now - timedelta(hours=1)).isoformat(),
            expires_at=(now + timedelta(hours=23)).isoformat(),
        )
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=approval_state,
            current_timestamp=now.isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_GRANTED.value

    def test_grant_includes_approver_count(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
        valid_approval_state,
    ):
        """Granted result should include approver count."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=valid_approval_state,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_GRANTED.value
        assert result.approver_count == 1
        assert result.required_approver_count == 1


# =============================================================================
# 6. Pending State Tests (4 tests)
# =============================================================================

class TestPendingState:
    """Test pending state conditions."""

    def test_no_approvers_is_pending(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """No approvers yet should result in PENDING."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_PENDING.value

    def test_pending_reason_awaiting_approval(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Pending for explicit approval should have correct reason."""
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.reason == PendingReason.AWAITING_APPROVAL.value

    def test_pending_reason_awaiting_dual(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Pending for dual approval should have correct reason."""
        now = datetime.utcnow()
        approval_state = ApprovalStateInput(
            approval_request_id="req-001",
            approvers=(
                ApproverInput(
                    approver_id="approver-001",
                    approver_role="lead",
                    approval_timestamp=now.isoformat(),
                    approval_reason="LGTM",
                ),
            ),
            required_approver_count=2,  # Need 2, have 1
            approval_type="dual_approval_required",
            created_at=(now - timedelta(hours=1)).isoformat(),
            expires_at=(now + timedelta(hours=23)).isoformat(),
        )
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=approval_state,
            current_timestamp=now.isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_PENDING.value
        assert result.reason == PendingReason.AWAITING_DUAL_APPROVAL.value

    def test_pending_reason_awaiting_confirmation(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Pending for confirmation should have correct reason."""
        recommendation = RecommendationInput(
            recommendation_id="rec-003",
            recommendation_type="mitigate",
            severity="low",
            approval_required="confirmation_required",
            status="pending",
            project_id="test-project",
            confidence=0.8,
        )
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result = orchestrator.evaluate(input_data)
        assert result.status == ApprovalStatus.APPROVAL_PENDING.value
        assert result.reason == PendingReason.AWAITING_CONFIRMATION.value


# =============================================================================
# 7. Determinism Tests (3 tests)
# =============================================================================

class TestDeterminism:
    """Test deterministic behavior."""

    def test_same_input_same_output(
        self,
        orchestrator,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Same inputs must always produce same output."""
        fixed_timestamp = "2024-01-15T10:00:00"
        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=fixed_timestamp,
        )
        result1 = orchestrator.evaluate(input_data)
        result2 = orchestrator.evaluate(input_data)
        assert result1.status == result2.status
        assert result1.reason == result2.reason
        assert result1.input_hash == result2.input_hash

    def test_input_hash_deterministic(
        self,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Input hash must be deterministic."""
        fixed_timestamp = "2024-01-15T10:00:00"
        input1 = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=fixed_timestamp,
        )
        input2 = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=fixed_timestamp,
        )
        assert input1.compute_hash() == input2.compute_hash()

    def test_different_input_different_hash(
        self,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Different inputs must produce different hashes."""
        input1 = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp="2024-01-15T10:00:00",
        )
        input2 = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp="2024-01-15T11:00:00",  # Different
        )
        assert input1.compute_hash() != input2.compute_hash()


# =============================================================================
# 8. Audit Tests (2 tests)
# =============================================================================

class TestAudit:
    """Test audit record writing."""

    def test_audit_record_written(
        self,
        temp_audit_dir,
        valid_eligibility_result,
        valid_recommendation,
        valid_lifecycle_state,
        valid_execution_gate,
        valid_requester,
    ):
        """Audit record must be written for every evaluation."""
        audit_file = temp_audit_dir / "approval_audit.jsonl"
        orchestrator = HumanApprovalOrchestrator(audit_file=audit_file)

        input_data = OrchestrationInput(
            eligibility_result=valid_eligibility_result,
            recommendation=valid_recommendation,
            lifecycle_state=valid_lifecycle_state,
            execution_gate=valid_execution_gate,
            requester=valid_requester,
            approval_state=None,
            current_timestamp=datetime.utcnow().isoformat(),
        )
        orchestrator.evaluate(input_data)

        assert audit_file.exists()
        with open(audit_file) as f:
            lines = f.readlines()
            assert len(lines) == 1
            record = json.loads(lines[0])
            assert "audit_id" in record
            assert "status" in record
            assert "timestamp" in record

    def test_audit_failure_causes_denial(self, temp_audit_dir):
        """Audit write failure must result in DENIAL."""
        # Create read-only directory to force failure
        audit_file = temp_audit_dir / "readonly" / "audit.jsonl"
        audit_file.parent.mkdir()
        audit_file.parent.chmod(0o444)

        try:
            orchestrator = HumanApprovalOrchestrator(audit_file=audit_file)
            # Create minimal valid input that would normally succeed
            input_data = create_orchestration_input(
                eligibility_result={
                    "decision": "automation_allowed_with_approval",
                    "matched_rules": [],
                    "input_hash": "abc",
                    "timestamp": "2024-01-01T00:00:00",
                    "engine_version": "18A.1.0",
                    "allowed_actions": [],
                },
                recommendation={
                    "recommendation_id": "rec-001",
                    "recommendation_type": "mitigate",
                    "severity": "medium",
                    "approval_required": "none_required",  # Would normally grant
                    "status": "pending",
                    "confidence": 0.8,
                },
                lifecycle_state={
                    "state": "AWAITING_FEEDBACK",
                    "project_id": "test",
                    "is_active": True,
                },
                execution_gate={
                    "gate_allows_action": True,
                },
                requester={
                    "requester_id": "user-001",
                    "requester_role": "developer",
                    "request_timestamp": "2024-01-01T00:00:00",
                },
                current_timestamp="2024-01-01T00:00:00",
            )
            result = orchestrator.evaluate(input_data)
            assert result.status == ApprovalStatus.APPROVAL_DENIED.value
            assert result.reason == DenialReason.AUDIT_WRITE_FAILED.value
        finally:
            audit_file.parent.chmod(0o755)


# =============================================================================
# 9. Store Tests (6 tests)
# =============================================================================

class TestApprovalStore:
    """Test approval store operations."""

    def test_create_request(self, store):
        """Creating a request should return a valid record."""
        record = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=1,
            project_id="test-project",
            request_reason="Need to deploy fix",
        )
        assert record.request_id.startswith("req-")
        assert record.requester_id == "user-001"
        assert record.recommendation_id == "rec-001"

    def test_get_request(self, store):
        """Getting a request by ID should return the correct record."""
        created = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=1,
        )
        fetched = store.get_request(created.request_id)
        assert fetched is not None
        assert fetched.request_id == created.request_id

    def test_record_approver_action(self, store):
        """Recording an approver action should create a record."""
        request = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=1,
        )
        action = store.record_approver_action(
            request_id=request.request_id,
            approver_id="approver-001",
            approver_role="lead",
            action=ApproverAction.APPROVE.value,
            reason="LGTM",
        )
        assert action.action_id.startswith("act-")
        assert action.action == ApproverAction.APPROVE.value

    def test_get_approval_count(self, store):
        """Getting approval count should return correct number."""
        request = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=2,
        )
        # Add two approvals
        store.record_approver_action(
            request_id=request.request_id,
            approver_id="approver-001",
            approver_role="lead",
            action=ApproverAction.APPROVE.value,
        )
        store.record_approver_action(
            request_id=request.request_id,
            approver_id="approver-002",
            approver_role="senior",
            action=ApproverAction.APPROVE.value,
        )
        count = store.get_approval_count(request.request_id)
        assert count == 2

    def test_record_decision(self, store):
        """Recording a decision should create a record."""
        request = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=1,
        )
        decision = store.record_decision(
            request_id=request.request_id,
            status=RequestStatus.APPROVED.value,
            approver_count=1,
            required_count=1,
            decided_by="approver-001",
        )
        assert decision.decision_id.startswith("dec-")
        assert decision.status == RequestStatus.APPROVED.value

    def test_is_request_decided(self, store):
        """Checking if request is decided should return correct status."""
        request = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=1,
        )
        assert store.is_request_decided(request.request_id) is False

        store.record_decision(
            request_id=request.request_id,
            status=RequestStatus.APPROVED.value,
            approver_count=1,
            required_count=1,
        )
        assert store.is_request_decided(request.request_id) is True


# =============================================================================
# 10. Integration Tests (2 tests)
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_create_orchestration_input_from_dicts(self):
        """Creating orchestration input from dictionaries should work."""
        input_data = create_orchestration_input(
            eligibility_result={
                "decision": "automation_allowed_with_approval",
                "matched_rules": [],
                "input_hash": "abc123",
                "timestamp": "2024-01-01T00:00:00",
                "engine_version": "18A.1.0",
                "allowed_actions": [],
            },
            recommendation={
                "recommendation_id": "rec-001",
                "recommendation_type": "mitigate",
                "severity": "medium",
                "approval_required": "explicit_approval_required",
                "status": "pending",
                "project_id": "test-project",
                "confidence": 0.8,
            },
            lifecycle_state={
                "state": "AWAITING_FEEDBACK",
                "project_id": "test-project",
                "is_active": True,
            },
            execution_gate={
                "gate_allows_action": True,
                "required_role": "developer",
            },
            requester={
                "requester_id": "user-001",
                "requester_role": "developer",
                "request_timestamp": "2024-01-01T00:00:00",
            },
            current_timestamp="2024-01-01T00:00:00",
        )
        assert input_data.eligibility_result is not None
        assert input_data.recommendation is not None
        assert input_data.lifecycle_state is not None
        assert input_data.execution_gate is not None
        assert input_data.requester is not None

    def test_full_approval_workflow(self, temp_audit_dir):
        """Test a full approval workflow from request to grant."""
        # Create store
        store = ApprovalStore(
            requests_file=temp_audit_dir / "requests.jsonl",
            decisions_file=temp_audit_dir / "decisions.jsonl",
            actions_file=temp_audit_dir / "actions.jsonl",
        )

        # Create orchestrator
        orchestrator = HumanApprovalOrchestrator(
            audit_file=temp_audit_dir / "audit.jsonl"
        )

        # Step 1: Create request
        request = store.create_request(
            requester_id="user-001",
            requester_role="developer",
            recommendation_id="rec-001",
            approval_type="explicit_approval_required",
            required_approver_count=1,
            project_id="test-project",
        )

        # Step 2: Check status (should be PENDING)
        input_data = create_orchestration_input(
            eligibility_result={
                "decision": "automation_allowed_with_approval",
                "matched_rules": [],
                "input_hash": "abc",
                "timestamp": datetime.utcnow().isoformat(),
                "engine_version": "18A.1.0",
                "allowed_actions": [],
            },
            recommendation={
                "recommendation_id": "rec-001",
                "recommendation_type": "mitigate",
                "severity": "medium",
                "approval_required": "explicit_approval_required",
                "status": "pending",
                "project_id": "test-project",
                "confidence": 0.8,
            },
            lifecycle_state={
                "state": "AWAITING_FEEDBACK",
                "project_id": "test-project",
                "is_active": True,
            },
            execution_gate={
                "gate_allows_action": True,
            },
            requester={
                "requester_id": "user-001",
                "requester_role": "developer",
                "request_timestamp": request.created_at,
            },
            approval_state=None,  # No approvers yet
            current_timestamp=datetime.utcnow().isoformat(),
        )
        result1 = orchestrator.evaluate(input_data)
        assert result1.status == ApprovalStatus.APPROVAL_PENDING.value

        # Step 3: Add approver
        store.record_approver_action(
            request_id=request.request_id,
            approver_id="approver-001",
            approver_role="lead",
            action=ApproverAction.APPROVE.value,
            reason="LGTM",
        )

        # Step 4: Check status again (should be GRANTED)
        now = datetime.utcnow()
        input_data2 = create_orchestration_input(
            eligibility_result={
                "decision": "automation_allowed_with_approval",
                "matched_rules": [],
                "input_hash": "abc",
                "timestamp": now.isoformat(),
                "engine_version": "18A.1.0",
                "allowed_actions": [],
            },
            recommendation={
                "recommendation_id": "rec-001",
                "recommendation_type": "mitigate",
                "severity": "medium",
                "approval_required": "explicit_approval_required",
                "status": "pending",
                "project_id": "test-project",
                "confidence": 0.8,
            },
            lifecycle_state={
                "state": "AWAITING_FEEDBACK",
                "project_id": "test-project",
                "is_active": True,
            },
            execution_gate={
                "gate_allows_action": True,
            },
            requester={
                "requester_id": "user-001",
                "requester_role": "developer",
                "request_timestamp": request.created_at,
            },
            approval_state={
                "approval_request_id": request.request_id,
                "approvers": [
                    {
                        "approver_id": "approver-001",
                        "approver_role": "lead",
                        "approval_timestamp": now.isoformat(),
                        "approval_reason": "LGTM",
                    },
                ],
                "required_approver_count": 1,
                "approval_type": "explicit_approval_required",
                "created_at": request.created_at,
                "expires_at": request.expires_at,
            },
            current_timestamp=now.isoformat(),
        )
        result2 = orchestrator.evaluate(input_data2)
        assert result2.status == ApprovalStatus.APPROVAL_GRANTED.value
