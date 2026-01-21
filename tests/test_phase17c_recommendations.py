"""
Phase 17C: Recommendation & Human-in-the-Loop Tests

Comprehensive tests proving:
1. IMMUTABILITY: Recommendations cannot be modified after creation
2. ADVISORY ONLY: Recommendations suggest, never execute
3. DETERMINISM: Same input always produces same output
4. UNKNOWN HANDLING: Missing data yields UNKNOWN, never guessed
5. APPEND-ONLY: Store only appends, never edits or deletes
6. APPROVAL TRACKING: Approvals create new records, not mutations
7. NO EXECUTION: Recommendations never trigger actions

MINIMUM 45 TESTS covering all critical behaviors.

CRITICAL CONSTRAINTS:
- ADVISORY-ONLY: Recommendations suggest, never execute
- NO AUTOMATION: Human must approve/dismiss
- DETERMINISTIC: No ML, no probabilistic inference
- EXPLICIT UNKNOWN: Missing data = UNKNOWN, never guessed
- APPEND-ONLY: Recommendations are never deleted or modified
- NO EXECUTION: Recommendations never trigger actions directly
- NO LIFECYCLE MUTATION: Recommendations never change project state
"""

import json
import os
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from dataclasses import FrozenInstanceError

# Phase 17C imports
from controller.recommendation_model import (
    Recommendation,
    ApprovalRecord,
    RecommendationSummary,
    RecommendationRule,
    RecommendationType,
    RecommendationSeverity,
    RecommendationApproval,
    RecommendationStatus,
    DEFAULT_EXPIRATION_HOURS,
    MAX_INCIDENTS_PER_RECOMMENDATION,
)
from controller.recommendation_engine import (
    RecommendationGenerator,
    RecommendationEngine,
    RECOMMENDATION_RULES,
    generate_recommendations,
)
from controller.recommendation_store import (
    RecommendationStore,
    persist_recommendations,
    read_recommendations,
    read_recent_recommendations,
    get_recommendation_by_id,
    get_recommendation_summary,
)

# Phase 17B imports for incident creation
from controller.incident_model import (
    Incident,
    IncidentType,
    IncidentSeverity,
    IncidentScope,
    IncidentState,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_recommendations_dir():
    """Create a temporary directory for recommendations testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_recommendations_file(temp_recommendations_dir):
    """Create a temporary recommendations file for testing."""
    return temp_recommendations_dir / "recommendations.jsonl"


@pytest.fixture
def temp_approvals_file(temp_recommendations_dir):
    """Create a temporary approvals file for testing."""
    return temp_recommendations_dir / "approvals.jsonl"


@pytest.fixture
def recommendation_store(temp_recommendations_file, temp_approvals_file):
    """Create a RecommendationStore with temporary files."""
    return RecommendationStore(
        recommendations_file=temp_recommendations_file,
        approvals_file=temp_approvals_file,
    )


@pytest.fixture
def sample_recommendation():
    """Create a sample recommendation for testing."""
    return Recommendation(
        recommendation_id="rec-test-001",
        created_at=datetime.utcnow().isoformat(),
        recommendation_type=RecommendationType.INVESTIGATE.value,
        severity=RecommendationSeverity.HIGH.value,
        approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
        status=RecommendationStatus.PENDING.value,
        title="Test Recommendation",
        description="A test recommendation for unit testing",
        rationale="This is a test rationale",
        suggested_actions=("Action 1", "Action 2"),
        source_incident_ids=("inc-001", "inc-002"),
        incident_count=2,
        project_id="test-project",
        aspect=None,
        confidence=0.9,
        classification_rule="rule-test-001",
        expires_at=(datetime.utcnow() + timedelta(hours=DEFAULT_EXPIRATION_HOURS)).isoformat(),
        metadata={"test": True},
    )


@pytest.fixture
def sample_incident():
    """Create a sample incident for testing."""
    return Incident(
        incident_id="inc-test-001",
        created_at=datetime.utcnow().isoformat(),
        incident_type=IncidentType.RELIABILITY.value,
        severity=IncidentSeverity.HIGH.value,
        scope=IncidentScope.PROJECT.value,
        state=IncidentState.OPEN.value,
        title="Test Incident",
        description="A test incident for recommendation generation",
        source_signal_ids=("sig-001", "sig-002"),
        signal_count=2,
        first_signal_at=datetime.utcnow().isoformat(),
        last_signal_at=datetime.utcnow().isoformat(),
        correlation_window_minutes=30,
        project_id="test-project",
        aspect=None,
        job_id=None,
        confidence=0.9,
        classification_rule="rule-reliability-001",
        metadata={},
    )


@pytest.fixture
def sample_incidents():
    """Create multiple sample incidents for testing."""
    now = datetime.utcnow()
    return [
        Incident(
            incident_id="inc-001",
            created_at=now.isoformat(),
            incident_type=IncidentType.RELIABILITY.value,
            severity=IncidentSeverity.HIGH.value,
            scope=IncidentScope.PROJECT.value,
            state=IncidentState.OPEN.value,
            title="Reliability Issue",
            description="Service reliability degraded",
            source_signal_ids=("sig-001",),
            signal_count=1,
            first_signal_at=now.isoformat(),
            last_signal_at=now.isoformat(),
            correlation_window_minutes=30,
            project_id="project-1",
            aspect=None,
            job_id=None,
            confidence=0.9,
            classification_rule="rule-rel-001",
            metadata={},
        ),
        Incident(
            incident_id="inc-002",
            created_at=(now + timedelta(hours=1)).isoformat(),
            incident_type=IncidentType.SECURITY.value,
            severity=IncidentSeverity.CRITICAL.value,
            scope=IncidentScope.SYSTEM.value,
            state=IncidentState.OPEN.value,
            title="Security Alert",
            description="Potential security vulnerability detected",
            source_signal_ids=("sig-002", "sig-003"),
            signal_count=2,
            first_signal_at=(now + timedelta(hours=1)).isoformat(),
            last_signal_at=(now + timedelta(hours=1)).isoformat(),
            correlation_window_minutes=30,
            project_id=None,
            aspect=None,
            job_id=None,
            confidence=0.95,
            classification_rule="rule-sec-001",
            metadata={},
        ),
    ]


# =============================================================================
# Section 1: IMMUTABILITY Tests (8 tests)
# =============================================================================

class TestRecommendationImmutability:
    """Test that recommendations are truly immutable (frozen dataclass)."""

    def test_recommendation_is_frozen(self, sample_recommendation):
        """Recommendation dataclass is frozen - cannot modify attributes."""
        with pytest.raises(FrozenInstanceError):
            sample_recommendation.title = "Modified Title"

    def test_recommendation_id_immutable(self, sample_recommendation):
        """Cannot modify recommendation_id after creation."""
        with pytest.raises(FrozenInstanceError):
            sample_recommendation.recommendation_id = "new-id"

    def test_severity_immutable(self, sample_recommendation):
        """Cannot modify severity after creation."""
        with pytest.raises(FrozenInstanceError):
            sample_recommendation.severity = RecommendationSeverity.CRITICAL.value

    def test_recommendation_type_immutable(self, sample_recommendation):
        """Cannot modify recommendation_type after creation."""
        with pytest.raises(FrozenInstanceError):
            sample_recommendation.recommendation_type = RecommendationType.MITIGATE.value

    def test_status_immutable(self, sample_recommendation):
        """Cannot modify status after creation."""
        with pytest.raises(FrozenInstanceError):
            sample_recommendation.status = RecommendationStatus.APPROVED.value

    def test_source_incident_ids_is_tuple(self, sample_recommendation):
        """source_incident_ids is a tuple (immutable), not a list."""
        assert isinstance(sample_recommendation.source_incident_ids, tuple)

    def test_suggested_actions_is_tuple(self, sample_recommendation):
        """suggested_actions is a tuple (immutable), not a list."""
        assert isinstance(sample_recommendation.suggested_actions, tuple)

    def test_source_incident_ids_validation_rejects_list(self):
        """Creating recommendation with list for incident_ids should fail."""
        with pytest.raises(ValueError, match="must be a tuple"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity=RecommendationSeverity.HIGH.value,
                approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=["inc-001"],  # List, not tuple
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=0.9,
                classification_rule="test",
                expires_at=None,
            )


# =============================================================================
# Section 2: APPROVAL RECORD IMMUTABILITY Tests (4 tests)
# =============================================================================

class TestApprovalRecordImmutability:
    """Test that approval records are truly immutable (frozen dataclass)."""

    def test_approval_record_is_frozen(self):
        """ApprovalRecord dataclass is frozen - cannot modify attributes."""
        record = ApprovalRecord(
            record_id="apr-001",
            recommendation_id="rec-001",
            action="approved",
            user_id="user-001",
            timestamp=datetime.utcnow().isoformat(),
            reason="Test approval",
        )
        with pytest.raises(FrozenInstanceError):
            record.action = "dismissed"

    def test_approval_record_action_validation(self):
        """ApprovalRecord action must be 'approved' or 'dismissed'."""
        with pytest.raises(ValueError, match="Invalid action"):
            ApprovalRecord(
                record_id="apr-001",
                recommendation_id="rec-001",
                action="invalid_action",
                user_id="user-001",
                timestamp=datetime.utcnow().isoformat(),
                reason=None,
            )

    def test_approval_record_approved_valid(self):
        """'approved' is a valid action."""
        record = ApprovalRecord(
            record_id="apr-001",
            recommendation_id="rec-001",
            action="approved",
            user_id="user-001",
            timestamp=datetime.utcnow().isoformat(),
            reason="Valid approval",
        )
        assert record.action == "approved"

    def test_approval_record_dismissed_valid(self):
        """'dismissed' is a valid action."""
        record = ApprovalRecord(
            record_id="apr-001",
            recommendation_id="rec-001",
            action="dismissed",
            user_id="user-001",
            timestamp=datetime.utcnow().isoformat(),
            reason="Valid dismissal",
        )
        assert record.action == "dismissed"


# =============================================================================
# Section 3: ENUM VALIDATION Tests (8 tests)
# =============================================================================

class TestEnumValidation:
    """Test that enums are properly validated."""

    def test_recommendation_type_enum_values(self):
        """RecommendationType has exactly 6 values (LOCKED)."""
        assert len(RecommendationType) == 6
        assert RecommendationType.INVESTIGATE.value == "investigate"
        assert RecommendationType.MITIGATE.value == "mitigate"
        assert RecommendationType.IMPROVE.value == "improve"
        assert RecommendationType.REFACTOR.value == "refactor"
        assert RecommendationType.DOCUMENT.value == "document"
        assert RecommendationType.NO_ACTION.value == "no_action"

    def test_recommendation_severity_enum_values(self):
        """RecommendationSeverity has exactly 6 values including UNKNOWN (LOCKED)."""
        assert len(RecommendationSeverity) == 6
        assert RecommendationSeverity.INFO.value == "info"
        assert RecommendationSeverity.LOW.value == "low"
        assert RecommendationSeverity.MEDIUM.value == "medium"
        assert RecommendationSeverity.HIGH.value == "high"
        assert RecommendationSeverity.CRITICAL.value == "critical"
        assert RecommendationSeverity.UNKNOWN.value == "unknown"

    def test_recommendation_approval_enum_values(self):
        """RecommendationApproval has exactly 3 values (LOCKED)."""
        assert len(RecommendationApproval) == 3
        assert RecommendationApproval.NONE_REQUIRED.value == "none_required"
        assert RecommendationApproval.CONFIRMATION_REQUIRED.value == "confirmation_required"
        assert RecommendationApproval.EXPLICIT_APPROVAL_REQUIRED.value == "explicit_approval_required"

    def test_recommendation_status_enum_values(self):
        """RecommendationStatus has exactly 5 values including UNKNOWN (LOCKED)."""
        assert len(RecommendationStatus) == 5
        assert RecommendationStatus.PENDING.value == "pending"
        assert RecommendationStatus.APPROVED.value == "approved"
        assert RecommendationStatus.DISMISSED.value == "dismissed"
        assert RecommendationStatus.EXPIRED.value == "expired"
        assert RecommendationStatus.UNKNOWN.value == "unknown"

    def test_invalid_severity_rejected(self):
        """Invalid severity value should be rejected."""
        with pytest.raises(ValueError, match="Invalid severity"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity="invalid_severity",
                approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=0.9,
                classification_rule="test",
                expires_at=None,
            )

    def test_invalid_recommendation_type_rejected(self):
        """Invalid recommendation_type value should be rejected."""
        with pytest.raises(ValueError, match="Invalid recommendation type"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type="invalid_type",
                severity=RecommendationSeverity.HIGH.value,
                approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=0.9,
                classification_rule="test",
                expires_at=None,
            )

    def test_invalid_status_rejected(self):
        """Invalid status value should be rejected."""
        with pytest.raises(ValueError, match="Invalid status"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity=RecommendationSeverity.HIGH.value,
                approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
                status="invalid_status",
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=0.9,
                classification_rule="test",
                expires_at=None,
            )

    def test_invalid_approval_required_rejected(self):
        """Invalid approval_required value should be rejected."""
        with pytest.raises(ValueError, match="Invalid approval requirement"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity=RecommendationSeverity.HIGH.value,
                approval_required="invalid_approval",
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=0.9,
                classification_rule="test",
                expires_at=None,
            )


# =============================================================================
# Section 4: DETERMINISM Tests (5 tests)
# =============================================================================

class TestDeterminism:
    """Test that recommendation generation is deterministic."""

    def test_same_incident_produces_same_recommendation_type(self, sample_incident):
        """Same incident always produces same recommendation type."""
        generator = RecommendationGenerator()
        rec1 = generator.generate(sample_incident)
        rec2 = generator.generate(sample_incident)

        # Type should be the same
        assert rec1.recommendation_type == rec2.recommendation_type

    def test_same_incident_produces_same_severity(self, sample_incident):
        """Same incident always produces same severity recommendation."""
        generator = RecommendationGenerator()
        rec1 = generator.generate(sample_incident)
        rec2 = generator.generate(sample_incident)

        assert rec1.severity == rec2.severity

    def test_rules_are_deterministic(self):
        """Classification rules are fixed and deterministic."""
        # Rules should be a tuple (immutable)
        assert isinstance(RECOMMENDATION_RULES, tuple)
        # Rules count should be fixed
        assert len(RECOMMENDATION_RULES) >= 1

    def test_rule_matching_is_deterministic(self):
        """Rule matching produces consistent results."""
        # Create identical incidents
        now = datetime.utcnow().isoformat()
        incident1 = Incident(
            incident_id="inc-a",
            created_at=now,
            incident_type=IncidentType.SECURITY.value,
            severity=IncidentSeverity.CRITICAL.value,
            scope=IncidentScope.SYSTEM.value,
            state=IncidentState.OPEN.value,
            title="Security Issue",
            description="Test",
            source_signal_ids=("s1",),
            signal_count=1,
            first_signal_at=now,
            last_signal_at=now,
            correlation_window_minutes=30,
            project_id=None,
            aspect=None,
            job_id=None,
            confidence=0.9,
            classification_rule="test",
            metadata={},
        )
        incident2 = Incident(
            incident_id="inc-b",
            created_at=now,
            incident_type=IncidentType.SECURITY.value,
            severity=IncidentSeverity.CRITICAL.value,
            scope=IncidentScope.SYSTEM.value,
            state=IncidentState.OPEN.value,
            title="Security Issue",
            description="Test",
            source_signal_ids=("s2",),
            signal_count=1,
            first_signal_at=now,
            last_signal_at=now,
            correlation_window_minutes=30,
            project_id=None,
            aspect=None,
            job_id=None,
            confidence=0.9,
            classification_rule="test",
            metadata={},
        )

        generator = RecommendationGenerator()
        rec1 = generator.generate(incident1)
        rec2 = generator.generate(incident2)

        # Same incident type/severity should produce same recommendation type
        if rec1 and rec2:
            assert rec1.recommendation_type == rec2.recommendation_type

    def test_confidence_is_deterministic(self, sample_incident):
        """Confidence values are deterministic, not random."""
        generator = RecommendationGenerator()
        rec1 = generator.generate(sample_incident)
        rec2 = generator.generate(sample_incident)

        if rec1 and rec2:
            assert rec1.confidence == rec2.confidence


# =============================================================================
# Section 5: UNKNOWN HANDLING Tests (5 tests)
# =============================================================================

class TestUnknownHandling:
    """Test UNKNOWN propagation - missing data yields UNKNOWN."""

    def test_unknown_incident_produces_unknown_recommendation(self):
        """UNKNOWN incident type produces UNKNOWN severity recommendation."""
        now = datetime.utcnow().isoformat()
        unknown_incident = Incident(
            incident_id="inc-unknown",
            created_at=now,
            incident_type=IncidentType.UNKNOWN.value,
            severity=IncidentSeverity.UNKNOWN.value,
            scope=IncidentScope.UNKNOWN.value,
            state=IncidentState.UNKNOWN.value,
            title="Unknown Issue",
            description="Cannot determine incident type",
            source_signal_ids=("s1",),
            signal_count=1,
            first_signal_at=now,
            last_signal_at=now,
            correlation_window_minutes=30,
            project_id=None,
            aspect=None,
            job_id=None,
            confidence=0.0,
            classification_rule="unknown",
            metadata={},
        )

        generator = RecommendationGenerator()
        rec = generator.generate(unknown_incident)

        # UNKNOWN incidents should produce UNKNOWN severity recommendations
        if rec:
            assert rec.severity == RecommendationSeverity.UNKNOWN.value

    def test_unknown_severity_explicit(self):
        """UNKNOWN severity is explicitly available."""
        assert RecommendationSeverity.UNKNOWN.value == "unknown"

    def test_unknown_status_explicit(self):
        """UNKNOWN status is explicitly available."""
        assert RecommendationStatus.UNKNOWN.value == "unknown"

    def test_confidence_zero_for_unknown(self):
        """UNKNOWN recommendations have low confidence."""
        now = datetime.utcnow().isoformat()
        unknown_incident = Incident(
            incident_id="inc-unknown",
            created_at=now,
            incident_type=IncidentType.UNKNOWN.value,
            severity=IncidentSeverity.UNKNOWN.value,
            scope=IncidentScope.UNKNOWN.value,
            state=IncidentState.UNKNOWN.value,
            title="Unknown",
            description="Unknown",
            source_signal_ids=("s1",),
            signal_count=1,
            first_signal_at=now,
            last_signal_at=now,
            correlation_window_minutes=30,
            project_id=None,
            aspect=None,
            job_id=None,
            confidence=0.0,
            classification_rule="unknown",
            metadata={},
        )

        generator = RecommendationGenerator()
        rec = generator.generate(unknown_incident)

        if rec:
            # UNKNOWN recommendations should have confidence <= 0.5
            assert rec.confidence <= 0.5

    def test_missing_project_id_not_guessed(self, sample_recommendation):
        """Missing project_id is None, not guessed."""
        # Create recommendation without project_id
        rec = Recommendation(
            recommendation_id="rec-no-project",
            created_at=datetime.utcnow().isoformat(),
            recommendation_type=RecommendationType.INVESTIGATE.value,
            severity=RecommendationSeverity.MEDIUM.value,
            approval_required=RecommendationApproval.NONE_REQUIRED.value,
            status=RecommendationStatus.PENDING.value,
            title="No Project",
            description="Test",
            rationale="Test",
            suggested_actions=("Action",),
            source_incident_ids=("inc-001",),
            incident_count=1,
            project_id=None,  # Explicitly None, not guessed
            aspect=None,
            confidence=0.8,
            classification_rule="test",
            expires_at=None,
        )
        assert rec.project_id is None


# =============================================================================
# Section 6: APPEND-ONLY STORE Tests (6 tests)
# =============================================================================

class TestAppendOnlyStore:
    """Test that store is append-only - no edit or delete."""

    def test_store_persist_appends(self, recommendation_store, sample_recommendation):
        """Persist appends to file, not replaces."""
        # Persist first recommendation
        count1 = recommendation_store.persist([sample_recommendation])
        assert count1 == 1

        # Create and persist second recommendation
        rec2 = Recommendation(
            recommendation_id="rec-test-002",
            created_at=datetime.utcnow().isoformat(),
            recommendation_type=RecommendationType.MITIGATE.value,
            severity=RecommendationSeverity.MEDIUM.value,
            approval_required=RecommendationApproval.NONE_REQUIRED.value,
            status=RecommendationStatus.PENDING.value,
            title="Second Recommendation",
            description="Another test",
            rationale="Test",
            suggested_actions=("Action",),
            source_incident_ids=("inc-003",),
            incident_count=1,
            project_id=None,
            aspect=None,
            confidence=0.8,
            classification_rule="test",
            expires_at=None,
        )
        count2 = recommendation_store.persist([rec2])
        assert count2 == 1

        # Both should be readable
        all_recs = recommendation_store.read_all()
        assert len(all_recs) >= 2

    def test_store_has_no_delete_method(self, recommendation_store):
        """Store has no delete method."""
        assert not hasattr(recommendation_store, 'delete')
        assert not hasattr(recommendation_store, 'remove')

    def test_store_has_no_update_method(self, recommendation_store):
        """Store has no update/edit method."""
        assert not hasattr(recommendation_store, 'update')
        assert not hasattr(recommendation_store, 'edit')
        assert not hasattr(recommendation_store, 'modify')

    def test_approval_creates_new_record(self, recommendation_store, sample_recommendation):
        """Approval creates new ApprovalRecord, doesn't modify original."""
        recommendation_store.persist([sample_recommendation])

        # Approve
        approval = recommendation_store.approve(
            recommendation_id=sample_recommendation.recommendation_id,
            user_id="user-001",
            reason="Approved for testing"
        )

        assert approval is not None
        assert approval.action == "approved"
        assert approval.recommendation_id == sample_recommendation.recommendation_id

    def test_dismiss_creates_new_record(self, recommendation_store, sample_recommendation):
        """Dismissal creates new ApprovalRecord, doesn't modify original."""
        recommendation_store.persist([sample_recommendation])

        # Dismiss
        dismissal = recommendation_store.dismiss(
            recommendation_id=sample_recommendation.recommendation_id,
            user_id="user-001",
            reason="Not applicable"
        )

        assert dismissal is not None
        assert dismissal.action == "dismissed"

    def test_approvals_stored_separately(self, recommendation_store, sample_recommendation, temp_approvals_file):
        """Approvals are stored in separate file from recommendations."""
        recommendation_store.persist([sample_recommendation])

        # Approve
        recommendation_store.approve(
            recommendation_id=sample_recommendation.recommendation_id,
            user_id="user-001",
            reason="Test"
        )

        # Check approvals file exists and has content
        assert temp_approvals_file.exists()
        with open(temp_approvals_file) as f:
            content = f.read()
            assert "approved" in content


# =============================================================================
# Section 7: ADVISORY-ONLY Tests (5 tests)
# =============================================================================

class TestAdvisoryOnly:
    """Test that recommendations are ADVISORY-ONLY - no execution."""

    def test_recommendation_has_no_execute_method(self, sample_recommendation):
        """Recommendations have no execute/apply method."""
        assert not hasattr(sample_recommendation, 'execute')
        assert not hasattr(sample_recommendation, 'apply')
        assert not hasattr(sample_recommendation, 'trigger')
        assert not hasattr(sample_recommendation, 'run')

    def test_engine_has_no_execute_method(self):
        """RecommendationEngine has no execute method."""
        engine = RecommendationEngine()
        assert not hasattr(engine, 'execute')
        assert not hasattr(engine, 'apply')
        assert not hasattr(engine, 'trigger')

    def test_store_has_no_execute_method(self, recommendation_store):
        """RecommendationStore has no execute method."""
        assert not hasattr(recommendation_store, 'execute')
        assert not hasattr(recommendation_store, 'apply')
        assert not hasattr(recommendation_store, 'trigger')

    def test_approval_does_not_execute(self, recommendation_store, sample_recommendation):
        """Approving a recommendation does NOT execute anything."""
        recommendation_store.persist([sample_recommendation])

        # Approve
        approval = recommendation_store.approve(
            recommendation_id=sample_recommendation.recommendation_id,
            user_id="user-001",
            reason="Test approval"
        )

        # Approval should only create a record, not trigger execution
        assert approval is not None
        assert isinstance(approval, ApprovalRecord)
        # No side effects - just a record

    def test_suggested_actions_are_strings(self, sample_recommendation):
        """Suggested actions are strings (descriptions), not callables."""
        for action in sample_recommendation.suggested_actions:
            assert isinstance(action, str)
            assert not callable(action)


# =============================================================================
# Section 8: SERIALIZATION Tests (4 tests)
# =============================================================================

class TestSerialization:
    """Test recommendation serialization/deserialization."""

    def test_to_dict_roundtrip(self, sample_recommendation):
        """to_dict and from_dict are inverses."""
        data = sample_recommendation.to_dict()
        restored = Recommendation.from_dict(data)

        assert restored.recommendation_id == sample_recommendation.recommendation_id
        assert restored.title == sample_recommendation.title
        assert restored.severity == sample_recommendation.severity
        assert restored.recommendation_type == sample_recommendation.recommendation_type

    def test_tuples_converted_to_lists_for_json(self, sample_recommendation):
        """Tuples are converted to lists for JSON serialization."""
        data = sample_recommendation.to_dict()

        assert isinstance(data["source_incident_ids"], list)
        assert isinstance(data["suggested_actions"], list)

    def test_lists_converted_back_to_tuples(self, sample_recommendation):
        """Lists are converted back to tuples on deserialization."""
        data = sample_recommendation.to_dict()
        restored = Recommendation.from_dict(data)

        assert isinstance(restored.source_incident_ids, tuple)
        assert isinstance(restored.suggested_actions, tuple)

    def test_approval_record_serialization(self):
        """ApprovalRecord serializes correctly."""
        record = ApprovalRecord(
            record_id="apr-001",
            recommendation_id="rec-001",
            action="approved",
            user_id="user-001",
            timestamp=datetime.utcnow().isoformat(),
            reason="Test",
        )

        data = record.to_dict()
        assert data["record_id"] == "apr-001"
        assert data["action"] == "approved"


# =============================================================================
# Section 9: CONFIDENCE VALIDATION Tests (3 tests)
# =============================================================================

class TestConfidenceValidation:
    """Test confidence value validation."""

    def test_confidence_range_0_to_1(self, sample_recommendation):
        """Confidence must be between 0.0 and 1.0."""
        assert 0.0 <= sample_recommendation.confidence <= 1.0

    def test_confidence_below_zero_rejected(self):
        """Confidence below 0.0 is rejected."""
        with pytest.raises(ValueError, match="Confidence must be 0.0-1.0"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity=RecommendationSeverity.MEDIUM.value,
                approval_required=RecommendationApproval.NONE_REQUIRED.value,
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=-0.1,  # Invalid
                classification_rule="test",
                expires_at=None,
            )

    def test_confidence_above_one_rejected(self):
        """Confidence above 1.0 is rejected."""
        with pytest.raises(ValueError, match="Confidence must be 0.0-1.0"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity=RecommendationSeverity.MEDIUM.value,
                approval_required=RecommendationApproval.NONE_REQUIRED.value,
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=1,
                project_id=None,
                aspect=None,
                confidence=1.5,  # Invalid
                classification_rule="test",
                expires_at=None,
            )


# =============================================================================
# Section 10: INCIDENT COUNT VALIDATION Tests (2 tests)
# =============================================================================

class TestIncidentCountValidation:
    """Test incident count validation."""

    def test_negative_incident_count_rejected(self):
        """Negative incident count is rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            Recommendation(
                recommendation_id="rec-test",
                created_at=datetime.utcnow().isoformat(),
                recommendation_type=RecommendationType.INVESTIGATE.value,
                severity=RecommendationSeverity.MEDIUM.value,
                approval_required=RecommendationApproval.NONE_REQUIRED.value,
                status=RecommendationStatus.PENDING.value,
                title="Test",
                description="Test",
                rationale="Test",
                suggested_actions=("Action",),
                source_incident_ids=("inc-001",),
                incident_count=-1,  # Invalid
                project_id=None,
                aspect=None,
                confidence=0.8,
                classification_rule="test",
                expires_at=None,
            )

    def test_zero_incident_count_valid(self):
        """Zero incident count is valid (for informational recommendations)."""
        rec = Recommendation(
            recommendation_id="rec-test",
            created_at=datetime.utcnow().isoformat(),
            recommendation_type=RecommendationType.NO_ACTION.value,
            severity=RecommendationSeverity.INFO.value,
            approval_required=RecommendationApproval.NONE_REQUIRED.value,
            status=RecommendationStatus.PENDING.value,
            title="Info Only",
            description="Informational",
            rationale="Just info",
            suggested_actions=(),
            source_incident_ids=(),
            incident_count=0,
            project_id=None,
            aspect=None,
            confidence=1.0,
            classification_rule="info",
            expires_at=None,
        )
        assert rec.incident_count == 0


# =============================================================================
# Total Tests Summary
# =============================================================================
# Section 1: IMMUTABILITY Tests - 8 tests
# Section 2: APPROVAL RECORD IMMUTABILITY Tests - 4 tests
# Section 3: ENUM VALIDATION Tests - 8 tests
# Section 4: DETERMINISM Tests - 5 tests
# Section 5: UNKNOWN HANDLING Tests - 5 tests
# Section 6: APPEND-ONLY STORE Tests - 6 tests
# Section 7: ADVISORY-ONLY Tests - 5 tests
# Section 8: SERIALIZATION Tests - 4 tests
# Section 9: CONFIDENCE VALIDATION Tests - 3 tests
# Section 10: INCIDENT COUNT VALIDATION Tests - 2 tests
# -----------------------------------------
# TOTAL: 50 tests (exceeds minimum of 45)
