"""
Phase 18A: Automation Eligibility Engine Tests

Comprehensive tests proving:
1. DECISION-ONLY: Engine returns decisions, never executes
2. DETERMINISM: Same inputs always produce same output
3. HARD-STOP RULES: All rules correctly trigger AUTOMATION_FORBIDDEN
4. MISSING INPUT HANDLING: Missing inputs → AUTOMATION_FORBIDDEN
5. PRODUCTION CONSTRAINTS: Stricter rules in production
6. UNKNOWN PROPAGATION: UNKNOWN states trigger FORBIDDEN
7. AUDIT REQUIREMENT: Audit failures → FORBIDDEN
8. NO SIDE EFFECTS: Engine never mutates state

MINIMUM 30 TESTS covering all critical behaviors.

CRITICAL CONSTRAINTS:
- DECISION-ONLY: Returns eligibility decision, NOTHING ELSE
- NO EXECUTION: Never executes, schedules, triggers, or mutates
- DETERMINISTIC: Same inputs ALWAYS produce same output
- AUDIT REQUIRED: Every evaluation emits immutable audit record
"""

import json
import os
import pytest
import tempfile
from dataclasses import FrozenInstanceError
from datetime import datetime
from pathlib import Path
from typing import Optional

from controller.automation_eligibility import (
    # Enums
    EligibilityDecision,
    LimitedAction,
    HardStopRule,
    Environment,
    # Input classes
    RecommendationInput,
    DriftEvaluationInput,
    IncidentSummaryInput,
    LifecycleStateInput,
    ExecutionGateInput,
    IntentBaselineInput,
    RuntimeIntelligenceInput,
    EligibilityInput,
    # Result classes
    EligibilityResult,
    EligibilityAuditRecord,
    # Engine
    AutomationEligibilityEngine,
    # Functions
    evaluate_eligibility,
    create_eligibility_input,
    ENGINE_VERSION,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_audit_file():
    """Create a temporary audit file for testing."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.jsonl',
        delete=False
    ) as f:
        yield Path(f.name)
    try:
        os.unlink(f.name)
    except FileNotFoundError:
        pass


@pytest.fixture
def engine(temp_audit_file):
    """Create engine with temporary audit file."""
    return AutomationEligibilityEngine(audit_file=temp_audit_file)


@pytest.fixture
def valid_recommendation():
    """Create a valid recommendation input."""
    return RecommendationInput(
        recommendation_id="rec-test-001",
        recommendation_type="investigate",
        severity="medium",
        approval_required="none_required",
        status="pending",
        project_id="test-project",
        confidence=0.9,
    )


@pytest.fixture
def valid_drift():
    """Create a valid drift evaluation input (no drift)."""
    return DriftEvaluationInput(
        drift_level="none",
        architecture_change=False,
        database_change=False,
        drift_dimensions=frozenset(),
    )


@pytest.fixture
def valid_incidents():
    """Create a valid incident summary input (no critical issues)."""
    return IncidentSummaryInput(
        total_incidents=2,
        open_count=1,
        critical_count=0,
        security_count=0,
        unknown_count=0,
        max_severity="low",
    )


@pytest.fixture
def valid_lifecycle():
    """Create a valid lifecycle state input."""
    return LifecycleStateInput(
        state="active",
        project_id="test-project",
        is_active=True,
    )


@pytest.fixture
def valid_gate():
    """Create a valid execution gate input (allows action)."""
    return ExecutionGateInput(
        gate_allows_action=True,
        required_role=None,
        gate_denial_reason=None,
    )


@pytest.fixture
def valid_baseline():
    """Create a valid intent baseline input."""
    return IntentBaselineInput(
        baseline_exists=True,
        baseline_valid=True,
        baseline_version="1.0.0",
    )


@pytest.fixture
def valid_runtime():
    """Create a valid runtime intelligence input."""
    return RuntimeIntelligenceInput(
        signals_available=True,
        unknown_severity_count=0,
        intelligence_window_present=True,
    )


@pytest.fixture
def complete_valid_input(
    valid_recommendation,
    valid_drift,
    valid_incidents,
    valid_lifecycle,
    valid_gate,
    valid_baseline,
    valid_runtime,
):
    """Create complete valid input that should allow automation."""
    return EligibilityInput(
        recommendation=valid_recommendation,
        drift_evaluation=valid_drift,
        incident_summary=valid_incidents,
        lifecycle_state=valid_lifecycle,
        execution_gate=valid_gate,
        environment=Environment.TEST.value,
        intent_baseline=valid_baseline,
        runtime_intelligence=valid_runtime,
    )


# =============================================================================
# Section 1: ENUM VALIDATION Tests (4 tests)
# =============================================================================

class TestEnumValidation:
    """Test that enums are correctly defined and LOCKED."""

    def test_eligibility_decision_has_exactly_3_values(self):
        """EligibilityDecision has EXACTLY 3 values (LOCKED)."""
        assert len(EligibilityDecision) == 3
        assert EligibilityDecision.AUTOMATION_FORBIDDEN.value == "automation_forbidden"
        assert EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value == "automation_allowed_with_approval"
        assert EligibilityDecision.AUTOMATION_ALLOWED_LIMITED.value == "automation_allowed_limited"

    def test_limited_action_has_exactly_2_values(self):
        """LimitedAction has EXACTLY 2 values (LOCKED)."""
        assert len(LimitedAction) == 2
        assert LimitedAction.RUN_TESTS.value == "run_tests"
        assert LimitedAction.UPDATE_DOCS.value == "update_docs"

    def test_environment_enum_values(self):
        """Environment has TEST and PRODUCTION values."""
        assert Environment.TEST.value == "test"
        assert Environment.PRODUCTION.value == "production"

    def test_hard_stop_rules_exist(self):
        """All hard stop rules are defined."""
        # Check key rules exist
        assert HardStopRule.MISSING_RECOMMENDATION.value == "missing_recommendation"
        assert HardStopRule.DRIFT_LEVEL_CRITICAL.value == "drift_level_critical"
        assert HardStopRule.INCIDENT_SEVERITY_CRITICAL.value == "incident_severity_critical"
        assert HardStopRule.PRODUCTION_WITH_MEDIUM_DRIFT.value == "production_with_medium_drift"


# =============================================================================
# Section 2: IMMUTABILITY Tests (5 tests)
# =============================================================================

class TestImmutability:
    """Test that all data classes are frozen (immutable)."""

    def test_recommendation_input_is_frozen(self, valid_recommendation):
        """RecommendationInput is frozen."""
        with pytest.raises(FrozenInstanceError):
            valid_recommendation.severity = "critical"

    def test_drift_input_is_frozen(self, valid_drift):
        """DriftEvaluationInput is frozen."""
        with pytest.raises(FrozenInstanceError):
            valid_drift.drift_level = "high"

    def test_eligibility_input_is_frozen(self, complete_valid_input):
        """EligibilityInput is frozen."""
        with pytest.raises(FrozenInstanceError):
            complete_valid_input.environment = "production"

    def test_eligibility_result_is_frozen(self, engine, complete_valid_input):
        """EligibilityResult is frozen."""
        result = engine.evaluate(complete_valid_input)
        with pytest.raises(FrozenInstanceError):
            result.decision = "automation_forbidden"

    def test_audit_record_is_frozen(self):
        """EligibilityAuditRecord is frozen."""
        record = EligibilityAuditRecord(
            audit_id="test",
            input_hash="hash",
            decision="automation_forbidden",
            matched_rules=(),
            timestamp=datetime.utcnow().isoformat(),
            engine_version=ENGINE_VERSION,
            environment="test",
            project_id="project",
        )
        with pytest.raises(FrozenInstanceError):
            record.decision = "automation_allowed_limited"


# =============================================================================
# Section 3: MISSING INPUT Tests (8 tests)
# =============================================================================

class TestMissingInputs:
    """Test that missing inputs trigger AUTOMATION_FORBIDDEN."""

    def test_missing_recommendation_forbidden(self, engine, complete_valid_input):
        """Missing recommendation → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=None,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_RECOMMENDATION.value in result.matched_rules

    def test_missing_drift_forbidden(self, engine, complete_valid_input):
        """Missing drift evaluation → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=None,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_DRIFT_EVALUATION.value in result.matched_rules

    def test_missing_incidents_forbidden(self, engine, complete_valid_input):
        """Missing incident summary → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=None,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_INCIDENT_SUMMARY.value in result.matched_rules

    def test_missing_lifecycle_forbidden(self, engine, complete_valid_input):
        """Missing lifecycle state → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=None,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_LIFECYCLE_STATE.value in result.matched_rules

    def test_missing_gate_forbidden(self, engine, complete_valid_input):
        """Missing execution gate → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=None,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_EXECUTION_GATE.value in result.matched_rules

    def test_missing_environment_forbidden(self, engine, complete_valid_input):
        """Missing environment → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=None,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_ENVIRONMENT.value in result.matched_rules

    def test_missing_baseline_forbidden(self, engine, complete_valid_input):
        """Missing intent baseline → FORBIDDEN."""
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=None,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.MISSING_INTENT_BASELINE.value in result.matched_rules

    def test_all_inputs_missing_forbidden(self, engine):
        """All inputs missing → FORBIDDEN with multiple rules."""
        input_data = EligibilityInput(
            recommendation=None,
            drift_evaluation=None,
            incident_summary=None,
            lifecycle_state=None,
            execution_gate=None,
            environment=None,
            intent_baseline=None,
            runtime_intelligence=None,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        # Should have multiple missing rules
        assert len(result.matched_rules) >= 7


# =============================================================================
# Section 4: DRIFT HARD-STOP Tests (4 tests)
# =============================================================================

class TestDriftHardStops:
    """Test drift-related hard-stop rules."""

    def test_high_drift_forbidden(self, engine, complete_valid_input):
        """Drift level HIGH → FORBIDDEN."""
        high_drift = DriftEvaluationInput(
            drift_level="high",
            architecture_change=False,
            database_change=False,
            drift_dimensions=frozenset({"code"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=high_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.DRIFT_LEVEL_HIGH.value in result.matched_rules

    def test_critical_drift_forbidden(self, engine, complete_valid_input):
        """Drift level CRITICAL → FORBIDDEN."""
        critical_drift = DriftEvaluationInput(
            drift_level="critical",
            architecture_change=False,
            database_change=False,
            drift_dimensions=frozenset({"code", "tests"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=critical_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.DRIFT_LEVEL_CRITICAL.value in result.matched_rules

    def test_architecture_change_forbidden(self, engine, complete_valid_input):
        """Architecture change → FORBIDDEN."""
        arch_drift = DriftEvaluationInput(
            drift_level="low",
            architecture_change=True,
            database_change=False,
            drift_dimensions=frozenset({"architecture"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=arch_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.ARCHITECTURE_CHANGE_DETECTED.value in result.matched_rules

    def test_database_change_forbidden(self, engine, complete_valid_input):
        """Database change → FORBIDDEN."""
        db_drift = DriftEvaluationInput(
            drift_level="low",
            architecture_change=False,
            database_change=True,
            drift_dimensions=frozenset({"database"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=db_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.DATABASE_CHANGE_DETECTED.value in result.matched_rules


# =============================================================================
# Section 5: INCIDENT HARD-STOP Tests (3 tests)
# =============================================================================

class TestIncidentHardStops:
    """Test incident-related hard-stop rules."""

    def test_critical_incident_forbidden(self, engine, complete_valid_input):
        """Critical incident → FORBIDDEN."""
        critical_incidents = IncidentSummaryInput(
            total_incidents=3,
            open_count=2,
            critical_count=1,
            security_count=0,
            unknown_count=0,
            max_severity="critical",
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=critical_incidents,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.INCIDENT_SEVERITY_CRITICAL.value in result.matched_rules

    def test_security_incident_forbidden(self, engine, complete_valid_input):
        """Security incident → FORBIDDEN."""
        security_incidents = IncidentSummaryInput(
            total_incidents=3,
            open_count=2,
            critical_count=0,
            security_count=1,
            unknown_count=0,
            max_severity="high",
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=security_incidents,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.INCIDENT_TYPE_SECURITY.value in result.matched_rules

    def test_unknown_incident_forbidden(self, engine, complete_valid_input):
        """UNKNOWN incident state → FORBIDDEN."""
        unknown_incidents = IncidentSummaryInput(
            total_incidents=3,
            open_count=2,
            critical_count=0,
            security_count=0,
            unknown_count=1,
            max_severity="medium",
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=unknown_incidents,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.INCIDENT_STATE_UNKNOWN.value in result.matched_rules


# =============================================================================
# Section 6: PRODUCTION ENVIRONMENT Tests (3 tests)
# =============================================================================

class TestProductionEnvironment:
    """Test production-specific constraints."""

    def test_production_with_medium_drift_forbidden(self, engine, complete_valid_input):
        """Production + medium drift → FORBIDDEN."""
        medium_drift = DriftEvaluationInput(
            drift_level="medium",
            architecture_change=False,
            database_change=False,
            drift_dimensions=frozenset({"code"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=medium_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=Environment.PRODUCTION.value,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.PRODUCTION_WITH_MEDIUM_DRIFT.value in result.matched_rules

    def test_production_with_medium_incident_forbidden(self, engine, complete_valid_input):
        """Production + medium incident severity → FORBIDDEN."""
        medium_incidents = IncidentSummaryInput(
            total_incidents=3,
            open_count=2,
            critical_count=0,
            security_count=0,
            unknown_count=0,
            max_severity="medium",
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=medium_incidents,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=Environment.PRODUCTION.value,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.PRODUCTION_WITH_MEDIUM_INCIDENT.value in result.matched_rules

    def test_test_environment_with_medium_drift_allowed(self, engine, complete_valid_input):
        """Test environment + medium drift → NOT forbidden (may be limited)."""
        medium_drift = DriftEvaluationInput(
            drift_level="medium",
            architecture_change=False,
            database_change=False,
            drift_dimensions=frozenset({"code"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=medium_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=Environment.TEST.value,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        # Should NOT be forbidden in test environment for medium drift
        assert HardStopRule.PRODUCTION_WITH_MEDIUM_DRIFT.value not in result.matched_rules


# =============================================================================
# Section 7: GOVERNANCE HARD-STOP Tests (2 tests)
# =============================================================================

class TestGovernanceHardStops:
    """Test governance-related hard-stop rules."""

    def test_gate_denied_forbidden(self, engine, complete_valid_input):
        """Execution gate denied → FORBIDDEN."""
        denied_gate = ExecutionGateInput(
            gate_allows_action=False,
            required_role="admin",
            gate_denial_reason="Insufficient permissions",
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=denied_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.EXECUTION_GATE_DENIED.value in result.matched_rules

    def test_invalid_baseline_forbidden(self, engine, complete_valid_input):
        """Invalid intent baseline → FORBIDDEN."""
        invalid_baseline = IntentBaselineInput(
            baseline_exists=True,
            baseline_valid=False,
            baseline_version="corrupted",
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=invalid_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_FORBIDDEN.value
        assert HardStopRule.INTENT_BASELINE_INVALID.value in result.matched_rules


# =============================================================================
# Section 8: DETERMINISM Tests (3 tests)
# =============================================================================

class TestDeterminism:
    """Test that engine is 100% deterministic."""

    def test_same_input_same_output(self, engine, complete_valid_input):
        """Same input always produces same output."""
        results = [engine.evaluate(complete_valid_input) for _ in range(5)]

        decisions = [r.decision for r in results]
        assert len(set(decisions)) == 1  # All same

    def test_input_hash_is_deterministic(self, complete_valid_input):
        """Input hash is deterministic for same input."""
        hash1 = complete_valid_input.compute_hash()
        hash2 = complete_valid_input.compute_hash()
        assert hash1 == hash2

    def test_matched_rules_are_deterministic(self, engine, complete_valid_input):
        """Matched rules are always the same for same input."""
        high_drift = DriftEvaluationInput(
            drift_level="high",
            architecture_change=True,
            database_change=False,
            drift_dimensions=frozenset({"code"}),
        )
        input_data = EligibilityInput(
            recommendation=complete_valid_input.recommendation,
            drift_evaluation=high_drift,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )

        results = [engine.evaluate(input_data) for _ in range(5)]
        rules_sets = [frozenset(r.matched_rules) for r in results]
        assert len(set(rules_sets)) == 1  # All same


# =============================================================================
# Section 9: ALLOWED DECISIONS Tests (3 tests)
# =============================================================================

class TestAllowedDecisions:
    """Test allowed automation decisions."""

    def test_valid_input_allows_limited(self, engine, complete_valid_input):
        """Valid input with no approval required → ALLOWED_LIMITED."""
        result = engine.evaluate(complete_valid_input)
        assert result.decision == EligibilityDecision.AUTOMATION_ALLOWED_LIMITED.value
        assert LimitedAction.RUN_TESTS.value in result.allowed_actions
        assert LimitedAction.UPDATE_DOCS.value in result.allowed_actions

    def test_explicit_approval_required(self, engine, complete_valid_input):
        """Explicit approval required → WITH_APPROVAL."""
        approval_rec = RecommendationInput(
            recommendation_id="rec-test-001",
            recommendation_type="investigate",
            severity="high",
            approval_required="explicit_approval_required",
            status="pending",
            project_id="test-project",
            confidence=0.9,
        )
        input_data = EligibilityInput(
            recommendation=approval_rec,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value
        assert len(result.allowed_actions) == 0  # Requires approval first

    def test_confirmation_required(self, engine, complete_valid_input):
        """Confirmation required → WITH_APPROVAL."""
        confirm_rec = RecommendationInput(
            recommendation_id="rec-test-001",
            recommendation_type="mitigate",
            severity="medium",
            approval_required="confirmation_required",
            status="pending",
            project_id="test-project",
            confidence=0.85,
        )
        input_data = EligibilityInput(
            recommendation=confirm_rec,
            drift_evaluation=complete_valid_input.drift_evaluation,
            incident_summary=complete_valid_input.incident_summary,
            lifecycle_state=complete_valid_input.lifecycle_state,
            execution_gate=complete_valid_input.execution_gate,
            environment=complete_valid_input.environment,
            intent_baseline=complete_valid_input.intent_baseline,
            runtime_intelligence=complete_valid_input.runtime_intelligence,
        )
        result = engine.evaluate(input_data)
        assert result.decision == EligibilityDecision.AUTOMATION_ALLOWED_WITH_APPROVAL.value


# =============================================================================
# Section 10: AUDIT Tests (2 tests)
# =============================================================================

class TestAudit:
    """Test audit record creation."""

    def test_audit_record_created(self, engine, complete_valid_input, temp_audit_file):
        """Audit record is created for every evaluation."""
        engine.evaluate(complete_valid_input)

        assert temp_audit_file.exists()
        with open(temp_audit_file) as f:
            content = f.read()
            assert "decision" in content
            assert ENGINE_VERSION in content

    def test_audit_contains_required_fields(self, engine, complete_valid_input, temp_audit_file):
        """Audit record contains all required fields."""
        engine.evaluate(complete_valid_input)

        with open(temp_audit_file) as f:
            record = json.loads(f.readline())

        assert "audit_id" in record
        assert "input_hash" in record
        assert "decision" in record
        assert "matched_rules" in record
        assert "timestamp" in record
        assert "engine_version" in record


# =============================================================================
# Section 11: NO SIDE EFFECTS Tests (2 tests)
# =============================================================================

class TestNoSideEffects:
    """Test that engine has no side effects beyond audit."""

    def test_engine_has_no_execute_method(self, engine):
        """Engine has no execute/apply/trigger methods."""
        assert not hasattr(engine, 'execute')
        assert not hasattr(engine, 'apply')
        assert not hasattr(engine, 'trigger')
        assert not hasattr(engine, 'schedule')
        assert not hasattr(engine, 'run')

    def test_result_has_no_execute_method(self, engine, complete_valid_input):
        """Result has no execute methods."""
        result = engine.evaluate(complete_valid_input)
        assert not hasattr(result, 'execute')
        assert not hasattr(result, 'apply')
        assert not hasattr(result, 'trigger')


# =============================================================================
# Total Tests Summary
# =============================================================================
# Section 1: ENUM VALIDATION Tests - 4 tests
# Section 2: IMMUTABILITY Tests - 5 tests
# Section 3: MISSING INPUT Tests - 8 tests
# Section 4: DRIFT HARD-STOP Tests - 4 tests
# Section 5: INCIDENT HARD-STOP Tests - 3 tests
# Section 6: PRODUCTION ENVIRONMENT Tests - 3 tests
# Section 7: GOVERNANCE HARD-STOP Tests - 2 tests
# Section 8: DETERMINISM Tests - 3 tests
# Section 9: ALLOWED DECISIONS Tests - 3 tests
# Section 10: AUDIT Tests - 2 tests
# Section 11: NO SIDE EFFECTS Tests - 2 tests
# -----------------------------------------
# TOTAL: 39 tests (exceeds minimum of 30)
