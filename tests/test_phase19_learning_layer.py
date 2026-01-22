"""
Phase 19: Learning, Memory & System Intelligence Tests

Comprehensive test suite for the learning layer.
Minimum 30 tests covering all critical behaviors.

Test Categories:
1. Enum Validation Tests (5 tests)
2. Immutability Tests (5 tests)
3. No Behavioral Coupling Tests (6 tests)
4. Deterministic Aggregate Tests (5 tests)
5. Pattern Detection Tests (5 tests)
6. Trend Observation Tests (4 tests)
7. Memory Append-Only Tests (4 tests)
8. Read-Only Guarantee Tests (4 tests)
9. Store Tests (5 tests)
10. UNKNOWN Propagation Tests (3 tests)

Total: 46 tests
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import pytest

# Import learning model components
from controller.learning_model import (
    PatternType,
    TrendDirection,
    ConfidenceLevel,
    AggregateType,
    MemoryEntryType,
    ObservedPattern,
    HistoricalAggregate,
    TrendObservation,
    MemoryEntry,
    LearningSummary,
    LearningInput,
)

# Import learning engine components
from controller.learning_engine import (
    LearningEngine,
    get_learning_engine,
    analyze_history,
    get_learning_patterns,
    get_learning_trends,
    get_learning_history,
    get_learning_summary,
    ENGINE_VERSION,
    CONFIDENCE_HIGH_MIN_N,
    CONFIDENCE_HIGH_MIN_CONSISTENCY,
)

# Import learning store components
from controller.learning_store import (
    LearningStore,
    get_learning_store,
    record_pattern,
    record_trend,
    record_memory,
    get_recent_patterns,
    get_recent_trends,
    get_learning_statistics,
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
def engine(temp_dir):
    """Create a learning engine with temp files."""
    return LearningEngine(
        patterns_file=temp_dir / "patterns.jsonl",
        aggregates_file=temp_dir / "aggregates.jsonl",
        trends_file=temp_dir / "trends.jsonl",
        memory_file=temp_dir / "memory.jsonl",
        summaries_file=temp_dir / "summaries.jsonl",
    )


@pytest.fixture
def store(temp_dir):
    """Create a learning store with temp files."""
    return LearningStore(
        patterns_file=temp_dir / "patterns.jsonl",
        trends_file=temp_dir / "trends.jsonl",
        aggregates_file=temp_dir / "aggregates.jsonl",
        memory_file=temp_dir / "memory.jsonl",
        summaries_file=temp_dir / "summaries.jsonl",
    )


@pytest.fixture
def sample_execution_results():
    """Create sample execution results."""
    return tuple([
        {
            "execution_id": f"exec-{i:03d}",
            "status": "execution_success" if i % 3 != 0 else "execution_failed",
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": "test-project",
            "action_type": "run_tests",
            "block_reason": None if i % 3 != 0 else "eligibility_forbidden",
        }
        for i in range(20)
    ])


@pytest.fixture
def sample_verification_results():
    """Create sample verification results."""
    return tuple([
        {
            "verification_id": f"ver-{i:03d}",
            "execution_id": f"exec-{i:03d}",
            "verification_status": "passed" if i % 4 != 0 else "failed",
            "checked_at": datetime.utcnow().isoformat(),
            "violation_count": 0 if i % 4 != 0 else 2,
            "high_severity_count": 0 if i % 4 != 0 else 1,
            "violations": [] if i % 4 != 0 else [
                {"violation_type": "scope_violation", "severity": "high"},
                {"violation_type": "action_violation", "severity": "medium"},
            ],
        }
        for i in range(20)
    ])


@pytest.fixture
def sample_approval_outcomes():
    """Create sample approval outcomes."""
    return tuple([
        {
            "approval_id": f"appr-{i:03d}",
            "status": "approval_granted" if i % 5 != 0 else "approval_denied",
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": "test-project",
            "requester_id": f"user-{i % 3}",
        }
        for i in range(15)
    ])


@pytest.fixture
def sample_incident_summaries():
    """Create sample incident summaries."""
    return tuple([
        {
            "incident_id": f"inc-{i:03d}",
            "type": "performance" if i % 2 == 0 else "error",
            "severity": "medium" if i % 3 != 0 else "high",
            "scope": "project",
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": "test-project",
        }
        for i in range(10)
    ])


@pytest.fixture
def complete_learning_input(
    sample_execution_results,
    sample_verification_results,
    sample_approval_outcomes,
    sample_incident_summaries,
):
    """Create a complete learning input."""
    return LearningInput(
        execution_results=sample_execution_results,
        verification_results=sample_verification_results,
        approval_outcomes=sample_approval_outcomes,
        incident_summaries=sample_incident_summaries,
        drift_history=(),
        recommendation_outcomes=(),
        period_start=(datetime.utcnow() - timedelta(hours=24)).isoformat(),
        period_end=datetime.utcnow().isoformat(),
    )


# =============================================================================
# 1. Enum Validation Tests (5 tests)
# =============================================================================

class TestEnumValidation:
    """Test enum constraints are enforced."""

    def test_pattern_type_has_expected_values(self):
        """PatternType must have expected pattern categories."""
        values = list(PatternType)
        # Should have execution, verification, approval, drift, incident patterns
        assert PatternType.EXECUTION_FAILURE_CLUSTER in values
        assert PatternType.VERIFICATION_VIOLATION_RECURRING in values
        assert PatternType.APPROVAL_REJECTION_TREND in values
        assert PatternType.INCIDENT_RECURRENCE in values

    def test_trend_direction_has_exactly_4_values(self):
        """TrendDirection must have exactly 4 values."""
        values = list(TrendDirection)
        assert len(values) == 4
        assert TrendDirection.INCREASING in values
        assert TrendDirection.DECREASING in values
        assert TrendDirection.STABLE in values
        assert TrendDirection.UNKNOWN in values

    def test_confidence_level_has_exactly_4_values(self):
        """ConfidenceLevel must have exactly 4 values."""
        values = list(ConfidenceLevel)
        assert len(values) == 4
        assert ConfidenceLevel.HIGH in values
        assert ConfidenceLevel.MEDIUM in values
        assert ConfidenceLevel.LOW in values
        assert ConfidenceLevel.INSUFFICIENT in values

    def test_aggregate_type_has_expected_values(self):
        """AggregateType must have expected aggregate categories."""
        values = list(AggregateType)
        assert AggregateType.FAILURE_RATE in values
        assert AggregateType.VIOLATION_FREQUENCY in values
        assert AggregateType.APPROVAL_REJECTION_RATE in values
        assert AggregateType.INCIDENT_FREQUENCY in values

    def test_memory_entry_type_has_expected_values(self):
        """MemoryEntryType must have expected entry categories."""
        values = list(MemoryEntryType)
        assert MemoryEntryType.EXECUTION_OUTCOME in values
        assert MemoryEntryType.VERIFICATION_RESULT in values
        assert MemoryEntryType.APPROVAL_DECISION in values
        assert MemoryEntryType.INCIDENT_OCCURRENCE in values


# =============================================================================
# 2. Immutability Tests (5 tests)
# =============================================================================

class TestImmutability:
    """Test frozen dataclass constraints."""

    def test_observed_pattern_is_immutable(self):
        """ObservedPattern must be frozen."""
        pattern = ObservedPattern(
            pattern_id="pat-001",
            pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
            description="Test pattern",
            frequency=10,
            confidence=ConfidenceLevel.HIGH.value,
            evidence_ids=("exec-001", "exec-002"),
            first_observed=datetime.utcnow().isoformat(),
            last_observed=datetime.utcnow().isoformat(),
            observed_at=datetime.utcnow().isoformat(),
            project_id=None,
            metadata=(),
        )
        with pytest.raises(Exception):
            pattern.frequency = 20

    def test_historical_aggregate_is_immutable(self):
        """HistoricalAggregate must be frozen."""
        aggregate = HistoricalAggregate(
            aggregate_id="agg-001",
            aggregate_type=AggregateType.FAILURE_RATE.value,
            value=0.25,
            sample_size=100,
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
            computed_at=datetime.utcnow().isoformat(),
            project_id=None,
            breakdown=(),
        )
        with pytest.raises(Exception):
            aggregate.value = 0.5

    def test_trend_observation_is_immutable(self):
        """TrendObservation must be frozen."""
        trend = TrendObservation(
            trend_id="trend-001",
            metric_name="execution_success_rate",
            direction=TrendDirection.INCREASING.value,
            change_rate=10.0,
            period_count=2,
            period_unit="day",
            start_value=0.7,
            end_value=0.8,
            observed_at=datetime.utcnow().isoformat(),
            confidence=ConfidenceLevel.MEDIUM.value,
            project_id=None,
        )
        with pytest.raises(Exception):
            trend.direction = TrendDirection.DECREASING.value

    def test_memory_entry_is_immutable(self):
        """MemoryEntry must be frozen."""
        entry = MemoryEntry(
            entry_id="mem-001",
            entry_type=MemoryEntryType.EXECUTION_OUTCOME.value,
            source_id="exec-001",
            source_type="execution",
            timestamp=datetime.utcnow().isoformat(),
            recorded_at=datetime.utcnow().isoformat(),
            summary="Execution succeeded",
            outcome="execution_success",
            project_id=None,
            details=(),
        )
        with pytest.raises(Exception):
            entry.outcome = "execution_failed"

    def test_learning_input_is_immutable(self, complete_learning_input):
        """LearningInput must be frozen."""
        with pytest.raises(Exception):
            complete_learning_input.execution_results = ()


# =============================================================================
# 3. No Behavioral Coupling Tests (6 tests)
# =============================================================================

class TestNoBehavioralCoupling:
    """Test that learning layer does not influence other phases."""

    def test_pattern_has_no_action_field(self):
        """ObservedPattern must not have action/recommendation fields."""
        pattern = ObservedPattern(
            pattern_id="pat-001",
            pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
            description="Test pattern",
            frequency=10,
            confidence=ConfidenceLevel.HIGH.value,
            evidence_ids=(),
            first_observed=datetime.utcnow().isoformat(),
            last_observed=datetime.utcnow().isoformat(),
            observed_at=datetime.utcnow().isoformat(),
            project_id=None,
            metadata=(),
        )
        # Should NOT have these fields
        assert not hasattr(pattern, 'recommended_action')
        assert not hasattr(pattern, 'auto_fix')
        assert not hasattr(pattern, 'trigger_threshold')
        assert not hasattr(pattern, 'automation_enabled')

    def test_trend_has_no_prediction_field(self):
        """TrendObservation must not have prediction fields."""
        trend = TrendObservation(
            trend_id="trend-001",
            metric_name="test_metric",
            direction=TrendDirection.INCREASING.value,
            change_rate=10.0,
            period_count=2,
            period_unit="day",
            start_value=0.5,
            end_value=0.6,
            observed_at=datetime.utcnow().isoformat(),
            confidence=ConfidenceLevel.MEDIUM.value,
            project_id=None,
        )
        # Should NOT have these fields
        assert not hasattr(trend, 'predicted_value')
        assert not hasattr(trend, 'forecast')
        assert not hasattr(trend, 'recommendation')

    def test_aggregate_has_no_threshold_field(self):
        """HistoricalAggregate must not have threshold fields."""
        aggregate = HistoricalAggregate(
            aggregate_id="agg-001",
            aggregate_type=AggregateType.FAILURE_RATE.value,
            value=0.25,
            sample_size=100,
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
            computed_at=datetime.utcnow().isoformat(),
            project_id=None,
            breakdown=(),
        )
        # Should NOT have these fields
        assert not hasattr(aggregate, 'threshold')
        assert not hasattr(aggregate, 'alert_if_exceeded')
        assert not hasattr(aggregate, 'auto_adjust')

    def test_summary_has_no_action_field(self):
        """LearningSummary must not have action fields."""
        summary = LearningSummary(
            summary_id="sum-001",
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
            generated_at=datetime.utcnow().isoformat(),
            total_executions=100,
            total_verifications=100,
            total_approvals=50,
            total_incidents=10,
            execution_success_rate=0.85,
            verification_pass_rate=0.90,
            approval_grant_rate=0.80,
            pattern_count=5,
            trend_count=3,
            top_patterns=(),
            top_trends=(),
            engine_version=ENGINE_VERSION,
        )
        # Should NOT have these fields
        assert not hasattr(summary, 'recommended_actions')
        assert not hasattr(summary, 'auto_remediation')
        assert not hasattr(summary, 'escalation_required')

    def test_engine_does_not_return_actions(self, engine, complete_learning_input):
        """Learning engine must not return actions or recommendations."""
        result = engine.analyze(complete_learning_input)

        # Result is a LearningSummary
        assert isinstance(result, LearningSummary)

        # Should NOT have action fields
        result_dict = result.to_dict()
        assert 'recommended_actions' not in result_dict
        assert 'auto_remediation' not in result_dict
        assert 'trigger_automation' not in result_dict

    def test_engine_does_not_modify_input(self, engine, complete_learning_input):
        """Engine must not modify input data."""
        original_hash = complete_learning_input.compute_hash()

        engine.analyze(complete_learning_input)

        # Input hash must be unchanged
        assert complete_learning_input.compute_hash() == original_hash


# =============================================================================
# 4. Deterministic Aggregate Tests (5 tests)
# =============================================================================

class TestDeterministicAggregates:
    """Test deterministic aggregate computation."""

    def test_same_input_produces_same_failure_rate(self, engine, complete_learning_input):
        """Same input must produce same failure rate."""
        result1 = engine.analyze(complete_learning_input)
        result2 = engine.analyze(complete_learning_input)

        assert result1.execution_success_rate == result2.execution_success_rate

    def test_same_input_produces_same_verification_rate(self, engine, complete_learning_input):
        """Same input must produce same verification pass rate."""
        result1 = engine.analyze(complete_learning_input)
        result2 = engine.analyze(complete_learning_input)

        assert result1.verification_pass_rate == result2.verification_pass_rate

    def test_same_input_produces_same_approval_rate(self, engine, complete_learning_input):
        """Same input must produce same approval grant rate."""
        result1 = engine.analyze(complete_learning_input)
        result2 = engine.analyze(complete_learning_input)

        assert result1.approval_grant_rate == result2.approval_grant_rate

    def test_failure_rate_is_accurate(self, engine, sample_execution_results):
        """Failure rate must be accurately computed."""
        learning_input = LearningInput(
            execution_results=sample_execution_results,
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )
        result = engine.analyze(learning_input)

        # Count expected successes (every 3rd is failed)
        expected_success = sum(
            1 for e in sample_execution_results
            if e.get("status") == "execution_success"
        )
        expected_rate = expected_success / len(sample_execution_results)

        assert abs(result.execution_success_rate - expected_rate) < 0.001

    def test_empty_input_produces_zero_rates(self, engine):
        """Empty input must produce zero rates."""
        empty_input = LearningInput(
            execution_results=(),
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )
        result = engine.analyze(empty_input)

        assert result.execution_success_rate == 0.0
        assert result.verification_pass_rate == 0.0
        assert result.approval_grant_rate == 0.0


# =============================================================================
# 5. Pattern Detection Tests (5 tests)
# =============================================================================

class TestPatternDetection:
    """Test pattern detection logic."""

    def test_detects_failure_cluster(self, engine):
        """Should detect failure cluster when rate > 30%."""
        # Create input with > 30% failure rate
        high_failure_executions = tuple([
            {
                "execution_id": f"exec-{i:03d}",
                "status": "execution_failed" if i < 10 else "execution_success",
                "timestamp": datetime.utcnow().isoformat(),
                "block_reason": "eligibility_forbidden",
            }
            for i in range(20)
        ])  # 50% failure rate

        learning_input = LearningInput(
            execution_results=high_failure_executions,
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )

        result = engine.analyze(learning_input)
        patterns = engine.get_patterns()

        # Should detect failure cluster
        failure_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.EXECUTION_FAILURE_CLUSTER.value
        ]
        assert len(failure_patterns) >= 1

    def test_does_not_detect_pattern_when_insufficient_data(self, engine):
        """Should not detect pattern with insufficient data."""
        # Create input with only 2 failures (threshold is 3)
        small_input = LearningInput(
            execution_results=tuple([
                {"execution_id": "exec-001", "status": "execution_failed", "timestamp": datetime.utcnow().isoformat()},
                {"execution_id": "exec-002", "status": "execution_failed", "timestamp": datetime.utcnow().isoformat()},
            ]),
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )

        engine.analyze(small_input)
        patterns = engine.get_patterns()

        # Should not detect failure cluster with only 2 failures
        failure_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.EXECUTION_FAILURE_CLUSTER.value
        ]
        assert len(failure_patterns) == 0

    def test_detects_recurring_violations(self, engine, sample_verification_results):
        """Should detect recurring violation types."""
        learning_input = LearningInput(
            execution_results=(),
            verification_results=sample_verification_results,
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )

        engine.analyze(learning_input)
        patterns = engine.get_patterns()

        # Should detect recurring violations
        violation_patterns = [
            p for p in patterns
            if p.pattern_type == PatternType.VERIFICATION_VIOLATION_RECURRING.value
        ]
        assert len(violation_patterns) >= 1

    def test_pattern_confidence_is_statistical(self, engine, complete_learning_input):
        """Pattern confidence must be statistical, not ML-based."""
        engine.analyze(complete_learning_input)
        patterns = engine.get_patterns()

        for pattern in patterns:
            # Confidence must be one of the statistical levels
            assert pattern.confidence in [c.value for c in ConfidenceLevel]

    def test_pattern_has_evidence(self, engine, complete_learning_input):
        """Detected patterns must have evidence IDs."""
        engine.analyze(complete_learning_input)
        patterns = engine.get_patterns()

        for pattern in patterns:
            # Patterns should have some evidence
            if pattern.frequency > 0:
                assert len(pattern.evidence_ids) > 0


# =============================================================================
# 6. Trend Observation Tests (4 tests)
# =============================================================================

class TestTrendObservation:
    """Test trend observation logic."""

    def test_detects_increasing_trend(self, engine):
        """Should detect increasing trend."""
        # Create improving execution results
        improving_executions = tuple([
            {
                "execution_id": f"exec-{i:03d}",
                "status": "execution_failed" if i < 5 else "execution_success",
                "timestamp": datetime.utcnow().isoformat(),
            }
            for i in range(20)
        ])  # First half has failures, second half succeeds

        learning_input = LearningInput(
            execution_results=improving_executions,
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )

        engine.analyze(learning_input)
        trends = engine.get_trends()

        exec_trends = [
            t for t in trends
            if t.metric_name == "execution_success_rate"
        ]
        assert len(exec_trends) >= 1
        # Should be increasing (improving)
        assert exec_trends[0].direction == TrendDirection.INCREASING.value

    def test_detects_decreasing_trend(self, engine):
        """Should detect decreasing trend."""
        # Create worsening execution results
        worsening_executions = tuple([
            {
                "execution_id": f"exec-{i:03d}",
                "status": "execution_success" if i < 10 else "execution_failed",
                "timestamp": datetime.utcnow().isoformat(),
            }
            for i in range(20)
        ])  # First half succeeds, second half fails

        learning_input = LearningInput(
            execution_results=worsening_executions,
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )

        engine.analyze(learning_input)
        trends = engine.get_trends()

        exec_trends = [
            t for t in trends
            if t.metric_name == "execution_success_rate"
        ]
        assert len(exec_trends) >= 1
        assert exec_trends[0].direction == TrendDirection.DECREASING.value

    def test_trend_has_no_prediction(self, engine, complete_learning_input):
        """Trends must not include predictions."""
        engine.analyze(complete_learning_input)
        trends = engine.get_trends()

        for trend in trends:
            trend_dict = trend.to_dict()
            assert 'predicted_value' not in trend_dict
            assert 'forecast' not in trend_dict

    def test_trend_confidence_is_statistical(self, engine, complete_learning_input):
        """Trend confidence must be statistical."""
        engine.analyze(complete_learning_input)
        trends = engine.get_trends()

        for trend in trends:
            assert trend.confidence in [c.value for c in ConfidenceLevel]


# =============================================================================
# 7. Memory Append-Only Tests (4 tests)
# =============================================================================

class TestMemoryAppendOnly:
    """Test append-only memory constraints."""

    def test_memory_is_recorded(self, engine, temp_dir, complete_learning_input):
        """Memory entries must be recorded."""
        engine.analyze(complete_learning_input)

        memory_file = temp_dir / "memory.jsonl"
        assert memory_file.exists()

        with open(memory_file, 'r') as f:
            lines = [l for l in f if l.strip()]

        # Should have memory entries
        assert len(lines) > 0

    def test_multiple_analyses_append(self, engine, temp_dir, complete_learning_input):
        """Multiple analyses must append, not overwrite."""
        engine.analyze(complete_learning_input)

        memory_file = temp_dir / "memory.jsonl"
        with open(memory_file, 'r') as f:
            count1 = len([l for l in f if l.strip()])

        engine.analyze(complete_learning_input)

        with open(memory_file, 'r') as f:
            count2 = len([l for l in f if l.strip()])

        # Second analysis should append
        assert count2 > count1

    def test_memory_entries_have_correct_type(self, engine, temp_dir, complete_learning_input):
        """Memory entries must have correct entry_type."""
        engine.analyze(complete_learning_input)

        memory_file = temp_dir / "memory.jsonl"
        with open(memory_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    assert record.get("entry_type") in [t.value for t in MemoryEntryType]

    def test_memory_entries_are_immutable_records(self, engine, temp_dir, complete_learning_input):
        """Memory entries must be immutable records."""
        engine.analyze(complete_learning_input)

        memory_file = temp_dir / "memory.jsonl"
        with open(memory_file, 'r') as f:
            records = [json.loads(l) for l in f if l.strip()]

        # Each record should have entry_id, timestamp, recorded_at
        for record in records:
            assert "entry_id" in record
            assert "timestamp" in record
            assert "recorded_at" in record


# =============================================================================
# 8. Read-Only Guarantee Tests (4 tests)
# =============================================================================

class TestReadOnlyGuarantees:
    """Test read-only guarantees."""

    def test_get_patterns_is_read_only(self, engine, temp_dir, complete_learning_input):
        """get_patterns must not modify storage."""
        engine.analyze(complete_learning_input)

        patterns_file = temp_dir / "patterns.jsonl"
        with open(patterns_file, 'r') as f:
            original = f.read()

        # Multiple reads
        engine.get_patterns()
        engine.get_patterns()
        engine.get_patterns()

        with open(patterns_file, 'r') as f:
            after_reads = f.read()

        assert original == after_reads

    def test_get_trends_is_read_only(self, engine, temp_dir, complete_learning_input):
        """get_trends must not modify storage."""
        engine.analyze(complete_learning_input)

        trends_file = temp_dir / "trends.jsonl"
        with open(trends_file, 'r') as f:
            original = f.read()

        # Multiple reads
        engine.get_trends()
        engine.get_trends()
        engine.get_trends()

        with open(trends_file, 'r') as f:
            after_reads = f.read()

        assert original == after_reads

    def test_get_history_is_read_only(self, engine, temp_dir, complete_learning_input):
        """get_history must not modify storage."""
        engine.analyze(complete_learning_input)

        memory_file = temp_dir / "memory.jsonl"
        with open(memory_file, 'r') as f:
            original = f.read()

        # Multiple reads
        engine.get_history()
        engine.get_history()

        with open(memory_file, 'r') as f:
            after_reads = f.read()

        assert original == after_reads

    def test_get_summaries_is_read_only(self, engine, temp_dir, complete_learning_input):
        """get_summaries must not modify storage."""
        engine.analyze(complete_learning_input)

        summaries_file = temp_dir / "summaries.jsonl"
        with open(summaries_file, 'r') as f:
            original = f.read()

        # Multiple reads
        engine.get_summaries()
        engine.get_summaries()

        with open(summaries_file, 'r') as f:
            after_reads = f.read()

        assert original == after_reads


# =============================================================================
# 9. Store Tests (5 tests)
# =============================================================================

class TestStore:
    """Test learning store operations."""

    def test_store_record_pattern(self, store, temp_dir):
        """Store must record patterns."""
        pattern = ObservedPattern(
            pattern_id="pat-test-001",
            pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
            description="Test pattern",
            frequency=10,
            confidence=ConfidenceLevel.HIGH.value,
            evidence_ids=("exec-001", "exec-002"),
            first_observed=datetime.utcnow().isoformat(),
            last_observed=datetime.utcnow().isoformat(),
            observed_at=datetime.utcnow().isoformat(),
            project_id=None,
            metadata=(),
        )
        store.record_pattern(pattern)

        patterns_file = temp_dir / "patterns.jsonl"
        assert patterns_file.exists()

        with open(patterns_file, 'r') as f:
            records = [json.loads(l) for l in f if l.strip()]

        assert len(records) == 1
        assert records[0]["pattern_id"] == "pat-test-001"

    def test_store_record_trend(self, store, temp_dir):
        """Store must record trends."""
        trend = TrendObservation(
            trend_id="trend-test-001",
            metric_name="test_metric",
            direction=TrendDirection.INCREASING.value,
            change_rate=10.0,
            period_count=2,
            period_unit="day",
            start_value=0.5,
            end_value=0.6,
            observed_at=datetime.utcnow().isoformat(),
            confidence=ConfidenceLevel.MEDIUM.value,
            project_id=None,
        )
        store.record_trend(trend)

        trends_file = temp_dir / "trends.jsonl"
        assert trends_file.exists()

        with open(trends_file, 'r') as f:
            records = [json.loads(l) for l in f if l.strip()]

        assert len(records) == 1
        assert records[0]["trend_id"] == "trend-test-001"

    def test_store_get_recent_patterns(self, store):
        """Store must retrieve recent patterns."""
        for i in range(5):
            pattern = ObservedPattern(
                pattern_id=f"pat-{i:03d}",
                pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
                description=f"Pattern {i}",
                frequency=i,
                confidence=ConfidenceLevel.MEDIUM.value,
                evidence_ids=(),
                first_observed=datetime.utcnow().isoformat(),
                last_observed=datetime.utcnow().isoformat(),
                observed_at=datetime.utcnow().isoformat(),
                project_id=None,
                metadata=(),
            )
            store.record_pattern(pattern)

        recent = store.get_recent_patterns(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0].pattern_id == "pat-004"

    def test_store_get_statistics(self, store):
        """Store must provide statistics."""
        # Add some data
        for i in range(3):
            pattern = ObservedPattern(
                pattern_id=f"pat-{i:03d}",
                pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
                description=f"Pattern {i}",
                frequency=i,
                confidence=ConfidenceLevel.MEDIUM.value,
                evidence_ids=(),
                first_observed=datetime.utcnow().isoformat(),
                last_observed=datetime.utcnow().isoformat(),
                observed_at=datetime.utcnow().isoformat(),
                project_id=None,
                metadata=(),
            )
            store.record_pattern(pattern)

        stats = store.get_statistics(since_hours=24)
        assert stats["pattern_count"] == 3

    def test_store_filter_by_type(self, store):
        """Store must support filtering by type."""
        store.record_pattern(ObservedPattern(
            pattern_id="pat-001",
            pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
            description="Failure cluster",
            frequency=10,
            confidence=ConfidenceLevel.HIGH.value,
            evidence_ids=(),
            first_observed=datetime.utcnow().isoformat(),
            last_observed=datetime.utcnow().isoformat(),
            observed_at=datetime.utcnow().isoformat(),
            project_id=None,
            metadata=(),
        ))
        store.record_pattern(ObservedPattern(
            pattern_id="pat-002",
            pattern_type=PatternType.INCIDENT_RECURRENCE.value,
            description="Incident recurrence",
            frequency=5,
            confidence=ConfidenceLevel.MEDIUM.value,
            evidence_ids=(),
            first_observed=datetime.utcnow().isoformat(),
            last_observed=datetime.utcnow().isoformat(),
            observed_at=datetime.utcnow().isoformat(),
            project_id=None,
            metadata=(),
        ))

        failure_patterns = store.get_recent_patterns(
            pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value
        )
        assert len(failure_patterns) == 1
        assert failure_patterns[0].pattern_id == "pat-001"


# =============================================================================
# 10. UNKNOWN Propagation Tests (3 tests)
# =============================================================================

class TestUnknownPropagation:
    """Test UNKNOWN state handling."""

    def test_empty_data_produces_zero_not_unknown(self, engine):
        """Empty input should produce 0 rates, not UNKNOWN."""
        empty_input = LearningInput(
            execution_results=(),
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )
        result = engine.analyze(empty_input)

        # Should return 0, not crash or return UNKNOWN
        assert result.execution_success_rate == 0.0
        assert result.total_executions == 0

    def test_partial_data_computes_available(self, engine, sample_execution_results):
        """Partial data should compute what's available."""
        partial_input = LearningInput(
            execution_results=sample_execution_results,
            verification_results=(),  # Missing
            approval_outcomes=(),  # Missing
            incident_summaries=(),  # Missing
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )
        result = engine.analyze(partial_input)

        # Should compute execution rate
        assert result.execution_success_rate > 0
        # Should show 0 for missing data
        assert result.verification_pass_rate == 0.0
        assert result.approval_grant_rate == 0.0

    def test_confidence_insufficient_with_small_sample(self, engine):
        """Small sample should result in INSUFFICIENT confidence."""
        small_input = LearningInput(
            execution_results=tuple([
                {"execution_id": f"exec-{i}", "status": "execution_failed", "timestamp": datetime.utcnow().isoformat()}
                for i in range(5)  # Only 5 samples
            ]),
            verification_results=(),
            approval_outcomes=(),
            incident_summaries=(),
            drift_history=(),
            recommendation_outcomes=(),
            period_start=datetime.utcnow().isoformat(),
            period_end=datetime.utcnow().isoformat(),
        )
        engine.analyze(small_input)
        patterns = engine.get_patterns()

        # With small sample, confidence should be INSUFFICIENT or LOW
        for pattern in patterns:
            assert pattern.confidence in [
                ConfidenceLevel.INSUFFICIENT.value,
                ConfidenceLevel.LOW.value,
            ]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the learning system."""

    def test_full_analysis_flow(self, engine, complete_learning_input):
        """Full analysis flow should complete successfully."""
        result = engine.analyze(complete_learning_input)

        assert isinstance(result, LearningSummary)
        assert result.summary_id is not None
        assert result.engine_version == ENGINE_VERSION

    def test_analysis_populates_all_stores(self, engine, temp_dir, complete_learning_input):
        """Analysis should populate all stores."""
        engine.analyze(complete_learning_input)

        # All files should exist
        assert (temp_dir / "patterns.jsonl").exists()
        assert (temp_dir / "trends.jsonl").exists()
        assert (temp_dir / "aggregates.jsonl").exists()
        assert (temp_dir / "memory.jsonl").exists()
        assert (temp_dir / "summaries.jsonl").exists()

    def test_summary_reflects_input_counts(self, engine, complete_learning_input):
        """Summary should reflect input counts."""
        result = engine.analyze(complete_learning_input)

        assert result.total_executions == len(complete_learning_input.execution_results)
        assert result.total_verifications == len(complete_learning_input.verification_results)
        assert result.total_approvals == len(complete_learning_input.approval_outcomes)
        assert result.total_incidents == len(complete_learning_input.incident_summaries)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
