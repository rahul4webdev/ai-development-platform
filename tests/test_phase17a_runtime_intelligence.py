"""
Phase 17A: Runtime Intelligence & Signal Collection Layer Tests

Tests for:
- SignalType and Severity enums (LOCKED)
- RuntimeSignal immutable dataclass
- SignalCollector (read-only collection)
- SignalPersister (append-only storage)
- RuntimeIntelligenceEngine (observation-only)
- SignalSummary (read-only aggregation)

CRITICAL TEST CONSTRAINTS:
- Verify OBSERVATION-ONLY behavior (no mutations)
- Verify UNKNOWN severity on missing data
- Verify append-only persistence (no deletions/modifications)
- Verify deterministic severity classification

Requirements: 25+ comprehensive tests
"""

import json
import os
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

# Import the modules under test
from controller.runtime_intelligence import (
    SignalType,
    Severity,
    SignalSource,
    SignalEnvironment,
    RuntimeSignal,
    SignalSummary,
    SignalCollector,
    SignalPersister,
    RuntimeIntelligenceEngine,
    DEFAULT_POLL_INTERVAL_SECONDS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_signals_dir():
    """Create a temporary directory for signal tests."""
    temp_dir = tempfile.mkdtemp(prefix="signals_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_signals_file(temp_signals_dir):
    """Create a temporary signals file path."""
    return temp_signals_dir / "signals.jsonl"


@pytest.fixture
def signal_persister(temp_signals_file):
    """Create a signal persister with temp file."""
    return SignalPersister(signals_file=temp_signals_file)


@pytest.fixture
def signal_collector():
    """Create a signal collector instance."""
    return SignalCollector()


@pytest.fixture
def runtime_engine(temp_signals_file):
    """Create a runtime intelligence engine with temp file."""
    return RuntimeIntelligenceEngine(
        poll_interval=1,  # Fast polling for tests
        signals_file=temp_signals_file,
    )


@pytest.fixture
def sample_signal() -> RuntimeSignal:
    """Sample runtime signal for testing."""
    return RuntimeSignal(
        signal_id="sig-test-001",
        timestamp=datetime.utcnow().isoformat(),
        signal_type=SignalType.SYSTEM_RESOURCE.value,
        severity=Severity.INFO.value,
        source=SignalSource.SYSTEM.value,
        project_id=None,
        aspect=None,
        environment=SignalEnvironment.NONE.value,
        raw_value=50.0,
        normalized_value=0.5,
        confidence=1.0,
        description="CPU usage: 50%",
        metadata={"metric": "cpu_percent"},
    )


@pytest.fixture
def sample_signals() -> List[RuntimeSignal]:
    """List of sample signals with various severities."""
    now = datetime.utcnow()
    return [
        RuntimeSignal(
            signal_id="sig-test-001",
            timestamp=now.isoformat(),
            signal_type=SignalType.SYSTEM_RESOURCE.value,
            severity=Severity.INFO.value,
            source=SignalSource.SYSTEM.value,
            project_id=None,
            aspect=None,
            environment=SignalEnvironment.NONE.value,
            raw_value=30.0,
            normalized_value=0.3,
            confidence=1.0,
            description="CPU usage: 30%",
            metadata={"metric": "cpu_percent"},
        ),
        RuntimeSignal(
            signal_id="sig-test-002",
            timestamp=(now - timedelta(minutes=5)).isoformat(),
            signal_type=SignalType.WORKER_QUEUE.value,
            severity=Severity.WARNING.value,
            source=SignalSource.CONTROLLER.value,
            project_id="test-project",
            aspect="backend",
            environment=SignalEnvironment.TEST.value,
            raw_value=3,
            normalized_value=1.0,
            confidence=1.0,
            description="Queue saturation: 100%",
            metadata={"queued_jobs": 3},
        ),
        RuntimeSignal(
            signal_id="sig-test-003",
            timestamp=(now - timedelta(minutes=10)).isoformat(),
            signal_type=SignalType.DRIFT_WARNING.value,
            severity=Severity.UNKNOWN.value,
            source=SignalSource.LIFECYCLE.value,
            project_id=None,
            aspect=None,
            environment=SignalEnvironment.NONE.value,
            raw_value=None,
            normalized_value=None,
            confidence=0.0,
            description="Lifecycle data unavailable",
            metadata={"reason": "file_not_found"},
        ),
    ]


# =============================================================================
# Test: Enums (LOCKED)
# =============================================================================

class TestSignalTypeEnum:
    """Test SignalType enum (LOCKED - 8 types)."""

    def test_signal_type_values_locked(self):
        """Verify all 8 signal types exist and are LOCKED."""
        expected_types = {
            "system_resource",
            "worker_queue",
            "job_failure",
            "test_regression",
            "deployment_failure",
            "drift_warning",
            "human_override",
            "config_anomaly",
        }
        actual_types = {t.value for t in SignalType}
        assert actual_types == expected_types, "SignalType enum must have exactly 8 LOCKED types"

    def test_signal_type_string_values(self):
        """Verify signal types are string values for JSON serialization."""
        for signal_type in SignalType:
            assert isinstance(signal_type.value, str)

    def test_signal_type_count(self):
        """Verify exactly 8 signal types (LOCKED)."""
        assert len(SignalType) == 8


class TestSeverityEnum:
    """Test Severity enum (LOCKED - 5 levels)."""

    def test_severity_values_locked(self):
        """Verify all 5 severity levels exist and are LOCKED."""
        expected_severities = {"info", "warning", "degraded", "critical", "unknown"}
        actual_severities = {s.value for s in Severity}
        assert actual_severities == expected_severities, "Severity enum must have exactly 5 LOCKED levels"

    def test_severity_includes_unknown(self):
        """Verify UNKNOWN severity exists for missing data handling."""
        assert Severity.UNKNOWN.value == "unknown"

    def test_severity_count(self):
        """Verify exactly 5 severity levels (LOCKED)."""
        assert len(Severity) == 5


class TestSignalSourceEnum:
    """Test SignalSource enum."""

    def test_signal_sources(self):
        """Verify signal sources exist."""
        assert SignalSource.SYSTEM.value == "system"
        assert SignalSource.CONTROLLER.value == "controller"
        assert SignalSource.CLAUDE.value == "claude"
        assert SignalSource.LIFECYCLE.value == "lifecycle"
        assert SignalSource.HUMAN.value == "human"


# =============================================================================
# Test: RuntimeSignal (Immutable Dataclass)
# =============================================================================

class TestRuntimeSignal:
    """Test RuntimeSignal immutable dataclass."""

    def test_signal_creation(self, sample_signal):
        """Verify signal can be created with valid data."""
        assert sample_signal.signal_id == "sig-test-001"
        assert sample_signal.signal_type == SignalType.SYSTEM_RESOURCE.value
        assert sample_signal.severity == Severity.INFO.value

    def test_signal_immutability(self, sample_signal):
        """Verify signal is frozen (immutable)."""
        with pytest.raises(AttributeError):
            sample_signal.severity = Severity.CRITICAL.value

    def test_signal_validation_confidence_too_low(self):
        """Verify confidence validation rejects values below 0."""
        with pytest.raises(ValueError, match="Confidence must be"):
            RuntimeSignal(
                signal_id="sig-invalid",
                timestamp=datetime.utcnow().isoformat(),
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity=Severity.INFO.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=50.0,
                normalized_value=0.5,
                confidence=-0.1,  # Invalid
                description="Test",
            )

    def test_signal_validation_confidence_too_high(self):
        """Verify confidence validation rejects values above 1."""
        with pytest.raises(ValueError, match="Confidence must be"):
            RuntimeSignal(
                signal_id="sig-invalid",
                timestamp=datetime.utcnow().isoformat(),
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity=Severity.INFO.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=50.0,
                normalized_value=0.5,
                confidence=1.5,  # Invalid
                description="Test",
            )

    def test_signal_validation_invalid_severity(self):
        """Verify severity validation rejects invalid values."""
        with pytest.raises(ValueError, match="Invalid severity"):
            RuntimeSignal(
                signal_id="sig-invalid",
                timestamp=datetime.utcnow().isoformat(),
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity="invalid_severity",  # Invalid
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=50.0,
                normalized_value=0.5,
                confidence=1.0,
                description="Test",
            )

    def test_signal_validation_invalid_signal_type(self):
        """Verify signal type validation rejects invalid values."""
        with pytest.raises(ValueError, match="Invalid signal type"):
            RuntimeSignal(
                signal_id="sig-invalid",
                timestamp=datetime.utcnow().isoformat(),
                signal_type="invalid_type",  # Invalid
                severity=Severity.INFO.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=50.0,
                normalized_value=0.5,
                confidence=1.0,
                description="Test",
            )

    def test_signal_to_dict(self, sample_signal):
        """Verify signal serializes to dictionary."""
        data = sample_signal.to_dict()
        assert isinstance(data, dict)
        assert data["signal_id"] == "sig-test-001"
        assert data["signal_type"] == SignalType.SYSTEM_RESOURCE.value

    def test_signal_from_dict(self, sample_signal):
        """Verify signal deserializes from dictionary."""
        data = sample_signal.to_dict()
        restored = RuntimeSignal.from_dict(data)
        assert restored.signal_id == sample_signal.signal_id
        assert restored.severity == sample_signal.severity

    def test_signal_roundtrip(self, sample_signal):
        """Verify signal survives JSON roundtrip."""
        json_str = json.dumps(sample_signal.to_dict())
        data = json.loads(json_str)
        restored = RuntimeSignal.from_dict(data)
        assert restored.signal_id == sample_signal.signal_id


# =============================================================================
# Test: SignalCollector (Read-Only)
# =============================================================================

class TestSignalCollector:
    """Test SignalCollector read-only behavior."""

    def test_generate_signal_id_unique(self, signal_collector):
        """Verify signal IDs are unique."""
        id1 = signal_collector._generate_signal_id()
        id2 = signal_collector._generate_signal_id()
        assert id1 != id2

    def test_generate_signal_id_format(self, signal_collector):
        """Verify signal ID format."""
        signal_id = signal_collector._generate_signal_id()
        assert signal_id.startswith("sig-")

    def test_create_unknown_signal(self, signal_collector):
        """Verify UNKNOWN signal creation for missing data."""
        unknown_signal = signal_collector._create_unknown_signal(
            SignalType.SYSTEM_RESOURCE,
            SignalSource.SYSTEM,
            "Test reason",
        )
        assert unknown_signal.severity == Severity.UNKNOWN.value
        assert unknown_signal.confidence == 0.0
        assert "Test reason" in unknown_signal.description

    @patch('controller.runtime_intelligence.psutil')
    def test_collect_system_signals_success(self, mock_psutil, signal_collector):
        """Verify system signals collection with mocked psutil."""
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = MagicMock(percent=60.0, available=4*1024*1024*1024)
        mock_psutil.disk_usage.return_value = MagicMock(percent=70.0, free=50*1024*1024*1024)

        signals = signal_collector.collect_system_signals()
        assert len(signals) >= 3  # CPU, Memory, Disk

        # Verify deterministic severity classification
        for sig in signals:
            assert sig.severity in [s.value for s in Severity]

    @patch('controller.runtime_intelligence.psutil')
    def test_collect_system_signals_unknown_on_failure(self, mock_psutil, signal_collector):
        """Verify UNKNOWN severity when psutil fails."""
        mock_psutil.cpu_percent.side_effect = Exception("Test error")
        mock_psutil.virtual_memory.side_effect = Exception("Test error")
        mock_psutil.disk_usage.side_effect = Exception("Test error")

        signals = signal_collector.collect_system_signals()
        # All signals should be UNKNOWN due to collection failure
        for sig in signals:
            assert sig.severity == Severity.UNKNOWN.value

    def test_classify_cpu_severity_deterministic(self, signal_collector):
        """Verify CPU severity classification is deterministic."""
        # INFO: < 70%
        assert signal_collector._classify_cpu_severity(30.0) == Severity.INFO
        assert signal_collector._classify_cpu_severity(69.0) == Severity.INFO

        # WARNING: 70-85%
        assert signal_collector._classify_cpu_severity(70.0) == Severity.WARNING
        assert signal_collector._classify_cpu_severity(84.0) == Severity.WARNING

        # DEGRADED: 85-95%
        assert signal_collector._classify_cpu_severity(85.0) == Severity.DEGRADED
        assert signal_collector._classify_cpu_severity(94.0) == Severity.DEGRADED

        # CRITICAL: >= 95%
        assert signal_collector._classify_cpu_severity(95.0) == Severity.CRITICAL
        assert signal_collector._classify_cpu_severity(100.0) == Severity.CRITICAL

    def test_classify_memory_severity_deterministic(self, signal_collector):
        """Verify memory severity classification is deterministic."""
        # INFO: < 70%
        assert signal_collector._classify_memory_severity(50.0) == Severity.INFO

        # WARNING: 70-85%
        assert signal_collector._classify_memory_severity(75.0) == Severity.WARNING

        # DEGRADED: 85-95%
        assert signal_collector._classify_memory_severity(90.0) == Severity.DEGRADED

        # CRITICAL: >= 95%
        assert signal_collector._classify_memory_severity(96.0) == Severity.CRITICAL

    def test_classify_disk_severity_deterministic(self, signal_collector):
        """Verify disk severity classification is deterministic."""
        # INFO: < 75%
        assert signal_collector._classify_disk_severity(50.0) == Severity.INFO

        # WARNING: 75-85%
        assert signal_collector._classify_disk_severity(80.0) == Severity.WARNING

        # DEGRADED: 85-95%
        assert signal_collector._classify_disk_severity(90.0) == Severity.DEGRADED

        # CRITICAL: >= 95%
        assert signal_collector._classify_disk_severity(96.0) == Severity.CRITICAL


# =============================================================================
# Test: SignalPersister (Append-Only)
# =============================================================================

class TestSignalPersister:
    """Test SignalPersister append-only behavior."""

    def test_persist_signals(self, signal_persister, sample_signals, temp_signals_file):
        """Verify signals are persisted."""
        count = signal_persister.persist(sample_signals)
        assert count == len(sample_signals)
        assert temp_signals_file.exists()

    def test_persist_append_only(self, signal_persister, sample_signals, temp_signals_file):
        """Verify persisting appends, never overwrites."""
        # First persist
        signal_persister.persist(sample_signals[:1])

        # Read line count
        with open(temp_signals_file) as f:
            lines1 = len(f.readlines())

        # Second persist
        signal_persister.persist(sample_signals[1:])

        # Read line count again
        with open(temp_signals_file) as f:
            lines2 = len(f.readlines())

        assert lines2 == lines1 + len(sample_signals[1:])

    def test_persist_empty_list(self, signal_persister):
        """Verify persisting empty list returns 0."""
        count = signal_persister.persist([])
        assert count == 0

    def test_read_signals(self, signal_persister, sample_signals):
        """Verify signals can be read back."""
        signal_persister.persist(sample_signals)
        read_signals = signal_persister.read_signals()
        assert len(read_signals) == len(sample_signals)

    def test_read_signals_filter_by_project(self, signal_persister, sample_signals):
        """Verify project filtering works."""
        signal_persister.persist(sample_signals)
        filtered = signal_persister.read_signals(project_id="test-project")
        assert len(filtered) == 1
        assert filtered[0].project_id == "test-project"

    def test_read_signals_filter_by_severity(self, signal_persister, sample_signals):
        """Verify severity filtering works."""
        signal_persister.persist(sample_signals)
        filtered = signal_persister.read_signals(severity=Severity.UNKNOWN)
        assert len(filtered) == 1
        assert filtered[0].severity == Severity.UNKNOWN.value

    def test_read_signals_filter_by_type(self, signal_persister, sample_signals):
        """Verify signal type filtering works."""
        signal_persister.persist(sample_signals)
        filtered = signal_persister.read_signals(signal_type=SignalType.WORKER_QUEUE)
        assert len(filtered) == 1
        assert filtered[0].signal_type == SignalType.WORKER_QUEUE.value

    def test_read_signals_filter_by_since(self, signal_persister, sample_signals):
        """Verify time filtering works."""
        signal_persister.persist(sample_signals)
        since = datetime.utcnow() - timedelta(minutes=7)
        filtered = signal_persister.read_signals(since=since)
        # Should exclude the 10-minute-old signal
        assert len(filtered) == 2

    def test_read_signals_limit(self, signal_persister, sample_signals):
        """Verify limit parameter works."""
        signal_persister.persist(sample_signals)
        filtered = signal_persister.read_signals(limit=1)
        assert len(filtered) == 1

    def test_read_signals_empty_file(self, signal_persister):
        """Verify reading non-existent file returns empty list."""
        signals = signal_persister.read_signals()
        assert signals == []

    def test_get_summary(self, signal_persister, sample_signals):
        """Verify summary generation."""
        signal_persister.persist(sample_signals)
        summary = signal_persister.get_summary()

        assert summary.total_signals == len(sample_signals)
        assert summary.unknown_count == 1  # One UNKNOWN in sample_signals
        assert Severity.INFO.value in summary.by_severity
        assert SignalType.SYSTEM_RESOURCE.value in summary.by_type

    def test_get_summary_observability_status(self, signal_persister, sample_signals):
        """Verify observability status determination."""
        signal_persister.persist(sample_signals)
        summary = signal_persister.get_summary()

        # With 1 unknown out of 3 signals, status should be "partial"
        assert summary.observability_status == "partial"


# =============================================================================
# Test: RuntimeIntelligenceEngine (Observation-Only)
# =============================================================================

class TestRuntimeIntelligenceEngine:
    """Test RuntimeIntelligenceEngine observation-only behavior."""

    def test_engine_creation(self, runtime_engine):
        """Verify engine creates correctly."""
        assert runtime_engine._poll_interval == 1
        assert runtime_engine._running is False

    @patch.object(SignalCollector, 'collect_all')
    def test_poll_once(self, mock_collect, runtime_engine):
        """Verify single poll cycle."""
        mock_collect.return_value = [
            RuntimeSignal(
                signal_id="sig-poll-test",
                timestamp=datetime.utcnow().isoformat(),
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity=Severity.INFO.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=50.0,
                normalized_value=0.5,
                confidence=1.0,
                description="Test signal",
            )
        ]

        persisted, signals = runtime_engine.poll_once()
        assert persisted == 1
        assert len(signals) == 1
        assert runtime_engine._poll_count == 1

    @patch.object(SignalCollector, 'collect_all')
    def test_poll_increments_counter(self, mock_collect, runtime_engine):
        """Verify poll count increments."""
        mock_collect.return_value = []

        runtime_engine.poll_once()
        assert runtime_engine._poll_count == 1

        runtime_engine.poll_once()
        assert runtime_engine._poll_count == 2

    @patch.object(SignalCollector, 'collect_all')
    def test_poll_updates_timestamp(self, mock_collect, runtime_engine):
        """Verify last poll timestamp is updated."""
        mock_collect.return_value = []

        assert runtime_engine._last_poll_timestamp is None
        runtime_engine.poll_once()
        assert runtime_engine._last_poll_timestamp is not None

    @patch.object(SignalCollector, 'collect_all')
    def test_poll_handles_failure(self, mock_collect, runtime_engine):
        """Verify poll handles collection failure gracefully."""
        mock_collect.side_effect = Exception("Test error")

        # Should not raise
        persisted, signals = runtime_engine.poll_once()
        assert persisted == 0
        assert signals == []

    def test_start_stop_polling(self, runtime_engine):
        """Verify polling can be started and stopped."""
        runtime_engine.start_polling()
        assert runtime_engine._running is True

        runtime_engine.stop_polling()
        assert runtime_engine._running is False

    def test_double_start_no_error(self, runtime_engine):
        """Verify double start doesn't cause error."""
        runtime_engine.start_polling()
        runtime_engine.start_polling()  # Should be no-op
        assert runtime_engine._running is True
        runtime_engine.stop_polling()

    def test_double_stop_no_error(self, runtime_engine):
        """Verify double stop doesn't cause error."""
        runtime_engine.start_polling()
        runtime_engine.stop_polling()
        runtime_engine.stop_polling()  # Should be no-op
        assert runtime_engine._running is False

    def test_get_status(self, runtime_engine):
        """Verify get_status returns expected structure."""
        status = runtime_engine.get_status()

        assert "running" in status
        assert "poll_interval_seconds" in status
        assert "last_poll_timestamp" in status
        assert "poll_count" in status
        assert "signals_file" in status

    @patch.object(SignalPersister, 'read_signals')
    def test_get_signals(self, mock_read, runtime_engine):
        """Verify get_signals delegates to persister."""
        mock_read.return_value = []

        result = runtime_engine.get_signals()
        mock_read.assert_called_once()
        assert result == []

    @patch.object(SignalPersister, 'get_summary')
    def test_get_summary(self, mock_summary, runtime_engine):
        """Verify get_summary delegates to persister."""
        mock_summary.return_value = SignalSummary(
            generated_at=datetime.utcnow().isoformat(),
            time_window_start=(datetime.utcnow() - timedelta(hours=24)).isoformat(),
            time_window_end=datetime.utcnow().isoformat(),
            total_signals=0,
            by_severity={},
            by_type={},
            by_source={},
            unknown_count=0,
            last_signal_timestamp=None,
            observability_status="no_data",
        )

        result = runtime_engine.get_summary()
        mock_summary.assert_called_once()
        assert result.total_signals == 0


# =============================================================================
# Test: Observation-Only Verification
# =============================================================================

class TestObservationOnlyBehavior:
    """Test that the system is OBSERVATION-ONLY."""

    def test_signal_immutability_enforced(self, sample_signal):
        """Verify signals cannot be modified after creation."""
        with pytest.raises(AttributeError):
            sample_signal.severity = Severity.CRITICAL.value

    def test_persister_append_only_no_delete(self, signal_persister, sample_signals, temp_signals_file):
        """Verify persister has no delete method."""
        signal_persister.persist(sample_signals)

        # There should be no delete method
        assert not hasattr(signal_persister, 'delete')
        assert not hasattr(signal_persister, 'delete_signal')
        assert not hasattr(signal_persister, 'remove')

    def test_persister_append_only_no_update(self, signal_persister):
        """Verify persister has no update method."""
        # There should be no update method
        assert not hasattr(signal_persister, 'update')
        assert not hasattr(signal_persister, 'update_signal')
        assert not hasattr(signal_persister, 'modify')

    def test_collector_no_mutation_methods(self, signal_collector):
        """Verify collector has no mutation methods."""
        # Collector should only collect (read)
        assert hasattr(signal_collector, 'collect_all')
        assert hasattr(signal_collector, 'collect_system_signals')

        # No mutation methods
        assert not hasattr(signal_collector, 'set')
        assert not hasattr(signal_collector, 'update')
        assert not hasattr(signal_collector, 'restart')
        assert not hasattr(signal_collector, 'kill')

    def test_engine_no_lifecycle_methods(self, runtime_engine):
        """Verify engine has no lifecycle mutation methods."""
        # Engine should only observe
        assert hasattr(runtime_engine, 'poll_once')
        assert hasattr(runtime_engine, 'get_signals')
        assert hasattr(runtime_engine, 'get_summary')

        # No lifecycle mutation methods
        assert not hasattr(runtime_engine, 'transition')
        assert not hasattr(runtime_engine, 'deploy')
        assert not hasattr(runtime_engine, 'execute_job')
        assert not hasattr(runtime_engine, 'fix')
        assert not hasattr(runtime_engine, 'heal')


# =============================================================================
# Test: UNKNOWN Severity on Missing Data
# =============================================================================

class TestUnknownSeverityBehavior:
    """Test UNKNOWN severity for missing data."""

    def test_unknown_severity_on_collection_failure(self, signal_collector):
        """Verify UNKNOWN signal is created on failure."""
        unknown = signal_collector._create_unknown_signal(
            SignalType.SYSTEM_RESOURCE,
            SignalSource.SYSTEM,
            "Data unavailable",
        )
        assert unknown.severity == Severity.UNKNOWN.value
        assert unknown.confidence == 0.0

    @patch('controller.runtime_intelligence.psutil')
    def test_system_unknown_on_psutil_failure(self, mock_psutil, signal_collector):
        """Verify system signals return UNKNOWN when psutil fails."""
        mock_psutil.cpu_percent.side_effect = Exception("Failed")
        mock_psutil.virtual_memory.side_effect = Exception("Failed")
        mock_psutil.disk_usage.side_effect = Exception("Failed")

        signals = signal_collector.collect_system_signals()
        for sig in signals:
            assert sig.severity == Severity.UNKNOWN.value, \
                "Missing data MUST produce UNKNOWN severity"

    def test_worker_unknown_on_missing_file(self, signal_collector):
        """Verify worker signals return UNKNOWN when state file missing."""
        # Patch the file path to a non-existent location
        with patch('controller.runtime_intelligence.Path') as mock_path:
            mock_instance = MagicMock()
            mock_instance.exists.return_value = False
            mock_path.return_value = mock_instance

            signals = signal_collector.collect_worker_signals()
            # At least one signal should be UNKNOWN
            unknown_signals = [s for s in signals if s.severity == Severity.UNKNOWN.value]
            assert len(unknown_signals) > 0


# =============================================================================
# Test: Signal Summary
# =============================================================================

class TestSignalSummary:
    """Test SignalSummary aggregation."""

    def test_summary_to_dict(self):
        """Verify summary serializes correctly."""
        now = datetime.utcnow()
        summary = SignalSummary(
            generated_at=now.isoformat(),
            time_window_start=(now - timedelta(hours=24)).isoformat(),
            time_window_end=now.isoformat(),
            total_signals=100,
            by_severity={"info": 80, "warning": 15, "critical": 5},
            by_type={"system_resource": 60, "worker_queue": 40},
            by_source={"system": 60, "controller": 40},
            unknown_count=0,
            last_signal_timestamp=now.isoformat(),
            observability_status="healthy",
        )

        data = summary.to_dict()
        assert data["total_signals"] == 100
        assert data["observability_status"] == "healthy"

    def test_summary_observability_healthy(self, signal_persister):
        """Verify healthy status when no UNKNOWN signals."""
        signal = RuntimeSignal(
            signal_id="sig-healthy",
            timestamp=datetime.utcnow().isoformat(),
            signal_type=SignalType.SYSTEM_RESOURCE.value,
            severity=Severity.INFO.value,
            source=SignalSource.SYSTEM.value,
            project_id=None,
            aspect=None,
            environment=SignalEnvironment.NONE.value,
            raw_value=50.0,
            normalized_value=0.5,
            confidence=1.0,
            description="Healthy signal",
        )
        signal_persister.persist([signal])
        summary = signal_persister.get_summary()

        assert summary.observability_status == "healthy"

    def test_summary_observability_no_data(self, signal_persister):
        """Verify no_data status when no signals exist."""
        summary = signal_persister.get_summary()
        assert summary.observability_status == "no_data"
