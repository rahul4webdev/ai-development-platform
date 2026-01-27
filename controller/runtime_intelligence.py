"""
Phase 17A: Runtime Intelligence & Signal Collection Layer

This module provides OBSERVATION-ONLY capabilities for the platform.
It collects signals, classifies severity, and persists data for analysis.

CRITICAL CONSTRAINTS:
- READ-ONLY: No lifecycle transitions, no deployments, no intent mutation
- DETERMINISTIC: No ML, no probabilistic inference
- EXPLICIT UNKNOWN: Missing data = UNKNOWN severity, never guessed
- APPEND-ONLY: Signals are never deleted or modified
- NO AUTO-FIX: Signals are reported, never acted upon

This phase gives the system eyes and ears, NOT hands.
"""

import json
import logging
import os
import psutil
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import threading

logger = logging.getLogger("runtime_intelligence")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OBSERVABILITY_DIR = Path(os.getenv(
    "OBSERVABILITY_DIR",
    "/home/aitesting.mybd.in/jobs/observability"
))

# Fallback for local development/testing
try:
    OBSERVABILITY_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    OBSERVABILITY_DIR = Path("/tmp/observability")
    OBSERVABILITY_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_FILE = OBSERVABILITY_DIR / "signals.jsonl"
POLL_AUDIT_FILE = OBSERVABILITY_DIR / "poll_audit.log"

# Polling configuration
DEFAULT_POLL_INTERVAL_SECONDS = 60
SIGNAL_RETENTION_DAYS = 30  # For query purposes only, never delete


# -----------------------------------------------------------------------------
# Signal Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class SignalType(str, Enum):
    """
    Types of runtime signals.

    This enum is LOCKED - do not add types without explicit approval.
    """
    SYSTEM_RESOURCE = "system_resource"
    WORKER_QUEUE = "worker_queue"
    JOB_FAILURE = "job_failure"
    TEST_REGRESSION = "test_regression"
    DEPLOYMENT_FAILURE = "deployment_failure"
    DRIFT_WARNING = "drift_warning"
    HUMAN_OVERRIDE = "human_override"
    CONFIG_ANOMALY = "config_anomaly"


# -----------------------------------------------------------------------------
# Severity Enum (LOCKED)
# -----------------------------------------------------------------------------
class Severity(str, Enum):
    """
    Signal severity levels.

    CRITICAL: Missing data or collection failure MUST produce UNKNOWN, not failure.
    """
    INFO = "info"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"  # MANDATORY when data is missing


# -----------------------------------------------------------------------------
# Signal Source Enum
# -----------------------------------------------------------------------------
class SignalSource(str, Enum):
    """Sources of runtime signals."""
    SYSTEM = "system"
    CONTROLLER = "controller"
    CLAUDE = "claude"
    LIFECYCLE = "lifecycle"
    HUMAN = "human"


# -----------------------------------------------------------------------------
# Environment Enum
# -----------------------------------------------------------------------------
class SignalEnvironment(str, Enum):
    """Environment context for signals."""
    TEST = "test"
    PROD = "prod"
    NONE = "none"


# -----------------------------------------------------------------------------
# Runtime Signal (Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeSignal:
    """
    Immutable runtime signal.

    Once created, a signal CANNOT be modified.
    This ensures audit integrity.
    """
    signal_id: str
    timestamp: str  # ISO format
    signal_type: str  # SignalType value
    severity: str  # Severity value
    source: str  # SignalSource value
    project_id: Optional[str]
    aspect: Optional[str]
    environment: str  # SignalEnvironment value
    raw_value: Any
    normalized_value: Any
    confidence: float  # 0.0 - 1.0, deterministic
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate signal on creation."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.severity not in [s.value for s in Severity]:
            raise ValueError(f"Invalid severity: {self.severity}")
        if self.signal_type not in [t.value for t in SignalType]:
            raise ValueError(f"Invalid signal type: {self.signal_type}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeSignal":
        """Create signal from dictionary."""
        # Handle frozen dataclass
        return cls(
            signal_id=data["signal_id"],
            timestamp=data["timestamp"],
            signal_type=data["signal_type"],
            severity=data["severity"],
            source=data["source"],
            project_id=data.get("project_id"),
            aspect=data.get("aspect"),
            environment=data.get("environment", SignalEnvironment.NONE.value),
            raw_value=data.get("raw_value"),
            normalized_value=data.get("normalized_value"),
            confidence=data.get("confidence", 0.0),
            description=data.get("description", ""),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Signal Summary (Read-Only Aggregation)
# -----------------------------------------------------------------------------
@dataclass
class SignalSummary:
    """
    Summary of signals for a time window.

    This is a READ-ONLY aggregation, never stored.
    """
    generated_at: str
    time_window_start: str
    time_window_end: str
    total_signals: int
    by_severity: Dict[str, int]  # severity -> count
    by_type: Dict[str, int]  # signal_type -> count
    by_source: Dict[str, int]  # source -> count
    unknown_count: int
    last_signal_timestamp: Optional[str]
    observability_status: str  # "healthy", "partial", "degraded"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Signal Collector (Read-Only Data Sources)
# -----------------------------------------------------------------------------
class SignalCollector:
    """
    Collects signals from various sources.

    CRITICAL: All collection is READ-ONLY.
    If a source is unavailable, emit UNKNOWN severity, not an error.
    """

    def __init__(self):
        self._signal_counter = 0
        self._lock = threading.Lock()

    def _generate_signal_id(self) -> str:
        """Generate unique signal ID."""
        with self._lock:
            self._signal_counter += 1
            return f"sig-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{self._signal_counter:06d}"

    def _create_unknown_signal(
        self,
        signal_type: SignalType,
        source: SignalSource,
        reason: str,
    ) -> RuntimeSignal:
        """Create an UNKNOWN severity signal when data is unavailable."""
        return RuntimeSignal(
            signal_id=self._generate_signal_id(),
            timestamp=datetime.utcnow().isoformat(),
            signal_type=signal_type.value,
            severity=Severity.UNKNOWN.value,
            source=source.value,
            project_id=None,
            aspect=None,
            environment=SignalEnvironment.NONE.value,
            raw_value=None,
            normalized_value=None,
            confidence=0.0,
            description=f"Data unavailable: {reason}",
            metadata={"reason": reason},
        )

    # -------------------------------------------------------------------------
    # System Resource Signals
    # -------------------------------------------------------------------------

    def collect_system_signals(self) -> List[RuntimeSignal]:
        """
        Collect system resource signals.

        If psutil is unavailable or fails, emit UNKNOWN signals.
        """
        signals = []
        now = datetime.utcnow().isoformat()

        # CPU Usage
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            severity = self._classify_cpu_severity(cpu_percent)
            signals.append(RuntimeSignal(
                signal_id=self._generate_signal_id(),
                timestamp=now,
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity=severity.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=cpu_percent,
                normalized_value=cpu_percent / 100.0,
                confidence=1.0,
                description=f"CPU usage: {cpu_percent:.1f}%",
                metadata={"metric": "cpu_percent"},
            ))
        except Exception as e:
            logger.warning(f"CPU signal collection failed: {e}")
            signals.append(self._create_unknown_signal(
                SignalType.SYSTEM_RESOURCE,
                SignalSource.SYSTEM,
                f"CPU metrics unavailable: {str(e)[:100]}",
            ))

        # Memory Usage
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            severity = self._classify_memory_severity(memory_percent)
            signals.append(RuntimeSignal(
                signal_id=self._generate_signal_id(),
                timestamp=now,
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity=severity.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=memory_percent,
                normalized_value=memory_percent / 100.0,
                confidence=1.0,
                description=f"Memory usage: {memory_percent:.1f}%",
                metadata={
                    "metric": "memory_percent",
                    "available_mb": memory.available / (1024 * 1024),
                },
            ))
        except Exception as e:
            logger.warning(f"Memory signal collection failed: {e}")
            signals.append(self._create_unknown_signal(
                SignalType.SYSTEM_RESOURCE,
                SignalSource.SYSTEM,
                f"Memory metrics unavailable: {str(e)[:100]}",
            ))

        # Disk Usage
        try:
            disk = psutil.disk_usage("/")
            disk_percent = disk.percent
            severity = self._classify_disk_severity(disk_percent)
            signals.append(RuntimeSignal(
                signal_id=self._generate_signal_id(),
                timestamp=now,
                signal_type=SignalType.SYSTEM_RESOURCE.value,
                severity=severity.value,
                source=SignalSource.SYSTEM.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=disk_percent,
                normalized_value=disk_percent / 100.0,
                confidence=1.0,
                description=f"Disk usage: {disk_percent:.1f}%",
                metadata={
                    "metric": "disk_percent",
                    "free_gb": disk.free / (1024 * 1024 * 1024),
                },
            ))
        except Exception as e:
            logger.warning(f"Disk signal collection failed: {e}")
            signals.append(self._create_unknown_signal(
                SignalType.SYSTEM_RESOURCE,
                SignalSource.SYSTEM,
                f"Disk metrics unavailable: {str(e)[:100]}",
            ))

        return signals

    def _classify_cpu_severity(self, cpu_percent: float) -> Severity:
        """Deterministic CPU severity classification."""
        if cpu_percent >= 95:
            return Severity.CRITICAL
        elif cpu_percent >= 85:
            return Severity.DEGRADED
        elif cpu_percent >= 70:
            return Severity.WARNING
        else:
            return Severity.INFO

    def _classify_memory_severity(self, memory_percent: float) -> Severity:
        """Deterministic memory severity classification."""
        if memory_percent >= 95:
            return Severity.CRITICAL
        elif memory_percent >= 85:
            return Severity.DEGRADED
        elif memory_percent >= 75:
            return Severity.WARNING
        else:
            return Severity.INFO

    def _classify_disk_severity(self, disk_percent: float) -> Severity:
        """Deterministic disk severity classification."""
        if disk_percent >= 95:
            return Severity.CRITICAL
        elif disk_percent >= 90:
            return Severity.DEGRADED
        elif disk_percent >= 80:
            return Severity.WARNING
        else:
            return Severity.INFO

    # -------------------------------------------------------------------------
    # Worker Queue Signals
    # -------------------------------------------------------------------------

    def collect_worker_signals(self) -> List[RuntimeSignal]:
        """
        Collect worker queue signals.

        Reads from job state without modifying it.
        """
        signals = []
        now = datetime.utcnow().isoformat()

        try:
            # Try to read job state file
            job_state_file = Path("/home/aitesting.mybd.in/jobs/job_state.json")
            if not job_state_file.exists():
                # Try fallback
                job_state_file = Path("/tmp/job_state.json")

            if job_state_file.exists():
                with open(job_state_file) as f:
                    job_state = json.load(f)

                jobs = job_state.get("jobs", {})
                queued_count = sum(1 for j in jobs.values() if j.get("state") == "queued")
                running_count = sum(1 for j in jobs.values() if j.get("state") == "running")
                failed_recent = sum(
                    1 for j in jobs.values()
                    if j.get("state") == "failed" and
                    self._is_recent(j.get("completed_at"), hours=1)
                )

                # Queue saturation signal
                max_concurrent = 3  # From Phase 14.10
                queue_saturation = (queued_count + running_count) / max_concurrent
                severity = self._classify_queue_severity(queue_saturation, queued_count)

                signals.append(RuntimeSignal(
                    signal_id=self._generate_signal_id(),
                    timestamp=now,
                    signal_type=SignalType.WORKER_QUEUE.value,
                    severity=severity.value,
                    source=SignalSource.CONTROLLER.value,
                    project_id=None,
                    aspect=None,
                    environment=SignalEnvironment.NONE.value,
                    raw_value={"queued": queued_count, "running": running_count},
                    normalized_value=min(1.0, queue_saturation),
                    confidence=1.0,
                    description=f"Worker queue: {queued_count} queued, {running_count} running",
                    metadata={
                        "queued_count": queued_count,
                        "running_count": running_count,
                        "failed_last_hour": failed_recent,
                        "saturation": queue_saturation,
                    },
                ))

                # Job failure signal if failures detected
                if failed_recent > 0:
                    failure_severity = self._classify_failure_severity(failed_recent)
                    signals.append(RuntimeSignal(
                        signal_id=self._generate_signal_id(),
                        timestamp=now,
                        signal_type=SignalType.JOB_FAILURE.value,
                        severity=failure_severity.value,
                        source=SignalSource.CLAUDE.value,
                        project_id=None,
                        aspect=None,
                        environment=SignalEnvironment.NONE.value,
                        raw_value=failed_recent,
                        normalized_value=min(1.0, failed_recent / 10.0),
                        confidence=1.0,
                        description=f"{failed_recent} job failures in last hour",
                        metadata={"failed_count": failed_recent, "window_hours": 1},
                    ))

            else:
                # Job state file not found - emit UNKNOWN
                signals.append(self._create_unknown_signal(
                    SignalType.WORKER_QUEUE,
                    SignalSource.CONTROLLER,
                    "Job state file not found",
                ))

        except Exception as e:
            logger.warning(f"Worker signal collection failed: {e}")
            signals.append(self._create_unknown_signal(
                SignalType.WORKER_QUEUE,
                SignalSource.CONTROLLER,
                f"Worker metrics unavailable: {str(e)[:100]}",
            ))

        return signals

    def _is_recent(self, timestamp_str: Optional[str], hours: int) -> bool:
        """Check if timestamp is within the last N hours."""
        if not timestamp_str:
            return False
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            return ts.replace(tzinfo=None) > cutoff
        except Exception:
            return False

    def _classify_queue_severity(self, saturation: float, queued: int) -> Severity:
        """Deterministic queue severity classification."""
        if saturation >= 3.0 or queued >= 10:
            return Severity.CRITICAL
        elif saturation >= 2.0 or queued >= 5:
            return Severity.DEGRADED
        elif saturation >= 1.0 or queued >= 3:
            return Severity.WARNING
        else:
            return Severity.INFO

    def _classify_failure_severity(self, failure_count: int) -> Severity:
        """Deterministic failure severity classification."""
        if failure_count >= 5:
            return Severity.CRITICAL
        elif failure_count >= 3:
            return Severity.DEGRADED
        elif failure_count >= 1:
            return Severity.WARNING
        else:
            return Severity.INFO

    # -------------------------------------------------------------------------
    # Lifecycle Signals
    # -------------------------------------------------------------------------

    def collect_lifecycle_signals(self) -> List[RuntimeSignal]:
        """
        Collect lifecycle-related signals.

        Detects patterns like repeated drift confirmations or excessive loops.
        """
        signals = []
        now = datetime.utcnow().isoformat()

        try:
            # Try to read lifecycle state
            lifecycle_file = Path("/home/aitesting.mybd.in/jobs/lifecycle/lifecycles.json")
            if not lifecycle_file.exists():
                lifecycle_file = Path("/tmp/lifecycle/lifecycles.json")

            if lifecycle_file.exists():
                with open(lifecycle_file) as f:
                    lifecycle_data = json.load(f)

                lifecycles = lifecycle_data.get("lifecycles", {})

                # Count lifecycle patterns
                change_mode_count = sum(
                    1 for lc in lifecycles.values()
                    if lc.get("mode") == "change"
                )
                high_cycle_count = sum(
                    1 for lc in lifecycles.values()
                    if lc.get("cycle_number", 0) > 5
                )

                # Emit signal if excessive change cycles detected
                if high_cycle_count > 0:
                    severity = Severity.WARNING if high_cycle_count < 3 else Severity.DEGRADED
                    signals.append(RuntimeSignal(
                        signal_id=self._generate_signal_id(),
                        timestamp=now,
                        signal_type=SignalType.DRIFT_WARNING.value,
                        severity=severity.value,
                        source=SignalSource.LIFECYCLE.value,
                        project_id=None,
                        aspect=None,
                        environment=SignalEnvironment.NONE.value,
                        raw_value=high_cycle_count,
                        normalized_value=min(1.0, high_cycle_count / 10.0),
                        confidence=1.0,
                        description=f"{high_cycle_count} lifecycles with >5 change cycles",
                        metadata={
                            "high_cycle_count": high_cycle_count,
                            "change_mode_count": change_mode_count,
                        },
                    ))
                else:
                    # Healthy lifecycle state
                    signals.append(RuntimeSignal(
                        signal_id=self._generate_signal_id(),
                        timestamp=now,
                        signal_type=SignalType.DRIFT_WARNING.value,
                        severity=Severity.INFO.value,
                        source=SignalSource.LIFECYCLE.value,
                        project_id=None,
                        aspect=None,
                        environment=SignalEnvironment.NONE.value,
                        raw_value=0,
                        normalized_value=0.0,
                        confidence=1.0,
                        description="Lifecycle patterns within normal range",
                        metadata={
                            "total_lifecycles": len(lifecycles),
                            "change_mode_count": change_mode_count,
                        },
                    ))

            else:
                # Lifecycle file not found - emit UNKNOWN
                signals.append(self._create_unknown_signal(
                    SignalType.DRIFT_WARNING,
                    SignalSource.LIFECYCLE,
                    "Lifecycle state file not found",
                ))

        except Exception as e:
            logger.warning(f"Lifecycle signal collection failed: {e}")
            signals.append(self._create_unknown_signal(
                SignalType.DRIFT_WARNING,
                SignalSource.LIFECYCLE,
                f"Lifecycle metrics unavailable: {str(e)[:100]}",
            ))

        return signals

    # -------------------------------------------------------------------------
    # Drift Contract Signals
    # -------------------------------------------------------------------------

    def collect_drift_signals(self) -> List[RuntimeSignal]:
        """
        Collect intent drift-related signals.

        Reads from contract audit logs without modifying them.
        """
        signals = []
        now = datetime.utcnow().isoformat()

        try:
            # Try to read contract audit log
            audit_file = Path("/home/aitesting.mybd.in/jobs/intent_contracts/contract_audit.log")
            if not audit_file.exists():
                audit_file = Path("/tmp/intent_contracts/contract_audit.log")

            if audit_file.exists():
                # Read last 100 lines to check recent activity
                recent_entries = []
                with open(audit_file) as f:
                    lines = f.readlines()[-100:]
                    for line in lines:
                        try:
                            entry = json.loads(line.strip())
                            if self._is_recent(entry.get("timestamp"), hours=24):
                                recent_entries.append(entry)
                        except json.JSONDecodeError:
                            continue

                # Count confirmation requests
                confirmation_count = sum(
                    1 for e in recent_entries
                    if e.get("action") == "CONFIRMATION_REQUESTED"
                )
                block_count = sum(
                    1 for e in recent_entries
                    if e.get("action") == "CONTRACT_EVALUATED" and
                    not e.get("can_proceed", True)
                )

                if confirmation_count > 0 or block_count > 0:
                    severity = Severity.INFO
                    if confirmation_count >= 5 or block_count >= 3:
                        severity = Severity.DEGRADED
                    elif confirmation_count >= 3 or block_count >= 1:
                        severity = Severity.WARNING

                    signals.append(RuntimeSignal(
                        signal_id=self._generate_signal_id(),
                        timestamp=now,
                        signal_type=SignalType.DRIFT_WARNING.value,
                        severity=severity.value,
                        source=SignalSource.CONTROLLER.value,
                        project_id=None,
                        aspect=None,
                        environment=SignalEnvironment.NONE.value,
                        raw_value={
                            "confirmations": confirmation_count,
                            "blocks": block_count,
                        },
                        normalized_value=min(1.0, (confirmation_count + block_count) / 10.0),
                        confidence=1.0,
                        description=f"Drift activity: {confirmation_count} confirmations, {block_count} blocks (24h)",
                        metadata={
                            "confirmation_count": confirmation_count,
                            "block_count": block_count,
                            "window_hours": 24,
                        },
                    ))
                else:
                    # No drift activity - healthy
                    signals.append(RuntimeSignal(
                        signal_id=self._generate_signal_id(),
                        timestamp=now,
                        signal_type=SignalType.DRIFT_WARNING.value,
                        severity=Severity.INFO.value,
                        source=SignalSource.CONTROLLER.value,
                        project_id=None,
                        aspect=None,
                        environment=SignalEnvironment.NONE.value,
                        raw_value={"confirmations": 0, "blocks": 0},
                        normalized_value=0.0,
                        confidence=1.0,
                        description="No drift activity in last 24 hours",
                        metadata={"window_hours": 24},
                    ))

            else:
                # Audit file not found - could be new system
                signals.append(RuntimeSignal(
                    signal_id=self._generate_signal_id(),
                    timestamp=now,
                    signal_type=SignalType.DRIFT_WARNING.value,
                    severity=Severity.INFO.value,
                    source=SignalSource.CONTROLLER.value,
                    project_id=None,
                    aspect=None,
                    environment=SignalEnvironment.NONE.value,
                    raw_value=None,
                    normalized_value=0.0,
                    confidence=0.5,  # Lower confidence without audit history
                    description="No drift audit history (new system or missing file)",
                    metadata={"reason": "audit_file_not_found"},
                ))

        except Exception as e:
            logger.warning(f"Drift signal collection failed: {e}")
            signals.append(self._create_unknown_signal(
                SignalType.DRIFT_WARNING,
                SignalSource.CONTROLLER,
                f"Drift metrics unavailable: {str(e)[:100]}",
            ))

        return signals

    # -------------------------------------------------------------------------
    # Phase 22: Deployment Failure Signals
    # -------------------------------------------------------------------------

    def emit_deployment_failure_signal(
        self,
        project_name: str,
        failure_type: str,
        failed_endpoints: List[Dict[str, Any]],
        rescue_attempt: int,
        max_attempts: int = 3,
        environment: str = "testing"
    ) -> RuntimeSignal:
        """
        Emit a signal for deployment validation failure.

        Phase 22: Rescue & Recovery System signal emission.

        This is called when post-deployment validation fails, allowing
        the runtime intelligence system to track deployment health.

        Args:
            project_name: Name of the project
            failure_type: Type of failure (HTTP_404, HTTP_500, etc.)
            failed_endpoints: List of endpoints that failed validation
            rescue_attempt: Current rescue attempt number
            max_attempts: Maximum allowed rescue attempts
            environment: Deployment environment (testing, production)

        Returns:
            The emitted RuntimeSignal
        """
        now = datetime.utcnow().isoformat()

        # Determine severity based on rescue attempts
        if rescue_attempt >= max_attempts:
            severity = Severity.CRITICAL
        elif rescue_attempt >= 2:
            severity = Severity.DEGRADED
        else:
            severity = Severity.WARNING

        # Create the signal
        signal = RuntimeSignal(
            signal_id=self._generate_signal_id(),
            timestamp=now,
            signal_type=SignalType.DEPLOYMENT_FAILURE.value,
            severity=severity.value,
            source=SignalSource.CONTROLLER.value,
            project_id=project_name,
            aspect=None,
            environment=environment,
            raw_value={
                "failure_type": failure_type,
                "failed_endpoint_count": len(failed_endpoints),
                "rescue_attempt": rescue_attempt,
                "max_attempts": max_attempts,
            },
            normalized_value=min(1.0, rescue_attempt / max_attempts),
            confidence=1.0,
            description=(
                f"Deployment validation failed for {project_name}: "
                f"{failure_type} ({len(failed_endpoints)} endpoints, "
                f"attempt {rescue_attempt}/{max_attempts})"
            ),
            metadata={
                "failure_type": failure_type,
                "failed_endpoints": [
                    {"url": e.get("url"), "error": e.get("error")}
                    for e in failed_endpoints[:10]  # Limit for storage
                ],
                "rescue_attempt": rescue_attempt,
                "max_attempts": max_attempts,
                "environment": environment,
            },
        )

        logger.info(
            f"Emitted deployment failure signal for {project_name}: "
            f"{failure_type} (attempt {rescue_attempt}/{max_attempts})"
        )

        return signal

    def emit_deployment_success_signal(
        self,
        project_name: str,
        validated_endpoints: List[str],
        was_rescue: bool = False,
        rescue_attempt: Optional[int] = None,
        environment: str = "testing"
    ) -> RuntimeSignal:
        """
        Emit a signal for successful deployment validation.

        Phase 22: Rescue & Recovery System success signal.

        Args:
            project_name: Name of the project
            validated_endpoints: List of successfully validated endpoints
            was_rescue: Whether this was after a rescue attempt
            rescue_attempt: Which rescue attempt succeeded (if applicable)
            environment: Deployment environment

        Returns:
            The emitted RuntimeSignal
        """
        now = datetime.utcnow().isoformat()

        description = f"Deployment validation passed for {project_name}"
        if was_rescue and rescue_attempt:
            description += f" (rescued on attempt {rescue_attempt})"

        signal = RuntimeSignal(
            signal_id=self._generate_signal_id(),
            timestamp=now,
            signal_type=SignalType.DEPLOYMENT_FAILURE.value,  # Same type, info severity
            severity=Severity.INFO.value,
            source=SignalSource.CONTROLLER.value,
            project_id=project_name,
            aspect=None,
            environment=environment,
            raw_value={
                "success": True,
                "endpoint_count": len(validated_endpoints),
                "was_rescue": was_rescue,
                "rescue_attempt": rescue_attempt,
            },
            normalized_value=0.0,  # No failure = 0
            confidence=1.0,
            description=description,
            metadata={
                "validated_endpoints": validated_endpoints[:10],
                "was_rescue": was_rescue,
                "rescue_attempt": rescue_attempt,
                "environment": environment,
            },
        )

        logger.info(f"Emitted deployment success signal for {project_name}")

        return signal

    # -------------------------------------------------------------------------
    # Collect All Signals
    # -------------------------------------------------------------------------

    def collect_all(self) -> List[RuntimeSignal]:
        """
        Collect all signals from all sources.

        CRITICAL: This method is READ-ONLY. It never modifies system state.
        """
        all_signals = []

        # System resources
        all_signals.extend(self.collect_system_signals())

        # Worker queue
        all_signals.extend(self.collect_worker_signals())

        # Lifecycle patterns
        all_signals.extend(self.collect_lifecycle_signals())

        # Drift activity
        all_signals.extend(self.collect_drift_signals())

        return all_signals


# -----------------------------------------------------------------------------
# Signal Persister (Append-Only)
# -----------------------------------------------------------------------------
class SignalPersister:
    """
    Persists signals to append-only storage.

    CRITICAL: Signals are NEVER deleted or modified.
    """

    def __init__(self, signals_file: Optional[Path] = None):
        self._signals_file = signals_file or SIGNALS_FILE
        self._lock = threading.Lock()

    def persist(self, signals: List[RuntimeSignal]) -> int:
        """
        Persist signals to storage.

        Returns: Number of signals persisted
        """
        if not signals:
            return 0

        with self._lock:
            try:
                # Ensure directory exists
                self._signals_file.parent.mkdir(parents=True, exist_ok=True)

                # Append signals (never overwrite)
                with open(self._signals_file, "a") as f:
                    for signal in signals:
                        f.write(json.dumps(signal.to_dict()) + "\n")
                    # fsync for durability
                    f.flush()
                    os.fsync(f.fileno())

                return len(signals)

            except Exception as e:
                logger.error(f"Signal persistence failed: {e}")
                return 0

    def read_signals(
        self,
        since: Optional[datetime] = None,
        project_id: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
        severity: Optional[Severity] = None,
        limit: int = 1000,
    ) -> List[RuntimeSignal]:
        """
        Read signals from storage with optional filters.

        This is a READ-ONLY operation.
        """
        signals = []

        if not self._signals_file.exists():
            return signals

        try:
            with open(self._signals_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        signal = RuntimeSignal.from_dict(data)

                        # Apply filters
                        if since:
                            signal_ts = datetime.fromisoformat(signal.timestamp)
                            if signal_ts < since:
                                continue

                        if project_id and signal.project_id != project_id:
                            continue

                        if signal_type and signal.signal_type != signal_type.value:
                            continue

                        if severity and signal.severity != severity.value:
                            continue

                        signals.append(signal)

                        if len(signals) >= limit:
                            break

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed signal: {e}")
                        continue

        except Exception as e:
            logger.error(f"Signal read failed: {e}")

        return signals

    def get_summary(
        self,
        since: Optional[datetime] = None,
    ) -> SignalSummary:
        """
        Generate a summary of signals.

        This is a READ-ONLY aggregation.
        """
        now = datetime.utcnow()
        if since is None:
            since = now - timedelta(hours=24)

        signals = self.read_signals(since=since, limit=10000)

        # Aggregate counts
        by_severity = {}
        by_type = {}
        by_source = {}
        unknown_count = 0
        last_timestamp = None

        for signal in signals:
            # By severity
            by_severity[signal.severity] = by_severity.get(signal.severity, 0) + 1

            # By type
            by_type[signal.signal_type] = by_type.get(signal.signal_type, 0) + 1

            # By source
            by_source[signal.source] = by_source.get(signal.source, 0) + 1

            # Unknown count
            if signal.severity == Severity.UNKNOWN.value:
                unknown_count += 1

            # Track last timestamp
            if last_timestamp is None or signal.timestamp > last_timestamp:
                last_timestamp = signal.timestamp

        # Determine observability status
        if unknown_count == 0 and len(signals) > 0:
            status = "healthy"
        elif unknown_count > 0 and unknown_count < len(signals) / 2:
            status = "partial"
        elif len(signals) == 0:
            status = "no_data"
        else:
            status = "degraded"

        return SignalSummary(
            generated_at=now.isoformat(),
            time_window_start=since.isoformat(),
            time_window_end=now.isoformat(),
            total_signals=len(signals),
            by_severity=by_severity,
            by_type=by_type,
            by_source=by_source,
            unknown_count=unknown_count,
            last_signal_timestamp=last_timestamp,
            observability_status=status,
        )


# -----------------------------------------------------------------------------
# Runtime Intelligence Engine
# -----------------------------------------------------------------------------
class RuntimeIntelligenceEngine:
    """
    Main engine for runtime intelligence.

    CRITICAL CONSTRAINTS:
    - READ-ONLY: No lifecycle transitions, no deployments
    - DETERMINISTIC: No ML, no probabilistic inference
    - EXPLICIT UNKNOWN: Missing data = UNKNOWN severity
    - APPEND-ONLY: Signals are never deleted
    """

    def __init__(
        self,
        poll_interval: int = DEFAULT_POLL_INTERVAL_SECONDS,
        signals_file: Optional[Path] = None,
    ):
        self._collector = SignalCollector()
        self._persister = SignalPersister(signals_file)
        self._poll_interval = poll_interval
        self._running = False
        self._poll_thread: Optional[threading.Thread] = None
        self._last_poll_timestamp: Optional[str] = None
        self._poll_count = 0

    def poll_once(self) -> Tuple[int, List[RuntimeSignal]]:
        """
        Perform a single poll cycle.

        Returns: (signals_persisted, signals)
        """
        try:
            # Collect all signals
            signals = self._collector.collect_all()

            # Persist signals
            persisted = self._persister.persist(signals)

            # Update state
            self._last_poll_timestamp = datetime.utcnow().isoformat()
            self._poll_count += 1

            # Audit log the poll
            self._log_poll_audit(len(signals), persisted)

            logger.debug(f"Poll cycle complete: {len(signals)} collected, {persisted} persisted")

            return persisted, signals

        except Exception as e:
            logger.error(f"Poll cycle failed: {e}")
            # Emit an UNKNOWN signal about the poll failure
            failure_signal = RuntimeSignal(
                signal_id=f"poll-failure-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                timestamp=datetime.utcnow().isoformat(),
                signal_type=SignalType.CONFIG_ANOMALY.value,
                severity=Severity.UNKNOWN.value,
                source=SignalSource.CONTROLLER.value,
                project_id=None,
                aspect=None,
                environment=SignalEnvironment.NONE.value,
                raw_value=str(e)[:200],
                normalized_value=None,
                confidence=0.0,
                description=f"Poll cycle failed: {str(e)[:100]}",
                metadata={"error": str(e)[:500]},
            )
            self._persister.persist([failure_signal])
            return 0, []

    def _log_poll_audit(self, collected: int, persisted: int) -> None:
        """Log poll cycle to audit trail."""
        try:
            POLL_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": "POLL_CYCLE",
                "collected": collected,
                "persisted": persisted,
                "poll_count": self._poll_count,
            }
            with open(POLL_AUDIT_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Poll audit log failed: {e}")

    def start_polling(self) -> None:
        """
        Start background polling.

        CRITICAL: Polling is READ-ONLY. It only collects and persists signals.
        """
        if self._running:
            logger.warning("Polling already running")
            return

        self._running = True
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        logger.info(f"Runtime intelligence polling started (interval={self._poll_interval}s)")

    def stop_polling(self) -> None:
        """Stop background polling."""
        self._running = False
        if self._poll_thread:
            self._poll_thread.join(timeout=5)
        logger.info("Runtime intelligence polling stopped")

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            self.poll_once()
            time.sleep(self._poll_interval)

    def get_signals(
        self,
        since: Optional[datetime] = None,
        project_id: Optional[str] = None,
        signal_type: Optional[SignalType] = None,
        severity: Optional[Severity] = None,
        limit: int = 100,
    ) -> List[RuntimeSignal]:
        """
        Get signals with optional filters.

        This is a READ-ONLY operation.
        """
        return self._persister.read_signals(
            since=since,
            project_id=project_id,
            signal_type=signal_type,
            severity=severity,
            limit=limit,
        )

    def get_summary(
        self,
        since: Optional[datetime] = None,
    ) -> SignalSummary:
        """
        Get signal summary.

        This is a READ-ONLY aggregation.
        """
        return self._persister.get_summary(since=since)

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "running": self._running,
            "poll_interval_seconds": self._poll_interval,
            "last_poll_timestamp": self._last_poll_timestamp,
            "poll_count": self._poll_count,
            "signals_file": str(self._persister._signals_file),
        }


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------
_engine: Optional[RuntimeIntelligenceEngine] = None


def get_runtime_engine() -> RuntimeIntelligenceEngine:
    """Get the global runtime intelligence engine."""
    global _engine
    if _engine is None:
        _engine = RuntimeIntelligenceEngine()
    return _engine


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
def poll_signals() -> Tuple[int, List[RuntimeSignal]]:
    """Perform a single poll cycle."""
    return get_runtime_engine().poll_once()


def get_signals(
    since: Optional[datetime] = None,
    project_id: Optional[str] = None,
    signal_type: Optional[SignalType] = None,
    severity: Optional[Severity] = None,
    limit: int = 100,
) -> List[RuntimeSignal]:
    """Get signals with optional filters."""
    return get_runtime_engine().get_signals(
        since=since,
        project_id=project_id,
        signal_type=signal_type,
        severity=severity,
        limit=limit,
    )


def get_signal_summary(since: Optional[datetime] = None) -> SignalSummary:
    """Get signal summary."""
    return get_runtime_engine().get_summary(since=since)


def start_signal_polling() -> None:
    """Start background signal polling."""
    get_runtime_engine().start_polling()


def stop_signal_polling() -> None:
    """Stop background signal polling."""
    get_runtime_engine().stop_polling()


logger.info("Runtime Intelligence module loaded (Phase 17A - OBSERVATION ONLY)")
