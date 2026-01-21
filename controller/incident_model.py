"""
Phase 17B: Incident Model & Classification Enums

This module defines the data structures for incident classification.
All structures are IMMUTABLE and OBSERVATION-ONLY.

CRITICAL CONSTRAINTS:
- READ-ONLY: No lifecycle changes, no deployments, no mutations
- DETERMINISTIC: No ML, no probabilistic inference
- EXPLICIT UNKNOWN: Missing data = UNKNOWN, never guessed
- APPEND-ONLY: Incidents are never deleted or modified
- NO ALERTS: Incidents are classified, not alerted
- NO RECOMMENDATIONS: Classification only, no suggested actions

This phase interprets signals, NOT takes action on them.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, FrozenSet


# -----------------------------------------------------------------------------
# Incident Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class IncidentType(str, Enum):
    """
    Types of incidents derived from signal correlation.

    This enum is LOCKED - do not add types without explicit approval.
    """
    PERFORMANCE = "performance"  # System slowdowns, high latency, resource contention
    RELIABILITY = "reliability"  # Service failures, job failures, test failures
    SECURITY = "security"  # Security-related signals, gate denials
    GOVERNANCE = "governance"  # Drift violations, contract breaches, human overrides
    RESOURCE = "resource"  # Resource exhaustion, disk/memory/CPU issues
    CONFIGURATION = "configuration"  # Config anomalies, misconfigurations
    UNKNOWN = "unknown"  # Cannot classify - data insufficient


# -----------------------------------------------------------------------------
# Incident Severity Enum (LOCKED)
# -----------------------------------------------------------------------------
class IncidentSeverity(str, Enum):
    """
    Incident severity levels.

    CRITICAL: Missing data or classification failure MUST produce UNKNOWN, not failure.
    """
    INFO = "info"  # Informational, no impact
    LOW = "low"  # Minor impact, no immediate action needed
    MEDIUM = "medium"  # Moderate impact, should be investigated
    HIGH = "high"  # Significant impact, requires attention
    CRITICAL = "critical"  # Severe impact, urgent attention needed
    UNKNOWN = "unknown"  # MANDATORY when data is insufficient


# -----------------------------------------------------------------------------
# Incident Scope Enum (LOCKED)
# -----------------------------------------------------------------------------
class IncidentScope(str, Enum):
    """
    Scope of the incident - where does it apply?
    """
    SYSTEM = "system"  # Affects entire system
    PROJECT = "project"  # Affects specific project
    PROJECT_ASPECT = "project_aspect"  # Affects specific project aspect
    JOB = "job"  # Affects specific job
    UNKNOWN = "unknown"  # Cannot determine scope


# -----------------------------------------------------------------------------
# Incident State Enum
# -----------------------------------------------------------------------------
class IncidentState(str, Enum):
    """
    State of the incident.

    Note: This is for CLASSIFICATION purposes only.
    Incidents are NEVER mutated - state transitions create new incidents.
    """
    OPEN = "open"  # Newly detected, signals still occurring
    CLOSED = "closed"  # No new signals in correlation window
    UNKNOWN = "unknown"  # Cannot determine state


# -----------------------------------------------------------------------------
# Correlation Window Constants
# -----------------------------------------------------------------------------
# Fixed time windows for signal correlation (in minutes)
CORRELATION_WINDOW_MINUTES = 15  # Signals within 15 min are correlated
MIN_SIGNALS_FOR_INCIDENT = 1  # Minimum signals to create incident
MAX_SIGNALS_PER_INCIDENT = 100  # Cap to prevent unbounded incidents


# -----------------------------------------------------------------------------
# Incident (Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Incident:
    """
    Immutable incident record.

    Once created, an incident CANNOT be modified.
    This ensures audit integrity and deterministic replay.

    Incidents are CORRELATED from signals, not manually created.
    """
    incident_id: str
    created_at: str  # ISO format
    incident_type: str  # IncidentType value
    severity: str  # IncidentSeverity value
    scope: str  # IncidentScope value
    state: str  # IncidentState value
    title: str  # Short descriptive title
    description: str  # Detailed description
    source_signal_ids: tuple  # Tuple of signal IDs (immutable)
    signal_count: int  # Number of correlated signals
    first_signal_at: str  # ISO format - when first signal occurred
    last_signal_at: str  # ISO format - when last signal occurred
    correlation_window_minutes: int  # Window used for correlation
    project_id: Optional[str]  # If project-scoped
    aspect: Optional[str]  # If aspect-scoped
    job_id: Optional[str]  # If job-scoped
    confidence: float  # 0.0 - 1.0, deterministic
    classification_rule: str  # Which rule classified this
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate incident on creation."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.severity not in [s.value for s in IncidentSeverity]:
            raise ValueError(f"Invalid severity: {self.severity}")
        if self.incident_type not in [t.value for t in IncidentType]:
            raise ValueError(f"Invalid incident type: {self.incident_type}")
        if self.scope not in [s.value for s in IncidentScope]:
            raise ValueError(f"Invalid scope: {self.scope}")
        if self.state not in [s.value for s in IncidentState]:
            raise ValueError(f"Invalid state: {self.state}")
        if self.signal_count < 0:
            raise ValueError(f"Signal count cannot be negative: {self.signal_count}")
        if not isinstance(self.source_signal_ids, tuple):
            raise ValueError("source_signal_ids must be a tuple for immutability")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert tuple to list for JSON serialization
        result["source_signal_ids"] = list(self.source_signal_ids)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Incident":
        """Create incident from dictionary."""
        # Convert list back to tuple
        signal_ids = data.get("source_signal_ids", [])
        if isinstance(signal_ids, list):
            signal_ids = tuple(signal_ids)

        return cls(
            incident_id=data["incident_id"],
            created_at=data["created_at"],
            incident_type=data["incident_type"],
            severity=data["severity"],
            scope=data["scope"],
            state=data["state"],
            title=data["title"],
            description=data["description"],
            source_signal_ids=signal_ids,
            signal_count=data["signal_count"],
            first_signal_at=data["first_signal_at"],
            last_signal_at=data["last_signal_at"],
            correlation_window_minutes=data.get("correlation_window_minutes", CORRELATION_WINDOW_MINUTES),
            project_id=data.get("project_id"),
            aspect=data.get("aspect"),
            job_id=data.get("job_id"),
            confidence=data.get("confidence", 0.0),
            classification_rule=data.get("classification_rule", "unknown"),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Incident Summary (Read-Only Aggregation)
# -----------------------------------------------------------------------------
@dataclass
class IncidentSummary:
    """
    Summary of incidents for a time window.

    This is a READ-ONLY aggregation, never stored.
    """
    generated_at: str
    time_window_start: str
    time_window_end: str
    total_incidents: int
    by_severity: Dict[str, int]  # severity -> count
    by_type: Dict[str, int]  # incident_type -> count
    by_scope: Dict[str, int]  # scope -> count
    by_state: Dict[str, int]  # state -> count
    unknown_count: int
    open_count: int
    recent_incidents: List[Dict[str, Any]]  # Summarized recent incidents

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Classification Rule (Read-Only)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ClassificationRule:
    """
    A deterministic rule for incident classification.

    Rules are FIXED and DETERMINISTIC - same input always produces same output.
    """
    rule_id: str
    name: str
    description: str
    signal_types: FrozenSet[str]  # Which signal types trigger this rule
    min_severity: str  # Minimum signal severity to trigger
    incident_type: str  # Resulting incident type
    scope_derivation: str  # How to derive scope (from_signal, system, etc.)
    confidence: float  # Confidence of this classification (0.0-1.0)

    def matches_signal_type(self, signal_type: str) -> bool:
        """Check if this rule matches a signal type."""
        return signal_type in self.signal_types

    def matches_severity(self, severity: str) -> bool:
        """Check if signal severity meets minimum threshold."""
        severity_order = [
            IncidentSeverity.INFO.value,
            IncidentSeverity.LOW.value,
            IncidentSeverity.MEDIUM.value,
            IncidentSeverity.HIGH.value,
            IncidentSeverity.CRITICAL.value,
        ]
        if severity == IncidentSeverity.UNKNOWN.value:
            return True  # UNKNOWN always matches
        if self.min_severity == IncidentSeverity.INFO.value:
            return True  # INFO is minimum, always matches

        try:
            signal_idx = severity_order.index(severity)
            min_idx = severity_order.index(self.min_severity)
            return signal_idx >= min_idx
        except ValueError:
            return False  # Unknown severity value
