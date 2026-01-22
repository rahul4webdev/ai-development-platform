"""
Phase 19: Learning, Memory & System Intelligence - Data Models

Frozen dataclasses and LOCKED enums for learning and pattern observation.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- NO BEHAVIORAL COUPLING: Never influences eligibility, approval, execution, or recommendations
- NO THRESHOLD MODIFICATION: Never changes system thresholds or limits
- NO ML INFERENCE: No machine learning, no optimization, no prediction
- NO AUTOMATION: Never triggers any automated actions
- 100% DETERMINISTIC: Same inputs = same aggregates
- READ-ONLY: Observes history, never changes it
- APPEND-ONLY: Memory is written, never modified

This module provides DATA MODELS for learning.
It is MEMORY, not INTELLIGENCE.
It provides INSIGHT, not ACTION.
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, Tuple


# -----------------------------------------------------------------------------
# Pattern Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class PatternType(str, Enum):
    """
    Types of observed patterns.

    This enum is LOCKED - do not add values without explicit approval.
    These are OBSERVATION categories, not ACTION triggers.
    """
    # Execution patterns
    EXECUTION_FAILURE_CLUSTER = "execution_failure_cluster"
    EXECUTION_SUCCESS_STREAK = "execution_success_streak"
    EXECUTION_BLOCK_FREQUENCY = "execution_block_frequency"

    # Verification patterns
    VERIFICATION_VIOLATION_RECURRING = "verification_violation_recurring"
    VERIFICATION_DOMAIN_HOTSPOT = "verification_domain_hotspot"

    # Approval patterns
    APPROVAL_REJECTION_TREND = "approval_rejection_trend"
    APPROVAL_DELAY_PATTERN = "approval_delay_pattern"

    # Drift patterns
    DRIFT_ESCALATION_TREND = "drift_escalation_trend"
    DRIFT_AREA_CONCENTRATION = "drift_area_concentration"

    # Incident patterns
    INCIDENT_RECURRENCE = "incident_recurrence"
    INCIDENT_SEVERITY_TREND = "incident_severity_trend"

    # General patterns
    TIME_BASED_PATTERN = "time_based_pattern"
    PROJECT_CORRELATION = "project_correlation"


# -----------------------------------------------------------------------------
# Trend Direction Enum (LOCKED - EXACTLY 4 VALUES)
# -----------------------------------------------------------------------------
class TrendDirection(str, Enum):
    """
    Direction of observed trends.

    This enum is LOCKED - EXACTLY 4 values.
    """
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


# -----------------------------------------------------------------------------
# Confidence Level Enum (LOCKED - EXACTLY 4 VALUES)
# -----------------------------------------------------------------------------
class ConfidenceLevel(str, Enum):
    """
    Statistical confidence level for patterns.

    This is STATISTICAL confidence, NOT ML confidence.
    Based on sample size and consistency, not model predictions.

    LOCKED - EXACTLY 4 values.
    """
    HIGH = "high"        # N >= 100, consistency >= 90%
    MEDIUM = "medium"    # N >= 30, consistency >= 70%
    LOW = "low"          # N >= 10, consistency >= 50%
    INSUFFICIENT = "insufficient"  # N < 10 or consistency < 50%


# -----------------------------------------------------------------------------
# Aggregate Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class AggregateType(str, Enum):
    """
    Types of historical aggregates.

    LOCKED - these are the ONLY aggregate types allowed.
    """
    FAILURE_RATE = "failure_rate"
    VIOLATION_FREQUENCY = "violation_frequency"
    APPROVAL_REJECTION_RATE = "approval_rejection_rate"
    DRIFT_TREND = "drift_trend"
    EXECUTION_BLOCK_RATE = "execution_block_rate"
    INCIDENT_FREQUENCY = "incident_frequency"
    VERIFICATION_PASS_RATE = "verification_pass_rate"


# -----------------------------------------------------------------------------
# Memory Entry Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class MemoryEntryType(str, Enum):
    """
    Types of memory entries.

    LOCKED - these are the ONLY memory entry types allowed.
    """
    EXECUTION_OUTCOME = "execution_outcome"
    VERIFICATION_RESULT = "verification_result"
    APPROVAL_DECISION = "approval_decision"
    INCIDENT_OCCURRENCE = "incident_occurrence"
    DRIFT_DETECTION = "drift_detection"
    PATTERN_OBSERVATION = "pattern_observation"
    AGGREGATE_SNAPSHOT = "aggregate_snapshot"


# -----------------------------------------------------------------------------
# Observed Pattern (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ObservedPattern:
    """
    An observed pattern in system history.

    FROZEN: Immutable once created.
    This is an OBSERVATION, not a RECOMMENDATION.
    Patterns are RECORDED, not ACTED UPON.
    """
    pattern_id: str
    pattern_type: str  # PatternType value
    description: str
    frequency: int  # How many times observed
    confidence: str  # ConfidenceLevel value (statistical, NOT ML)
    evidence_ids: Tuple[str, ...]  # IDs of supporting records
    first_observed: str  # ISO format
    last_observed: str  # ISO format
    observed_at: str  # When this pattern was recorded
    project_id: Optional[str]  # If project-specific
    metadata: Tuple[Tuple[str, str], ...]  # Immutable key-value pairs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
            "observed_at": self.observed_at,
            "project_id": self.project_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObservedPattern":
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            metadata = tuple(metadata.items())
        return cls(
            pattern_id=data["pattern_id"],
            pattern_type=data["pattern_type"],
            description=data["description"],
            frequency=data["frequency"],
            confidence=data["confidence"],
            evidence_ids=tuple(data.get("evidence_ids", [])),
            first_observed=data["first_observed"],
            last_observed=data["last_observed"],
            observed_at=data["observed_at"],
            project_id=data.get("project_id"),
            metadata=metadata if isinstance(metadata, tuple) else tuple(metadata.items()),
        )


# -----------------------------------------------------------------------------
# Historical Aggregate (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class HistoricalAggregate:
    """
    A historical aggregate computed from system history.

    FROZEN: Immutable once created.
    This is a STATISTIC, not an INPUT to decisions.
    Aggregates are RECORDED, not USED for automation.
    """
    aggregate_id: str
    aggregate_type: str  # AggregateType value
    value: float  # The computed value (rate, count, etc.)
    sample_size: int  # Number of records in aggregate
    period_start: str  # ISO format
    period_end: str  # ISO format
    computed_at: str  # When this aggregate was computed
    project_id: Optional[str]  # If project-specific
    breakdown: Tuple[Tuple[str, float], ...]  # Sub-breakdowns (immutable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "value": self.value,
            "sample_size": self.sample_size,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "computed_at": self.computed_at,
            "project_id": self.project_id,
            "breakdown": dict(self.breakdown),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoricalAggregate":
        breakdown = data.get("breakdown", {})
        if isinstance(breakdown, dict):
            breakdown = tuple(breakdown.items())
        return cls(
            aggregate_id=data["aggregate_id"],
            aggregate_type=data["aggregate_type"],
            value=data["value"],
            sample_size=data["sample_size"],
            period_start=data["period_start"],
            period_end=data["period_end"],
            computed_at=data["computed_at"],
            project_id=data.get("project_id"),
            breakdown=breakdown if isinstance(breakdown, tuple) else tuple(breakdown.items()),
        )


# -----------------------------------------------------------------------------
# Trend Observation (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class TrendObservation:
    """
    An observed trend over time.

    FROZEN: Immutable once created.
    This is an OBSERVATION, not a PREDICTION.
    Trends are RECORDED for human review.
    """
    trend_id: str
    metric_name: str
    direction: str  # TrendDirection value
    change_rate: float  # Percentage change per period
    period_count: int  # Number of periods analyzed
    period_unit: str  # "hour", "day", "week"
    start_value: float
    end_value: float
    observed_at: str  # ISO format
    confidence: str  # ConfidenceLevel value
    project_id: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend_id": self.trend_id,
            "metric_name": self.metric_name,
            "direction": self.direction,
            "change_rate": self.change_rate,
            "period_count": self.period_count,
            "period_unit": self.period_unit,
            "start_value": self.start_value,
            "end_value": self.end_value,
            "observed_at": self.observed_at,
            "confidence": self.confidence,
            "project_id": self.project_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrendObservation":
        return cls(
            trend_id=data["trend_id"],
            metric_name=data["metric_name"],
            direction=data["direction"],
            change_rate=data["change_rate"],
            period_count=data["period_count"],
            period_unit=data["period_unit"],
            start_value=data["start_value"],
            end_value=data["end_value"],
            observed_at=data["observed_at"],
            confidence=data["confidence"],
            project_id=data.get("project_id"),
        )


# -----------------------------------------------------------------------------
# Memory Entry (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MemoryEntry:
    """
    A single entry in system memory.

    FROZEN: Immutable once created.
    Memory is APPEND-ONLY, never modified.
    """
    entry_id: str
    entry_type: str  # MemoryEntryType value
    source_id: str  # ID of the source record (execution_id, verification_id, etc.)
    source_type: str  # Type of source ("execution", "verification", etc.)
    timestamp: str  # When the original event occurred
    recorded_at: str  # When this memory was recorded
    summary: str  # Brief summary of the event
    outcome: str  # Result/outcome (passed, failed, approved, etc.)
    project_id: Optional[str]
    details: Tuple[Tuple[str, str], ...]  # Additional details (immutable)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "recorded_at": self.recorded_at,
            "summary": self.summary,
            "outcome": self.outcome,
            "project_id": self.project_id,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        details = data.get("details", {})
        if isinstance(details, dict):
            details = tuple(details.items())
        return cls(
            entry_id=data["entry_id"],
            entry_type=data["entry_type"],
            source_id=data["source_id"],
            source_type=data["source_type"],
            timestamp=data["timestamp"],
            recorded_at=data["recorded_at"],
            summary=data["summary"],
            outcome=data["outcome"],
            project_id=data.get("project_id"),
            details=details if isinstance(details, tuple) else tuple(details.items()),
        )


# -----------------------------------------------------------------------------
# Learning Summary (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LearningSummary:
    """
    Summary of learning observations for a time period.

    FROZEN: Immutable once created.
    This is a REPORT, not an ACTION item.
    """
    summary_id: str
    period_start: str
    period_end: str
    generated_at: str
    total_executions: int
    total_verifications: int
    total_approvals: int
    total_incidents: int
    execution_success_rate: float
    verification_pass_rate: float
    approval_grant_rate: float
    pattern_count: int
    trend_count: int
    top_patterns: Tuple[str, ...]  # Top pattern IDs
    top_trends: Tuple[str, ...]  # Top trend IDs
    engine_version: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_id": self.summary_id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "generated_at": self.generated_at,
            "total_executions": self.total_executions,
            "total_verifications": self.total_verifications,
            "total_approvals": self.total_approvals,
            "total_incidents": self.total_incidents,
            "execution_success_rate": self.execution_success_rate,
            "verification_pass_rate": self.verification_pass_rate,
            "approval_grant_rate": self.approval_grant_rate,
            "pattern_count": self.pattern_count,
            "trend_count": self.trend_count,
            "top_patterns": list(self.top_patterns),
            "top_trends": list(self.top_trends),
            "engine_version": self.engine_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LearningSummary":
        return cls(
            summary_id=data["summary_id"],
            period_start=data["period_start"],
            period_end=data["period_end"],
            generated_at=data["generated_at"],
            total_executions=data["total_executions"],
            total_verifications=data["total_verifications"],
            total_approvals=data["total_approvals"],
            total_incidents=data["total_incidents"],
            execution_success_rate=data["execution_success_rate"],
            verification_pass_rate=data["verification_pass_rate"],
            approval_grant_rate=data["approval_grant_rate"],
            pattern_count=data["pattern_count"],
            trend_count=data["trend_count"],
            top_patterns=tuple(data.get("top_patterns", [])),
            top_trends=tuple(data.get("top_trends", [])),
            engine_version=data["engine_version"],
        )


# -----------------------------------------------------------------------------
# Learning Input (Frozen - All Inputs Combined)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class LearningInput:
    """
    Complete input for learning analysis.

    All fields are READ-ONLY snapshots from other phases.
    """
    execution_results: Tuple[Dict[str, Any], ...]  # From Phase 18C
    verification_results: Tuple[Dict[str, Any], ...]  # From Phase 18D
    approval_outcomes: Tuple[Dict[str, Any], ...]  # From Phase 18B
    incident_summaries: Tuple[Dict[str, Any], ...]  # From Phase 17B
    drift_history: Tuple[Dict[str, Any], ...]  # From Phase 16F
    recommendation_outcomes: Tuple[Dict[str, Any], ...]  # From Phase 17C
    period_start: str
    period_end: str

    def compute_hash(self) -> str:
        """Compute deterministic hash of all inputs."""
        data = {
            "execution_count": len(self.execution_results),
            "verification_count": len(self.verification_results),
            "approval_count": len(self.approval_outcomes),
            "incident_count": len(self.incident_summaries),
            "drift_count": len(self.drift_history),
            "recommendation_count": len(self.recommendation_outcomes),
            "period_start": self.period_start,
            "period_end": self.period_end,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()


# -----------------------------------------------------------------------------
# Constraints Documentation
# -----------------------------------------------------------------------------
"""
PHASE 19 CONSTRAINTS
====================

This module provides DATA MODELS ONLY. It does NOT:
- Influence eligibility decisions (Phase 18A)
- Influence approval decisions (Phase 18B)
- Influence execution dispatch (Phase 18C)
- Influence verification outcomes (Phase 18D)
- Modify system thresholds
- Perform ML inference
- Perform optimization
- Trigger automated actions

WHAT IT PROVIDES:
- Historical aggregates (rates, frequencies, counts)
- Pattern observations (recurring events, clusters)
- Trend observations (increasing, decreasing, stable)
- Memory entries (append-only history)
- Learning summaries (human-readable reports)

ALL OUTPUTS ARE:
- Frozen (immutable)
- Append-only (never modified)
- For human insight (not automation)
- Deterministic (same inputs = same outputs)

CONFIDENCE LEVELS ARE STATISTICAL:
- Based on sample size and consistency
- NOT based on ML model predictions
- HIGH: N >= 100, consistency >= 90%
- MEDIUM: N >= 30, consistency >= 70%
- LOW: N >= 10, consistency >= 50%
- INSUFFICIENT: N < 10 or consistency < 50%
"""
