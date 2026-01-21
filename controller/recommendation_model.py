"""
Phase 17C: Recommendation Model & Classification Enums

This module defines the data structures for recommendations.
All structures are IMMUTABLE and ADVISORY-ONLY.

CRITICAL CONSTRAINTS:
- ADVISORY-ONLY: Recommendations suggest, never execute
- NO AUTOMATION: Human must approve/dismiss
- DETERMINISTIC: No ML, no probabilistic inference
- EXPLICIT UNKNOWN: Missing data = UNKNOWN, never guessed
- APPEND-ONLY: Recommendations are never deleted or modified
- NO EXECUTION: Recommendations never trigger actions directly
- NO LIFECYCLE MUTATION: Recommendations never change project state

This phase advises humans, NOT takes action.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, FrozenSet


# -----------------------------------------------------------------------------
# Recommendation Type Enum (LOCKED)
# -----------------------------------------------------------------------------
class RecommendationType(str, Enum):
    """
    Types of recommendations derived from incident analysis.

    This enum is LOCKED - do not add types without explicit approval.
    """
    INVESTIGATE = "investigate"  # Needs human investigation
    MITIGATE = "mitigate"  # Suggests mitigation steps
    IMPROVE = "improve"  # Suggests improvement actions
    REFACTOR = "refactor"  # Suggests code/config refactoring
    DOCUMENT = "document"  # Suggests documentation updates
    NO_ACTION = "no_action"  # No action recommended (informational)


# -----------------------------------------------------------------------------
# Recommendation Severity Enum (LOCKED)
# -----------------------------------------------------------------------------
class RecommendationSeverity(str, Enum):
    """
    Recommendation severity levels.

    CRITICAL: Missing data or classification failure MUST produce UNKNOWN, not failure.
    """
    INFO = "info"  # Informational, low priority
    LOW = "low"  # Minor priority
    MEDIUM = "medium"  # Moderate priority
    HIGH = "high"  # High priority, needs attention
    CRITICAL = "critical"  # Critical priority, urgent
    UNKNOWN = "unknown"  # MANDATORY when data is insufficient


# -----------------------------------------------------------------------------
# Recommendation Approval Enum (LOCKED)
# -----------------------------------------------------------------------------
class RecommendationApproval(str, Enum):
    """
    Approval requirements for recommendations.

    Determines whether human approval is needed before marking as actioned.
    """
    NONE_REQUIRED = "none_required"  # Info only, no approval needed
    CONFIRMATION_REQUIRED = "confirmation_required"  # Simple confirmation
    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"  # Detailed approval


# -----------------------------------------------------------------------------
# Recommendation Status Enum (LOCKED)
# -----------------------------------------------------------------------------
class RecommendationStatus(str, Enum):
    """
    Status of a recommendation.

    Note: This is for TRACKING purposes only.
    Status changes create NEW records, not mutations.
    """
    PENDING = "pending"  # Awaiting human review
    APPROVED = "approved"  # Human approved the recommendation
    DISMISSED = "dismissed"  # Human dismissed the recommendation
    EXPIRED = "expired"  # Recommendation aged out
    UNKNOWN = "unknown"  # Cannot determine status


# -----------------------------------------------------------------------------
# Recommendation (Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Recommendation:
    """
    Immutable recommendation record.

    Once created, a recommendation CANNOT be modified.
    This ensures audit integrity and deterministic replay.

    Recommendations are DERIVED from incidents, not manually created.
    They are ADVISORY ONLY - they never trigger execution.
    """
    recommendation_id: str
    created_at: str  # ISO format
    recommendation_type: str  # RecommendationType value
    severity: str  # RecommendationSeverity value
    approval_required: str  # RecommendationApproval value
    status: str  # RecommendationStatus value
    title: str  # Short descriptive title
    description: str  # Detailed description of recommendation
    rationale: str  # Why this recommendation was generated
    suggested_actions: tuple  # Tuple of action strings (immutable)
    source_incident_ids: tuple  # Tuple of incident IDs (immutable)
    incident_count: int  # Number of linked incidents
    project_id: Optional[str]  # If project-scoped
    aspect: Optional[str]  # If aspect-scoped
    confidence: float  # 0.0 - 1.0, deterministic
    classification_rule: str  # Which rule generated this
    expires_at: Optional[str]  # ISO format, when recommendation expires
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate recommendation on creation."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")
        if self.severity not in [s.value for s in RecommendationSeverity]:
            raise ValueError(f"Invalid severity: {self.severity}")
        if self.recommendation_type not in [t.value for t in RecommendationType]:
            raise ValueError(f"Invalid recommendation type: {self.recommendation_type}")
        if self.approval_required not in [a.value for a in RecommendationApproval]:
            raise ValueError(f"Invalid approval requirement: {self.approval_required}")
        if self.status not in [s.value for s in RecommendationStatus]:
            raise ValueError(f"Invalid status: {self.status}")
        if self.incident_count < 0:
            raise ValueError(f"Incident count cannot be negative: {self.incident_count}")
        if not isinstance(self.source_incident_ids, tuple):
            raise ValueError("source_incident_ids must be a tuple for immutability")
        if not isinstance(self.suggested_actions, tuple):
            raise ValueError("suggested_actions must be a tuple for immutability")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        # Convert tuples to lists for JSON serialization
        result["source_incident_ids"] = list(self.source_incident_ids)
        result["suggested_actions"] = list(self.suggested_actions)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recommendation":
        """Create recommendation from dictionary."""
        # Convert lists back to tuples
        incident_ids = data.get("source_incident_ids", [])
        if isinstance(incident_ids, list):
            incident_ids = tuple(incident_ids)

        actions = data.get("suggested_actions", [])
        if isinstance(actions, list):
            actions = tuple(actions)

        return cls(
            recommendation_id=data["recommendation_id"],
            created_at=data["created_at"],
            recommendation_type=data["recommendation_type"],
            severity=data["severity"],
            approval_required=data["approval_required"],
            status=data["status"],
            title=data["title"],
            description=data["description"],
            rationale=data["rationale"],
            suggested_actions=actions,
            source_incident_ids=incident_ids,
            incident_count=data["incident_count"],
            project_id=data.get("project_id"),
            aspect=data.get("aspect"),
            confidence=data.get("confidence", 0.0),
            classification_rule=data.get("classification_rule", "unknown"),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Approval Record (Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ApprovalRecord:
    """
    Immutable record of an approval or dismissal action.

    This creates an audit trail for all human decisions on recommendations.
    """
    record_id: str
    recommendation_id: str
    action: str  # "approved" or "dismissed"
    user_id: str
    timestamp: str  # ISO format
    reason: Optional[str]  # User-provided reason
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate record on creation."""
        if self.action not in ["approved", "dismissed"]:
            raise ValueError(f"Invalid action: {self.action}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRecord":
        """Create record from dictionary."""
        return cls(
            record_id=data["record_id"],
            recommendation_id=data["recommendation_id"],
            action=data["action"],
            user_id=data["user_id"],
            timestamp=data["timestamp"],
            reason=data.get("reason"),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Recommendation Summary (Read-Only Aggregation)
# -----------------------------------------------------------------------------
@dataclass
class RecommendationSummary:
    """
    Summary of recommendations for a time window.

    This is a READ-ONLY aggregation, never stored.
    """
    generated_at: str
    time_window_start: str
    time_window_end: str
    total_recommendations: int
    by_severity: Dict[str, int]  # severity -> count
    by_type: Dict[str, int]  # recommendation_type -> count
    by_status: Dict[str, int]  # status -> count
    by_approval: Dict[str, int]  # approval_required -> count
    pending_count: int
    pending_approval_count: int  # Those requiring explicit approval
    unknown_count: int
    recent_recommendations: List[Dict[str, Any]]  # Summarized recent recommendations

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# -----------------------------------------------------------------------------
# Recommendation Rule (Read-Only)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RecommendationRule:
    """
    A deterministic rule for recommendation generation.

    Rules are FIXED and DETERMINISTIC - same input always produces same output.
    """
    rule_id: str
    name: str
    description: str
    incident_types: FrozenSet[str]  # Which incident types trigger this rule
    min_severity: str  # Minimum incident severity to trigger
    recommendation_type: str  # Resulting recommendation type
    approval_required: str  # Approval requirement for this recommendation
    action_template: tuple  # Template for suggested actions (tuple for immutability)
    confidence: float  # Confidence of this recommendation (0.0-1.0)

    def matches_incident_type(self, incident_type: str) -> bool:
        """Check if this rule matches an incident type."""
        return incident_type in self.incident_types

    def matches_severity(self, severity: str) -> bool:
        """Check if incident severity meets minimum threshold."""
        severity_order = [
            RecommendationSeverity.INFO.value,
            RecommendationSeverity.LOW.value,
            RecommendationSeverity.MEDIUM.value,
            RecommendationSeverity.HIGH.value,
            RecommendationSeverity.CRITICAL.value,
        ]
        if severity == RecommendationSeverity.UNKNOWN.value:
            return True  # UNKNOWN always matches (but produces UNKNOWN recommendation)
        if self.min_severity == RecommendationSeverity.INFO.value:
            return True  # INFO is minimum, always matches

        try:
            incident_idx = severity_order.index(severity)
            min_idx = severity_order.index(self.min_severity)
            return incident_idx >= min_idx
        except ValueError:
            return False  # Unknown severity value


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Default expiration window for recommendations (in hours)
DEFAULT_EXPIRATION_HOURS = 168  # 7 days

# Maximum incidents per recommendation
MAX_INCIDENTS_PER_RECOMMENDATION = 50
