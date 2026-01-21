"""
Phase 17C: Recommendation Engine

This module generates recommendations from incidents using deterministic rules.
All processing is ADVISORY-ONLY - no execution, no automation.

CRITICAL CONSTRAINTS:
- RULE-BASED ONLY: No ML, no AI guessing, no probabilistic output
- DETERMINISTIC: Same input always produces same output
- ADVISORY-ONLY: Recommendations suggest actions, never execute them
- UNKNOWN PROPAGATES: If incident is UNKNOWN, recommendation is UNKNOWN
- NO EXECUTION: This engine NEVER triggers any action
- NO LIFECYCLE MUTATION: Recommendations do not change project state
- IMMUTABLE OUTPUT: Generated recommendations are frozen

This phase interprets incidents, NOT takes action on them.
"""

import hashlib
import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict, Any

from .recommendation_model import (
    Recommendation,
    RecommendationRule,
    RecommendationType,
    RecommendationSeverity,
    RecommendationApproval,
    RecommendationStatus,
    DEFAULT_EXPIRATION_HOURS,
    MAX_INCIDENTS_PER_RECOMMENDATION,
)
from .incident_model import (
    Incident,
    IncidentType,
    IncidentSeverity,
)

logger = logging.getLogger("recommendation_engine")


# -----------------------------------------------------------------------------
# Recommendation Rules (LOCKED, DETERMINISTIC)
# -----------------------------------------------------------------------------
RECOMMENDATION_RULES: Tuple[RecommendationRule, ...] = (
    # Resource incidents -> Investigate/Mitigate
    RecommendationRule(
        rule_id="rule-resource-investigate",
        name="Resource Issue Investigation",
        description="Investigate resource exhaustion incidents",
        incident_types=frozenset({IncidentType.RESOURCE.value}),
        min_severity=RecommendationSeverity.MEDIUM.value,
        recommendation_type=RecommendationType.INVESTIGATE.value,
        approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
        action_template=(
            "Review system resource metrics",
            "Identify resource-consuming processes",
            "Evaluate scaling options",
        ),
        confidence=0.85,
    ),
    RecommendationRule(
        rule_id="rule-resource-mitigate",
        name="Resource Issue Mitigation",
        description="Mitigate critical resource exhaustion",
        incident_types=frozenset({IncidentType.RESOURCE.value}),
        min_severity=RecommendationSeverity.CRITICAL.value,
        recommendation_type=RecommendationType.MITIGATE.value,
        approval_required=RecommendationApproval.EXPLICIT_APPROVAL_REQUIRED.value,
        action_template=(
            "Identify and terminate non-essential processes",
            "Increase resource limits if possible",
            "Enable resource monitoring alerts",
            "Schedule maintenance window",
        ),
        confidence=0.9,
    ),

    # Reliability incidents -> Investigate/Improve
    RecommendationRule(
        rule_id="rule-reliability-investigate",
        name="Reliability Issue Investigation",
        description="Investigate service/job failures",
        incident_types=frozenset({IncidentType.RELIABILITY.value}),
        min_severity=RecommendationSeverity.LOW.value,
        recommendation_type=RecommendationType.INVESTIGATE.value,
        approval_required=RecommendationApproval.NONE_REQUIRED.value,
        action_template=(
            "Review failure logs",
            "Check recent changes",
            "Verify dependencies",
        ),
        confidence=0.8,
    ),
    RecommendationRule(
        rule_id="rule-reliability-improve",
        name="Reliability Improvement",
        description="Improve reliability after failures",
        incident_types=frozenset({IncidentType.RELIABILITY.value}),
        min_severity=RecommendationSeverity.HIGH.value,
        recommendation_type=RecommendationType.IMPROVE.value,
        approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
        action_template=(
            "Add retry logic for transient failures",
            "Implement circuit breaker pattern",
            "Add health checks",
            "Review test coverage",
        ),
        confidence=0.85,
    ),

    # Security incidents -> Investigate (always explicit approval)
    RecommendationRule(
        rule_id="rule-security-investigate",
        name="Security Incident Investigation",
        description="Investigate security-related incidents",
        incident_types=frozenset({IncidentType.SECURITY.value}),
        min_severity=RecommendationSeverity.LOW.value,
        recommendation_type=RecommendationType.INVESTIGATE.value,
        approval_required=RecommendationApproval.EXPLICIT_APPROVAL_REQUIRED.value,
        action_template=(
            "Review audit logs",
            "Check for unauthorized access",
            "Verify gate decisions",
            "Review permission changes",
        ),
        confidence=0.9,
    ),

    # Governance incidents -> Investigate/Document
    RecommendationRule(
        rule_id="rule-governance-investigate",
        name="Governance Investigation",
        description="Investigate drift/contract violations",
        incident_types=frozenset({IncidentType.GOVERNANCE.value}),
        min_severity=RecommendationSeverity.MEDIUM.value,
        recommendation_type=RecommendationType.INVESTIGATE.value,
        approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
        action_template=(
            "Review intent baseline",
            "Check for unauthorized changes",
            "Verify contract compliance",
        ),
        confidence=0.85,
    ),
    RecommendationRule(
        rule_id="rule-governance-document",
        name="Governance Documentation",
        description="Document governance changes",
        incident_types=frozenset({IncidentType.GOVERNANCE.value}),
        min_severity=RecommendationSeverity.HIGH.value,
        recommendation_type=RecommendationType.DOCUMENT.value,
        approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
        action_template=(
            "Update project documentation",
            "Record rebaseline decision",
            "Update contract definitions",
        ),
        confidence=0.8,
    ),

    # Configuration incidents -> Investigate/Refactor
    RecommendationRule(
        rule_id="rule-config-investigate",
        name="Configuration Investigation",
        description="Investigate configuration anomalies",
        incident_types=frozenset({IncidentType.CONFIGURATION.value}),
        min_severity=RecommendationSeverity.LOW.value,
        recommendation_type=RecommendationType.INVESTIGATE.value,
        approval_required=RecommendationApproval.NONE_REQUIRED.value,
        action_template=(
            "Review configuration changes",
            "Check for inconsistencies",
            "Verify environment settings",
        ),
        confidence=0.8,
    ),
    RecommendationRule(
        rule_id="rule-config-refactor",
        name="Configuration Refactoring",
        description="Refactor problematic configurations",
        incident_types=frozenset({IncidentType.CONFIGURATION.value}),
        min_severity=RecommendationSeverity.HIGH.value,
        recommendation_type=RecommendationType.REFACTOR.value,
        approval_required=RecommendationApproval.EXPLICIT_APPROVAL_REQUIRED.value,
        action_template=(
            "Standardize configuration format",
            "Move to environment variables",
            "Implement configuration validation",
            "Document configuration schema",
        ),
        confidence=0.85,
    ),

    # Performance incidents -> Investigate/Improve
    RecommendationRule(
        rule_id="rule-performance-investigate",
        name="Performance Investigation",
        description="Investigate performance degradation",
        incident_types=frozenset({IncidentType.PERFORMANCE.value}),
        min_severity=RecommendationSeverity.MEDIUM.value,
        recommendation_type=RecommendationType.INVESTIGATE.value,
        approval_required=RecommendationApproval.NONE_REQUIRED.value,
        action_template=(
            "Profile application performance",
            "Identify slow queries/operations",
            "Review resource utilization",
        ),
        confidence=0.8,
    ),
    RecommendationRule(
        rule_id="rule-performance-improve",
        name="Performance Improvement",
        description="Improve system performance",
        incident_types=frozenset({IncidentType.PERFORMANCE.value}),
        min_severity=RecommendationSeverity.HIGH.value,
        recommendation_type=RecommendationType.IMPROVE.value,
        approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
        action_template=(
            "Optimize slow operations",
            "Implement caching",
            "Review database indices",
            "Consider horizontal scaling",
        ),
        confidence=0.85,
    ),
)


# -----------------------------------------------------------------------------
# Recommendation Generator
# -----------------------------------------------------------------------------
class RecommendationGenerator:
    """
    Generates recommendations from incidents using deterministic rules.

    CRITICAL: This class ONLY generates recommendations. It does not:
    - Execute any actions
    - Modify lifecycle state
    - Trigger deployments
    - Send alerts

    All output is ADVISORY ONLY for human review.
    """

    def __init__(self):
        """Initialize the generator."""
        self._rules = RECOMMENDATION_RULES

    def generate(self, incident: Incident) -> Optional[Recommendation]:
        """
        Generate a recommendation from a single incident.

        Uses deterministic rule matching - same incident always produces
        same recommendation.

        Returns None if no rule matches or incident is insufficient.
        """
        # UNKNOWN incidents produce UNKNOWN recommendations
        if incident.incident_type == IncidentType.UNKNOWN.value:
            return self._create_unknown_recommendation(incident)

        # Find matching rule (first match wins, rules ordered by priority)
        matched_rule = None
        for rule in self._rules:
            if rule.matches_incident_type(incident.incident_type):
                if rule.matches_severity(incident.severity):
                    matched_rule = rule
                    break

        if matched_rule is None:
            # No matching rule - create no-action recommendation
            return self._create_no_action_recommendation(incident)

        return self._create_recommendation(incident, matched_rule)

    def generate_from_incidents(
        self,
        incidents: List[Incident],
    ) -> List[Recommendation]:
        """
        Generate recommendations from multiple incidents.

        Groups related incidents and generates consolidated recommendations.

        Returns: List of recommendations (may be fewer than input incidents)
        """
        if not incidents:
            return []

        # Group incidents by type and project for consolidation
        groups: Dict[str, List[Incident]] = {}
        for incident in incidents[:MAX_INCIDENTS_PER_RECOMMENDATION]:
            key = f"{incident.incident_type}:{incident.project_id or 'system'}"
            if key not in groups:
                groups[key] = []
            groups[key].append(incident)

        recommendations = []
        for group_key, group_incidents in groups.items():
            # Use most severe incident as primary
            primary = max(
                group_incidents,
                key=lambda i: self._severity_to_int(i.severity)
            )
            recommendation = self.generate(primary)
            if recommendation:
                # Update with all incident IDs
                all_ids = tuple(i.incident_id for i in group_incidents)
                recommendation = self._update_incident_ids(recommendation, all_ids)
                recommendations.append(recommendation)

        return recommendations

    def _create_recommendation(
        self,
        incident: Incident,
        rule: RecommendationRule,
    ) -> Recommendation:
        """Create recommendation from incident and matched rule."""
        now = datetime.utcnow()
        rec_id = self._generate_id(incident, rule, now)

        # Derive severity from incident severity (map to recommendation severity)
        severity = self._map_incident_to_recommendation_severity(incident.severity)

        # Calculate expiration
        expires_at = (now + timedelta(hours=DEFAULT_EXPIRATION_HOURS)).isoformat()

        # Calculate confidence (rule confidence * incident confidence)
        confidence = min(rule.confidence * incident.confidence, 1.0)

        return Recommendation(
            recommendation_id=rec_id,
            created_at=now.isoformat(),
            recommendation_type=rule.recommendation_type,
            severity=severity,
            approval_required=rule.approval_required,
            status=RecommendationStatus.PENDING.value,
            title=f"{rule.name}: {incident.title[:50]}",
            description=self._generate_description(incident, rule),
            rationale=f"Generated by {rule.rule_id} from incident {incident.incident_id}",
            suggested_actions=rule.action_template,
            source_incident_ids=(incident.incident_id,),
            incident_count=1,
            project_id=incident.project_id,
            aspect=incident.aspect,
            confidence=confidence,
            classification_rule=rule.rule_id,
            expires_at=expires_at,
            metadata={
                "incident_type": incident.incident_type,
                "incident_severity": incident.severity,
            },
        )

    def _create_unknown_recommendation(self, incident: Incident) -> Recommendation:
        """Create UNKNOWN recommendation from UNKNOWN incident."""
        now = datetime.utcnow()
        rec_id = f"rec-unknown-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

        return Recommendation(
            recommendation_id=rec_id,
            created_at=now.isoformat(),
            recommendation_type=RecommendationType.INVESTIGATE.value,
            severity=RecommendationSeverity.UNKNOWN.value,
            approval_required=RecommendationApproval.CONFIRMATION_REQUIRED.value,
            status=RecommendationStatus.PENDING.value,
            title="Unknown Incident Investigation",
            description="An incident could not be classified. Manual investigation required.",
            rationale=f"Generated from unclassified incident {incident.incident_id}",
            suggested_actions=(
                "Review incident details manually",
                "Check source signals for context",
                "Determine appropriate action",
            ),
            source_incident_ids=(incident.incident_id,),
            incident_count=1,
            project_id=incident.project_id,
            aspect=incident.aspect,
            confidence=0.0,
            classification_rule="unknown",
            expires_at=(now + timedelta(hours=DEFAULT_EXPIRATION_HOURS)).isoformat(),
            metadata={"reason": "Unknown incident type"},
        )

    def _create_no_action_recommendation(self, incident: Incident) -> Recommendation:
        """Create NO_ACTION recommendation when no rule matches."""
        now = datetime.utcnow()
        rec_id = f"rec-noaction-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

        return Recommendation(
            recommendation_id=rec_id,
            created_at=now.isoformat(),
            recommendation_type=RecommendationType.NO_ACTION.value,
            severity=RecommendationSeverity.INFO.value,
            approval_required=RecommendationApproval.NONE_REQUIRED.value,
            status=RecommendationStatus.PENDING.value,
            title=f"Info: {incident.title[:60]}",
            description="No specific action recommended. For information only.",
            rationale=f"No matching rule for incident {incident.incident_id}",
            suggested_actions=("Monitor for changes",),
            source_incident_ids=(incident.incident_id,),
            incident_count=1,
            project_id=incident.project_id,
            aspect=incident.aspect,
            confidence=1.0,  # High confidence in no-action
            classification_rule="no-match",
            expires_at=(now + timedelta(hours=DEFAULT_EXPIRATION_HOURS)).isoformat(),
            metadata={},
        )

    def _update_incident_ids(
        self,
        recommendation: Recommendation,
        incident_ids: Tuple[str, ...],
    ) -> Recommendation:
        """Create new recommendation with updated incident IDs (immutable update)."""
        return Recommendation(
            recommendation_id=recommendation.recommendation_id,
            created_at=recommendation.created_at,
            recommendation_type=recommendation.recommendation_type,
            severity=recommendation.severity,
            approval_required=recommendation.approval_required,
            status=recommendation.status,
            title=recommendation.title,
            description=recommendation.description,
            rationale=recommendation.rationale,
            suggested_actions=recommendation.suggested_actions,
            source_incident_ids=incident_ids,
            incident_count=len(incident_ids),
            project_id=recommendation.project_id,
            aspect=recommendation.aspect,
            confidence=recommendation.confidence,
            classification_rule=recommendation.classification_rule,
            expires_at=recommendation.expires_at,
            metadata=recommendation.metadata,
        )

    def _generate_id(
        self,
        incident: Incident,
        rule: RecommendationRule,
        timestamp: datetime,
    ) -> str:
        """Generate deterministic recommendation ID."""
        content = f"{incident.incident_id}:{rule.rule_id}:{timestamp.isoformat()}"
        hash_suffix = hashlib.sha256(content.encode()).hexdigest()[:8]
        return f"rec-{timestamp.strftime('%Y%m%d%H%M%S')}-{hash_suffix}"

    def _generate_description(
        self,
        incident: Incident,
        rule: RecommendationRule,
    ) -> str:
        """Generate recommendation description."""
        return (
            f"{rule.description}. "
            f"Based on {incident.incident_type} incident "
            f"with {incident.severity} severity. "
            f"{incident.description[:100]}"
        )

    def _map_incident_to_recommendation_severity(self, incident_severity: str) -> str:
        """Map incident severity to recommendation severity."""
        # Direct mapping (both use same severity scale)
        mapping = {
            IncidentSeverity.INFO.value: RecommendationSeverity.INFO.value,
            IncidentSeverity.LOW.value: RecommendationSeverity.LOW.value,
            IncidentSeverity.MEDIUM.value: RecommendationSeverity.MEDIUM.value,
            IncidentSeverity.HIGH.value: RecommendationSeverity.HIGH.value,
            IncidentSeverity.CRITICAL.value: RecommendationSeverity.CRITICAL.value,
            IncidentSeverity.UNKNOWN.value: RecommendationSeverity.UNKNOWN.value,
        }
        return mapping.get(incident_severity, RecommendationSeverity.UNKNOWN.value)

    def _severity_to_int(self, severity: str) -> int:
        """Convert severity to integer for comparison."""
        order = {
            RecommendationSeverity.INFO.value: 0,
            RecommendationSeverity.LOW.value: 1,
            RecommendationSeverity.MEDIUM.value: 2,
            RecommendationSeverity.HIGH.value: 3,
            RecommendationSeverity.CRITICAL.value: 4,
            RecommendationSeverity.UNKNOWN.value: -1,
        }
        return order.get(severity, -1)


# -----------------------------------------------------------------------------
# Recommendation Engine (Main Interface)
# -----------------------------------------------------------------------------
class RecommendationEngine:
    """
    Main engine for generating recommendations from incidents.

    OBSERVATION-ONLY: This engine generates recommendations but does not:
    - Execute any actions
    - Modify any state
    - Trigger any automation

    All output is for human review and approval.
    """

    def __init__(self):
        """Initialize the engine."""
        self._generator = RecommendationGenerator()

    def generate_recommendations(
        self,
        incidents: List[Incident],
    ) -> List[Recommendation]:
        """
        Generate recommendations from incidents.

        Returns: List of recommendations (ADVISORY ONLY)
        """
        if not incidents:
            return []

        return self._generator.generate_from_incidents(incidents)

    def generate_single(self, incident: Incident) -> Optional[Recommendation]:
        """
        Generate a single recommendation from one incident.

        Returns: Recommendation or None
        """
        return self._generator.generate(incident)


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------
_engine: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Get the global recommendation engine instance."""
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


def generate_recommendations(incidents: List[Incident]) -> List[Recommendation]:
    """Generate recommendations from incidents."""
    return get_recommendation_engine().generate_recommendations(incidents)


def generate_single_recommendation(incident: Incident) -> Optional[Recommendation]:
    """Generate a single recommendation from one incident."""
    return get_recommendation_engine().generate_single(incident)


logger.info("Recommendation Engine module loaded (Phase 17C - ADVISORY-ONLY)")
