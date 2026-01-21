"""
Phase 16F: Intent Drift Detection Engine

Detects and classifies drift between baseline intent and current intent.

This engine compares the project's original intent (baseline) against its
current state to detect if the project is evolving beyond its original scope.

DRIFT CLASSIFICATION (LOCKED):
- NONE: No meaningful change, allow silently
- LOW: Minor extension, log only
- MEDIUM: Scope expansion, require user confirmation
- HIGH: Architectural/domain change, hard block until approved
- CRITICAL: Violates original purpose, freeze project

DRIFT DIMENSIONS (ALL REQUIRED):
1. Purpose Drift - Core purpose/problem space change
2. Module Drift - Functional module additions/removals
3. Architecture Drift - Architecture class change
4. Database Drift - Database type change
5. Surface Area Drift - API/domain expansion
6. Non-Functional Drift - Security, scale, compliance changes

Each axis contributes to a deterministic Drift Score (0-100).
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set, FrozenSet, Tuple

logger = logging.getLogger("intent_drift_engine")


# -----------------------------------------------------------------------------
# Drift Classification Levels
# -----------------------------------------------------------------------------
class DriftLevel(str, Enum):
    """
    Drift severity levels.

    These are LOCKED and define the system's response to detected drift.
    """
    NONE = "none"           # No meaningful change - allow silently
    LOW = "low"             # Minor extension - log only
    MEDIUM = "medium"       # Scope expansion - require user confirmation
    HIGH = "high"           # Architectural/domain change - hard block
    CRITICAL = "critical"   # Violates original purpose - freeze project


class DriftDimension(str, Enum):
    """
    Dimensions along which drift is measured.

    Each dimension contributes independently to the overall drift score.
    """
    PURPOSE = "purpose"           # Core purpose/problem space
    MODULE = "module"             # Functional modules
    ARCHITECTURE = "architecture" # Architecture class
    DATABASE = "database"         # Database type
    SURFACE_AREA = "surface_area" # APIs, domains, endpoints
    NON_FUNCTIONAL = "non_functional"  # Security, scale, compliance


# -----------------------------------------------------------------------------
# Drift Score Configuration
# -----------------------------------------------------------------------------
# Weight for each dimension in overall score calculation
DIMENSION_WEIGHTS = {
    DriftDimension.PURPOSE: 0.30,       # Purpose is most important
    DriftDimension.ARCHITECTURE: 0.25,  # Architecture changes are severe
    DriftDimension.MODULE: 0.20,        # Module changes are significant
    DriftDimension.DATABASE: 0.10,      # Database changes are breaking
    DriftDimension.SURFACE_AREA: 0.10,  # Surface area expansion
    DriftDimension.NON_FUNCTIONAL: 0.05,  # Non-functional changes
}

# Threshold scores for drift levels (0-100)
DRIFT_THRESHOLDS = {
    DriftLevel.NONE: 5,      # 0-5: No meaningful drift
    DriftLevel.LOW: 20,      # 6-20: Minor drift
    DriftLevel.MEDIUM: 50,   # 21-50: Moderate drift
    DriftLevel.HIGH: 80,     # 51-80: Severe drift
    DriftLevel.CRITICAL: 100,  # 81-100: Critical drift
}

# Breaking changes that automatically escalate to HIGH
BREAKING_ARCHITECTURE_CHANGES = frozenset([
    ("monolith", "microservices"),
    ("api_only", "fullstack"),
    ("frontend_only", "fullstack"),
    ("backend_only", "fullstack"),
])

# Breaking database changes that escalate
BREAKING_DATABASE_CHANGES = frozenset([
    ("none", "postgresql"),
    ("none", "mysql"),
    ("none", "mongodb"),
    ("sqlite", "postgresql"),
    ("sqlite", "mysql"),
    ("postgresql", "mongodb"),
    ("mysql", "mongodb"),
])


# -----------------------------------------------------------------------------
# Drift Analysis Result
# -----------------------------------------------------------------------------
@dataclass
class DimensionDrift:
    """Drift analysis for a single dimension."""
    dimension: str  # DriftDimension value
    score: float  # 0-100
    level: str  # DriftLevel value
    baseline_value: Any  # What it was
    current_value: Any  # What it is now
    added: List[str]  # Items added
    removed: List[str]  # Items removed
    explanation: str  # Human-readable explanation

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "score": self.score,
            "level": self.level,
            "baseline_value": (
                list(self.baseline_value) if isinstance(self.baseline_value, (set, frozenset))
                else self.baseline_value
            ),
            "current_value": (
                list(self.current_value) if isinstance(self.current_value, (set, frozenset))
                else self.current_value
            ),
            "added": self.added,
            "removed": self.removed,
            "explanation": self.explanation,
        }


@dataclass
class DriftAnalysisResult:
    """
    Complete drift analysis result.

    Contains per-dimension analysis and overall drift classification.
    """
    project_id: str
    baseline_id: str
    analysis_timestamp: str

    # Overall scores
    overall_score: float  # 0-100
    overall_level: str  # DriftLevel value
    requires_action: bool  # True if MEDIUM or higher

    # Per-dimension analysis
    dimension_drifts: List[DimensionDrift]

    # Summary
    summary: str  # Human-readable summary
    recommended_action: str  # What should be done

    # Blocking info
    blocks_execution: bool  # True if HIGH or CRITICAL
    requires_confirmation: bool  # True if MEDIUM
    requires_rebaseline: bool  # True if drift is intentional

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "baseline_id": self.baseline_id,
            "analysis_timestamp": self.analysis_timestamp,
            "overall_score": self.overall_score,
            "overall_level": self.overall_level,
            "requires_action": self.requires_action,
            "dimension_drifts": [d.to_dict() for d in self.dimension_drifts],
            "summary": self.summary,
            "recommended_action": self.recommended_action,
            "blocks_execution": self.blocks_execution,
            "requires_confirmation": self.requires_confirmation,
            "requires_rebaseline": self.requires_rebaseline,
        }

    def get_blocking_dimensions(self) -> List[DimensionDrift]:
        """Get dimensions that are causing blocks."""
        return [
            d for d in self.dimension_drifts
            if d.level in [DriftLevel.HIGH.value, DriftLevel.CRITICAL.value]
        ]


# -----------------------------------------------------------------------------
# Intent Drift Engine
# -----------------------------------------------------------------------------
class IntentDriftEngine:
    """
    Analyzes drift between baseline intent and current intent.

    DETERMINISTIC: Same inputs always produce same outputs.
    NO LLM CALLS: All analysis is rule-based.
    """

    def analyze_drift(
        self,
        project_id: str,
        baseline_id: str,
        baseline_intent: Dict[str, Any],
        current_intent: Dict[str, Any],
    ) -> DriftAnalysisResult:
        """
        Analyze drift between baseline and current intent.

        Args:
            project_id: Project identifier
            baseline_id: Baseline identifier
            baseline_intent: NormalizedIntent as dict (baseline)
            current_intent: NormalizedIntent as dict (current)

        Returns:
            DriftAnalysisResult with complete analysis
        """
        now = datetime.utcnow().isoformat()
        dimension_drifts = []

        # Analyze each dimension
        dimension_drifts.append(self._analyze_purpose_drift(
            baseline_intent, current_intent
        ))
        dimension_drifts.append(self._analyze_module_drift(
            baseline_intent, current_intent
        ))
        dimension_drifts.append(self._analyze_architecture_drift(
            baseline_intent, current_intent
        ))
        dimension_drifts.append(self._analyze_database_drift(
            baseline_intent, current_intent
        ))
        dimension_drifts.append(self._analyze_surface_area_drift(
            baseline_intent, current_intent
        ))
        dimension_drifts.append(self._analyze_non_functional_drift(
            baseline_intent, current_intent
        ))

        # Calculate overall score
        overall_score = self._calculate_overall_score(dimension_drifts)
        overall_level = self._score_to_level(overall_score)

        # Check for breaking changes that override score
        overall_level = self._check_breaking_overrides(dimension_drifts, overall_level)

        # Determine actions
        blocks_execution = overall_level in [DriftLevel.HIGH.value, DriftLevel.CRITICAL.value]
        requires_confirmation = overall_level == DriftLevel.MEDIUM.value
        requires_action = blocks_execution or requires_confirmation
        requires_rebaseline = overall_score >= DRIFT_THRESHOLDS[DriftLevel.MEDIUM]

        # Generate summary and recommendation
        summary = self._generate_summary(dimension_drifts, overall_level)
        recommended_action = self._generate_recommendation(overall_level, dimension_drifts)

        return DriftAnalysisResult(
            project_id=project_id,
            baseline_id=baseline_id,
            analysis_timestamp=now,
            overall_score=round(overall_score, 2),
            overall_level=overall_level,
            requires_action=requires_action,
            dimension_drifts=dimension_drifts,
            summary=summary,
            recommended_action=recommended_action,
            blocks_execution=blocks_execution,
            requires_confirmation=requires_confirmation,
            requires_rebaseline=requires_rebaseline,
        )

    # -------------------------------------------------------------------------
    # Dimension-Specific Analysis
    # -------------------------------------------------------------------------

    def _analyze_purpose_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> DimensionDrift:
        """Analyze drift in project purpose."""
        baseline_purpose = set(baseline.get("purpose_keywords", []))
        current_purpose = set(current.get("purpose_keywords", []))

        added = list(current_purpose - baseline_purpose)
        removed = list(baseline_purpose - current_purpose)

        # Score calculation
        if not baseline_purpose:
            score = 0.0 if not current_purpose else 50.0
        else:
            # Jaccard-based similarity
            intersection = len(baseline_purpose & current_purpose)
            union = len(baseline_purpose | current_purpose)
            similarity = intersection / union if union > 0 else 1.0
            score = (1 - similarity) * 100

            # Penalty for removed core purposes (more severe than additions)
            if removed:
                score = min(100, score + len(removed) * 15)

        level = self._score_to_level(score)

        # Generate explanation
        if not added and not removed:
            explanation = "No change in project purpose"
        else:
            parts = []
            if added:
                parts.append(f"Added purposes: {', '.join(added)}")
            if removed:
                parts.append(f"Removed purposes: {', '.join(removed)}")
            explanation = "; ".join(parts)

        return DimensionDrift(
            dimension=DriftDimension.PURPOSE.value,
            score=round(score, 2),
            level=level,
            baseline_value=list(baseline_purpose),
            current_value=list(current_purpose),
            added=added,
            removed=removed,
            explanation=explanation,
        )

    def _analyze_module_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> DimensionDrift:
        """Analyze drift in functional modules."""
        baseline_modules = set(baseline.get("functional_modules", []))
        current_modules = set(current.get("functional_modules", []))

        added = list(current_modules - baseline_modules)
        removed = list(baseline_modules - current_modules)

        # Score calculation
        if not baseline_modules:
            score = 0.0 if not current_modules else 30.0
        else:
            # Calculate change ratio
            total_baseline = len(baseline_modules)
            added_ratio = len(added) / total_baseline if total_baseline > 0 else 0
            removed_ratio = len(removed) / total_baseline if total_baseline > 0 else 0

            # Additions are less severe than removals
            # 1 module added to 3 = 33% ratio * 50 = 16.67 (LOW)
            # 2 modules added to 3 = 66% ratio * 50 = 33 (MEDIUM)
            # 1 module removed from 3 = 33% ratio * 70 = 23 (MEDIUM)
            score = added_ratio * 50 + removed_ratio * 70
            score = min(100, score)

        level = self._score_to_level(score)

        # Generate explanation
        if not added and not removed:
            explanation = "No change in functional modules"
        else:
            parts = []
            if added:
                parts.append(f"Added modules: {', '.join(added)}")
            if removed:
                parts.append(f"Removed modules: {', '.join(removed)}")
            explanation = "; ".join(parts)

        return DimensionDrift(
            dimension=DriftDimension.MODULE.value,
            score=round(score, 2),
            level=level,
            baseline_value=list(baseline_modules),
            current_value=list(current_modules),
            added=added,
            removed=removed,
            explanation=explanation,
        )

    def _analyze_architecture_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> DimensionDrift:
        """Analyze drift in architecture class."""
        baseline_arch = baseline.get("architecture_class", "unknown")
        current_arch = current.get("architecture_class", "unknown")

        added = []
        removed = []

        # Any architecture change is significant
        if baseline_arch == current_arch:
            score = 0.0
            explanation = f"Architecture unchanged: {baseline_arch}"
        else:
            # Check if it's a breaking change
            change_pair = (baseline_arch, current_arch)
            if change_pair in BREAKING_ARCHITECTURE_CHANGES:
                score = 90.0  # Breaking change
                explanation = f"BREAKING: Architecture changed from {baseline_arch} to {current_arch}"
            else:
                score = 60.0  # Non-breaking but significant
                explanation = f"Architecture changed from {baseline_arch} to {current_arch}"

            added = [current_arch]
            removed = [baseline_arch]

        level = self._score_to_level(score)

        return DimensionDrift(
            dimension=DriftDimension.ARCHITECTURE.value,
            score=round(score, 2),
            level=level,
            baseline_value=baseline_arch,
            current_value=current_arch,
            added=added,
            removed=removed,
            explanation=explanation,
        )

    def _analyze_database_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> DimensionDrift:
        """Analyze drift in database type."""
        baseline_db = baseline.get("database_type", "unknown")
        current_db = current.get("database_type", "unknown")

        added = []
        removed = []

        if baseline_db == current_db:
            score = 0.0
            explanation = f"Database unchanged: {baseline_db}"
        else:
            # Check if it's a breaking change
            change_pair = (baseline_db, current_db)
            if change_pair in BREAKING_DATABASE_CHANGES:
                score = 85.0  # Breaking database change
                explanation = f"BREAKING: Database changed from {baseline_db} to {current_db}"
            else:
                score = 40.0  # Non-breaking but notable
                explanation = f"Database changed from {baseline_db} to {current_db}"

            added = [current_db]
            removed = [baseline_db]

        level = self._score_to_level(score)

        return DimensionDrift(
            dimension=DriftDimension.DATABASE.value,
            score=round(score, 2),
            level=level,
            baseline_value=baseline_db,
            current_value=current_db,
            added=added,
            removed=removed,
            explanation=explanation,
        )

    def _analyze_surface_area_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> DimensionDrift:
        """Analyze drift in API surface area (domains, topology)."""
        baseline_topology = set(baseline.get("domain_topology", []))
        current_topology = set(current.get("domain_topology", []))

        added = list(current_topology - baseline_topology)
        removed = list(baseline_topology - current_topology)

        # Score based on expansion
        if not added and not removed:
            score = 0.0
            explanation = "No change in surface area"
        else:
            # Surface area expansion is significant
            expansion = len(added)
            contraction = len(removed)

            # Each new domain adds to score
            score = expansion * 25 + contraction * 15
            score = min(100, score)

            parts = []
            if added:
                parts.append(f"Added domains: {', '.join(added)}")
            if removed:
                parts.append(f"Removed domains: {', '.join(removed)}")
            explanation = "; ".join(parts)

        level = self._score_to_level(score)

        return DimensionDrift(
            dimension=DriftDimension.SURFACE_AREA.value,
            score=round(score, 2),
            level=level,
            baseline_value=list(baseline_topology),
            current_value=list(current_topology),
            added=added,
            removed=removed,
            explanation=explanation,
        )

    def _analyze_non_functional_drift(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> DimensionDrift:
        """Analyze drift in non-functional aspects (users, scale)."""
        baseline_users = set(baseline.get("target_users", []))
        current_users = set(current.get("target_users", []))

        added = list(current_users - baseline_users)
        removed = list(baseline_users - current_users)

        if not added and not removed:
            score = 0.0
            explanation = "No change in target users"
        else:
            # User base changes are moderate
            score = len(added) * 20 + len(removed) * 10
            score = min(100, score)

            parts = []
            if added:
                parts.append(f"Added user types: {', '.join(added)}")
            if removed:
                parts.append(f"Removed user types: {', '.join(removed)}")
            explanation = "; ".join(parts)

        level = self._score_to_level(score)

        return DimensionDrift(
            dimension=DriftDimension.NON_FUNCTIONAL.value,
            score=round(score, 2),
            level=level,
            baseline_value=list(baseline_users),
            current_value=list(current_users),
            added=added,
            removed=removed,
            explanation=explanation,
        )

    # -------------------------------------------------------------------------
    # Score Calculation
    # -------------------------------------------------------------------------

    def _calculate_overall_score(self, dimension_drifts: List[DimensionDrift]) -> float:
        """Calculate weighted overall drift score."""
        total_score = 0.0

        for drift in dimension_drifts:
            dimension = DriftDimension(drift.dimension)
            weight = DIMENSION_WEIGHTS.get(dimension, 0.1)
            total_score += drift.score * weight

        return total_score

    def _score_to_level(self, score: float) -> str:
        """Convert numeric score to drift level."""
        if score <= DRIFT_THRESHOLDS[DriftLevel.NONE]:
            return DriftLevel.NONE.value
        elif score <= DRIFT_THRESHOLDS[DriftLevel.LOW]:
            return DriftLevel.LOW.value
        elif score <= DRIFT_THRESHOLDS[DriftLevel.MEDIUM]:
            return DriftLevel.MEDIUM.value
        elif score <= DRIFT_THRESHOLDS[DriftLevel.HIGH]:
            return DriftLevel.HIGH.value
        else:
            return DriftLevel.CRITICAL.value

    def _check_breaking_overrides(
        self,
        dimension_drifts: List[DimensionDrift],
        current_level: str,
    ) -> str:
        """Check for breaking changes that override the calculated level."""
        for drift in dimension_drifts:
            # Any HIGH or CRITICAL dimension forces overall to at least HIGH
            if drift.level == DriftLevel.CRITICAL.value:
                return DriftLevel.CRITICAL.value
            elif drift.level == DriftLevel.HIGH.value:
                if current_level not in [DriftLevel.HIGH.value, DriftLevel.CRITICAL.value]:
                    return DriftLevel.HIGH.value
        return current_level

    # -------------------------------------------------------------------------
    # Summary Generation
    # -------------------------------------------------------------------------

    def _generate_summary(
        self,
        dimension_drifts: List[DimensionDrift],
        overall_level: str,
    ) -> str:
        """Generate human-readable summary of drift analysis."""
        if overall_level == DriftLevel.NONE.value:
            return "No significant drift detected. Project remains aligned with baseline intent."

        if overall_level == DriftLevel.LOW.value:
            return "Minor drift detected. Project has small extensions but remains largely aligned."

        if overall_level == DriftLevel.MEDIUM.value:
            drifting = [d for d in dimension_drifts if d.level in [
                DriftLevel.MEDIUM.value, DriftLevel.HIGH.value, DriftLevel.CRITICAL.value
            ]]
            dims = ", ".join([d.dimension for d in drifting])
            return f"Moderate drift detected in: {dims}. User confirmation required before proceeding."

        if overall_level == DriftLevel.HIGH.value:
            drifting = [d for d in dimension_drifts if d.level in [
                DriftLevel.HIGH.value, DriftLevel.CRITICAL.value
            ]]
            dims = ", ".join([d.dimension for d in drifting])
            return f"SEVERE drift detected in: {dims}. Execution BLOCKED until approved."

        # CRITICAL
        return "CRITICAL: Project has fundamentally changed from its original purpose. Project FROZEN."

    def _generate_recommendation(
        self,
        overall_level: str,
        dimension_drifts: List[DimensionDrift],
    ) -> str:
        """Generate recommended action based on drift level."""
        if overall_level == DriftLevel.NONE.value:
            return "No action required. Continue as normal."

        if overall_level == DriftLevel.LOW.value:
            return "Logged for awareness. No action required."

        if overall_level == DriftLevel.MEDIUM.value:
            return "User must confirm intent change before Claude can proceed."

        if overall_level == DriftLevel.HIGH.value:
            return "Request rebaseline approval to update project intent, or revert changes."

        # CRITICAL
        return "Project must be reviewed. Rebaseline or archive required before any further work."


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------
_engine: Optional[IntentDriftEngine] = None


def get_drift_engine() -> IntentDriftEngine:
    """Get the global drift engine instance."""
    global _engine
    if _engine is None:
        _engine = IntentDriftEngine()
    return _engine


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
def analyze_drift(
    project_id: str,
    baseline_id: str,
    baseline_intent: Dict[str, Any],
    current_intent: Dict[str, Any],
) -> DriftAnalysisResult:
    """Analyze drift between baseline and current intent."""
    return get_drift_engine().analyze_drift(
        project_id=project_id,
        baseline_id=baseline_id,
        baseline_intent=baseline_intent,
        current_intent=current_intent,
    )


def check_drift_blocks_execution(
    project_id: str,
    baseline_intent: Dict[str, Any],
    current_intent: Dict[str, Any],
    baseline_id: str = "unknown",
) -> Tuple[bool, str, Optional[DriftAnalysisResult]]:
    """
    Quick check if drift blocks execution.

    Returns: (blocks, reason, analysis)
    """
    analysis = analyze_drift(
        project_id=project_id,
        baseline_id=baseline_id,
        baseline_intent=baseline_intent,
        current_intent=current_intent,
    )

    if analysis.blocks_execution:
        return True, analysis.summary, analysis
    elif analysis.requires_confirmation:
        return True, f"Confirmation required: {analysis.summary}", analysis
    else:
        return False, "", analysis
