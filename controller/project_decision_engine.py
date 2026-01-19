"""
Phase 16E: Project Decision Engine

Deterministic decision making for project creation and conflict resolution.

This module implements the LOCKED decision matrix:
| Condition                    | Result                          |
|-----------------------------|---------------------------------|
| Same fingerprint            | Ask user: reuse / extend        |
| Same project, changed CHD   | CHANGE_MODE lifecycle           |
| Breaking architecture       | New version                     |
| Totally new fingerprint     | New project                     |

HARD CONSTRAINTS:
- NO silent decisions allowed
- Every decision must be explainable
- User confirmation required for ambiguous cases
- All decisions are logged for audit
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

from controller.project_identity import (
    ProjectIdentity,
    NormalizedIntent,
    ProjectIdentityManager,
    IntentExtractor,
    FingerprintGenerator,
    create_project_identity,
)

logger = logging.getLogger("project_decision_engine")


# -----------------------------------------------------------------------------
# Decision Types
# -----------------------------------------------------------------------------
class DecisionType(str, Enum):
    """Types of project creation decisions."""
    NEW_PROJECT = "NEW_PROJECT"  # Create brand new project
    REUSE_PROJECT = "REUSE_PROJECT"  # Use existing project as-is
    CHANGE_MODE = "CHANGE_MODE"  # Start change cycle on existing project
    NEW_VERSION = "NEW_VERSION"  # Create new major version
    CONFLICT_DETECTED = "CONFLICT_DETECTED"  # Requires user resolution


class ConflictType(str, Enum):
    """Types of conflicts detected."""
    EXACT_DUPLICATE = "EXACT_DUPLICATE"  # Same fingerprint
    HIGH_SIMILARITY = "HIGH_SIMILARITY"  # Very similar (>0.85)
    SIMILAR_PURPOSE = "SIMILAR_PURPOSE"  # Same purpose, different scope
    BREAKING_CHANGE = "BREAKING_CHANGE"  # Architecture change
    SCOPE_EXPANSION = "SCOPE_EXPANSION"  # Adding new modules
    SCOPE_REDUCTION = "SCOPE_REDUCTION"  # Removing modules
    NONE = "NONE"  # No conflict


class UserChoice(str, Enum):
    """User choices for conflict resolution."""
    IMPROVE_EXISTING = "IMPROVE_EXISTING"  # Add to existing project
    ADD_MODULE = "ADD_MODULE"  # Add new module to project
    CREATE_VERSION = "CREATE_VERSION"  # Create new version
    CREATE_NEW = "CREATE_NEW"  # Force create new project
    CANCEL = "CANCEL"  # Abort operation


# -----------------------------------------------------------------------------
# Decision Result
# -----------------------------------------------------------------------------
@dataclass
class DecisionResult:
    """
    Result of a project decision.

    Contains the decision type, confidence level, explanation,
    and whether user confirmation is required.
    """
    decision: DecisionType
    confidence: float  # 0.0 to 1.0
    explanation: str  # Human-readable explanation
    requires_user_confirmation: bool
    conflict_type: ConflictType = ConflictType.NONE
    existing_project_id: Optional[str] = None
    existing_project_name: Optional[str] = None
    similarity_score: Optional[float] = None
    suggested_action: Optional[str] = None
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "decision": self.decision.value,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "requires_user_confirmation": self.requires_user_confirmation,
            "conflict_type": self.conflict_type.value,
            "existing_project": {
                "project_id": self.existing_project_id,
                "project_name": self.existing_project_name,
            } if self.existing_project_id else None,
            "similarity_score": self.similarity_score,
            "suggested_action": self.suggested_action,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------------
# Architecture Change Detector
# -----------------------------------------------------------------------------
class ArchitectureChangeDetector:
    """
    Detects breaking architecture changes between project versions.

    Breaking changes include:
    - Switching from monolith to microservices (or vice versa)
    - Changing primary database type
    - Major topology changes (adding/removing major domains)
    """

    # Changes considered "breaking"
    BREAKING_ARCHITECTURE_TRANSITIONS = [
        ("monolith", "microservices"),
        ("microservices", "monolith"),
        ("api_only", "fullstack"),
        ("fullstack", "api_only"),
    ]

    BREAKING_DATABASE_TRANSITIONS = [
        # SQL to NoSQL and vice versa
        ("postgresql", "mongodb"),
        ("mysql", "mongodb"),
        ("mongodb", "postgresql"),
        ("mongodb", "mysql"),
    ]

    @classmethod
    def detect_breaking_changes(
        cls,
        old_intent: NormalizedIntent,
        new_intent: NormalizedIntent,
    ) -> Tuple[bool, List[str]]:
        """
        Detect breaking architecture changes.

        Returns:
            Tuple of (has_breaking_changes, list_of_changes)
        """
        changes = []

        # Check architecture transition
        old_arch = old_intent.architecture_class
        new_arch = new_intent.architecture_class
        if (old_arch, new_arch) in cls.BREAKING_ARCHITECTURE_TRANSITIONS:
            changes.append(
                f"Architecture change: {old_arch} -> {new_arch}"
            )

        # Check database transition
        old_db = old_intent.database_type
        new_db = new_intent.database_type
        if (old_db, new_db) in cls.BREAKING_DATABASE_TRANSITIONS:
            changes.append(
                f"Database change: {old_db} -> {new_db}"
            )

        # Check major topology removal
        removed_topology = old_intent.domain_topology - new_intent.domain_topology
        critical_domains = {"api", "backend", "frontend"}
        removed_critical = removed_topology & critical_domains
        if removed_critical:
            changes.append(
                f"Critical domain(s) removed: {', '.join(removed_critical)}"
            )

        # Check purpose change
        old_purposes = old_intent.purpose_keywords
        new_purposes = new_intent.purpose_keywords
        if old_purposes and new_purposes and not (old_purposes & new_purposes):
            changes.append(
                f"Complete purpose change: {old_purposes} -> {new_purposes}"
            )

        return len(changes) > 0, changes


# -----------------------------------------------------------------------------
# Scope Change Detector
# -----------------------------------------------------------------------------
class ScopeChangeDetector:
    """
    Detects scope changes between project versions.

    Scope changes include:
    - Adding new functional modules
    - Removing existing modules
    - Adding new domain aspects
    """

    @classmethod
    def detect_scope_changes(
        cls,
        old_intent: NormalizedIntent,
        new_intent: NormalizedIntent,
    ) -> Dict[str, Any]:
        """
        Detect scope changes between intents.

        Returns:
            Dictionary with change details
        """
        # Module changes
        added_modules = new_intent.functional_modules - old_intent.functional_modules
        removed_modules = old_intent.functional_modules - new_intent.functional_modules

        # Topology changes
        added_topology = new_intent.domain_topology - old_intent.domain_topology
        removed_topology = old_intent.domain_topology - new_intent.domain_topology

        # Purpose changes
        added_purposes = new_intent.purpose_keywords - old_intent.purpose_keywords
        removed_purposes = old_intent.purpose_keywords - new_intent.purpose_keywords

        # Determine change type
        is_expansion = bool(added_modules or added_topology or added_purposes)
        is_reduction = bool(removed_modules or removed_topology or removed_purposes)

        change_type = "none"
        if is_expansion and not is_reduction:
            change_type = "expansion"
        elif is_reduction and not is_expansion:
            change_type = "reduction"
        elif is_expansion and is_reduction:
            change_type = "modification"

        return {
            "change_type": change_type,
            "added_modules": list(added_modules),
            "removed_modules": list(removed_modules),
            "added_topology": list(added_topology),
            "removed_topology": list(removed_topology),
            "added_purposes": list(added_purposes),
            "removed_purposes": list(removed_purposes),
            "is_significant": len(added_modules) + len(removed_modules) >= 2,
        }


# -----------------------------------------------------------------------------
# Project Decision Engine
# -----------------------------------------------------------------------------
class ProjectDecisionEngine:
    """
    Core decision engine for project creation and conflict resolution.

    This engine implements the decision matrix and ensures:
    - No silent decisions
    - Full explainability
    - User confirmation for ambiguous cases
    """

    # Similarity thresholds
    EXACT_MATCH_THRESHOLD = 1.0
    HIGH_SIMILARITY_THRESHOLD = 0.85
    MODERATE_SIMILARITY_THRESHOLD = 0.70
    LOW_SIMILARITY_THRESHOLD = 0.50

    @classmethod
    def evaluate(
        cls,
        new_description: str,
        new_requirements: Optional[str],
        new_tech_stack: Optional[Dict[str, Any]],
        new_aspects: Optional[List[str]],
        existing_identities: List[Tuple[ProjectIdentity, str]],  # (identity, project_name)
    ) -> DecisionResult:
        """
        Evaluate a project creation request against existing projects.

        Args:
            new_description: New project description
            new_requirements: New project requirements
            new_tech_stack: New project tech stack
            new_aspects: New project aspects
            existing_identities: List of (identity, project_name) tuples

        Returns:
            DecisionResult with recommended action
        """
        # Extract intent for new project
        new_intent = IntentExtractor.extract_intent(
            description=new_description,
            requirements=new_requirements,
            tech_stack=new_tech_stack,
            aspects=new_aspects,
        )

        # Generate fingerprint
        new_fingerprint = FingerprintGenerator.generate(new_intent)

        # If no existing projects, create new
        if not existing_identities:
            return DecisionResult(
                decision=DecisionType.NEW_PROJECT,
                confidence=1.0,
                explanation="No existing projects found. Creating new project.",
                requires_user_confirmation=False,
                conflict_type=ConflictType.NONE,
                metadata={
                    "fingerprint": new_fingerprint,
                    "intent": new_intent.to_dict(),
                },
            )

        # Check for exact fingerprint match
        for existing_id, project_name in existing_identities:
            if existing_id.fingerprint == new_fingerprint:
                return cls._handle_exact_match(
                    existing_id, project_name, new_intent, new_fingerprint
                )

        # Find similar projects
        best_match = None
        best_score = 0.0
        best_name = None

        for existing_id, project_name in existing_identities:
            similarity = FingerprintGenerator.compute_similarity(
                new_intent, existing_id.normalized_intent
            )
            if similarity > best_score:
                best_score = similarity
                best_match = existing_id
                best_name = project_name

        # Evaluate based on similarity
        if best_score >= cls.HIGH_SIMILARITY_THRESHOLD:
            return cls._handle_high_similarity(
                best_match, best_name, new_intent, new_fingerprint, best_score
            )
        elif best_score >= cls.MODERATE_SIMILARITY_THRESHOLD:
            return cls._handle_moderate_similarity(
                best_match, best_name, new_intent, new_fingerprint, best_score
            )
        elif best_score >= cls.LOW_SIMILARITY_THRESHOLD:
            return cls._handle_low_similarity(
                best_match, best_name, new_intent, new_fingerprint, best_score
            )
        else:
            # New project - no significant similarity
            return DecisionResult(
                decision=DecisionType.NEW_PROJECT,
                confidence=1.0 - best_score,
                explanation=(
                    f"No similar projects found (best match: {best_score:.0%} similarity). "
                    "Creating new project."
                ),
                requires_user_confirmation=False,
                conflict_type=ConflictType.NONE,
                similarity_score=best_score,
                existing_project_id=best_match.project_id if best_match else None,
                existing_project_name=best_name,
                metadata={
                    "fingerprint": new_fingerprint,
                    "intent": new_intent.to_dict(),
                },
            )

    @classmethod
    def _handle_exact_match(
        cls,
        existing: ProjectIdentity,
        project_name: str,
        new_intent: NormalizedIntent,
        new_fingerprint: str,
    ) -> DecisionResult:
        """Handle exact fingerprint match - requires user choice."""
        return DecisionResult(
            decision=DecisionType.CONFLICT_DETECTED,
            confidence=1.0,
            explanation=(
                f"Exact duplicate detected! Project '{project_name}' has identical "
                f"semantic fingerprint. This appears to be the same project."
            ),
            requires_user_confirmation=True,
            conflict_type=ConflictType.EXACT_DUPLICATE,
            existing_project_id=existing.project_id,
            existing_project_name=project_name,
            similarity_score=1.0,
            suggested_action="Consider improving the existing project instead of creating a duplicate.",
            alternatives=[
                {
                    "choice": UserChoice.IMPROVE_EXISTING.value,
                    "label": "Improve existing project",
                    "description": f"Add changes to '{project_name}' using CHANGE_MODE",
                },
                {
                    "choice": UserChoice.ADD_MODULE.value,
                    "label": "Add new module",
                    "description": f"Add a new module to '{project_name}'",
                },
                {
                    "choice": UserChoice.CREATE_NEW.value,
                    "label": "Create anyway",
                    "description": "Force create a new project (not recommended)",
                },
                {
                    "choice": UserChoice.CANCEL.value,
                    "label": "Cancel",
                    "description": "Abort project creation",
                },
            ],
            metadata={
                "new_fingerprint": new_fingerprint,
                "existing_fingerprint": existing.fingerprint,
                "intent": new_intent.to_dict(),
            },
        )

    @classmethod
    def _handle_high_similarity(
        cls,
        existing: ProjectIdentity,
        project_name: str,
        new_intent: NormalizedIntent,
        new_fingerprint: str,
        similarity: float,
    ) -> DecisionResult:
        """Handle high similarity - check for breaking changes."""
        # Check for breaking architecture changes
        has_breaking, breaking_changes = ArchitectureChangeDetector.detect_breaking_changes(
            existing.normalized_intent, new_intent
        )

        if has_breaking:
            return DecisionResult(
                decision=DecisionType.CONFLICT_DETECTED,
                confidence=0.9,
                explanation=(
                    f"High similarity ({similarity:.0%}) with '{project_name}' but with "
                    f"breaking architecture changes: {'; '.join(breaking_changes)}. "
                    "This may require a new version."
                ),
                requires_user_confirmation=True,
                conflict_type=ConflictType.BREAKING_CHANGE,
                existing_project_id=existing.project_id,
                existing_project_name=project_name,
                similarity_score=similarity,
                suggested_action="Create a new version of the project with the new architecture.",
                alternatives=[
                    {
                        "choice": UserChoice.CREATE_VERSION.value,
                        "label": "Create new version",
                        "description": f"Create v2 of '{project_name}' with new architecture",
                    },
                    {
                        "choice": UserChoice.IMPROVE_EXISTING.value,
                        "label": "Modify existing",
                        "description": f"Apply changes to '{project_name}' (may break things)",
                    },
                    {
                        "choice": UserChoice.CREATE_NEW.value,
                        "label": "Create new project",
                        "description": "Create as separate project",
                    },
                    {
                        "choice": UserChoice.CANCEL.value,
                        "label": "Cancel",
                        "description": "Abort project creation",
                    },
                ],
                metadata={
                    "breaking_changes": breaking_changes,
                    "fingerprint": new_fingerprint,
                    "intent": new_intent.to_dict(),
                },
            )

        # Check scope changes
        scope_changes = ScopeChangeDetector.detect_scope_changes(
            existing.normalized_intent, new_intent
        )

        if scope_changes["change_type"] == "expansion":
            return DecisionResult(
                decision=DecisionType.CONFLICT_DETECTED,
                confidence=0.85,
                explanation=(
                    f"Very similar to '{project_name}' ({similarity:.0%}) with scope expansion. "
                    f"New modules: {', '.join(scope_changes['added_modules']) or 'none'}. "
                    "Consider adding to existing project."
                ),
                requires_user_confirmation=True,
                conflict_type=ConflictType.SCOPE_EXPANSION,
                existing_project_id=existing.project_id,
                existing_project_name=project_name,
                similarity_score=similarity,
                suggested_action="Add the new functionality to the existing project.",
                alternatives=[
                    {
                        "choice": UserChoice.IMPROVE_EXISTING.value,
                        "label": "Improve existing",
                        "description": f"Add new modules to '{project_name}'",
                    },
                    {
                        "choice": UserChoice.ADD_MODULE.value,
                        "label": "Add as module",
                        "description": "Add as a new module/aspect",
                    },
                    {
                        "choice": UserChoice.CREATE_NEW.value,
                        "label": "Create new project",
                        "description": "Create as separate project",
                    },
                    {
                        "choice": UserChoice.CANCEL.value,
                        "label": "Cancel",
                        "description": "Abort project creation",
                    },
                ],
                metadata={
                    "scope_changes": scope_changes,
                    "fingerprint": new_fingerprint,
                    "intent": new_intent.to_dict(),
                },
            )

        # High similarity, no breaking changes - suggest CHANGE_MODE
        return DecisionResult(
            decision=DecisionType.CONFLICT_DETECTED,
            confidence=0.9,
            explanation=(
                f"Very similar to existing project '{project_name}' ({similarity:.0%}). "
                "The intent appears to be an improvement or modification."
            ),
            requires_user_confirmation=True,
            conflict_type=ConflictType.HIGH_SIMILARITY,
            existing_project_id=existing.project_id,
            existing_project_name=project_name,
            similarity_score=similarity,
            suggested_action="Use CHANGE_MODE to improve the existing project.",
            alternatives=[
                {
                    "choice": UserChoice.IMPROVE_EXISTING.value,
                    "label": "Improve existing (recommended)",
                    "description": f"Start a change cycle on '{project_name}'",
                },
                {
                    "choice": UserChoice.CREATE_NEW.value,
                    "label": "Create new project",
                    "description": "Create as separate project anyway",
                },
                {
                    "choice": UserChoice.CANCEL.value,
                    "label": "Cancel",
                    "description": "Abort project creation",
                },
            ],
            metadata={
                "scope_changes": scope_changes,
                "fingerprint": new_fingerprint,
                "intent": new_intent.to_dict(),
            },
        )

    @classmethod
    def _handle_moderate_similarity(
        cls,
        existing: ProjectIdentity,
        project_name: str,
        new_intent: NormalizedIntent,
        new_fingerprint: str,
        similarity: float,
    ) -> DecisionResult:
        """Handle moderate similarity - may be related project."""
        return DecisionResult(
            decision=DecisionType.CONFLICT_DETECTED,
            confidence=0.7,
            explanation=(
                f"Moderate similarity ({similarity:.0%}) with '{project_name}'. "
                "These projects may be related but serve different purposes."
            ),
            requires_user_confirmation=True,
            conflict_type=ConflictType.SIMILAR_PURPOSE,
            existing_project_id=existing.project_id,
            existing_project_name=project_name,
            similarity_score=similarity,
            suggested_action="Review if this is an extension of the existing project or truly new.",
            alternatives=[
                {
                    "choice": UserChoice.CREATE_NEW.value,
                    "label": "Create new project",
                    "description": "This is a different project",
                },
                {
                    "choice": UserChoice.ADD_MODULE.value,
                    "label": "Add as module",
                    "description": f"Add as a new module to '{project_name}'",
                },
                {
                    "choice": UserChoice.IMPROVE_EXISTING.value,
                    "label": "Improve existing",
                    "description": f"Modify '{project_name}'",
                },
                {
                    "choice": UserChoice.CANCEL.value,
                    "label": "Cancel",
                    "description": "Abort and reconsider",
                },
            ],
            metadata={
                "fingerprint": new_fingerprint,
                "intent": new_intent.to_dict(),
            },
        )

    @classmethod
    def _handle_low_similarity(
        cls,
        existing: ProjectIdentity,
        project_name: str,
        new_intent: NormalizedIntent,
        new_fingerprint: str,
        similarity: float,
    ) -> DecisionResult:
        """Handle low similarity - likely different project."""
        return DecisionResult(
            decision=DecisionType.NEW_PROJECT,
            confidence=0.8,
            explanation=(
                f"Low similarity ({similarity:.0%}) with nearest project '{project_name}'. "
                "This appears to be a new project."
            ),
            requires_user_confirmation=False,
            conflict_type=ConflictType.NONE,
            existing_project_id=existing.project_id,
            existing_project_name=project_name,
            similarity_score=similarity,
            metadata={
                "fingerprint": new_fingerprint,
                "intent": new_intent.to_dict(),
            },
        )

    @classmethod
    def resolve_with_user_choice(
        cls,
        original_result: DecisionResult,
        user_choice: UserChoice,
    ) -> DecisionResult:
        """
        Resolve a conflict based on user choice.

        Args:
            original_result: The original decision result with conflict
            user_choice: The user's choice for resolution

        Returns:
            New DecisionResult with resolved action
        """
        if user_choice == UserChoice.IMPROVE_EXISTING:
            return DecisionResult(
                decision=DecisionType.CHANGE_MODE,
                confidence=1.0,
                explanation=(
                    f"User chose to improve existing project. "
                    f"Starting CHANGE_MODE on '{original_result.existing_project_name}'."
                ),
                requires_user_confirmation=False,
                conflict_type=original_result.conflict_type,
                existing_project_id=original_result.existing_project_id,
                existing_project_name=original_result.existing_project_name,
                metadata={
                    "user_choice": user_choice.value,
                    "original_decision": original_result.decision.value,
                },
            )

        elif user_choice == UserChoice.ADD_MODULE:
            return DecisionResult(
                decision=DecisionType.CHANGE_MODE,
                confidence=1.0,
                explanation=(
                    f"User chose to add new module. "
                    f"Starting CHANGE_MODE on '{original_result.existing_project_name}' "
                    "with module addition."
                ),
                requires_user_confirmation=False,
                conflict_type=original_result.conflict_type,
                existing_project_id=original_result.existing_project_id,
                existing_project_name=original_result.existing_project_name,
                metadata={
                    "user_choice": user_choice.value,
                    "change_type": "add_module",
                    "original_decision": original_result.decision.value,
                },
            )

        elif user_choice == UserChoice.CREATE_VERSION:
            return DecisionResult(
                decision=DecisionType.NEW_VERSION,
                confidence=1.0,
                explanation=(
                    f"User chose to create new version. "
                    f"Creating v2 of '{original_result.existing_project_name}'."
                ),
                requires_user_confirmation=False,
                conflict_type=original_result.conflict_type,
                existing_project_id=original_result.existing_project_id,
                existing_project_name=original_result.existing_project_name,
                metadata={
                    "user_choice": user_choice.value,
                    "version": "v2",
                    "original_decision": original_result.decision.value,
                },
            )

        elif user_choice == UserChoice.CREATE_NEW:
            return DecisionResult(
                decision=DecisionType.NEW_PROJECT,
                confidence=1.0,
                explanation=(
                    "User explicitly chose to create new project despite similarity."
                ),
                requires_user_confirmation=False,
                conflict_type=original_result.conflict_type,
                existing_project_id=original_result.existing_project_id,
                existing_project_name=original_result.existing_project_name,
                metadata={
                    "user_choice": user_choice.value,
                    "forced": True,
                    "original_decision": original_result.decision.value,
                },
            )

        else:  # CANCEL
            return DecisionResult(
                decision=DecisionType.CONFLICT_DETECTED,
                confidence=1.0,
                explanation="User cancelled project creation.",
                requires_user_confirmation=False,
                conflict_type=ConflictType.NONE,
                metadata={
                    "user_choice": user_choice.value,
                    "cancelled": True,
                },
            )


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------
def evaluate_project_creation(
    description: str,
    requirements: Optional[str] = None,
    tech_stack: Optional[Dict[str, Any]] = None,
    aspects: Optional[List[str]] = None,
    existing_identities: Optional[List[Tuple[ProjectIdentity, str]]] = None,
) -> DecisionResult:
    """
    Evaluate a project creation request.

    Args:
        description: Project description
        requirements: Raw requirements
        tech_stack: Technology stack
        aspects: Project aspects
        existing_identities: List of (identity, project_name) tuples

    Returns:
        DecisionResult with recommended action
    """
    return ProjectDecisionEngine.evaluate(
        new_description=description,
        new_requirements=requirements,
        new_tech_stack=tech_stack,
        new_aspects=aspects,
        existing_identities=existing_identities or [],
    )


def resolve_conflict(
    original_result: DecisionResult,
    user_choice: str,
) -> DecisionResult:
    """
    Resolve a conflict based on user choice.

    Args:
        original_result: The original decision result
        user_choice: String value of UserChoice enum

    Returns:
        Resolved DecisionResult
    """
    try:
        choice = UserChoice(user_choice)
    except ValueError:
        raise ValueError(f"Invalid user choice: {user_choice}")

    return ProjectDecisionEngine.resolve_with_user_choice(original_result, choice)
