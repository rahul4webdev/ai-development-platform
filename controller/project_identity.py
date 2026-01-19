"""
Phase 16E: Project Identity Engine

Deterministic fingerprinting and identity resolution for projects.

This module provides:
1. ProjectIdentity - Immutable identity dataclass
2. Fingerprint generation from semantic project attributes
3. Intent normalization for comparison

HARD CONSTRAINTS:
- Fingerprint MUST be deterministic (same input = same output)
- Fingerprint MUST be stable (order-independent)
- Fingerprint MUST NOT include: repo URLs, secrets, domains, paths, credentials, env vars
- Fingerprint MUST be derived ONLY from semantic project attributes

FINGERPRINT SOURCES (ALLOWED):
- Project purpose / problem statement
- Functional scope (modules, aspects)
- Domain topology (api / admin / frontend)
- Database type (mysql / postgres / mongo / sqlite)
- Architecture class (monolith / service / api-only)
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Set, FrozenSet, Tuple

logger = logging.getLogger("project_identity")


# -----------------------------------------------------------------------------
# Enums for Classification
# -----------------------------------------------------------------------------
class ArchitectureClass(str, Enum):
    """Architecture classification for projects."""
    MONOLITH = "monolith"
    MICROSERVICES = "microservices"
    API_ONLY = "api_only"
    FULLSTACK = "fullstack"
    FRONTEND_ONLY = "frontend_only"
    BACKEND_ONLY = "backend_only"
    UNKNOWN = "unknown"


class DatabaseType(str, Enum):
    """Database type classification."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    SQLITE = "sqlite"
    REDIS = "redis"
    ELASTICSEARCH = "elasticsearch"
    NONE = "none"
    UNKNOWN = "unknown"


class DomainTopology(str, Enum):
    """Domain/aspect topology."""
    API = "api"
    ADMIN = "admin"
    FRONTEND = "frontend"
    BACKEND = "backend"
    CORE = "core"
    MOBILE = "mobile"
    WEB = "web"


# -----------------------------------------------------------------------------
# Normalized Intent
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class NormalizedIntent:
    """
    Normalized representation of project intent.

    This is the semantic essence of what the project does,
    stripped of implementation details that don't affect identity.
    """
    purpose_keywords: FrozenSet[str]  # Core purpose keywords
    functional_modules: FrozenSet[str]  # Major functional areas
    domain_topology: FrozenSet[str]  # api, admin, frontend, etc.
    database_type: str  # Primary database
    architecture_class: str  # monolith, microservices, etc.
    target_users: FrozenSet[str]  # Who uses this (customers, admins, etc.)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "purpose_keywords": sorted(self.purpose_keywords),
            "functional_modules": sorted(self.functional_modules),
            "domain_topology": sorted(self.domain_topology),
            "database_type": self.database_type,
            "architecture_class": self.architecture_class,
            "target_users": sorted(self.target_users),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NormalizedIntent":
        """Create from dictionary."""
        return cls(
            purpose_keywords=frozenset(data.get("purpose_keywords", [])),
            functional_modules=frozenset(data.get("functional_modules", [])),
            domain_topology=frozenset(data.get("domain_topology", [])),
            database_type=data.get("database_type", "unknown"),
            architecture_class=data.get("architecture_class", "unknown"),
            target_users=frozenset(data.get("target_users", [])),
        )


# -----------------------------------------------------------------------------
# Project Identity
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ProjectIdentity:
    """
    Immutable project identity.

    This represents the unique semantic identity of a project,
    independent of its name, location, or implementation details.
    """
    project_id: str
    fingerprint: str
    normalized_intent: NormalizedIntent
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "project_id": self.project_id,
            "fingerprint": self.fingerprint,
            "normalized_intent": self.normalized_intent.to_dict(),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectIdentity":
        """Create from dictionary."""
        return cls(
            project_id=data["project_id"],
            fingerprint=data["fingerprint"],
            normalized_intent=NormalizedIntent.from_dict(data["normalized_intent"]),
            created_at=data["created_at"],
        )


# -----------------------------------------------------------------------------
# Intent Extraction
# -----------------------------------------------------------------------------
class IntentExtractor:
    """
    Extracts normalized intent from project description and requirements.

    Uses keyword-based classification to identify:
    - Purpose (what the project does)
    - Functional modules (major features)
    - Domain topology (aspects)
    - Database type
    - Architecture class
    - Target users
    """

    # Purpose keyword patterns
    PURPOSE_PATTERNS = {
        "ecommerce": ["ecommerce", "e-commerce", "online store", "shopping", "cart", "checkout", "product catalog"],
        "crm": ["crm", "customer relationship", "lead management", "sales pipeline", "contact management"],
        "cms": ["cms", "content management", "blog", "articles", "posts", "pages"],
        "erp": ["erp", "enterprise resource", "inventory", "procurement", "supply chain"],
        "social": ["social", "community", "forum", "messaging", "chat", "posts", "comments", "likes"],
        "analytics": ["analytics", "dashboard", "reporting", "metrics", "statistics", "insights"],
        "booking": ["booking", "reservation", "appointment", "scheduling", "calendar"],
        "education": ["education", "learning", "courses", "students", "teachers", "lms"],
        "healthcare": ["healthcare", "medical", "patient", "clinic", "hospital", "health"],
        "finance": ["finance", "banking", "payment", "transaction", "wallet", "accounting"],
        "hr": ["hr", "human resources", "employee", "payroll", "recruitment", "attendance"],
        "project_management": ["project management", "task", "kanban", "sprint", "backlog", "agile"],
        "fitness": ["fitness", "workout", "exercise", "gym", "health tracker", "nutrition"],
        "food": ["food", "restaurant", "menu", "order", "delivery", "recipe"],
        "travel": ["travel", "hotel", "flight", "booking", "itinerary", "tourism"],
        "real_estate": ["real estate", "property", "listing", "rental", "housing"],
        "authentication": ["authentication", "auth", "login", "signup", "oauth", "sso"],
        "api": ["api", "rest", "graphql", "endpoint", "microservice"],
    }

    # Functional module patterns
    MODULE_PATTERNS = {
        "user_management": ["user", "users", "profile", "account", "registration"],
        "authentication": ["login", "logout", "auth", "password", "2fa", "oauth"],
        "authorization": ["role", "permission", "rbac", "access control"],
        "payment": ["payment", "stripe", "paypal", "checkout", "billing"],
        "notification": ["notification", "email", "sms", "push", "alert"],
        "search": ["search", "filter", "query", "elasticsearch"],
        "file_management": ["upload", "download", "file", "storage", "s3"],
        "reporting": ["report", "export", "csv", "pdf", "analytics"],
        "messaging": ["message", "chat", "inbox", "conversation"],
        "api_gateway": ["api gateway", "rate limit", "throttle"],
        "caching": ["cache", "redis", "memcached"],
        "queue": ["queue", "job", "worker", "background"],
        "logging": ["log", "audit", "tracking"],
        "admin_panel": ["admin", "backoffice", "management"],
        "dashboard": ["dashboard", "overview", "summary"],
    }

    # Database detection patterns
    DATABASE_PATTERNS = {
        DatabaseType.POSTGRESQL.value: ["postgresql", "postgres", "pg", "psql"],
        DatabaseType.MYSQL.value: ["mysql", "mariadb"],
        DatabaseType.MONGODB.value: ["mongodb", "mongo", "nosql", "document db"],
        DatabaseType.SQLITE.value: ["sqlite", "sqlite3"],
        DatabaseType.REDIS.value: ["redis"],
        DatabaseType.ELASTICSEARCH.value: ["elasticsearch", "elastic"],
    }

    # Architecture patterns
    ARCHITECTURE_PATTERNS = {
        ArchitectureClass.MICROSERVICES.value: ["microservice", "micro-service", "distributed", "service mesh"],
        ArchitectureClass.MONOLITH.value: ["monolith", "monolithic", "single app"],
        ArchitectureClass.API_ONLY.value: ["api only", "api-only", "headless", "backend only"],
        ArchitectureClass.FULLSTACK.value: ["fullstack", "full-stack", "full stack"],
        ArchitectureClass.FRONTEND_ONLY.value: ["frontend only", "spa", "single page"],
    }

    # Domain topology patterns
    TOPOLOGY_PATTERNS = {
        DomainTopology.API.value: ["api", "rest api", "graphql", "endpoint"],
        DomainTopology.ADMIN.value: ["admin", "backoffice", "back-office", "management panel"],
        DomainTopology.FRONTEND.value: ["frontend", "front-end", "ui", "user interface"],
        DomainTopology.BACKEND.value: ["backend", "back-end", "server"],
        DomainTopology.MOBILE.value: ["mobile", "ios", "android", "react native", "flutter"],
        DomainTopology.WEB.value: ["web", "website", "webapp", "web app"],
    }

    # Target user patterns
    USER_PATTERNS = {
        "customers": ["customer", "user", "client", "buyer", "shopper"],
        "admins": ["admin", "administrator", "moderator", "staff"],
        "developers": ["developer", "api user", "integration"],
        "internal": ["internal", "employee", "team member"],
        "public": ["public", "visitor", "guest"],
    }

    @classmethod
    def extract_intent(
        cls,
        description: str,
        requirements: Optional[str] = None,
        tech_stack: Optional[Dict[str, Any]] = None,
        aspects: Optional[List[str]] = None,
    ) -> NormalizedIntent:
        """
        Extract normalized intent from project inputs.

        Args:
            description: Project description
            requirements: Raw requirements text
            tech_stack: Technology stack configuration
            aspects: List of project aspects

        Returns:
            NormalizedIntent with extracted semantic information
        """
        # Combine all text for analysis
        full_text = cls._normalize_text(description)
        if requirements:
            full_text += " " + cls._normalize_text(requirements)
        if tech_stack:
            full_text += " " + cls._normalize_text(json.dumps(tech_stack))

        # Extract each component
        purpose_keywords = cls._extract_matches(full_text, cls.PURPOSE_PATTERNS)
        functional_modules = cls._extract_matches(full_text, cls.MODULE_PATTERNS)
        domain_topology = cls._extract_topology(full_text, aspects)
        database_type = cls._extract_database(full_text, tech_stack)
        architecture_class = cls._extract_architecture(full_text, tech_stack)
        target_users = cls._extract_matches(full_text, cls.USER_PATTERNS)

        return NormalizedIntent(
            purpose_keywords=frozenset(purpose_keywords),
            functional_modules=frozenset(functional_modules),
            domain_topology=frozenset(domain_topology),
            database_type=database_type,
            architecture_class=architecture_class,
            target_users=frozenset(target_users) if target_users else frozenset(["customers"]),
        )

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize text for pattern matching."""
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        # Remove paths
        text = re.sub(r'/[\w/.-]+', '', text)
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        # Remove special characters except spaces
        text = re.sub(r'[^a-z0-9\s-]', ' ', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @classmethod
    def _extract_matches(cls, text: str, patterns: Dict[str, List[str]]) -> Set[str]:
        """Extract matching categories from text."""
        matches = set()
        for category, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text:
                    matches.add(category)
                    break
        return matches

    @classmethod
    def _extract_topology(
        cls,
        text: str,
        aspects: Optional[List[str]] = None,
    ) -> Set[str]:
        """Extract domain topology from text and aspects."""
        topology = set()

        # From text patterns
        for domain, keywords in cls.TOPOLOGY_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    topology.add(domain)
                    break

        # From explicit aspects
        if aspects:
            aspect_mapping = {
                "api": DomainTopology.API.value,
                "admin": DomainTopology.ADMIN.value,
                "frontend": DomainTopology.FRONTEND.value,
                "backend": DomainTopology.BACKEND.value,
                "core": DomainTopology.BACKEND.value,
                "mobile": DomainTopology.MOBILE.value,
                "web": DomainTopology.WEB.value,
            }
            for aspect in aspects:
                if aspect.lower() in aspect_mapping:
                    topology.add(aspect_mapping[aspect.lower()])

        # Default if nothing found
        if not topology:
            topology.add(DomainTopology.BACKEND.value)

        return topology

    @classmethod
    def _extract_database(
        cls,
        text: str,
        tech_stack: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract primary database type."""
        # Check tech stack first
        if tech_stack:
            db_config = tech_stack.get("database", {})
            if isinstance(db_config, dict):
                db_type = db_config.get("type", "").lower()
            else:
                db_type = str(db_config).lower()

            for db_enum, keywords in cls.DATABASE_PATTERNS.items():
                for keyword in keywords:
                    if keyword in db_type:
                        return db_enum

        # Check text
        for db_enum, keywords in cls.DATABASE_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    return db_enum

        return DatabaseType.UNKNOWN.value

    @classmethod
    def _extract_architecture(
        cls,
        text: str,
        tech_stack: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Extract architecture class."""
        # Check tech stack
        if tech_stack:
            arch = tech_stack.get("architecture", "").lower()
            for arch_enum, keywords in cls.ARCHITECTURE_PATTERNS.items():
                for keyword in keywords:
                    if keyword in arch:
                        return arch_enum

        # Check text
        for arch_enum, keywords in cls.ARCHITECTURE_PATTERNS.items():
            for keyword in keywords:
                if keyword in text:
                    return arch_enum

        return ArchitectureClass.UNKNOWN.value


# -----------------------------------------------------------------------------
# Fingerprint Generator
# -----------------------------------------------------------------------------
class FingerprintGenerator:
    """
    Generates deterministic, stable fingerprints from normalized intent.

    PROPERTIES:
    - Deterministic: Same input always produces same output
    - Stable: Order of elements doesn't affect result
    - Semantic: Based on meaning, not syntax
    - Secure: Uses SHA-256 hashing
    """

    @classmethod
    def generate(cls, intent: NormalizedIntent) -> str:
        """
        Generate fingerprint from normalized intent.

        The fingerprint is a SHA-256 hash of the canonical representation
        of the normalized intent.
        """
        # Build canonical representation (sorted for stability)
        canonical = cls._build_canonical(intent)

        # Generate SHA-256 hash
        fingerprint = hashlib.sha256(canonical.encode('utf-8')).hexdigest()

        logger.debug(f"Generated fingerprint: {fingerprint[:16]}... from intent")
        return fingerprint

    @classmethod
    def _build_canonical(cls, intent: NormalizedIntent) -> str:
        """
        Build canonical string representation of intent.

        All collections are sorted to ensure order-independence.
        """
        parts = [
            f"purpose:{','.join(sorted(intent.purpose_keywords))}",
            f"modules:{','.join(sorted(intent.functional_modules))}",
            f"topology:{','.join(sorted(intent.domain_topology))}",
            f"database:{intent.database_type}",
            f"architecture:{intent.architecture_class}",
            f"users:{','.join(sorted(intent.target_users))}",
        ]
        return "|".join(parts)

    @classmethod
    def compute_similarity(
        cls,
        intent1: NormalizedIntent,
        intent2: NormalizedIntent,
    ) -> float:
        """
        Compute similarity score between two intents.

        Returns a score from 0.0 (completely different) to 1.0 (identical).
        """
        scores = []

        # Purpose similarity (high weight)
        purpose_sim = cls._set_similarity(
            intent1.purpose_keywords,
            intent2.purpose_keywords,
        )
        scores.append((purpose_sim, 0.30))

        # Module similarity (high weight)
        module_sim = cls._set_similarity(
            intent1.functional_modules,
            intent2.functional_modules,
        )
        scores.append((module_sim, 0.25))

        # Topology similarity (medium weight)
        topology_sim = cls._set_similarity(
            intent1.domain_topology,
            intent2.domain_topology,
        )
        scores.append((topology_sim, 0.15))

        # Database match (medium weight)
        db_match = 1.0 if intent1.database_type == intent2.database_type else 0.0
        scores.append((db_match, 0.15))

        # Architecture match (medium weight)
        arch_match = 1.0 if intent1.architecture_class == intent2.architecture_class else 0.0
        scores.append((arch_match, 0.10))

        # User similarity (low weight)
        user_sim = cls._set_similarity(
            intent1.target_users,
            intent2.target_users,
        )
        scores.append((user_sim, 0.05))

        # Weighted average
        total = sum(score * weight for score, weight in scores)
        return round(total, 4)

    @classmethod
    def _set_similarity(cls, set1: FrozenSet[str], set2: FrozenSet[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0


# -----------------------------------------------------------------------------
# Identity Manager
# -----------------------------------------------------------------------------
class ProjectIdentityManager:
    """
    Manages project identity creation and comparison.

    Provides high-level methods for:
    - Creating identities from project data
    - Comparing identities for conflicts
    - Finding similar projects
    """

    @classmethod
    def create_identity(
        cls,
        project_id: str,
        description: str,
        requirements: Optional[str] = None,
        tech_stack: Optional[Dict[str, Any]] = None,
        aspects: Optional[List[str]] = None,
    ) -> ProjectIdentity:
        """
        Create a project identity from project data.

        Args:
            project_id: Unique project identifier
            description: Project description
            requirements: Raw requirements text
            tech_stack: Technology stack configuration
            aspects: List of project aspects

        Returns:
            ProjectIdentity with computed fingerprint
        """
        # Extract normalized intent
        intent = IntentExtractor.extract_intent(
            description=description,
            requirements=requirements,
            tech_stack=tech_stack,
            aspects=aspects,
        )

        # Generate fingerprint
        fingerprint = FingerprintGenerator.generate(intent)

        # Create identity
        identity = ProjectIdentity(
            project_id=project_id,
            fingerprint=fingerprint,
            normalized_intent=intent,
            created_at=datetime.utcnow().isoformat(),
        )

        logger.info(
            f"Created identity for project {project_id}: "
            f"fingerprint={fingerprint[:16]}..."
        )

        return identity

    @classmethod
    def compare_identities(
        cls,
        identity1: ProjectIdentity,
        identity2: ProjectIdentity,
    ) -> Tuple[bool, float, str]:
        """
        Compare two project identities.

        Returns:
            Tuple of (exact_match, similarity_score, explanation)
        """
        # Check exact fingerprint match
        if identity1.fingerprint == identity2.fingerprint:
            return True, 1.0, "Exact fingerprint match - identical semantic intent"

        # Compute similarity
        similarity = FingerprintGenerator.compute_similarity(
            identity1.normalized_intent,
            identity2.normalized_intent,
        )

        # Generate explanation
        if similarity >= 0.9:
            explanation = "Very high similarity - likely the same project with minor variations"
        elif similarity >= 0.7:
            explanation = "High similarity - similar purpose and modules, may be related"
        elif similarity >= 0.5:
            explanation = "Moderate similarity - some overlap in functionality"
        elif similarity >= 0.3:
            explanation = "Low similarity - different purpose but shared modules"
        else:
            explanation = "Minimal similarity - likely unrelated projects"

        return False, similarity, explanation

    @classmethod
    def find_similar_in_registry(
        cls,
        identity: ProjectIdentity,
        registry_identities: List[ProjectIdentity],
        threshold: float = 0.7,
    ) -> List[Tuple[ProjectIdentity, float, str]]:
        """
        Find similar projects in the registry.

        Args:
            identity: Identity to compare
            registry_identities: List of existing identities
            threshold: Minimum similarity score to include

        Returns:
            List of (matching_identity, similarity_score, explanation)
        """
        matches = []

        for existing in registry_identities:
            exact, similarity, explanation = cls.compare_identities(identity, existing)

            if exact:
                matches.append((existing, 1.0, "Exact match"))
            elif similarity >= threshold:
                matches.append((existing, similarity, explanation))

        # Sort by similarity descending
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------
def create_project_identity(
    project_id: str,
    description: str,
    requirements: Optional[str] = None,
    tech_stack: Optional[Dict[str, Any]] = None,
    aspects: Optional[List[str]] = None,
) -> ProjectIdentity:
    """Create a project identity."""
    return ProjectIdentityManager.create_identity(
        project_id=project_id,
        description=description,
        requirements=requirements,
        tech_stack=tech_stack,
        aspects=aspects,
    )


def compare_project_identities(
    identity1: ProjectIdentity,
    identity2: ProjectIdentity,
) -> Tuple[bool, float, str]:
    """Compare two project identities."""
    return ProjectIdentityManager.compare_identities(identity1, identity2)


def extract_normalized_intent(
    description: str,
    requirements: Optional[str] = None,
    tech_stack: Optional[Dict[str, Any]] = None,
    aspects: Optional[List[str]] = None,
) -> NormalizedIntent:
    """Extract normalized intent from project data."""
    return IntentExtractor.extract_intent(
        description=description,
        requirements=requirements,
        tech_stack=tech_stack,
        aspects=aspects,
    )


def generate_fingerprint(intent: NormalizedIntent) -> str:
    """Generate fingerprint from normalized intent."""
    return FingerprintGenerator.generate(intent)


def compute_intent_similarity(
    intent1: NormalizedIntent,
    intent2: NormalizedIntent,
) -> float:
    """Compute similarity between two normalized intents."""
    return FingerprintGenerator.compute_similarity(intent1, intent2)
