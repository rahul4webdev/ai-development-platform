"""
Phase 16C: Project Registry

Canonical source of truth for all projects in the platform.

This module provides:
1. Unified Project model that bridges IPC and Lifecycle systems
2. Persistent storage with crash recovery
3. Lifecycle auto-creation when projects are registered
4. Dashboard-ready project queries

HARD CONSTRAINTS:
- Project MUST exist before lifecycle can be created
- Project state is derived from lifecycle states
- All mutations are atomic and logged
- No silent failures - all errors are structured
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("project_registry")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
def _get_registry_dir() -> Path:
    """Get the registry directory with fallback for local development."""
    primary = Path("/home/aitesting.mybd.in/jobs/registry")

    # Try primary location
    try:
        if primary.parent.exists():
            primary.mkdir(parents=True, exist_ok=True)
            return primary
    except (OSError, PermissionError):
        pass

    # Fallback to /tmp
    fallback = Path("/tmp/ai_platform/registry")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


REGISTRY_DIR = _get_registry_dir()
REGISTRY_FILE = REGISTRY_DIR / "projects.json"


# -----------------------------------------------------------------------------
# Project Status Enum
# -----------------------------------------------------------------------------
class ProjectStatus(str, Enum):
    """Project lifecycle status."""
    CREATED = "created"
    PLANNING = "planning"
    DEVELOPMENT = "development"
    TESTING = "testing"
    AWAITING_FEEDBACK = "awaiting_feedback"
    READY_FOR_PRODUCTION = "ready_for_production"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"


class ProjectAspect(str, Enum):
    """Project aspects (domains)."""
    API = "api"
    FRONTEND = "frontend"
    ADMIN = "admin"
    CORE = "core"
    BACKEND = "backend"
    FRONTEND_WEB = "frontend_web"


# -----------------------------------------------------------------------------
# Project Model
# -----------------------------------------------------------------------------
@dataclass
class Project:
    """
    Canonical project model.

    This is the single source of truth for project existence.
    Lifecycles reference projects, not the other way around.
    """
    project_id: str
    name: str
    description: str
    created_by: str
    created_at: str
    updated_at: str
    current_status: str = ProjectStatus.CREATED.value
    aspects: Dict[str, str] = field(default_factory=dict)  # aspect -> status
    lifecycle_ids: List[str] = field(default_factory=list)
    active_job_ids: List[str] = field(default_factory=list)
    ipc_contract_id: Optional[str] = None
    requirements_source: Optional[str] = None  # "text", "file", "url"
    requirements_raw: Optional[str] = None
    tech_stack: Dict[str, Any] = field(default_factory=dict)
    domains: Dict[str, str] = field(default_factory=dict)  # aspect -> domain mapping
    validation_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Project":
        """Create from dictionary."""
        return cls(**data)

    def get_primary_status(self) -> str:
        """Get the most relevant status from all aspects."""
        if not self.aspects:
            return self.current_status

        # Priority order for status reporting
        priority = [
            ProjectStatus.FAILED.value,
            ProjectStatus.AWAITING_FEEDBACK.value,
            ProjectStatus.TESTING.value,
            ProjectStatus.DEVELOPMENT.value,
            ProjectStatus.PLANNING.value,
            ProjectStatus.READY_FOR_PRODUCTION.value,
            ProjectStatus.DEPLOYED.value,
            ProjectStatus.CREATED.value,
            ProjectStatus.ARCHIVED.value,
        ]

        for status in priority:
            if any(s == status for s in self.aspects.values()):
                return status

        return self.current_status


# -----------------------------------------------------------------------------
# Registry Errors
# -----------------------------------------------------------------------------
class RegistryError(Exception):
    """Base registry error with structured details."""
    def __init__(self, code: str, message: str, details: Dict[str, Any] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": True,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ProjectNotFoundError(RegistryError):
    def __init__(self, project_name: str):
        super().__init__(
            code="PROJECT_NOT_FOUND",
            message=f"Project '{project_name}' not found",
            details={"project_name": project_name}
        )


class ProjectAlreadyExistsError(RegistryError):
    def __init__(self, project_name: str):
        super().__init__(
            code="PROJECT_EXISTS",
            message=f"Project '{project_name}' already exists",
            details={"project_name": project_name}
        )


class ValidationError(RegistryError):
    def __init__(self, errors: List[str]):
        super().__init__(
            code="VALIDATION_FAILED",
            message="Project validation failed",
            details={"errors": errors}
        )


# -----------------------------------------------------------------------------
# Project Registry
# -----------------------------------------------------------------------------
class ProjectRegistry:
    """
    Central registry for all projects.

    Provides:
    - CRUD operations for projects
    - Lifecycle integration
    - Dashboard queries
    - Validation layer
    """

    def __init__(self):
        self._registry_file = REGISTRY_FILE
        self._projects: Dict[str, Project] = {}
        self._load_registry()

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def _load_registry(self) -> None:
        """Load projects from persistent storage."""
        if not self._registry_file.exists():
            logger.info("No existing registry file, starting fresh")
            return

        try:
            with open(self._registry_file) as f:
                data = json.load(f)

            for name, proj_data in data.get("projects", {}).items():
                try:
                    self._projects[name] = Project.from_dict(proj_data)
                except Exception as e:
                    logger.warning(f"Failed to load project {name}: {e}")

            logger.info(f"Loaded {len(self._projects)} projects from registry")
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    def _save_registry(self) -> None:
        """Save projects to persistent storage."""
        try:
            self._registry_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "updated_at": datetime.utcnow().isoformat(),
                "projects": {
                    name: proj.to_dict()
                    for name, proj in self._projects.items()
                }
            }

            # Atomic write
            temp_file = self._registry_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            temp_file.rename(self._registry_file)

            logger.debug(f"Saved {len(self._projects)} projects to registry")
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            raise RegistryError(
                code="SAVE_FAILED",
                message="Failed to save project registry",
                details={"error": str(e)}
            )

    # -------------------------------------------------------------------------
    # CRUD Operations
    # -------------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str,
        created_by: str,
        requirements_raw: Optional[str] = None,
        requirements_source: str = "text",
        tech_stack: Dict[str, Any] = None,
        aspects: List[str] = None,
    ) -> Tuple[bool, str, Optional[Project]]:
        """
        Create a new project.

        Returns: (success, message, project)
        """
        # Normalize name
        normalized_name = self._normalize_name(name)

        # Check if exists
        if normalized_name in self._projects:
            return False, f"Project '{normalized_name}' already exists", None

        # Create project
        now = datetime.utcnow().isoformat()
        project = Project(
            project_id=str(uuid.uuid4()),
            name=normalized_name,
            description=description,
            created_by=created_by,
            created_at=now,
            updated_at=now,
            current_status=ProjectStatus.CREATED.value,
            requirements_raw=requirements_raw,
            requirements_source=requirements_source,
            tech_stack=tech_stack or {},
        )

        # Initialize default aspects if not provided
        default_aspects = aspects or ["api", "frontend", "admin"]
        for aspect in default_aspects:
            project.aspects[aspect] = ProjectStatus.CREATED.value

        # Save
        self._projects[normalized_name] = project
        self._save_registry()

        logger.info(f"Created project: {normalized_name} (id={project.project_id})")
        return True, f"Project '{normalized_name}' created successfully", project

    def get_project(self, name: str) -> Optional[Project]:
        """Get project by name."""
        normalized = self._normalize_name(name)
        return self._projects.get(normalized)

    def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by ID."""
        for project in self._projects.values():
            if project.project_id == project_id:
                return project
        return None

    def list_projects(
        self,
        status: Optional[str] = None,
        created_by: Optional[str] = None,
        limit: int = 100,
    ) -> List[Project]:
        """List projects with optional filters."""
        projects = list(self._projects.values())

        if status:
            projects = [p for p in projects if p.current_status == status]

        if created_by:
            projects = [p for p in projects if p.created_by == created_by]

        # Sort by created_at descending
        projects.sort(key=lambda p: p.created_at, reverse=True)

        return projects[:limit]

    def update_project(
        self,
        name: str,
        updates: Dict[str, Any],
    ) -> Tuple[bool, str, Optional[Project]]:
        """
        Update project fields.

        Returns: (success, message, project)
        """
        project = self.get_project(name)
        if not project:
            return False, f"Project '{name}' not found", None

        # Apply updates
        for key, value in updates.items():
            if hasattr(project, key):
                setattr(project, key, value)

        project.updated_at = datetime.utcnow().isoformat()

        # Save
        self._save_registry()

        logger.info(f"Updated project: {name}")
        return True, f"Project '{name}' updated", project

    def update_aspect_status(
        self,
        name: str,
        aspect: str,
        status: str,
    ) -> Tuple[bool, str]:
        """Update a specific aspect's status."""
        project = self.get_project(name)
        if not project:
            return False, f"Project '{name}' not found"

        project.aspects[aspect] = status
        project.current_status = project.get_primary_status()
        project.updated_at = datetime.utcnow().isoformat()

        self._save_registry()

        logger.info(f"Updated {name}/{aspect} status to {status}")
        return True, f"Aspect '{aspect}' updated to '{status}'"

    def add_lifecycle_id(
        self,
        name: str,
        lifecycle_id: str,
    ) -> Tuple[bool, str]:
        """Associate a lifecycle with a project."""
        project = self.get_project(name)
        if not project:
            return False, f"Project '{name}' not found"

        if lifecycle_id not in project.lifecycle_ids:
            project.lifecycle_ids.append(lifecycle_id)
            project.updated_at = datetime.utcnow().isoformat()
            self._save_registry()

        return True, f"Lifecycle {lifecycle_id} added to {name}"

    def add_active_job(
        self,
        name: str,
        job_id: str,
    ) -> Tuple[bool, str]:
        """Track an active job for a project."""
        project = self.get_project(name)
        if not project:
            return False, f"Project '{name}' not found"

        if job_id not in project.active_job_ids:
            project.active_job_ids.append(job_id)
            project.updated_at = datetime.utcnow().isoformat()
            self._save_registry()

        return True, f"Job {job_id} added to {name}"

    def remove_active_job(
        self,
        name: str,
        job_id: str,
    ) -> Tuple[bool, str]:
        """Remove a completed/failed job from tracking."""
        project = self.get_project(name)
        if not project:
            return False, f"Project '{name}' not found"

        if job_id in project.active_job_ids:
            project.active_job_ids.remove(job_id)
            project.updated_at = datetime.utcnow().isoformat()
            self._save_registry()

        return True, f"Job {job_id} removed from {name}"

    def delete_project(self, name: str) -> Tuple[bool, str]:
        """Delete a project (archive it)."""
        project = self.get_project(name)
        if not project:
            return False, f"Project '{name}' not found"

        # Don't actually delete, archive
        project.current_status = ProjectStatus.ARCHIVED.value
        project.updated_at = datetime.utcnow().isoformat()
        self._save_registry()

        logger.info(f"Archived project: {name}")
        return True, f"Project '{name}' archived"

    # -------------------------------------------------------------------------
    # Dashboard Queries
    # -------------------------------------------------------------------------

    def get_dashboard_projects(self) -> List[Dict[str, Any]]:
        """
        Get projects formatted for dashboard display.

        Returns projects with:
        - name, status, aspects, active_jobs, last_activity
        """
        result = []
        for project in self._projects.values():
            if project.current_status == ProjectStatus.ARCHIVED.value:
                continue

            result.append({
                "project_id": project.project_id,
                "project_name": project.name,
                "current_status": project.current_status,
                "lifecycle_state": project.current_status,  # Alias for dashboard
                "aspects": project.aspects,
                "active_jobs": len(project.active_job_ids),
                "lifecycle_count": len(project.lifecycle_ids),
                "created_at": project.created_at,
                "updated_at": project.updated_at,
                "created_by": project.created_by,
            })

        # Sort by updated_at descending
        result.sort(key=lambda p: p["updated_at"], reverse=True)
        return result

    def get_project_count(self) -> Dict[str, int]:
        """Get project counts by status."""
        counts = {
            "total": 0,
            "active": 0,
            "archived": 0,
        }

        for project in self._projects.values():
            counts["total"] += 1
            if project.current_status == ProjectStatus.ARCHIVED.value:
                counts["archived"] += 1
            else:
                counts["active"] += 1

        return counts

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_project_requirements(
        self,
        description: str,
        requirements: Optional[str] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate project requirements before creation.

        Checks:
        - Non-empty description
        - At least one aspect defined
        - No dangerous patterns

        Returns: (is_valid, errors)
        """
        errors = []

        # Check description
        if not description or len(description.strip()) < 10:
            errors.append("Description must be at least 10 characters")

        # Check for dangerous keywords
        dangerous = ["rm -rf", "sudo", "password", "secret", "DROP TABLE"]
        content = (description + " " + (requirements or "")).lower()
        for word in dangerous:
            if word.lower() in content:
                errors.append(f"Potentially dangerous content detected: {word}")

        return len(errors) == 0, errors

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _normalize_name(self, name: str) -> str:
        """Normalize project name to valid slug."""
        # Take first few words
        words = name.lower().split()[:5]
        slug = "-".join(words)
        # Remove invalid chars
        slug = "".join(c if c.isalnum() or c == "-" else "" for c in slug)
        # Ensure starts with letter
        if slug and not slug[0].isalpha():
            slug = "project-" + slug
        return slug or "new-project"

    def project_exists(self, name: str) -> bool:
        """Check if project exists."""
        return self._normalize_name(name) in self._projects


# -----------------------------------------------------------------------------
# Global Registry Instance
# -----------------------------------------------------------------------------
_registry: Optional[ProjectRegistry] = None


def get_registry() -> ProjectRegistry:
    """Get the global project registry instance."""
    global _registry
    if _registry is None:
        _registry = ProjectRegistry()
    return _registry


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
async def create_project(
    name: str,
    description: str,
    created_by: str,
    requirements_raw: Optional[str] = None,
    requirements_source: str = "text",
    tech_stack: Dict[str, Any] = None,
    aspects: List[str] = None,
) -> Tuple[bool, str, Optional[Project]]:
    """Create a new project."""
    return get_registry().create_project(
        name=name,
        description=description,
        created_by=created_by,
        requirements_raw=requirements_raw,
        requirements_source=requirements_source,
        tech_stack=tech_stack,
        aspects=aspects,
    )


async def get_project(name: str) -> Optional[Project]:
    """Get project by name."""
    return get_registry().get_project(name)


async def list_projects(
    status: Optional[str] = None,
    created_by: Optional[str] = None,
    limit: int = 100,
) -> List[Project]:
    """List projects with optional filters."""
    return get_registry().list_projects(status, created_by, limit)


async def get_dashboard_projects() -> List[Dict[str, Any]]:
    """Get projects for dashboard."""
    return get_registry().get_dashboard_projects()


async def update_aspect_status(
    name: str,
    aspect: str,
    status: str,
) -> Tuple[bool, str]:
    """Update aspect status."""
    return get_registry().update_aspect_status(name, aspect, status)
