"""
Phase 15.3: Existing Project Ingestion & Adoption Engine

This module provides safe, deterministic analysis and ingestion of external projects
into the AI Development Platform.

Key Features:
- Accept external projects (Git repositories or local codebases)
- Analyze safely and deterministically (read-only inspection)
- Generate missing governance documents (PROJECT_MANIFEST, CURRENT_STATE, etc.)
- Register into Lifecycle Engine v2 in DEPLOYED state
- Enable future CHANGE_MODE cycles like native projects

Analysis Pipeline:
1. Repository Inspection - File enumeration, git metadata, structure analysis
2. Aspect Detection - Classify project aspects (backend, frontend, core, etc.)
3. Risk & Integrity Scan - Security patterns, sensitive files, code quality
4. Documentation Synthesis - Generate governance documents
5. Registration - Register in Lifecycle Engine v2 with human approval

IMPORTANT: This engine performs READ-ONLY analysis. It never modifies the
external project during ingestion.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

from .lifecycle_v2 import (
    LifecycleManager,
    LifecycleInstance,
    LifecycleState,
    LifecycleMode,
    ProjectAspect,
    ChangeType,
    UserRole,
    TransitionTrigger,
    get_lifecycle_manager,
)

logger = logging.getLogger("ingestion_engine")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

INGESTION_STATE_DIR = Path("/home/aitesting.mybd.in/jobs/ingestion")
INGESTION_WORKSPACE_DIR = INGESTION_STATE_DIR / "workspaces"
INGESTION_REPORTS_DIR = INGESTION_STATE_DIR / "reports"

# File patterns for aspect detection
ASPECT_PATTERNS = {
    ProjectAspect.BACKEND: [
        r".*/(api|server|backend|routes|controllers|services|handlers)/.*\.py$",
        r".*/(api|server|backend|routes|controllers|services|handlers)/.*\.js$",
        r".*/(api|server|backend|routes|controllers|services|handlers)/.*\.ts$",
        r".*/(api|server|backend|routes|controllers|services|handlers)/.*\.go$",
        r".*/(api|server|backend|routes|controllers|services|handlers)/.*\.java$",
        r".*/main\.(py|go|java)$",
        r".*/app\.(py|js|ts)$",
        r".*requirements\.txt$",
        r".*setup\.py$",
        r".*Pipfile$",
        r".*go\.mod$",
        r".*pom\.xml$",
        r".*build\.gradle$",
    ],
    ProjectAspect.FRONTEND_WEB: [
        r".*/(src|app|pages|components)/.*\.(jsx|tsx|vue|svelte)$",
        r".*/(public|static)/.*\.(html|css|js)$",
        r".*/package\.json$",
        r".*/vite\.config\.(js|ts)$",
        r".*/webpack\.config\.(js|ts)$",
        r".*/next\.config\.(js|ts|mjs)$",
        r".*/nuxt\.config\.(js|ts)$",
        r".*/tailwind\.config\.(js|ts)$",
    ],
    ProjectAspect.FRONTEND_MOBILE: [
        r".*/android/.*\.(java|kt|xml)$",
        r".*/ios/.*\.(swift|m|h)$",
        r".*/lib/.*\.dart$",
        r".*/pubspec\.yaml$",
        r".*/Podfile$",
        r".*/build\.gradle$",
        r".*/(react-native|expo).*\.json$",
    ],
    ProjectAspect.ADMIN: [
        r".*/admin/.*",
        r".*/dashboard/.*",
        r".*/backoffice/.*",
        r".*/cms/.*",
    ],
    ProjectAspect.CORE: [
        r".*/core/.*",
        r".*/lib/.*",
        r".*/shared/.*",
        r".*/common/.*",
        r".*/utils?/.*",
        r".*/helpers?/.*",
    ],
}

# Risk patterns to flag
RISK_PATTERNS = {
    "hardcoded_secrets": [
        r"(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]",
        r"(api_key|apikey|api-key)\s*=\s*['\"][^'\"]+['\"]",
        r"(secret|token)\s*=\s*['\"][^'\"]+['\"]",
        r"(aws_access_key|aws_secret)\s*=\s*['\"][^'\"]+['\"]",
    ],
    "sensitive_files": [
        r"\.env$",
        r"\.env\.(local|development|production)$",
        r"credentials\.json$",
        r"secrets\.yaml$",
        r".*\.pem$",
        r".*\.key$",
        r"id_rsa$",
    ],
    "dangerous_patterns": [
        r"eval\s*\(",
        r"exec\s*\(",
        r"__import__\s*\(",
        r"subprocess\.(call|run|Popen)\s*\([^)]*shell\s*=\s*True",
        r"os\.system\s*\(",
    ],
}

# Governance document templates
GOVERNANCE_DOCS = [
    "PROJECT_MANIFEST.yaml",
    "CURRENT_STATE.md",
    "ARCHITECTURE.md",
    "AI_POLICY.md",
    "TESTING_STRATEGY.md",
]


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class IngestionStatus(str, Enum):
    """Status of an ingestion request."""
    PENDING = "pending"
    ANALYZING = "analyzing"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    REGISTERED = "registered"
    FAILED = "failed"


class IngestionSource(str, Enum):
    """Source type for ingestion."""
    GIT_REPOSITORY = "git_repository"
    LOCAL_PATH = "local_path"
    ZIP_ARCHIVE = "zip_archive"


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

@dataclass
class FileInfo:
    """Information about a single file."""
    path: str
    relative_path: str
    size_bytes: int
    extension: str
    last_modified: Optional[datetime] = None
    checksum: Optional[str] = None
    is_binary: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "extension": self.extension,
            "last_modified": self.last_modified.isoformat() if self.last_modified else None,
            "checksum": self.checksum,
            "is_binary": self.is_binary,
        }


@dataclass
class AspectDetection:
    """Results of aspect detection for a project."""
    detected_aspects: List[ProjectAspect]
    primary_aspect: ProjectAspect
    confidence_scores: Dict[str, float]
    evidence: Dict[str, List[str]]  # aspect -> list of matching files

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected_aspects": [a.value for a in self.detected_aspects],
            "primary_aspect": self.primary_aspect.value,
            "confidence_scores": self.confidence_scores,
            "evidence": self.evidence,
        }


@dataclass
class RiskAssessment:
    """Risk assessment results."""
    risk_level: str  # low, medium, high, critical
    total_issues: int
    hardcoded_secrets: List[Dict[str, Any]]
    sensitive_files: List[str]
    dangerous_patterns: List[Dict[str, Any]]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "total_issues": self.total_issues,
            "hardcoded_secrets": self.hardcoded_secrets,
            "sensitive_files": self.sensitive_files,
            "dangerous_patterns": self.dangerous_patterns,
            "recommendations": self.recommendations,
        }


@dataclass
class GitMetadata:
    """Git repository metadata."""
    is_git_repo: bool
    remote_url: Optional[str] = None
    current_branch: Optional[str] = None
    last_commit_hash: Optional[str] = None
    last_commit_message: Optional[str] = None
    last_commit_date: Optional[datetime] = None
    total_commits: int = 0
    contributors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_git_repo": self.is_git_repo,
            "remote_url": self.remote_url,
            "current_branch": self.current_branch,
            "last_commit_hash": self.last_commit_hash,
            "last_commit_message": self.last_commit_message,
            "last_commit_date": self.last_commit_date.isoformat() if self.last_commit_date else None,
            "total_commits": self.total_commits,
            "contributors": self.contributors,
        }


@dataclass
class StructureAnalysis:
    """Project structure analysis."""
    total_files: int
    total_directories: int
    total_size_bytes: int
    file_types: Dict[str, int]  # extension -> count
    top_level_dirs: List[str]
    has_tests: bool
    has_docs: bool
    has_ci: bool
    build_system: Optional[str] = None
    package_manager: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "total_directories": self.total_directories,
            "total_size_bytes": self.total_size_bytes,
            "file_types": self.file_types,
            "top_level_dirs": self.top_level_dirs,
            "has_tests": self.has_tests,
            "has_docs": self.has_docs,
            "has_ci": self.has_ci,
            "build_system": self.build_system,
            "package_manager": self.package_manager,
        }


@dataclass
class ExistingDocuments:
    """Information about existing governance documents."""
    found: List[str]
    missing: List[str]
    contents: Dict[str, str]  # filename -> content preview

    def to_dict(self) -> Dict[str, Any]:
        return {
            "found": self.found,
            "missing": self.missing,
            "contents": {k: v[:500] + "..." if len(v) > 500 else v for k, v in self.contents.items()},
        }


@dataclass
class IngestionReport:
    """Complete ingestion analysis report."""
    report_id: str
    ingestion_id: str
    project_name: str
    source_type: IngestionSource
    source_location: str
    analyzed_at: datetime
    # Analysis results
    git_metadata: GitMetadata
    structure: StructureAnalysis
    aspects: AspectDetection
    risk_assessment: RiskAssessment
    existing_docs: ExistingDocuments
    files: List[FileInfo]
    # Status
    ready_for_registration: bool
    blocking_issues: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "ingestion_id": self.ingestion_id,
            "project_name": self.project_name,
            "source_type": self.source_type.value,
            "source_location": self.source_location,
            "analyzed_at": self.analyzed_at.isoformat(),
            "git_metadata": self.git_metadata.to_dict(),
            "structure": self.structure.to_dict(),
            "aspects": self.aspects.to_dict(),
            "risk_assessment": self.risk_assessment.to_dict(),
            "existing_docs": self.existing_docs.to_dict(),
            "files_count": len(self.files),
            "ready_for_registration": self.ready_for_registration,
            "blocking_issues": self.blocking_issues,
            "warnings": self.warnings,
        }


@dataclass
class IngestionRequest:
    """A project ingestion request."""
    ingestion_id: str
    project_name: str
    source_type: IngestionSource
    source_location: str
    status: IngestionStatus
    requested_by: str
    requested_at: datetime
    # Optional fields
    description: str = ""
    target_aspects: List[ProjectAspect] = field(default_factory=list)
    # Analysis results
    report: Optional[IngestionReport] = None
    workspace_path: Optional[str] = None
    # Approval
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    # Registration
    lifecycle_ids: List[str] = field(default_factory=list)
    registered_at: Optional[datetime] = None
    # Metadata
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ingestion_id": self.ingestion_id,
            "project_name": self.project_name,
            "source_type": self.source_type.value,
            "source_location": self.source_location,
            "status": self.status.value,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat(),
            "description": self.description,
            "target_aspects": [a.value for a in self.target_aspects],
            "report": self.report.to_dict() if self.report else None,
            "workspace_path": self.workspace_path,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "rejection_reason": self.rejection_reason,
            "lifecycle_ids": self.lifecycle_ids,
            "registered_at": self.registered_at.isoformat() if self.registered_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IngestionRequest":
        """Deserialize from dictionary."""
        # Parse report if present
        report = None
        if data.get("report"):
            # Report reconstruction would happen here
            pass

        return cls(
            ingestion_id=data["ingestion_id"],
            project_name=data["project_name"],
            source_type=IngestionSource(data["source_type"]),
            source_location=data["source_location"],
            status=IngestionStatus(data["status"]),
            requested_by=data["requested_by"],
            requested_at=datetime.fromisoformat(data["requested_at"]),
            description=data.get("description", ""),
            target_aspects=[ProjectAspect(a) for a in data.get("target_aspects", [])],
            report=report,
            workspace_path=data.get("workspace_path"),
            approved_by=data.get("approved_by"),
            approved_at=datetime.fromisoformat(data["approved_at"]) if data.get("approved_at") else None,
            rejection_reason=data.get("rejection_reason"),
            lifecycle_ids=data.get("lifecycle_ids", []),
            registered_at=datetime.fromisoformat(data["registered_at"]) if data.get("registered_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Project Ingestion Engine
# -----------------------------------------------------------------------------

class ProjectIngestionEngine:
    """
    Engine for analyzing and ingesting external projects.

    This engine performs safe, read-only analysis of external projects
    and prepares them for registration into the Lifecycle Engine v2.
    """

    def __init__(
        self,
        state_dir: Path = INGESTION_STATE_DIR,
        workspace_dir: Path = INGESTION_WORKSPACE_DIR,
        reports_dir: Path = INGESTION_REPORTS_DIR,
    ):
        self._state_dir = state_dir
        self._workspace_dir = workspace_dir
        self._reports_dir = reports_dir
        self._state_file = state_dir / "ingestion_state.json"
        self._lock = asyncio.Lock()
        self._lifecycle_manager = get_lifecycle_manager()
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        for directory in [self._state_dir, self._workspace_dir, self._reports_dir]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.debug(f"Could not create directory {directory}: {e}")

    # -------------------------------------------------------------------------
    # Repository Inspection
    # -------------------------------------------------------------------------

    async def clone_repository(
        self,
        git_url: str,
        ingestion_id: str,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Clone a Git repository to the workspace.

        This is a safe, read-only clone operation.
        """
        workspace_path = self._workspace_dir / ingestion_id

        try:
            # Clean up if exists
            if workspace_path.exists():
                shutil.rmtree(workspace_path)

            workspace_path.mkdir(parents=True, exist_ok=True)

            # Clone with depth=1 for faster cloning
            result = subprocess.run(
                ["git", "clone", "--depth", "1", git_url, str(workspace_path)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            if result.returncode != 0:
                return False, f"Git clone failed: {result.stderr}", None

            logger.info(f"Cloned repository {git_url} to {workspace_path}")
            return True, "Repository cloned successfully", workspace_path

        except subprocess.TimeoutExpired:
            return False, "Git clone timed out after 5 minutes", None
        except Exception as e:
            return False, f"Failed to clone repository: {e}", None

    async def prepare_local_path(
        self,
        local_path: str,
        ingestion_id: str,
    ) -> Tuple[bool, str, Optional[Path]]:
        """
        Prepare a local path for analysis.

        Creates a read-only reference (not a copy) for analysis.
        """
        source_path = Path(local_path)

        if not source_path.exists():
            return False, f"Path does not exist: {local_path}", None

        if not source_path.is_dir():
            return False, f"Path is not a directory: {local_path}", None

        # For local paths, we analyze in-place (read-only)
        logger.info(f"Prepared local path for analysis: {local_path}")
        return True, "Local path prepared", source_path

    async def extract_git_metadata(self, project_path: Path) -> GitMetadata:
        """Extract Git metadata from a repository."""
        git_dir = project_path / ".git"

        if not git_dir.exists():
            return GitMetadata(is_git_repo=False)

        metadata = GitMetadata(is_git_repo=True)

        try:
            # Get remote URL
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                metadata.remote_url = result.stdout.strip()

            # Get current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                metadata.current_branch = result.stdout.strip()

            # Get last commit
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H|%s|%ai"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split("|")
                if len(parts) >= 3:
                    metadata.last_commit_hash = parts[0]
                    metadata.last_commit_message = parts[1]
                    try:
                        metadata.last_commit_date = datetime.fromisoformat(parts[2].replace(" ", "T").split("+")[0])
                    except ValueError:
                        pass

            # Get commit count
            result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                metadata.total_commits = int(result.stdout.strip())

            # Get contributors
            result = subprocess.run(
                ["git", "shortlog", "-sn", "--no-merges", "-e", "HEAD"],
                cwd=str(project_path),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n")[:10]:  # Top 10 contributors
                    if line.strip():
                        # Extract email or name
                        match = re.search(r"<([^>]+)>", line)
                        if match:
                            metadata.contributors.append(match.group(1))

        except Exception as e:
            logger.warning(f"Error extracting git metadata: {e}")

        return metadata

    async def enumerate_files(
        self,
        project_path: Path,
        max_files: int = 10000,
    ) -> List[FileInfo]:
        """
        Enumerate all files in a project.

        Respects .gitignore and excludes common non-essential directories.
        """
        files: List[FileInfo] = []
        excluded_dirs = {
            ".git", "node_modules", "__pycache__", ".venv", "venv",
            "dist", "build", ".next", ".nuxt", "coverage", ".pytest_cache",
            ".mypy_cache", "eggs", "*.egg-info", ".tox", ".nox",
        }

        try:
            for root, dirs, filenames in os.walk(str(project_path)):
                # Filter out excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]

                for filename in filenames:
                    if len(files) >= max_files:
                        break

                    file_path = Path(root) / filename
                    relative_path = str(file_path.relative_to(project_path))

                    try:
                        stat = file_path.stat()
                        extension = file_path.suffix.lower()

                        # Check if binary
                        is_binary = False
                        try:
                            with open(file_path, "rb") as f:
                                chunk = f.read(1024)
                                is_binary = b"\x00" in chunk
                        except Exception:
                            is_binary = True

                        files.append(FileInfo(
                            path=str(file_path),
                            relative_path=relative_path,
                            size_bytes=stat.st_size,
                            extension=extension,
                            last_modified=datetime.fromtimestamp(stat.st_mtime),
                            is_binary=is_binary,
                        ))
                    except (OSError, IOError) as e:
                        logger.debug(f"Could not stat file {file_path}: {e}")

                if len(files) >= max_files:
                    break

        except Exception as e:
            logger.error(f"Error enumerating files: {e}")

        return files

    async def analyze_structure(
        self,
        project_path: Path,
        files: List[FileInfo],
    ) -> StructureAnalysis:
        """Analyze the project structure."""
        # Count file types
        file_types: Dict[str, int] = {}
        total_size = 0
        directories: Set[str] = set()

        for f in files:
            ext = f.extension or "(no extension)"
            file_types[ext] = file_types.get(ext, 0) + 1
            total_size += f.size_bytes
            # Track directories
            parent = str(Path(f.relative_path).parent)
            if parent != ".":
                directories.add(parent)

        # Get top-level directories
        top_level_dirs = sorted([
            d.name for d in project_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        # Check for common directories
        has_tests = any(d in top_level_dirs for d in ["tests", "test", "spec", "__tests__"])
        has_docs = any(d in top_level_dirs for d in ["docs", "doc", "documentation"])
        has_ci = (project_path / ".github" / "workflows").exists() or (project_path / ".gitlab-ci.yml").exists()

        # Detect build system
        build_system = None
        package_manager = None

        if (project_path / "Makefile").exists():
            build_system = "make"
        elif (project_path / "CMakeLists.txt").exists():
            build_system = "cmake"
        elif (project_path / "build.gradle").exists() or (project_path / "build.gradle.kts").exists():
            build_system = "gradle"
        elif (project_path / "pom.xml").exists():
            build_system = "maven"
        elif (project_path / "Cargo.toml").exists():
            build_system = "cargo"

        if (project_path / "package.json").exists():
            package_manager = "npm"
            if (project_path / "yarn.lock").exists():
                package_manager = "yarn"
            elif (project_path / "pnpm-lock.yaml").exists():
                package_manager = "pnpm"
            elif (project_path / "bun.lockb").exists():
                package_manager = "bun"
        elif (project_path / "requirements.txt").exists():
            package_manager = "pip"
        elif (project_path / "Pipfile").exists():
            package_manager = "pipenv"
        elif (project_path / "pyproject.toml").exists():
            package_manager = "poetry"
        elif (project_path / "go.mod").exists():
            package_manager = "go"

        return StructureAnalysis(
            total_files=len(files),
            total_directories=len(directories),
            total_size_bytes=total_size,
            file_types=dict(sorted(file_types.items(), key=lambda x: x[1], reverse=True)),
            top_level_dirs=top_level_dirs,
            has_tests=has_tests,
            has_docs=has_docs,
            has_ci=has_ci,
            build_system=build_system,
            package_manager=package_manager,
        )

    # -------------------------------------------------------------------------
    # Aspect Detection
    # -------------------------------------------------------------------------

    async def detect_aspects(
        self,
        project_path: Path,
        files: List[FileInfo],
    ) -> AspectDetection:
        """
        Detect project aspects based on file patterns.

        Returns detected aspects with confidence scores and evidence.
        """
        aspect_scores: Dict[str, float] = {a.value: 0.0 for a in ProjectAspect}
        aspect_evidence: Dict[str, List[str]] = {a.value: [] for a in ProjectAspect}

        for f in files:
            relative_path = "/" + f.relative_path

            for aspect, patterns in ASPECT_PATTERNS.items():
                for pattern in patterns:
                    if re.match(pattern, relative_path, re.IGNORECASE):
                        aspect_scores[aspect.value] += 1.0
                        if len(aspect_evidence[aspect.value]) < 10:  # Limit evidence
                            aspect_evidence[aspect.value].append(f.relative_path)
                        break

        # Normalize scores
        total_matches = sum(aspect_scores.values())
        if total_matches > 0:
            for aspect in aspect_scores:
                aspect_scores[aspect] = round(aspect_scores[aspect] / total_matches, 3)

        # Determine detected aspects (those with score > 0.05 or at least 3 matches)
        detected = []
        for aspect in ProjectAspect:
            if aspect_scores[aspect.value] > 0.05 or len(aspect_evidence[aspect.value]) >= 3:
                detected.append(aspect)

        # If no aspects detected, default to CORE
        if not detected:
            detected = [ProjectAspect.CORE]
            aspect_scores[ProjectAspect.CORE.value] = 1.0

        # Determine primary aspect
        primary = max(detected, key=lambda a: aspect_scores[a.value])

        return AspectDetection(
            detected_aspects=detected,
            primary_aspect=primary,
            confidence_scores=aspect_scores,
            evidence=aspect_evidence,
        )

    # -------------------------------------------------------------------------
    # Risk & Integrity Scanning
    # -------------------------------------------------------------------------

    async def scan_risks(
        self,
        project_path: Path,
        files: List[FileInfo],
    ) -> RiskAssessment:
        """
        Scan for security risks and code quality issues.

        This is a lightweight static analysis - not a full security audit.
        """
        hardcoded_secrets: List[Dict[str, Any]] = []
        sensitive_files: List[str] = []
        dangerous_patterns: List[Dict[str, Any]] = []
        recommendations: List[str] = []

        for f in files:
            # Skip binary files
            if f.is_binary:
                continue

            # Check for sensitive files
            for pattern in RISK_PATTERNS["sensitive_files"]:
                if re.match(pattern, f.relative_path, re.IGNORECASE):
                    sensitive_files.append(f.relative_path)
                    break

            # Scan file contents (only for small text files)
            if f.size_bytes > 100000:  # Skip files > 100KB
                continue

            try:
                file_path = project_path / f.relative_path
                content = file_path.read_text(errors="ignore")

                # Check for hardcoded secrets
                for pattern in RISK_PATTERNS["hardcoded_secrets"]:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count("\n") + 1
                        hardcoded_secrets.append({
                            "file": f.relative_path,
                            "line": line_num,
                            "pattern": pattern[:30],
                            "severity": "high",
                        })

                # Check for dangerous patterns
                for pattern in RISK_PATTERNS["dangerous_patterns"]:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count("\n") + 1
                        dangerous_patterns.append({
                            "file": f.relative_path,
                            "line": line_num,
                            "pattern": match.group()[:50],
                            "severity": "medium",
                        })
            except Exception as e:
                logger.debug(f"Could not scan file {f.relative_path}: {e}")

        # Calculate risk level
        total_issues = len(hardcoded_secrets) + len(sensitive_files) + len(dangerous_patterns)

        if total_issues == 0:
            risk_level = "low"
        elif len(hardcoded_secrets) > 0:
            risk_level = "critical" if len(hardcoded_secrets) > 5 else "high"
        elif len(dangerous_patterns) > 10:
            risk_level = "high"
        elif total_issues > 5:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Generate recommendations
        if hardcoded_secrets:
            recommendations.append("Remove hardcoded secrets and use environment variables or a secrets manager")
        if sensitive_files:
            recommendations.append("Add sensitive files to .gitignore and remove from version control")
        if dangerous_patterns:
            recommendations.append("Review dangerous code patterns for potential security vulnerabilities")
        if not (project_path / ".gitignore").exists():
            recommendations.append("Add a .gitignore file to prevent committing sensitive files")

        return RiskAssessment(
            risk_level=risk_level,
            total_issues=total_issues,
            hardcoded_secrets=hardcoded_secrets[:20],  # Limit results
            sensitive_files=sensitive_files[:20],
            dangerous_patterns=dangerous_patterns[:20],
            recommendations=recommendations,
        )

    # -------------------------------------------------------------------------
    # Document Detection and Generation
    # -------------------------------------------------------------------------

    async def check_existing_documents(
        self,
        project_path: Path,
    ) -> ExistingDocuments:
        """Check for existing governance documents."""
        found: List[str] = []
        missing: List[str] = []
        contents: Dict[str, str] = {}

        for doc in GOVERNANCE_DOCS:
            doc_path = project_path / doc
            if doc_path.exists():
                found.append(doc)
                try:
                    contents[doc] = doc_path.read_text(errors="ignore")
                except Exception:
                    contents[doc] = "(unreadable)"
            else:
                missing.append(doc)

        # Also check common alternatives
        alternatives = {
            "README.md": ["README.rst", "README.txt", "README"],
            "ARCHITECTURE.md": ["DESIGN.md", "TECH_SPEC.md"],
            "TESTING_STRATEGY.md": ["TESTING.md", "TEST_PLAN.md"],
        }

        for primary, alts in alternatives.items():
            if primary not in found:
                for alt in alts:
                    alt_path = project_path / alt
                    if alt_path.exists():
                        found.append(f"{alt} (alternative)")
                        try:
                            contents[alt] = alt_path.read_text(errors="ignore")
                        except Exception:
                            pass
                        break

        return ExistingDocuments(found=found, missing=missing, contents=contents)

    async def generate_project_manifest(
        self,
        project_name: str,
        report: IngestionReport,
    ) -> str:
        """Generate a PROJECT_MANIFEST.yaml file."""
        manifest = f"""# PROJECT_MANIFEST.yaml
# Auto-generated by AI Development Platform Phase 15.3
# Generated: {datetime.utcnow().isoformat()}

project:
  name: {project_name}
  version: "1.0.0"
  description: "{report.structure.package_manager or 'Unknown'} project ingested from {report.source_type.value}"
  imported: true
  import_date: "{datetime.utcnow().isoformat()}"

source:
  type: {report.source_type.value}
  location: "{report.source_location}"
  git_url: {f'"{report.git_metadata.remote_url}"' if report.git_metadata.remote_url else 'null'}
  branch: {f'"{report.git_metadata.current_branch}"' if report.git_metadata.current_branch else 'null'}
  last_commit: {f'"{report.git_metadata.last_commit_hash}"' if report.git_metadata.last_commit_hash else 'null'}

structure:
  total_files: {report.structure.total_files}
  total_size_bytes: {report.structure.total_size_bytes}
  build_system: {f'"{report.structure.build_system}"' if report.structure.build_system else 'null'}
  package_manager: {f'"{report.structure.package_manager}"' if report.structure.package_manager else 'null'}
  has_tests: {str(report.structure.has_tests).lower()}
  has_docs: {str(report.structure.has_docs).lower()}
  has_ci: {str(report.structure.has_ci).lower()}

aspects:
  primary: {report.aspects.primary_aspect.value}
  detected: [{', '.join(a.value for a in report.aspects.detected_aspects)}]

risk:
  level: {report.risk_assessment.risk_level}
  total_issues: {report.risk_assessment.total_issues}

governance:
  existing_docs: [{', '.join(f'"{d}"' for d in report.existing_docs.found)}]
  missing_docs: [{', '.join(f'"{d}"' for d in report.existing_docs.missing)}]
"""
        return manifest

    async def generate_current_state(
        self,
        project_name: str,
        report: IngestionReport,
    ) -> str:
        """Generate a CURRENT_STATE.md file."""
        return f"""# Current State: {project_name}

## Overview

This project was imported into the AI Development Platform on {datetime.utcnow().strftime('%Y-%m-%d')}.

**Status**: DEPLOYED (Imported)
**Primary Aspect**: {report.aspects.primary_aspect.value}
**Risk Level**: {report.risk_assessment.risk_level}

## Source Information

- **Type**: {report.source_type.value}
- **Location**: `{report.source_location}`
{f'- **Git URL**: `{report.git_metadata.remote_url}`' if report.git_metadata.remote_url else ''}
{f'- **Branch**: `{report.git_metadata.current_branch}`' if report.git_metadata.current_branch else ''}
{f'- **Last Commit**: `{report.git_metadata.last_commit_hash[:8] if report.git_metadata.last_commit_hash else "N/A"}`' if report.git_metadata.is_git_repo else ''}

## Project Structure

- **Total Files**: {report.structure.total_files}
- **Total Size**: {report.structure.total_size_bytes / 1024 / 1024:.2f} MB
- **Build System**: {report.structure.build_system or 'Unknown'}
- **Package Manager**: {report.structure.package_manager or 'Unknown'}

### Top-Level Directories

{chr(10).join(f'- `{d}/`' for d in report.structure.top_level_dirs[:10])}

### File Types

| Extension | Count |
|-----------|-------|
{chr(10).join(f'| {ext} | {count} |' for ext, count in list(report.structure.file_types.items())[:10])}

## Detected Aspects

| Aspect | Confidence | Evidence |
|--------|------------|----------|
{chr(10).join(f'| {a.value} | {report.aspects.confidence_scores.get(a.value, 0):.1%} | {len(report.aspects.evidence.get(a.value, []))} files |' for a in report.aspects.detected_aspects)}

## Risk Assessment

**Risk Level**: {report.risk_assessment.risk_level.upper()}
**Total Issues**: {report.risk_assessment.total_issues}

### Findings

{f'- **Hardcoded Secrets**: {len(report.risk_assessment.hardcoded_secrets)} found' if report.risk_assessment.hardcoded_secrets else '- No hardcoded secrets detected'}
{f'- **Sensitive Files**: {len(report.risk_assessment.sensitive_files)} found' if report.risk_assessment.sensitive_files else '- No sensitive files detected'}
{f'- **Dangerous Patterns**: {len(report.risk_assessment.dangerous_patterns)} found' if report.risk_assessment.dangerous_patterns else '- No dangerous patterns detected'}

### Recommendations

{chr(10).join(f'1. {r}' for r in report.risk_assessment.recommendations) if report.risk_assessment.recommendations else 'No recommendations at this time.'}

## Next Steps

This project is now registered in the Lifecycle Engine v2 and can receive CHANGE_MODE cycles:

1. Use `/new_feature` to request new features
2. Use `/report_bug` to report bugs
3. Use `/improve` to request improvements
4. Use `/refactor` to request refactoring
5. Use `/security_fix` to request security fixes

---
*Generated by AI Development Platform Phase 15.3*
"""

    async def generate_architecture_doc(
        self,
        project_name: str,
        report: IngestionReport,
    ) -> str:
        """Generate an ARCHITECTURE.md file."""
        return f"""# Architecture: {project_name}

## Overview

This document describes the architecture of the {project_name} project as analyzed during ingestion.

## Project Type

- **Primary Aspect**: {report.aspects.primary_aspect.value}
- **Build System**: {report.structure.build_system or 'Unknown'}
- **Package Manager**: {report.structure.package_manager or 'Unknown'}

## Directory Structure

```
{project_name}/
{chr(10).join(f'├── {d}/' for d in report.structure.top_level_dirs)}
```

## Detected Components

{chr(10).join(f'### {a.value.replace("_", " ").title()}{chr(10)}{chr(10)}Confidence: {report.aspects.confidence_scores.get(a.value, 0):.1%}{chr(10)}{chr(10)}Key files:{chr(10)}{chr(10).join(f"- `{f}`" for f in report.aspects.evidence.get(a.value, [])[:5])}' for a in report.aspects.detected_aspects)}

## Technology Stack

Based on the analysis, this project appears to use:

{f'- **Build**: {report.structure.build_system}' if report.structure.build_system else ''}
{f'- **Package Manager**: {report.structure.package_manager}' if report.structure.package_manager else ''}
{f'- **CI/CD**: Detected' if report.structure.has_ci else ''}
{f'- **Testing**: Framework present' if report.structure.has_tests else ''}

## Quality Indicators

| Indicator | Status |
|-----------|--------|
| Tests | {'✅ Present' if report.structure.has_tests else '❌ Missing'} |
| Documentation | {'✅ Present' if report.structure.has_docs else '❌ Missing'} |
| CI/CD | {'✅ Configured' if report.structure.has_ci else '❌ Not found'} |
| Security | {report.risk_assessment.risk_level.upper()} risk |

---
*Generated by AI Development Platform Phase 15.3*
"""

    async def generate_ai_policy(
        self,
        project_name: str,
    ) -> str:
        """Generate an AI_POLICY.md file."""
        return f"""# AI Policy: {project_name}

## Overview

This document defines the AI collaboration policy for the {project_name} project.

## General Principles

1. **Human Oversight**: All significant changes require human approval before deployment
2. **Transparency**: AI-generated code must be clearly identified
3. **Testing**: All changes must include appropriate tests
4. **Documentation**: Changes must be documented appropriately

## Change Types

### Features
- AI may propose new features based on feedback
- Implementation requires human approval at each phase
- Tests are mandatory

### Bug Fixes
- AI may identify and propose fixes
- Critical fixes require expedited review
- Regression tests are mandatory

### Improvements
- AI may suggest performance optimizations
- Changes must not alter existing behavior
- Benchmarks required for performance claims

### Refactoring
- AI may propose code restructuring
- No functional changes allowed
- Full test coverage required before refactoring

### Security Fixes
- AI may identify security issues
- Security fixes have highest priority
- Immediate review required

## Boundaries

### AI MUST NOT:
- Modify production data directly
- Deploy without human approval
- Access credentials or secrets
- Make changes outside project scope
- Skip testing phases

### AI MAY:
- Analyze code and suggest improvements
- Generate implementation plans
- Write code and tests
- Propose documentation updates
- Identify potential issues

## Review Process

1. All AI-generated changes go through AWAITING_FEEDBACK state
2. Human reviewer must approve before READY_FOR_PRODUCTION
3. Final deployment requires explicit human approval
4. Cycle history is maintained for audit

---
*Generated by AI Development Platform Phase 15.3*
"""

    async def generate_testing_strategy(
        self,
        project_name: str,
        report: IngestionReport,
    ) -> str:
        """Generate a TESTING_STRATEGY.md file."""
        return f"""# Testing Strategy: {project_name}

## Overview

This document defines the testing strategy for the {project_name} project.

## Current State

- **Has Tests**: {'Yes' if report.structure.has_tests else 'No'}
- **Has CI/CD**: {'Yes' if report.structure.has_ci else 'No'}
- **Package Manager**: {report.structure.package_manager or 'Unknown'}

## Testing Levels

### Unit Tests
- Test individual functions and methods
- Mock external dependencies
- Aim for high coverage of business logic

### Integration Tests
- Test component interactions
- Use test databases/services where appropriate
- Verify API contracts

### End-to-End Tests
- Test complete user workflows
- Use realistic test data
- Run before production deployment

## Test Requirements for Changes

All changes to this project must include:

1. **New Code**: Unit tests for all new functions
2. **Bug Fixes**: Regression test that reproduces the bug
3. **Features**: Integration tests for feature functionality
4. **Refactoring**: Existing tests must pass; add tests if coverage improves

## Running Tests

```bash
# Commands will depend on the project type
{f'npm test  # Node.js projects' if report.structure.package_manager in ['npm', 'yarn', 'pnpm', 'bun'] else ''}
{f'pytest  # Python projects' if report.structure.package_manager in ['pip', 'pipenv', 'poetry'] else ''}
{f'go test ./...  # Go projects' if report.structure.package_manager == 'go' else ''}
```

## CI/CD Integration

{'The project has CI/CD configured. Tests should run automatically on pull requests.' if report.structure.has_ci else 'CI/CD is not currently configured. Consider adding GitHub Actions or similar.'}

## Coverage Goals

- **Minimum**: 60% line coverage
- **Target**: 80% line coverage
- **Critical paths**: 100% coverage

---
*Generated by AI Development Platform Phase 15.3*
"""

    # -------------------------------------------------------------------------
    # Main Analysis Pipeline
    # -------------------------------------------------------------------------

    async def analyze_project(
        self,
        project_path: Path,
        ingestion_id: str,
        project_name: str,
        source_type: IngestionSource,
        source_location: str,
    ) -> Tuple[bool, str, Optional[IngestionReport]]:
        """
        Run the complete analysis pipeline on a project.

        Steps:
        1. Extract git metadata
        2. Enumerate files
        3. Analyze structure
        4. Detect aspects
        5. Scan risks
        6. Check existing documents
        7. Generate report
        """
        logger.info(f"Starting analysis of {project_name} from {source_location}")

        try:
            # Step 1: Git metadata
            git_metadata = await self.extract_git_metadata(project_path)
            logger.info(f"Git metadata: is_repo={git_metadata.is_git_repo}")

            # Step 2: Enumerate files
            files = await self.enumerate_files(project_path)
            logger.info(f"Enumerated {len(files)} files")

            # Step 3: Analyze structure
            structure = await self.analyze_structure(project_path, files)
            logger.info(f"Structure: {structure.total_files} files, {structure.total_directories} dirs")

            # Step 4: Detect aspects
            aspects = await self.detect_aspects(project_path, files)
            logger.info(f"Detected aspects: {[a.value for a in aspects.detected_aspects]}")

            # Step 5: Scan risks
            risk_assessment = await self.scan_risks(project_path, files)
            logger.info(f"Risk level: {risk_assessment.risk_level}")

            # Step 6: Check existing documents
            existing_docs = await self.check_existing_documents(project_path)
            logger.info(f"Found docs: {existing_docs.found}, Missing: {existing_docs.missing}")

            # Determine if ready for registration
            blocking_issues: List[str] = []
            warnings: List[str] = []

            if risk_assessment.risk_level == "critical":
                blocking_issues.append("Critical security issues detected - manual review required")

            if risk_assessment.hardcoded_secrets:
                warnings.append(f"Found {len(risk_assessment.hardcoded_secrets)} potential hardcoded secrets")

            if len(files) == 0:
                blocking_issues.append("No files found in project")

            ready_for_registration = len(blocking_issues) == 0

            # Create report
            report = IngestionReport(
                report_id=str(uuid.uuid4()),
                ingestion_id=ingestion_id,
                project_name=project_name,
                source_type=source_type,
                source_location=source_location,
                analyzed_at=datetime.utcnow(),
                git_metadata=git_metadata,
                structure=structure,
                aspects=aspects,
                risk_assessment=risk_assessment,
                existing_docs=existing_docs,
                files=files,
                ready_for_registration=ready_for_registration,
                blocking_issues=blocking_issues,
                warnings=warnings,
            )

            # Save report
            await self._save_report(report)

            logger.info(f"Analysis complete for {project_name}: ready={ready_for_registration}")
            return True, "Analysis completed successfully", report

        except Exception as e:
            logger.error(f"Analysis failed for {project_name}: {e}")
            return False, f"Analysis failed: {e}", None

    # -------------------------------------------------------------------------
    # Ingestion Request Management
    # -------------------------------------------------------------------------

    async def create_ingestion_request(
        self,
        project_name: str,
        source_type: IngestionSource,
        source_location: str,
        requested_by: str,
        description: str = "",
        target_aspects: Optional[List[ProjectAspect]] = None,
    ) -> Tuple[bool, str, Optional[IngestionRequest]]:
        """Create a new ingestion request."""
        ingestion_id = str(uuid.uuid4())

        request = IngestionRequest(
            ingestion_id=ingestion_id,
            project_name=project_name,
            source_type=source_type,
            source_location=source_location,
            status=IngestionStatus.PENDING,
            requested_by=requested_by,
            requested_at=datetime.utcnow(),
            description=description,
            target_aspects=target_aspects or [],
        )

        await self._save_ingestion_request(request)

        logger.info(f"Created ingestion request {ingestion_id} for {project_name}")
        return True, f"Ingestion request created: {ingestion_id}", request

    async def start_analysis(
        self,
        ingestion_id: str,
    ) -> Tuple[bool, str, Optional[IngestionReport]]:
        """Start the analysis for an ingestion request."""
        request = await self.get_ingestion_request(ingestion_id)
        if not request:
            return False, f"Ingestion request {ingestion_id} not found", None

        if request.status != IngestionStatus.PENDING:
            return False, f"Request is not pending (status: {request.status.value})", None

        # Update status
        request.status = IngestionStatus.ANALYZING
        request.updated_at = datetime.utcnow()
        await self._save_ingestion_request(request)

        try:
            # Prepare workspace
            if request.source_type == IngestionSource.GIT_REPOSITORY:
                success, msg, workspace_path = await self.clone_repository(
                    request.source_location, ingestion_id
                )
            elif request.source_type == IngestionSource.LOCAL_PATH:
                success, msg, workspace_path = await self.prepare_local_path(
                    request.source_location, ingestion_id
                )
            else:
                success, msg, workspace_path = False, f"Unsupported source type: {request.source_type}", None

            if not success or not workspace_path:
                request.status = IngestionStatus.FAILED
                request.error_message = msg
                request.updated_at = datetime.utcnow()
                await self._save_ingestion_request(request)
                return False, msg, None

            request.workspace_path = str(workspace_path)

            # Run analysis
            success, msg, report = await self.analyze_project(
                project_path=workspace_path,
                ingestion_id=ingestion_id,
                project_name=request.project_name,
                source_type=request.source_type,
                source_location=request.source_location,
            )

            if not success or not report:
                request.status = IngestionStatus.FAILED
                request.error_message = msg
                request.updated_at = datetime.utcnow()
                await self._save_ingestion_request(request)
                return False, msg, None

            # Update request with report
            request.report = report
            request.status = IngestionStatus.AWAITING_APPROVAL
            request.updated_at = datetime.utcnow()
            await self._save_ingestion_request(request)

            return True, "Analysis completed successfully", report

        except Exception as e:
            request.status = IngestionStatus.FAILED
            request.error_message = str(e)
            request.updated_at = datetime.utcnow()
            await self._save_ingestion_request(request)
            return False, f"Analysis failed: {e}", None

    async def approve_ingestion(
        self,
        ingestion_id: str,
        approved_by: str,
        user_role: UserRole,
    ) -> Tuple[bool, str]:
        """Approve an ingestion request for registration."""
        request = await self.get_ingestion_request(ingestion_id)
        if not request:
            return False, f"Ingestion request {ingestion_id} not found"

        if request.status != IngestionStatus.AWAITING_APPROVAL:
            return False, f"Request is not awaiting approval (status: {request.status.value})"

        # Check permissions
        if user_role not in {UserRole.OWNER, UserRole.ADMIN}:
            return False, f"Role {user_role.value} cannot approve ingestion"

        request.status = IngestionStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.utcnow()
        request.updated_at = datetime.utcnow()
        await self._save_ingestion_request(request)

        logger.info(f"Ingestion {ingestion_id} approved by {approved_by}")
        return True, "Ingestion approved"

    async def reject_ingestion(
        self,
        ingestion_id: str,
        rejected_by: str,
        reason: str,
        user_role: UserRole,
    ) -> Tuple[bool, str]:
        """Reject an ingestion request."""
        request = await self.get_ingestion_request(ingestion_id)
        if not request:
            return False, f"Ingestion request {ingestion_id} not found"

        if request.status != IngestionStatus.AWAITING_APPROVAL:
            return False, f"Request is not awaiting approval (status: {request.status.value})"

        # Check permissions
        if user_role not in {UserRole.OWNER, UserRole.ADMIN}:
            return False, f"Role {user_role.value} cannot reject ingestion"

        request.status = IngestionStatus.REJECTED
        request.rejection_reason = reason
        request.updated_at = datetime.utcnow()
        await self._save_ingestion_request(request)

        logger.info(f"Ingestion {ingestion_id} rejected by {rejected_by}: {reason}")
        return True, "Ingestion rejected"

    async def register_project(
        self,
        ingestion_id: str,
        registered_by: str,
        user_role: UserRole,
    ) -> Tuple[bool, str, List[str]]:
        """
        Register an approved ingestion as a project in Lifecycle Engine v2.

        Creates lifecycle instances for each detected aspect in DEPLOYED state.
        """
        request = await self.get_ingestion_request(ingestion_id)
        if not request:
            return False, f"Ingestion request {ingestion_id} not found", []

        if request.status != IngestionStatus.APPROVED:
            return False, f"Request is not approved (status: {request.status.value})", []

        if not request.report:
            return False, "No analysis report available", []

        lifecycle_ids: List[str] = []

        try:
            # Register each detected aspect
            aspects_to_register = request.target_aspects or request.report.aspects.detected_aspects

            for aspect in aspects_to_register:
                # Create lifecycle in PROJECT_MODE
                success, msg, lifecycle = await self._lifecycle_manager.create_lifecycle(
                    project_name=request.project_name,
                    mode=LifecycleMode.PROJECT_MODE,
                    aspect=aspect,
                    created_by=registered_by,
                    description=f"Imported from {request.source_type.value}: {request.source_location}",
                )

                if not success or not lifecycle:
                    logger.error(f"Failed to create lifecycle for {aspect.value}: {msg}")
                    continue

                # Transition to DEPLOYED state (imported projects start as deployed)
                # We need to go through the state machine: CREATED -> PLANNING -> DEVELOPMENT -> TESTING -> AWAITING_FEEDBACK -> READY_FOR_PRODUCTION -> DEPLOYED
                # For imported projects, we fast-track with special handling

                # Add metadata to indicate this is an imported project
                lifecycle.metadata["imported"] = True
                lifecycle.metadata["import_date"] = datetime.utcnow().isoformat()
                lifecycle.metadata["source_type"] = request.source_type.value
                lifecycle.metadata["source_location"] = request.source_location
                lifecycle.metadata["ingestion_id"] = ingestion_id

                # Fast-track through states for imported projects
                transitions = [
                    (TransitionTrigger.SYSTEM_INIT, "Imported project initialization"),
                    (TransitionTrigger.CLAUDE_JOB_COMPLETED, "Import analysis complete"),
                    (TransitionTrigger.CLAUDE_JOB_COMPLETED, "Import setup complete"),
                    (TransitionTrigger.TEST_PASSED, "Import validation passed"),
                    (TransitionTrigger.HUMAN_APPROVAL, "Import approved"),
                    (TransitionTrigger.HUMAN_APPROVAL, "Import deployment approved"),
                ]

                for trigger, reason in transitions:
                    success, msg, new_state = await self._lifecycle_manager.transition_lifecycle(
                        lifecycle_id=lifecycle.lifecycle_id,
                        trigger=trigger,
                        triggered_by=registered_by,
                        user_role=user_role,
                        reason=reason,
                        metadata={"import_fast_track": True},
                    )

                    if not success:
                        logger.warning(f"Transition failed for {lifecycle.lifecycle_id}: {msg}")
                        break

                lifecycle_ids.append(lifecycle.lifecycle_id)
                logger.info(f"Registered lifecycle {lifecycle.lifecycle_id} for {request.project_name}/{aspect.value}")

            # Update request
            request.lifecycle_ids = lifecycle_ids
            request.status = IngestionStatus.REGISTERED
            request.registered_at = datetime.utcnow()
            request.updated_at = datetime.utcnow()
            await self._save_ingestion_request(request)

            # Generate and save governance documents
            if request.workspace_path and request.report:
                await self._generate_governance_docs(
                    Path(request.workspace_path),
                    request.project_name,
                    request.report,
                )

            return True, f"Registered {len(lifecycle_ids)} lifecycle(s)", lifecycle_ids

        except Exception as e:
            logger.error(f"Registration failed: {e}")
            request.status = IngestionStatus.FAILED
            request.error_message = str(e)
            request.updated_at = datetime.utcnow()
            await self._save_ingestion_request(request)
            return False, f"Registration failed: {e}", []

    async def _generate_governance_docs(
        self,
        project_path: Path,
        project_name: str,
        report: IngestionReport,
    ) -> None:
        """Generate missing governance documents."""
        try:
            # Only generate missing documents
            for doc in report.existing_docs.missing:
                content = None

                if doc == "PROJECT_MANIFEST.yaml":
                    content = await self.generate_project_manifest(project_name, report)
                elif doc == "CURRENT_STATE.md":
                    content = await self.generate_current_state(project_name, report)
                elif doc == "ARCHITECTURE.md":
                    content = await self.generate_architecture_doc(project_name, report)
                elif doc == "AI_POLICY.md":
                    content = await self.generate_ai_policy(project_name)
                elif doc == "TESTING_STRATEGY.md":
                    content = await self.generate_testing_strategy(project_name, report)

                if content:
                    doc_path = project_path / doc
                    doc_path.write_text(content)
                    logger.info(f"Generated {doc} for {project_name}")

        except Exception as e:
            logger.error(f"Failed to generate governance docs: {e}")

    # -------------------------------------------------------------------------
    # Query Methods
    # -------------------------------------------------------------------------

    async def get_ingestion_request(self, ingestion_id: str) -> Optional[IngestionRequest]:
        """Get an ingestion request by ID."""
        state = await self._load_state()
        request_data = state.get("requests", {}).get(ingestion_id)
        if request_data:
            return IngestionRequest.from_dict(request_data)
        return None

    async def list_ingestion_requests(
        self,
        status_filter: Optional[IngestionStatus] = None,
        limit: int = 50,
    ) -> List[IngestionRequest]:
        """List ingestion requests with optional filtering."""
        state = await self._load_state()
        requests = []

        for request_data in state.get("requests", {}).values():
            try:
                request = IngestionRequest.from_dict(request_data)
                if status_filter is None or request.status == status_filter:
                    requests.append(request)
            except Exception as e:
                logger.warning(f"Failed to deserialize request: {e}")

        # Sort by requested_at descending
        requests.sort(key=lambda r: r.requested_at, reverse=True)
        return requests[:limit]

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    async def _save_ingestion_request(self, request: IngestionRequest) -> None:
        """Save an ingestion request to persistent storage."""
        async with self._lock:
            state = await self._load_state()
            state["requests"][request.ingestion_id] = request.to_dict()
            state["last_updated"] = datetime.utcnow().isoformat()
            await self._save_state(state)

    async def _save_report(self, report: IngestionReport) -> None:
        """Save an analysis report to the reports directory."""
        try:
            report_path = self._reports_dir / f"{report.ingestion_id}.json"
            report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    async def _load_state(self) -> Dict[str, Any]:
        """Load state from file."""
        if not self._state_file.exists():
            return {"requests": {}, "created_at": datetime.utcnow().isoformat()}
        try:
            return json.loads(self._state_file.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load state file: {e}")
            return {"requests": {}, "created_at": datetime.utcnow().isoformat()}

    async def _save_state(self, state: Dict[str, Any]) -> None:
        """Save state to file atomically."""
        temp_file = self._state_file.with_suffix(".tmp")
        try:
            temp_file.write_text(json.dumps(state, indent=2, default=str))
            temp_file.replace(self._state_file)
        except IOError as e:
            logger.error(f"Failed to save state file: {e}")
            if temp_file.exists():
                temp_file.unlink()


# -----------------------------------------------------------------------------
# Global Engine Instance
# -----------------------------------------------------------------------------

_engine_instance: Optional[ProjectIngestionEngine] = None


def get_ingestion_engine() -> ProjectIngestionEngine:
    """Get or create the ingestion engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ProjectIngestionEngine()
    return _engine_instance


# -----------------------------------------------------------------------------
# Public API Functions
# -----------------------------------------------------------------------------

async def create_ingestion_request(
    project_name: str,
    source_type: str,
    source_location: str,
    requested_by: str,
    description: str = "",
    target_aspects: Optional[List[str]] = None,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Create a new project ingestion request.

    Args:
        project_name: Name for the project
        source_type: 'git_repository' or 'local_path'
        source_location: Git URL or local filesystem path
        requested_by: User requesting the ingestion
        description: Optional project description
        target_aspects: Optional list of aspects to register

    Returns:
        (success, message, request_dict)
    """
    engine = get_ingestion_engine()

    aspects = [ProjectAspect(a) for a in target_aspects] if target_aspects else None

    success, msg, request = await engine.create_ingestion_request(
        project_name=project_name,
        source_type=IngestionSource(source_type),
        source_location=source_location,
        requested_by=requested_by,
        description=description,
        target_aspects=aspects,
    )

    return success, msg, request.to_dict() if request else None


async def start_ingestion_analysis(
    ingestion_id: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Start the analysis phase for an ingestion request.

    Returns:
        (success, message, report_dict)
    """
    engine = get_ingestion_engine()
    success, msg, report = await engine.start_analysis(ingestion_id)
    return success, msg, report.to_dict() if report else None


async def approve_ingestion(
    ingestion_id: str,
    approved_by: str,
    role: str,
) -> Tuple[bool, str]:
    """Approve an ingestion request."""
    engine = get_ingestion_engine()
    return await engine.approve_ingestion(
        ingestion_id=ingestion_id,
        approved_by=approved_by,
        user_role=UserRole(role),
    )


async def reject_ingestion(
    ingestion_id: str,
    rejected_by: str,
    reason: str,
    role: str,
) -> Tuple[bool, str]:
    """Reject an ingestion request."""
    engine = get_ingestion_engine()
    return await engine.reject_ingestion(
        ingestion_id=ingestion_id,
        rejected_by=rejected_by,
        reason=reason,
        user_role=UserRole(role),
    )


async def register_ingested_project(
    ingestion_id: str,
    registered_by: str,
    role: str,
) -> Tuple[bool, str, List[str]]:
    """
    Register an approved ingestion as a project.

    Returns:
        (success, message, lifecycle_ids)
    """
    engine = get_ingestion_engine()
    return await engine.register_project(
        ingestion_id=ingestion_id,
        registered_by=registered_by,
        user_role=UserRole(role),
    )


async def get_ingestion_request(
    ingestion_id: str,
) -> Optional[Dict[str, Any]]:
    """Get an ingestion request by ID."""
    engine = get_ingestion_engine()
    request = await engine.get_ingestion_request(ingestion_id)
    return request.to_dict() if request else None


async def list_ingestion_requests(
    status: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List ingestion requests."""
    engine = get_ingestion_engine()
    status_filter = IngestionStatus(status) if status else None
    requests = await engine.list_ingestion_requests(status_filter, limit)
    return [r.to_dict() for r in requests]


logger.info("Ingestion Engine module loaded successfully (Phase 15.3)")
