"""
Phase 16B-16E: Platform Dashboard & Observability Layer

READ-ONLY control plane for comprehensive system visibility.

HARD CONSTRAINTS:
- ❌ No business logic duplication
- ❌ No lifecycle mutation
- ❌ No Claude execution
- ✅ Read-only aggregation only
- ✅ Deterministic output
- ✅ Zero hallucinated data

This module provides:
1. Project Overview - All projects with lifecycle states
2. Claude Activity Panel - Jobs, workers, queue
3. Lifecycle Timeline - State transitions, approvals, feedback
4. Deployment View - TEST/PROD deployments
5. Security & Audit - Gate denials, policy violations
6. Identity Grouping - Projects grouped by fingerprint (Phase 16E)

All data is aggregated from existing sources:
- lifecycle_v2.py - Lifecycle state and history
- claude_backend.py - Job and scheduler state
- execution_gate.py - Audit trail and gate decisions
- project_registry.py - Project registry with identity (Phase 16E)
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("dashboard_backend")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
LIFECYCLE_STATE_DIR = Path("/home/aitesting.mybd.in/jobs/lifecycle")
EXECUTION_AUDIT_LOG = Path("/home/aitesting.mybd.in/jobs/execution_audit.log")
JOBS_DIR = Path("/home/aitesting.mybd.in/jobs")

# Fallbacks for local development
if not LIFECYCLE_STATE_DIR.exists():
    LIFECYCLE_STATE_DIR = Path("/tmp/lifecycle")
if not JOBS_DIR.exists():
    JOBS_DIR = Path("/tmp/jobs")


# -----------------------------------------------------------------------------
# Data Models (Read-Only Views)
# -----------------------------------------------------------------------------

class SystemHealth(str, Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ProjectOverview:
    """Read-only view of a project's current state."""
    project_id: str
    project_name: str
    mode: str  # PROJECT_MODE or CHANGE_MODE
    current_lifecycle_state: str
    aspects: Dict[str, str]  # aspect -> state
    active_cycle_number: int
    last_test_deployment: Optional[Dict[str, Any]]
    last_prod_deployment: Optional[Dict[str, Any]]
    current_claude_job: Optional[Dict[str, Any]]
    lifecycle_id: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    # Phase 16E: Identity fields
    fingerprint: Optional[str] = None
    version: str = "v1"
    parent_project_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "mode": self.mode,
            "current_lifecycle_state": self.current_lifecycle_state,
            "aspects": self.aspects,
            "active_cycle_number": self.active_cycle_number,
            "last_test_deployment": self.last_test_deployment,
            "last_prod_deployment": self.last_prod_deployment,
            "current_claude_job": self.current_claude_job,
            "lifecycle_id": self.lifecycle_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # Phase 16E: Identity information
            "fingerprint": self.fingerprint,
            "version": self.version,
            "parent_project_id": self.parent_project_id,
        }
        # Include identity_info if attached
        if hasattr(self, "_identity_info"):
            result.update(self._identity_info)
        return result


@dataclass
class ClaudeActivityPanel:
    """Read-only view of Claude job activity."""
    active_jobs: List[Dict[str, Any]]
    queued_jobs: List[Dict[str, Any]]
    completed_jobs_today: int
    failed_jobs_today: int
    worker_utilization: Dict[str, Any]
    job_worker_mapping: Dict[str, str]  # job_id -> worker_id
    gate_decisions_today: Dict[str, int]  # ALLOWED/DENIED counts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_jobs": self.active_jobs,
            "queued_jobs": self.queued_jobs,
            "completed_jobs_today": self.completed_jobs_today,
            "failed_jobs_today": self.failed_jobs_today,
            "worker_utilization": self.worker_utilization,
            "job_worker_mapping": self.job_worker_mapping,
            "gate_decisions_today": self.gate_decisions_today,
        }


@dataclass
class LifecycleTimeline:
    """Read-only view of lifecycle history."""
    lifecycle_id: str
    project_name: str
    aspect: str
    current_state: str
    transition_history: List[Dict[str, Any]]
    approvals: List[Dict[str, Any]]
    rejections: List[Dict[str, Any]]
    feedback_entries: List[Dict[str, Any]]
    cycle_history: List[Dict[str, Any]]
    change_summaries: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lifecycle_id": self.lifecycle_id,
            "project_name": self.project_name,
            "aspect": self.aspect,
            "current_state": self.current_state,
            "transition_history": self.transition_history,
            "approvals": self.approvals,
            "rejections": self.rejections,
            "feedback_entries": self.feedback_entries,
            "cycle_history": self.cycle_history,
            "change_summaries": self.change_summaries,
        }


@dataclass
class DeploymentView:
    """Read-only view of deployments."""
    deployment_type: str  # TEST or PROD
    commit_hash: Optional[str]
    timestamp: Optional[str]
    what_changed: str
    approver_id: Optional[str]
    lifecycle_id: str
    project_name: str
    aspect: str
    success: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_type": self.deployment_type,
            "commit_hash": self.commit_hash,
            "timestamp": self.timestamp,
            "what_changed": self.what_changed,
            "approver_id": self.approver_id,
            "lifecycle_id": self.lifecycle_id,
            "project_name": self.project_name,
            "aspect": self.aspect,
            "success": self.success,
        }


@dataclass
class AuditEvent:
    """Read-only view of an audit event."""
    timestamp: str
    event_type: str  # GATE_DENIAL, POLICY_VIOLATION, WORKSPACE_VIOLATION
    job_id: str
    project_name: str
    action: str
    user_id: str
    user_role: str
    reason: str
    severity: str  # INFO, WARNING, CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "job_id": self.job_id,
            "project_name": self.project_name,
            "action": self.action,
            "user_id": self.user_id,
            "user_role": self.user_role,
            "reason": self.reason,
            "severity": self.severity,
        }


@dataclass
class DashboardSummary:
    """Top-level dashboard summary for quick overview."""
    system_health: SystemHealth
    timestamp: str
    # Counts
    total_projects: int
    active_projects: int
    total_lifecycles: int
    active_lifecycles: int
    # Jobs
    active_jobs: int
    queued_jobs: int
    completed_today: int
    failed_today: int
    # Security
    gate_denials_today: int
    policy_violations_today: int
    # Services
    claude_cli_available: bool
    telegram_bot_operational: bool
    controller_healthy: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_health": self.system_health.value,
            "timestamp": self.timestamp,
            "projects": {
                "total": self.total_projects,
                "active": self.active_projects,
            },
            "lifecycles": {
                "total": self.total_lifecycles,
                "active": self.active_lifecycles,
            },
            "jobs": {
                "active": self.active_jobs,
                "queued": self.queued_jobs,
                "completed_today": self.completed_today,
                "failed_today": self.failed_today,
            },
            "security": {
                "gate_denials_today": self.gate_denials_today,
                "policy_violations_today": self.policy_violations_today,
            },
            "services": {
                "claude_cli": self.claude_cli_available,
                "telegram_bot": self.telegram_bot_operational,
                "controller": self.controller_healthy,
            },
        }


# -----------------------------------------------------------------------------
# Dashboard Backend (Read-Only Aggregator)
# -----------------------------------------------------------------------------

class DashboardBackend:
    """
    Read-only aggregator for platform observability.

    IMPORTANT: This class ONLY reads data. It does not:
    - Modify lifecycle state
    - Execute Claude jobs
    - Change any system configuration

    All methods are deterministic and return data that exists.
    If data cannot be determined, it returns "unknown" or None.
    """

    def __init__(self):
        self._lifecycle_dir = LIFECYCLE_STATE_DIR
        self._audit_log = EXECUTION_AUDIT_LOG
        self._jobs_dir = JOBS_DIR

    # -------------------------------------------------------------------------
    # Summary Dashboard
    # -------------------------------------------------------------------------

    async def get_dashboard_summary(self) -> DashboardSummary:
        """
        Get top-level dashboard summary.

        Returns aggregated counts and health status.
        All data is read-only from existing sources.
        Data sources: Project Registry (primary), Lifecycle files (secondary)
        """
        timestamp = datetime.utcnow().isoformat()

        # Get project data from registry (Phase 16C)
        registry_projects = await self._read_registry_projects()
        registry_project_names = {p.get("project_name") for p in registry_projects}

        # Get lifecycle data
        lifecycles = await self._read_all_lifecycles()
        active_lifecycles = [
            lc for lc in lifecycles
            if lc.get("state") not in ["archived", "rejected"]
        ]

        # Combine unique projects from both sources
        lifecycle_projects = set(lc.get("project_name") for lc in lifecycles)
        projects = registry_project_names | lifecycle_projects

        active_registry = {
            p.get("project_name") for p in registry_projects
            if p.get("current_status") not in ["archived", "failed"]
        }
        active_lifecycle = set(lc.get("project_name") for lc in active_lifecycles)
        active_projects = active_registry | active_lifecycle

        # Get job data
        job_stats = await self._get_job_statistics()

        # Get audit data
        audit_stats = await self._get_audit_statistics()

        # Check service health
        services = await self._check_service_health()

        # Determine overall health
        health = self._determine_system_health(
            active_jobs=job_stats["active"],
            failed_today=job_stats["failed_today"],
            gate_denials=audit_stats["denials_today"],
            services=services,
        )

        return DashboardSummary(
            system_health=health,
            timestamp=timestamp,
            total_projects=len(projects),
            active_projects=len(active_projects),
            total_lifecycles=len(lifecycles),
            active_lifecycles=len(active_lifecycles),
            active_jobs=job_stats["active"],
            queued_jobs=job_stats["queued"],
            completed_today=job_stats["completed_today"],
            failed_today=job_stats["failed_today"],
            gate_denials_today=audit_stats["denials_today"],
            policy_violations_today=audit_stats["violations_today"],
            claude_cli_available=services.get("claude_cli", False),
            telegram_bot_operational=services.get("telegram_bot", False),
            controller_healthy=services.get("controller", True),
        )

    # -------------------------------------------------------------------------
    # Project Overview
    # -------------------------------------------------------------------------

    async def get_all_projects(self) -> List[ProjectOverview]:
        """
        Get overview of all projects.

        Returns list of projects with their current state.
        Data is aggregated from:
        1. Project Registry (primary source - Phase 16C/16E)
        2. Lifecycle state files (secondary source)

        This ensures projects appear in dashboard immediately after creation.

        Phase 16E: Projects now include fingerprint and version information.
        """
        result = []
        seen_projects = set()

        # Source 1: Project Registry (Phase 16C/16E)
        registry_projects = await self._read_registry_projects()
        for proj in registry_projects:
            project_name = proj.get("project_name", "unknown")
            seen_projects.add(project_name)

            overview = ProjectOverview(
                project_id=proj.get("project_id", "unknown"),
                project_name=project_name,
                mode=proj.get("mode", "project_mode"),
                current_lifecycle_state=proj.get("current_status", "created"),
                aspects=proj.get("aspects", {}),
                active_cycle_number=1,
                last_test_deployment=None,
                last_prod_deployment=None,
                current_claude_job=None,
                lifecycle_id=proj.get("lifecycle_ids", [None])[0] if proj.get("lifecycle_ids") else None,
                created_at=proj.get("created_at"),
                updated_at=proj.get("updated_at"),
                # Phase 16E: Identity fields
                fingerprint=proj.get("fingerprint"),
                version=proj.get("version", "v1"),
                parent_project_id=proj.get("parent_project_id"),
            )

            result.append(overview)

        # Source 2: Lifecycle state files (for projects not in registry)
        lifecycles = await self._read_all_lifecycles()

        # Group by project
        projects_map: Dict[str, List[Dict]] = {}
        for lc in lifecycles:
            project_name = lc.get("project_name", "unknown")
            if project_name in seen_projects:
                continue  # Already added from registry
            if project_name not in projects_map:
                projects_map[project_name] = []
            projects_map[project_name].append(lc)

        for project_name, project_lifecycles in projects_map.items():
            # Find primary lifecycle (most recent active)
            active_lcs = [
                lc for lc in project_lifecycles
                if lc.get("state") not in ["archived", "rejected"]
            ]
            primary_lc = active_lcs[0] if active_lcs else project_lifecycles[0]

            # Build aspect map
            aspects = {}
            for lc in project_lifecycles:
                aspect = lc.get("aspect", "core")
                aspects[aspect] = lc.get("state", "unknown")

            # Get current job if any
            current_job = None
            job_id = primary_lc.get("current_claude_job_id")
            if job_id:
                current_job = await self._get_job_summary(job_id)

            result.append(ProjectOverview(
                project_id=primary_lc.get("lifecycle_id", "unknown"),
                project_name=project_name,
                mode=primary_lc.get("mode", "project_mode"),
                current_lifecycle_state=primary_lc.get("state", "unknown"),
                aspects=aspects,
                active_cycle_number=primary_lc.get("cycle_number", 1),
                last_test_deployment=self._extract_deployment(primary_lc, "test"),
                last_prod_deployment=self._extract_deployment(primary_lc, "prod"),
                current_claude_job=current_job,
                lifecycle_id=primary_lc.get("lifecycle_id"),
                created_at=primary_lc.get("created_at"),
                updated_at=primary_lc.get("updated_at"),
            ))

        return result

    async def get_projects_grouped_by_identity(self) -> Dict[str, Any]:
        """
        Get projects grouped by fingerprint/identity (Phase 16E).

        Returns projects organized by their unique identity, showing
        all versions of the same logical project together.

        This enables the dashboard to:
        1. Show project families (same identity, different versions)
        2. Detect and display project evolution
        3. Provide consistent project grouping
        """
        grouped = await self._read_registry_projects_grouped()

        if not grouped:
            # Fall back to ungrouped view
            projects = await self.get_all_projects()
            return {
                "grouped": False,
                "project_families": [],
                "ungrouped_projects": [p.to_dict() for p in projects],
            }

        # Build project families
        families = []
        for fingerprint, versions in grouped.items():
            if not versions:
                continue

            # Sort by version
            sorted_versions = sorted(
                versions,
                key=lambda x: x.get("version", "v1")
            )

            primary = sorted_versions[-1]  # Latest version
            families.append({
                "fingerprint": fingerprint,
                "primary_project": primary.get("project_name"),
                "total_versions": len(versions),
                "versions": sorted_versions,
                "created_at": sorted_versions[0].get("created_at"),
                "latest_update": primary.get("updated_at"),
            })

        return {
            "grouped": True,
            "total_families": len(families),
            "project_families": families,
            "ungrouped_projects": [],
        }

    async def get_project_detail(self, project_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed view of a specific project.

        Returns comprehensive project data including all aspects and history.
        """
        lifecycles = await self._read_all_lifecycles()
        project_lifecycles = [
            lc for lc in lifecycles
            if lc.get("project_name") == project_name
        ]

        if not project_lifecycles:
            return None

        # Find primary lifecycle
        active_lcs = [
            lc for lc in project_lifecycles
            if lc.get("state") not in ["archived", "rejected"]
        ]
        primary_lc = active_lcs[0] if active_lcs else project_lifecycles[0]

        # Build comprehensive view
        aspects_detail = {}
        for lc in project_lifecycles:
            aspect = lc.get("aspect", "core")
            aspects_detail[aspect] = {
                "state": lc.get("state"),
                "lifecycle_id": lc.get("lifecycle_id"),
                "cycle_number": lc.get("cycle_number", 1),
                "current_job": lc.get("current_claude_job_id"),
                "last_test_passed": lc.get("last_test_passed"),
                "feedback_count": len(lc.get("feedback_history", [])),
            }

        return {
            "project_name": project_name,
            "primary_lifecycle_id": primary_lc.get("lifecycle_id"),
            "mode": primary_lc.get("mode"),
            "overall_state": primary_lc.get("state"),
            "aspects": aspects_detail,
            "all_lifecycles": [
                {
                    "lifecycle_id": lc.get("lifecycle_id"),
                    "aspect": lc.get("aspect"),
                    "state": lc.get("state"),
                    "mode": lc.get("mode"),
                }
                for lc in project_lifecycles
            ],
            "total_cycles": sum(
                lc.get("cycle_number", 1) for lc in project_lifecycles
            ),
        }

    # -------------------------------------------------------------------------
    # Claude Activity Panel
    # -------------------------------------------------------------------------

    async def get_claude_activity(self) -> ClaudeActivityPanel:
        """
        Get Claude job activity panel.

        Returns active jobs, queue state, and worker utilization.
        Data is read from job state files and scheduler.
        """
        job_stats = await self._get_job_statistics()
        active_jobs = await self._get_active_jobs()
        queued_jobs = await self._get_queued_jobs()

        # Get worker utilization from scheduler
        worker_util = await self._get_worker_utilization()

        # Get gate decisions from audit log
        audit_stats = await self._get_audit_statistics()

        return ClaudeActivityPanel(
            active_jobs=active_jobs,
            queued_jobs=queued_jobs,
            completed_jobs_today=job_stats["completed_today"],
            failed_jobs_today=job_stats["failed_today"],
            worker_utilization=worker_util,
            job_worker_mapping=worker_util.get("job_mapping", {}),
            gate_decisions_today={
                "allowed": audit_stats.get("allowed_today", 0),
                "denied": audit_stats.get("denials_today", 0),
            },
        )

    async def get_jobs_list(
        self,
        state: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Get list of Claude jobs with optional filtering.

        Parameters:
            state: Filter by job state (queued, running, completed, failed)
            project: Filter by project name
            limit: Maximum number of jobs to return

        Returns list of job summaries.
        """
        jobs = await self._read_all_jobs()

        # Apply filters
        if state:
            jobs = [j for j in jobs if j.get("state") == state]
        if project:
            jobs = [j for j in jobs if j.get("project_name") == project]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)

        return jobs[:limit]

    # -------------------------------------------------------------------------
    # Lifecycle Timeline
    # -------------------------------------------------------------------------

    async def get_lifecycle_timeline(self, lifecycle_id: str) -> Optional[LifecycleTimeline]:
        """
        Get detailed timeline for a specific lifecycle.

        Returns full state transition history, approvals, feedback, etc.
        """
        lifecycle = await self._read_lifecycle(lifecycle_id)
        if not lifecycle:
            return None

        # Extract timeline events
        transition_history = []
        approvals = []
        rejections = []
        feedback_entries = lifecycle.get("feedback_history", [])

        # Build transition history from cycle_history
        for cycle in lifecycle.get("cycle_history", []):
            transition_history.append({
                "cycle": cycle.get("cycle_number"),
                "from_state": cycle.get("from_state"),
                "to_state": cycle.get("to_state"),
                "timestamp": cycle.get("timestamp"),
                "trigger": cycle.get("trigger"),
            })

        # Extract approvals/rejections from metadata
        for entry in lifecycle.get("metadata", {}).get("approval_history", []):
            if entry.get("decision") == "approved":
                approvals.append(entry)
            elif entry.get("decision") == "rejected":
                rejections.append(entry)

        return LifecycleTimeline(
            lifecycle_id=lifecycle_id,
            project_name=lifecycle.get("project_name", "unknown"),
            aspect=lifecycle.get("aspect", "core"),
            current_state=lifecycle.get("state", "unknown"),
            transition_history=transition_history,
            approvals=approvals,
            rejections=rejections,
            feedback_entries=feedback_entries,
            cycle_history=lifecycle.get("cycle_history", []),
            change_summaries=[
                cycle.get("change_summary", "")
                for cycle in lifecycle.get("cycle_history", [])
                if cycle.get("change_summary")
            ],
        )

    async def get_all_lifecycles(
        self,
        state: Optional[str] = None,
        project: Optional[str] = None,
        aspect: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of all lifecycles with optional filtering.

        Parameters:
            state: Filter by lifecycle state
            project: Filter by project name
            aspect: Filter by aspect

        Returns list of lifecycle summaries.
        """
        lifecycles = await self._read_all_lifecycles()

        # Apply filters
        if state:
            lifecycles = [lc for lc in lifecycles if lc.get("state") == state]
        if project:
            lifecycles = [lc for lc in lifecycles if lc.get("project_name") == project]
        if aspect:
            lifecycles = [lc for lc in lifecycles if lc.get("aspect") == aspect]

        return lifecycles

    # -------------------------------------------------------------------------
    # Audit & Security
    # -------------------------------------------------------------------------

    async def get_audit_events(
        self,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: Optional[str] = None,
    ) -> List[AuditEvent]:
        """
        Get audit events with optional filtering.

        Parameters:
            event_type: Filter by event type (GATE_DENIAL, POLICY_VIOLATION, etc.)
            limit: Maximum events to return
            since: ISO timestamp to filter events after

        Returns list of audit events.
        """
        events = await self._read_audit_log()

        # Apply filters
        if event_type:
            events = [e for e in events if e.get("event_type") == event_type]

        if since:
            events = [
                e for e in events
                if e.get("timestamp", "") >= since
            ]

        # Sort by timestamp descending
        events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

        result = []
        for e in events[:limit]:
            result.append(AuditEvent(
                timestamp=e.get("timestamp", "unknown"),
                event_type=e.get("gate_decision", "UNKNOWN"),
                job_id=e.get("job_id", "unknown"),
                project_name=e.get("project_name", "unknown"),
                action=e.get("executed_action", "unknown"),
                user_id=e.get("requesting_user_id", "unknown"),
                user_role=e.get("requesting_user_role", "unknown"),
                reason=e.get("denied_reason") or "N/A",
                severity="WARNING" if e.get("gate_decision") == "DENIED" else "INFO",
            ))

        return result

    async def get_security_summary(self) -> Dict[str, Any]:
        """
        Get security summary for dashboard.

        Returns counts and recent security events.
        """
        audit_stats = await self._get_audit_statistics()
        recent_denials = await self.get_audit_events(event_type="DENIED", limit=10)

        return {
            "gate_denials_today": audit_stats["denials_today"],
            "policy_violations_today": audit_stats["violations_today"],
            "total_events_today": audit_stats["total_today"],
            "recent_denials": [d.to_dict() for d in recent_denials],
            "security_status": "OK" if audit_stats["denials_today"] < 10 else "ELEVATED",
        }

    # -------------------------------------------------------------------------
    # Private Helper Methods (Read-Only)
    # -------------------------------------------------------------------------

    async def _read_registry_projects(self) -> List[Dict[str, Any]]:
        """Read projects from Project Registry (Phase 16C)."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            return registry.get_dashboard_projects()
        except ImportError:
            logger.debug("Project registry not available")
            return []
        except Exception as e:
            logger.warning(f"Failed to read project registry: {e}")
            return []

    async def _read_registry_projects_grouped(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Read projects from Project Registry grouped by identity (Phase 16E).

        Returns:
            Dict with keys as fingerprints and values as lists of project versions.
        """
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            return registry.get_dashboard_projects_grouped()
        except ImportError:
            logger.debug("Project registry not available for grouped view")
            return {}
        except Exception as e:
            logger.warning(f"Failed to read grouped projects: {e}")
            return {}

    async def _read_all_lifecycles(self) -> List[Dict[str, Any]]:
        """Read all lifecycle state files."""
        lifecycles = []
        lifecycle_dir = self._lifecycle_dir

        if not lifecycle_dir.exists():
            return lifecycles

        for state_file in lifecycle_dir.glob("*.json"):
            if state_file.name.startswith("lifecycle_"):
                continue  # Skip aggregate files
            try:
                with open(state_file) as f:
                    data = json.load(f)
                    lifecycles.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read lifecycle file {state_file}: {e}")

        return lifecycles

    async def _read_lifecycle(self, lifecycle_id: str) -> Optional[Dict[str, Any]]:
        """Read a specific lifecycle state file."""
        state_file = self._lifecycle_dir / f"{lifecycle_id}.json"
        if not state_file.exists():
            return None

        try:
            with open(state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read lifecycle {lifecycle_id}: {e}")
            return None

    async def _read_all_jobs(self) -> List[Dict[str, Any]]:
        """Read all job state files."""
        jobs = []
        jobs_dir = self._jobs_dir

        if not jobs_dir.exists():
            return jobs

        for job_dir in jobs_dir.iterdir():
            if not job_dir.is_dir() or not job_dir.name.startswith("job-"):
                continue

            state_file = job_dir / "job_state.json"
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        data = json.load(f)
                        jobs.append(data)
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Failed to read job state {state_file}: {e}")

        return jobs

    async def _get_job_summary(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific job."""
        job_dir = self._jobs_dir / f"job-{job_id}"
        state_file = job_dir / "job_state.json"

        if not state_file.exists():
            # Try without job- prefix
            job_dir = self._jobs_dir / job_id
            state_file = job_dir / "job_state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file) as f:
                data = json.load(f)
                return {
                    "job_id": data.get("job_id"),
                    "state": data.get("state"),
                    "task_type": data.get("task_type"),
                    "created_at": data.get("created_at"),
                }
        except (json.JSONDecodeError, IOError):
            return None

    async def _get_job_statistics(self) -> Dict[str, int]:
        """Get job statistics for today."""
        jobs = await self._read_all_jobs()
        today = datetime.utcnow().date().isoformat()

        active = 0
        queued = 0
        completed_today = 0
        failed_today = 0

        for job in jobs:
            state = job.get("state", "")
            completed_at = job.get("completed_at", "")

            if state == "running":
                active += 1
            elif state == "queued":
                queued += 1
            elif state == "completed" and completed_at.startswith(today):
                completed_today += 1
            elif state == "failed" and completed_at.startswith(today):
                failed_today += 1

        return {
            "active": active,
            "queued": queued,
            "completed_today": completed_today,
            "failed_today": failed_today,
        }

    async def _get_active_jobs(self) -> List[Dict[str, Any]]:
        """Get list of currently active jobs."""
        jobs = await self._read_all_jobs()
        return [
            {
                "job_id": j.get("job_id"),
                "project_name": j.get("project_name"),
                "task_type": j.get("task_type"),
                "started_at": j.get("started_at"),
                "aspect": j.get("aspect"),
            }
            for j in jobs
            if j.get("state") == "running"
        ]

    async def _get_queued_jobs(self) -> List[Dict[str, Any]]:
        """Get list of queued jobs."""
        jobs = await self._read_all_jobs()
        queued = [
            {
                "job_id": j.get("job_id"),
                "project_name": j.get("project_name"),
                "task_type": j.get("task_type"),
                "created_at": j.get("created_at"),
                "priority": j.get("priority", 50),
                "aspect": j.get("aspect"),
            }
            for j in jobs
            if j.get("state") == "queued"
        ]
        # Sort by priority descending, then created_at ascending
        queued.sort(key=lambda j: (-j.get("priority", 50), j.get("created_at", "")))
        return queued

    async def _get_worker_utilization(self) -> Dict[str, Any]:
        """Get worker utilization from scheduler."""
        # Try to read scheduler state if available
        scheduler_state = self._jobs_dir / "scheduler_state.json"
        if scheduler_state.exists():
            try:
                with open(scheduler_state) as f:
                    data = json.load(f)
                    return {
                        "total_workers": data.get("max_workers", 3),
                        "active_workers": data.get("active_workers", 0),
                        "utilization_percent": data.get("utilization", 0),
                        "job_mapping": data.get("job_mapping", {}),
                    }
            except (json.JSONDecodeError, IOError):
                pass

        # Fallback: estimate from active jobs
        active_jobs = await self._get_active_jobs()
        return {
            "total_workers": 3,  # Default MAX_CONCURRENT_JOBS
            "active_workers": len(active_jobs),
            "utilization_percent": round(len(active_jobs) / 3 * 100, 1),
            "job_mapping": {},
        }

    async def _read_audit_log(self) -> List[Dict[str, Any]]:
        """Read audit log entries."""
        events = []
        if not self._audit_log.exists():
            return events

        try:
            with open(self._audit_log) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        except IOError as e:
            logger.warning(f"Failed to read audit log: {e}")

        return events

    async def _get_audit_statistics(self) -> Dict[str, int]:
        """Get audit statistics for today."""
        events = await self._read_audit_log()
        today = datetime.utcnow().date().isoformat()

        denials = 0
        allowed = 0
        violations = 0
        total = 0

        for event in events:
            timestamp = event.get("timestamp", "")
            if not timestamp.startswith(today):
                continue

            total += 1
            decision = event.get("gate_decision", "")
            if decision == "DENIED":
                denials += 1
                if "policy" in event.get("denied_reason", "").lower():
                    violations += 1
            elif decision == "ALLOWED":
                allowed += 1

        return {
            "denials_today": denials,
            "allowed_today": allowed,
            "violations_today": violations,
            "total_today": total,
        }

    async def _check_service_health(self) -> Dict[str, bool]:
        """Check health of platform services."""
        import subprocess

        services = {
            "claude_cli": False,
            "telegram_bot": False,
            "controller": True,  # We're running, so controller is healthy
        }

        # Check Claude CLI
        try:
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                timeout=5,
            )
            services["claude_cli"] = result.returncode == 0
        except Exception:
            pass

        # Check Telegram bot via systemd
        try:
            result = subprocess.run(
                ["systemctl", "is-active", "ai-telegram-bot"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            services["telegram_bot"] = result.stdout.strip() == "active"
        except Exception:
            pass

        return services

    def _determine_system_health(
        self,
        active_jobs: int,
        failed_today: int,
        gate_denials: int,
        services: Dict[str, bool],
    ) -> SystemHealth:
        """Determine overall system health based on metrics."""
        # Critical: Essential services down
        if not services.get("controller", True):
            return SystemHealth.CRITICAL

        # Degraded: High failure rate or many denials
        if failed_today > 5 or gate_denials > 20:
            return SystemHealth.DEGRADED

        # Degraded: Claude CLI not available
        if not services.get("claude_cli", False):
            return SystemHealth.DEGRADED

        # Healthy
        return SystemHealth.HEALTHY

    def _extract_deployment(
        self,
        lifecycle: Dict[str, Any],
        deploy_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Extract deployment info from lifecycle metadata."""
        deployments = lifecycle.get("metadata", {}).get("deployments", [])
        for dep in reversed(deployments):
            if dep.get("type") == deploy_type:
                return {
                    "timestamp": dep.get("timestamp"),
                    "commit": dep.get("commit_hash"),
                    "approver": dep.get("approved_by"),
                }
        return None


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------

# Singleton instance
_dashboard: Optional[DashboardBackend] = None


def get_dashboard() -> DashboardBackend:
    """Get the dashboard backend singleton."""
    global _dashboard
    if _dashboard is None:
        _dashboard = DashboardBackend()
    return _dashboard


async def get_dashboard_summary() -> Dict[str, Any]:
    """Get dashboard summary."""
    dashboard = get_dashboard()
    summary = await dashboard.get_dashboard_summary()
    return summary.to_dict()


async def get_all_projects() -> List[Dict[str, Any]]:
    """Get all projects overview."""
    dashboard = get_dashboard()
    projects = await dashboard.get_all_projects()
    return [p.to_dict() for p in projects]


async def get_project_detail(project_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed project view."""
    dashboard = get_dashboard()
    return await dashboard.get_project_detail(project_name)


async def get_claude_activity() -> Dict[str, Any]:
    """Get Claude activity panel."""
    dashboard = get_dashboard()
    activity = await dashboard.get_claude_activity()
    return activity.to_dict()


async def get_jobs_list(
    state: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get jobs list with filtering."""
    dashboard = get_dashboard()
    return await dashboard.get_jobs_list(state=state, project=project, limit=limit)


async def get_lifecycle_timeline(lifecycle_id: str) -> Optional[Dict[str, Any]]:
    """Get lifecycle timeline."""
    dashboard = get_dashboard()
    timeline = await dashboard.get_lifecycle_timeline(lifecycle_id)
    return timeline.to_dict() if timeline else None


async def get_all_lifecycles(
    state: Optional[str] = None,
    project: Optional[str] = None,
    aspect: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get all lifecycles with filtering."""
    dashboard = get_dashboard()
    return await dashboard.get_all_lifecycles(state=state, project=project, aspect=aspect)


async def get_audit_events(
    event_type: Optional[str] = None,
    limit: int = 100,
    since: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get audit events."""
    dashboard = get_dashboard()
    events = await dashboard.get_audit_events(event_type=event_type, limit=limit, since=since)
    return [e.to_dict() for e in events]


async def get_security_summary() -> Dict[str, Any]:
    """Get security summary."""
    dashboard = get_dashboard()
    return await dashboard.get_security_summary()
