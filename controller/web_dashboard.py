"""
Web Dashboard for AI Development Platform

This module provides a comprehensive web-based dashboard with:
1. Login authentication (email/password)
2. Project overview with real-time status
3. In-depth project logging from start to end
4. Project cleanup functionality
5. Job monitoring and management (Claude jobs & queue)
6. Lifecycle management view
7. Audit trail and security events
8. Runtime monitoring (signals, incidents, recommendations)
9. Execution tracking and verification
10. System health overview

Security:
- Session-based authentication
- CSRF protection
- Password hashing with SHA-256
"""

import hashlib
import json
import logging
import secrets
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, HTTPException, Request, Response, Depends, Form, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

# Import Jinja2 directly for inline templates (no filesystem templates needed)
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Template = None

logger = logging.getLogger("web_dashboard")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Hardcoded credentials as requested
DASHBOARD_USERS = {
    "rahul4webdev@gmail.com": {
        "password_hash": hashlib.sha256("Admin@#2024".encode()).hexdigest(),
        "name": "Rahul",
        "role": "admin"
    }
}

# Session storage (in-memory, for production use Redis/DB)
_sessions: Dict[str, Dict[str, Any]] = {}
SESSION_EXPIRY_HOURS = 24

# Paths
PROJECTS_DIR = Path("/home/aitesting.mybd.in/projects")
JOBS_DIR = Path("/home/aitesting.mybd.in/jobs")
REGISTRY_DIR = JOBS_DIR / "registry"
LIFECYCLE_DIR = JOBS_DIR / "lifecycle"
INTENT_BASELINES_DIR = JOBS_DIR / "intent_baselines"
INTENT_CONTRACTS_DIR = JOBS_DIR / "intent_contracts"
LOGS_DIR = JOBS_DIR / "logs"
AUDIT_DIR = JOBS_DIR / "audit"
SIGNALS_DIR = JOBS_DIR / "signals"
INCIDENTS_DIR = JOBS_DIR / "incidents"
RECOMMENDATIONS_DIR = JOBS_DIR / "recommendations"
EXECUTIONS_DIR = JOBS_DIR / "executions"

# Fallbacks for local development
if not PROJECTS_DIR.exists():
    PROJECTS_DIR = Path(__file__).parent.parent / "projects"
if not JOBS_DIR.exists():
    JOBS_DIR = Path("/tmp/jobs")

# -----------------------------------------------------------------------------
# Router Setup
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/dashboard", tags=["Web Dashboard"])


# -----------------------------------------------------------------------------
# Session Management
# -----------------------------------------------------------------------------
def create_session(email: str) -> str:
    """Create a new session for authenticated user."""
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "email": email,
        "user": DASHBOARD_USERS[email],
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": (datetime.utcnow() + timedelta(hours=SESSION_EXPIRY_HOURS)).isoformat()
    }
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data if valid."""
    if not session_id or session_id not in _sessions:
        return None

    session = _sessions[session_id]
    expires_at = datetime.fromisoformat(session["expires_at"])

    if datetime.utcnow() > expires_at:
        del _sessions[session_id]
        return None

    return session


def delete_session(session_id: str) -> None:
    """Delete a session."""
    if session_id in _sessions:
        del _sessions[session_id]


def get_current_user(request: Request) -> Optional[Dict[str, Any]]:
    """Get current authenticated user from request."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        return None

    session = get_session(session_id)
    if not session:
        return None

    return session["user"]


def require_auth(request: Request) -> Dict[str, Any]:
    """Dependency to require authentication."""
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# -----------------------------------------------------------------------------
# Data Access Functions
# -----------------------------------------------------------------------------
def get_all_projects() -> List[Dict[str, Any]]:
    """Get all projects from registry and filesystem."""
    projects = []
    seen = set()

    # Source 1: Project Registry
    registry_file = REGISTRY_DIR / "projects.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                data = json.load(f)
                for name, proj in data.get("projects", {}).items():
                    seen.add(name)
                    projects.append({
                        "name": name,
                        "status": proj.get("current_status", "unknown"),
                        "created_at": proj.get("created_at", ""),
                        "updated_at": proj.get("updated_at", ""),
                        "fingerprint": proj.get("identity", {}).get("fingerprint", "")[:16] + "..." if proj.get("identity", {}).get("fingerprint") else "",
                        "description": proj.get("description", "")[:100],
                        "source": "registry"
                    })
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read registry: {e}")

    # Source 2: Projects directory
    if PROJECTS_DIR.exists():
        for project_dir in PROJECTS_DIR.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                if project_dir.name not in seen:
                    ipc_file = project_dir / "INTERNAL_PROJECT_CONTRACT.yaml"
                    status = "unknown"
                    if ipc_file.exists():
                        status = "has_ipc"

                    projects.append({
                        "name": project_dir.name,
                        "status": status,
                        "created_at": datetime.fromtimestamp(project_dir.stat().st_ctime).isoformat(),
                        "updated_at": datetime.fromtimestamp(project_dir.stat().st_mtime).isoformat(),
                        "fingerprint": "",
                        "description": "",
                        "source": "filesystem"
                    })

    # Sort by updated_at descending
    projects.sort(key=lambda p: p.get("updated_at", ""), reverse=True)
    return projects


def get_all_jobs() -> List[Dict[str, Any]]:
    """Get all jobs from job state."""
    jobs = []
    job_state_file = JOBS_DIR / "job_state.json"

    if job_state_file.exists():
        try:
            with open(job_state_file) as f:
                data = json.load(f)
                for job_id, job in data.get("jobs", {}).items():
                    jobs.append({
                        "job_id": job_id,
                        "project_name": job.get("project_name", "unknown"),
                        "task_type": job.get("task_type", "unknown"),
                        "state": job.get("state", "unknown"),
                        "aspect": job.get("aspect", "core"),
                        "priority": job.get("priority", 0),
                        "created_at": job.get("created_at", ""),
                        "started_at": job.get("started_at", ""),
                        "completed_at": job.get("completed_at", ""),
                        "result": job.get("result", "")[:200] if job.get("result") else "",
                        "error": job.get("error", "")[:200] if job.get("error") else ""
                    })
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read job state: {e}")

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return jobs


def get_job_queue() -> List[Dict[str, Any]]:
    """Get jobs in queue (queued state)."""
    all_jobs = get_all_jobs()
    return [j for j in all_jobs if j.get("state") == "queued"]


def get_running_jobs() -> List[Dict[str, Any]]:
    """Get currently running jobs."""
    all_jobs = get_all_jobs()
    return [j for j in all_jobs if j.get("state") == "running"]


def get_all_lifecycles() -> List[Dict[str, Any]]:
    """Get all lifecycle records."""
    lifecycles = []

    if LIFECYCLE_DIR.exists():
        for lc_file in LIFECYCLE_DIR.glob("*.json"):
            try:
                with open(lc_file) as f:
                    lc = json.load(f)
                    lifecycles.append({
                        "lifecycle_id": lc.get("lifecycle_id", lc_file.stem),
                        "project_name": lc.get("project_name", "unknown"),
                        "aspect": lc.get("aspect", "core"),
                        "mode": lc.get("mode", "unknown"),
                        "state": lc.get("state", "unknown"),
                        "cycle_count": len(lc.get("cycle_history", [])),
                        "created_at": lc.get("created_at", ""),
                        "updated_at": lc.get("updated_at", ""),
                        "feedback_pending": lc.get("feedback_pending", False)
                    })
            except (json.JSONDecodeError, IOError):
                pass

    # Sort by updated_at descending
    lifecycles.sort(key=lambda l: l.get("updated_at", ""), reverse=True)
    return lifecycles


def get_lifecycle_details(lifecycle_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed lifecycle information."""
    lc_file = LIFECYCLE_DIR / f"{lifecycle_id}.json"

    if lc_file.exists():
        try:
            with open(lc_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    return None


def get_audit_events(limit: int = 100) -> List[Dict[str, Any]]:
    """Get audit events."""
    events = []

    # Check audit directory
    if AUDIT_DIR.exists():
        for audit_file in sorted(AUDIT_DIR.glob("*.json"), reverse=True)[:limit]:
            try:
                with open(audit_file) as f:
                    events.extend(json.load(f) if isinstance(json.load(f), list) else [json.load(f)])
            except (json.JSONDecodeError, IOError):
                pass

    # Also check job state for audit-like entries
    job_state_file = JOBS_DIR / "job_state.json"
    if job_state_file.exists():
        try:
            with open(job_state_file) as f:
                data = json.load(f)
                for job_id, job in data.get("jobs", {}).items():
                    if job.get("completed_at"):
                        events.append({
                            "timestamp": job.get("completed_at"),
                            "event_type": "job_completed",
                            "actor": "system",
                            "action": f"Job {job.get('task_type', 'unknown')} completed",
                            "project": job.get("project_name", "unknown"),
                            "details": {"state": job.get("state"), "job_id": job_id}
                        })
        except (json.JSONDecodeError, IOError):
            pass

    # Sort by timestamp descending
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:limit]


def get_runtime_signals() -> List[Dict[str, Any]]:
    """Get runtime signals."""
    signals = []

    if SIGNALS_DIR.exists():
        for sig_file in SIGNALS_DIR.glob("*.json"):
            try:
                with open(sig_file) as f:
                    sig = json.load(f)
                    signals.append(sig)
            except (json.JSONDecodeError, IOError):
                pass

    signals.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    return signals


def get_incidents(limit: int = 50) -> List[Dict[str, Any]]:
    """Get incidents."""
    incidents = []

    if INCIDENTS_DIR.exists():
        for inc_file in INCIDENTS_DIR.glob("*.json"):
            try:
                with open(inc_file) as f:
                    inc = json.load(f)
                    incidents.append(inc)
            except (json.JSONDecodeError, IOError):
                pass

    incidents.sort(key=lambda i: i.get("timestamp", ""), reverse=True)
    return incidents[:limit]


def get_recommendations(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recommendations."""
    recommendations = []

    if RECOMMENDATIONS_DIR.exists():
        for rec_file in RECOMMENDATIONS_DIR.glob("*.json"):
            try:
                with open(rec_file) as f:
                    rec = json.load(f)
                    recommendations.append(rec)
            except (json.JSONDecodeError, IOError):
                pass

    recommendations.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return recommendations[:limit]


def get_executions(limit: int = 50) -> List[Dict[str, Any]]:
    """Get execution records."""
    executions = []

    if EXECUTIONS_DIR.exists():
        for exec_file in EXECUTIONS_DIR.glob("*.json"):
            try:
                with open(exec_file) as f:
                    execution = json.load(f)
                    executions.append(execution)
            except (json.JSONDecodeError, IOError):
                pass

    executions.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return executions[:limit]


def get_project_logs(project_name: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Get detailed logs for a project."""
    logs = []

    # Source 1: Job logs
    job_state_file = JOBS_DIR / "job_state.json"
    if job_state_file.exists():
        try:
            with open(job_state_file) as f:
                data = json.load(f)
                for job_id, job in data.get("jobs", {}).items():
                    if job.get("project_name") == project_name:
                        logs.append({
                            "timestamp": job.get("created_at", ""),
                            "type": "job",
                            "event": f"Job created: {job.get('task_type', 'unknown')}",
                            "details": {
                                "job_id": job_id,
                                "state": job.get("state", "unknown"),
                                "task_type": job.get("task_type", ""),
                                "aspect": job.get("aspect", ""),
                            }
                        })
                        if job.get("started_at"):
                            logs.append({
                                "timestamp": job.get("started_at"),
                                "type": "job",
                                "event": f"Job started: {job_id[:8]}...",
                                "details": {"job_id": job_id}
                            })
                        if job.get("completed_at"):
                            logs.append({
                                "timestamp": job.get("completed_at"),
                                "type": "job",
                                "event": f"Job completed: {job.get('state', 'unknown')}",
                                "details": {
                                    "job_id": job_id,
                                    "result": job.get("result", "")
                                }
                            })
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read job state: {e}")

    # Source 2: Lifecycle logs
    if LIFECYCLE_DIR.exists():
        for lc_file in LIFECYCLE_DIR.glob("*.json"):
            try:
                with open(lc_file) as f:
                    lc = json.load(f)
                    if lc.get("project_name") == project_name:
                        # Add state transitions
                        for cycle in lc.get("cycle_history", []):
                            logs.append({
                                "timestamp": cycle.get("timestamp", ""),
                                "type": "lifecycle",
                                "event": f"State: {cycle.get('from_state', '?')} → {cycle.get('to_state', '?')}",
                                "details": {
                                    "cycle": cycle.get("cycle_number"),
                                    "trigger": cycle.get("trigger", ""),
                                    "aspect": lc.get("aspect", "core")
                                }
                            })
                        # Add feedback entries
                        for fb in lc.get("feedback_history", []):
                            logs.append({
                                "timestamp": fb.get("timestamp", ""),
                                "type": "feedback",
                                "event": f"Feedback: {fb.get('type', 'unknown')}",
                                "details": fb
                            })
            except (json.JSONDecodeError, IOError):
                pass

    # Source 3: Registry events
    registry_file = REGISTRY_DIR / "projects.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                data = json.load(f)
                proj = data.get("projects", {}).get(project_name, {})
                if proj:
                    logs.append({
                        "timestamp": proj.get("created_at", ""),
                        "type": "registry",
                        "event": "Project registered",
                        "details": {
                            "created_by": proj.get("created_by", "unknown"),
                            "fingerprint": proj.get("identity", {}).get("fingerprint", "")[:16]
                        }
                    })
        except (json.JSONDecodeError, IOError):
            pass

    # Source 4: Project-specific log file
    project_log = LOGS_DIR / f"{project_name}.log"
    if project_log.exists():
        try:
            with open(project_log) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entry = json.loads(line)
                            logs.append(entry)
                        except json.JSONDecodeError:
                            # Plain text log line
                            logs.append({
                                "timestamp": "",
                                "type": "log",
                                "event": line[:200],
                                "details": {}
                            })
        except IOError:
            pass

    # Sort by timestamp descending and limit
    logs.sort(key=lambda l: l.get("timestamp", ""), reverse=True)
    return logs[:limit]


def get_project_details(project_name: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a project."""
    details = {
        "name": project_name,
        "exists": False,
        "registry": None,
        "ipc": None,
        "lifecycle": [],
        "jobs": [],
        "files": [],
        "cleanup_info": {}
    }

    # Check registry
    registry_file = REGISTRY_DIR / "projects.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                data = json.load(f)
                if project_name in data.get("projects", {}):
                    details["registry"] = data["projects"][project_name]
                    details["exists"] = True
        except (json.JSONDecodeError, IOError):
            pass

    # Check project directory
    project_dir = PROJECTS_DIR / project_name
    if project_dir.exists():
        details["exists"] = True
        details["project_dir"] = str(project_dir)

        # List files
        try:
            for f in project_dir.rglob("*"):
                if f.is_file():
                    details["files"].append({
                        "path": str(f.relative_to(project_dir)),
                        "size": f.stat().st_size,
                        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
                    })
        except Exception as e:
            logger.warning(f"Failed to list project files: {e}")

        # Check IPC
        ipc_file = project_dir / "INTERNAL_PROJECT_CONTRACT.yaml"
        if ipc_file.exists():
            try:
                import yaml
                with open(ipc_file) as f:
                    details["ipc"] = yaml.safe_load(f)
            except Exception:
                pass

    # Check lifecycle
    if LIFECYCLE_DIR.exists():
        for lc_file in LIFECYCLE_DIR.glob("*.json"):
            try:
                with open(lc_file) as f:
                    lc = json.load(f)
                    if lc.get("project_name") == project_name:
                        details["lifecycle"].append({
                            "lifecycle_id": lc.get("lifecycle_id"),
                            "aspect": lc.get("aspect"),
                            "state": lc.get("state"),
                            "mode": lc.get("mode")
                        })
                        details["exists"] = True
            except (json.JSONDecodeError, IOError):
                pass

    # Check jobs
    job_state_file = JOBS_DIR / "job_state.json"
    if job_state_file.exists():
        try:
            with open(job_state_file) as f:
                data = json.load(f)
                for job_id, job in data.get("jobs", {}).items():
                    if job.get("project_name") == project_name:
                        details["jobs"].append({
                            "job_id": job_id,
                            "state": job.get("state"),
                            "task_type": job.get("task_type"),
                            "created_at": job.get("created_at")
                        })
                        details["exists"] = True
        except (json.JSONDecodeError, IOError):
            pass

    # Build cleanup info
    details["cleanup_info"] = {
        "has_registry_entry": details["registry"] is not None,
        "has_project_dir": project_dir.exists() if project_dir else False,
        "has_lifecycle": len(details["lifecycle"]) > 0,
        "has_jobs": len(details["jobs"]) > 0,
        "intent_baseline": (INTENT_BASELINES_DIR / f"{project_name}.json").exists() if INTENT_BASELINES_DIR.exists() else False,
        "intent_contract": (INTENT_CONTRACTS_DIR / f"{project_name}.json").exists() if INTENT_CONTRACTS_DIR.exists() else False,
    }

    return details if details["exists"] else None


def cleanup_project(project_name: str) -> Dict[str, Any]:
    """
    Completely remove a project from all storage locations.

    Returns dict with cleanup results.
    """
    results = {
        "project_name": project_name,
        "success": True,
        "cleaned": [],
        "errors": [],
        "timestamp": datetime.utcnow().isoformat()
    }

    # 1. Remove from registry
    registry_file = REGISTRY_DIR / "projects.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                data = json.load(f)

            if project_name in data.get("projects", {}):
                del data["projects"][project_name]
                with open(registry_file, "w") as f:
                    json.dump(data, f, indent=2)
                results["cleaned"].append("registry")
                logger.info(f"Removed {project_name} from registry")
        except Exception as e:
            results["errors"].append(f"Registry cleanup failed: {e}")
            logger.error(f"Failed to clean registry for {project_name}: {e}")

    # 2. Remove project directory
    project_dir = PROJECTS_DIR / project_name
    if project_dir.exists():
        try:
            shutil.rmtree(project_dir)
            results["cleaned"].append("project_directory")
            logger.info(f"Removed project directory: {project_dir}")
        except Exception as e:
            results["errors"].append(f"Project directory cleanup failed: {e}")
            logger.error(f"Failed to remove project directory {project_dir}: {e}")

    # 3. Remove from job state
    job_state_file = JOBS_DIR / "job_state.json"
    if job_state_file.exists():
        try:
            with open(job_state_file) as f:
                data = json.load(f)

            jobs_to_remove = [
                job_id for job_id, job in data.get("jobs", {}).items()
                if job.get("project_name") == project_name
            ]

            for job_id in jobs_to_remove:
                del data["jobs"][job_id]

            if jobs_to_remove:
                with open(job_state_file, "w") as f:
                    json.dump(data, f, indent=2)
                results["cleaned"].append(f"jobs ({len(jobs_to_remove)})")
                logger.info(f"Removed {len(jobs_to_remove)} jobs for {project_name}")
        except Exception as e:
            results["errors"].append(f"Job state cleanup failed: {e}")
            logger.error(f"Failed to clean job state for {project_name}: {e}")

    # 4. Remove lifecycle files
    if LIFECYCLE_DIR.exists():
        try:
            removed_lc = 0
            for lc_file in LIFECYCLE_DIR.glob("*.json"):
                try:
                    with open(lc_file) as f:
                        lc = json.load(f)
                    if lc.get("project_name") == project_name:
                        lc_file.unlink()
                        removed_lc += 1
                except (json.JSONDecodeError, IOError):
                    pass

            if removed_lc:
                results["cleaned"].append(f"lifecycle ({removed_lc})")
                logger.info(f"Removed {removed_lc} lifecycle files for {project_name}")
        except Exception as e:
            results["errors"].append(f"Lifecycle cleanup failed: {e}")

    # 5. Remove intent baseline
    intent_baseline = INTENT_BASELINES_DIR / f"{project_name}.json"
    if intent_baseline.exists():
        try:
            intent_baseline.unlink()
            results["cleaned"].append("intent_baseline")
            logger.info(f"Removed intent baseline for {project_name}")
        except Exception as e:
            results["errors"].append(f"Intent baseline cleanup failed: {e}")

    # 6. Remove intent contract
    intent_contract = INTENT_CONTRACTS_DIR / f"{project_name}.json"
    if intent_contract.exists():
        try:
            intent_contract.unlink()
            results["cleaned"].append("intent_contract")
            logger.info(f"Removed intent contract for {project_name}")
        except Exception as e:
            results["errors"].append(f"Intent contract cleanup failed: {e}")

    # 7. Remove project log file
    if LOGS_DIR.exists():
        project_log = LOGS_DIR / f"{project_name}.log"
        if project_log.exists():
            try:
                project_log.unlink()
                results["cleaned"].append("log_file")
            except Exception as e:
                results["errors"].append(f"Log file cleanup failed: {e}")

    results["success"] = len(results["errors"]) == 0
    return results


def get_system_stats() -> Dict[str, Any]:
    """Get system statistics for dashboard."""
    stats = {
        "total_projects": 0,
        "active_jobs": 0,
        "queued_jobs": 0,
        "completed_jobs_today": 0,
        "failed_jobs_today": 0,
        "active_lifecycles": 0,
        "pending_feedback": 0,
        "total_incidents": 0,
        "open_recommendations": 0,
        "system_health": "healthy"
    }

    # Count projects
    projects = get_all_projects()
    stats["total_projects"] = len(projects)

    # Count jobs
    job_state_file = JOBS_DIR / "job_state.json"
    if job_state_file.exists():
        try:
            with open(job_state_file) as f:
                data = json.load(f)

            today = datetime.utcnow().date().isoformat()
            for job in data.get("jobs", {}).values():
                state = job.get("state", "")
                if state == "running":
                    stats["active_jobs"] += 1
                elif state == "queued":
                    stats["queued_jobs"] += 1
                elif state == "completed" and job.get("completed_at", "").startswith(today):
                    stats["completed_jobs_today"] += 1
                elif state == "failed" and job.get("completed_at", "").startswith(today):
                    stats["failed_jobs_today"] += 1
        except (json.JSONDecodeError, IOError):
            pass

    # Count lifecycles
    lifecycles = get_all_lifecycles()
    stats["active_lifecycles"] = len([l for l in lifecycles if l.get("state") not in ["completed", "archived"]])
    stats["pending_feedback"] = len([l for l in lifecycles if l.get("feedback_pending")])

    # Count incidents
    incidents = get_incidents(100)
    stats["total_incidents"] = len(incidents)

    # Count recommendations
    recommendations = get_recommendations(100)
    stats["open_recommendations"] = len([r for r in recommendations if r.get("status") == "pending"])

    # Determine health
    if stats["failed_jobs_today"] > 5:
        stats["system_health"] = "degraded"
    elif stats["total_incidents"] > 10:
        stats["system_health"] = "warning"

    return stats


# -----------------------------------------------------------------------------
# HTML Templates (inline for simplicity)
# -----------------------------------------------------------------------------
BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa;
    min-height: 100vh;
}
.navbar {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 16px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.navbar h1 { color: white; font-size: 20px; }
.navbar .user-info {
    display: flex;
    align-items: center;
    gap: 16px;
}
.navbar .user-info span { color: #a0aec0; }
.navbar a {
    color: #a0aec0;
    text-decoration: none;
    padding: 8px 16px;
    border-radius: 6px;
    transition: background 0.2s;
}
.navbar a:hover { background: rgba(255,255,255,0.1); color: white; }
.navbar a.logout { color: #f87171; }
.navbar a.logout:hover { background: rgba(248,113,113,0.2); }

.nav-links {
    display: flex;
    gap: 8px;
}
.nav-links a {
    color: #a0aec0;
    text-decoration: none;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 14px;
}
.nav-links a:hover { background: rgba(255,255,255,0.1); color: white; }
.nav-links a.active { background: rgba(74,144,217,0.3); color: white; }

.container { max-width: 1400px; margin: 0 auto; padding: 24px; }

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}
.page-header h1 { font-size: 28px; color: #1a1a2e; }

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 24px;
}
.stat-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.stat-card h3 { color: #666; font-size: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
.stat-card .value { font-size: 28px; font-weight: 700; color: #1a1a2e; margin-top: 4px; }
.stat-card.healthy .value { color: #10b981; }
.stat-card.warning .value { color: #f59e0b; }
.stat-card.error .value { color: #ef4444; }

.section {
    background: white;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 24px;
    overflow: hidden;
}
.section-header {
    background: #f8fafc;
    padding: 16px 24px;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.section-header h2 { font-size: 18px; color: #1a1a2e; }
.section-body { padding: 0; }

table { width: 100%; border-collapse: collapse; }
th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #e2e8f0; }
th { background: #f8fafc; font-weight: 600; color: #4a5568; font-size: 12px; text-transform: uppercase; }
tr:hover { background: #f8fafc; }
td { font-size: 14px; }

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge.planning { background: #dbeafe; color: #1d4ed8; }
.badge.running { background: #dcfce7; color: #15803d; }
.badge.completed { background: #d1fae5; color: #047857; }
.badge.failed { background: #fee2e2; color: #dc2626; }
.badge.queued { background: #fef3c7; color: #b45309; }
.badge.created { background: #e0e7ff; color: #4338ca; }
.badge.pending { background: #fef3c7; color: #b45309; }
.badge.approved { background: #dcfce7; color: #15803d; }
.badge.dismissed { background: #f1f5f9; color: #64748b; }
.badge.unknown { background: #f1f5f9; color: #64748b; }
.badge.healthy { background: #dcfce7; color: #15803d; }
.badge.degraded { background: #fef3c7; color: #b45309; }
.badge.error { background: #fee2e2; color: #dc2626; }
.badge.warning { background: #fef3c7; color: #b45309; }
.badge.project { background: #e0e7ff; color: #4338ca; }
.badge.change { background: #fce7f3; color: #be185d; }

.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 6px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none;
    display: inline-block;
}
.btn-primary { background: #4a90d9; color: white; }
.btn-primary:hover { background: #357abd; }
.btn-danger { background: #ef4444; color: white; }
.btn-danger:hover { background: #dc2626; }
.btn-outline {
    background: white;
    border: 1px solid #e2e8f0;
    color: #4a5568;
}
.btn-outline:hover { background: #f8fafc; }
.btn-sm { padding: 4px 8px; font-size: 12px; }

.action-links a {
    color: #4a90d9;
    text-decoration: none;
    margin-right: 12px;
    font-size: 13px;
}
.action-links a:hover { text-decoration: underline; }
.action-links a.danger { color: #ef4444; }

.empty-state {
    text-align: center;
    padding: 48px;
    color: #9ca3af;
}
.empty-state h3 { margin-bottom: 8px; color: #6b7280; }

.refresh-btn {
    background: none;
    border: 1px solid #e2e8f0;
    padding: 8px 16px;
    border-radius: 6px;
    cursor: pointer;
    color: #4a5568;
    font-size: 14px;
}
.refresh-btn:hover { background: #f8fafc; }

.mono { font-family: 'SF Mono', Monaco, monospace; font-size: 12px; }

.truncate {
    max-width: 300px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.breadcrumb { margin-bottom: 20px; color: #6b7280; font-size: 14px; }
.breadcrumb a { color: #4a90d9; text-decoration: none; }
.breadcrumb a:hover { text-decoration: underline; }

.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    padding: 20px;
}
.info-item label { display: block; color: #6b7280; font-size: 12px; margin-bottom: 4px; text-transform: uppercase; }
.info-item span { font-size: 14px; color: #1a1a2e; }

.log-entry {
    background: white;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-left: 4px solid #e2e8f0;
}
.log-entry.job { border-left-color: #4a90d9; }
.log-entry.lifecycle { border-left-color: #10b981; }
.log-entry.feedback { border-left-color: #f59e0b; }
.log-entry.registry { border-left-color: #8b5cf6; }
.log-entry.log { border-left-color: #6b7280; }

.log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}
.log-type {
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
    background: #f1f5f9;
}
.log-time { color: #6b7280; font-size: 12px; }
.log-event { font-size: 14px; color: #1a1a2e; margin-bottom: 8px; }
.log-details {
    background: #f8fafc;
    padding: 12px;
    border-radius: 6px;
    font-family: monospace;
    font-size: 12px;
    color: #4a5568;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
}

.warning-box {
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
}
.warning-box h2 { color: #b45309; margin-bottom: 8px; font-size: 18px; }
.warning-box p { color: #92400e; }

.result-box {
    background: white;
    border-radius: 12px;
    padding: 24px;
    margin-top: 24px;
}
.result-box.success { border: 2px solid #10b981; }
.result-box.error { border: 2px solid #ef4444; }
.result-box h3 { margin-bottom: 12px; }
.result-box ul { margin-left: 20px; margin-top: 8px; }

.tabs {
    display: flex;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 24px;
}
.tabs a {
    padding: 12px 20px;
    text-decoration: none;
    color: #6b7280;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
}
.tabs a:hover { color: #1a1a2e; }
.tabs a.active {
    color: #4a90d9;
    border-bottom-color: #4a90d9;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
}
.card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.card h3 { font-size: 16px; color: #1a1a2e; margin-bottom: 12px; }
.card p { font-size: 13px; color: #6b7280; margin-bottom: 8px; }
.card .meta { font-size: 12px; color: #9ca3af; }
"""

LOGIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Platform - Login</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .login-container {
            background: rgba(255,255,255,0.95);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 400px;
        }
        .logo { text-align: center; margin-bottom: 30px; }
        .logo h1 { color: #1a1a2e; font-size: 24px; }
        .logo p { color: #666; font-size: 14px; margin-top: 5px; }
        .form-group { margin-bottom: 20px; }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            color: #333;
            font-weight: 500;
        }
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        .form-group input:focus {
            outline: none;
            border-color: #4a90d9;
        }
        .btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, #4a90d9 0%, #357abd 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(74,144,217,0.4);
        }
        .error {
            background: #fee2e2;
            color: #dc2626;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <h1>AI Development Platform</h1>
            <p>Control Panel</p>
        </div>
        {% if error %}
        <div class="error">{{ error }}</div>
        {% endif %}
        <form method="POST" action="/dashboard/login">
            <div class="form-group">
                <label for="email">Email</label>
                <input type="email" id="email" name="email" required placeholder="Enter your email">
            </div>
            <div class="form-group">
                <label for="password">Password</label>
                <input type="password" id="password" name="password" required placeholder="Enter your password">
            </div>
            <button type="submit" class="btn">Sign In</button>
        </form>
    </div>
</body>
</html>
"""

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Platform Dashboard</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/live">Live</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }} ({{ user.role }})</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Projects</h3>
                <div class="value">{{ stats.total_projects }}</div>
            </div>
            <div class="stat-card">
                <h3>Active Jobs</h3>
                <div class="value">{{ stats.active_jobs }}</div>
            </div>
            <div class="stat-card">
                <h3>Queued Jobs</h3>
                <div class="value">{{ stats.queued_jobs }}</div>
            </div>
            <div class="stat-card">
                <h3>Active Lifecycles</h3>
                <div class="value">{{ stats.active_lifecycles }}</div>
            </div>
            <div class="stat-card {% if stats.pending_feedback > 0 %}warning{% endif %}">
                <h3>Pending Feedback</h3>
                <div class="value">{{ stats.pending_feedback }}</div>
            </div>
            <div class="stat-card {% if stats.failed_jobs_today > 0 %}error{% else %}healthy{% endif %}">
                <h3>Failed Today</h3>
                <div class="value">{{ stats.failed_jobs_today }}</div>
            </div>
            <div class="stat-card {% if stats.system_health == 'healthy' %}healthy{% elif stats.system_health == 'warning' %}warning{% else %}error{% endif %}">
                <h3>System Health</h3>
                <div class="value">{{ stats.system_health | upper }}</div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Projects</h2>
                <button class="refresh-btn" onclick="location.reload()">Refresh</button>
            </div>
            <div class="section-body">
                {% if projects %}
                <table>
                    <thead>
                        <tr>
                            <th>Project Name</th>
                            <th>Status</th>
                            <th>Created</th>
                            <th>Updated</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for project in projects %}
                        <tr>
                            <td>
                                <strong>{{ project.name }}</strong>
                                {% if project.fingerprint %}
                                <br><small class="mono" style="color: #9ca3af;">{{ project.fingerprint }}</small>
                                {% endif %}
                            </td>
                            <td><span class="badge {{ project.status }}">{{ project.status }}</span></td>
                            <td>{{ project.created_at[:16] if project.created_at else '-' }}</td>
                            <td>{{ project.updated_at[:16] if project.updated_at else '-' }}</td>
                            <td class="action-links">
                                <a href="/dashboard/project/{{ project.name }}">View</a>
                                <a href="/dashboard/project/{{ project.name }}/logs">Logs</a>
                                <a href="/dashboard/project/{{ project.name }}/cleanup" class="danger">Cleanup</a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <div class="empty-state">
                    <h3>No Projects Found</h3>
                    <p>Create a new project via Telegram bot to get started.</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

JOBS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jobs - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs" class="active">Jobs</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="page-header">
            <h1>Claude Jobs</h1>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Running</h3>
                <div class="value">{{ running_count }}</div>
            </div>
            <div class="stat-card">
                <h3>Queued</h3>
                <div class="value">{{ queued_count }}</div>
            </div>
            <div class="stat-card">
                <h3>Completed Today</h3>
                <div class="value">{{ completed_today }}</div>
            </div>
            <div class="stat-card">
                <h3>Failed Today</h3>
                <div class="value">{{ failed_today }}</div>
            </div>
        </div>

        {% if running_jobs %}
        <div class="section">
            <div class="section-header">
                <h2>Currently Running ({{ running_jobs | length }})</h2>
            </div>
            <div class="section-body">
                <table>
                    <thead>
                        <tr>
                            <th>Job ID</th>
                            <th>Project</th>
                            <th>Type</th>
                            <th>Aspect</th>
                            <th>Started</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for job in running_jobs %}
                        <tr>
                            <td class="mono">{{ job.job_id[:12] }}...</td>
                            <td>{{ job.project_name }}</td>
                            <td><span class="badge running">{{ job.task_type }}</span></td>
                            <td>{{ job.aspect }}</td>
                            <td>{{ job.started_at[:16] if job.started_at else '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}

        {% if queue_jobs %}
        <div class="section">
            <div class="section-header">
                <h2>Queue ({{ queue_jobs | length }})</h2>
            </div>
            <div class="section-body">
                <table>
                    <thead>
                        <tr>
                            <th>Job ID</th>
                            <th>Project</th>
                            <th>Type</th>
                            <th>Priority</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for job in queue_jobs %}
                        <tr>
                            <td class="mono">{{ job.job_id[:12] }}...</td>
                            <td>{{ job.project_name }}</td>
                            <td>{{ job.task_type }}</td>
                            <td>{{ job.priority }}</td>
                            <td>{{ job.created_at[:16] if job.created_at else '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <div class="section-header">
                <h2>All Jobs ({{ jobs | length }})</h2>
            </div>
            <div class="section-body">
                {% if jobs %}
                <table>
                    <thead>
                        <tr>
                            <th>Job ID</th>
                            <th>Project</th>
                            <th>Type</th>
                            <th>State</th>
                            <th>Created</th>
                            <th>Completed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for job in jobs[:50] %}
                        <tr>
                            <td class="mono">{{ job.job_id[:12] }}...</td>
                            <td>{{ job.project_name }}</td>
                            <td>{{ job.task_type }}</td>
                            <td><span class="badge {{ job.state }}">{{ job.state }}</span></td>
                            <td>{{ job.created_at[:16] if job.created_at else '-' }}</td>
                            <td>{{ job.completed_at[:16] if job.completed_at else '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <div class="empty-state">
                    <h3>No Jobs Found</h3>
                    <p>Jobs will appear here when created via the Telegram bot.</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        setTimeout(() => location.reload(), 15000);
    </script>
</body>
</html>
"""

LIFECYCLES_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lifecycles - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/lifecycles" class="active">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="page-header">
            <h1>Lifecycle Management</h1>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Lifecycles</h3>
                <div class="value">{{ lifecycles | length }}</div>
            </div>
            <div class="stat-card {% if pending_feedback > 0 %}warning{% endif %}">
                <h3>Pending Feedback</h3>
                <div class="value">{{ pending_feedback }}</div>
            </div>
            <div class="stat-card">
                <h3>Project Mode</h3>
                <div class="value">{{ project_mode_count }}</div>
            </div>
            <div class="stat-card">
                <h3>Change Mode</h3>
                <div class="value">{{ change_mode_count }}</div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Active Lifecycles</h2>
            </div>
            <div class="section-body">
                {% if lifecycles %}
                <table>
                    <thead>
                        <tr>
                            <th>Lifecycle ID</th>
                            <th>Project</th>
                            <th>Aspect</th>
                            <th>Mode</th>
                            <th>State</th>
                            <th>Cycles</th>
                            <th>Updated</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for lc in lifecycles %}
                        <tr>
                            <td class="mono">{{ lc.lifecycle_id[:12] }}...</td>
                            <td>{{ lc.project_name }}</td>
                            <td>{{ lc.aspect }}</td>
                            <td><span class="badge {{ lc.mode }}">{{ lc.mode }}</span></td>
                            <td>
                                <span class="badge {{ lc.state }}">{{ lc.state }}</span>
                                {% if lc.feedback_pending %}
                                <span class="badge warning">FEEDBACK</span>
                                {% endif %}
                            </td>
                            <td>{{ lc.cycle_count }}</td>
                            <td>{{ lc.updated_at[:16] if lc.updated_at else '-' }}</td>
                            <td class="action-links">
                                <a href="/dashboard/lifecycle/{{ lc.lifecycle_id }}">View</a>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <div class="empty-state">
                    <h3>No Lifecycles Found</h3>
                    <p>Lifecycles are created when projects start development.</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

LIFECYCLE_DETAIL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lifecycle {{ lifecycle_id }} - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/lifecycles" class="active">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="breadcrumb">
            <a href="/dashboard/main">Dashboard</a> / <a href="/dashboard/lifecycles">Lifecycles</a> / {{ lifecycle.lifecycle_id[:16] }}...
        </div>

        <div class="page-header">
            <h1>Lifecycle Details</h1>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Overview</h2>
            </div>
            <div class="info-grid">
                <div class="info-item">
                    <label>Lifecycle ID</label>
                    <span class="mono">{{ lifecycle.lifecycle_id }}</span>
                </div>
                <div class="info-item">
                    <label>Project</label>
                    <span>{{ lifecycle.project_name }}</span>
                </div>
                <div class="info-item">
                    <label>Aspect</label>
                    <span>{{ lifecycle.aspect }}</span>
                </div>
                <div class="info-item">
                    <label>Mode</label>
                    <span class="badge {{ lifecycle.mode }}">{{ lifecycle.mode }}</span>
                </div>
                <div class="info-item">
                    <label>Current State</label>
                    <span class="badge {{ lifecycle.state }}">{{ lifecycle.state }}</span>
                </div>
                <div class="info-item">
                    <label>Feedback Pending</label>
                    <span>{{ 'Yes' if lifecycle.feedback_pending else 'No' }}</span>
                </div>
            </div>
        </div>

        {% if lifecycle.cycle_history %}
        <div class="section">
            <div class="section-header">
                <h2>Cycle History ({{ lifecycle.cycle_history | length }})</h2>
            </div>
            <div class="section-body">
                <table>
                    <thead>
                        <tr>
                            <th>Cycle</th>
                            <th>From State</th>
                            <th>To State</th>
                            <th>Trigger</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for cycle in lifecycle.cycle_history | reverse %}
                        <tr>
                            <td>{{ cycle.cycle_number }}</td>
                            <td>{{ cycle.from_state }}</td>
                            <td>{{ cycle.to_state }}</td>
                            <td>{{ cycle.trigger }}</td>
                            <td>{{ cycle.timestamp[:19] if cycle.timestamp else '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}

        {% if lifecycle.feedback_history %}
        <div class="section">
            <div class="section-header">
                <h2>Feedback History</h2>
            </div>
            <div class="section-body">
                {% for fb in lifecycle.feedback_history | reverse %}
                <div class="log-entry feedback">
                    <div class="log-header">
                        <span class="log-type">{{ fb.type }}</span>
                        <span class="log-time">{{ fb.timestamp }}</span>
                    </div>
                    <div class="log-event">{{ fb.content[:200] if fb.content else 'No content' }}...</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

AUDIT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audit Trail - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit" class="active">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="page-header">
            <h1>Audit Trail</h1>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Recent Events ({{ events | length }})</h2>
            </div>
            <div class="section-body">
                {% if events %}
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Event Type</th>
                            <th>Actor</th>
                            <th>Action</th>
                            <th>Project</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for event in events %}
                        <tr>
                            <td>{{ event.timestamp[:19] if event.timestamp else '-' }}</td>
                            <td><span class="badge {{ event.event_type }}">{{ event.event_type }}</span></td>
                            <td>{{ event.actor or 'system' }}</td>
                            <td class="truncate">{{ event.action }}</td>
                            <td>{{ event.project or '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                {% else %}
                <div class="empty-state">
                    <h3>No Audit Events</h3>
                    <p>System events will be logged here.</p>
                </div>
                {% endif %}
            </div>
        </div>
    </div>

    <script>
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

RUNTIME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Runtime - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime" class="active">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="page-header">
            <h1>Runtime Monitoring</h1>
            <button class="refresh-btn" onclick="location.reload()">Refresh</button>
        </div>

        <div class="tabs">
            <a href="#signals" class="active" onclick="showTab('signals')">Signals</a>
            <a href="#incidents" onclick="showTab('incidents')">Incidents ({{ incidents | length }})</a>
            <a href="#recommendations" onclick="showTab('recommendations')">Recommendations ({{ recommendations | length }})</a>
        </div>

        <div id="signals" class="tab-content">
            <div class="section">
                <div class="section-header">
                    <h2>Runtime Signals</h2>
                </div>
                <div class="section-body">
                    {% if signals %}
                    <table>
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Type</th>
                                <th>Source</th>
                                <th>Message</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for signal in signals[:30] %}
                            <tr>
                                <td>{{ signal.timestamp[:19] if signal.timestamp else '-' }}</td>
                                <td><span class="badge {{ signal.type }}">{{ signal.type }}</span></td>
                                <td>{{ signal.source or 'system' }}</td>
                                <td class="truncate">{{ signal.message or signal.content or '-' }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% else %}
                    <div class="empty-state">
                        <h3>No Signals</h3>
                        <p>Runtime signals will appear here.</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>

        <div id="incidents" class="tab-content" style="display: none;">
            <div class="section">
                <div class="section-header">
                    <h2>Incidents</h2>
                </div>
                <div class="section-body">
                    {% if incidents %}
                    <table>
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Severity</th>
                                <th>Category</th>
                                <th>Description</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for inc in incidents %}
                            <tr>
                                <td>{{ inc.timestamp[:19] if inc.timestamp else '-' }}</td>
                                <td><span class="badge {{ inc.severity }}">{{ inc.severity }}</span></td>
                                <td>{{ inc.category or 'general' }}</td>
                                <td class="truncate">{{ inc.description or inc.message or '-' }}</td>
                                <td><span class="badge {{ inc.status }}">{{ inc.status or 'open' }}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% else %}
                    <div class="empty-state">
                        <h3>No Incidents</h3>
                        <p>No incidents have been recorded.</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>

        <div id="recommendations" class="tab-content" style="display: none;">
            <div class="section">
                <div class="section-header">
                    <h2>Recommendations</h2>
                </div>
                <div class="section-body">
                    {% if recommendations %}
                    <table>
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Type</th>
                                <th>Recommendation</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for rec in recommendations %}
                            <tr>
                                <td>{{ rec.timestamp[:19] if rec.timestamp else '-' }}</td>
                                <td><span class="badge {{ rec.type }}">{{ rec.type or 'general' }}</span></td>
                                <td class="truncate">{{ rec.content or rec.recommendation or '-' }}</td>
                                <td><span class="badge {{ rec.status }}">{{ rec.status or 'pending' }}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                    {% else %}
                    <div class="empty-state">
                        <h3>No Recommendations</h3>
                        <p>System recommendations will appear here.</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>

    <script>
        function showTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(t => t.style.display = 'none');
            document.querySelectorAll('.tabs a').forEach(a => a.classList.remove('active'));
            document.getElementById(tabId).style.display = 'block';
            event.target.classList.add('active');
        }
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
"""

PROJECT_DETAIL_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ project.name }} - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/live">Live</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="breadcrumb">
            <a href="/dashboard/main">Dashboard</a> / Project / {{ project.name }}
        </div>

        <div class="page-header">
            <h1>{{ project.name }}</h1>
            <a href="/dashboard/project/{{ project.name }}/cleanup" class="btn btn-danger">Cleanup Project</a>
        </div>

        {% if project.registry %}
        <div class="section">
            <div class="section-header">
                <h2>Registry Information</h2>
            </div>
            <div class="info-grid">
                <div class="info-item">
                    <label>Status</label>
                    <span class="badge {{ project.registry.current_status }}">{{ project.registry.current_status }}</span>
                </div>
                <div class="info-item">
                    <label>Created By</label>
                    <span>{{ project.registry.created_by or 'Unknown' }}</span>
                </div>
                <div class="info-item">
                    <label>Created At</label>
                    <span>{{ project.registry.created_at }}</span>
                </div>
                <div class="info-item">
                    <label>Fingerprint</label>
                    <span class="mono">{{ project.registry.identity.fingerprint[:32] if project.registry.identity else 'N/A' }}...</span>
                </div>
            </div>
        </div>
        {% endif %}

        <div class="section">
            <div class="section-header">
                <h2>Storage Locations</h2>
            </div>
            <div class="section-body">
                <table>
                    <tr>
                        <td>Registry Entry</td>
                        <td><span class="badge {% if project.cleanup_info.has_registry_entry %}healthy{% else %}unknown{% endif %}">
                            {% if project.cleanup_info.has_registry_entry %}EXISTS{% else %}NONE{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Project Directory</td>
                        <td><span class="badge {% if project.cleanup_info.has_project_dir %}healthy{% else %}unknown{% endif %}">
                            {% if project.cleanup_info.has_project_dir %}EXISTS{% else %}NONE{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Lifecycle State</td>
                        <td><span class="badge {% if project.cleanup_info.has_lifecycle %}healthy{% else %}unknown{% endif %}">
                            {% if project.cleanup_info.has_lifecycle %}EXISTS ({{ project.lifecycle | length }}){% else %}NONE{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Jobs</td>
                        <td><span class="badge {% if project.cleanup_info.has_jobs %}healthy{% else %}unknown{% endif %}">
                            {% if project.cleanup_info.has_jobs %}EXISTS ({{ project.jobs | length }}){% else %}NONE{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Intent Baseline</td>
                        <td><span class="badge {% if project.cleanup_info.intent_baseline %}healthy{% else %}unknown{% endif %}">
                            {% if project.cleanup_info.intent_baseline %}EXISTS{% else %}NONE{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Intent Contract</td>
                        <td><span class="badge {% if project.cleanup_info.intent_contract %}healthy{% else %}unknown{% endif %}">
                            {% if project.cleanup_info.intent_contract %}EXISTS{% else %}NONE{% endif %}
                        </span></td>
                    </tr>
                </table>
            </div>
        </div>

        {% if project.jobs %}
        <div class="section">
            <div class="section-header">
                <h2>Jobs ({{ project.jobs | length }})</h2>
            </div>
            <div class="section-body">
                <table>
                    <thead>
                        <tr>
                            <th>Job ID</th>
                            <th>Type</th>
                            <th>State</th>
                            <th>Created</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for job in project.jobs %}
                        <tr>
                            <td class="mono">{{ job.job_id[:12] }}...</td>
                            <td>{{ job.task_type }}</td>
                            <td><span class="badge {{ job.state }}">{{ job.state }}</span></td>
                            <td>{{ job.created_at[:16] if job.created_at else '-' }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}

        {% if project.files %}
        <div class="section">
            <div class="section-header">
                <h2>Files ({{ project.files | length }})</h2>
            </div>
            <div class="section-body" style="max-height: 400px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Path</th>
                            <th>Size</th>
                            <th>Modified</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for file in project.files[:50] %}
                        <tr>
                            <td class="mono">{{ file.path }}</td>
                            <td>{{ file.size }} bytes</td>
                            <td>{{ file.modified[:16] }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

PROJECT_LOGS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ project_name }} Logs - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/live">Live</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="breadcrumb">
            <a href="/dashboard/main">Dashboard</a> / <a href="/dashboard/project/{{ project_name }}">{{ project_name }}</a> / Logs
        </div>

        <div class="page-header">
            <h1>{{ project_name }} - Activity Logs</h1>
            <span style="color: #6b7280;">{{ logs | length }} entries</span>
        </div>

        {% if logs %}
            {% for log in logs %}
            <div class="log-entry {{ log.type }}">
                <div class="log-header">
                    <span class="log-type">{{ log.type }}</span>
                    <span class="log-time">{{ log.timestamp }}</span>
                </div>
                <div class="log-event">{{ log.event }}</div>
                {% if log.details %}
                <div class="log-details">{{ log.details | tojson(indent=2) }}</div>
                {% endif %}
            </div>
            {% endfor %}
        {% else %}
        <div class="empty-state">
            <h3>No Logs Found</h3>
            <p>No activity has been recorded for this project yet.</p>
        </div>
        {% endif %}
    </div>

    <script>
        setTimeout(() => location.reload(), 10000);
    </script>
</body>
</html>
"""

CLEANUP_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cleanup {{ project_name }} - AI Platform</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container" style="max-width: 800px;">
        {% if result %}
        <div class="result-box {% if result.success %}success{% else %}error{% endif %}">
            <h3>{% if result.success %}Cleanup Complete{% else %}Cleanup Completed with Errors{% endif %}</h3>
            {% if result.cleaned %}
            <p><strong>Cleaned:</strong></p>
            <ul>
                {% for item in result.cleaned %}
                <li>{{ item }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            {% if result.errors %}
            <p style="margin-top: 12px;"><strong>Errors:</strong></p>
            <ul style="color: #ef4444;">
                {% for error in result.errors %}
                <li>{{ error }}</li>
                {% endfor %}
            </ul>
            {% endif %}
            <a href="/dashboard/main" class="btn btn-outline" style="margin-top: 16px;">Back to Dashboard</a>
        </div>
        {% else %}
        <div class="warning-box">
            <h2>Warning: Destructive Action</h2>
            <p>You are about to permanently delete all data associated with project <strong>{{ project_name }}</strong>. This action cannot be undone.</p>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Cleanup Project: {{ project_name }}</h2>
            </div>
            <div class="section-body">
                <p style="padding: 20px; color: #6b7280;">The following data will be removed:</p>
                <table>
                    <tr>
                        <td>Registry Entry</td>
                        <td><span class="badge {% if cleanup_info.has_registry_entry %}warning{% else %}unknown{% endif %}">
                            {% if cleanup_info.has_registry_entry %}Will be removed{% else %}Not found{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Project Directory & Files</td>
                        <td><span class="badge {% if cleanup_info.has_project_dir %}warning{% else %}unknown{% endif %}">
                            {% if cleanup_info.has_project_dir %}Will be removed{% else %}Not found{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Lifecycle State</td>
                        <td><span class="badge {% if cleanup_info.has_lifecycle %}warning{% else %}unknown{% endif %}">
                            {% if cleanup_info.has_lifecycle %}Will be removed{% else %}Not found{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Job Records</td>
                        <td><span class="badge {% if cleanup_info.has_jobs %}warning{% else %}unknown{% endif %}">
                            {% if cleanup_info.has_jobs %}Will be removed{% else %}Not found{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Intent Baseline</td>
                        <td><span class="badge {% if cleanup_info.intent_baseline %}warning{% else %}unknown{% endif %}">
                            {% if cleanup_info.intent_baseline %}Will be removed{% else %}Not found{% endif %}
                        </span></td>
                    </tr>
                    <tr>
                        <td>Intent Contract</td>
                        <td><span class="badge {% if cleanup_info.intent_contract %}warning{% else %}unknown{% endif %}">
                            {% if cleanup_info.intent_contract %}Will be removed{% else %}Not found{% endif %}
                        </span></td>
                    </tr>
                </table>
            </div>
        </div>

        <form method="POST" action="/dashboard/project/{{ project_name }}/cleanup" style="margin-top: 24px;">
            <input type="hidden" name="confirm" value="yes">
            <div style="display: flex; gap: 12px;">
                <button type="submit" class="btn btn-danger">Delete All Project Data</button>
                <a href="/dashboard/project/{{ project_name }}" class="btn btn-outline">Cancel</a>
            </div>
        </form>
        {% endif %}
    </div>
</body>
</html>
"""

LIVE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Status - AI Platform</title>
    <style>""" + BASE_CSS + """
        .live-container { display: grid; grid-template-columns: 300px 1fr; gap: 20px; height: calc(100vh - 120px); }
        .job-list { background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; }
        .job-list-header { padding: 16px; background: #1a1a2e; color: white; font-weight: 600; }
        .job-list-body { flex: 1; overflow-y: auto; }
        .job-item { padding: 12px 16px; border-bottom: 1px solid #e5e7eb; cursor: pointer; transition: background 0.2s; }
        .job-item:hover { background: #f3f4f6; }
        .job-item.active { background: #e0f2fe; border-left: 3px solid #0ea5e9; }
        .job-item.running { border-left: 3px solid #22c55e; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .job-item .job-name { font-weight: 500; color: #1a1a2e; font-size: 14px; }
        .job-item .job-meta { font-size: 12px; color: #6b7280; margin-top: 4px; }
        .job-item .job-status { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: 500; margin-top: 6px; }
        .job-item .job-status.running { background: #dcfce7; color: #166534; }
        .job-item .job-status.completed { background: #dbeafe; color: #1e40af; }
        .job-item .job-status.failed { background: #fee2e2; color: #991b1b; }
        .job-item .job-status.queued { background: #fef3c7; color: #92400e; }
        .log-panel { background: #1a1a2e; border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }
        .log-header { padding: 16px; background: #16213e; color: white; display: flex; justify-content: space-between; align-items: center; }
        .log-header h3 { font-size: 16px; }
        .log-controls { display: flex; gap: 8px; }
        .log-controls button { padding: 6px 12px; border: 1px solid rgba(255,255,255,0.2); background: transparent; color: white; border-radius: 6px; cursor: pointer; font-size: 12px; }
        .log-controls button:hover { background: rgba(255,255,255,0.1); }
        .log-controls button.active { background: #0ea5e9; border-color: #0ea5e9; }
        .log-body { flex: 1; overflow-y: auto; padding: 16px; font-family: 'Monaco', 'Menlo', monospace; font-size: 13px; line-height: 1.6; color: #e2e8f0; }
        .log-line { margin-bottom: 2px; white-space: pre-wrap; word-break: break-all; }
        .log-line.info { color: #60a5fa; }
        .log-line.error { color: #f87171; }
        .log-line.warn { color: #fbbf24; }
        .log-line.success { color: #4ade80; }
        .log-line.timestamp { color: #94a3b8; }
        .live-indicator { display: flex; align-items: center; gap: 8px; }
        .live-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; animation: blink 1s infinite; }
        @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
        .no-jobs { padding: 40px; text-align: center; color: #6b7280; }
        .refresh-timer { font-size: 11px; color: rgba(255,255,255,0.6); }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>AI Development Platform</h1>
        <div class="nav-links">
            <a href="/dashboard/main">Overview</a>
            <a href="/dashboard/jobs">Jobs</a>
            <a href="/dashboard/live" class="active">Live</a>
            <a href="/dashboard/lifecycles">Lifecycles</a>
            <a href="/dashboard/audit">Audit</a>
            <a href="/dashboard/runtime">Runtime</a>
        </div>
        <div class="user-info">
            <span>{{ user.name }}</span>
            <a href="/dashboard/logout" class="logout">Logout</a>
        </div>
    </nav>

    <div class="container">
        <div class="live-container">
            <div class="job-list">
                <div class="job-list-header">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span>Active & Recent Jobs</span>
                        <span class="refresh-timer" id="refresh-timer">Refreshing...</span>
                    </div>
                </div>
                <div class="job-list-body" id="job-list">
                    {% if jobs %}
                        {% for job in jobs %}
                        <div class="job-item {% if job.state == 'running' %}running{% endif %} {% if loop.first %}active{% endif %}"
                             data-job-id="{{ job.job_id }}" onclick="selectJob('{{ job.job_id }}')">
                            <div class="job-name">{{ job.project_name }}</div>
                            <div class="job-meta">{{ job.task_type }} - {{ job.job_id[:8] }}...</div>
                            <span class="job-status {{ job.state }}">{{ job.state | upper }}</span>
                        </div>
                        {% endfor %}
                    {% else %}
                        <div class="no-jobs">No jobs found</div>
                    {% endif %}
                </div>
            </div>

            <div class="log-panel">
                <div class="log-header">
                    <h3>
                        <span class="live-indicator" id="live-indicator" style="display: none;">
                            <span class="live-dot"></span>
                            <span>LIVE</span>
                        </span>
                        <span id="log-title">Select a job to view logs</span>
                    </h3>
                    <div class="log-controls">
                        <button id="btn-auto-scroll" class="active" onclick="toggleAutoScroll()">Auto-scroll</button>
                        <button onclick="clearLogs()">Clear</button>
                        <button onclick="downloadLogs()">Download</button>
                    </div>
                </div>
                <div class="log-body" id="log-body">
                    <div class="log-line info">Select a job from the left panel to view its logs.</div>
                    <div class="log-line info">Running jobs will stream logs in real-time.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentJobId = null;
        let autoScroll = true;
        let refreshInterval = null;
        let logRefreshInterval = null;

        function selectJob(jobId) {
            currentJobId = jobId;

            // Update UI
            document.querySelectorAll('.job-item').forEach(el => el.classList.remove('active'));
            document.querySelector(`[data-job-id="${jobId}"]`)?.classList.add('active');

            // Clear and load logs
            document.getElementById('log-body').innerHTML = '<div class="log-line info">Loading logs...</div>';
            document.getElementById('log-title').textContent = 'Job: ' + jobId.substring(0, 12) + '...';

            loadLogs(jobId);

            // Check if job is running
            const jobItem = document.querySelector(`[data-job-id="${jobId}"]`);
            const isRunning = jobItem?.classList.contains('running');
            document.getElementById('live-indicator').style.display = isRunning ? 'flex' : 'none';

            // Set up log refresh for running jobs
            if (logRefreshInterval) clearInterval(logRefreshInterval);
            if (isRunning) {
                logRefreshInterval = setInterval(() => loadLogs(jobId), 2000);
            }
        }

        async function loadLogs(jobId) {
            try {
                const response = await fetch(`/dashboard/api/job/${jobId}/logs`);
                const data = await response.json();

                const logBody = document.getElementById('log-body');
                let html = '';

                if (data.job_log) {
                    html += '<div class="log-line info">═══ JOB LOG ═══</div>';
                    data.job_log.split('\\n').forEach(line => {
                        const cls = line.includes('[ERROR]') ? 'error' :
                                   line.includes('[WARN]') ? 'warn' :
                                   line.includes('[INFO]') ? 'info' : '';
                        html += `<div class="log-line ${cls}">${escapeHtml(line)}</div>`;
                    });
                }

                if (data.stdout) {
                    html += '<div class="log-line info" style="margin-top: 16px;">═══ CLAUDE OUTPUT ═══</div>';
                    try {
                        const jsonOutput = JSON.parse(data.stdout);
                        if (jsonOutput.result) {
                            html += `<div class="log-line success">${escapeHtml(jsonOutput.result.substring(0, 2000))}</div>`;
                        }
                    } catch (e) {
                        html += `<div class="log-line">${escapeHtml(data.stdout.substring(0, 5000))}</div>`;
                    }
                }

                if (data.stderr && data.stderr.trim()) {
                    html += '<div class="log-line error" style="margin-top: 16px;">═══ ERRORS ═══</div>';
                    html += `<div class="log-line error">${escapeHtml(data.stderr)}</div>`;
                }

                if (!html) {
                    html = '<div class="log-line info">No logs available yet.</div>';
                }

                logBody.innerHTML = html;

                if (autoScroll) {
                    logBody.scrollTop = logBody.scrollHeight;
                }
            } catch (err) {
                document.getElementById('log-body').innerHTML = `<div class="log-line error">Error loading logs: ${err.message}</div>`;
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function toggleAutoScroll() {
            autoScroll = !autoScroll;
            document.getElementById('btn-auto-scroll').classList.toggle('active', autoScroll);
        }

        function clearLogs() {
            document.getElementById('log-body').innerHTML = '<div class="log-line info">Logs cleared.</div>';
        }

        function downloadLogs() {
            const logs = document.getElementById('log-body').innerText;
            const blob = new Blob([logs], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `job-${currentJobId || 'logs'}.txt`;
            a.click();
        }

        async function refreshJobList() {
            try {
                const response = await fetch('/dashboard/api/jobs/recent');
                const data = await response.json();

                const jobList = document.getElementById('job-list');
                if (data.jobs && data.jobs.length > 0) {
                    let html = '';
                    data.jobs.forEach((job, index) => {
                        const isActive = job.job_id === currentJobId;
                        const isRunning = job.state === 'running';
                        html += `
                            <div class="job-item ${isRunning ? 'running' : ''} ${isActive ? 'active' : ''}"
                                 data-job-id="${job.job_id}" onclick="selectJob('${job.job_id}')">
                                <div class="job-name">${job.project_name}</div>
                                <div class="job-meta">${job.task_type} - ${job.job_id.substring(0, 8)}...</div>
                                <span class="job-status ${job.state}">${job.state.toUpperCase()}</span>
                            </div>
                        `;
                    });
                    jobList.innerHTML = html;

                    // Update live indicator
                    const currentJob = data.jobs.find(j => j.job_id === currentJobId);
                    if (currentJob) {
                        document.getElementById('live-indicator').style.display = currentJob.state === 'running' ? 'flex' : 'none';
                    }
                }

                document.getElementById('refresh-timer').textContent = 'Updated just now';
            } catch (err) {
                console.error('Failed to refresh job list:', err);
            }
        }

        // Initial setup
        {% if jobs and jobs[0] %}
        selectJob('{{ jobs[0].job_id }}');
        {% endif %}

        // Refresh job list every 5 seconds
        refreshInterval = setInterval(refreshJobList, 5000);
        refreshJobList();
    </script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@router.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Dashboard home - redirects to login or main dashboard."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)
    return RedirectResponse(url="/dashboard/main", status_code=302)


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = None):
    """Login page."""
    user = get_current_user(request)
    if user:
        return RedirectResponse(url="/dashboard/main", status_code=302)

    template = Template(LOGIN_HTML)
    return HTMLResponse(template.render(error=error))


@router.post("/login")
async def login_submit(
    request: Request,
    email: str = Form(...),
    password: str = Form(...)
):
    """Handle login form submission."""
    # Verify credentials
    if email not in DASHBOARD_USERS:
        return RedirectResponse(
            url="/dashboard/login?error=Invalid+email+or+password",
            status_code=302
        )

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    if password_hash != DASHBOARD_USERS[email]["password_hash"]:
        return RedirectResponse(
            url="/dashboard/login?error=Invalid+email+or+password",
            status_code=302
        )

    # Create session
    session_id = create_session(email)

    # Redirect to dashboard with session cookie
    response = RedirectResponse(url="/dashboard/main", status_code=302)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        max_age=SESSION_EXPIRY_HOURS * 3600,
        samesite="lax"
    )
    return response


@router.get("/logout")
async def logout(request: Request):
    """Logout and clear session."""
    session_id = request.cookies.get("session_id")
    if session_id:
        delete_session(session_id)

    response = RedirectResponse(url="/dashboard/login", status_code=302)
    response.delete_cookie("session_id")
    return response


@router.get("/main", response_class=HTMLResponse)
async def main_dashboard(request: Request):
    """Main dashboard page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    projects = get_all_projects()
    stats = get_system_stats()

    template = Template(DASHBOARD_HTML)
    return HTMLResponse(template.render(
        user=user,
        projects=projects,
        stats=stats
    ))


@router.get("/jobs", response_class=HTMLResponse)
async def jobs_page(request: Request):
    """Jobs management page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    jobs = get_all_jobs()
    running_jobs = get_running_jobs()
    queue_jobs = get_job_queue()

    today = datetime.utcnow().date().isoformat()
    completed_today = len([j for j in jobs if j.get("state") == "completed" and j.get("completed_at", "").startswith(today)])
    failed_today = len([j for j in jobs if j.get("state") == "failed" and j.get("completed_at", "").startswith(today)])

    template = Template(JOBS_HTML)
    return HTMLResponse(template.render(
        user=user,
        jobs=jobs,
        running_jobs=running_jobs,
        queue_jobs=queue_jobs,
        running_count=len(running_jobs),
        queued_count=len(queue_jobs),
        completed_today=completed_today,
        failed_today=failed_today
    ))


@router.get("/live", response_class=HTMLResponse)
async def live_page(request: Request):
    """Live status and log streaming page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    # Get recent jobs (running first, then recent completed/failed)
    all_jobs = get_all_jobs()
    running_jobs = [j for j in all_jobs if j.get("state") == "running"]
    other_jobs = [j for j in all_jobs if j.get("state") != "running"]
    # Sort by created_at descending
    other_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    jobs = running_jobs + other_jobs[:20]  # Running + last 20

    template = Template(LIVE_HTML)
    return HTMLResponse(template.render(
        user=user,
        jobs=jobs
    ))


@router.get("/lifecycles", response_class=HTMLResponse)
async def lifecycles_page(request: Request):
    """Lifecycles management page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    lifecycles = get_all_lifecycles()
    pending_feedback = len([l for l in lifecycles if l.get("feedback_pending")])
    project_mode_count = len([l for l in lifecycles if l.get("mode") == "project"])
    change_mode_count = len([l for l in lifecycles if l.get("mode") == "change"])

    template = Template(LIFECYCLES_HTML)
    return HTMLResponse(template.render(
        user=user,
        lifecycles=lifecycles,
        pending_feedback=pending_feedback,
        project_mode_count=project_mode_count,
        change_mode_count=change_mode_count
    ))


@router.get("/lifecycle/{lifecycle_id}", response_class=HTMLResponse)
async def lifecycle_detail_page(request: Request, lifecycle_id: str):
    """Lifecycle detail page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    lifecycle = get_lifecycle_details(lifecycle_id)
    if not lifecycle:
        raise HTTPException(status_code=404, detail="Lifecycle not found")

    template = Template(LIFECYCLE_DETAIL_HTML)
    return HTMLResponse(template.render(
        user=user,
        lifecycle=lifecycle,
        lifecycle_id=lifecycle_id
    ))


@router.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    """Audit trail page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    events = get_audit_events(100)

    template = Template(AUDIT_HTML)
    return HTMLResponse(template.render(
        user=user,
        events=events
    ))


@router.get("/runtime", response_class=HTMLResponse)
async def runtime_page(request: Request):
    """Runtime monitoring page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    signals = get_runtime_signals()
    incidents = get_incidents()
    recommendations = get_recommendations()

    template = Template(RUNTIME_HTML)
    return HTMLResponse(template.render(
        user=user,
        signals=signals,
        incidents=incidents,
        recommendations=recommendations
    ))


@router.get("/project/{project_name}", response_class=HTMLResponse)
async def project_detail_page(request: Request, project_name: str):
    """Project detail page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    project = get_project_details(project_name)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    template = Template(PROJECT_DETAIL_HTML)
    return HTMLResponse(template.render(user=user, project=project))


@router.get("/project/{project_name}/logs", response_class=HTMLResponse)
async def project_logs_page(request: Request, project_name: str):
    """Project logs page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    logs = get_project_logs(project_name)

    template = Template(PROJECT_LOGS_HTML)
    return HTMLResponse(template.render(
        user=user,
        project_name=project_name,
        logs=logs
    ))


@router.get("/project/{project_name}/cleanup", response_class=HTMLResponse)
async def project_cleanup_page(request: Request, project_name: str):
    """Project cleanup confirmation page."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    project = get_project_details(project_name)
    cleanup_info = project["cleanup_info"] if project else {}

    template = Template(CLEANUP_HTML)
    return HTMLResponse(template.render(
        user=user,
        project_name=project_name,
        cleanup_info=cleanup_info,
        result=None
    ))


@router.post("/project/{project_name}/cleanup", response_class=HTMLResponse)
async def project_cleanup_submit(
    request: Request,
    project_name: str,
    confirm: str = Form(...)
):
    """Handle project cleanup."""
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/dashboard/login", status_code=302)

    if confirm != "yes":
        return RedirectResponse(
            url=f"/dashboard/project/{project_name}/cleanup",
            status_code=302
        )

    # Perform cleanup
    result = cleanup_project(project_name)

    logger.info(f"Project cleanup performed by {user['name']}: {project_name}")
    logger.info(f"Cleanup result: {result}")

    template = Template(CLEANUP_HTML)
    return HTMLResponse(template.render(
        user=user,
        project_name=project_name,
        cleanup_info={},
        result=result
    ))


# -----------------------------------------------------------------------------
# API Endpoints (JSON)
# -----------------------------------------------------------------------------
@router.get("/api/projects")
async def api_list_projects(user: Dict = Depends(require_auth)):
    """API: List all projects."""
    return {"projects": get_all_projects()}


@router.get("/api/project/{project_name}")
async def api_project_detail(project_name: str, user: Dict = Depends(require_auth)):
    """API: Get project details."""
    project = get_project_details(project_name)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.get("/api/project/{project_name}/logs")
async def api_project_logs(
    project_name: str,
    limit: int = 100,
    user: Dict = Depends(require_auth)
):
    """API: Get project logs."""
    return {"logs": get_project_logs(project_name, limit)}


@router.delete("/api/project/{project_name}")
async def api_cleanup_project(project_name: str, user: Dict = Depends(require_auth)):
    """API: Cleanup/delete a project."""
    result = cleanup_project(project_name)
    logger.info(f"API cleanup by {user['name']}: {project_name}")
    return result


@router.get("/api/stats")
async def api_system_stats(user: Dict = Depends(require_auth)):
    """API: Get system statistics."""
    return get_system_stats()


@router.get("/api/jobs")
async def api_list_jobs(user: Dict = Depends(require_auth)):
    """API: List all jobs."""
    return {"jobs": get_all_jobs()}


@router.get("/api/jobs/queue")
async def api_job_queue(user: Dict = Depends(require_auth)):
    """API: Get job queue."""
    return {"queue": get_job_queue()}


@router.get("/api/jobs/running")
async def api_running_jobs(user: Dict = Depends(require_auth)):
    """API: Get running jobs."""
    return {"running": get_running_jobs()}


@router.get("/api/lifecycles")
async def api_list_lifecycles(user: Dict = Depends(require_auth)):
    """API: List all lifecycles."""
    return {"lifecycles": get_all_lifecycles()}


@router.get("/api/lifecycle/{lifecycle_id}")
async def api_lifecycle_detail(lifecycle_id: str, user: Dict = Depends(require_auth)):
    """API: Get lifecycle details."""
    lifecycle = get_lifecycle_details(lifecycle_id)
    if not lifecycle:
        raise HTTPException(status_code=404, detail="Lifecycle not found")
    return lifecycle


@router.get("/api/audit")
async def api_audit_events(limit: int = 100, user: Dict = Depends(require_auth)):
    """API: Get audit events."""
    return {"events": get_audit_events(limit)}


@router.get("/api/runtime/signals")
async def api_runtime_signals(user: Dict = Depends(require_auth)):
    """API: Get runtime signals."""
    return {"signals": get_runtime_signals()}


@router.get("/api/runtime/incidents")
async def api_incidents(limit: int = 50, user: Dict = Depends(require_auth)):
    """API: Get incidents."""
    return {"incidents": get_incidents(limit)}


@router.get("/api/runtime/recommendations")
async def api_recommendations(limit: int = 50, user: Dict = Depends(require_auth)):
    """API: Get recommendations."""
    return {"recommendations": get_recommendations(limit)}


@router.get("/api/jobs/recent")
async def api_recent_jobs(limit: int = 25, user: Dict = Depends(require_auth)):
    """API: Get recent jobs for live page (running first, then recent)."""
    all_jobs = get_all_jobs()
    running_jobs = [j for j in all_jobs if j.get("state") == "running"]
    other_jobs = [j for j in all_jobs if j.get("state") != "running"]
    other_jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    jobs = running_jobs + other_jobs[:limit]
    return {"jobs": jobs}


@router.get("/api/job/{job_id}/logs")
async def api_job_logs(job_id: str, user: Dict = Depends(require_auth)):
    """API: Get job logs for live streaming."""
    # Find job workspace
    job_dir = JOBS_DIR / f"job-{job_id}"
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    result = {
        "job_id": job_id,
        "job_log": "",
        "stdout": "",
        "stderr": "",
        "execution_summary": None
    }

    # Read job log from /var/log/claude-jobs
    log_file = Path("/var/log/claude-jobs") / f"job-{job_id}.log"
    if log_file.exists():
        try:
            result["job_log"] = log_file.read_text()[-50000:]  # Last 50KB
        except Exception as e:
            result["job_log"] = f"Error reading log: {e}"

    # Read stdout
    stdout_file = job_dir / "logs" / "stdout.log"
    if stdout_file.exists():
        try:
            result["stdout"] = stdout_file.read_text()[-100000:]  # Last 100KB
        except Exception as e:
            result["stdout"] = f"Error reading stdout: {e}"

    # Read stderr
    stderr_file = job_dir / "logs" / "stderr.log"
    if stderr_file.exists():
        try:
            result["stderr"] = stderr_file.read_text()[-20000:]  # Last 20KB
        except Exception as e:
            result["stderr"] = f"Error reading stderr: {e}"

    # Read execution summary
    summary_file = job_dir / "EXECUTION_SUMMARY.yaml"
    if summary_file.exists():
        try:
            result["execution_summary"] = summary_file.read_text()
        except Exception:
            pass

    return result


logger.info("Enhanced web dashboard module loaded with Live Status page")
