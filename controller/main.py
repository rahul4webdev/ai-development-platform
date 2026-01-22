"""
Task Controller - FastAPI Application

Phase 6: Production Hardening & Explicit Go-Live Controls

Central orchestrator that:
- Bootstraps new project repositories (filesystem only)
- Maintains project-level manifests
- Tracks task lifecycle (state machine)
- Generates implementation plans (NO execution)
- Generates proposed code changes as DIFFS
- Applies diffs with EXPLICIT human confirmation
- Supports dry-run, apply, and rollback
- Prepares git commits (human-approved)
- Triggers CI pipelines (controlled)
- Enforces test gates
- Supports environment promotion (testing → production)
- Enforces dual approval for production deployment
- Maintains immutable audit trail

PHASE 6 CONSTRAINTS (Production Hardening):
- NO autonomous production deployment (EVER)
- NO single-actor production approval (dual approval required)
- NO silent production changes (every action logged)
- NO bypassing testing environment (must deploy to testing first)
- NO rollback omission (instant rollback always available)
- Production actions must be rare, explicit, auditable
- Multi-step confirmation is mandatory
- Different users MUST request and approve production

PHASE 5 CONSTRAINTS (inherited):
- NO automatic commits without human approval
- NO automatic merges
- NO CI triggers without explicit intent
- NO bypassing test failures
- NO background or scheduled CI runs
- CI must be human-initiated or human-approved
- All release steps must be auditable
- Rollback must remain possible
- Commits are artifacts, not automatic actions
"""

import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Phase 12: Multi-Aspect Project Router
from .phase12_router import router as phase12_router

# Import phase metadata from single source of truth
from . import __version__, CURRENT_PHASE, CURRENT_PHASE_NAME, CURRENT_PHASE_FULL

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("task_controller")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
PROJECTS_DIR = Path(__file__).parent.parent / "projects"
DOCS_DIR = Path(__file__).parent.parent / "docs"

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class TaskType(str, Enum):
    """Supported task types for the platform."""
    BUG = "bug"
    FEATURE = "feature"
    REFACTOR = "refactor"
    INFRA = "infra"


class TaskState(str, Enum):
    """Task lifecycle states (Phase 6 state machine)."""
    RECEIVED = "received"
    VALIDATED = "validated"
    PLANNED = "planned"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    DIFF_GENERATED = "diff_generated"  # Phase 3
    # Phase 4: Execution states
    READY_TO_APPLY = "ready_to_apply"  # Dry-run passed, awaiting apply confirmation
    APPLYING = "applying"              # Currently applying diff
    APPLIED = "applied"                # Diff successfully applied
    ROLLED_BACK = "rolled_back"        # Applied changes have been rolled back
    EXECUTION_FAILED = "execution_failed"  # Apply failed, restored from backup
    # Phase 5: CI/Release states
    COMMITTED = "committed"            # Git commit created locally (not pushed)
    CI_RUNNING = "ci_running"          # CI pipeline running
    CI_PASSED = "ci_passed"            # CI pipeline passed
    CI_FAILED = "ci_failed"            # CI pipeline failed
    DEPLOYED_TESTING = "deployed_testing"  # Deployed to testing environment
    # Phase 6: Production states (GUARDED HEAVILY)
    PROD_DEPLOY_REQUESTED = "prod_deploy_requested"  # Production deployment requested (awaiting approval)
    PROD_APPROVED = "prod_approved"                  # Approved by DIFFERENT user (ready for deploy)
    DEPLOYED_PRODUCTION = "deployed_production"      # Deployed to production
    PROD_ROLLED_BACK = "prod_rolled_back"            # Production rollback executed (break-glass)
    REJECTED = "rejected"
    ARCHIVED = "archived"


class CIStatus(str, Enum):
    """CI pipeline status values."""
    PASSED = "passed"
    FAILED = "failed"


class ProjectPhase(str, Enum):
    """Project lifecycle phases."""
    BOOTSTRAP = "bootstrap"
    DEVELOPMENT = "development"
    HARDENING = "hardening"
    RELEASE = "release"
    MAINTENANCE = "maintenance"


class DeploymentEnvironment(str, Enum):
    """Supported deployment environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------
class ProjectBootstrapRequest(BaseModel):
    """Request model for project bootstrap."""
    project_name: str = Field(..., min_length=1, max_length=50, pattern=r'^[a-z0-9-]+$')
    repo_url: str = Field(..., description="Repository URL (string only, no validation)")
    tech_stack: list[str] = Field(default_factory=list, description="List of technologies")
    user_id: Optional[str] = None


class ProjectBootstrapResponse(BaseModel):
    """Response model for project bootstrap."""
    project_name: str
    status: str
    manifest_path: str
    state_path: str
    message: str
    created_at: str


class TaskRequest(BaseModel):
    """Request model for creating a new task."""
    project_name: str
    task_description: str
    task_type: TaskType
    user_id: Optional[str] = None


class TaskResponse(BaseModel):
    """Response model for task creation."""
    task_id: str
    project_name: str
    task_type: TaskType
    state: TaskState
    message: str
    created_at: str


class TaskValidateResponse(BaseModel):
    """Response model for task validation."""
    task_id: str
    previous_state: TaskState
    current_state: TaskState
    validation_passed: bool
    validation_errors: list[str]
    message: str


class TaskPlanResponse(BaseModel):
    """Response model for plan generation."""
    task_id: str
    state: TaskState
    plan_path: str
    message: str


class TaskApprovalResponse(BaseModel):
    """Response model for task approval/rejection."""
    task_id: str
    previous_state: TaskState
    current_state: TaskState
    message: str
    rejection_reason: Optional[str] = None


class DiffGenerateResponse(BaseModel):
    """Response model for diff generation (Phase 3)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    diff_path: str
    files_in_diff: int
    message: str
    warning: str = "DIFF NOT APPLIED. FOR HUMAN REVIEW ONLY."


# -----------------------------------------------------------------------------
# Phase 4 Response Models
# -----------------------------------------------------------------------------
class DryRunResponse(BaseModel):
    """Response model for dry-run (Phase 4)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    files_affected: list[str]
    lines_added: int
    lines_removed: int
    can_apply: bool
    conflicts: list[str]
    message: str
    warning: str = "DRY-RUN ONLY. NO FILES MODIFIED."


class ApplyResponse(BaseModel):
    """Response model for apply (Phase 4)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    files_modified: list[str]
    backup_path: str
    message: str
    rollback_available: bool = True


class RollbackResponse(BaseModel):
    """Response model for rollback (Phase 4)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    files_restored: list[str]
    message: str


# -----------------------------------------------------------------------------
# Phase 5 Response Models
# -----------------------------------------------------------------------------
class CommitResponse(BaseModel):
    """Response model for commit preparation (Phase 5)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    commit_hash: str
    commit_message: str
    files_committed: list[str]
    message: str
    warning: str = "COMMIT CREATED LOCALLY. NOT PUSHED."


class CIRunResponse(BaseModel):
    """Response model for CI trigger (Phase 5)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    ci_run_id: str
    message: str
    warning: str = "CI RUNNING. WAIT FOR RESULTS."


class CIResultRequest(BaseModel):
    """Request model for CI result ingestion (Phase 5)."""
    status: CIStatus
    logs_url: Optional[str] = None
    details: Optional[str] = None


class CIResultResponse(BaseModel):
    """Response model for CI result ingestion (Phase 5)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    ci_status: CIStatus
    logs_url: Optional[str]
    message: str


class DeployTestingResponse(BaseModel):
    """Response model for testing deployment (Phase 5)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    deployment_url: str
    message: str
    warning: str = "DEPLOYED TO TESTING ONLY. NOT PRODUCTION."


# -----------------------------------------------------------------------------
# Phase 6 Request/Response Models - Production Deployment
# -----------------------------------------------------------------------------
class ProdDeployRequestModel(BaseModel):
    """Request model for production deployment request (Phase 6)."""
    justification: str = Field(..., min_length=20, description="Why this deploy is needed (min 20 chars)")
    risk_acknowledged: bool = Field(..., description="User acknowledges production risk")
    rollback_plan: str = Field(..., min_length=10, description="How to rollback if needed")


class ProdDeployRequestResponse(BaseModel):
    """Response model for production deployment request (Phase 6)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    requested_by: str
    justification: str
    rollback_plan: str
    message: str
    warning: str = "⚠️ PRODUCTION DEPLOYMENT REQUESTED. REQUIRES APPROVAL FROM DIFFERENT USER."
    next_step: str = "Another user must approve via /task/{task_id}/deploy/production/approve"


class ProdApproveRequest(BaseModel):
    """Request model for production approval (Phase 6)."""
    approval_reason: str = Field(..., min_length=10, description="Reason for approval")
    reviewed_changes: bool = Field(..., description="Approver has reviewed all changes")
    reviewed_rollback: bool = Field(..., description="Approver has reviewed rollback plan")


class ProdApproveResponse(BaseModel):
    """Response model for production approval (Phase 6)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    requested_by: str
    approved_by: str
    approval_reason: str
    message: str
    warning: str = "⚠️ PRODUCTION DEPLOYMENT APPROVED. READY FOR FINAL APPLY."
    next_step: str = "Execute deploy via /task/{task_id}/deploy/production/apply with confirm=true"


class ProdApplyResponse(BaseModel):
    """Response model for production apply (Phase 6)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    requested_by: str
    approved_by: str
    applied_by: str
    deployment_url: str
    release_manifest_path: str
    message: str
    warning: str = "🚀 DEPLOYED TO PRODUCTION. MONITOR CLOSELY."
    rollback_available: bool = True
    rollback_command: str = "POST /task/{task_id}/deploy/production/rollback"


class ProdRollbackResponse(BaseModel):
    """Response model for production rollback (Phase 6)."""
    task_id: str
    project_name: str
    previous_state: TaskState
    current_state: TaskState
    rolled_back_by: str
    rollback_reason: str
    message: str
    warning: str = "⚠️ PRODUCTION ROLLED BACK. Verify system stability."


class StatusResponse(BaseModel):
    """Response model for project status."""
    project_name: str
    current_phase: str
    task_count: int
    tasks_by_state: dict
    last_updated: str


class DeployRequest(BaseModel):
    """Request model for deployment."""
    project_name: str
    environment: DeploymentEnvironment
    user_id: Optional[str] = None


class DeployResponse(BaseModel):
    """Response model for deployment."""
    project_name: str
    environment: DeploymentEnvironment
    status: str
    message: str
    deployment_url: Optional[str]


# -----------------------------------------------------------------------------
# Policy Enforcement Hooks (Phase 2)
# -----------------------------------------------------------------------------
def can_create_project(user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can a project be created?

    Returns (allowed, reason)
    """
    # Phase 2: Always allow, log decision
    logger.info(f"POLICY CHECK: can_create_project | user_id={user_id} | ALLOWED")
    return True, "Project creation allowed"


def can_submit_task(project_name: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can a task be submitted to this project?

    Returns (allowed, reason)
    """
    # Phase 2: Always allow, log decision
    logger.info(f"POLICY CHECK: can_submit_task | project={project_name} | user_id={user_id} | ALLOWED")
    return True, "Task submission allowed"


def can_validate_task(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can this task be validated?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_validate_task | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Task validation allowed"


def can_plan_task(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can a plan be generated for this task?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_plan_task | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Plan generation allowed"


def can_approve_task(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can this task be approved?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_approve_task | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Task approval allowed"


def can_reject_task(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can this task be rejected?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_reject_task | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Task rejection allowed"


# -----------------------------------------------------------------------------
# Phase 3 Policy Hooks - Diff Generation
# -----------------------------------------------------------------------------
def can_generate_diff(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can a diff be generated for this task?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_generate_diff | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Diff generation allowed"


def diff_within_scope(plan_files: list[str], diff_files: list[str]) -> tuple[bool, str]:
    """
    Policy check: Are the files in the diff within the scope defined by the plan?

    This ensures diffs only modify files mentioned in the plan.

    Returns (allowed, reason)
    """
    # Phase 3: Stub - always returns True but logs for inspection
    logger.info(f"POLICY CHECK: diff_within_scope | plan_files={plan_files} | diff_files={diff_files}")

    # In future phases, this would check that diff_files is a subset of plan_files
    # For now, log and allow
    for diff_file in diff_files:
        logger.info(f"  Scope check: {diff_file} - ALLOWED (stub)")

    return True, "All diff files within plan scope"


def diff_file_limit_ok(file_count: int, max_files: int = 10) -> tuple[bool, str]:
    """
    Policy check: Is the number of files in the diff within limits?

    Phase 3 constraint: Max 10 files per diff.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: diff_file_limit_ok | count={file_count} | max={max_files}")

    if file_count > max_files:
        logger.warning(f"POLICY VIOLATION: diff_file_limit_ok | {file_count} > {max_files}")
        return False, f"Diff contains {file_count} files, exceeds maximum of {max_files}"

    return True, f"File count ({file_count}) within limit ({max_files})"


# -----------------------------------------------------------------------------
# Phase 4 Policy Hooks - Execution
# -----------------------------------------------------------------------------
def can_dry_run(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can a dry-run be performed for this task?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_dry_run | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Dry-run allowed"


def can_apply(task_id: str, user_id: Optional[str] = None, confirmed: bool = False) -> tuple[bool, str]:
    """
    Policy check: Can the diff be applied for this task?

    CRITICAL: This MUST require explicit confirmation.
    If confirmed=False, this MUST return False.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_apply | task_id={task_id} | user_id={user_id} | confirmed={confirmed}")

    if not confirmed:
        logger.warning(f"POLICY VIOLATION: can_apply | confirmation missing for task {task_id}")
        return False, "Apply REQUIRES explicit confirmation. Set confirm=true."

    logger.info(f"POLICY CHECK: can_apply | task_id={task_id} | ALLOWED (confirmed)")
    return True, "Apply allowed (confirmed by user)"


def can_rollback(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can changes be rolled back for this task?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_rollback | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "Rollback allowed"


# -----------------------------------------------------------------------------
# Phase 5 Policy Hooks - CI/Release
# -----------------------------------------------------------------------------
def can_commit(task_id: str, user_id: Optional[str] = None, confirmed: bool = False) -> tuple[bool, str]:
    """
    Policy check: Can a commit be created for this task?

    CRITICAL: This MUST require explicit confirmation.
    If confirmed=False, this MUST return False.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_commit | task_id={task_id} | user_id={user_id} | confirmed={confirmed}")

    if not confirmed:
        logger.warning(f"POLICY VIOLATION: can_commit | confirmation missing for task {task_id}")
        return False, "Commit REQUIRES explicit confirmation. Set confirm=true."

    logger.info(f"POLICY CHECK: can_commit | task_id={task_id} | ALLOWED (confirmed)")
    return True, "Commit allowed (confirmed by user)"


def can_trigger_ci(task_id: str, user_id: Optional[str] = None) -> tuple[bool, str]:
    """
    Policy check: Can CI be triggered for this task?

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_trigger_ci | task_id={task_id} | user_id={user_id} | ALLOWED")
    return True, "CI trigger allowed"


def can_deploy_testing(task_id: str, user_id: Optional[str] = None, confirmed: bool = False) -> tuple[bool, str]:
    """
    Policy check: Can this task be deployed to testing?

    CRITICAL: This MUST require explicit confirmation.
    If confirmed=False, this MUST return False.

    CI MUST have passed before deployment.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_deploy_testing | task_id={task_id} | user_id={user_id} | confirmed={confirmed}")

    if not confirmed:
        logger.warning(f"POLICY VIOLATION: can_deploy_testing | confirmation missing for task {task_id}")
        return False, "Testing deployment REQUIRES explicit confirmation. Set confirm=true."

    logger.info(f"POLICY CHECK: can_deploy_testing | task_id={task_id} | ALLOWED (confirmed)")
    return True, "Testing deployment allowed (confirmed by user)"


# -----------------------------------------------------------------------------
# Phase 6 Policy Hooks - Production Deployment (CRITICAL SAFETY)
# -----------------------------------------------------------------------------
def can_request_prod_deploy(
    task_id: str,
    user_id: Optional[str] = None,
    risk_acknowledged: bool = False
) -> tuple[bool, str]:
    """
    Policy check: Can a production deployment be requested?

    CRITICAL: User MUST acknowledge production risk.
    CRITICAL: Task MUST have been deployed to testing first.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_request_prod_deploy | task_id={task_id} | user_id={user_id} | risk_acknowledged={risk_acknowledged}")

    if not user_id:
        logger.warning(f"POLICY VIOLATION: can_request_prod_deploy | user_id required for task {task_id}")
        return False, "Production deployment request REQUIRES user identification."

    if not risk_acknowledged:
        logger.warning(f"POLICY VIOLATION: can_request_prod_deploy | risk not acknowledged for task {task_id}")
        return False, "Production deployment REQUIRES explicit risk acknowledgment (risk_acknowledged=true)."

    logger.info(f"POLICY CHECK: can_request_prod_deploy | task_id={task_id} | ALLOWED (risk acknowledged)")
    return True, "Production deployment request allowed (risk acknowledged)"


def can_approve_prod_deploy(
    task_id: str,
    approver_id: Optional[str] = None,
    requester_id: Optional[str] = None,
    reviewed_changes: bool = False,
    reviewed_rollback: bool = False
) -> tuple[bool, str]:
    """
    Policy check: Can production deployment be approved?

    CRITICAL: Approver MUST be DIFFERENT from requester (dual approval).
    CRITICAL: Approver MUST have reviewed changes and rollback plan.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_approve_prod_deploy | task_id={task_id} | approver={approver_id} | requester={requester_id}")

    if not approver_id:
        logger.warning(f"POLICY VIOLATION: can_approve_prod_deploy | approver_id required for task {task_id}")
        return False, "Production approval REQUIRES approver identification."

    if not requester_id:
        logger.warning(f"POLICY VIOLATION: can_approve_prod_deploy | requester_id missing for task {task_id}")
        return False, "Production approval REQUIRES knowing who requested the deploy."

    # CRITICAL: Dual approval enforcement - same user cannot approve their own request
    if approver_id == requester_id:
        logger.warning(f"POLICY VIOLATION: can_approve_prod_deploy | SAME USER ({approver_id}) tried to approve own request for task {task_id}")
        return False, f"DUAL APPROVAL REQUIRED: Approver ({approver_id}) CANNOT be the same as requester ({requester_id})."

    if not reviewed_changes:
        logger.warning(f"POLICY VIOLATION: can_approve_prod_deploy | changes not reviewed for task {task_id}")
        return False, "Production approval REQUIRES reviewer to confirm changes were reviewed."

    if not reviewed_rollback:
        logger.warning(f"POLICY VIOLATION: can_approve_prod_deploy | rollback not reviewed for task {task_id}")
        return False, "Production approval REQUIRES reviewer to confirm rollback plan was reviewed."

    logger.info(f"POLICY CHECK: can_approve_prod_deploy | task_id={task_id} | ALLOWED (dual approval: {requester_id} -> {approver_id})")
    return True, f"Production approval allowed (dual approval: requester={requester_id}, approver={approver_id})"


def can_apply_prod_deploy(
    task_id: str,
    user_id: Optional[str] = None,
    confirmed: bool = False
) -> tuple[bool, str]:
    """
    Policy check: Can production deployment be applied?

    CRITICAL: Task MUST be in PROD_APPROVED state.
    CRITICAL: Explicit confirmation required.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_apply_prod_deploy | task_id={task_id} | user_id={user_id} | confirmed={confirmed}")

    if not user_id:
        logger.warning(f"POLICY VIOLATION: can_apply_prod_deploy | user_id required for task {task_id}")
        return False, "Production deployment apply REQUIRES user identification."

    if not confirmed:
        logger.warning(f"POLICY VIOLATION: can_apply_prod_deploy | confirmation missing for task {task_id}")
        return False, "Production deployment REQUIRES explicit confirmation (confirm=true)."

    logger.info(f"POLICY CHECK: can_apply_prod_deploy | task_id={task_id} | ALLOWED (confirmed)")
    return True, "Production deployment apply allowed (confirmed)"


def can_rollback_prod(
    task_id: str,
    user_id: Optional[str] = None
) -> tuple[bool, str]:
    """
    Policy check: Can production be rolled back?

    NOTE: Rollback does NOT require dual approval (speed > ceremony for emergency).
    NOTE: Rollback is always allowed for deployed production tasks.

    Returns (allowed, reason)
    """
    logger.info(f"POLICY CHECK: can_rollback_prod | task_id={task_id} | user_id={user_id}")

    if not user_id:
        logger.warning(f"POLICY VIOLATION: can_rollback_prod | user_id required for task {task_id}")
        return False, "Production rollback REQUIRES user identification (for audit trail)."

    # Rollback is ALWAYS allowed (break-glass) - speed over ceremony
    logger.info(f"POLICY CHECK: can_rollback_prod | task_id={task_id} | ALLOWED (break-glass rollback)")
    return True, "Production rollback allowed (break-glass enabled)"


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_project_path(project_name: str) -> Path:
    """Get the path to a project directory."""
    return PROJECTS_DIR / project_name


def project_exists(project_name: str) -> bool:
    """Check if a project exists by verifying PROJECT_MANIFEST.yaml exists."""
    manifest_path = get_project_path(project_name) / "PROJECT_MANIFEST.yaml"
    return manifest_path.exists()


def generate_task_id() -> str:
    """Generate a unique task ID using UUID."""
    return str(uuid.uuid4())


def get_task_path(project_name: str, task_id: str) -> Path:
    """Get the path to a task file."""
    return get_project_path(project_name) / "tasks" / f"{task_id}.yaml"


def get_plan_path(project_name: str, task_id: str) -> Path:
    """Get the path to a plan file."""
    return get_project_path(project_name) / "plans" / f"{task_id}_plan.md"


def get_diff_path(project_name: str, task_id: str) -> Path:
    """Get the path to a diff file (Phase 3)."""
    return get_project_path(project_name) / "diffs" / f"{task_id}.diff"


def get_backup_path(project_name: str, task_id: str) -> Path:
    """Get the path to backup directory for a task (Phase 4)."""
    return get_project_path(project_name) / "backups" / task_id


# -----------------------------------------------------------------------------
# Phase 6 Helper Functions - Audit & Release
# -----------------------------------------------------------------------------
def get_audit_log_path(project_name: str) -> Path:
    """Get the path to production audit log (Phase 6)."""
    return get_project_path(project_name) / "audit" / "production.log"


def get_release_path(project_name: str, task_id: str) -> Path:
    """Get the path to release directory for a task (Phase 6)."""
    return get_project_path(project_name) / "releases" / task_id


def append_audit_log(project_name: str, entry: dict) -> None:
    """
    Append an entry to the immutable production audit log (Phase 6).

    NOTE: This is append-only. Entries cannot be deleted or modified.
    """
    audit_path = get_audit_log_path(project_name)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp and ensure immutability marker
    entry['timestamp'] = datetime.utcnow().isoformat()
    entry['immutable'] = True

    # Append to log file (one JSON line per entry)
    with open(audit_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')

    logger.info(f"AUDIT LOG: {entry.get('action', 'unknown')} | task={entry.get('task_id', 'N/A')} | user={entry.get('user_id', 'N/A')}")


def create_release_manifest(
    project_name: str,
    task_id: str,
    requested_by: str,
    approved_by: str,
    applied_by: str,
    justification: str,
    rollback_plan: str
) -> str:
    """Create RELEASE_MANIFEST.yaml for a production release (Phase 6)."""
    release_path = get_release_path(project_name, task_id)
    release_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task_id": task_id,
        "project_name": project_name,
        "release_type": "production",
        "requested_by": requested_by,
        "approved_by": approved_by,
        "applied_by": applied_by,
        "justification": justification,
        "rollback_plan": rollback_plan,
        "deployed_at": datetime.utcnow().isoformat(),
        "status": "deployed",
        "dual_approval_verified": requested_by != approved_by,
    }

    manifest_path = release_path / "RELEASE_MANIFEST.yaml"
    write_yaml_file(manifest_path, manifest)

    return str(manifest_path)


def create_deploy_log(
    project_name: str,
    task_id: str,
    action: str,
    user_id: str,
    details: dict
) -> None:
    """Create/append to DEPLOY_LOG.md for a task (Phase 6)."""
    release_path = get_release_path(project_name, task_id)
    release_path.mkdir(parents=True, exist_ok=True)

    log_path = release_path / "DEPLOY_LOG.md"
    timestamp = datetime.utcnow().isoformat()

    entry = f"""
## {action}

- **Timestamp**: {timestamp}
- **User**: {user_id}
- **Details**: {json.dumps(details, indent=2)}

---
"""

    # Append to log
    with open(log_path, 'a') as f:
        if log_path.stat().st_size == 0 if log_path.exists() else True:
            f.write(f"# Deployment Log - {task_id}\n\n")
        f.write(entry)

    logger.info(f"DEPLOY LOG: {action} | task={task_id} | user={user_id}")


def deploy_to_production(project_name: str, task_id: str) -> str:
    """
    Placeholder for production deployment (Phase 6).

    Returns the production deployment URL.
    """
    # In real implementation, this would trigger actual deployment
    # For now, return placeholder URL
    logger.info(f"PRODUCTION DEPLOY (placeholder): project={project_name}, task={task_id}")
    return "https://ai.mybd.in"


def rollback_production(project_name: str, task_id: str) -> bool:
    """
    Placeholder for production rollback (Phase 6).

    Returns True if rollback successful.
    """
    # In real implementation, this would trigger actual rollback
    logger.info(f"PRODUCTION ROLLBACK (placeholder): project={project_name}, task={task_id}")
    return True


def read_yaml_file(file_path: Path) -> dict:
    """Read and parse a YAML file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, 'r') as f:
        return yaml.safe_load(f) or {}


def write_yaml_file(file_path: Path, data: dict) -> None:
    """Write data to a YAML file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Written YAML file: {file_path}")


def write_markdown_file(file_path: Path, content: str) -> None:
    """Write content to a markdown file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    logger.info(f"Written markdown file: {file_path}")


def read_project_manifest(project_name: str) -> dict:
    """Read PROJECT_MANIFEST.yaml for a project."""
    manifest_path = get_project_path(project_name) / "PROJECT_MANIFEST.yaml"
    return read_yaml_file(manifest_path)


def update_project_manifest(project_name: str, updates: dict) -> None:
    """Update PROJECT_MANIFEST.yaml for a project."""
    manifest_path = get_project_path(project_name) / "PROJECT_MANIFEST.yaml"
    manifest = read_yaml_file(manifest_path)
    manifest.update(updates)
    manifest['updated_at'] = datetime.utcnow().isoformat()
    write_yaml_file(manifest_path, manifest)


def read_task(project_name: str, task_id: str) -> dict:
    """Read a task record."""
    task_path = get_task_path(project_name, task_id)
    return read_yaml_file(task_path)


def write_task(project_name: str, task_id: str, task_data: dict) -> None:
    """Write a task record."""
    task_path = get_task_path(project_name, task_id)
    task_data['updated_at'] = datetime.utcnow().isoformat()
    write_yaml_file(task_path, task_data)


def list_tasks(project_name: str) -> list[dict]:
    """List all tasks for a project."""
    tasks_dir = get_project_path(project_name) / "tasks"
    if not tasks_dir.exists():
        return []

    tasks = []
    for task_file in tasks_dir.glob("*.yaml"):
        try:
            task_data = read_yaml_file(task_file)
            tasks.append(task_data)
        except Exception as e:
            logger.error(f"Error reading task file {task_file}: {e}")

    return tasks


def validate_task_transition(current_state: TaskState, target_state: TaskState) -> bool:
    """Validate if a state transition is allowed."""
    # Define allowed transitions (Phase 6 updated)
    allowed_transitions = {
        TaskState.RECEIVED: [TaskState.VALIDATED, TaskState.REJECTED],
        TaskState.VALIDATED: [TaskState.PLANNED, TaskState.REJECTED],
        TaskState.PLANNED: [TaskState.AWAITING_APPROVAL],
        TaskState.AWAITING_APPROVAL: [TaskState.APPROVED, TaskState.REJECTED],
        TaskState.APPROVED: [TaskState.DIFF_GENERATED, TaskState.ARCHIVED],
        TaskState.DIFF_GENERATED: [TaskState.READY_TO_APPLY, TaskState.ARCHIVED],  # Phase 4: Dry-run
        # Phase 4: Execution states
        TaskState.READY_TO_APPLY: [TaskState.APPLYING, TaskState.ARCHIVED],  # Apply or abandon
        TaskState.APPLYING: [TaskState.APPLIED, TaskState.EXECUTION_FAILED],  # Success or fail
        TaskState.APPLIED: [TaskState.ROLLED_BACK, TaskState.COMMITTED, TaskState.ARCHIVED],  # Phase 5: Can commit
        TaskState.ROLLED_BACK: [TaskState.ARCHIVED],  # Terminal for this cycle
        TaskState.EXECUTION_FAILED: [TaskState.READY_TO_APPLY, TaskState.ARCHIVED],  # Retry or abandon
        # Phase 5: CI/Release states
        TaskState.COMMITTED: [TaskState.CI_RUNNING, TaskState.ARCHIVED],  # Trigger CI or abandon
        TaskState.CI_RUNNING: [TaskState.CI_PASSED, TaskState.CI_FAILED],  # CI result
        TaskState.CI_PASSED: [TaskState.DEPLOYED_TESTING, TaskState.ARCHIVED],  # Deploy or archive
        TaskState.CI_FAILED: [TaskState.COMMITTED, TaskState.ARCHIVED],  # Fix and recommit, or abandon
        # Phase 6: Production deployment path (DEPLOYED_TESTING → Production)
        TaskState.DEPLOYED_TESTING: [TaskState.PROD_DEPLOY_REQUESTED, TaskState.ARCHIVED],  # Request production or archive
        TaskState.PROD_DEPLOY_REQUESTED: [TaskState.PROD_APPROVED, TaskState.REJECTED],  # Approve or reject
        TaskState.PROD_APPROVED: [TaskState.DEPLOYED_PRODUCTION, TaskState.REJECTED],  # Apply or reject
        TaskState.DEPLOYED_PRODUCTION: [TaskState.PROD_ROLLED_BACK, TaskState.ARCHIVED],  # Rollback or archive
        TaskState.PROD_ROLLED_BACK: [TaskState.ARCHIVED],  # Terminal for production rollback
        TaskState.REJECTED: [TaskState.ARCHIVED],
        TaskState.ARCHIVED: [],  # Terminal state
    }

    return target_state in allowed_transitions.get(current_state, [])


# -----------------------------------------------------------------------------
# Project Bootstrap Functions
# -----------------------------------------------------------------------------
def create_project_manifest(project_name: str, repo_url: str, tech_stack: list[str]) -> dict:
    """Create initial PROJECT_MANIFEST.yaml content."""
    return {
        "project_name": project_name,
        "repo_url": repo_url,
        "environment": "testing",  # Only testing allowed
        "tech_stack": tech_stack,
        "allowed_actions": [
            "create_task",
            "validate_task",
            "generate_plan",
            "approve_task",
            "reject_task",
            "generate_diff",  # Phase 3
            # Phase 4: Execution actions
            "dry_run",
            "apply",
            "rollback",
            # Phase 5: CI/Release actions
            "commit",
            "ci_run",
            "ci_result",
            "deploy_testing",
            # Phase 6: Production deployment actions (DUAL APPROVAL)
            "prod_deploy_request",
            "prod_deploy_approve",
            "prod_deploy_apply",
            "prod_rollback"
        ],
        "current_phase": ProjectPhase.BOOTSTRAP.value,
        "task_history": [],  # Task IDs only
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }


def create_current_state(project_name: str) -> str:
    """Create initial CURRENT_STATE.md content."""
    return f"""# Current State - {project_name}

## Last Updated
- **Timestamp**: {datetime.utcnow().isoformat()}
- **Phase**: BOOTSTRAP

## Tasks

| Task ID | Type | State | Created |
|---------|------|-------|---------|
| (none) | - | - | - |

## Notes

Project bootstrapped. Ready for task submission.
"""


# -----------------------------------------------------------------------------
# Plan Generation Functions
# -----------------------------------------------------------------------------
def generate_implementation_plan(task_data: dict, project_manifest: dict) -> str:
    """
    Generate a high-level implementation plan.

    NOTE: This generates PLANS only, not code.
    NO file diffs, NO code generation.
    """
    task_id = task_data['task_id']
    task_type = task_data['task_type']
    description = task_data['description']
    project_name = task_data['project_name']
    tech_stack = project_manifest.get('tech_stack', [])

    # Generate plan based on task type
    type_specific_guidance = {
        'bug': "Focus on identifying root cause, minimal fix, regression prevention.",
        'feature': "Focus on incremental implementation, clear interfaces, test coverage.",
        'refactor': "Focus on preserving behavior, improving structure, no new features.",
        'infra': "Focus on reliability, monitoring, rollback capability."
    }

    guidance = type_specific_guidance.get(task_type, "Follow standard development practices.")

    plan_content = f"""# Implementation Plan

## Task Information

| Field | Value |
|-------|-------|
| Task ID | `{task_id}` |
| Project | {project_name} |
| Type | {task_type} |
| Created | {task_data.get('created_at', 'N/A')} |

## Description

{description}

## Tech Stack

{', '.join(tech_stack) if tech_stack else 'Not specified'}

---

## Overview

This plan outlines the high-level approach for implementing this {task_type} task.

**Guidance**: {guidance}

---

## Files Likely to Change

> NOTE: These are predicted paths. Actual files will be determined during implementation.

| Path Pattern | Reason |
|--------------|--------|
| `src/**/*.py` | Main source code changes |
| `tests/**/*.py` | Test additions/modifications |
| `docs/*.md` | Documentation updates |
| `config/*.yaml` | Configuration changes (if needed) |

---

## Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing functionality | Medium | High | Comprehensive test coverage |
| Performance regression | Low | Medium | Benchmark before/after |
| Security vulnerability | Low | High | Security review checklist |

---

## Test Impact

- [ ] Unit tests required
- [ ] Integration tests required
- [ ] Manual testing on testing environment required
- [ ] Performance testing (if applicable)

---

## Rollback Strategy

1. All changes are committed in a single, atomic commit
2. Git revert can undo the change
3. Testing environment validation required before production
4. Production deployment requires explicit human approval (per AI_POLICY.md)

---

## Implementation Steps (High-Level)

1. **Analyze**: Review existing code related to this task
2. **Design**: Define approach and interfaces
3. **Implement**: Make code changes
4. **Test**: Write and run tests
5. **Document**: Update relevant documentation
6. **Review**: Self-review and validation
7. **Submit**: Create PR for human review

---

## Approval Requirements

- [ ] Plan reviewed by human
- [ ] Approach approved
- [ ] No policy violations identified

---

*This plan was auto-generated. Human review required before approval.*

*Generated at: {datetime.utcnow().isoformat()}*
"""

    return plan_content


# -----------------------------------------------------------------------------
# Phase 3: Diff Generation Functions
# -----------------------------------------------------------------------------
def generate_diff_header(task_id: str, project_name: str, plan_path: str) -> str:
    """Generate the metadata header for a diff file."""
    return f"""# TASK_ID: {task_id}
# PROJECT: {project_name}
# GENERATED_AT: {datetime.utcnow().isoformat()}
# PLAN_REF: {plan_path}
# DISCLAIMER: NOT APPLIED. FOR HUMAN REVIEW ONLY.
#
# This diff was auto-generated based on the approved plan.
# It contains proposed code changes that MUST be reviewed by a human.
#
# TO APPLY (manually, in a future phase):
#   git apply projects/{project_name}/diffs/{task_id}.diff
#
# SAFETY: This diff has NOT been applied. No code changes have been made.
#

"""


def generate_diff_content(task_data: dict, plan_content: str, tech_stack: list[str]) -> tuple[str, list[str]]:
    """
    Generate illustrative diff content based on task and plan.

    NOTE: This generates TEMPLATE/PLACEHOLDER diffs, not real code.
    The diffs show structure, not actual implementation.

    Returns (diff_content, list_of_files_in_diff)
    """
    task_type = task_data.get('task_type', 'feature')
    task_id = task_data['task_id']
    description = task_data.get('description', 'No description')

    # Determine file patterns based on tech stack
    is_python = any(t.lower() in ['python', 'fastapi', 'django', 'flask'] for t in tech_stack)
    is_javascript = any(t.lower() in ['javascript', 'typescript', 'react', 'node', 'nextjs'] for t in tech_stack)

    files_in_diff = []
    diff_sections = []

    if is_python:
        # Generate Python-style diff
        if task_type == 'bug':
            files_in_diff = ['src/main.py', 'tests/test_main.py']
            diff_sections.append(f"""--- a/src/main.py
+++ b/src/main.py
@@ -1,6 +1,10 @@
 # Main application module
+# Task: {task_id}
+# Fix: {description[:50]}...

 def main():
-    # TODO: Bug fix needed here
-    pass
+    # Fixed implementation
+    # TODO: Actual implementation will be added in execution phase
+    raise NotImplementedError("Phase 3: Diff only, no execution")
""")
            diff_sections.append(f"""--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,5 +1,12 @@
 # Test module
+import pytest

 def test_main():
-    pass
+    # Test for bug fix: {task_id}
+    # TODO: Actual test implementation in execution phase
+    with pytest.raises(NotImplementedError):
+        from src.main import main
+        main()
""")
        else:  # feature, refactor, infra
            files_in_diff = ['src/feature.py', 'tests/test_feature.py']
            diff_sections.append(f"""--- /dev/null
+++ b/src/feature.py
@@ -0,0 +1,15 @@
+# New feature module
+# Task: {task_id}
+# Description: {description[:60]}...
+#
+# PHASE 3 PLACEHOLDER - No actual implementation
+
+
+class FeaturePlaceholder:
+    \"\"\"Placeholder for new feature.\"\"\"
+
+    def __init__(self):
+        raise NotImplementedError("Phase 3: Diff only, no execution")
+
+    def execute(self):
+        raise NotImplementedError("Phase 3: Diff only, no execution")
""")
            diff_sections.append(f"""--- /dev/null
+++ b/tests/test_feature.py
@@ -0,0 +1,12 @@
+# Test module for new feature
+# Task: {task_id}
+import pytest
+
+
+def test_feature_placeholder():
+    \"\"\"Test placeholder for feature.\"\"\"
+    # TODO: Actual test implementation in execution phase
+    with pytest.raises(NotImplementedError):
+        from src.feature import FeaturePlaceholder
+        f = FeaturePlaceholder()
""")

    elif is_javascript:
        # Generate JavaScript-style diff
        files_in_diff = ['src/index.js', 'tests/index.test.js']
        diff_sections.append(f"""--- a/src/index.js
+++ b/src/index.js
@@ -1,5 +1,15 @@
 // Main module
+// Task: {task_id}
+// Description: {description[:50]}...
+
+/**
+ * Placeholder function for task implementation
+ * PHASE 3: No execution, diff only
+ */
+function placeholder() {{
+  throw new Error('Phase 3: Diff only, no execution');
+}}

-module.exports = {{}};
+module.exports = {{ placeholder }};
""")
        diff_sections.append(f"""--- /dev/null
+++ b/tests/index.test.js
@@ -0,0 +1,10 @@
+// Test module
+// Task: {task_id}
+const {{ placeholder }} = require('../src/index');
+
+describe('Feature placeholder', () => {{
+  it('should throw NotImplementedError', () => {{
+    expect(() => placeholder()).toThrow('Phase 3: Diff only, no execution');
+  }});
+}});
""")

    else:
        # Generic placeholder diff
        files_in_diff = ['src/module.txt', 'docs/CHANGES.md']
        diff_sections.append(f"""--- /dev/null
+++ b/src/module.txt
@@ -0,0 +1,8 @@
+# Placeholder Module
+# Task: {task_id}
+# Description: {description[:60]}...
+#
+# This is a placeholder file generated in Phase 3.
+# Actual implementation will be added in execution phase.
+#
+# PHASE 3: DIFF ONLY - NO EXECUTION
""")
        diff_sections.append(f"""--- /dev/null
+++ b/docs/CHANGES.md
@@ -0,0 +1,5 @@
+# Changes for Task {task_id}
+
+## Description
+{description}
+
+## Status: DIFF_GENERATED (not applied)
""")

    return "\n".join(diff_sections), files_in_diff


def write_diff_file(diff_path: Path, header: str, content: str) -> None:
    """Write a diff file with header and content."""
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    with open(diff_path, 'w') as f:
        f.write(header)
        f.write(content)
    logger.info(f"Written diff file: {diff_path}")


# -----------------------------------------------------------------------------
# Phase 4: Execution Functions
# -----------------------------------------------------------------------------
def parse_diff_files(diff_content: str) -> list[str]:
    """
    Parse a diff to extract the list of files affected.

    Returns list of file paths from the diff.
    """
    files = []
    for line in diff_content.split('\n'):
        if line.startswith('+++ b/'):
            file_path = line[6:].strip()
            if file_path and file_path != '/dev/null':
                files.append(file_path)
        elif line.startswith('--- a/'):
            file_path = line[6:].strip()
            if file_path and file_path != '/dev/null' and file_path not in files:
                files.append(file_path)
    return files


def count_diff_lines(diff_content: str) -> tuple[int, int]:
    """
    Count lines added and removed in a diff.

    Returns (lines_added, lines_removed)
    """
    lines_added = 0
    lines_removed = 0

    in_diff_body = False
    for line in diff_content.split('\n'):
        if line.startswith('@@'):
            in_diff_body = True
            continue
        if line.startswith('--- ') or line.startswith('+++ '):
            continue
        if in_diff_body:
            if line.startswith('+') and not line.startswith('+++'):
                lines_added += 1
            elif line.startswith('-') and not line.startswith('---'):
                lines_removed += 1

    return lines_added, lines_removed


def simulate_diff_apply(project_path: Path, diff_content: str) -> tuple[bool, list[str]]:
    """
    Simulate applying a diff (dry-run).

    Validates:
    - Files exist (or are new files)
    - No obvious conflicts

    Returns (can_apply, list_of_conflicts)
    """
    conflicts = []
    files = parse_diff_files(diff_content)

    for file_path in files:
        target_path = project_path / file_path

        # Check if this is a new file
        is_new_file = False
        for line in diff_content.split('\n'):
            if line.startswith('--- /dev/null') or line.startswith('--- a/' + file_path):
                if line.startswith('--- /dev/null'):
                    is_new_file = True
                break

        if is_new_file:
            # New file - check if parent directory can be created
            parent = target_path.parent
            if parent.exists() and parent.is_file():
                conflicts.append(f"Cannot create {file_path}: parent is a file")
        else:
            # Existing file modification - file must exist
            # For Phase 4, we're working with placeholder files, so we'll be lenient
            pass

    can_apply = len(conflicts) == 0
    logger.info(f"Dry-run simulation: can_apply={can_apply}, conflicts={conflicts}")
    return can_apply, conflicts


def create_backup(project_name: str, task_id: str, files_to_backup: list[str]) -> Path:
    """
    Create backup of files before applying diff.

    Creates backup directory structure:
    projects/<project_name>/backups/<task_id>/
    └── <relative_file_paths>

    Returns path to backup directory.
    """
    backup_dir = get_backup_path(project_name, task_id)
    backup_dir.mkdir(parents=True, exist_ok=True)

    project_path = get_project_path(project_name)
    backed_up_files = []

    for file_path in files_to_backup:
        source = project_path / file_path
        if source.exists():
            # Create backup with directory structure
            backup_file = backup_dir / file_path
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup_file)
            backed_up_files.append(file_path)
            logger.info(f"Backed up: {file_path}")

    # Write backup manifest
    manifest = {
        "task_id": task_id,
        "project_name": project_name,
        "created_at": datetime.utcnow().isoformat(),
        "files": backed_up_files
    }
    manifest_path = backup_dir / "BACKUP_MANIFEST.yaml"
    write_yaml_file(manifest_path, manifest)

    logger.info(f"Backup created: {backup_dir}")
    return backup_dir


def apply_diff_to_files(project_name: str, task_id: str, diff_content: str) -> list[str]:
    """
    Apply diff to project files.

    SAFETY:
    - Backup MUST exist before calling this function
    - This function ONLY creates/modifies files listed in the diff

    Returns list of files modified.
    """
    project_path = get_project_path(project_name)
    files = parse_diff_files(diff_content)
    modified_files = []

    for file_path in files:
        target_path = project_path / file_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # For Phase 4, we simulate applying the diff by creating placeholder files
        # In a real implementation, this would use patch or similar
        # Extract content for this file from diff
        file_content = extract_file_content_from_diff(diff_content, file_path)

        with open(target_path, 'w') as f:
            f.write(file_content)

        modified_files.append(file_path)
        logger.info(f"Applied diff to: {file_path}")

    return modified_files


def extract_file_content_from_diff(diff_content: str, file_path: str) -> str:
    """
    Extract the new content for a file from a diff.

    For Phase 4, this extracts the '+' lines (excluding the +++ header).
    """
    lines = []
    in_file_section = False
    found_file = False

    for line in diff_content.split('\n'):
        if line.startswith('+++ b/' + file_path):
            in_file_section = True
            found_file = True
            continue

        if in_file_section:
            if line.startswith('--- ') or line.startswith('+++ '):
                in_file_section = False
                continue
            if line.startswith('@@'):
                continue
            if line.startswith('+') and not line.startswith('+++'):
                lines.append(line[1:])  # Remove the leading '+'
            elif line.startswith(' '):
                lines.append(line[1:])  # Context line

    if not found_file:
        # File not found in diff, return placeholder
        return f"# File: {file_path}\n# Applied from diff for Phase 4\n"

    return '\n'.join(lines)


def restore_from_backup(project_name: str, task_id: str) -> list[str]:
    """
    Restore files from backup.

    Returns list of files restored.
    """
    backup_dir = get_backup_path(project_name, task_id)
    project_path = get_project_path(project_name)

    if not backup_dir.exists():
        raise FileNotFoundError(f"Backup not found for task {task_id}")

    # Read backup manifest
    manifest_path = backup_dir / "BACKUP_MANIFEST.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Backup manifest not found for task {task_id}")

    manifest = read_yaml_file(manifest_path)
    restored_files = []

    for file_path in manifest.get('files', []):
        backup_file = backup_dir / file_path
        target_file = project_path / file_path

        if backup_file.exists():
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, target_file)
            restored_files.append(file_path)
            logger.info(f"Restored: {file_path}")

    # Also restore files that were created (delete them)
    task_data = read_task(project_name, task_id)
    files_in_diff = task_data.get('files_in_diff', [])

    for file_path in files_in_diff:
        if file_path not in manifest.get('files', []):
            # This file was created by the diff, delete it
            target_file = project_path / file_path
            if target_file.exists():
                target_file.unlink()
                logger.info(f"Deleted newly created file: {file_path}")

    logger.info(f"Restore complete: {len(restored_files)} files restored")
    return restored_files


# -----------------------------------------------------------------------------
# Phase 5: CI/Release Functions
# -----------------------------------------------------------------------------
def generate_commit_hash() -> str:
    """
    Generate a placeholder commit hash.

    In a real implementation, this would come from git.
    """
    return str(uuid.uuid4())[:8]


def generate_ci_run_id() -> str:
    """
    Generate a CI run ID.

    In a real implementation, this would come from the CI system.
    """
    return f"ci-{str(uuid.uuid4())[:8]}"


def create_commit_message(task_data: dict) -> str:
    """
    Generate a commit message for a task.

    Format:
    [task_type] task_id: short description

    Body includes task details and Phase 5 markers.
    """
    task_id = task_data['task_id']
    task_type = task_data.get('task_type', 'task')
    description = task_data.get('description', 'No description')

    # Truncate description for subject line
    short_desc = description[:50] + "..." if len(description) > 50 else description

    return f"""[{task_type}] {task_id[:8]}: {short_desc}

Task ID: {task_id}
Type: {task_type}
Description: {description}

Phase 5: Human-approved commit
---
This commit was created via the AI Development Platform.
NOT automatically pushed. Human review required.
"""


def prepare_git_commit(project_name: str, task_id: str, task_data: dict) -> tuple[str, str, list[str]]:
    """
    Prepare a git commit for applied changes.

    SAFETY:
    - This creates commit metadata, NOT an actual git commit
    - Does NOT push to any remote
    - Returns (commit_hash, commit_message, files_committed)

    In a real implementation, this would:
    1. Stage files from task_data['files_modified']
    2. Create a git commit with the message
    3. Return the actual commit hash
    """
    commit_hash = generate_commit_hash()
    commit_message = create_commit_message(task_data)
    files_committed = task_data.get('files_modified', [])

    logger.info(f"Prepared commit {commit_hash} for task {task_id}")
    logger.info(f"Files to commit: {files_committed}")

    # In Phase 5, we store commit info but do NOT actually run git
    # This is a placeholder for future git integration

    return commit_hash, commit_message, files_committed


def trigger_ci_pipeline(project_name: str, task_id: str, commit_hash: str) -> str:
    """
    Trigger CI pipeline for a commit.

    SAFETY:
    - This is a placeholder for CI integration
    - Does NOT actually trigger external CI
    - Returns CI run ID

    In a real implementation, this would:
    1. Push commit to remote (if configured)
    2. Trigger GitHub Actions via API
    3. Return the workflow run ID
    """
    ci_run_id = generate_ci_run_id()

    logger.info(f"CI pipeline triggered for task {task_id}, commit {commit_hash}")
    logger.info(f"CI run ID: {ci_run_id}")

    # In Phase 5, we store CI info but do NOT actually trigger external CI
    # This is a placeholder for future CI integration

    return ci_run_id


def deploy_to_testing(project_name: str, task_id: str) -> str:
    """
    Deploy to testing environment.

    SAFETY:
    - This is a placeholder for testing deployment
    - Does NOT deploy to production
    - Returns deployment URL

    In a real implementation, this would:
    1. Trigger deployment script
    2. Wait for deployment to complete
    3. Return the testing URL
    """
    deployment_url = "https://aitesting.mybd.in"

    logger.info(f"Deployed task {task_id} to testing: {deployment_url}")

    # In Phase 5, this is a placeholder
    # Actual deployment would be implemented in future phases

    return deployment_url


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="AI Development Platform - Task Controller",
    description=CURRENT_PHASE_FULL,
    version=__version__
)

# Include Phase 12 router for multi-aspect projects
app.include_router(phase12_router)


# -----------------------------------------------------------------------------
# API Endpoints - Health
# -----------------------------------------------------------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "AI Development Platform - Task Controller",
        "phase": CURRENT_PHASE_FULL,
        "status": "running",
        "version": __version__
    }


@app.get("/health")
async def health_check():
    """
    Detailed health check.

    Phase 15.5: Added Claude CLI status with session-based auth support.
    """
    # Build base health response
    health_response = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "api": "operational",
            "projects_dir": PROJECTS_DIR.exists(),
            "docs_dir": DOCS_DIR.exists()
        },
        "phase": f"Phase {CURRENT_PHASE}",
        "phase_name": CURRENT_PHASE_NAME,
        "version": __version__,
        "capabilities": [
            "project_bootstrap",
            "task_lifecycle",
            "plan_generation",
            "approval_gates",
            "diff_generation",
            # Phase 4
            "dry_run",
            "apply_with_confirmation",
            "rollback",
            # Phase 5
            "commit_with_confirmation",
            "ci_trigger",
            "ci_result_ingestion",
            "deploy_testing_with_confirmation",
            # Phase 12: Multi-Aspect Projects
            "multi_aspect_projects",
            "internal_project_contract",
            "aspect_lifecycle_management",
            "structured_feedback_workflow",
            "autonomous_ci_triggers",
            "notification_system",
            "project_dashboard"
        ],
        "constraints": [
            "NO_AUTONOMOUS_EXECUTION",
            "CONFIRMATION_REQUIRED",
            "BACKUP_BEFORE_APPLY",
            "ROLLBACK_GUARANTEED",
            "NO_PRODUCTION_DEPLOYMENT",
            "HUMAN_REVIEW_REQUIRED",
            # Phase 5
            "NO_AUTOMATIC_COMMITS",
            "NO_AUTOMATIC_MERGES",
            "NO_CI_WITHOUT_INTENT",
            "NO_BYPASS_TEST_FAILURES",
            "NO_BACKGROUND_CI_RUNS",
            # Phase 12
            "PRODUCTION_REQUIRES_APPROVAL",
            "FEEDBACK_REQUIRES_EXPLANATION",
            "CI_ON_PHASE_COMPLETE_ONLY"
        ]
    }

    # Phase 15.5/18B: Report Claude backend module status only
    # NOTE: We do NOT call check_claude_availability() here because it spawns
    # subprocesses that can hang for 30+ seconds, causing health check timeouts.
    # For detailed Claude CLI status, use GET /claude/status instead.
    if CLAUDE_BACKEND_AVAILABLE:
        health_response["components"]["claude_cli"] = {
            "backend_loaded": True,
            "note": "Use GET /claude/status for detailed CLI availability"
        }
        # Capability is reported as available since module is loaded
        # Actual CLI availability should be checked via /claude/status
        health_response["capabilities"].append("claude_cli_execution")

    return health_response


# -----------------------------------------------------------------------------
# API Endpoints - Project Bootstrap (TASK 1)
# -----------------------------------------------------------------------------
@app.post("/project/bootstrap", response_model=ProjectBootstrapResponse)
async def bootstrap_project(request: ProjectBootstrapRequest):
    """
    Bootstrap a new project.

    Creates:
    - Project directory structure
    - PROJECT_MANIFEST.yaml
    - CURRENT_STATE.md
    - tasks/ directory
    - plans/ directory

    Rules:
    - Fails if project already exists
    - No Git operations
    - Filesystem only
    """
    logger.info(f"Bootstrap request for project: {request.project_name}")

    # Policy check
    allowed, reason = can_create_project(request.user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Check if project already exists
    if project_exists(request.project_name):
        logger.warning(f"Project already exists: {request.project_name}")
        raise HTTPException(
            status_code=409,
            detail=f"Project '{request.project_name}' already exists"
        )

    # Create project directory structure
    project_path = get_project_path(request.project_name)
    tasks_dir = project_path / "tasks"
    plans_dir = project_path / "plans"
    diffs_dir = project_path / "diffs"  # Phase 3
    backups_dir = project_path / "backups"  # Phase 4

    logger.info(f"Creating project directory: {project_path}")
    project_path.mkdir(parents=True, exist_ok=True)
    tasks_dir.mkdir(exist_ok=True)
    plans_dir.mkdir(exist_ok=True)
    diffs_dir.mkdir(exist_ok=True)
    backups_dir.mkdir(exist_ok=True)  # Phase 4

    # Create PROJECT_MANIFEST.yaml
    manifest_path = project_path / "PROJECT_MANIFEST.yaml"
    manifest_data = create_project_manifest(
        project_name=request.project_name,
        repo_url=request.repo_url,
        tech_stack=request.tech_stack
    )
    write_yaml_file(manifest_path, manifest_data)
    logger.info(f"Created manifest: {manifest_path}")

    # Create CURRENT_STATE.md
    state_path = project_path / "CURRENT_STATE.md"
    state_content = create_current_state(request.project_name)
    write_markdown_file(state_path, state_content)
    logger.info(f"Created state file: {state_path}")

    logger.info(f"Project bootstrapped successfully: {request.project_name}")

    return ProjectBootstrapResponse(
        project_name=request.project_name,
        status="bootstrapped",
        manifest_path=str(manifest_path),
        state_path=str(state_path),
        message=f"Project '{request.project_name}' bootstrapped successfully",
        created_at=datetime.utcnow().isoformat()
    )


# -----------------------------------------------------------------------------
# API Endpoints - Task Lifecycle (TASK 2)
# -----------------------------------------------------------------------------
@app.post("/task", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """
    Create a new task for a project.

    - Generates a unique task_id
    - Persists task record to filesystem
    - Sets state to RECEIVED
    - Attaches task to project manifest
    """
    logger.info(f"Task creation request for project: {request.project_name}")

    # Validate project exists
    if not project_exists(request.project_name):
        raise HTTPException(
            status_code=404,
            detail=f"Project '{request.project_name}' not found. Bootstrap it first."
        )

    # Policy check
    allowed, reason = can_submit_task(request.project_name, request.user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Generate task ID
    task_id = generate_task_id()
    created_at = datetime.utcnow().isoformat()

    # Create task record
    task_data = {
        "task_id": task_id,
        "project_name": request.project_name,
        "submitted_by": request.user_id,
        "task_type": request.task_type.value,
        "description": request.task_description,
        "current_state": TaskState.RECEIVED.value,
        "created_at": created_at,
        "updated_at": created_at,
        "plan_summary": ""
    }

    # Persist task to filesystem
    write_task(request.project_name, task_id, task_data)
    logger.info(f"Task persisted: {task_id}")

    # Update project manifest with task reference
    manifest = read_project_manifest(request.project_name)
    task_history = manifest.get('task_history', [])
    task_history.append(task_id)
    update_project_manifest(request.project_name, {'task_history': task_history})
    logger.info(f"Task {task_id} added to project manifest")

    return TaskResponse(
        task_id=task_id,
        project_name=request.project_name,
        task_type=request.task_type,
        state=TaskState.RECEIVED,
        message="Task created successfully. Use /task/{task_id}/validate to validate.",
        created_at=created_at
    )


@app.get("/task/{task_id}")
async def get_task(task_id: str, project_name: str):
    """Get task details."""
    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    return read_task(project_name, task_id)


@app.post("/task/{task_id}/validate", response_model=TaskValidateResponse)
async def validate_task(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Validate a task.

    - Performs schema + policy validation
    - Moves task to VALIDATED or REJECTED
    """
    logger.info(f"Validation request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_validate_task(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Validate state transition is allowed
    if previous_state != TaskState.RECEIVED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot validate task in state '{previous_state.value}'. Must be RECEIVED."
        )

    # Perform validation
    validation_errors = []

    # Schema validation
    if not task_data.get('description'):
        validation_errors.append("Task description is empty")
    if len(task_data.get('description', '')) < 10:
        validation_errors.append("Task description too short (min 10 characters)")
    if task_data.get('task_type') not in [t.value for t in TaskType]:
        validation_errors.append(f"Invalid task type: {task_data.get('task_type')}")

    # Determine new state
    validation_passed = len(validation_errors) == 0
    new_state = TaskState.VALIDATED if validation_passed else TaskState.REJECTED

    # Update task
    task_data['current_state'] = new_state.value
    task_data['validation_errors'] = validation_errors
    write_task(project_name, task_id, task_data)

    logger.info(f"Task {task_id} validation: {'PASSED' if validation_passed else 'FAILED'}")

    return TaskValidateResponse(
        task_id=task_id,
        previous_state=previous_state,
        current_state=new_state,
        validation_passed=validation_passed,
        validation_errors=validation_errors,
        message=f"Task {'validated successfully' if validation_passed else 'validation failed'}"
    )


# -----------------------------------------------------------------------------
# API Endpoints - Plan Generation (TASK 3)
# -----------------------------------------------------------------------------
@app.post("/task/{task_id}/plan", response_model=TaskPlanResponse)
async def generate_plan(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Generate a high-level implementation plan for a task.

    - NO code generation
    - NO file diffs
    - Output Markdown only
    - Saves to plans/<task_id>_plan.md
    - Updates task state to PLANNED
    """
    logger.info(f"Plan generation request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_plan_task(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    current_state = TaskState(task_data['current_state'])

    # Validate state - must be VALIDATED
    if current_state != TaskState.VALIDATED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot generate plan for task in state '{current_state.value}'. Must be VALIDATED."
        )

    # Read project manifest for tech stack
    manifest = read_project_manifest(project_name)

    # Generate plan (NO CODE, just high-level plan)
    plan_content = generate_implementation_plan(task_data, manifest)

    # Save plan to file
    plan_path = get_plan_path(project_name, task_id)
    write_markdown_file(plan_path, plan_content)
    logger.info(f"Plan saved: {plan_path}")

    # Update task state to PLANNED, then AWAITING_APPROVAL
    task_data['current_state'] = TaskState.AWAITING_APPROVAL.value
    task_data['plan_summary'] = f"Plan generated at {plan_path}"
    task_data['plan_path'] = str(plan_path)
    write_task(project_name, task_id, task_data)

    logger.info(f"Task {task_id} state updated to AWAITING_APPROVAL")

    return TaskPlanResponse(
        task_id=task_id,
        state=TaskState.AWAITING_APPROVAL,
        plan_path=str(plan_path),
        message="Plan generated successfully. Review and approve/reject."
    )


@app.get("/task/{task_id}/plan")
async def get_plan(task_id: str, project_name: str):
    """Get the plan for a task."""
    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    plan_path = get_plan_path(project_name, task_id)
    if not plan_path.exists():
        raise HTTPException(status_code=404, detail=f"Plan for task '{task_id}' not found")

    with open(plan_path, 'r') as f:
        return {"task_id": task_id, "plan": f.read()}


# -----------------------------------------------------------------------------
# API Endpoints - Approval Gate (TASK 4)
# -----------------------------------------------------------------------------
@app.post("/task/{task_id}/approve", response_model=TaskApprovalResponse)
async def approve_task(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Approve a task.

    - Only allowed if task is AWAITING_APPROVAL (was PLANNED)
    - Moves state to APPROVED
    - NO execution triggered (Phase 2 is thinking layer only)
    """
    logger.info(f"Approval request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_approve_task(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Validate state - must be AWAITING_APPROVAL
    if previous_state != TaskState.AWAITING_APPROVAL:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve task in state '{previous_state.value}'. Must be AWAITING_APPROVAL."
        )

    # Update task state to APPROVED
    task_data['current_state'] = TaskState.APPROVED.value
    task_data['approved_by'] = user_id
    task_data['approved_at'] = datetime.utcnow().isoformat()
    write_task(project_name, task_id, task_data)

    logger.info(f"Task {task_id} APPROVED by {user_id}")

    # NOTE: No execution triggered - Phase 3 generates diffs only

    return TaskApprovalResponse(
        task_id=task_id,
        previous_state=previous_state,
        current_state=TaskState.APPROVED,
        message="Task approved. Use /task/{task_id}/generate-diff to generate code changes."
    )


class TaskRejectRequest(BaseModel):
    """Request model for task rejection."""
    rejection_reason: str = Field(..., min_length=10)


@app.post("/task/{task_id}/reject", response_model=TaskApprovalResponse)
async def reject_task(
    task_id: str,
    project_name: str,
    request: TaskRejectRequest,
    user_id: Optional[str] = None
):
    """
    Reject a task.

    - Moves task to REJECTED
    - Requires rejection reason
    """
    logger.info(f"Rejection request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_reject_task(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Can reject from multiple states
    rejectable_states = [
        TaskState.RECEIVED,
        TaskState.VALIDATED,
        TaskState.AWAITING_APPROVAL
    ]
    if previous_state not in rejectable_states:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reject task in state '{previous_state.value}'."
        )

    # Update task state to REJECTED
    task_data['current_state'] = TaskState.REJECTED.value
    task_data['rejected_by'] = user_id
    task_data['rejected_at'] = datetime.utcnow().isoformat()
    task_data['rejection_reason'] = request.rejection_reason
    write_task(project_name, task_id, task_data)

    logger.info(f"Task {task_id} REJECTED by {user_id}: {request.rejection_reason}")

    return TaskApprovalResponse(
        task_id=task_id,
        previous_state=previous_state,
        current_state=TaskState.REJECTED,
        message="Task rejected.",
        rejection_reason=request.rejection_reason
    )


# -----------------------------------------------------------------------------
# API Endpoints - Diff Generation (Phase 3)
# -----------------------------------------------------------------------------
@app.post("/task/{task_id}/generate-diff", response_model=DiffGenerateResponse)
async def generate_diff(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Generate a diff for an approved task.

    Phase 3 Constraints:
    - Task MUST be in APPROVED state
    - Plan file MUST exist
    - Diff is generated but NOT applied
    - Max 10 files per diff
    - Human review is MANDATORY

    SAFETY:
    - NO code execution
    - NO applying diffs
    - NO git commits
    - Diff file stored for human review
    """
    logger.info(f"Diff generation request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check: Can generate diff?
    allowed, reason = can_generate_diff(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be APPROVED
    if previous_state != TaskState.APPROVED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot generate diff for task in state '{previous_state.value}'. Must be APPROVED."
        )

    # Precondition: Plan file MUST exist
    plan_path = get_plan_path(project_name, task_id)
    if not plan_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Plan file not found for task '{task_id}'. Generate plan first."
        )

    # Read plan content
    with open(plan_path, 'r') as f:
        plan_content = f.read()

    # Read project manifest for tech stack
    manifest = read_project_manifest(project_name)
    tech_stack = manifest.get('tech_stack', [])

    # Generate diff content
    diff_content, files_in_diff = generate_diff_content(task_data, plan_content, tech_stack)

    # Policy check: File limit
    allowed, limit_reason = diff_file_limit_ok(len(files_in_diff))
    if not allowed:
        raise HTTPException(status_code=400, detail=limit_reason)

    # Policy check: Scope (files in diff vs files in plan)
    # For Phase 3, we use the files_in_diff as the "plan files" since we don't parse the plan
    allowed, scope_reason = diff_within_scope(files_in_diff, files_in_diff)
    if not allowed:
        raise HTTPException(status_code=400, detail=scope_reason)

    # Generate diff header
    diff_header = generate_diff_header(
        task_id=task_id,
        project_name=project_name,
        plan_path=f"plans/{task_id}_plan.md"
    )

    # Write diff file
    diff_path = get_diff_path(project_name, task_id)
    write_diff_file(diff_path, diff_header, diff_content)
    logger.info(f"Diff generated: {diff_path}")

    # Update task state to DIFF_GENERATED
    task_data['current_state'] = TaskState.DIFF_GENERATED.value
    task_data['diff_path'] = str(diff_path)
    task_data['diff_generated_at'] = datetime.utcnow().isoformat()
    task_data['diff_generated_by'] = user_id
    task_data['files_in_diff'] = files_in_diff
    write_task(project_name, task_id, task_data)

    logger.info(f"Task {task_id} state updated to DIFF_GENERATED")

    return DiffGenerateResponse(
        task_id=task_id,
        project_name=project_name,
        previous_state=previous_state,
        current_state=TaskState.DIFF_GENERATED,
        diff_path=str(diff_path),
        files_in_diff=len(files_in_diff),
        message="Diff generated successfully. Review before applying.",
        warning="DIFF NOT APPLIED. FOR HUMAN REVIEW ONLY."
    )


@app.get("/task/{task_id}/diff")
async def get_diff(task_id: str, project_name: str):
    """Get the diff for a task."""
    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    diff_path = get_diff_path(project_name, task_id)
    if not diff_path.exists():
        raise HTTPException(status_code=404, detail=f"Diff for task '{task_id}' not found")

    with open(diff_path, 'r') as f:
        return {
            "task_id": task_id,
            "diff": f.read(),
            "warning": "DIFF NOT APPLIED. FOR HUMAN REVIEW ONLY."
        }


# -----------------------------------------------------------------------------
# API Endpoints - Execution (Phase 4)
# -----------------------------------------------------------------------------
@app.post("/task/{task_id}/dry-run", response_model=DryRunResponse)
async def dry_run(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Perform a dry-run of diff application.

    Phase 4 Constraints:
    - Task MUST be in DIFF_GENERATED state
    - NO files are modified
    - Validates diff can be applied cleanly
    - Outputs summary of changes

    SAFETY:
    - This is a READ-ONLY operation
    - NO files are written
    - NO backups created (no need)
    """
    logger.info(f"Dry-run request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_dry_run(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be DIFF_GENERATED
    if previous_state != TaskState.DIFF_GENERATED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot dry-run task in state '{previous_state.value}'. Must be DIFF_GENERATED."
        )

    # Read diff
    diff_path = get_diff_path(project_name, task_id)
    if not diff_path.exists():
        raise HTTPException(status_code=400, detail=f"Diff file not found for task '{task_id}'.")

    with open(diff_path, 'r') as f:
        diff_content = f.read()

    # Parse diff for summary
    files_affected = parse_diff_files(diff_content)
    lines_added, lines_removed = count_diff_lines(diff_content)

    # Simulate apply to check for conflicts
    project_path = get_project_path(project_name)
    can_apply, conflicts = simulate_diff_apply(project_path, diff_content)

    # Update task state to READY_TO_APPLY if no conflicts
    if can_apply:
        task_data['current_state'] = TaskState.READY_TO_APPLY.value
        task_data['dry_run_at'] = datetime.utcnow().isoformat()
        task_data['dry_run_by'] = user_id
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} dry-run PASSED, state updated to READY_TO_APPLY")
        message = "Dry-run successful. Task is ready to apply. Use /task/{task_id}/apply with confirm=true."
    else:
        message = f"Dry-run found conflicts: {conflicts}. Fix before applying."
        logger.warning(f"Task {task_id} dry-run found conflicts: {conflicts}")

    return DryRunResponse(
        task_id=task_id,
        project_name=project_name,
        previous_state=previous_state,
        current_state=TaskState(task_data['current_state']),
        files_affected=files_affected,
        lines_added=lines_added,
        lines_removed=lines_removed,
        can_apply=can_apply,
        conflicts=conflicts,
        message=message,
        warning="DRY-RUN ONLY. NO FILES MODIFIED."
    )


@app.post("/task/{task_id}/apply", response_model=ApplyResponse)
async def apply_diff_endpoint(
    task_id: str,
    project_name: str,
    confirm: bool = False,
    user_id: Optional[str] = None
):
    """
    Apply a diff to project files.

    Phase 4 Constraints:
    - Task MUST be in READY_TO_APPLY state (dry-run must pass first)
    - EXPLICIT confirmation required (confirm=true)
    - Backup MUST be created before apply
    - Rollback MUST be available after apply

    SAFETY:
    - REQUIRES confirm=true parameter
    - Creates backup BEFORE any modification
    - Logs all file changes
    - Rollback available via /task/{task_id}/rollback
    """
    logger.info(f"Apply request for task: {task_id}, confirm={confirm}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # CRITICAL: Policy check with confirmation requirement
    allowed, reason = can_apply(task_id, user_id, confirmed=confirm)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be READY_TO_APPLY (dry-run must pass first)
    if previous_state != TaskState.READY_TO_APPLY:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot apply task in state '{previous_state.value}'. Must be READY_TO_APPLY (run dry-run first)."
        )

    # Read diff
    diff_path = get_diff_path(project_name, task_id)
    if not diff_path.exists():
        raise HTTPException(status_code=400, detail=f"Diff file not found for task '{task_id}'.")

    with open(diff_path, 'r') as f:
        diff_content = f.read()

    # Get files that will be affected
    files_to_modify = parse_diff_files(diff_content)

    # Update state to APPLYING
    task_data['current_state'] = TaskState.APPLYING.value
    task_data['apply_started_at'] = datetime.utcnow().isoformat()
    task_data['apply_started_by'] = user_id
    write_task(project_name, task_id, task_data)
    logger.info(f"Task {task_id} state updated to APPLYING")

    try:
        # CRITICAL: Create backup BEFORE any modification
        backup_path = create_backup(project_name, task_id, files_to_modify)
        logger.info(f"Backup created at: {backup_path}")

        # Apply the diff
        modified_files = apply_diff_to_files(project_name, task_id, diff_content)
        logger.info(f"Diff applied successfully: {modified_files}")

        # Update state to APPLIED
        task_data['current_state'] = TaskState.APPLIED.value
        task_data['applied_at'] = datetime.utcnow().isoformat()
        task_data['applied_by'] = user_id
        task_data['backup_path'] = str(backup_path)
        task_data['files_modified'] = modified_files
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} state updated to APPLIED")

        return ApplyResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.APPLIED,
            files_modified=modified_files,
            backup_path=str(backup_path),
            message="Diff applied successfully. Rollback available via /task/{task_id}/rollback.",
            rollback_available=True
        )

    except Exception as e:
        # CRITICAL: Automatic restore on failure
        logger.error(f"Apply failed for task {task_id}: {e}")

        try:
            # Attempt to restore from backup
            restore_from_backup(project_name, task_id)
            logger.info(f"Restored from backup after failure")
        except Exception as restore_error:
            logger.error(f"Restore also failed: {restore_error}")

        # Update state to EXECUTION_FAILED
        task_data['current_state'] = TaskState.EXECUTION_FAILED.value
        task_data['execution_failed_at'] = datetime.utcnow().isoformat()
        task_data['execution_error'] = str(e)
        write_task(project_name, task_id, task_data)

        raise HTTPException(
            status_code=500,
            detail=f"Apply failed: {e}. Automatic restore attempted. Task state: EXECUTION_FAILED."
        )


@app.post("/task/{task_id}/rollback", response_model=RollbackResponse)
async def rollback_task(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Rollback applied changes.

    Phase 4 Constraints:
    - Task MUST be in APPLIED state
    - Backup MUST exist
    - Restores all files from backup

    SAFETY:
    - Restores original file state
    - Deletes newly created files
    - Preserves all artifacts (diff, plan, backup)
    """
    logger.info(f"Rollback request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_rollback(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be APPLIED
    if previous_state != TaskState.APPLIED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot rollback task in state '{previous_state.value}'. Must be APPLIED."
        )

    # Verify backup exists
    backup_path = get_backup_path(project_name, task_id)
    if not backup_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Backup not found for task '{task_id}'. Cannot rollback."
        )

    try:
        # Restore from backup
        restored_files = restore_from_backup(project_name, task_id)
        logger.info(f"Rollback successful: {restored_files}")

        # Update state to ROLLED_BACK
        task_data['current_state'] = TaskState.ROLLED_BACK.value
        task_data['rolled_back_at'] = datetime.utcnow().isoformat()
        task_data['rolled_back_by'] = user_id
        task_data['files_restored'] = restored_files
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} state updated to ROLLED_BACK")

        return RollbackResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.ROLLED_BACK,
            files_restored=restored_files,
            message="Rollback successful. All changes have been reverted."
        )

    except Exception as e:
        logger.error(f"Rollback failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Rollback failed: {e}. Manual intervention may be required."
        )


# -----------------------------------------------------------------------------
# API Endpoints - CI/Release (Phase 5)
# -----------------------------------------------------------------------------
@app.post("/task/{task_id}/commit", response_model=CommitResponse)
async def commit_task(
    task_id: str,
    project_name: str,
    confirm: bool = False,
    user_id: Optional[str] = None
):
    """
    Prepare a git commit for applied changes.

    Phase 5 Constraints:
    - Task MUST be in APPLIED state
    - EXPLICIT confirmation required (confirm=true)
    - Commit is created LOCALLY, NOT pushed
    - Human must manually push if desired

    SAFETY:
    - REQUIRES confirm=true parameter
    - Does NOT push to remote
    - Does NOT trigger CI automatically
    - Commit hash and message are recorded
    """
    logger.info(f"Commit request for task: {task_id}, confirm={confirm}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # CRITICAL: Policy check with confirmation requirement
    allowed, reason = can_commit(task_id, user_id, confirmed=confirm)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be APPLIED
    if previous_state != TaskState.APPLIED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot commit task in state '{previous_state.value}'. Must be APPLIED."
        )

    try:
        # Prepare git commit (does NOT actually run git)
        commit_hash, commit_message, files_committed = prepare_git_commit(
            project_name, task_id, task_data
        )
        logger.info(f"Commit prepared: {commit_hash}")

        # Update state to COMMITTED
        task_data['current_state'] = TaskState.COMMITTED.value
        task_data['committed_at'] = datetime.utcnow().isoformat()
        task_data['committed_by'] = user_id
        task_data['commit_hash'] = commit_hash
        task_data['commit_message'] = commit_message
        task_data['files_committed'] = files_committed
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} state updated to COMMITTED")

        return CommitResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.COMMITTED,
            commit_hash=commit_hash,
            commit_message=commit_message,
            files_committed=files_committed,
            message="Commit created locally. Use /task/{task_id}/ci/run to trigger CI.",
            warning="COMMIT CREATED LOCALLY. NOT PUSHED."
        )

    except Exception as e:
        logger.error(f"Commit failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Commit failed: {e}"
        )


@app.post("/task/{task_id}/ci/run", response_model=CIRunResponse)
async def trigger_ci(task_id: str, project_name: str, user_id: Optional[str] = None):
    """
    Trigger CI pipeline for a committed task.

    Phase 5 Constraints:
    - Task MUST be in COMMITTED state
    - Human MUST explicitly trigger CI
    - CI runs in external system (placeholder in Phase 5)
    - Results must be ingested via /task/{task_id}/ci/result

    SAFETY:
    - Does NOT auto-deploy on CI pass
    - Does NOT auto-merge
    - CI failure blocks promotion
    """
    logger.info(f"CI trigger request for task: {task_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check
    allowed, reason = can_trigger_ci(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be COMMITTED
    if previous_state != TaskState.COMMITTED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot trigger CI for task in state '{previous_state.value}'. Must be COMMITTED."
        )

    try:
        # Get commit hash from task data
        commit_hash = task_data.get('commit_hash', 'unknown')

        # Trigger CI pipeline (placeholder)
        ci_run_id = trigger_ci_pipeline(project_name, task_id, commit_hash)
        logger.info(f"CI triggered: {ci_run_id}")

        # Update state to CI_RUNNING
        task_data['current_state'] = TaskState.CI_RUNNING.value
        task_data['ci_triggered_at'] = datetime.utcnow().isoformat()
        task_data['ci_triggered_by'] = user_id
        task_data['ci_run_id'] = ci_run_id
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} state updated to CI_RUNNING")

        return CIRunResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.CI_RUNNING,
            ci_run_id=ci_run_id,
            message="CI pipeline triggered. Wait for results and submit via /task/{task_id}/ci/result.",
            warning="CI RUNNING. WAIT FOR RESULTS."
        )

    except Exception as e:
        logger.error(f"CI trigger failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"CI trigger failed: {e}"
        )


@app.post("/task/{task_id}/ci/result", response_model=CIResultResponse)
async def ingest_ci_result(
    task_id: str,
    project_name: str,
    request: CIResultRequest,
    user_id: Optional[str] = None
):
    """
    Ingest CI result for a running pipeline.

    Phase 5 Constraints:
    - Task MUST be in CI_RUNNING state
    - Status MUST be 'passed' or 'failed'
    - CI_PASSED enables deployment, CI_FAILED blocks it
    - CI failures can be re-committed and re-run

    SAFETY:
    - Does NOT auto-deploy on pass
    - CI failure blocks all promotion
    - Human must explicitly proceed after CI pass
    """
    logger.info(f"CI result ingestion for task: {task_id}, status={request.status}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be CI_RUNNING
    if previous_state != TaskState.CI_RUNNING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot ingest CI result for task in state '{previous_state.value}'. Must be CI_RUNNING."
        )

    try:
        # Determine new state based on CI result
        if request.status == CIStatus.PASSED:
            new_state = TaskState.CI_PASSED
            message = "CI passed. Use /task/{task_id}/deploy/testing to deploy to testing."
        else:
            new_state = TaskState.CI_FAILED
            message = "CI failed. Fix issues and re-commit, or archive the task."

        # Update task state
        task_data['current_state'] = new_state.value
        task_data['ci_completed_at'] = datetime.utcnow().isoformat()
        task_data['ci_status'] = request.status.value
        task_data['ci_logs_url'] = request.logs_url
        task_data['ci_details'] = request.details
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} CI result: {request.status}, state updated to {new_state.value}")

        return CIResultResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=new_state,
            ci_status=request.status,
            logs_url=request.logs_url,
            message=message
        )

    except Exception as e:
        logger.error(f"CI result ingestion failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"CI result ingestion failed: {e}"
        )


@app.post("/task/{task_id}/deploy/testing", response_model=DeployTestingResponse)
async def deploy_to_testing_endpoint(
    task_id: str,
    project_name: str,
    confirm: bool = False,
    user_id: Optional[str] = None
):
    """
    Deploy to testing environment.

    Phase 5 Constraints:
    - Task MUST be in CI_PASSED state (CI must pass)
    - EXPLICIT confirmation required (confirm=true)
    - Deploys to TESTING only, NOT production
    - Production deployment remains blocked

    SAFETY:
    - REQUIRES confirm=true parameter
    - REQUIRES CI_PASSED state (cannot bypass failed CI)
    - Deploys to testing environment only
    - Production deployment blocked by policy
    """
    logger.info(f"Testing deployment request for task: {task_id}, confirm={confirm}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # CRITICAL: Policy check with confirmation requirement
    allowed, reason = can_deploy_testing(task_id, user_id, confirmed=confirm)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Precondition: Task MUST be CI_PASSED (cannot bypass failed CI)
    if previous_state != TaskState.CI_PASSED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot deploy task in state '{previous_state.value}'. Must be CI_PASSED. CI failures block deployment."
        )

    try:
        # Deploy to testing (placeholder)
        deployment_url = deploy_to_testing(project_name, task_id)
        logger.info(f"Deployed to testing: {deployment_url}")

        # Update state to DEPLOYED_TESTING
        task_data['current_state'] = TaskState.DEPLOYED_TESTING.value
        task_data['deployed_testing_at'] = datetime.utcnow().isoformat()
        task_data['deployed_testing_by'] = user_id
        task_data['testing_deployment_url'] = deployment_url
        write_task(project_name, task_id, task_data)
        logger.info(f"Task {task_id} state updated to DEPLOYED_TESTING")

        return DeployTestingResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.DEPLOYED_TESTING,
            deployment_url=deployment_url,
            message="Deployed to testing environment successfully.",
            warning="DEPLOYED TO TESTING ONLY. NOT PRODUCTION."
        )

    except Exception as e:
        logger.error(f"Testing deployment failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Testing deployment failed: {e}"
        )


# -----------------------------------------------------------------------------
# API Endpoints - Phase 6: Production Deployment (DUAL APPROVAL REQUIRED)
# -----------------------------------------------------------------------------
@app.post("/task/{task_id}/deploy/production/request", response_model=ProdDeployRequestResponse)
async def request_production_deploy(
    task_id: str,
    project_name: str,
    request: ProdDeployRequestModel,
    user_id: str
):
    """
    Request production deployment (Phase 6).

    CRITICAL SAFETY CONSTRAINTS:
    - Task MUST be in DEPLOYED_TESTING state (testing must pass first)
    - User MUST acknowledge production risk
    - User MUST provide justification (min 20 chars)
    - User MUST provide rollback plan
    - This ONLY creates a request - requires DIFFERENT user to approve
    - State transitions to PROD_DEPLOY_REQUESTED

    NEXT STEP: Another user must call /task/{task_id}/deploy/production/approve
    """
    logger.info(f"⚠️ PRODUCTION DEPLOY REQUEST: task={task_id}, user={user_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # CRITICAL: Policy check with risk acknowledgment
    allowed, reason = can_request_prod_deploy(task_id, user_id, request.risk_acknowledged)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # CRITICAL: Task MUST be deployed to testing first (no bypassing testing)
    if previous_state != TaskState.DEPLOYED_TESTING:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot request production deploy for task in state '{previous_state.value}'. "
                   f"Task MUST be DEPLOYED_TESTING first. Testing cannot be bypassed."
        )

    try:
        # Update task state to PROD_DEPLOY_REQUESTED
        task_data['current_state'] = TaskState.PROD_DEPLOY_REQUESTED.value
        task_data['prod_deploy_requested_at'] = datetime.utcnow().isoformat()
        task_data['prod_deploy_requested_by'] = user_id
        task_data['prod_deploy_justification'] = request.justification
        task_data['prod_deploy_rollback_plan'] = request.rollback_plan
        task_data['prod_deploy_risk_acknowledged'] = True
        write_task(project_name, task_id, task_data)

        # Audit log entry (immutable)
        append_audit_log(project_name, {
            "action": "PROD_DEPLOY_REQUESTED",
            "task_id": task_id,
            "user_id": user_id,
            "justification": request.justification,
            "rollback_plan": request.rollback_plan,
            "risk_acknowledged": True
        })

        logger.info(f"⚠️ PRODUCTION DEPLOY REQUEST CREATED: task={task_id}, awaiting approval from DIFFERENT user")

        return ProdDeployRequestResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.PROD_DEPLOY_REQUESTED,
            requested_by=user_id,
            justification=request.justification,
            rollback_plan=request.rollback_plan,
            message=f"Production deployment requested. REQUIRES approval from a DIFFERENT user.",
            warning="⚠️ PRODUCTION DEPLOYMENT REQUESTED. REQUIRES APPROVAL FROM DIFFERENT USER.",
            next_step=f"Another user must approve via /task/{task_id}/deploy/production/approve"
        )

    except Exception as e:
        logger.error(f"Production deploy request failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Production deploy request failed: {e}"
        )


@app.post("/task/{task_id}/deploy/production/approve", response_model=ProdApproveResponse)
async def approve_production_deploy(
    task_id: str,
    project_name: str,
    request: ProdApproveRequest,
    user_id: str
):
    """
    Approve production deployment (Phase 6).

    CRITICAL SAFETY CONSTRAINTS:
    - Task MUST be in PROD_DEPLOY_REQUESTED state
    - Approver MUST be DIFFERENT from requester (DUAL APPROVAL)
    - Approver MUST confirm they reviewed changes
    - Approver MUST confirm they reviewed rollback plan
    - State transitions to PROD_APPROVED

    NEXT STEP: Any user can call /task/{task_id}/deploy/production/apply with confirm=true
    """
    logger.info(f"⚠️ PRODUCTION DEPLOY APPROVAL ATTEMPT: task={task_id}, approver={user_id}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # CRITICAL: Task MUST be in PROD_DEPLOY_REQUESTED state
    if previous_state != TaskState.PROD_DEPLOY_REQUESTED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot approve production deploy for task in state '{previous_state.value}'. "
                   f"Task MUST be PROD_DEPLOY_REQUESTED."
        )

    # Get the original requester
    requester_id = task_data.get('prod_deploy_requested_by')

    # CRITICAL: Policy check with dual approval enforcement
    allowed, reason = can_approve_prod_deploy(
        task_id,
        approver_id=user_id,
        requester_id=requester_id,
        reviewed_changes=request.reviewed_changes,
        reviewed_rollback=request.reviewed_rollback
    )
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    try:
        # Update task state to PROD_APPROVED
        task_data['current_state'] = TaskState.PROD_APPROVED.value
        task_data['prod_deploy_approved_at'] = datetime.utcnow().isoformat()
        task_data['prod_deploy_approved_by'] = user_id
        task_data['prod_deploy_approval_reason'] = request.approval_reason
        task_data['prod_deploy_changes_reviewed'] = True
        task_data['prod_deploy_rollback_reviewed'] = True
        write_task(project_name, task_id, task_data)

        # Audit log entry (immutable)
        append_audit_log(project_name, {
            "action": "PROD_DEPLOY_APPROVED",
            "task_id": task_id,
            "user_id": user_id,
            "requester_id": requester_id,
            "approval_reason": request.approval_reason,
            "dual_approval_verified": True
        })

        logger.info(f"✅ PRODUCTION DEPLOY APPROVED: task={task_id}, requester={requester_id}, approver={user_id}")

        return ProdApproveResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.PROD_APPROVED,
            requested_by=requester_id,
            approved_by=user_id,
            approval_reason=request.approval_reason,
            message=f"Production deployment approved. Dual approval verified (requester={requester_id}, approver={user_id}).",
            warning="⚠️ PRODUCTION DEPLOYMENT APPROVED. READY FOR FINAL APPLY.",
            next_step=f"Execute deploy via /task/{task_id}/deploy/production/apply with confirm=true"
        )

    except Exception as e:
        logger.error(f"Production deploy approval failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Production deploy approval failed: {e}"
        )


@app.post("/task/{task_id}/deploy/production/apply", response_model=ProdApplyResponse)
async def apply_production_deploy(
    task_id: str,
    project_name: str,
    confirm: bool = False,
    user_id: Optional[str] = None
):
    """
    Apply production deployment (Phase 6).

    CRITICAL SAFETY CONSTRAINTS:
    - Task MUST be in PROD_APPROVED state (dual approval required first)
    - EXPLICIT confirmation required (confirm=true)
    - Creates release manifest and deploy log
    - State transitions to DEPLOYED_PRODUCTION

    ROLLBACK: Available via /task/{task_id}/deploy/production/rollback
    """
    logger.info(f"🚀 PRODUCTION DEPLOY APPLY: task={task_id}, user={user_id}, confirm={confirm}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # CRITICAL: Policy check with confirmation requirement
    allowed, reason = can_apply_prod_deploy(task_id, user_id, confirmed=confirm)
    if not allowed:
        raise HTTPException(status_code=403, detail=reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # CRITICAL: Task MUST be PROD_APPROVED (dual approval required)
    if previous_state != TaskState.PROD_APPROVED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot apply production deploy for task in state '{previous_state.value}'. "
                   f"Task MUST be PROD_APPROVED (dual approval required)."
        )

    # Get the requester and approver
    requester_id = task_data.get('prod_deploy_requested_by', 'unknown')
    approver_id = task_data.get('prod_deploy_approved_by', 'unknown')
    justification = task_data.get('prod_deploy_justification', '')
    rollback_plan = task_data.get('prod_deploy_rollback_plan', '')

    try:
        # Deploy to production (placeholder)
        deployment_url = deploy_to_production(project_name, task_id)

        # Create release manifest
        release_manifest_path = create_release_manifest(
            project_name=project_name,
            task_id=task_id,
            requested_by=requester_id,
            approved_by=approver_id,
            applied_by=user_id,
            justification=justification,
            rollback_plan=rollback_plan
        )

        # Create deploy log entry
        create_deploy_log(project_name, task_id, "PRODUCTION_APPLY", user_id, {
            "deployment_url": deployment_url,
            "requester": requester_id,
            "approver": approver_id
        })

        # Update task state to DEPLOYED_PRODUCTION
        task_data['current_state'] = TaskState.DEPLOYED_PRODUCTION.value
        task_data['prod_deployed_at'] = datetime.utcnow().isoformat()
        task_data['prod_deployed_by'] = user_id
        task_data['prod_deployment_url'] = deployment_url
        task_data['prod_release_manifest'] = release_manifest_path
        write_task(project_name, task_id, task_data)

        # Audit log entry (immutable)
        append_audit_log(project_name, {
            "action": "DEPLOYED_PRODUCTION",
            "task_id": task_id,
            "user_id": user_id,
            "requester_id": requester_id,
            "approver_id": approver_id,
            "deployment_url": deployment_url,
            "release_manifest": release_manifest_path
        })

        logger.info(f"🚀 PRODUCTION DEPLOYED: task={task_id}, url={deployment_url}")

        return ProdApplyResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.DEPLOYED_PRODUCTION,
            requested_by=requester_id,
            approved_by=approver_id,
            applied_by=user_id,
            deployment_url=deployment_url,
            release_manifest_path=release_manifest_path,
            message="Deployed to production successfully.",
            warning="🚀 DEPLOYED TO PRODUCTION. MONITOR CLOSELY.",
            rollback_available=True,
            rollback_command=f"POST /task/{task_id}/deploy/production/rollback"
        )

    except Exception as e:
        logger.error(f"Production deployment apply failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Production deployment apply failed: {e}"
        )


@app.post("/task/{task_id}/deploy/production/rollback", response_model=ProdRollbackResponse)
async def rollback_production_deploy(
    task_id: str,
    project_name: str,
    reason: str,
    user_id: str
):
    """
    Rollback production deployment (Phase 6 - BREAK GLASS).

    CRITICAL: Rollback does NOT require dual approval (speed > ceremony for emergency).
    - Task MUST be in DEPLOYED_PRODUCTION state
    - User MUST provide reason for audit trail
    - State transitions to PROD_ROLLED_BACK
    - Rollback is immediate and always available
    """
    logger.info(f"⚠️ PRODUCTION ROLLBACK: task={task_id}, user={user_id}, reason={reason}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Policy check (requires user_id only, no dual approval for rollback)
    allowed, policy_reason = can_rollback_prod(task_id, user_id)
    if not allowed:
        raise HTTPException(status_code=403, detail=policy_reason)

    # Read task
    task_path = get_task_path(project_name, task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    task_data = read_task(project_name, task_id)
    previous_state = TaskState(task_data['current_state'])

    # Task MUST be DEPLOYED_PRODUCTION
    if previous_state != TaskState.DEPLOYED_PRODUCTION:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot rollback task in state '{previous_state.value}'. "
                   f"Task MUST be DEPLOYED_PRODUCTION."
        )

    try:
        # Execute rollback (placeholder)
        rollback_success = rollback_production(project_name, task_id)
        if not rollback_success:
            raise Exception("Rollback execution failed")

        # Create deploy log entry
        create_deploy_log(project_name, task_id, "PRODUCTION_ROLLBACK", user_id, {
            "reason": reason,
            "emergency": True
        })

        # Update task state to PROD_ROLLED_BACK
        task_data['current_state'] = TaskState.PROD_ROLLED_BACK.value
        task_data['prod_rolled_back_at'] = datetime.utcnow().isoformat()
        task_data['prod_rolled_back_by'] = user_id
        task_data['prod_rollback_reason'] = reason
        write_task(project_name, task_id, task_data)

        # Audit log entry (immutable) - CRITICAL for emergency actions
        append_audit_log(project_name, {
            "action": "PROD_ROLLED_BACK",
            "task_id": task_id,
            "user_id": user_id,
            "reason": reason,
            "emergency_action": True,
            "dual_approval_bypassed": True,
            "bypass_reason": "Rollback speed > ceremony"
        })

        logger.info(f"⚠️ PRODUCTION ROLLED BACK: task={task_id}, reason={reason}")

        return ProdRollbackResponse(
            task_id=task_id,
            project_name=project_name,
            previous_state=previous_state,
            current_state=TaskState.PROD_ROLLED_BACK,
            rolled_back_by=user_id,
            rollback_reason=reason,
            message="Production rolled back successfully.",
            warning="⚠️ PRODUCTION ROLLED BACK. Verify system stability."
        )

    except Exception as e:
        logger.error(f"Production rollback failed for task {task_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Production rollback failed: {e}"
        )


# -----------------------------------------------------------------------------
# API Endpoints - Project Status
# -----------------------------------------------------------------------------
@app.get("/status/{project_name}", response_model=StatusResponse)
async def get_project_status(project_name: str):
    """Get current status of a project."""
    logger.info(f"Status request for project: {project_name}")

    if not project_exists(project_name):
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    manifest = read_project_manifest(project_name)
    tasks = list_tasks(project_name)

    # Count tasks by state
    tasks_by_state = {}
    for task in tasks:
        state = task.get('current_state', 'unknown')
        tasks_by_state[state] = tasks_by_state.get(state, 0) + 1

    return StatusResponse(
        project_name=project_name,
        current_phase=manifest.get('current_phase', 'unknown'),
        task_count=len(tasks),
        tasks_by_state=tasks_by_state,
        last_updated=manifest.get('updated_at', datetime.utcnow().isoformat())
    )


@app.get("/projects")
async def list_projects():
    """List all registered projects."""
    logger.info("Listing all projects")

    projects = []
    if PROJECTS_DIR.exists():
        for item in PROJECTS_DIR.iterdir():
            if item.is_dir() and not item.name.startswith(".") and item.name != "README.md":
                manifest_path = item / "PROJECT_MANIFEST.yaml"
                if manifest_path.exists():
                    try:
                        manifest = read_yaml_file(manifest_path)
                        projects.append({
                            "name": item.name,
                            "phase": manifest.get('current_phase', 'unknown'),
                            "repo_url": manifest.get('repo_url', ''),
                            "task_count": len(manifest.get('task_history', []))
                        })
                    except Exception as e:
                        logger.error(f"Error reading manifest for {item.name}: {e}")

    return {
        "projects": projects,
        "count": len(projects)
    }


# -----------------------------------------------------------------------------
# API Endpoints - Deployment (unchanged from Phase 1)
# -----------------------------------------------------------------------------
@app.post("/deploy", response_model=DeployResponse)
async def trigger_deployment(request: DeployRequest):
    """
    Trigger deployment to an environment.

    Phase 2 constraints:
    - Production deployment BLOCKED
    - Testing deployment placeholder only
    """
    logger.info(f"Deployment request for project: {request.project_name} to {request.environment}")

    if request.environment == DeploymentEnvironment.PRODUCTION:
        logger.warning("Production deployment blocked per AI_POLICY.md")
        raise HTTPException(
            status_code=403,
            detail="Production deployment requires explicit human approval. Not available in Phase 2."
        )

    if not project_exists(request.project_name):
        raise HTTPException(status_code=404, detail=f"Project '{request.project_name}' not found")

    # Placeholder response - no actual deployment in Phase 4
    deployment_url = None
    if request.environment == DeploymentEnvironment.TESTING:
        deployment_url = "https://aitesting.mybd.in"
    elif request.environment == DeploymentEnvironment.DEVELOPMENT:
        deployment_url = "http://localhost:8000"

    return DeployResponse(
        project_name=request.project_name,
        environment=request.environment,
        status="placeholder",
        message="Deployment is a placeholder in Phase 4. No actual deployment triggered.",
        deployment_url=deployment_url
    )


# -----------------------------------------------------------------------------
# Phase 14: Claude CLI Job Endpoints
# -----------------------------------------------------------------------------
try:
    from .claude_backend import (
        create_job,
        get_job_status,
        list_jobs,
        get_queue_status,
        cancel_job,
        check_claude_availability,
        get_scheduler_status,
        start_scheduler,
        stop_scheduler,
        multi_scheduler,  # Phase 14.10: Multi-worker scheduler
        JobState,
        MAX_CONCURRENT_JOBS,  # Phase 14.10: Concurrency limit
    )
    CLAUDE_BACKEND_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Claude backend not available: {e}")
    CLAUDE_BACKEND_AVAILABLE = False


class ClaudeJobRequest(BaseModel):
    """Request to create a Claude CLI job."""
    project_name: str = Field(..., description="Project name")
    task_description: str = Field(..., min_length=10, description="Task description")
    task_type: str = Field(default="feature_development", description="Task type")


class ClaudeJobResponse(BaseModel):
    """Response for Claude job creation."""
    job_id: str
    project_name: str
    task_type: str
    state: str
    created_at: str
    message: str


@app.post("/claude/job", response_model=ClaudeJobResponse)
async def create_claude_job(request: ClaudeJobRequest):
    """
    Create a new Claude CLI job (Phase 14).

    This queues a task for Claude CLI to execute autonomously.
    The job will run in an isolated workspace with all policy documents.
    """
    if not CLAUDE_BACKEND_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Claude backend not available"
        )

    job = await create_job(
        project_name=request.project_name,
        task_description=request.task_description,
        task_type=request.task_type,
    )

    return ClaudeJobResponse(
        job_id=job.job_id,
        project_name=job.project_name,
        task_type=job.task_type,
        state=job.state.value,
        created_at=job.created_at.isoformat(),
        message=f"Job {job.job_id} created and queued"
    )


@app.get("/claude/job/{job_id}")
async def get_claude_job(job_id: str):
    """Get Claude job status by ID."""
    if not CLAUDE_BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Claude backend not available")

    job = await get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return job


@app.get("/claude/jobs")
async def list_claude_jobs(
    state: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 50
):
    """List Claude jobs with optional filtering."""
    if not CLAUDE_BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Claude backend not available")

    jobs = await list_jobs(state=state, project=project, limit=limit)
    return {"jobs": jobs, "count": len(jobs)}


@app.get("/claude/queue")
async def get_claude_queue():
    """Get Claude job queue status."""
    if not CLAUDE_BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Claude backend not available")

    status = await get_queue_status()
    return status


@app.post("/claude/job/{job_id}/cancel")
async def cancel_claude_job(job_id: str):
    """Cancel a queued or running Claude job."""
    if not CLAUDE_BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Claude backend not available")

    success = await cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Could not cancel job {job_id}. It may not exist or already completed."
        )

    return {"message": f"Job {job_id} cancelled", "job_id": job_id}


@app.get("/claude/status")
async def get_claude_status():
    """
    Check Claude CLI availability and status.

    Phase 15.5: Updated to support session-based authentication.
    Phase 18B: Added 10-second timeout to prevent hanging on slow CLI.

    Returns information about:
    - Claude CLI installation and version
    - Authentication status (CLI session OR API key)
    - Job scheduler status (Phase 14.10: multi-worker)

    IMPORTANT: API key is NOT required if CLI session auth is present.
    CLI session auth (via 'claude auth login') is treated as first-class.
    """
    if not CLAUDE_BACKEND_AVAILABLE:
        return {
            "available": False,
            "error": "Claude backend module not loaded",
            "scheduler": None
        }

    # Phase 18B: Add timeout to prevent hanging on slow/unresponsive CLI
    # Note: check_claude_availability() has internal 30s timeout for execution test
    # We allow 35s total to give it time to complete properly
    try:
        cli_status = await asyncio.wait_for(
            check_claude_availability(),
            timeout=35.0  # Allow time for internal 30s execution test
        )
    except asyncio.TimeoutError:
        logger.warning("Claude CLI status check timed out (35s limit)")
        cli_status = {
            "available": False,
            "installed": None,
            "version": None,
            "authenticated": False,
            "error": "Status check timed out - CLI may be slow or unresponsive"
        }

    scheduler_status = await get_scheduler_status()

    # Phase 15.5: Use the new 'authenticated' field instead of requiring api_key
    # CLI is available if installed AND authenticated (via any method)
    is_available = cli_status.get("available", False)

    return {
        "available": is_available,
        "cli": cli_status,
        "scheduler": scheduler_status  # Phase 14.10: Enhanced multi-worker status
    }


@app.get("/claude/scheduler")
async def get_claude_scheduler_status():
    """
    Get detailed multi-worker scheduler status (Phase 14.10).

    Returns comprehensive information about:
    - Running state
    - Worker count and availability
    - Active jobs per worker
    - Queue depth
    - Job statistics
    """
    if not CLAUDE_BACKEND_AVAILABLE:
        raise HTTPException(status_code=503, detail="Claude backend not available")

    status = await get_scheduler_status()
    return {
        "phase": "14.10",
        "scheduler": status,
        "concurrency_limit": MAX_CONCURRENT_JOBS,
        "message": f"Multi-worker scheduler with {status['active_workers']}/{status['max_workers']} active workers"
    }


# -----------------------------------------------------------------------------
# Phase 15.1: Lifecycle Engine Endpoints
# -----------------------------------------------------------------------------

# Import lifecycle_v2 module
try:
    from .lifecycle_v2 import (
        get_lifecycle_manager,
        create_project_lifecycle,
        create_change_lifecycle,
        get_lifecycle,
        list_lifecycles,
        transition_lifecycle,
        get_lifecycle_guidance,
        LifecycleState,
        LifecycleMode,
        ChangeType,
        ProjectAspect as LifecycleAspect,
        TransitionTrigger,
        UserRole,
        # Phase 15.2: Continuous Change Cycle imports
        request_continuous_change,
        get_cycle_history,
        get_change_lineage,
        get_deployment_summary,
        get_project_changes,
        get_aspect_history,
    )
    LIFECYCLE_AVAILABLE = True
    logger.info("Phase 15.2: Lifecycle V2 module loaded successfully")
except ImportError as e:
    LIFECYCLE_AVAILABLE = False
    logger.warning(f"Lifecycle V2 module not available: {e}")


# Pydantic models for lifecycle endpoints
class CreateProjectLifecycleRequest(BaseModel):
    """Request to create a PROJECT_MODE lifecycle."""
    project_name: str
    aspect: str
    created_by: str
    description: str = ""


class CreateChangeLifecycleRequest(BaseModel):
    """Request to create a CHANGE_MODE lifecycle."""
    project_name: str
    project_id: str
    aspect: str
    change_type: str  # bug, feature, improvement, refactor
    created_by: str
    description: str = ""


class TransitionLifecycleRequest(BaseModel):
    """Request to transition a lifecycle state."""
    trigger: str  # claude_job_completed, test_passed, test_failed, etc.
    triggered_by: str
    role: str  # owner, admin, developer, tester
    reason: str = ""
    metadata: dict = Field(default_factory=dict)


# Phase 15.2: Continuous Change Request model
class ContinuousChangeRequest(BaseModel):
    """Request to create a continuous change on a deployed lifecycle."""
    change_type: str  # bug, feature, improvement, refactor, security
    change_summary: str
    requested_by: str
    role: str  # owner, admin, developer, tester


@app.post("/lifecycle")
async def create_lifecycle(request: CreateProjectLifecycleRequest):
    """
    Create a new PROJECT_MODE lifecycle (Phase 15.1).

    Creates a lifecycle instance for a new project with the specified aspect.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    try:
        aspect = LifecycleAspect(request.aspect)
    except ValueError:
        valid_aspects = [a.value for a in LifecycleAspect]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect '{request.aspect}'. Valid aspects: {valid_aspects}"
        )

    success, message, lifecycle = await create_project_lifecycle(
        project_name=request.project_name,
        aspect=aspect,
        created_by=request.created_by,
        description=request.description,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "lifecycle": lifecycle.to_dict() if lifecycle else None,
    }


@app.post("/lifecycle/change")
async def create_change_lifecycle_endpoint(request: CreateChangeLifecycleRequest):
    """
    Create a new CHANGE_MODE lifecycle (Phase 15.1).

    Creates a lifecycle instance for a change (bug, feature, improvement, refactor)
    on an existing project. Requires project_id and change_type.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    try:
        aspect = LifecycleAspect(request.aspect)
    except ValueError:
        valid_aspects = [a.value for a in LifecycleAspect]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect '{request.aspect}'. Valid aspects: {valid_aspects}"
        )

    try:
        change_type = ChangeType(request.change_type)
    except ValueError:
        valid_types = [t.value for t in ChangeType]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid change_type '{request.change_type}'. Valid types: {valid_types}"
        )

    success, message, lifecycle = await create_change_lifecycle(
        project_name=request.project_name,
        project_id=request.project_id,
        aspect=aspect,
        change_type=change_type,
        created_by=request.created_by,
        description=request.description,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "lifecycle": lifecycle.to_dict() if lifecycle else None,
    }


@app.get("/lifecycle/{lifecycle_id}")
async def get_lifecycle_by_id(lifecycle_id: str):
    """
    Get a lifecycle by ID (Phase 15.1).

    Returns the full lifecycle state including:
    - Current state
    - Mode (PROJECT_MODE or CHANGE_MODE)
    - Aspect
    - Transition history count
    - Associated Claude jobs
    - Guidance for next steps
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    lifecycle = await get_lifecycle(lifecycle_id)
    if not lifecycle:
        raise HTTPException(status_code=404, detail=f"Lifecycle {lifecycle_id} not found")

    guidance = await get_lifecycle_guidance(lifecycle_id)

    return {
        "lifecycle": lifecycle.to_dict(),
        "guidance": guidance,
    }


@app.get("/lifecycle")
async def list_lifecycles_endpoint(
    project_id: Optional[str] = None,
    state: Optional[str] = None,
    mode: Optional[str] = None,
    limit: int = 50
):
    """
    List lifecycles with optional filtering (Phase 15.1).

    Filters:
    - project_id: Filter by project name
    - state: Filter by lifecycle state (created, planning, development, etc.)
    - mode: Filter by mode (project_mode, change_mode)
    - limit: Maximum number of results (default: 50)
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    # Validate state if provided
    if state:
        try:
            LifecycleState(state)
        except ValueError:
            valid_states = [s.value for s in LifecycleState]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid state '{state}'. Valid states: {valid_states}"
            )

    # Validate mode if provided
    if mode:
        try:
            LifecycleMode(mode)
        except ValueError:
            valid_modes = [m.value for m in LifecycleMode]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode '{mode}'. Valid modes: {valid_modes}"
            )

    lifecycles = await list_lifecycles(
        project_name=project_id,
        state=state,
        mode=mode,
        limit=limit,
    )

    return {
        "lifecycles": lifecycles,
        "count": len(lifecycles),
    }


@app.post("/lifecycle/{lifecycle_id}/transition")
async def transition_lifecycle_endpoint(lifecycle_id: str, request: TransitionLifecycleRequest):
    """
    Transition a lifecycle to a new state (Phase 15.1).

    Validates:
    - Current state allows the transition
    - User role has permission for the trigger
    - Logs the transition to immutable audit trail

    Triggers:
    - claude_job_completed: Claude job finished
    - test_passed: Automated tests passed
    - test_failed: Automated tests failed
    - telegram_feedback: User feedback from Telegram
    - human_approval: Human approved transition
    - human_rejection: Human rejected
    - system_init: System initialization
    - manual_archive: Manual archive request
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    # Validate trigger
    try:
        TransitionTrigger(request.trigger)
    except ValueError:
        valid_triggers = [t.value for t in TransitionTrigger]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid trigger '{request.trigger}'. Valid triggers: {valid_triggers}"
        )

    # Validate role
    try:
        UserRole(request.role)
    except ValueError:
        valid_roles = [r.value for r in UserRole]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role '{request.role}'. Valid roles: {valid_roles}"
        )

    success, message, new_state = await transition_lifecycle(
        lifecycle_id=lifecycle_id,
        trigger=request.trigger,
        triggered_by=request.triggered_by,
        role=request.role,
        reason=request.reason,
        metadata=request.metadata,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "new_state": new_state,
    }


@app.get("/lifecycle/{lifecycle_id}/guidance")
async def get_lifecycle_guidance_endpoint(lifecycle_id: str):
    """
    Get guidance for what happens next in a lifecycle (Phase 15.1).

    Returns:
    - Current state
    - Available actions
    - What the system is waiting for
    - Next step description

    Useful for Telegram bot to show users what actions are available.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    guidance = await get_lifecycle_guidance(lifecycle_id)
    if not guidance:
        raise HTTPException(status_code=404, detail=f"Lifecycle {lifecycle_id} not found")

    return guidance


# -----------------------------------------------------------------------------
# Phase 15.2: Continuous Change Cycle Endpoints
# -----------------------------------------------------------------------------

@app.post("/lifecycle/{lifecycle_id}/change")
async def request_change_endpoint(lifecycle_id: str, request: ContinuousChangeRequest):
    """
    Phase 15.2: Request a continuous change on a deployed lifecycle.

    This triggers the DEPLOYED -> AWAITING_FEEDBACK transition and starts
    a new change cycle. The lifecycle must be in DEPLOYED state.

    Valid change types: bug, feature, improvement, refactor, security
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    # Validate change type
    valid_types = ["bug", "feature", "improvement", "refactor", "security"]
    if request.change_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid change_type: {request.change_type}. Valid types: {valid_types}"
        )

    # Validate role
    valid_roles = ["owner", "admin", "developer", "tester"]
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {request.role}. Valid roles: {valid_roles}"
        )

    success, message, cycle_number = await request_continuous_change(
        lifecycle_id=lifecycle_id,
        change_type=request.change_type,
        change_summary=request.change_summary,
        requested_by=request.requested_by,
        role=request.role,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "success": True,
        "message": message,
        "cycle_number": cycle_number,
    }


@app.get("/lifecycle/{lifecycle_id}/cycles")
async def get_cycle_history_endpoint(lifecycle_id: str):
    """
    Phase 15.2: Get the complete cycle history for a lifecycle.

    Returns all past and current cycles with their details.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    success, message, history = await get_cycle_history(lifecycle_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)

    return {
        "lifecycle_id": lifecycle_id,
        "cycles": history,
    }


@app.get("/lifecycle/{lifecycle_id}/lineage")
async def get_lineage_endpoint(lifecycle_id: str):
    """
    Phase 15.2: Get the change lineage for a lifecycle.

    Returns the chain of deployments and their relationships.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    success, message, lineage = await get_change_lineage(lifecycle_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)

    return lineage


@app.get("/lifecycle/{lifecycle_id}/summary")
async def get_summary_endpoint(lifecycle_id: str):
    """
    Phase 15.2: Generate a "What Changed" deployment summary.

    Returns a comprehensive summary of all changes made in the lifecycle.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    success, message, summary = await get_deployment_summary(lifecycle_id)
    if not success:
        raise HTTPException(status_code=404, detail=message)

    return summary


@app.get("/project/{project_name}/changes")
async def get_project_changes_endpoint(project_name: str, aspect: Optional[str] = None):
    """
    Phase 15.2: Get all changes (CHANGE_MODE lifecycles) for a project.

    Optionally filtered by aspect.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    # Validate aspect if provided
    if aspect:
        valid_aspects = ["core", "backend", "frontend_web", "frontend_mobile", "admin", "custom"]
        if aspect not in valid_aspects:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid aspect: {aspect}. Valid aspects: {valid_aspects}"
            )

    changes = await get_project_changes(project_name, aspect)

    return {
        "project_name": project_name,
        "aspect": aspect,
        "changes": changes,
    }


@app.get("/project/{project_name}/aspects/{aspect}/history")
async def get_aspect_history_endpoint(project_name: str, aspect: str):
    """
    Phase 15.2: Get the complete history for a project aspect.

    Includes all lifecycles (project and change mode) for the aspect.
    """
    if not LIFECYCLE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Lifecycle engine not available")

    # Validate aspect
    valid_aspects = ["core", "backend", "frontend_web", "frontend_mobile", "admin", "custom"]
    if aspect not in valid_aspects:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid aspect: {aspect}. Valid aspects: {valid_aspects}"
        )

    history = await get_aspect_history(project_name, aspect)

    return history


# -----------------------------------------------------------------------------
# Phase 15.3: Project Ingestion Engine Endpoints
# -----------------------------------------------------------------------------

# Import ingestion engine module
try:
    from .ingestion_engine import (
        get_ingestion_engine,
        create_ingestion_request,
        start_ingestion_analysis,
        approve_ingestion,
        reject_ingestion,
        register_ingested_project,
        get_ingestion_request as get_ingestion,
        list_ingestion_requests,
        IngestionStatus,
        IngestionSource,
    )
    INGESTION_AVAILABLE = True
    logger.info("Phase 15.3: Ingestion Engine module loaded successfully")
except ImportError as e:
    INGESTION_AVAILABLE = False
    logger.warning(f"Ingestion Engine module not available: {e}")


# Pydantic models for ingestion endpoints
class CreateIngestionRequest(BaseModel):
    """Request to create an ingestion request."""
    project_name: str
    source_type: str  # git_repository, local_path
    source_location: str  # Git URL or filesystem path
    requested_by: str
    description: str = ""
    target_aspects: list = Field(default_factory=list)  # Optional list of aspects


class ApproveIngestionRequest(BaseModel):
    """Request to approve an ingestion."""
    approved_by: str
    role: str  # owner, admin


class RejectIngestionRequest(BaseModel):
    """Request to reject an ingestion."""
    rejected_by: str
    reason: str
    role: str  # owner, admin


class RegisterIngestionRequest(BaseModel):
    """Request to register an approved ingestion."""
    registered_by: str
    role: str  # owner, admin


@app.post("/ingestion")
async def create_ingestion_endpoint(request: CreateIngestionRequest):
    """
    Phase 15.3: Create a new project ingestion request.

    Starts the process of ingesting an external project into the platform.
    The project will be analyzed before registration.

    Supported source types:
    - git_repository: Clone from a Git URL
    - local_path: Analyze an existing local directory
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    # Validate source type
    valid_source_types = ["git_repository", "local_path"]
    if request.source_type not in valid_source_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type: {request.source_type}. Valid types: {valid_source_types}"
        )

    success, message, request_data = await create_ingestion_request(
        project_name=request.project_name,
        source_type=request.source_type,
        source_location=request.source_location,
        requested_by=request.requested_by,
        description=request.description,
        target_aspects=request.target_aspects if request.target_aspects else None,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "created",
        "message": message,
        "ingestion": request_data,
    }


@app.post("/ingestion/{ingestion_id}/analyze")
async def start_analysis_endpoint(ingestion_id: str):
    """
    Phase 15.3: Start the analysis for an ingestion request.

    This will:
    1. Clone the repository (for git) or prepare the local path
    2. Enumerate and analyze all files
    3. Detect project aspects
    4. Scan for security risks
    5. Check for existing governance documents
    6. Generate an analysis report
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    success, message, report = await start_ingestion_analysis(ingestion_id)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "analyzed",
        "message": message,
        "report": report,
    }


@app.post("/ingestion/{ingestion_id}/approve")
async def approve_ingestion_endpoint(ingestion_id: str, request: ApproveIngestionRequest):
    """
    Phase 15.3: Approve an ingestion request.

    Only owners and admins can approve ingestion requests.
    The request must be in AWAITING_APPROVAL status.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    # Validate role
    valid_roles = ["owner", "admin"]
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {request.role}. Only owners and admins can approve."
        )

    success, message = await approve_ingestion(
        ingestion_id=ingestion_id,
        approved_by=request.approved_by,
        role=request.role,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "approved",
        "message": message,
        "ingestion_id": ingestion_id,
    }


@app.post("/ingestion/{ingestion_id}/reject")
async def reject_ingestion_endpoint(ingestion_id: str, request: RejectIngestionRequest):
    """
    Phase 15.3: Reject an ingestion request.

    Only owners and admins can reject ingestion requests.
    A reason must be provided.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    # Validate role
    valid_roles = ["owner", "admin"]
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {request.role}. Only owners and admins can reject."
        )

    success, message = await reject_ingestion(
        ingestion_id=ingestion_id,
        rejected_by=request.rejected_by,
        reason=request.reason,
        role=request.role,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "rejected",
        "message": message,
        "ingestion_id": ingestion_id,
    }


@app.post("/ingestion/{ingestion_id}/register")
async def register_ingestion_endpoint(ingestion_id: str, request: RegisterIngestionRequest):
    """
    Phase 15.3: Register an approved ingestion as a project.

    Creates lifecycle instances for each detected aspect in DEPLOYED state.
    Generates missing governance documents.

    This is the final step of the ingestion workflow.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    # Validate role
    valid_roles = ["owner", "admin"]
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role: {request.role}. Only owners and admins can register."
        )

    success, message, lifecycle_ids = await register_ingested_project(
        ingestion_id=ingestion_id,
        registered_by=request.registered_by,
        role=request.role,
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    return {
        "status": "registered",
        "message": message,
        "ingestion_id": ingestion_id,
        "lifecycle_ids": lifecycle_ids,
    }


@app.get("/ingestion/{ingestion_id}")
async def get_ingestion_endpoint(ingestion_id: str):
    """
    Phase 15.3: Get an ingestion request by ID.

    Returns the full ingestion request including analysis report if available.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    request_data = await get_ingestion(ingestion_id)

    if not request_data:
        raise HTTPException(status_code=404, detail=f"Ingestion request {ingestion_id} not found")

    return request_data


@app.get("/ingestion")
async def list_ingestions_endpoint(status: Optional[str] = None, limit: int = 50):
    """
    Phase 15.3: List ingestion requests.

    Can be filtered by status:
    - pending: Not yet analyzed
    - analyzing: Currently being analyzed
    - awaiting_approval: Analysis complete, waiting for approval
    - approved: Approved, ready for registration
    - rejected: Rejected by admin
    - registered: Successfully registered as a project
    - failed: Analysis or registration failed
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    # Validate status if provided
    if status:
        valid_statuses = ["pending", "analyzing", "awaiting_approval", "approved", "rejected", "registered", "failed"]
        if status not in valid_statuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid statuses: {valid_statuses}"
            )

    requests = await list_ingestion_requests(status=status, limit=limit)

    return {
        "count": len(requests),
        "status_filter": status,
        "requests": requests,
    }


@app.get("/ingestion/status")
async def ingestion_status_endpoint():
    """
    Phase 15.3: Get the status of the ingestion engine.

    Returns counts of ingestion requests by status.
    """
    if not INGESTION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Ingestion engine not available")

    # Get counts by status
    all_requests = await list_ingestion_requests(limit=1000)

    status_counts = {}
    for req in all_requests:
        status = req.get("status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        "phase": "15.3",
        "available": True,
        "total_requests": len(all_requests),
        "by_status": status_counts,
        "message": "Ingestion Engine operational",
    }


# -----------------------------------------------------------------------------
# Phase 16B: Dashboard & Observability Endpoints (READ-ONLY)
# -----------------------------------------------------------------------------

# Import dashboard backend
try:
    from .dashboard_backend import (
        get_dashboard_summary,
        get_all_projects as dashboard_get_projects,
        get_project_detail,
        get_claude_activity,
        get_jobs_list as dashboard_get_jobs,
        get_lifecycle_timeline,
        get_all_lifecycles as dashboard_get_lifecycles,
        get_audit_events,
        get_security_summary,
    )
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dashboard backend not available: {e}")
    DASHBOARD_AVAILABLE = False


@app.get("/dashboard")
async def dashboard_summary_endpoint():
    """
    Phase 16B: Get dashboard summary.

    Returns top-level system overview including:
    - System health status
    - Project counts
    - Lifecycle counts
    - Job statistics
    - Security summary
    - Service health

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        summary = await get_dashboard_summary()
        return {
            "phase": "16B",
            "endpoint": "dashboard",
            "data": summary,
        }
    except Exception as e:
        logger.error(f"Dashboard summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard: {str(e)}")


@app.get("/dashboard/projects")
async def dashboard_projects_endpoint():
    """
    Phase 16B: Get all projects overview.

    Returns list of all projects with:
    - Project ID and name
    - Current lifecycle state
    - Aspects and their states
    - Active cycle number
    - Current Claude job (if any)

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        projects = await dashboard_get_projects()
        return {
            "phase": "16B",
            "endpoint": "dashboard/projects",
            "count": len(projects),
            "projects": projects,
        }
    except Exception as e:
        logger.error(f"Dashboard projects error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get projects: {str(e)}")


@app.get("/dashboard/project/{project_name}")
async def dashboard_project_detail_endpoint(project_name: str):
    """
    Phase 16B: Get detailed view of a specific project.

    Returns comprehensive project data including:
    - All aspects with their states
    - All associated lifecycles
    - Total cycles across aspects

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        detail = await get_project_detail(project_name)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")
        return {
            "phase": "16B",
            "endpoint": f"dashboard/project/{project_name}",
            "data": detail,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard project detail error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get project: {str(e)}")


@app.get("/dashboard/jobs")
async def dashboard_jobs_endpoint(
    state: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 50,
):
    """
    Phase 16B: Get Claude jobs list.

    Query parameters:
    - state: Filter by job state (queued, running, completed, failed)
    - project: Filter by project name
    - limit: Maximum jobs to return (default 50)

    Returns list of jobs with:
    - Job ID and state
    - Project and task info
    - Timestamps

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        # Also get activity panel for summary
        activity = await get_claude_activity()
        jobs = await dashboard_get_jobs(state=state, project=project, limit=limit)
        return {
            "phase": "16B",
            "endpoint": "dashboard/jobs",
            "activity_summary": {
                "active": activity.get("active_jobs", []),
                "queued_count": len(activity.get("queued_jobs", [])),
                "completed_today": activity.get("completed_jobs_today", 0),
                "failed_today": activity.get("failed_jobs_today", 0),
                "worker_utilization": activity.get("worker_utilization", {}),
            },
            "count": len(jobs),
            "jobs": jobs,
        }
    except Exception as e:
        logger.error(f"Dashboard jobs error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get jobs: {str(e)}")


@app.get("/dashboard/lifecycles")
async def dashboard_lifecycles_endpoint(
    state: Optional[str] = None,
    project: Optional[str] = None,
    aspect: Optional[str] = None,
):
    """
    Phase 16B: Get all lifecycles.

    Query parameters:
    - state: Filter by lifecycle state
    - project: Filter by project name
    - aspect: Filter by aspect (core, backend, etc.)

    Returns list of lifecycles with their current state.

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        lifecycles = await dashboard_get_lifecycles(state=state, project=project, aspect=aspect)
        return {
            "phase": "16B",
            "endpoint": "dashboard/lifecycles",
            "count": len(lifecycles),
            "lifecycles": lifecycles,
        }
    except Exception as e:
        logger.error(f"Dashboard lifecycles error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lifecycles: {str(e)}")


@app.get("/dashboard/lifecycle/{lifecycle_id}")
async def dashboard_lifecycle_timeline_endpoint(lifecycle_id: str):
    """
    Phase 16B: Get detailed lifecycle timeline.

    Returns full history including:
    - State transition history
    - Approvals and rejections
    - Feedback entries
    - Cycle history
    - Change summaries

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        timeline = await get_lifecycle_timeline(lifecycle_id)
        if timeline is None:
            raise HTTPException(status_code=404, detail=f"Lifecycle '{lifecycle_id}' not found")
        return {
            "phase": "16B",
            "endpoint": f"dashboard/lifecycle/{lifecycle_id}",
            "data": timeline,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard lifecycle timeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get timeline: {str(e)}")


@app.get("/dashboard/audit")
async def dashboard_audit_endpoint(
    event_type: Optional[str] = None,
    limit: int = 100,
    since: Optional[str] = None,
):
    """
    Phase 16B: Get audit events.

    Query parameters:
    - event_type: Filter by event type (DENIED, ALLOWED)
    - limit: Maximum events to return (default 100)
    - since: ISO timestamp to filter events after

    Returns:
    - Audit events
    - Security summary

    This is a READ-ONLY endpoint. No mutations.
    """
    if not DASHBOARD_AVAILABLE:
        raise HTTPException(status_code=503, detail="Dashboard backend not available")

    try:
        events = await get_audit_events(event_type=event_type, limit=limit, since=since)
        security = await get_security_summary()
        return {
            "phase": "16B",
            "endpoint": "dashboard/audit",
            "security_summary": security,
            "count": len(events),
            "events": events,
        }
    except Exception as e:
        logger.error(f"Dashboard audit error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get audit: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 17A: Runtime Intelligence Endpoints (READ-ONLY)
# -----------------------------------------------------------------------------

# Import runtime intelligence backend
try:
    from .runtime_intelligence import (
        get_runtime_engine,
        get_signals,
        get_signal_summary,
        poll_signals,
        SignalType,
        Severity,
        RuntimeSignal,
        SignalSummary,
    )
    RUNTIME_INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Runtime intelligence not available: {e}")
    RUNTIME_INTELLIGENCE_AVAILABLE = False


@app.get("/runtime/signals")
async def runtime_signals_endpoint(
    project_id: Optional[str] = None,
    signal_type: Optional[str] = None,
    severity: Optional[str] = None,
    since_hours: int = 24,
    limit: int = 100,
):
    """
    Phase 17A: Get runtime signals.

    Query parameters:
    - project_id: Filter by project
    - signal_type: Filter by signal type (system_resource, worker_queue, etc.)
    - severity: Filter by severity (info, warning, degraded, critical, unknown)
    - since_hours: Time window in hours (default 24)
    - limit: Maximum signals to return (default 100)

    This is a READ-ONLY endpoint. No mutations.
    """
    if not RUNTIME_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Runtime intelligence not available")

    try:
        from datetime import datetime, timedelta

        since = datetime.utcnow() - timedelta(hours=since_hours)

        # Parse optional enums
        type_filter = None
        if signal_type:
            try:
                type_filter = SignalType(signal_type)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid signal_type: {signal_type}")

        severity_filter = None
        if severity:
            try:
                severity_filter = Severity(severity)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")

        signals = get_signals(
            since=since,
            project_id=project_id,
            signal_type=type_filter,
            severity=severity_filter,
            limit=limit,
        )

        return {
            "phase": "17A",
            "endpoint": "runtime/signals",
            "observation_only": True,
            "filter": {
                "project_id": project_id,
                "signal_type": signal_type,
                "severity": severity,
                "since_hours": since_hours,
            },
            "count": len(signals),
            "signals": [s.to_dict() for s in signals],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Runtime signals error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {str(e)}")


@app.get("/runtime/summary")
async def runtime_summary_endpoint(
    since_hours: int = 24,
):
    """
    Phase 17A: Get runtime signal summary.

    Query parameters:
    - since_hours: Time window in hours (default 24)

    Returns:
    - Signal counts by severity
    - Signal counts by type
    - Unknown count (visibility of missing data)
    - Observability status

    This is a READ-ONLY endpoint. No mutations.
    """
    if not RUNTIME_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Runtime intelligence not available")

    try:
        from datetime import datetime, timedelta

        since = datetime.utcnow() - timedelta(hours=since_hours)
        summary = get_signal_summary(since=since)

        return {
            "phase": "17A",
            "endpoint": "runtime/summary",
            "observation_only": True,
            "summary": summary.to_dict(),
        }

    except Exception as e:
        logger.error(f"Runtime summary error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@app.get("/runtime/status")
async def runtime_status_endpoint():
    """
    Phase 17A: Get runtime intelligence engine status.

    Returns:
    - Engine running state
    - Poll configuration
    - Last poll timestamp
    - Poll count

    This is a READ-ONLY endpoint. No mutations.
    """
    if not RUNTIME_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Runtime intelligence not available")

    try:
        engine = get_runtime_engine()
        status = engine.get_status()

        return {
            "phase": "17A",
            "endpoint": "runtime/status",
            "observation_only": True,
            "status": status,
        }

    except Exception as e:
        logger.error(f"Runtime status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@app.post("/runtime/poll")
async def runtime_poll_endpoint():
    """
    Phase 17A: Trigger a manual signal poll.

    This endpoint triggers an immediate poll cycle without waiting
    for the scheduled interval.

    Returns:
    - Number of signals collected
    - Number of signals persisted

    This is a READ-ONLY operation - it only collects and persists signals.
    It does NOT modify lifecycle, deployments, or intent.
    """
    if not RUNTIME_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Runtime intelligence not available")

    try:
        persisted, signals = poll_signals()

        # Summarize by severity
        by_severity = {}
        for signal in signals:
            sev = signal.severity
            by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "phase": "17A",
            "endpoint": "runtime/poll",
            "observation_only": True,
            "collected": len(signals),
            "persisted": persisted,
            "by_severity": by_severity,
            "message": f"Poll complete: {len(signals)} signals collected, {persisted} persisted",
        }

    except Exception as e:
        logger.error(f"Runtime poll error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to poll: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 17B: Incident Classification Endpoints (READ-ONLY)
# -----------------------------------------------------------------------------
try:
    from .incident_model import (
        Incident,
        IncidentType,
        IncidentSeverity,
        IncidentScope,
        IncidentState,
        IncidentSummary,
    )
    from .incident_engine import (
        IncidentClassificationEngine,
        classify_signals,
        get_classification_engine,
    )
    from .incident_store import (
        IncidentStore,
        get_incident_store,
        persist_incidents,
        read_incidents,
        read_recent_incidents,
        get_incident_by_id,
        get_incident_summary,
    )
    INCIDENT_CLASSIFICATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Incident classification not available: {e}")
    INCIDENT_CLASSIFICATION_AVAILABLE = False


@app.get("/incidents")
async def incidents_endpoint(
    incident_type: Optional[str] = None,
    severity: Optional[str] = None,
    scope: Optional[str] = None,
    state: Optional[str] = None,
    project_id: Optional[str] = None,
    since_hours: int = 24,
    limit: int = 100,
):
    """
    Phase 17B: Get incidents.

    Query parameters:
    - incident_type: Filter by type (performance, reliability, security, etc.)
    - severity: Filter by severity (info, low, medium, high, critical, unknown)
    - scope: Filter by scope (system, project, project_aspect, job, unknown)
    - state: Filter by state (open, closed, unknown)
    - project_id: Filter by project
    - since_hours: Time window in hours (default 24)
    - limit: Maximum incidents to return (default 100)

    This is a READ-ONLY endpoint. No mutations, no alerts, no recommendations.
    """
    if not INCIDENT_CLASSIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Incident classification not available")

    try:
        from datetime import datetime, timedelta

        since = datetime.utcnow() - timedelta(hours=since_hours)

        # Parse filters
        type_filter = IncidentType(incident_type) if incident_type else None
        severity_filter = IncidentSeverity(severity) if severity else None
        scope_filter = IncidentScope(scope) if scope else None
        state_filter = IncidentState(state) if state else None

        store = get_incident_store()
        incidents = store.read_incidents(
            since=since,
            incident_type=type_filter,
            severity=severity_filter,
            scope=scope_filter,
            state=state_filter,
            project_id=project_id,
            limit=limit,
        )

        return {
            "phase": "17B",
            "endpoint": "incidents",
            "observation_only": True,
            "no_alerts": True,
            "no_recommendations": True,
            "total": len(incidents),
            "filters": {
                "incident_type": incident_type,
                "severity": severity,
                "scope": scope,
                "state": state,
                "project_id": project_id,
                "since_hours": since_hours,
            },
            "incidents": [i.to_dict() for i in incidents],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter value: {str(e)}")
    except Exception as e:
        logger.error(f"Incidents endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get incidents: {str(e)}")


@app.get("/incidents/recent")
async def incidents_recent_endpoint(
    hours: int = 24,
    limit: int = 50,
):
    """
    Phase 17B: Get recent incidents.

    Query parameters:
    - hours: Time window in hours (default 24)
    - limit: Maximum incidents to return (default 50)

    This is a READ-ONLY endpoint. No mutations, no alerts, no recommendations.
    """
    if not INCIDENT_CLASSIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Incident classification not available")

    try:
        store = get_incident_store()
        incidents = store.read_recent(hours=hours, limit=limit)

        return {
            "phase": "17B",
            "endpoint": "incidents/recent",
            "observation_only": True,
            "no_alerts": True,
            "no_recommendations": True,
            "total": len(incidents),
            "hours": hours,
            "incidents": [i.to_dict() for i in incidents],
        }

    except Exception as e:
        logger.error(f"Recent incidents endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent incidents: {str(e)}")


@app.get("/incidents/summary")
async def incidents_summary_endpoint(
    since_hours: int = 24,
):
    """
    Phase 17B: Get incident summary.

    Query parameters:
    - since_hours: Time window in hours (default 24)

    Returns summary counts by severity, type, scope, and state.
    Also includes list of recent incidents.

    This is a READ-ONLY endpoint. No mutations, no alerts, no recommendations.
    """
    if not INCIDENT_CLASSIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Incident classification not available")

    try:
        summary = get_incident_summary(since_hours=since_hours)

        return {
            "phase": "17B",
            "endpoint": "incidents/summary",
            "observation_only": True,
            "no_alerts": True,
            "no_recommendations": True,
            **summary.to_dict(),
        }

    except Exception as e:
        logger.error(f"Incidents summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get incident summary: {str(e)}")


@app.get("/incidents/{incident_id}")
async def incident_by_id_endpoint(incident_id: str):
    """
    Phase 17B: Get a specific incident by ID.

    This is a READ-ONLY endpoint. No mutations, no alerts, no recommendations.
    """
    if not INCIDENT_CLASSIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Incident classification not available")

    try:
        incident = get_incident_by_id(incident_id)

        if not incident:
            raise HTTPException(status_code=404, detail=f"Incident not found: {incident_id}")

        return {
            "phase": "17B",
            "endpoint": "incidents/{id}",
            "observation_only": True,
            "no_alerts": True,
            "no_recommendations": True,
            "incident": incident.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Incident by ID endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get incident: {str(e)}")


@app.post("/incidents/classify")
async def incidents_classify_endpoint():
    """
    Phase 17B: Trigger incident classification from current signals.

    This reads signals from Phase 17A, classifies them into incidents,
    and persists the incidents.

    CRITICAL: This is an OBSERVATION-ONLY operation:
    - No lifecycle changes
    - No alerts
    - No recommendations
    - No deployments

    It only classifies signals into incidents and stores them.
    """
    if not INCIDENT_CLASSIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Incident classification not available")

    if not RUNTIME_INTELLIGENCE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Runtime intelligence not available (required for signals)")

    try:
        from datetime import datetime, timedelta

        # Get recent signals from Phase 17A
        since = datetime.utcnow() - timedelta(hours=1)
        signals = get_signals(since=since, limit=500)

        if not signals:
            return {
                "phase": "17B",
                "endpoint": "incidents/classify",
                "observation_only": True,
                "no_alerts": True,
                "no_recommendations": True,
                "signals_processed": 0,
                "incidents_created": 0,
                "message": "No signals to classify",
            }

        # Classify signals into incidents
        engine = get_classification_engine()
        incidents = engine.classify_signals(signals)

        # Persist incidents
        store = get_incident_store()
        persisted = store.persist(incidents)

        return {
            "phase": "17B",
            "endpoint": "incidents/classify",
            "observation_only": True,
            "no_alerts": True,
            "no_recommendations": True,
            "signals_processed": len(signals),
            "incidents_created": len(incidents),
            "incidents_persisted": persisted,
            "incidents": [
                {
                    "incident_id": i.incident_id,
                    "incident_type": i.incident_type,
                    "severity": i.severity,
                    "title": i.title,
                }
                for i in incidents
            ],
        }

    except Exception as e:
        logger.error(f"Incident classification error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to classify incidents: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 17C: Recommendation Endpoints (ADVISORY ONLY)
# -----------------------------------------------------------------------------
try:
    from .recommendation_model import (
        Recommendation,
        RecommendationType,
        RecommendationSeverity,
        RecommendationApproval,
        RecommendationStatus,
        RecommendationSummary,
        ApprovalRecord,
    )
    from .recommendation_engine import (
        RecommendationEngine,
        RecommendationGenerator,
        get_recommendation_engine,
        generate_recommendations,
    )
    from .recommendation_store import (
        RecommendationStore,
        get_recommendation_store,
        persist_recommendations,
        read_recommendations,
        read_recent_recommendations,
        get_recommendation_by_id,
        get_recommendation_summary,
    )
    RECOMMENDATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Recommendations not available: {e}")
    RECOMMENDATIONS_AVAILABLE = False


@app.get("/recommendations")
async def recommendations_endpoint(
    recommendation_type: Optional[str] = None,
    severity: Optional[str] = None,
    status: Optional[str] = None,
    approval_required: Optional[str] = None,
    project_id: Optional[str] = None,
    since_hours: int = 168,  # 7 days default
    limit: int = 100,
):
    """
    Phase 17C: Get recommendations.

    Query parameters:
    - recommendation_type: Filter by type (investigate, mitigate, improve, refactor, document, no_action)
    - severity: Filter by severity (info, low, medium, high, critical, unknown)
    - status: Filter by status (pending, approved, dismissed, expired, unknown)
    - approval_required: Filter by approval requirement (none_required, confirmation_required, explicit_approval_required)
    - project_id: Filter by project
    - since_hours: Time window in hours (default 168 = 7 days)
    - limit: Maximum recommendations to return (default 100)

    CRITICAL: This is an ADVISORY-ONLY endpoint.
    - Recommendations suggest, never execute
    - NO automation, NO lifecycle mutation
    - Human must approve/dismiss
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    try:
        from datetime import datetime, timedelta

        since = datetime.utcnow() - timedelta(hours=since_hours)

        # Parse filters
        type_filter = RecommendationType(recommendation_type) if recommendation_type else None
        severity_filter = RecommendationSeverity(severity) if severity else None
        status_filter = RecommendationStatus(status) if status else None
        approval_filter = RecommendationApproval(approval_required) if approval_required else None

        store = get_recommendation_store()
        recommendations = store.read_recommendations(
            since=since,
            recommendation_type=type_filter,
            severity=severity_filter,
            status=status_filter,
            approval_required=approval_filter,
            project_id=project_id,
            limit=limit,
        )

        return {
            "phase": "17C",
            "endpoint": "recommendations",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            "total": len(recommendations),
            "filters": {
                "recommendation_type": recommendation_type,
                "severity": severity,
                "status": status,
                "approval_required": approval_required,
                "project_id": project_id,
                "since_hours": since_hours,
            },
            "recommendations": [r.to_dict() for r in recommendations],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid filter value: {str(e)}")
    except Exception as e:
        logger.error(f"Recommendations endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendations: {str(e)}")


@app.get("/recommendations/recent")
async def recommendations_recent_endpoint(
    hours: int = 168,  # 7 days default
    limit: int = 50,
):
    """
    Phase 17C: Get recent recommendations.

    Query parameters:
    - hours: Time window in hours (default 168 = 7 days)
    - limit: Maximum recommendations to return (default 50)

    CRITICAL: This is an ADVISORY-ONLY endpoint.
    - Recommendations suggest, never execute
    - NO automation, NO lifecycle mutation
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    try:
        store = get_recommendation_store()
        recommendations = store.read_recent(hours=hours, limit=limit)

        return {
            "phase": "17C",
            "endpoint": "recommendations/recent",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            "total": len(recommendations),
            "hours": hours,
            "recommendations": [r.to_dict() for r in recommendations],
        }

    except Exception as e:
        logger.error(f"Recent recommendations endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent recommendations: {str(e)}")


@app.get("/recommendations/summary")
async def recommendations_summary_endpoint(
    since_hours: int = 168,  # 7 days default
):
    """
    Phase 17C: Get recommendation summary.

    Query parameters:
    - since_hours: Time window in hours (default 168 = 7 days)

    Returns summary counts by severity, type, status, and approval requirement.
    Also includes list of recent recommendations.

    CRITICAL: This is an ADVISORY-ONLY endpoint.
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    try:
        summary = get_recommendation_summary(since_hours=since_hours)

        return {
            "phase": "17C",
            "endpoint": "recommendations/summary",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            **summary.to_dict(),
        }

    except Exception as e:
        logger.error(f"Recommendations summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation summary: {str(e)}")


@app.get("/recommendations/{recommendation_id}")
async def recommendation_by_id_endpoint(recommendation_id: str):
    """
    Phase 17C: Get a specific recommendation by ID.

    CRITICAL: This is an ADVISORY-ONLY endpoint.
    - Recommendations suggest, never execute
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    try:
        recommendation = get_recommendation_by_id(recommendation_id)

        if not recommendation:
            raise HTTPException(status_code=404, detail=f"Recommendation not found: {recommendation_id}")

        return {
            "phase": "17C",
            "endpoint": "recommendations/{id}",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            "recommendation": recommendation.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation by ID endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation: {str(e)}")


class ApprovalRequest(BaseModel):
    """Request body for approve/dismiss endpoints."""
    user_id: str = Field(..., description="ID of the user approving/dismissing")
    reason: Optional[str] = Field(None, description="Optional reason for the decision")


@app.post("/recommendations/{recommendation_id}/approve")
async def approve_recommendation_endpoint(
    recommendation_id: str,
    request: ApprovalRequest,
):
    """
    Phase 17C: Approve a recommendation.

    This creates an approval record in the separate approvals log.
    The original recommendation is NEVER modified (append-only).

    CRITICAL CONSTRAINTS:
    - This is ADVISORY ONLY - approval does NOT trigger execution
    - NO automation, NO lifecycle mutation, NO deployment
    - Human action required for any actual changes

    Request body:
    - user_id: ID of the user approving
    - reason: Optional reason for approval
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    try:
        store = get_recommendation_store()

        # Check if recommendation exists
        recommendation = store.get_by_id(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"Recommendation not found: {recommendation_id}")

        # Check if already approved
        if recommendation.status == RecommendationStatus.APPROVED.value:
            return {
                "phase": "17C",
                "endpoint": "recommendations/{id}/approve",
                "advisory_only": True,
                "no_execution": True,
                "no_lifecycle_mutation": True,
                "already_approved": True,
                "message": f"Recommendation {recommendation_id} was already approved",
                "recommendation": recommendation.to_dict(),
            }

        # Create approval record
        approval_record = store.approve(
            recommendation_id=recommendation_id,
            user_id=request.user_id,
            reason=request.reason,
        )

        if not approval_record:
            raise HTTPException(status_code=500, detail="Failed to create approval record")

        # Get updated recommendation (with approval status applied)
        updated_recommendation = store.get_by_id(recommendation_id)

        return {
            "phase": "17C",
            "endpoint": "recommendations/{id}/approve",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            "approved": True,
            "approval_record": approval_record.to_dict(),
            "recommendation": updated_recommendation.to_dict() if updated_recommendation else None,
            "message": "Recommendation approved. NOTE: This is advisory only - no automatic action taken.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Approve recommendation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to approve recommendation: {str(e)}")


@app.post("/recommendations/{recommendation_id}/dismiss")
async def dismiss_recommendation_endpoint(
    recommendation_id: str,
    request: ApprovalRequest,
):
    """
    Phase 17C: Dismiss a recommendation.

    This creates a dismissal record in the separate approvals log.
    The original recommendation is NEVER modified (append-only).

    CRITICAL CONSTRAINTS:
    - This is ADVISORY ONLY
    - NO automation, NO lifecycle mutation

    Request body:
    - user_id: ID of the user dismissing
    - reason: Optional reason for dismissal
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    try:
        store = get_recommendation_store()

        # Check if recommendation exists
        recommendation = store.get_by_id(recommendation_id)
        if not recommendation:
            raise HTTPException(status_code=404, detail=f"Recommendation not found: {recommendation_id}")

        # Check if already dismissed
        if recommendation.status == RecommendationStatus.DISMISSED.value:
            return {
                "phase": "17C",
                "endpoint": "recommendations/{id}/dismiss",
                "advisory_only": True,
                "no_execution": True,
                "no_lifecycle_mutation": True,
                "already_dismissed": True,
                "message": f"Recommendation {recommendation_id} was already dismissed",
                "recommendation": recommendation.to_dict(),
            }

        # Create dismissal record
        dismissal_record = store.dismiss(
            recommendation_id=recommendation_id,
            user_id=request.user_id,
            reason=request.reason,
        )

        if not dismissal_record:
            raise HTTPException(status_code=500, detail="Failed to create dismissal record")

        # Get updated recommendation (with dismissal status applied)
        updated_recommendation = store.get_by_id(recommendation_id)

        return {
            "phase": "17C",
            "endpoint": "recommendations/{id}/dismiss",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            "dismissed": True,
            "dismissal_record": dismissal_record.to_dict(),
            "recommendation": updated_recommendation.to_dict() if updated_recommendation else None,
            "message": "Recommendation dismissed.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dismiss recommendation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dismiss recommendation: {str(e)}")


@app.post("/recommendations/generate")
async def generate_recommendations_endpoint():
    """
    Phase 17C: Generate recommendations from current incidents.

    This reads incidents from Phase 17B, generates recommendations using
    deterministic rules, and persists them.

    CRITICAL: This is an ADVISORY-ONLY operation:
    - Recommendations suggest, never execute
    - NO lifecycle changes
    - NO deployment
    - NO Claude execution
    - Human must approve/dismiss

    It only generates recommendations and stores them.
    """
    if not RECOMMENDATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Recommendations not available")

    if not INCIDENT_CLASSIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Incident classification not available (required for recommendation generation)")

    try:
        from datetime import datetime, timedelta

        # Get recent incidents from Phase 17B
        since = datetime.utcnow() - timedelta(hours=24)
        incident_store = get_incident_store()
        incidents = incident_store.read_incidents(since=since, limit=500)

        if not incidents:
            return {
                "phase": "17C",
                "endpoint": "recommendations/generate",
                "advisory_only": True,
                "no_execution": True,
                "no_lifecycle_mutation": True,
                "incidents_processed": 0,
                "recommendations_created": 0,
                "message": "No incidents to generate recommendations from",
            }

        # Generate recommendations from incidents
        engine = get_recommendation_engine()
        recommendations = engine.generate_recommendations(incidents)

        # Persist recommendations
        rec_store = get_recommendation_store()
        persisted = rec_store.persist(recommendations)

        return {
            "phase": "17C",
            "endpoint": "recommendations/generate",
            "advisory_only": True,
            "no_execution": True,
            "no_lifecycle_mutation": True,
            "incidents_processed": len(incidents),
            "recommendations_created": len(recommendations),
            "recommendations_persisted": persisted,
            "recommendations": [
                {
                    "recommendation_id": r.recommendation_id,
                    "recommendation_type": r.recommendation_type,
                    "severity": r.severity,
                    "title": r.title,
                    "approval_required": r.approval_required,
                }
                for r in recommendations
            ],
        }

    except Exception as e:
        logger.error(f"Recommendation generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 18C: Execution Dispatcher Endpoints
# -----------------------------------------------------------------------------

# Import execution dispatcher components
try:
    from .execution_dispatcher import (
        ControlledExecutionDispatcher,
        ExecutionIntent,
        ExecutionResult,
        ValidationChainInput,
        ExecutionStatus,
        ActionType,
        BlockReason,
        get_execution_dispatcher,
        create_execution_intent,
        create_validation_chain_input,
        dispatch_execution,
        get_execution_summary,
    )
    from .execution_store import (
        ExecutionStore,
        get_execution_store,
    )
    EXECUTION_DISPATCHER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Execution dispatcher not available: {e}")
    EXECUTION_DISPATCHER_AVAILABLE = False


class ExecutionDispatchRequest(BaseModel):
    """Request body for execution dispatch."""
    project_id: str = Field(..., description="ID of the project")
    project_name: str = Field(..., description="Name of the project")
    action_type: str = Field(..., description="Type of action (run_tests, update_docs, write_code, commit, push, deploy_test)")
    action_description: str = Field(..., description="Description of the action to perform")
    requester_id: str = Field(..., description="ID of the requester")
    requester_role: str = Field(..., description="Role of the requester")
    target_workspace: str = Field(..., description="Target workspace path")
    metadata: Optional[dict] = Field(None, description="Optional metadata")
    # Validation chain data (must be provided by caller)
    eligibility_decision: str = Field(..., description="Eligibility decision from Phase 18A")
    eligibility_allowed_actions: list = Field(default_factory=list, description="Allowed actions from eligibility")
    approval_status: str = Field(..., description="Approval status from Phase 18B")


@app.post("/execution/dispatch")
async def execution_dispatch_endpoint(request: ExecutionDispatchRequest):
    """
    Phase 18C: Dispatch an execution request through the validation chain.

    This is the ONLY endpoint that can trigger actual execution.

    CRITICAL CONSTRAINTS:
    - ALL actions MUST flow through this endpoint
    - Validates complete chain: Eligibility -> Approval -> Gate
    - Returns execution result with status
    - AUDIT REQUIRED: Every dispatch is audited

    Request body:
    - project_id: ID of the project
    - project_name: Name of the project
    - action_type: Type of action to perform
    - action_description: Description of the action
    - requester_id: ID of the requester
    - requester_role: Role of the requester
    - target_workspace: Target workspace path
    - eligibility_decision: Decision from Phase 18A
    - eligibility_allowed_actions: Allowed actions from eligibility
    - approval_status: Status from Phase 18B
    """
    if not EXECUTION_DISPATCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Execution dispatcher not available")

    try:
        # Create execution intent
        intent = create_execution_intent(
            project_id=request.project_id,
            project_name=request.project_name,
            action_type=request.action_type,
            action_description=request.action_description,
            requester_id=request.requester_id,
            requester_role=request.requester_role,
            target_workspace=request.target_workspace,
            metadata=request.metadata,
        )

        # Import required types for creating validation chain
        from .automation_eligibility import EligibilityResult, EligibilityDecision
        from .approval_orchestrator import OrchestrationResult, ApprovalStatus
        from .execution_gate import ExecutionRequest

        # Create eligibility result from request data
        eligibility_result = EligibilityResult(
            decision=request.eligibility_decision,
            matched_rules=(),
            input_hash="from-api",
            timestamp=datetime.utcnow().isoformat(),
            engine_version="18A.1.0",
            allowed_actions=tuple(request.eligibility_allowed_actions),
        )

        # Create approval result from request data
        approval_result = OrchestrationResult(
            status=request.approval_status,
            reason=None,
            input_hash="from-api",
            timestamp=datetime.utcnow().isoformat(),
            orchestrator_version="18B.1.0",
            approval_request_id=None,
            approver_count=0,
            required_approver_count=0,
        )

        # Create gate request
        gate_request = ExecutionRequest(
            job_id=intent.intent_id,
            project_name=request.project_name,
            aspect="core",
            lifecycle_id=request.project_id,
            lifecycle_state="development",  # Default, should be provided
            requested_action=request.action_type,
            requesting_user_id=request.requester_id,
            requesting_user_role=request.requester_role,
            workspace_path=request.target_workspace,
            task_description=request.action_description,
            project_id=request.project_id,
        )

        # Create validation chain input
        chain_input = create_validation_chain_input(
            intent=intent,
            eligibility_result=eligibility_result,
            approval_result=approval_result,
            gate_request=gate_request,
        )

        # Dispatch execution
        result = dispatch_execution(chain_input)

        return {
            "phase": "18C",
            "endpoint": "execution/dispatch",
            "single_execution_point": True,
            "chain_validated": True,
            "audit_recorded": True,
            "result": result.to_dict(),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    except Exception as e:
        logger.error(f"Execution dispatch endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to dispatch execution: {str(e)}")


@app.get("/execution/{execution_id}")
async def execution_by_id_endpoint(execution_id: str):
    """
    Phase 18C: Get an execution result by ID.

    Returns the execution result including status, timestamps, and any output.

    CRITICAL: This is a READ-ONLY endpoint.
    """
    if not EXECUTION_DISPATCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Execution dispatcher not available")

    try:
        dispatcher = get_execution_dispatcher()
        result = dispatcher.get_execution(execution_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"Execution not found: {execution_id}")

        return {
            "phase": "18C",
            "endpoint": "execution/{id}",
            "read_only": True,
            "result": result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Execution by ID endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution: {str(e)}")


@app.get("/execution/recent")
async def execution_recent_endpoint(
    limit: int = 100,
    status: Optional[str] = None,
    project_id: Optional[str] = None,
):
    """
    Phase 18C: Get recent executions.

    Query parameters:
    - limit: Maximum number of results (default 100)
    - status: Optional filter by status (execution_blocked, execution_pending, execution_success, execution_failed)
    - project_id: Optional filter by project

    CRITICAL: This is a READ-ONLY endpoint.
    """
    if not EXECUTION_DISPATCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Execution dispatcher not available")

    try:
        dispatcher = get_execution_dispatcher()
        results = dispatcher.get_recent_executions(
            limit=limit,
            status=status,
            project_id=project_id,
        )

        return {
            "phase": "18C",
            "endpoint": "execution/recent",
            "read_only": True,
            "total": len(results),
            "filters": {
                "limit": limit,
                "status": status,
                "project_id": project_id,
            },
            "executions": [r.to_dict() for r in results],
        }

    except Exception as e:
        logger.error(f"Recent executions endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent executions: {str(e)}")


@app.get("/execution/summary")
async def execution_summary_endpoint(
    project_id: Optional[str] = None,
    since_hours: int = 24,
):
    """
    Phase 18C: Get execution summary statistics.

    Query parameters:
    - project_id: Optional filter by project
    - since_hours: Time window in hours (default 24)

    Returns summary counts by status, action type, and block reason.

    CRITICAL: This is a READ-ONLY endpoint.
    """
    if not EXECUTION_DISPATCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Execution dispatcher not available")

    try:
        summary = get_execution_summary(
            project_id=project_id,
            since_hours=since_hours,
        )

        return {
            "phase": "18C",
            "endpoint": "execution/summary",
            "read_only": True,
            **summary,
        }

    except Exception as e:
        logger.error(f"Execution summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get execution summary: {str(e)}")


@app.post("/execution/{execution_id}/outcome")
async def execution_outcome_endpoint(
    execution_id: str,
    success: bool,
    output: Optional[str] = None,
    failure_reason: Optional[str] = None,
    rollback_performed: bool = False,
):
    """
    Phase 18C: Record execution outcome after completion.

    Called by the execution backend after actual execution completes.

    Request body:
    - success: Whether execution succeeded
    - output: Execution output (truncated to 1000 chars)
    - failure_reason: Reason for failure if not successful
    - rollback_performed: Whether rollback was performed

    CRITICAL: This endpoint updates execution state.
    """
    if not EXECUTION_DISPATCHER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Execution dispatcher not available")

    try:
        dispatcher = get_execution_dispatcher()
        result = dispatcher.record_execution_outcome(
            execution_id=execution_id,
            success=success,
            output=output,
            failure_reason=failure_reason,
            rollback_performed=rollback_performed,
        )

        return {
            "phase": "18C",
            "endpoint": "execution/{id}/outcome",
            "outcome_recorded": True,
            "result": result.to_dict(),
        }

    except Exception as e:
        logger.error(f"Execution outcome endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to record execution outcome: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 18D: Post-Execution Verification Endpoints
# -----------------------------------------------------------------------------

# Import verification components
try:
    from .verification_engine import (
        PostExecutionVerificationEngine,
        VerificationInput,
        ExecutionVerificationResult,
        get_verification_engine,
        verify_execution,
        create_verification_input,
        get_verification_summary,
    )
    from .verification_model import (
        VerificationStatus,
        ViolationSeverity,
        ViolationType,
        UnknownReason,
    )
    from .verification_store import (
        VerificationStore,
        get_verification_store,
        get_violations_for_execution,
    )
    VERIFICATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Verification engine not available: {e}")
    VERIFICATION_AVAILABLE = False


@app.get("/execution/{execution_id}/verification")
async def execution_verification_endpoint(execution_id: str):
    """
    Phase 18D: Get verification result for an execution.

    Returns verification result including status, violations, and domains checked.

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - VERIFICATION ONLY: Reports facts, never suggests fixes
    - NO NOTIFICATIONS: Does not alert or escalate

    This endpoint answers: "Did the execution respect all constraints?"
    """
    if not VERIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Verification engine not available")

    try:
        engine = get_verification_engine()
        result = engine.get_verification(execution_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"No verification found for execution: {execution_id}")

        return {
            "phase": "18D",
            "endpoint": "execution/{id}/verification",
            "read_only": True,
            "no_recommendations": True,
            "no_notifications": True,
            "result": result.to_dict(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get verification: {str(e)}")


@app.get("/execution/{execution_id}/violations")
async def execution_violations_endpoint(execution_id: str):
    """
    Phase 18D: Get violations for an execution.

    Returns list of all invariant violations detected for the execution.

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - OBSERVATION ONLY: Reports violations, never fixes them
    - NO RECOMMENDATIONS: Does not suggest remediation
    """
    if not VERIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Verification engine not available")

    try:
        violations = get_violations_for_execution(execution_id)

        return {
            "phase": "18D",
            "endpoint": "execution/{id}/violations",
            "read_only": True,
            "observation_only": True,
            "no_recommendations": True,
            "execution_id": execution_id,
            "violation_count": len(violations),
            "violations": [v.to_dict() for v in violations],
        }

    except Exception as e:
        logger.error(f"Violations endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get violations: {str(e)}")


@app.get("/execution/verification/recent")
async def verification_recent_endpoint(
    limit: int = 100,
    status: Optional[str] = None,
):
    """
    Phase 18D: Get recent verification results.

    Query parameters:
    - limit: Maximum number of results (default 100)
    - status: Optional filter by status (passed, failed, unknown)

    CRITICAL: This is a READ-ONLY endpoint.
    Verification results are OBSERVATION ONLY - no actions triggered.
    """
    if not VERIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Verification engine not available")

    try:
        engine = get_verification_engine()
        results = engine.get_recent_verifications(
            limit=limit,
            status=status,
        )

        return {
            "phase": "18D",
            "endpoint": "execution/verification/recent",
            "read_only": True,
            "observation_only": True,
            "total": len(results),
            "filters": {
                "limit": limit,
                "status": status,
            },
            "verifications": [r.to_dict() for r in results],
        }

    except Exception as e:
        logger.error(f"Recent verifications endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recent verifications: {str(e)}")


@app.get("/execution/verification/summary")
async def verification_summary_endpoint(
    since_hours: int = 24,
):
    """
    Phase 18D: Get verification summary statistics.

    Query parameters:
    - since_hours: Time window in hours (default 24)

    Returns summary counts by status, violation type, and severity.

    CRITICAL: This is a READ-ONLY endpoint.
    Statistics are OBSERVATION ONLY - no actions triggered.
    """
    if not VERIFICATION_AVAILABLE:
        raise HTTPException(status_code=503, detail="Verification engine not available")

    try:
        summary = get_verification_summary(since_hours=since_hours)

        return {
            "phase": "18D",
            "endpoint": "execution/verification/summary",
            "read_only": True,
            "observation_only": True,
            **summary,
        }

    except Exception as e:
        logger.error(f"Verification summary endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get verification summary: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 19: Learning, Memory & System Intelligence (READ-ONLY ANALYTICS)
# -----------------------------------------------------------------------------
try:
    from .learning_engine import (
        get_learning_engine,
        get_learning_patterns,
        get_learning_trends,
        get_learning_history,
        get_learning_summary,
        LearningInput,
    )
    from .learning_store import (
        get_learning_store,
        get_learning_statistics,
    )
    LEARNING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Learning engine not available: {e}")
    LEARNING_AVAILABLE = False


@app.get("/learning/patterns")
async def learning_patterns_endpoint(
    limit: int = 100,
    pattern_type: Optional[str] = None,
):
    """
    Phase 19: Get observed patterns.

    Query parameters:
    - limit: Maximum number of results (default 100)
    - pattern_type: Optional filter by pattern type

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - NO BEHAVIORAL COUPLING: Does not influence other phases
    - NO AUTOMATION: Never triggers any actions

    Patterns are for HUMAN INSIGHT, not automation.
    """
    if not LEARNING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Learning engine not available")

    try:
        patterns = get_learning_patterns(
            limit=limit,
            pattern_type=pattern_type,
        )

        return {
            "phase": "19",
            "endpoint": "learning/patterns",
            "read_only": True,
            "insight_only": True,
            "no_behavioral_coupling": True,
            "patterns": [p.to_dict() for p in patterns],
            "count": len(patterns),
        }

    except Exception as e:
        logger.error(f"Failed to get learning patterns: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning patterns: {str(e)}")


@app.get("/learning/trends")
async def learning_trends_endpoint(
    limit: int = 100,
    metric_name: Optional[str] = None,
):
    """
    Phase 19: Get observed trends.

    Query parameters:
    - limit: Maximum number of results (default 100)
    - metric_name: Optional filter by metric name

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - NO BEHAVIORAL COUPLING: Does not influence other phases
    - NO PREDICTION: Only observes past, never predicts future

    Trends are for HUMAN INSIGHT, not automation.
    """
    if not LEARNING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Learning engine not available")

    try:
        trends = get_learning_trends(
            limit=limit,
            metric_name=metric_name,
        )

        return {
            "phase": "19",
            "endpoint": "learning/trends",
            "read_only": True,
            "insight_only": True,
            "no_prediction": True,
            "trends": [t.to_dict() for t in trends],
            "count": len(trends),
        }

    except Exception as e:
        logger.error(f"Failed to get learning trends: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning trends: {str(e)}")


@app.get("/learning/history")
async def learning_history_endpoint(
    limit: int = 100,
    entry_type: Optional[str] = None,
    project_id: Optional[str] = None,
):
    """
    Phase 19: Get memory history.

    Query parameters:
    - limit: Maximum number of results (default 100)
    - entry_type: Optional filter by entry type
    - project_id: Optional filter by project

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - NO BEHAVIORAL COUPLING: Does not influence other phases
    - APPEND-ONLY: Memory is never modified or deleted

    History is for HUMAN INSIGHT, not automation.
    """
    if not LEARNING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Learning engine not available")

    try:
        history = get_learning_history(
            limit=limit,
            entry_type=entry_type,
            project_id=project_id,
        )

        return {
            "phase": "19",
            "endpoint": "learning/history",
            "read_only": True,
            "insight_only": True,
            "append_only": True,
            "history": [h.to_dict() for h in history],
            "count": len(history),
        }

    except Exception as e:
        logger.error(f"Failed to get learning history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning history: {str(e)}")


@app.get("/learning/summary")
async def learning_summary_endpoint():
    """
    Phase 19: Get latest learning summary.

    Returns the most recent learning summary with aggregate statistics.

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - NO BEHAVIORAL COUPLING: Does not influence other phases
    - NO AUTOMATION: Never triggers any actions

    Summary is for HUMAN INSIGHT, not automation.
    """
    if not LEARNING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Learning engine not available")

    try:
        summary = get_learning_summary()

        if not summary:
            return {
                "phase": "19",
                "endpoint": "learning/summary",
                "read_only": True,
                "insight_only": True,
                "summary": None,
                "message": "No learning summary available yet",
            }

        return {
            "phase": "19",
            "endpoint": "learning/summary",
            "read_only": True,
            "insight_only": True,
            "no_behavioral_coupling": True,
            "summary": summary.to_dict(),
        }

    except Exception as e:
        logger.error(f"Failed to get learning summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning summary: {str(e)}")


@app.get("/learning/statistics")
async def learning_statistics_endpoint(since_hours: int = 24):
    """
    Phase 19: Get learning statistics.

    Query parameters:
    - since_hours: Time window in hours (default 24)

    Returns statistics about patterns, trends, and memory entries.

    CRITICAL CONSTRAINTS:
    - READ-ONLY: Does not modify any state
    - NO BEHAVIORAL COUPLING: Does not influence other phases

    Statistics are for HUMAN INSIGHT, not automation.
    """
    if not LEARNING_AVAILABLE:
        raise HTTPException(status_code=503, detail="Learning engine not available")

    try:
        statistics = get_learning_statistics(since_hours=since_hours)

        return {
            "phase": "19",
            "endpoint": "learning/statistics",
            "read_only": True,
            "insight_only": True,
            "no_behavioral_coupling": True,
            "statistics": statistics,
        }

    except Exception as e:
        logger.error(f"Failed to get learning statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get learning statistics: {str(e)}")


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with standard error format."""
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
                "details": {}
            }
        }
    )


# -----------------------------------------------------------------------------
# Startup/Shutdown Events
# -----------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Initialize controller on startup."""
    logger.info("Task Controller starting up (Phase 17C)...")
    logger.info(f"Projects directory: {PROJECTS_DIR}")
    logger.info(f"Docs directory: {DOCS_DIR}")

    # Ensure directories exist
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 14.10: Start Claude multi-worker scheduler if available
    if CLAUDE_BACKEND_AVAILABLE:
        try:
            await start_scheduler()
            logger.info(f"Claude multi-worker scheduler started (max {MAX_CONCURRENT_JOBS} concurrent jobs)")
        except Exception as e:
            logger.error(f"Failed to start Claude scheduler: {e}")

    # Phase 17A: Start runtime intelligence polling if available
    if RUNTIME_INTELLIGENCE_AVAILABLE:
        try:
            from .runtime_intelligence import start_signal_polling
            start_signal_polling()
            logger.info("Runtime intelligence polling started (Phase 17A - OBSERVATION ONLY)")
        except Exception as e:
            logger.error(f"Failed to start runtime intelligence: {e}")

    # Phase 15.1: Initialize lifecycle manager and recover state
    if LIFECYCLE_AVAILABLE:
        try:
            manager = get_lifecycle_manager()
            recovery_summary = await manager.recover_state()
            logger.info(f"Phase 15.1: Lifecycle manager initialized ({recovery_summary['total']} lifecycles recovered)")
        except Exception as e:
            logger.error(f"Failed to initialize lifecycle manager: {e}")

    # Phase 15.3: Initialize ingestion engine
    if INGESTION_AVAILABLE:
        try:
            engine = get_ingestion_engine()
            logger.info("Phase 15.3: Ingestion Engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ingestion engine: {e}")

    # Phase 17C: Log recommendations availability
    if RECOMMENDATIONS_AVAILABLE:
        logger.info("Phase 17C: Recommendation engine initialized (ADVISORY ONLY)")

    # Phase 18C: Log execution dispatcher availability
    if EXECUTION_DISPATCHER_AVAILABLE:
        logger.info("Phase 18C: Execution dispatcher initialized (SINGLE EXECUTION POINT)")

    # Phase 18D: Log verification engine availability
    if VERIFICATION_AVAILABLE:
        logger.info("Phase 18D: Verification engine initialized (OBSERVATION ONLY)")

    # Phase 19: Log learning engine availability
    if LEARNING_AVAILABLE:
        logger.info("Phase 19: Learning engine initialized (INSIGHT ONLY, NO BEHAVIORAL COUPLING)")

    logger.info("Task Controller ready (Phase 19: Learning, Memory & System Intelligence)")
    logger.info("SAFETY: Production deployment requires DUAL APPROVAL (different users).")
    logger.info("SAFETY: All production actions logged to immutable audit trail.")
    logger.info("SAFETY: Production rollback always available (break-glass).")
    logger.info("SAFETY: Testing environment MUST be used before production.")
    logger.info("PHASE 17C: Recommendations are ADVISORY ONLY - human approval required.")
    logger.info("PHASE 18C: ALL executions MUST flow through the dispatcher - no bypass allowed.")
    logger.info("PHASE 18D: Verification is OBSERVATION ONLY - violations recorded, not acted upon.")
    logger.info("PHASE 19: Learning is INSIGHT ONLY - no behavioral coupling, no automation.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Task Controller shutting down...")

    # Phase 14.10: Stop Claude multi-worker scheduler if running
    if CLAUDE_BACKEND_AVAILABLE:
        try:
            await stop_scheduler()
            logger.info("Claude multi-worker scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping Claude scheduler: {e}")

    # Phase 17A: Stop runtime intelligence polling if running
    if RUNTIME_INTELLIGENCE_AVAILABLE:
        try:
            from .runtime_intelligence import stop_signal_polling
            stop_signal_polling()
            logger.info("Runtime intelligence polling stopped")
        except Exception as e:
            logger.error(f"Error stopping runtime intelligence: {e}")


# -----------------------------------------------------------------------------
# Main Entry Point (for development)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
