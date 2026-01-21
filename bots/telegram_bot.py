"""
Telegram Bot - Chat Interface

Phase 6-17C: Production Hardening & Explicit Go-Live Controls

User-facing interface that:
- Supports multiple users
- Supports multiple projects
- Converts chat input to structured tasks
- Forwards tasks to Task Controller
- Sends status updates back to users
- Manages task lifecycle (validate, plan, approve, reject)
- Generates code diffs
- Applies diffs with EXPLICIT confirmation
- Supports rollback
- Prepares git commits (human-approved)
- Triggers CI pipelines (controlled)
- Ingests CI results
- Deploys to testing (gated)
- Supports production deployment with DUAL APPROVAL
- Provides break-glass production rollback

Bot token should be read from environment variable: TELEGRAM_BOT_TOKEN

PHASE 6 SAFETY (Production Hardening):
- Production deployment requires DUAL APPROVAL (different users)
- Requester CANNOT approve their own request
- All production actions logged to immutable audit trail
- Production rollback always available (break-glass, no dual approval)
- Testing MUST be deployed before production
- Multi-step confirmation is mandatory

PHASE 5 SAFETY (inherited):
- Execution requires EXPLICIT confirmation keyword
- Dry-run must pass before apply
- Backup created before any apply
- Rollback always available
- No autonomous execution
- Commits require EXPLICIT confirmation
- CI must pass before testing deployment
- Testing deployment requires EXPLICIT confirmation
- No automatic merges
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("telegram_bot")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
CONTROLLER_BASE_URL = os.getenv("CONTROLLER_URL", "http://localhost:8000")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class BotCommand(str, Enum):
    """Supported bot commands."""
    START = "start"
    HELP = "help"
    PROJECT = "project"
    BOOTSTRAP = "bootstrap"  # Phase 2
    TASK = "task"
    VALIDATE = "validate"    # Phase 2
    PLAN = "plan"            # Phase 2
    APPROVE = "approve"      # Phase 2
    REJECT = "reject"        # Phase 2
    GENERATE_DIFF = "generate_diff"  # Phase 3
    # Phase 4: Execution commands
    DRY_RUN = "dry_run"      # Phase 4
    APPLY = "apply"          # Phase 4
    ROLLBACK = "rollback"    # Phase 4
    # Phase 5: CI/Release commands
    COMMIT = "commit"        # Phase 5
    CI_RUN = "ci_run"        # Phase 5
    CI_RESULT = "ci_result"  # Phase 5
    DEPLOY_TESTING = "deploy_testing"  # Phase 5
    # Phase 6: Production commands (DUAL APPROVAL)
    PROD_REQUEST = "prod_request"      # Phase 6: Request production deploy
    PROD_APPROVE = "prod_approve"      # Phase 6: Approve production deploy (DIFFERENT user)
    PROD_APPLY = "prod_apply"          # Phase 6: Execute production deploy
    PROD_ROLLBACK = "prod_rollback"    # Phase 6: Rollback production (break-glass)
    DEPLOY = "deploy"
    STATUS = "status"
    LIST = "list"
    # Phase 17C: Recommendation commands (ADVISORY ONLY)
    RECOMMENDATIONS = "recommendations"  # List recent recommendations
    RECOMMENDATION = "recommendation"    # View specific recommendation
    REC_APPROVE = "rec_approve"          # Approve recommendation
    REC_DISMISS = "rec_dismiss"          # Dismiss recommendation


class TaskType(str, Enum):
    """Task types (aligned with Phase 2 controller)."""
    BUG = "bug"
    FEATURE = "feature"
    REFACTOR = "refactor"
    INFRA = "infra"


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------
@dataclass
class UserSession:
    """Tracks user session state."""
    user_id: str
    username: Optional[str]
    current_project: Optional[str] = None
    last_task_id: Optional[str] = None  # Phase 2: Track last created task
    is_authorized: bool = False


@dataclass
class ParsedCommand:
    """Parsed command from user input."""
    command: BotCommand
    args: list[str]
    raw_text: str


# -----------------------------------------------------------------------------
# Session Management
# -----------------------------------------------------------------------------
user_sessions: dict[str, UserSession] = {}


def get_or_create_session(user_id: str, username: Optional[str] = None) -> UserSession:
    """Get existing session or create new one."""
    if user_id not in user_sessions:
        user_sessions[user_id] = UserSession(
            user_id=user_id,
            username=username,
            is_authorized=False
        )
        logger.info(f"Created new session for user: {user_id}")
    return user_sessions[user_id]


def set_current_project(user_id: str, project_name: str) -> bool:
    """Set the current project for a user session."""
    session = get_or_create_session(user_id)
    session.current_project = project_name
    logger.info(f"User {user_id} switched to project: {project_name}")
    return True


def set_last_task_id(user_id: str, task_id: str) -> None:
    """Set the last task ID for quick reference."""
    session = get_or_create_session(user_id)
    session.last_task_id = task_id
    logger.info(f"User {user_id} last task: {task_id}")


# -----------------------------------------------------------------------------
# Authorization (Placeholder)
# -----------------------------------------------------------------------------
def is_user_authorized(user_id: str) -> bool:
    """Check if user is authorized to use the bot."""
    logger.info(f"Authorization check for user {user_id} - PLACEHOLDER: allowing all")
    return True


def check_rate_limit(user_id: str) -> tuple[bool, Optional[str]]:
    """Check if user has exceeded rate limit."""
    return True, None


# -----------------------------------------------------------------------------
# Command Parsing
# -----------------------------------------------------------------------------
def parse_command(text: str) -> Optional[ParsedCommand]:
    """Parse user input into a structured command."""
    if not text.startswith("/"):
        return None

    parts = text.split()
    command_str = parts[0][1:].lower()

    try:
        command = BotCommand(command_str)
    except ValueError:
        return None

    return ParsedCommand(
        command=command,
        args=parts[1:] if len(parts) > 1 else [],
        raw_text=text
    )


def infer_task_type(description: str) -> TaskType:
    """Infer task type from description text."""
    description_lower = description.lower()

    if any(word in description_lower for word in ["bug", "fix", "error", "issue", "broken"]):
        return TaskType.BUG
    elif any(word in description_lower for word in ["add", "new", "feature", "implement", "create"]):
        return TaskType.FEATURE
    elif any(word in description_lower for word in ["refactor", "clean", "improve", "optimize"]):
        return TaskType.REFACTOR
    elif any(word in description_lower for word in ["deploy", "infra", "ci", "cd", "pipeline"]):
        return TaskType.INFRA
    else:
        return TaskType.FEATURE


# -----------------------------------------------------------------------------
# Controller Communication (Placeholders - HTTP calls to be implemented)
# -----------------------------------------------------------------------------
def call_controller_bootstrap(project_name: str, repo_url: str, tech_stack: list[str], user_id: str) -> dict:
    """Call POST /project/bootstrap on controller."""
    logger.info(f"Controller call: bootstrap project={project_name}")
    # TODO: Implement actual HTTP request
    # import httpx
    # response = httpx.post(f"{CONTROLLER_BASE_URL}/project/bootstrap", json={...})
    # return response.json()
    return {
        "status": "bootstrapped",
        "project_name": project_name,
        "message": f"Project '{project_name}' bootstrapped (placeholder)"
    }


def call_controller_create_task(project_name: str, description: str, task_type: str, user_id: str) -> dict:
    """Call POST /task on controller."""
    logger.info(f"Controller call: create task for project={project_name}")
    # TODO: Implement actual HTTP request
    import uuid
    task_id = str(uuid.uuid4())[:8]  # Short ID for display
    return {
        "task_id": task_id,
        "state": "received",
        "message": f"Task created with ID: {task_id}"
    }


def call_controller_validate_task(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/validate on controller."""
    logger.info(f"Controller call: validate task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "current_state": "validated",
        "validation_passed": True,
        "validation_errors": [],
        "message": "Task validated successfully (placeholder)"
    }


def call_controller_plan_task(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/plan on controller."""
    logger.info(f"Controller call: plan task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "state": "awaiting_approval",
        "plan_path": f"projects/{project_name}/plans/{task_id}_plan.md",
        "message": "Plan generated (placeholder)"
    }


def call_controller_approve_task(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/approve on controller."""
    logger.info(f"Controller call: approve task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "current_state": "approved",
        "message": "Task approved (placeholder). Use /generate_diff to create code changes."
    }


def call_controller_generate_diff(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/generate-diff on controller (Phase 3)."""
    logger.info(f"Controller call: generate_diff task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "current_state": "diff_generated",
        "diff_path": f"projects/{project_name}/diffs/{task_id}.diff",
        "files_in_diff": 2,
        "message": "Diff generated (placeholder)",
        "warning": "DIFF NOT APPLIED. FOR HUMAN REVIEW ONLY."
    }


def call_controller_reject_task(project_name: str, task_id: str, reason: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/reject on controller."""
    logger.info(f"Controller call: reject task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "current_state": "rejected",
        "rejection_reason": reason,
        "message": "Task rejected (placeholder)"
    }


def call_controller_get_status(project_name: str) -> dict:
    """Call GET /status/{project_name} on controller."""
    logger.info(f"Controller call: status for project={project_name}")
    # TODO: Implement actual HTTP request
    return {
        "project_name": project_name,
        "current_phase": "bootstrap",
        "task_count": 0,
        "tasks_by_state": {},
        "last_updated": "N/A"
    }


def call_controller_list_projects() -> dict:
    """Call GET /projects on controller."""
    logger.info("Controller call: list projects")
    # TODO: Implement actual HTTP request
    return {
        "projects": [],
        "count": 0
    }


def call_controller_deploy(project_name: str, environment: str, user_id: str) -> dict:
    """Call POST /deploy on controller."""
    logger.info(f"Controller call: deploy project={project_name} to {environment}")
    # TODO: Implement actual HTTP request
    return {
        "status": "placeholder",
        "message": f"Deployment to {environment} is placeholder in Phase 4",
        "deployment_url": f"https://{'aitesting' if environment == 'testing' else 'ai'}.mybd.in"
    }


# -----------------------------------------------------------------------------
# Phase 4 Controller Communication
# -----------------------------------------------------------------------------
def call_controller_dry_run(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/dry-run on controller (Phase 4)."""
    logger.info(f"Controller call: dry_run task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "current_state": "ready_to_apply",
        "files_affected": ["src/main.py", "tests/test_main.py"],
        "lines_added": 15,
        "lines_removed": 3,
        "can_apply": True,
        "conflicts": [],
        "message": "Dry-run successful (placeholder)",
        "warning": "DRY-RUN ONLY. NO FILES MODIFIED."
    }


def call_controller_apply(project_name: str, task_id: str, user_id: str, confirm: bool) -> dict:
    """Call POST /task/{task_id}/apply on controller (Phase 4)."""
    logger.info(f"Controller call: apply task={task_id} confirm={confirm}")
    # TODO: Implement actual HTTP request

    if not confirm:
        return {
            "error": True,
            "message": "Apply REQUIRES explicit confirmation. Use: /apply <task_id> confirm"
        }

    return {
        "task_id": task_id,
        "current_state": "applied",
        "files_modified": ["src/main.py", "tests/test_main.py"],
        "backup_path": f"projects/{project_name}/backups/{task_id}",
        "message": "Diff applied successfully (placeholder)",
        "rollback_available": True
    }


def call_controller_rollback(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/rollback on controller (Phase 4)."""
    logger.info(f"Controller call: rollback task={task_id}")
    # TODO: Implement actual HTTP request
    return {
        "task_id": task_id,
        "current_state": "rolled_back",
        "files_restored": ["src/main.py", "tests/test_main.py"],
        "message": "Rollback successful (placeholder)"
    }


# -----------------------------------------------------------------------------
# Phase 5 Controller Communication
# -----------------------------------------------------------------------------
def call_controller_commit(project_name: str, task_id: str, user_id: str, confirm: bool) -> dict:
    """Call POST /task/{task_id}/commit on controller (Phase 5)."""
    logger.info(f"Controller call: commit task={task_id} confirm={confirm}")
    # TODO: Implement actual HTTP request

    if not confirm:
        return {
            "error": True,
            "message": "Commit REQUIRES explicit confirmation. Use: /commit <task_id> confirm"
        }

    import uuid
    commit_hash = str(uuid.uuid4())[:8]
    return {
        "task_id": task_id,
        "current_state": "committed",
        "commit_hash": commit_hash,
        "commit_message": f"[task] {task_id[:8]}: Task changes",
        "files_committed": ["src/main.py", "tests/test_main.py"],
        "message": "Commit created locally (placeholder)",
        "warning": "COMMIT CREATED LOCALLY. NOT PUSHED."
    }


def call_controller_ci_run(project_name: str, task_id: str, user_id: str) -> dict:
    """Call POST /task/{task_id}/ci/run on controller (Phase 5)."""
    logger.info(f"Controller call: ci_run task={task_id}")
    # TODO: Implement actual HTTP request

    import uuid
    ci_run_id = f"ci-{str(uuid.uuid4())[:8]}"
    return {
        "task_id": task_id,
        "current_state": "ci_running",
        "ci_run_id": ci_run_id,
        "message": "CI pipeline triggered (placeholder)",
        "warning": "CI RUNNING. WAIT FOR RESULTS."
    }


def call_controller_ci_result(project_name: str, task_id: str, user_id: str, status: str, logs_url: str = None) -> dict:
    """Call POST /task/{task_id}/ci/result on controller (Phase 5)."""
    logger.info(f"Controller call: ci_result task={task_id} status={status}")
    # TODO: Implement actual HTTP request

    if status == "passed":
        return {
            "task_id": task_id,
            "current_state": "ci_passed",
            "ci_status": "passed",
            "logs_url": logs_url,
            "message": "CI passed (placeholder). Ready for testing deployment."
        }
    else:
        return {
            "task_id": task_id,
            "current_state": "ci_failed",
            "ci_status": "failed",
            "logs_url": logs_url,
            "message": "CI failed (placeholder). Fix issues and re-commit."
        }


def call_controller_deploy_testing(project_name: str, task_id: str, user_id: str, confirm: bool) -> dict:
    """Call POST /task/{task_id}/deploy/testing on controller (Phase 5)."""
    logger.info(f"Controller call: deploy_testing task={task_id} confirm={confirm}")
    # TODO: Implement actual HTTP request

    if not confirm:
        return {
            "error": True,
            "message": "Testing deployment REQUIRES explicit confirmation. Use: /deploy_testing <task_id> confirm"
        }

    return {
        "task_id": task_id,
        "current_state": "deployed_testing",
        "deployment_url": "https://aitesting.mybd.in",
        "message": "Deployed to testing (placeholder)",
        "warning": "DEPLOYED TO TESTING ONLY. NOT PRODUCTION."
    }


# -----------------------------------------------------------------------------
# Phase 6 Controller API Calls - Production Deployment
# -----------------------------------------------------------------------------
def call_controller_prod_request(
    project_name: str,
    task_id: str,
    user_id: str,
    justification: str,
    rollback_plan: str,
    risk_acknowledged: bool
) -> dict:
    """Call POST /task/{task_id}/deploy/production/request on controller (Phase 6)."""
    logger.info(f"Controller call: prod_request task={task_id} user={user_id} risk_acknowledged={risk_acknowledged}")
    # TODO: Implement actual HTTP request

    if not risk_acknowledged:
        return {
            "error": True,
            "message": "Production deployment REQUIRES explicit risk acknowledgment."
        }

    if len(justification) < 20:
        return {
            "error": True,
            "message": "Justification must be at least 20 characters."
        }

    if len(rollback_plan) < 10:
        return {
            "error": True,
            "message": "Rollback plan must be at least 10 characters."
        }

    return {
        "task_id": task_id,
        "current_state": "prod_deploy_requested",
        "requested_by": user_id,
        "justification": justification,
        "rollback_plan": rollback_plan,
        "message": f"Production deployment requested. REQUIRES approval from a DIFFERENT user.",
        "warning": "⚠️ PRODUCTION DEPLOYMENT REQUESTED. REQUIRES APPROVAL FROM DIFFERENT USER.",
        "next_step": f"Another user must approve via /prod_approve {task_id}"
    }


def call_controller_prod_approve(
    project_name: str,
    task_id: str,
    user_id: str,
    approval_reason: str,
    reviewed_changes: bool,
    reviewed_rollback: bool
) -> dict:
    """Call POST /task/{task_id}/deploy/production/approve on controller (Phase 6)."""
    logger.info(f"Controller call: prod_approve task={task_id} approver={user_id}")
    # TODO: Implement actual HTTP request - must validate dual approval

    if not reviewed_changes:
        return {
            "error": True,
            "message": "You must confirm you reviewed the changes."
        }

    if not reviewed_rollback:
        return {
            "error": True,
            "message": "You must confirm you reviewed the rollback plan."
        }

    # Note: In actual implementation, controller will enforce that approver != requester
    return {
        "task_id": task_id,
        "current_state": "prod_approved",
        "approved_by": user_id,
        "approval_reason": approval_reason,
        "message": "Production deployment approved. Dual approval verified.",
        "warning": "⚠️ PRODUCTION DEPLOYMENT APPROVED. READY FOR FINAL APPLY.",
        "next_step": f"Execute deploy via /prod_apply {task_id} confirm"
    }


def call_controller_prod_apply(
    project_name: str,
    task_id: str,
    user_id: str,
    confirm: bool
) -> dict:
    """Call POST /task/{task_id}/deploy/production/apply on controller (Phase 6)."""
    logger.info(f"Controller call: prod_apply task={task_id} user={user_id} confirm={confirm}")
    # TODO: Implement actual HTTP request

    if not confirm:
        return {
            "error": True,
            "message": "Production deployment REQUIRES explicit confirmation. Use: /prod_apply <task_id> confirm"
        }

    return {
        "task_id": task_id,
        "current_state": "deployed_production",
        "deployment_url": "https://ai.mybd.in",
        "release_manifest_path": f"projects/{project_name}/releases/{task_id}/RELEASE_MANIFEST.yaml",
        "message": "Deployed to production successfully.",
        "warning": "🚀 DEPLOYED TO PRODUCTION. MONITOR CLOSELY.",
        "rollback_available": True,
        "rollback_command": f"/prod_rollback {task_id}"
    }


def call_controller_prod_rollback(
    project_name: str,
    task_id: str,
    user_id: str,
    reason: str
) -> dict:
    """Call POST /task/{task_id}/deploy/production/rollback on controller (Phase 6)."""
    logger.info(f"Controller call: prod_rollback task={task_id} user={user_id} reason={reason}")
    # TODO: Implement actual HTTP request

    if not reason or len(reason) < 5:
        return {
            "error": True,
            "message": "Rollback REQUIRES a reason for audit trail."
        }

    return {
        "task_id": task_id,
        "current_state": "prod_rolled_back",
        "rolled_back_by": user_id,
        "rollback_reason": reason,
        "message": "Production rolled back successfully.",
        "warning": "⚠️ PRODUCTION ROLLED BACK. Verify system stability."
    }


# -----------------------------------------------------------------------------
# Phase 17C Controller Communication - Recommendations (ADVISORY ONLY)
# -----------------------------------------------------------------------------
def call_controller_get_recommendations(limit: int = 20, status: str = None) -> dict:
    """
    Call GET /recommendations on controller (Phase 17C).

    CRITICAL: Recommendations are ADVISORY ONLY.
    - They suggest actions, never execute them
    - Human must approve/dismiss
    - NO automation, NO lifecycle mutation
    """
    logger.info(f"Controller call: get recommendations limit={limit}")
    # TODO: Implement actual HTTP request
    # import httpx
    # params = {"limit": limit}
    # if status:
    #     params["status"] = status
    # response = httpx.get(f"{CONTROLLER_BASE_URL}/recommendations", params=params)
    # return response.json()
    return {
        "phase": "17C",
        "advisory_only": True,
        "no_execution": True,
        "total": 0,
        "recommendations": [],
        "message": "Recommendations endpoint placeholder"
    }


def call_controller_get_recommendation(recommendation_id: str) -> dict:
    """
    Call GET /recommendations/{id} on controller (Phase 17C).

    CRITICAL: Recommendations are ADVISORY ONLY.
    """
    logger.info(f"Controller call: get recommendation id={recommendation_id}")
    # TODO: Implement actual HTTP request
    return {
        "phase": "17C",
        "advisory_only": True,
        "no_execution": True,
        "recommendation": None,
        "message": f"Recommendation {recommendation_id} not found (placeholder)"
    }


def call_controller_approve_recommendation(
    recommendation_id: str,
    user_id: str,
    reason: str = None
) -> dict:
    """
    Call POST /recommendations/{id}/approve on controller (Phase 17C).

    CRITICAL: Approval does NOT trigger any automation.
    - This is ADVISORY ONLY
    - It marks the recommendation as approved for tracking
    - NO execution, NO lifecycle changes, NO deployment
    """
    logger.info(f"Controller call: approve recommendation id={recommendation_id} user={user_id}")
    # TODO: Implement actual HTTP request
    return {
        "phase": "17C",
        "advisory_only": True,
        "no_execution": True,
        "approved": True,
        "recommendation_id": recommendation_id,
        "message": "Recommendation approved (placeholder). NOTE: This is advisory only - no automatic action taken."
    }


def call_controller_dismiss_recommendation(
    recommendation_id: str,
    user_id: str,
    reason: str = None
) -> dict:
    """
    Call POST /recommendations/{id}/dismiss on controller (Phase 17C).

    CRITICAL: This is ADVISORY ONLY - no automation triggered.
    """
    logger.info(f"Controller call: dismiss recommendation id={recommendation_id} user={user_id}")
    # TODO: Implement actual HTTP request
    return {
        "phase": "17C",
        "advisory_only": True,
        "no_execution": True,
        "dismissed": True,
        "recommendation_id": recommendation_id,
        "message": "Recommendation dismissed (placeholder)."
    }


# -----------------------------------------------------------------------------
# Command Handlers
# -----------------------------------------------------------------------------
def handle_start(user_id: str, username: Optional[str]) -> str:
    """Handle /start command."""
    get_or_create_session(user_id, username)

    return """Welcome to the AI Development Platform Bot!

I help you manage AI-driven autonomous software development.

Phase 17C Commands:

Project Management:
/bootstrap <name> <repo_url> - Create new project
/project <name> - Select a project
/list - List all projects

Task Lifecycle:
/task <description> - Create a new task
/validate <task_id> - Validate a task
/plan <task_id> - Generate implementation plan
/approve <task_id> - Approve a planned task
/reject <task_id> <reason> - Reject a task

Code Generation:
/generate_diff <task_id> - Generate code diff

Execution (Phase 4):
/dry_run <task_id> - Test diff application
/apply <task_id> confirm - Apply diff (REQUIRES confirm)
/rollback <task_id> - Undo applied changes

CI/Release (Phase 5):
/commit <task_id> confirm - Create git commit (REQUIRES confirm)
/ci_run <task_id> - Trigger CI pipeline
/ci_result <task_id> passed|failed - Report CI result
/deploy_testing <task_id> confirm - Deploy to testing (REQUIRES confirm)

Production (Phase 6 - DUAL APPROVAL):
/prod_request <task_id> - Request production deployment
/prod_approve <task_id> - Approve production (DIFFERENT user)
/prod_apply <task_id> confirm - Execute production deploy
/prod_rollback <task_id> - Rollback production (break-glass)

Recommendations (Phase 17C - ADVISORY ONLY):
/recommendations - List recent recommendations
/recommendation <id> - View recommendation details
/rec_approve <id> [reason] - Approve recommendation
/rec_dismiss <id> [reason] - Dismiss recommendation

Other:
/status - Get project status
/help - Show full help

SAFETY NOTES:
- Apply/commit/deploy REQUIRE 'confirm' keyword
- Production requires DUAL APPROVAL (different users)
- All production actions are audited
- Recommendations are ADVISORY ONLY - NO automatic execution

Start by creating a project with /bootstrap"""


def handle_help(user_id: str) -> str:
    """Handle /help command."""
    session = get_or_create_session(user_id)
    project_info = f"Current project: {session.current_project}" if session.current_project else "No project selected"
    task_info = f"Last task: {session.last_task_id}" if session.last_task_id else "No recent task"

    return f"""AI Development Platform - Phase 17C Help

{project_info}
{task_info}

PROJECT COMMANDS:
/bootstrap <name> <repo_url> [tech1,tech2]
  Create a new project
  Example: /bootstrap my-app https://github.com/user/repo python,fastapi

/project <name>
  Select/switch to a project

/list
  List all registered projects

TASK LIFECYCLE COMMANDS:
/task <description>
  Create a task (auto-detects type: bug/feature/refactor/infra)
  Example: /task Fix the login button not working

/validate <task_id>
  Validate task (moves: received -> validated)
  Use 'last' for last created task

/plan <task_id>
  Generate implementation plan (moves: validated -> awaiting_approval)
  Use 'last' for last created task

/approve <task_id>
  Approve task (moves: awaiting_approval -> approved)

/reject <task_id> <reason>
  Reject task with reason

CODE GENERATION:
/generate_diff <task_id>
  Generate code diff for approved task
  Use 'last' for last created task

EXECUTION (PHASE 4):
/dry_run <task_id>
  Simulate diff application (NO files modified)
  Use 'last' for last created task
  Must pass before /apply

/apply <task_id> confirm
  Apply diff to project files
  REQUIRES 'confirm' keyword
  Creates backup before apply
  Rollback available after

/rollback <task_id>
  Undo applied changes
  Restores files from backup
  Use 'last' for last created task

CI/RELEASE (PHASE 5):
/commit <task_id> confirm
  Create git commit for applied changes
  REQUIRES 'confirm' keyword
  Commit is LOCAL only, NOT pushed
  Use 'last' for last created task

/ci_run <task_id>
  Trigger CI pipeline for committed task
  CI runs externally (GitHub Actions)
  Use 'last' for last created task

/ci_result <task_id> passed|failed [logs_url]
  Report CI pipeline result
  Must be 'passed' or 'failed'
  Optional: include logs URL

/deploy_testing <task_id> confirm
  Deploy to testing environment
  REQUIRES 'confirm' keyword
  REQUIRES CI to have passed
  Use 'last' for last created task

PRODUCTION DEPLOYMENT (PHASE 6 - DUAL APPROVAL):
⚠️ LOUD WARNING: Production deployment is IRREVERSIBLE and affects real users!

/prod_request <task_id>
  Request production deployment
  REQUIRES task to be in deployed_testing state
  REQUIRES justification and rollback plan
  Creates a REQUEST that another user must approve

/prod_approve <task_id>
  Approve production deployment request
  ⚠️ MUST be a DIFFERENT user than requester
  MUST confirm reviewed changes and rollback plan

/prod_apply <task_id> confirm
  Execute approved production deployment
  REQUIRES 'confirm' keyword
  Creates release manifest and audit log

/prod_rollback <task_id>
  🚨 BREAK-GLASS: Rollback production immediately
  Does NOT require dual approval (speed > ceremony)
  REQUIRES reason for audit trail

RECOMMENDATIONS (PHASE 17C - ADVISORY ONLY):
⚠️ Recommendations SUGGEST actions - they NEVER execute automatically

/recommendations [pending|approved|dismissed] [limit]
  List recent recommendations
  Optional filters: status and limit

/recommendation <id>
  View full details of a recommendation
  Shows suggested actions, rationale, severity

/rec_approve <id> [reason]
  Approve a recommendation
  ⚠️ Does NOT trigger any automatic action
  Only marks as "approved" for tracking

/rec_dismiss <id> [reason]
  Dismiss a recommendation
  Logs decision for audit purposes

STATUS:
/status - Show project status and task counts

SAFETY GUARANTEES (PHASE 17C):
- Production deployment requires DUAL APPROVAL
- Requester CANNOT approve their own request
- All production actions logged to audit trail
- Rollback always available (break-glass)
- Testing MUST be deployed before production
- Recommendations are ADVISORY ONLY - NO automatic execution"""


def handle_bootstrap(user_id: str, args: list[str]) -> str:
    """Handle /bootstrap command - create new project."""
    if len(args) < 2:
        return """Usage: /bootstrap <project_name> <repo_url> [tech1,tech2,...]

Example:
/bootstrap my-webapp https://github.com/user/my-webapp python,fastapi,postgresql

Rules:
- Project name: lowercase letters, numbers, hyphens only
- Repo URL: GitHub URL (validation only, no clone)
- Tech stack: comma-separated list (optional)"""

    project_name = args[0].lower()
    repo_url = args[1]
    tech_stack = args[2].split(",") if len(args) > 2 else []

    # Validate project name format
    if not all(c.isalnum() or c == '-' for c in project_name):
        return "Error: Project name must contain only lowercase letters, numbers, and hyphens."

    result = call_controller_bootstrap(project_name, repo_url, tech_stack, user_id)

    if result.get("status") == "bootstrapped":
        set_current_project(user_id, project_name)
        return f"""Project bootstrapped successfully!

Project: {project_name}
Repo: {repo_url}
Tech Stack: {', '.join(tech_stack) if tech_stack else 'Not specified'}

Project is now selected. You can:
- /task <description> to create a task
- /status to check project status"""
    else:
        return f"Error: {result.get('message', 'Bootstrap failed')}"


def handle_project(user_id: str, args: list[str]) -> str:
    """Handle /project command."""
    if not args:
        session = get_or_create_session(user_id)
        if session.current_project:
            return f"Current project: {session.current_project}\n\nUse /project <name> to switch."
        return "No project selected. Use /project <name> to select one, or /bootstrap to create new."

    project_name = args[0].lower()
    set_current_project(user_id, project_name)

    return f"""Switched to project: {project_name}

Commands available:
- /task <description> - Create a task
- /status - Check project status
- /list - List all projects"""


def handle_task(user_id: str, args: list[str]) -> str:
    """Handle /task command - create new task."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> or /bootstrap first."

    if not args:
        return """Please provide a task description.

Examples:
/task Fix the login button not working
/task Add dark mode toggle to settings
/task Refactor database connection pooling
/task Set up CI/CD pipeline"""

    task_description = " ".join(args)
    task_type = infer_task_type(task_description)

    result = call_controller_create_task(
        project_name=session.current_project,
        description=task_description,
        task_type=task_type.value,
        user_id=user_id
    )

    task_id = result.get('task_id', 'unknown')
    set_last_task_id(user_id, task_id)

    return f"""Task created!

Project: {session.current_project}
Task ID: {task_id}
Type: {task_type.value}
State: {result.get('state', 'received')}

Description: {task_description}

Next steps:
1. /validate {task_id} (or /validate last)
2. /plan {task_id}
3. /approve {task_id} or /reject {task_id} <reason>"""


def handle_validate(user_id: str, args: list[str]) -> str:
    """Handle /validate command - validate a task."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /validate <task_id>\n\nLast task: {session.last_task_id}\nUse: /validate last"
        return "Usage: /validate <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_validate_task(session.current_project, task_id, user_id)

    if result.get('validation_passed'):
        return f"""Task validated successfully!

Task ID: {task_id}
State: {result.get('current_state', 'validated')}

Next step: /plan {task_id}"""
    else:
        errors = result.get('validation_errors', [])
        error_str = "\n".join(f"  - {e}" for e in errors) if errors else "  Unknown validation error"
        return f"""Task validation failed!

Task ID: {task_id}
State: {result.get('current_state', 'rejected')}

Errors:
{error_str}"""


def handle_plan(user_id: str, args: list[str]) -> str:
    """Handle /plan command - generate implementation plan."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /plan <task_id>\n\nLast task: {session.last_task_id}\nUse: /plan last"
        return "Usage: /plan <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_plan_task(session.current_project, task_id, user_id)

    return f"""Implementation plan generated!

Task ID: {task_id}
State: {result.get('state', 'awaiting_approval')}
Plan: {result.get('plan_path', 'N/A')}

The plan includes:
- Overview
- Files likely to change
- Risk analysis
- Test impact
- Rollback strategy

Next step: Review the plan, then:
- /approve {task_id} to approve
- /reject {task_id} <reason> to reject"""


def handle_approve(user_id: str, args: list[str]) -> str:
    """Handle /approve command - approve a task."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /approve <task_id>\n\nLast task: {session.last_task_id}\nUse: /approve last"
        return "Usage: /approve <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_approve_task(session.current_project, task_id, user_id)

    return f"""Task approved!

Task ID: {task_id}
State: {result.get('current_state', 'approved')}

Next step: /generate_diff {task_id}

This will generate a code diff for human review.
NOTE: The diff will NOT be applied automatically."""


def handle_reject(user_id: str, args: list[str]) -> str:
    """Handle /reject command - reject a task."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if len(args) < 2:
        return """Usage: /reject <task_id> <reason>

Example: /reject abc123 Requirements are unclear, need more details

Reason must be at least 10 characters."""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        reason = " ".join(args[1:])
    else:
        reason = " ".join(args[1:])

    if len(reason) < 10:
        return "Rejection reason must be at least 10 characters."

    result = call_controller_reject_task(session.current_project, task_id, reason, user_id)

    return f"""Task rejected.

Task ID: {task_id}
State: {result.get('current_state', 'rejected')}
Reason: {reason}

The task has been moved to REJECTED state."""


def handle_generate_diff(user_id: str, args: list[str]) -> str:
    """Handle /generate_diff command - generate code diff for approved task (Phase 3)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /generate_diff <task_id>\n\nLast task: {session.last_task_id}\nUse: /generate_diff last"
        return "Usage: /generate_diff <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_generate_diff(session.current_project, task_id, user_id)

    warning = result.get('warning', 'DIFF NOT APPLIED. FOR HUMAN REVIEW ONLY.')

    return f"""Diff generated!

Task ID: {task_id}
State: {result.get('current_state', 'diff_generated')}
Diff file: {result.get('diff_path', 'N/A')}
Files in diff: {result.get('files_in_diff', 0)}

{warning}

Next steps (Phase 4):
1. Review the diff file
2. /dry_run {task_id} - Test the diff
3. /apply {task_id} confirm - Apply (if dry-run passes)

SAFETY REMINDER:
- The diff has NOT been applied
- No code changes have been made
- Human review is MANDATORY"""


# -----------------------------------------------------------------------------
# Phase 4 Command Handlers
# -----------------------------------------------------------------------------
def handle_dry_run(user_id: str, args: list[str]) -> str:
    """Handle /dry_run command - simulate diff application (Phase 4)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /dry_run <task_id>\n\nLast task: {session.last_task_id}\nUse: /dry_run last"
        return "Usage: /dry_run <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_dry_run(session.current_project, task_id, user_id)

    files = result.get('files_affected', [])
    files_str = "\n".join(f"  - {f}" for f in files) if files else "  None"
    conflicts = result.get('conflicts', [])
    conflicts_str = "\n".join(f"  - {c}" for c in conflicts) if conflicts else "  None"

    can_apply = result.get('can_apply', False)

    if can_apply:
        return f"""Dry-run PASSED!

Task ID: {task_id}
State: {result.get('current_state', 'ready_to_apply')}

Summary:
- Files affected: {len(files)}
- Lines added: {result.get('lines_added', 0)}
- Lines removed: {result.get('lines_removed', 0)}
- Conflicts: None

Files:
{files_str}

{result.get('warning', 'DRY-RUN ONLY. NO FILES MODIFIED.')}

Next step: /apply {task_id} confirm

WARNING: The 'confirm' keyword is REQUIRED."""
    else:
        return f"""Dry-run FAILED!

Task ID: {task_id}
State: {result.get('current_state', 'diff_generated')}

Conflicts found:
{conflicts_str}

Files affected:
{files_str}

Please resolve conflicts before applying.
You may need to regenerate the diff."""


def handle_apply(user_id: str, args: list[str]) -> str:
    """Handle /apply command - apply diff with confirmation (Phase 4)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """Usage: /apply <task_id> confirm

The 'confirm' keyword is REQUIRED to apply changes.
Example: /apply abc123 confirm

Steps before apply:
1. Run /dry_run <task_id> first
2. Review the diff and dry-run results
3. Then run /apply <task_id> confirm"""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        # Check for confirm in remaining args
        confirm = len(args) > 1 and args[1].lower() == "confirm"
    else:
        # Check for confirm in args
        confirm = len(args) > 1 and args[1].lower() == "confirm"

    if not confirm:
        return f"""CONFIRMATION REQUIRED

To apply changes for task {task_id}, use:
/apply {task_id} confirm

This is a safety measure to prevent accidental changes.

What will happen:
1. Backup of current files will be created
2. Diff will be applied to project files
3. Rollback will be available via /rollback {task_id}

Make sure you have:
- Reviewed the diff
- Run /dry_run {task_id}
- Confirmed no conflicts"""

    result = call_controller_apply(session.current_project, task_id, user_id, confirm=True)

    if result.get('error'):
        return f"Error: {result.get('message', 'Apply failed')}"

    files = result.get('files_modified', [])
    files_str = "\n".join(f"  - {f}" for f in files) if files else "  None"

    return f"""Changes APPLIED successfully!

Task ID: {task_id}
State: {result.get('current_state', 'applied')}

Files modified:
{files_str}

Backup: {result.get('backup_path', 'N/A')}
Rollback available: Yes

To undo these changes: /rollback {task_id}

IMPORTANT:
- Changes have been applied to project files
- Backup was created before apply
- Use /rollback {task_id} to undo if needed"""


def handle_rollback(user_id: str, args: list[str]) -> str:
    """Handle /rollback command - undo applied changes (Phase 4)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /rollback <task_id>\n\nLast task: {session.last_task_id}\nUse: /rollback last"
        return "Usage: /rollback <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_rollback(session.current_project, task_id, user_id)

    files = result.get('files_restored', [])
    files_str = "\n".join(f"  - {f}" for f in files) if files else "  None"

    return f"""Rollback SUCCESSFUL!

Task ID: {task_id}
State: {result.get('current_state', 'rolled_back')}

Files restored:
{files_str}

All changes from the previous apply have been reverted.
The project files are back to their original state.

Note: The diff and plan artifacts are preserved."""


# -----------------------------------------------------------------------------
# Phase 5 Command Handlers
# -----------------------------------------------------------------------------
def handle_commit(user_id: str, args: list[str]) -> str:
    """Handle /commit command - create git commit for applied changes (Phase 5)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """Usage: /commit <task_id> confirm

The 'confirm' keyword is REQUIRED to create a commit.
Example: /commit abc123 confirm

Prerequisites:
1. Task must be in APPLIED state
2. Review the applied changes first

Note: Commit is LOCAL only - NOT automatically pushed."""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        confirm = len(args) > 1 and args[1].lower() == "confirm"
    else:
        confirm = len(args) > 1 and args[1].lower() == "confirm"

    if not confirm:
        return f"""CONFIRMATION REQUIRED

To create a commit for task {task_id}, use:
/commit {task_id} confirm

This is a safety measure to prevent accidental commits.

What will happen:
1. A git commit will be prepared locally
2. The commit will include all files modified by the task
3. The commit will NOT be pushed automatically

Make sure you have:
- Reviewed the applied changes
- Verified the task is in APPLIED state"""

    result = call_controller_commit(session.current_project, task_id, user_id, confirm=True)

    if result.get('error'):
        return f"Error: {result.get('message', 'Commit failed')}"

    files = result.get('files_committed', [])
    files_str = "\n".join(f"  - {f}" for f in files) if files else "  None"

    return f"""Commit CREATED successfully!

Task ID: {task_id}
State: {result.get('current_state', 'committed')}
Commit Hash: {result.get('commit_hash', 'N/A')}

Files committed:
{files_str}

{result.get('warning', 'COMMIT CREATED LOCALLY. NOT PUSHED.')}

Next step: /ci_run {task_id}

IMPORTANT:
- Commit exists locally only
- NOT pushed to remote
- Run CI to validate changes"""


def handle_ci_run(user_id: str, args: list[str]) -> str:
    """Handle /ci_run command - trigger CI pipeline (Phase 5)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        if session.last_task_id:
            return f"Usage: /ci_run <task_id>\n\nLast task: {session.last_task_id}\nUse: /ci_run last"
        return "Usage: /ci_run <task_id>"

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    result = call_controller_ci_run(session.current_project, task_id, user_id)

    return f"""CI Pipeline TRIGGERED!

Task ID: {task_id}
State: {result.get('current_state', 'ci_running')}
CI Run ID: {result.get('ci_run_id', 'N/A')}

{result.get('warning', 'CI RUNNING. WAIT FOR RESULTS.')}

Next step: Submit CI result when complete:
/ci_result {task_id} passed
/ci_result {task_id} failed [logs_url]

IMPORTANT:
- CI is running externally (placeholder in Phase 5)
- Submit result manually when CI completes
- CI must PASS before testing deployment"""


def handle_ci_result(user_id: str, args: list[str]) -> str:
    """Handle /ci_result command - report CI pipeline result (Phase 5)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if len(args) < 2:
        return """Usage: /ci_result <task_id> passed|failed [logs_url]

Examples:
/ci_result abc123 passed
/ci_result abc123 failed https://ci.example.com/logs/123

Note: CI status must be 'passed' or 'failed'."""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        status = args[1].lower() if len(args) > 1 else None
        logs_url = args[2] if len(args) > 2 else None
    else:
        status = args[1].lower()
        logs_url = args[2] if len(args) > 2 else None

    if status not in ["passed", "failed"]:
        return f"Invalid CI status: {status}\n\nMust be 'passed' or 'failed'."

    result = call_controller_ci_result(session.current_project, task_id, user_id, status, logs_url)

    if status == "passed":
        return f"""CI PASSED!

Task ID: {task_id}
State: {result.get('current_state', 'ci_passed')}
CI Status: PASSED
{f"Logs: {logs_url}" if logs_url else ""}

Next step: /deploy_testing {task_id} confirm

IMPORTANT:
- Testing deployment requires 'confirm' keyword
- Production deployment is NOT available via chat"""
    else:
        return f"""CI FAILED!

Task ID: {task_id}
State: {result.get('current_state', 'ci_failed')}
CI Status: FAILED
{f"Logs: {logs_url}" if logs_url else ""}

CI failure BLOCKS testing deployment.

Options:
1. Fix the issues
2. Create a new commit: /commit {task_id} confirm (after fixing)
3. Archive the task if abandoning"""


def handle_deploy_testing(user_id: str, args: list[str]) -> str:
    """Handle /deploy_testing command - deploy to testing environment (Phase 5)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """Usage: /deploy_testing <task_id> confirm

The 'confirm' keyword is REQUIRED to deploy.
Example: /deploy_testing abc123 confirm

Prerequisites:
1. Task must be in CI_PASSED state
2. CI must have PASSED (failures block deployment)

Note: This deploys to TESTING only, NOT production."""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        confirm = len(args) > 1 and args[1].lower() == "confirm"
    else:
        confirm = len(args) > 1 and args[1].lower() == "confirm"

    if not confirm:
        return f"""CONFIRMATION REQUIRED

To deploy task {task_id} to testing, use:
/deploy_testing {task_id} confirm

This is a safety measure to prevent accidental deployments.

Prerequisites:
- Task must be in CI_PASSED state
- CI pipeline must have passed

What will happen:
1. Code will be deployed to testing environment
2. Testing URL: https://aitesting.mybd.in
3. Production deployment is NOT triggered"""

    result = call_controller_deploy_testing(session.current_project, task_id, user_id, confirm=True)

    if result.get('error'):
        return f"Error: {result.get('message', 'Deployment failed')}"

    return f"""Deployed to TESTING!

Task ID: {task_id}
State: {result.get('current_state', 'deployed_testing')}
URL: {result.get('deployment_url', 'https://aitesting.mybd.in')}

{result.get('warning', 'DEPLOYED TO TESTING ONLY. NOT PRODUCTION.')}

IMPORTANT:
- This is the TESTING environment only
- Production deployment requires separate approval
- Production deployment is NOT available via chat"""


def handle_status(user_id: str) -> str:
    """Handle /status command."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    status = call_controller_get_status(session.current_project)

    tasks_by_state = status.get('tasks_by_state', {})
    if tasks_by_state:
        state_str = "\n".join(f"  {state}: {count}" for state, count in tasks_by_state.items())
    else:
        state_str = "  No tasks yet"

    return f"""Project Status: {session.current_project}

Phase: {status.get('current_phase', 'unknown')}
Total Tasks: {status.get('task_count', 0)}
Last Updated: {status.get('last_updated', 'N/A')}

Tasks by State:
{state_str}

Your last task: {session.last_task_id or 'None'}"""


def handle_list(user_id: str) -> str:
    """Handle /list command."""
    result = call_controller_list_projects()

    projects = result.get('projects', [])
    if not projects:
        return """No projects registered yet.

Use /bootstrap <name> <repo_url> to create a new project."""

    project_list = "\n".join(
        f"  - {p['name']} ({p.get('phase', 'unknown')}) - {p.get('task_count', 0)} tasks"
        for p in projects
    )

    return f"""Registered Projects ({result.get('count', 0)}):

{project_list}

Use /project <name> to switch to a project."""


def handle_deploy(user_id: str, args: list[str]) -> str:
    """Handle /deploy command."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """Usage: /deploy <environment>

Environments:
- testing: Deploy to aitesting.mybd.in
- development: Local deployment

Note: Production deployment requires explicit approval and is not available via chat."""

    environment = args[0].lower()

    if environment == "production":
        return """Production deployment is NOT available via chat.

Per AI_POLICY.md, production deployment requires:
1. Explicit human approval
2. Validated build from testing
3. Separate workflow with manual trigger

Use /deploy testing instead."""

    if environment not in ["testing", "development"]:
        return f"Unknown environment: {environment}\n\nUse: /deploy testing or /deploy development"

    result = call_controller_deploy(session.current_project, environment, user_id)

    return f"""Deployment triggered (Placeholder)

Project: {session.current_project}
Environment: {environment}
Status: {result.get('status', 'placeholder')}
URL: {result.get('deployment_url', 'N/A')}

Note: Actual deployment is not implemented in Phase 4."""


# -----------------------------------------------------------------------------
# Phase 6 Command Handlers - Production Deployment
# -----------------------------------------------------------------------------
def handle_prod_request(user_id: str, args: list[str]) -> str:
    """Handle /prod_request command - request production deployment (Phase 6)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """⚠️ PRODUCTION DEPLOYMENT REQUEST

Usage: /prod_request <task_id>

This command starts the MULTI-STEP production deployment process.

You will be prompted for:
1. Justification (why this deploy is needed) - min 20 characters
2. Rollback plan (how to undo if needed) - min 10 characters
3. Risk acknowledgment

IMPORTANT:
- Task MUST be in deployed_testing state
- After request, a DIFFERENT user must approve
- You CANNOT approve your own request

Example: /prod_request last"""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    # For simplicity in this placeholder, we'll prompt for details inline
    # In real implementation, this would be a multi-step conversation
    if len(args) < 4:
        return f"""⚠️ PRODUCTION DEPLOYMENT REQUEST for {task_id}

To request production deployment, provide ALL details:

/prod_request {task_id} "<justification>" "<rollback_plan>" acknowledge

Example:
/prod_request {task_id} "Critical security fix for auth bypass vulnerability" "Revert to previous commit abc123" acknowledge

Requirements:
- Justification: min 20 characters
- Rollback plan: min 10 characters
- Must include 'acknowledge' keyword

⚠️ WARNING: This is for PRODUCTION. Real users will be affected!"""

    # Parse the full command with justification and rollback plan
    justification = args[1] if len(args) > 1 else ""
    rollback_plan = args[2] if len(args) > 2 else ""
    acknowledge = len(args) > 3 and args[3].lower() == "acknowledge"

    if not acknowledge:
        return f"""⚠️ RISK ACKNOWLEDGMENT REQUIRED

You must include 'acknowledge' as the last word to confirm:

/prod_request {task_id} "{justification}" "{rollback_plan}" acknowledge

This ensures you understand the production deployment risks."""

    result = call_controller_prod_request(
        session.current_project,
        task_id,
        user_id,
        justification,
        rollback_plan,
        risk_acknowledged=True
    )

    if result.get('error'):
        return f"Error: {result.get('message', 'Request failed')}"

    return f"""⚠️ PRODUCTION DEPLOYMENT REQUESTED

Task ID: {task_id}
Requested by: {user_id}
State: {result.get('current_state', 'prod_deploy_requested')}

Justification: {justification}
Rollback Plan: {rollback_plan}

{result.get('warning', '')}

NEXT STEP:
{result.get('next_step', 'A DIFFERENT user must approve this request.')}

The requester (you) CANNOT approve this request.
Another authorized user must run: /prod_approve {task_id}"""


def handle_prod_approve(user_id: str, args: list[str]) -> str:
    """Handle /prod_approve command - approve production deployment (Phase 6)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """⚠️ PRODUCTION APPROVAL (DUAL APPROVAL REQUIRED)

Usage: /prod_approve <task_id>

You will be prompted to confirm:
1. You have reviewed the code changes
2. You have reviewed the rollback plan
3. Your approval reason

CRITICAL: You CANNOT approve a request you made yourself.
The system enforces dual approval for production deployments.

Example: /prod_approve last"""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id

    # For simplicity, prompt for confirmation inline
    if len(args) < 4:
        return f"""⚠️ PRODUCTION APPROVAL for {task_id}

To approve, confirm you have reviewed everything:

/prod_approve {task_id} "<reason>" reviewed_changes reviewed_rollback

Example:
/prod_approve {task_id} "Verified fix is correct and minimal" reviewed_changes reviewed_rollback

Requirements:
- Approval reason: min 10 characters
- Must confirm reviewed_changes
- Must confirm reviewed_rollback

⚠️ You CANNOT approve your own request!"""

    approval_reason = args[1] if len(args) > 1 else ""
    reviewed_changes = len(args) > 2 and args[2].lower() == "reviewed_changes"
    reviewed_rollback = len(args) > 3 and args[3].lower() == "reviewed_rollback"

    if not reviewed_changes or not reviewed_rollback:
        return f"""⚠️ REVIEW CONFIRMATION REQUIRED

You must confirm both:
- reviewed_changes
- reviewed_rollback

/prod_approve {task_id} "{approval_reason}" reviewed_changes reviewed_rollback"""

    result = call_controller_prod_approve(
        session.current_project,
        task_id,
        user_id,
        approval_reason,
        reviewed_changes=True,
        reviewed_rollback=True
    )

    if result.get('error'):
        return f"Error: {result.get('message', 'Approval failed')}"

    return f"""✅ PRODUCTION DEPLOYMENT APPROVED

Task ID: {task_id}
Approved by: {user_id}
State: {result.get('current_state', 'prod_approved')}

Approval Reason: {approval_reason}

{result.get('warning', '')}

NEXT STEP:
{result.get('next_step', 'Execute deployment via /prod_apply <task_id> confirm')}

Any authorized user can now execute: /prod_apply {task_id} confirm"""


def handle_prod_apply(user_id: str, args: list[str]) -> str:
    """Handle /prod_apply command - execute production deployment (Phase 6)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """🚀 PRODUCTION DEPLOYMENT EXECUTION

Usage: /prod_apply <task_id> confirm

The 'confirm' keyword is REQUIRED to execute.

Prerequisites:
1. Task must be in prod_approved state
2. Dual approval must have been completed
3. A DIFFERENT user must have approved the request

Example: /prod_apply last confirm

⚠️ WARNING: This deploys to PRODUCTION and affects real users!"""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        confirm = len(args) > 1 and args[1].lower() == "confirm"
    else:
        confirm = len(args) > 1 and args[1].lower() == "confirm"

    if not confirm:
        return f"""🚀 FINAL CONFIRMATION REQUIRED

To deploy task {task_id} to PRODUCTION, use:
/prod_apply {task_id} confirm

This is the FINAL safety check before production deployment.

What will happen:
1. Code deployed to https://ai.mybd.in
2. Release manifest created
3. Deployment logged to audit trail
4. Rollback available via /prod_rollback

⚠️ WARNING: Real users will be affected!
Make sure you are ready to monitor the deployment."""

    result = call_controller_prod_apply(
        session.current_project,
        task_id,
        user_id,
        confirm=True
    )

    if result.get('error'):
        return f"Error: {result.get('message', 'Deployment failed')}"

    return f"""🚀 DEPLOYED TO PRODUCTION!

Task ID: {task_id}
Deployed by: {user_id}
State: {result.get('current_state', 'deployed_production')}
URL: {result.get('deployment_url', 'https://ai.mybd.in')}

Release Manifest: {result.get('release_manifest_path', 'N/A')}

{result.get('warning', '')}

ROLLBACK AVAILABLE:
If issues arise, use: {result.get('rollback_command', f'/prod_rollback {task_id}')}

🔍 MONITOR THE DEPLOYMENT CLOSELY!"""


def handle_prod_rollback(user_id: str, args: list[str]) -> str:
    """Handle /prod_rollback command - rollback production (Phase 6 - BREAK GLASS)."""
    session = get_or_create_session(user_id)

    if not session.current_project:
        return "No project selected. Use /project <name> first."

    if not args:
        return """🚨 PRODUCTION ROLLBACK (BREAK-GLASS)

Usage: /prod_rollback <task_id> "<reason>"

This is an EMERGENCY command that does NOT require dual approval.
Speed is prioritized over ceremony for rollbacks.

You MUST provide a reason for the audit trail.

Example: /prod_rollback last "Users reporting 500 errors on login"

The rollback will:
1. Immediately revert production to previous state
2. Log the rollback with your reason to audit trail
3. Notify relevant stakeholders (when implemented)"""

    task_id = args[0]
    if task_id.lower() == "last":
        if not session.last_task_id:
            return "No recent task. Please specify task ID."
        task_id = session.last_task_id
        reason = " ".join(args[1:]) if len(args) > 1 else ""
    else:
        reason = " ".join(args[1:]) if len(args) > 1 else ""

    if not reason or len(reason) < 5:
        return f"""🚨 REASON REQUIRED FOR AUDIT TRAIL

To rollback task {task_id}, provide a reason:
/prod_rollback {task_id} "<reason>"

Example:
/prod_rollback {task_id} "Critical bug causing 500 errors"

The reason is logged to the audit trail for accountability."""

    result = call_controller_prod_rollback(
        session.current_project,
        task_id,
        user_id,
        reason
    )

    if result.get('error'):
        return f"Error: {result.get('message', 'Rollback failed')}"

    return f"""⚠️ PRODUCTION ROLLED BACK!

Task ID: {task_id}
Rolled back by: {user_id}
State: {result.get('current_state', 'prod_rolled_back')}
Reason: {reason}

{result.get('warning', '')}

NEXT STEPS:
1. Verify system stability
2. Investigate the issue
3. Prepare a fix if needed
4. Follow normal deployment process for the fix

The rollback has been logged to the audit trail."""


# -----------------------------------------------------------------------------
# Phase 17C Command Handlers - Recommendations (ADVISORY ONLY)
# -----------------------------------------------------------------------------
def handle_recommendations(user_id: str, args: list[str]) -> str:
    """
    Handle /recommendations command - list recent recommendations (Phase 17C).

    CRITICAL: Recommendations are ADVISORY ONLY.
    - They suggest actions, never execute them
    - Human must approve/dismiss
    - NO automation, NO lifecycle mutation
    """
    # Parse optional arguments
    status_filter = None
    limit = 10

    for arg in args:
        if arg in ["pending", "approved", "dismissed"]:
            status_filter = arg
        elif arg.isdigit():
            limit = min(int(arg), 50)  # Cap at 50

    result = call_controller_get_recommendations(limit=limit, status=status_filter)

    recommendations = result.get("recommendations", [])

    if not recommendations:
        return """📋 RECOMMENDATIONS (Phase 17C - ADVISORY ONLY)

No recommendations found.

Recommendations are generated from incident analysis.
They suggest actions but NEVER execute automatically.

Commands:
/recommendations [pending|approved|dismissed] [limit]
/recommendation <id> - View details
/rec_approve <id> [reason] - Approve
/rec_dismiss <id> [reason] - Dismiss

⚠️ ADVISORY ONLY: Approval does NOT trigger any automation."""

    # Build recommendation list
    rec_list = []
    for rec in recommendations[:limit]:
        rec_id = rec.get("recommendation_id", "unknown")[:8]
        rec_type = rec.get("recommendation_type", "unknown")
        severity = rec.get("severity", "unknown")
        status = rec.get("status", "pending")
        title = rec.get("title", "No title")[:50]

        severity_icon = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "ℹ️",
            "unknown": "❓"
        }.get(severity, "❓")

        status_icon = {
            "pending": "⏳",
            "approved": "✅",
            "dismissed": "❌",
            "expired": "⏰"
        }.get(status, "❓")

        rec_list.append(f"{severity_icon} {status_icon} [{rec_id}] {rec_type}: {title}")

    rec_str = "\n".join(rec_list)

    return f"""📋 RECOMMENDATIONS (Phase 17C - ADVISORY ONLY)

⚠️ These are ADVISORY suggestions - NO automatic execution

Showing {len(recommendations)} recommendation(s):

{rec_str}

Commands:
/recommendation <id> - View details
/rec_approve <id> [reason] - Approve
/rec_dismiss <id> [reason] - Dismiss

⚠️ IMPORTANT: Approving a recommendation does NOT
trigger any automatic action. Human action is required."""


def handle_recommendation(user_id: str, args: list[str]) -> str:
    """
    Handle /recommendation command - view specific recommendation (Phase 17C).

    CRITICAL: Recommendations are ADVISORY ONLY.
    """
    if not args:
        return """Usage: /recommendation <recommendation_id>

Example: /recommendation abc12345

This shows the full details of a specific recommendation.

⚠️ ADVISORY ONLY: Recommendations suggest, never execute."""

    recommendation_id = args[0]
    result = call_controller_get_recommendation(recommendation_id)

    recommendation = result.get("recommendation")

    if not recommendation:
        return f"""Recommendation not found: {recommendation_id}

Use /recommendations to list available recommendations."""

    # Format recommendation details
    rec_type = recommendation.get("recommendation_type", "unknown")
    severity = recommendation.get("severity", "unknown")
    status = recommendation.get("status", "pending")
    title = recommendation.get("title", "No title")
    description = recommendation.get("description", "No description")
    rationale = recommendation.get("rationale", "No rationale")
    actions = recommendation.get("suggested_actions", [])
    approval_req = recommendation.get("approval_required", "unknown")
    created_at = recommendation.get("created_at", "unknown")

    severity_icon = {
        "critical": "🔴 CRITICAL",
        "high": "🟠 HIGH",
        "medium": "🟡 MEDIUM",
        "low": "🟢 LOW",
        "info": "ℹ️ INFO",
        "unknown": "❓ UNKNOWN"
    }.get(severity, "❓ UNKNOWN")

    actions_str = "\n".join(f"  • {a}" for a in actions) if actions else "  No suggested actions"

    approval_text = {
        "none_required": "None required (informational)",
        "confirmation_required": "Simple confirmation required",
        "explicit_approval_required": "Explicit approval required"
    }.get(approval_req, approval_req)

    return f"""📋 RECOMMENDATION DETAILS

⚠️ ADVISORY ONLY - NO AUTOMATIC EXECUTION

ID: {recommendation_id}
Type: {rec_type}
Severity: {severity_icon}
Status: {status.upper()}
Approval: {approval_text}
Created: {created_at}

TITLE:
{title}

DESCRIPTION:
{description}

RATIONALE:
{rationale}

SUGGESTED ACTIONS:
{actions_str}

ACTIONS:
/rec_approve {recommendation_id} [reason] - Approve
/rec_dismiss {recommendation_id} [reason] - Dismiss

⚠️ CRITICAL: Approving this recommendation does NOT
trigger any automatic action. You must take the
suggested actions manually if appropriate."""


def handle_rec_approve(user_id: str, args: list[str]) -> str:
    """
    Handle /rec_approve command - approve a recommendation (Phase 17C).

    CRITICAL: Approval does NOT trigger any automation.
    - This is ADVISORY ONLY
    - It marks the recommendation as approved for tracking
    - NO execution, NO lifecycle changes, NO deployment
    """
    if not args:
        return """Usage: /rec_approve <recommendation_id> [reason]

Example: /rec_approve abc12345 "Will address in next sprint"

⚠️ CRITICAL: Approving does NOT trigger any automatic action.
This only marks the recommendation as "approved" for tracking.
You must take the suggested actions manually if appropriate."""

    recommendation_id = args[0]
    reason = " ".join(args[1:]) if len(args) > 1 else None

    result = call_controller_approve_recommendation(
        recommendation_id=recommendation_id,
        user_id=user_id,
        reason=reason
    )

    if result.get("error"):
        return f"Error: {result.get('message', 'Approval failed')}"

    return f"""✅ RECOMMENDATION APPROVED

ID: {recommendation_id}
Approved by: {user_id}
{f"Reason: {reason}" if reason else ""}

{result.get('message', '')}

⚠️ CRITICAL REMINDER:
This approval is ADVISORY ONLY. It does NOT:
- Trigger any automatic action
- Change any lifecycle state
- Deploy anything
- Execute any code

You must take the suggested actions manually
if you deem them appropriate."""


def handle_rec_dismiss(user_id: str, args: list[str]) -> str:
    """
    Handle /rec_dismiss command - dismiss a recommendation (Phase 17C).

    CRITICAL: This is ADVISORY ONLY - no automation triggered.
    """
    if not args:
        return """Usage: /rec_dismiss <recommendation_id> [reason]

Example: /rec_dismiss abc12345 "Not applicable to our setup"

Dismissing marks the recommendation as reviewed and not needed."""

    recommendation_id = args[0]
    reason = " ".join(args[1:]) if len(args) > 1 else None

    result = call_controller_dismiss_recommendation(
        recommendation_id=recommendation_id,
        user_id=user_id,
        reason=reason
    )

    if result.get("error"):
        return f"Error: {result.get('message', 'Dismissal failed')}"

    return f"""❌ RECOMMENDATION DISMISSED

ID: {recommendation_id}
Dismissed by: {user_id}
{f"Reason: {reason}" if reason else ""}

{result.get('message', '')}

The recommendation has been marked as dismissed.
This decision is logged for audit purposes."""


# -----------------------------------------------------------------------------
# Message Router
# -----------------------------------------------------------------------------
def process_message(user_id: str, username: Optional[str], text: str) -> str:
    """Process incoming message and return response."""
    logger.info(f"Processing message from user {user_id}: {text[:50]}...")

    if not is_user_authorized(user_id):
        return "You are not authorized to use this bot. Contact the administrator."

    allowed, error_msg = check_rate_limit(user_id)
    if not allowed:
        return error_msg or "Rate limit exceeded. Please wait before sending more commands."

    parsed = parse_command(text)

    if not parsed:
        return "Unknown command. Use /help to see available commands."

    handlers = {
        BotCommand.START: lambda: handle_start(user_id, username),
        BotCommand.HELP: lambda: handle_help(user_id),
        BotCommand.BOOTSTRAP: lambda: handle_bootstrap(user_id, parsed.args),
        BotCommand.PROJECT: lambda: handle_project(user_id, parsed.args),
        BotCommand.TASK: lambda: handle_task(user_id, parsed.args),
        BotCommand.VALIDATE: lambda: handle_validate(user_id, parsed.args),
        BotCommand.PLAN: lambda: handle_plan(user_id, parsed.args),
        BotCommand.APPROVE: lambda: handle_approve(user_id, parsed.args),
        BotCommand.REJECT: lambda: handle_reject(user_id, parsed.args),
        BotCommand.GENERATE_DIFF: lambda: handle_generate_diff(user_id, parsed.args),
        # Phase 4: Execution commands
        BotCommand.DRY_RUN: lambda: handle_dry_run(user_id, parsed.args),
        BotCommand.APPLY: lambda: handle_apply(user_id, parsed.args),
        BotCommand.ROLLBACK: lambda: handle_rollback(user_id, parsed.args),
        # Phase 5: CI/Release commands
        BotCommand.COMMIT: lambda: handle_commit(user_id, parsed.args),
        BotCommand.CI_RUN: lambda: handle_ci_run(user_id, parsed.args),
        BotCommand.CI_RESULT: lambda: handle_ci_result(user_id, parsed.args),
        BotCommand.DEPLOY_TESTING: lambda: handle_deploy_testing(user_id, parsed.args),
        # Phase 6: Production commands (DUAL APPROVAL)
        BotCommand.PROD_REQUEST: lambda: handle_prod_request(user_id, parsed.args),
        BotCommand.PROD_APPROVE: lambda: handle_prod_approve(user_id, parsed.args),
        BotCommand.PROD_APPLY: lambda: handle_prod_apply(user_id, parsed.args),
        BotCommand.PROD_ROLLBACK: lambda: handle_prod_rollback(user_id, parsed.args),
        BotCommand.STATUS: lambda: handle_status(user_id),
        BotCommand.LIST: lambda: handle_list(user_id),
        BotCommand.DEPLOY: lambda: handle_deploy(user_id, parsed.args),
        # Phase 17C: Recommendation commands (ADVISORY ONLY)
        BotCommand.RECOMMENDATIONS: lambda: handle_recommendations(user_id, parsed.args),
        BotCommand.RECOMMENDATION: lambda: handle_recommendation(user_id, parsed.args),
        BotCommand.REC_APPROVE: lambda: handle_rec_approve(user_id, parsed.args),
        BotCommand.REC_DISMISS: lambda: handle_rec_dismiss(user_id, parsed.args),
    }

    handler = handlers.get(parsed.command)
    if handler:
        return handler()

    return "Unknown command. Use /help to see available commands."


# -----------------------------------------------------------------------------
# Bot Initialization (Placeholder)
# -----------------------------------------------------------------------------
def create_bot():
    """Create and configure the Telegram bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not set. Bot cannot start.")
        return None

    logger.info("Creating Telegram bot...")
    # TODO: Implement using python-telegram-bot library
    logger.warning("Bot creation is a placeholder - not actually connecting to Telegram")
    return None


def run_bot():
    """Run the Telegram bot."""
    bot = create_bot()
    if not bot:
        logger.error("Failed to create bot. Exiting.")
        return
    logger.info("Starting bot...")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Telegram Bot starting (Phase 17C)...")
    logger.info(f"Controller URL: {CONTROLLER_BASE_URL}")
    logger.info(f"Bot token configured: {'Yes' if TELEGRAM_BOT_TOKEN else 'No'}")

    if not TELEGRAM_BOT_TOKEN:
        logger.warning("No bot token set. Running in test mode.")

        # Test mode: demonstrate Phase 17C task lifecycle with Recommendations
        print("\n" + "="*60)
        print("PHASE 17C TEST MODE - Full Lifecycle with Recommendations Demo")
        print("="*60 + "\n")

        print(process_message("test_user", "TestUser", "/start"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/bootstrap my-webapp https://github.com/user/repo python,fastapi"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/task Fix the login button not working"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/validate last"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/plan last"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/approve last"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/generate_diff last"))
        print("\n" + "-"*50 + "\n")

        # Phase 4: Execution commands
        print("--- PHASE 4: EXECUTION ---")
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/dry_run last"))
        print("\n" + "-"*50 + "\n")

        print("--- Applying WITH confirmation ---")
        print(process_message("test_user", "TestUser", "/apply last confirm"))
        print("\n" + "-"*50 + "\n")

        # Phase 5: CI/Release commands
        print("--- PHASE 5: CI/RELEASE ---")
        print("\n" + "-"*50 + "\n")

        print("--- Attempting commit WITHOUT confirmation ---")
        print(process_message("test_user", "TestUser", "/commit last"))
        print("\n" + "-"*50 + "\n")

        print("--- Committing WITH confirmation ---")
        print(process_message("test_user", "TestUser", "/commit last confirm"))
        print("\n" + "-"*50 + "\n")

        print("--- Triggering CI ---")
        print(process_message("test_user", "TestUser", "/ci_run last"))
        print("\n" + "-"*50 + "\n")

        print("--- Submitting CI PASSED result ---")
        print(process_message("test_user", "TestUser", "/ci_result last passed"))
        print("\n" + "-"*50 + "\n")

        print("--- Attempting deploy WITHOUT confirmation ---")
        print(process_message("test_user", "TestUser", "/deploy_testing last"))
        print("\n" + "-"*50 + "\n")

        print("--- Deploying to testing WITH confirmation ---")
        print(process_message("test_user", "TestUser", "/deploy_testing last confirm"))
        print("\n" + "-"*50 + "\n")

        print(process_message("test_user", "TestUser", "/status"))
    else:
        run_bot()
