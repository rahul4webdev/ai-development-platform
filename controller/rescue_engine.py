"""
Rescue Engine for AI Development Platform
Phase 22: Rescue & Recovery System

Creates and manages rescue jobs for deployment failures.

Features:
- Rescue job creation with failure-specific instructions
- Max attempt tracking (3 attempts per project)
- State persistence across restarts
- Server-aware diagnostic instructions

CONSTRAINTS:
- Max 3 rescue attempts per project per deployment
- Human notification at every step
- No auto-production deployment
- All attempts logged for audit
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("rescue_engine")

# Configuration
RESCUE_MAX_ATTEMPTS = 3
RESCUE_STATE_FILE = Path("/home/aitesting.mybd.in/jobs/rescue_state.json")

# Fallback for local development
if not RESCUE_STATE_FILE.parent.exists():
    RESCUE_STATE_FILE = Path("/tmp/rescue_state.json")


@dataclass
class RescueAttempt:
    """Tracks a single rescue attempt."""
    attempt_number: int
    job_id: str
    failure_type: str
    created_at: str  # ISO format
    completed_at: Optional[str] = None
    success: bool = False
    outcome: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "RescueAttempt":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ProjectRescueState:
    """Tracks rescue state for a project."""
    project_name: str
    deployment_job_id: str
    attempts: List[RescueAttempt] = field(default_factory=list)
    last_failure_type: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved: bool = False
    resolved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "deployment_job_id": self.deployment_job_id,
            "attempts": [a.to_dict() for a in self.attempts],
            "last_failure_type": self.last_failure_type,
            "created_at": self.created_at,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProjectRescueState":
        """Create from dictionary."""
        attempts = [RescueAttempt.from_dict(a) for a in data.get("attempts", [])]
        return cls(
            project_name=data["project_name"],
            deployment_job_id=data["deployment_job_id"],
            attempts=attempts,
            last_failure_type=data.get("last_failure_type"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            resolved=data.get("resolved", False),
            resolved_at=data.get("resolved_at")
        )


# Rescue task template
RESCUE_TASK_TEMPLATE = """RESCUE JOB - Deployment Failure Recovery

PROJECT: {project_name}
FAILURE TYPE: {failure_type}
DEPLOYMENT JOB: {deployment_job_id}
ATTEMPT: {attempt_number}/{max_attempts}
GITHUB REPO: {github_repo}

==============================================================================
FAILED ENDPOINTS WITH ACTUAL ERRORS
==============================================================================
{failed_endpoints_details}

==============================================================================
SUCCESSFUL ENDPOINTS
==============================================================================
{successful_endpoints_list}

==============================================================================
DIAGNOSTIC INFORMATION
==============================================================================
{diagnostic_info}

==============================================================================
SUGGESTED FIXES
==============================================================================
{suggested_fixes_list}

{server_instructions}

==============================================================================
PROJECT DIRECTORY
==============================================================================
The project code is available at: {project_directory}

You do NOT need to clone the repository. The code is already on the platform.

==============================================================================
INSTRUCTIONS
==============================================================================
1. Analyze the ACTUAL ERROR RESPONSES above for each failing endpoint
2. Read the project code in {project_directory}
3. Read the CHD (requirements_raw in registry) for deployment configuration
4. Identify the root cause based on:
   - The failure type: {failure_type}
   - The actual error responses from each domain
   - The deployment configuration in CHD
5. Implement the required fix:
   - Fix code issues, configuration, or deployment scripts
   - Update GitHub Actions workflow if needed
6. After implementing fixes:
   a. Commit changes with clear message
   b. Push to GitHub: git push origin main
   c. If workflow needs manual trigger, note it in RESCUE_LOG.md
7. Document the fix in logs/RESCUE_LOG.md

==============================================================================
CRITICAL RULES
==============================================================================
- Focus ONLY on fixing the deployment issue
- Read actual error responses carefully - they contain the root cause
- Push changes to GitHub - deployment happens via GitHub Actions workflow
- Do NOT deploy directly via SSH
- If the issue cannot be fixed automatically, document in logs/BLOCKERS.md

VALIDATION WILL RE-RUN AUTOMATICALLY AFTER DEPLOYMENT COMPLETES.
"""

# Server-specific instructions
SERVER_INSTRUCTIONS_KNOWN = """
SERVER ENVIRONMENT: {environment}
WEB SERVER: {web_server}
CONFIG PATH: {config_path}
VHOST PATH: {vhost_path}

Use the server-specific paths above when checking configurations.
"""

SERVER_INSTRUCTIONS_UNKNOWN = """
DISCOVERY PHASE - Server Environment Unknown

Before proceeding with fixes, discover the server environment:

1. Check for control panels:
   - ls /usr/local/CyberCP 2>/dev/null && echo "CyberPanel detected"
   - ls /usr/local/cpanel 2>/dev/null && echo "cPanel detected"
   - ls /usr/local/psa 2>/dev/null && echo "Plesk detected"
   - ls /www/server/panel 2>/dev/null && echo "aaPanel detected"

2. Check web server:
   - which nginx && nginx -v
   - which apache2 || which httpd
   - ls /usr/local/lsws 2>/dev/null && echo "OpenLiteSpeed detected"

3. Find virtual host configs:
   - CyberPanel: /usr/local/lsws/conf/vhosts/{domain}/vhostconf.conf
   - Nginx: /etc/nginx/sites-available/* or /etc/nginx/conf.d/*
   - Apache: /etc/apache2/sites-available/* or /etc/httpd/conf.d/*

4. Document discovered environment in logs/SERVER_DISCOVERY.md

5. Proceed with appropriate fix based on discovered server type.
"""


class RescueJobEngine:
    """
    Manages rescue jobs for deployment failures.

    CONSTRAINTS:
    - Max 3 rescue attempts per project per deployment
    - Human notification required after max attempts
    - All attempts logged for audit
    """

    def __init__(self):
        self._state: Dict[str, ProjectRescueState] = {}
        self._load_state()

    def _load_state(self) -> None:
        """Load rescue state from file."""
        try:
            if RESCUE_STATE_FILE.exists():
                with open(RESCUE_STATE_FILE) as f:
                    data = json.load(f)
                    for key, value in data.items():
                        self._state[key] = ProjectRescueState.from_dict(value)
                logger.info(f"Loaded rescue state: {len(self._state)} projects")
        except Exception as e:
            logger.warning(f"Failed to load rescue state: {e}")
            self._state = {}

    def _save_state(self) -> None:
        """Save rescue state to file."""
        try:
            RESCUE_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(RESCUE_STATE_FILE, "w") as f:
                json.dump(
                    {k: v.to_dict() for k, v in self._state.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save rescue state: {e}")

    def can_attempt_rescue(
        self,
        project_name: str,
        deployment_job_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Check if rescue attempt is allowed.

        Args:
            project_name: Name of the project
            deployment_job_id: Optional deployment job ID to check specific deployment

        Returns:
            (can_rescue, reason)
        """
        state = self._state.get(project_name)

        if state is None:
            return True, "No previous rescue attempts"

        # Check if this is a different deployment
        if deployment_job_id and state.deployment_job_id != deployment_job_id:
            # New deployment - reset state
            return True, "New deployment - resetting rescue state"

        # Check if already resolved
        if state.resolved:
            return True, "Previous rescue was successful"

        # Check attempt count
        if len(state.attempts) >= RESCUE_MAX_ATTEMPTS:
            return False, f"Max rescue attempts ({RESCUE_MAX_ATTEMPTS}) reached"

        return True, f"Attempt {len(state.attempts) + 1} of {RESCUE_MAX_ATTEMPTS}"

    async def create_rescue_job(
        self,
        project_name: str,
        failure: Any,  # DeploymentFailure from deployment_validator
        source_deployment_job_id: str,
        controller_client: Any,  # ControllerClient
        server_config: Optional[Any] = None  # ServerConfig from server_detector
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Create a rescue job for a deployment failure.

        Args:
            project_name: Name of the project
            failure: DeploymentFailure object
            source_deployment_job_id: ID of the failed deployment job
            controller_client: Client to create jobs
            server_config: Optional server configuration

        Returns:
            (success, message, job_id)
        """
        # Check if we can attempt rescue
        can_rescue, reason = self.can_attempt_rescue(project_name, source_deployment_job_id)

        if not can_rescue:
            return False, reason, None

        # Get or create state
        state = self._state.get(project_name)
        if state is None or state.deployment_job_id != source_deployment_job_id:
            state = ProjectRescueState(
                project_name=project_name,
                deployment_job_id=source_deployment_job_id
            )
            self._state[project_name] = state

        attempt_number = len(state.attempts) + 1

        # Generate task description
        task_description = self._generate_rescue_task_description(
            project_name=project_name,
            failure=failure,
            attempt_number=attempt_number,
            server_config=server_config
        )

        try:
            # Create the rescue job
            result = await controller_client.create_claude_job(
                project_name=project_name,
                task_description=task_description,
                task_type="rescue",
                copy_from_job=source_deployment_job_id,
                copy_artifacts=["PLANNING_OUTPUT.yaml", "CURRENT_STATE.md", "DEPLOYMENT.md", project_name]
            )

            if result.get("job_id"):
                # Record the attempt
                attempt = RescueAttempt(
                    attempt_number=attempt_number,
                    job_id=result["job_id"],
                    failure_type=failure.failure_type.value if hasattr(failure.failure_type, 'value') else str(failure.failure_type),
                    created_at=datetime.utcnow().isoformat()
                )
                state.attempts.append(attempt)
                state.last_failure_type = attempt.failure_type
                self._save_state()

                logger.info(f"Created rescue job {result['job_id']} for {project_name} (attempt {attempt_number})")
                return True, f"Rescue job created (attempt {attempt_number}/{RESCUE_MAX_ATTEMPTS})", result["job_id"]
            else:
                error = result.get("error", {}).get("message", "Unknown error")
                return False, f"Failed to create rescue job: {error}", None

        except Exception as e:
            logger.error(f"Failed to create rescue job: {e}")
            return False, f"Failed to create rescue job: {e}", None

    def _generate_rescue_task_description(
        self,
        project_name: str,
        failure: Any,
        attempt_number: int,
        server_config: Optional[Any] = None
    ) -> str:
        """Generate task description for Claude."""
        from controller.deployment_validator import (
            get_project_directory,
            get_github_repo_url
        )

        # Format failed endpoints with detailed error info
        failed_details_parts = []
        for f in failure.failed_urls:
            url = f.get('url', 'Unknown')
            status = f.get('status_code', 'N/A')
            error = f.get('error', 'Unknown error')
            response_time = f.get('response_time_ms', 'N/A')
            response_preview = f.get('response_body_preview', '')

            detail = f"""
URL: {url}
Status Code: {status}
Error: {error}
Response Time: {response_time}ms
Response Body Preview:
{response_preview[:1000] if response_preview else 'No response body'}
---"""
            failed_details_parts.append(detail)

        failed_endpoints_details = "\n".join(failed_details_parts) or "No failed endpoints"

        # Format successful endpoints
        successful_endpoints_list = "\n".join([
            f"- {url}" for url in failure.successful_urls
        ]) or "- None"

        # Format diagnostic info
        diagnostic_info = json.dumps(failure.diagnostic_info, indent=2)

        # Format suggested fixes
        suggested_fixes_list = "\n".join([
            f"- {fix}" for fix in failure.suggested_fixes
        ]) or "- See server-specific instructions below"

        # Get project directory
        project_dir = get_project_directory(project_name)
        project_directory = str(project_dir) if project_dir else f"/home/aitesting.mybd.in/public_html/projects/{project_name}"

        # Get GitHub repo URL
        github_repo = get_github_repo_url(project_name) or "Not found in CHD"

        # Server instructions
        if server_config and hasattr(server_config, 'environment'):
            from controller.server_detector import ServerEnvironment
            if server_config.environment != ServerEnvironment.UNKNOWN:
                server_instructions = SERVER_INSTRUCTIONS_KNOWN.format(
                    environment=server_config.environment.value,
                    web_server=server_config.web_server.value if hasattr(server_config.web_server, 'value') else server_config.web_server,
                    config_path=server_config.config_path,
                    vhost_path=server_config.vhost_path
                )
            else:
                server_instructions = SERVER_INSTRUCTIONS_UNKNOWN
        else:
            server_instructions = SERVER_INSTRUCTIONS_UNKNOWN

        return RESCUE_TASK_TEMPLATE.format(
            project_name=project_name,
            failure_type=failure.failure_type.value if hasattr(failure.failure_type, 'value') else str(failure.failure_type),
            deployment_job_id=failure.deployment_job_id,
            attempt_number=attempt_number,
            max_attempts=RESCUE_MAX_ATTEMPTS,
            failed_endpoints_details=failed_endpoints_details,
            successful_endpoints_list=successful_endpoints_list,
            diagnostic_info=diagnostic_info,
            suggested_fixes_list=suggested_fixes_list,
            server_instructions=server_instructions,
            project_directory=project_directory,
            github_repo=github_repo
        )

    def mark_rescue_success(self, project_name: str, job_id: str) -> None:
        """Mark a rescue attempt as successful."""
        state = self._state.get(project_name)
        if state:
            state.resolved = True
            state.resolved_at = datetime.utcnow().isoformat()
            for attempt in state.attempts:
                if attempt.job_id == job_id:
                    attempt.success = True
                    attempt.completed_at = datetime.utcnow().isoformat()
                    attempt.outcome = "Validation passed after rescue"
                    break
            self._save_state()
            logger.info(f"Marked rescue success for {project_name}")

    def mark_rescue_failure(self, project_name: str, job_id: str, reason: str) -> None:
        """Mark a rescue attempt as failed."""
        state = self._state.get(project_name)
        if state:
            for attempt in state.attempts:
                if attempt.job_id == job_id:
                    attempt.completed_at = datetime.utcnow().isoformat()
                    attempt.outcome = reason
                    break
            self._save_state()
            logger.info(f"Marked rescue failure for {project_name}: {reason}")

    def get_rescue_history(self, project_name: str) -> List[RescueAttempt]:
        """Get rescue attempt history for a project."""
        state = self._state.get(project_name)
        return state.attempts if state else []

    def get_rescue_state(self, project_name: str) -> Optional[ProjectRescueState]:
        """Get full rescue state for a project."""
        return self._state.get(project_name)

    def clear_rescue_state(self, project_name: str) -> None:
        """Clear rescue state for a project (use when starting fresh deployment)."""
        if project_name in self._state:
            del self._state[project_name]
            self._save_state()
            logger.info(f"Cleared rescue state for {project_name}")


# Singleton instance
_engine: Optional[RescueJobEngine] = None


def get_rescue_engine() -> RescueJobEngine:
    """Get singleton rescue engine instance."""
    global _engine
    if _engine is None:
        _engine = RescueJobEngine()
    return _engine
