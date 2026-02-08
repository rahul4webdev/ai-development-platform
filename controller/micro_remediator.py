"""
Phase 24.5: Self-Healing Micro-Remediation Layer

Deterministic failure classification and targeted fix generation
for individual E2E check failures detected by the Deep E2E Verifier.

CONSTRAINTS:
- Max 3 micro-remediation attempts per project per failure pattern
- Deterministic classification rules (no ML, no heuristics)
- All state stored in Project.metadata (no separate state file)
- Fail-open: ImportError/exceptions must not crash the caller
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("micro_remediator")

MICRO_REMEDIATION_MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FailureOwner(str, Enum):
    FRONTEND = "frontend"
    BACKEND = "backend"
    INFRA = "infra"
    DEPLOYMENT = "deployment"
    PLATFORM = "platform"
    UNKNOWN = "unknown"


class FailurePattern(str, Enum):
    CORS = "cors"
    HTTP_404 = "http_404"
    HTTP_500 = "http_500"
    HTTP_503 = "http_503"
    AUTH_FAILURE = "auth_failure"
    DATA_MISSING = "data_missing"
    CONSOLE_ERROR = "console_error"
    BUILD_FAILURE = "build_failure"
    RESCUE_JOB_FAILED = "rescue_job_failed"
    UNKNOWN = "unknown"


class RemediationStatus(str, Enum):
    PENDING = "pending"
    FIXING = "fixing"
    RETESTING = "retesting"
    RESOLVED = "resolved"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MicroRemediationTask:
    project: str
    failure_pattern: str      # FailurePattern value
    owner: str                # FailureOwner value
    evidence: str             # Human-readable evidence from the check
    suggested_fix: str        # One-line suggested fix
    requires_redeploy: bool
    check_type: str           # E2ECheckType value that triggered this
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Suggested fix map
# ---------------------------------------------------------------------------

_SUGGESTED_FIXES: Dict[Tuple[str, str], str] = {
    (FailurePattern.CORS.value, FailureOwner.BACKEND.value):
        "Add frontend origin to CORS allowed_origins in backend configuration",
    (FailurePattern.HTTP_404.value, FailureOwner.FRONTEND.value):
        "Verify frontend routes and SPA fallback (nginx try_files or .htaccess rewrite)",
    (FailurePattern.HTTP_404.value, FailureOwner.BACKEND.value):
        "Verify API routes are registered and server is running",
    (FailurePattern.HTTP_500.value, FailureOwner.BACKEND.value):
        "Check backend logs for unhandled exceptions; verify database connection and environment variables",
    (FailurePattern.HTTP_503.value, FailureOwner.INFRA.value):
        "Service not running: verify uvicorn process on expected port; run start_server.sh; check api.log for startup errors",
    (FailurePattern.AUTH_FAILURE.value, FailureOwner.BACKEND.value):
        "Verify auth endpoints are working: check /register, /login, /me routes and database user table",
    (FailurePattern.DATA_MISSING.value, FailureOwner.BACKEND.value):
        "Verify API returns expected fields in response; check database schema and seed data",
    (FailurePattern.CONSOLE_ERROR.value, FailureOwner.FRONTEND.value):
        "Check frontend build output and JavaScript console errors",
    (FailurePattern.BUILD_FAILURE.value, FailureOwner.FRONTEND.value):
        "Verify frontend build completes successfully; check build logs for errors",
    (FailurePattern.BUILD_FAILURE.value, FailureOwner.BACKEND.value):
        "Verify backend starts successfully; check application logs",
    (FailurePattern.BUILD_FAILURE.value, FailureOwner.DEPLOYMENT.value):
        "Verify deployment pipeline: check CI workflow, commit SHA, and deploy artifacts",
}

_DEFAULT_SUGGESTED_FIX = "Investigate the failure manually; check application logs"


# ---------------------------------------------------------------------------
# Prompt templates per pattern
# ---------------------------------------------------------------------------

_PATTERN_STRATEGIES: Dict[str, str] = {
    FailurePattern.CORS.value: (
        "1. Find CORS middleware/configuration in the backend code\n"
        "2. Add the frontend origin URL to the allowed origins list\n"
        "3. Ensure OPTIONS preflight requests are handled\n"
        "4. Verify the Access-Control-Allow-Origin header is returned"
    ),
    FailurePattern.HTTP_404.value: (
        "1. Check if the route/page exists in the application code\n"
        "2. For frontend SPAs: verify .htaccess or nginx config has fallback to index.html\n"
        "3. For backend APIs: verify route registration in the main app file\n"
        "4. Check if the URL path matches the configured routes exactly"
    ),
    FailurePattern.HTTP_500.value: (
        "1. Check server logs for the stack trace\n"
        "2. Verify database connection and migrations are current\n"
        "3. Check environment variables are set correctly\n"
        "4. Look for unhandled exceptions in request handlers"
    ),
    FailurePattern.HTTP_503.value: (
        "1. SSH to the server and check if uvicorn is running: ps aux | grep uvicorn\n"
        "2. If not running, check scripts/start_server.sh exists and is executable\n"
        "3. Run: cd /home/<domain>/public_html && bash scripts/start_server.sh\n"
        "4. Check api.log for startup errors (import failures, missing deps, DB connection)\n"
        "5. Verify the reverse proxy (LiteSpeed/nginx) points to the correct port\n"
        "6. Ensure the deploy workflow verifies the process is running after start"
    ),
    FailurePattern.AUTH_FAILURE.value: (
        "1. Verify /api/auth/register creates users in the database\n"
        "2. Verify /api/auth/login returns a valid JWT/token\n"
        "3. Verify /api/auth/me accepts the token and returns user data\n"
        "4. Check password hashing and token generation logic"
    ),
    FailurePattern.DATA_MISSING.value: (
        "1. Check the API endpoint returns all expected fields\n"
        "2. Verify database has the required tables and columns\n"
        "3. Check serialization logic includes all fields\n"
        "4. Verify default values are returned for empty datasets"
    ),
    FailurePattern.CONSOLE_ERROR.value: (
        "1. Check frontend build for JavaScript errors\n"
        "2. Verify all imports and dependencies resolve correctly\n"
        "3. Check for missing environment variables in frontend config\n"
        "4. Verify API URLs are configured correctly in the frontend"
    ),
    FailurePattern.BUILD_FAILURE.value: (
        "1. Check build logs for compilation errors\n"
        "2. Verify all dependencies are installed\n"
        "3. Check for version mismatches in package.json or requirements.txt\n"
        "4. Verify CI/CD pipeline runs the correct build commands"
    ),
}


# ---------------------------------------------------------------------------
# MicroRemediator
# ---------------------------------------------------------------------------

class MicroRemediator:
    """
    Deterministic failure classifier and remediation task generator.

    classify_failure() uses keyword/pattern matching on E2ECheckResult dict fields.
    No ML, no probabilistic inference. Returns UNKNOWN for unclassifiable failures.
    """

    def classify_failure(
        self, test_failure: Dict[str, Any]
    ) -> Tuple[FailurePattern, FailureOwner]:
        """
        Classify a failed E2E check into (pattern, owner).

        Args:
            test_failure: Dict with keys matching E2ECheckResult fields:
                check_type, passed, message, details, response_time_ms, checked_at

        Returns:
            (FailurePattern, FailureOwner) tuple
        """
        check_type = test_failure.get("check_type", "")
        message = (test_failure.get("message") or "").lower()

        # CORS check
        if check_type == "cors_browser_origin":
            return FailurePattern.CORS, FailureOwner.BACKEND

        # Frontend pages
        if check_type == "frontend_pages":
            if "404" in message:
                return FailurePattern.HTTP_404, FailureOwner.FRONTEND
            if "not html" in message or "content-type" in message:
                return FailurePattern.BUILD_FAILURE, FailureOwner.FRONTEND
            return FailurePattern.CONSOLE_ERROR, FailureOwner.FRONTEND

        # Auth E2E
        if check_type == "auth_login_e2e":
            return FailurePattern.AUTH_FAILURE, FailureOwner.BACKEND

        # 503 — service not running (check before other API patterns)
        if "503" in message:
            return FailurePattern.HTTP_503, FailureOwner.INFRA

        # API schema
        if check_type == "api_schema_valid":
            if "404" in message:
                return FailurePattern.HTTP_404, FailureOwner.BACKEND
            if "500" in message:
                return FailurePattern.HTTP_500, FailureOwner.BACKEND
            return FailurePattern.BUILD_FAILURE, FailureOwner.BACKEND

        # Business data
        if check_type == "api_business_data":
            return FailurePattern.DATA_MISSING, FailureOwner.BACKEND

        # Commit SHA match
        if check_type == "commit_sha_match":
            return FailurePattern.BUILD_FAILURE, FailureOwner.DEPLOYMENT

        # Default
        return FailurePattern.UNKNOWN, FailureOwner.UNKNOWN

    def create_remediation_task(
        self, test_failure: Dict[str, Any]
    ) -> MicroRemediationTask:
        """
        Create a remediation task from a failed E2E check result dict.

        Args:
            test_failure: Dict with E2ECheckResult fields plus optional
                '_project_name' key for the project name.

        Returns:
            Frozen MicroRemediationTask
        """
        pattern, owner = self.classify_failure(test_failure)
        suggested_fix = _SUGGESTED_FIXES.get(
            (pattern.value, owner.value), _DEFAULT_SUGGESTED_FIX
        )

        return MicroRemediationTask(
            project=test_failure.get("_project_name", "unknown"),
            failure_pattern=pattern.value,
            owner=owner.value,
            evidence=test_failure.get("message", "No message"),
            suggested_fix=suggested_fix,
            requires_redeploy=(pattern != FailurePattern.UNKNOWN),
            check_type=test_failure.get("check_type", "unknown"),
        )

    def generate_claude_fix_prompt(self, task: MicroRemediationTask) -> str:
        """
        Generate a focused Claude CLI fix prompt for a remediation task.

        Returns:
            Structured prompt string ready for create_claude_job()
        """
        strategy = _PATTERN_STRATEGIES.get(
            task.failure_pattern,
            "1. Investigate the failure evidence\n2. Check application logs\n3. Apply the suggested fix",
        )

        return (
            f"MICRO-REMEDIATION JOB — Targeted Auto-Fix\n"
            f"{'=' * 60}\n\n"
            f"PROJECT: {task.project}\n"
            f"FAILURE PATTERN: {task.failure_pattern}\n"
            f"OWNER: {task.owner}\n"
            f"CHECK TYPE: {task.check_type}\n\n"
            f"{'=' * 60}\n"
            f"EVIDENCE\n"
            f"{'=' * 60}\n"
            f"{task.evidence}\n\n"
            f"{'=' * 60}\n"
            f"SUGGESTED FIX\n"
            f"{'=' * 60}\n"
            f"{task.suggested_fix}\n\n"
            f"{'=' * 60}\n"
            f"FIX STRATEGY\n"
            f"{'=' * 60}\n"
            f"{strategy}\n\n"
            f"{'=' * 60}\n"
            f"SCOPE CONSTRAINTS\n"
            f"{'=' * 60}\n"
            f"- Only fix the {task.owner} component\n"
            f"- Do NOT modify other components\n"
            f"- This is a MICRO fix — make the smallest change possible\n"
            f"- Do NOT refactor or restructure code beyond the fix\n\n"
            f"{'=' * 60}\n"
            f"INSTRUCTIONS\n"
            f"{'=' * 60}\n"
            f"1. Read the project code and identify the root cause\n"
            f"2. Apply the minimal fix for the identified failure\n"
            f"3. Verify the fix addresses the evidence above\n"
            f"4. Commit changes with a clear message\n"
            f"5. Push to GitHub: git push origin main\n\n"
            f"VALIDATION WILL RE-RUN AUTOMATICALLY AFTER DEPLOYMENT.\n"
        )

    # ------------------------------------------------------------------
    # State management (reads/writes Project.metadata)
    # ------------------------------------------------------------------

    @staticmethod
    def get_remediation_state(project_name: str) -> Dict[str, Any]:
        """Read micro-remediation state from Project.metadata."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            project = registry.get_project(project_name)
            if project:
                return dict(project.metadata.get("micro_remediation", {}))
        except Exception as e:
            logger.warning(f"Failed to read remediation state for {project_name}: {e}")
        return {}

    @staticmethod
    def update_remediation_state(
        project_name: str, state: Dict[str, Any]
    ) -> None:
        """Write micro-remediation state to Project.metadata."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            project = registry.get_project(project_name)
            if project:
                project.metadata["micro_remediation"] = state
                registry.update_project(project_name, {"metadata": project.metadata})
                logger.info(f"Updated remediation state for {project_name}: status={state.get('status')}")
        except Exception as e:
            logger.error(f"Failed to update remediation state for {project_name}: {e}")

    @staticmethod
    def can_attempt_remediation(
        project_name: str,
    ) -> Tuple[bool, str]:
        """
        Check if micro-remediation can be attempted for a project.

        Returns:
            (can_attempt, reason) tuple
        """
        state = MicroRemediator.get_remediation_state(project_name)

        if not state:
            return True, "No previous remediation attempts"

        status = state.get("status")
        attempts = state.get("attempts", 0)

        if status == RemediationStatus.RESOLVED.value:
            return True, "Previous remediation was resolved"

        if status == RemediationStatus.FAILED.value:
            return False, f"Remediation failed after {attempts} attempts — manual intervention required"

        if status in (RemediationStatus.FIXING.value, RemediationStatus.RETESTING.value):
            return False, f"Remediation already in progress (status={status})"

        if attempts >= MICRO_REMEDIATION_MAX_ATTEMPTS:
            return False, f"Max remediation attempts ({MICRO_REMEDIATION_MAX_ATTEMPTS}) reached"

        return True, f"Attempt {attempts + 1} of {MICRO_REMEDIATION_MAX_ATTEMPTS}"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_remediator: Optional[MicroRemediator] = None


def get_micro_remediator() -> MicroRemediator:
    global _remediator
    if _remediator is None:
        _remediator = MicroRemediator()
    return _remediator
