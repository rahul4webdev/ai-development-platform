"""
CI Monitor - GitHub Workflow Status Monitoring
Phase 21: CI Failure Auto-Detection and Self-Healing

This module monitors GitHub Actions workflow runs and automatically:
1. Detects CI failures
2. Parses error logs to identify failure type
3. Creates bug_fix jobs to resolve issues
4. Triggers re-deployment after fixes

IMPORTANT:
- This is a polling-based monitor (GitHub webhooks are not always available)
- All decisions are logged for audit
- No production deployment without human approval
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set

import httpx

logger = logging.getLogger("ci_monitor")

# Configuration
GITHUB_API_BASE = "https://api.github.com"
CI_MONITOR_INTERVAL = int(os.getenv("CI_MONITOR_INTERVAL", "60"))  # seconds
CI_MONITOR_STATE_FILE = Path(os.getenv("CI_MONITOR_STATE", "/tmp/ci_monitor_state.json"))
MAX_AUTO_FIX_ATTEMPTS = 3  # Maximum automatic fix attempts before requiring human


class CIFailureType(str, Enum):
    """
    Classification of CI failures.

    LOCKED: Each type has specific remediation strategies.
    """
    LINT_FLAKE8 = "lint_flake8"  # Python flake8 errors
    LINT_BLACK = "lint_black"    # Python black formatting
    LINT_ESLINT = "lint_eslint"  # JavaScript/React ESLint
    TEST_PYTEST = "test_pytest"  # Python test failures
    TEST_JEST = "test_jest"      # JavaScript test failures
    BUILD_NPM = "build_npm"      # npm install/build failures
    BUILD_PIP = "build_pip"      # pip install failures
    DEPLOY_SSH = "deploy_ssh"    # SSH/deployment failures
    UNKNOWN = "unknown"          # Cannot classify


@dataclass
class CIFailure:
    """Represents a CI failure with classification and remediation info."""
    run_id: int
    repo: str
    job_name: str
    step_name: str
    failure_type: CIFailureType
    error_message: str
    error_files: List[str] = field(default_factory=list)
    suggested_fix: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "repo": self.repo,
            "job_name": self.job_name,
            "step_name": self.step_name,
            "failure_type": self.failure_type.value,
            "error_message": self.error_message[:500],  # Truncate for storage
            "error_files": self.error_files,
            "suggested_fix": self.suggested_fix,
            "detected_at": self.detected_at.isoformat()
        }


class CIFailureClassifier:
    """
    Classifies CI failures based on log content.

    Rules are deterministic pattern matching - no ML.
    """

    # Pattern -> (failure_type, fix_suggestion)
    PATTERNS = [
        # Flake8 errors
        (r"F401.*imported but unused", CIFailureType.LINT_FLAKE8,
         "Remove unused imports or add to __all__"),
        (r"E501.*line too long", CIFailureType.LINT_FLAKE8,
         "Break long lines or use black to auto-format"),
        (r"flake8.*exit code", CIFailureType.LINT_FLAKE8,
         "Fix flake8 linting errors"),

        # Black formatting
        (r"would reformat", CIFailureType.LINT_BLACK,
         "Run 'black .' to auto-format code"),
        (r"black.*--check.*failed", CIFailureType.LINT_BLACK,
         "Run 'black .' to auto-format code"),

        # ESLint errors
        (r"react-hooks/exhaustive-deps", CIFailureType.LINT_ESLINT,
         "Add missing dependencies to useEffect or wrap functions in useCallback"),
        (r"eslint.*error", CIFailureType.LINT_ESLINT,
         "Fix ESLint errors in JavaScript/React code"),
        (r"Treating warnings as errors", CIFailureType.LINT_ESLINT,
         "Fix ESLint warnings - they are treated as errors in CI"),

        # Pytest failures
        (r"FAILED.*test_", CIFailureType.TEST_PYTEST,
         "Fix failing test cases"),
        (r"pytest.*exit code 1", CIFailureType.TEST_PYTEST,
         "Fix pytest test failures"),
        (r"coverage.*--cov-fail-under", CIFailureType.TEST_PYTEST,
         "Increase test coverage to meet threshold"),

        # npm/Node failures
        (r"npm ci.*can only install", CIFailureType.BUILD_NPM,
         "Use 'npm install --legacy-peer-deps' instead of 'npm ci'"),
        (r"npm error code EUSAGE", CIFailureType.BUILD_NPM,
         "Fix package.json/package-lock.json sync issues"),
        (r"peer dependency.*conflict", CIFailureType.BUILD_NPM,
         "Use --legacy-peer-deps flag for npm install"),

        # pip/Python failures
        (r"pip.*error.*No matching distribution", CIFailureType.BUILD_PIP,
         "Fix package version constraints in requirements.txt"),
        (r"bcrypt.*passlib", CIFailureType.BUILD_PIP,
         "Pin bcrypt>=4.0.0,<4.2.0 for passlib compatibility"),

        # SSH/Deploy failures
        (r"Permission denied.*ssh", CIFailureType.DEPLOY_SSH,
         "Check SSH key configuration in GitHub secrets"),
        (r"Connection refused", CIFailureType.DEPLOY_SSH,
         "Verify server is reachable and SSH port is correct"),
    ]

    @classmethod
    def classify(cls, log_content: str) -> tuple[CIFailureType, Optional[str]]:
        """
        Classify failure type from log content.

        Returns:
            Tuple of (failure_type, suggested_fix)
        """
        for pattern, failure_type, suggestion in cls.PATTERNS:
            if re.search(pattern, log_content, re.IGNORECASE):
                return failure_type, suggestion

        return CIFailureType.UNKNOWN, None

    @classmethod
    def extract_error_files(cls, log_content: str) -> List[str]:
        """Extract file paths mentioned in error messages."""
        # Common patterns for file paths in error messages
        patterns = [
            r'(?:File |at |in )["\'"]?([a-zA-Z0-9_/\-\.]+\.(?:py|js|ts|tsx|jsx))["\'"]?',
            r'([a-zA-Z0-9_/\-\.]+\.(?:py|js|ts|tsx|jsx)):\d+:\d+',
        ]

        files = set()
        for pattern in patterns:
            matches = re.findall(pattern, log_content)
            files.update(matches)

        return list(files)


class CIMonitor:
    """
    Monitor GitHub Actions workflow runs for failures.

    Polls GitHub API to check workflow status and triggers
    automatic remediation when failures are detected.
    """

    def __init__(
        self,
        github_token: str,
        notification_callback=None,
        create_job_callback=None
    ):
        self.github_token = github_token
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.notification_callback = notification_callback
        self.create_job_callback = create_job_callback
        self._monitored_repos: Set[str] = set()
        self._processed_runs: Set[int] = set()
        self._fix_attempts: Dict[str, int] = {}  # repo -> attempt count
        self._running = False
        self._load_state()

    def _load_state(self):
        """Load monitor state from file."""
        if CI_MONITOR_STATE_FILE.exists():
            try:
                state = json.loads(CI_MONITOR_STATE_FILE.read_text())
                self._processed_runs = set(state.get("processed_runs", []))
                self._fix_attempts = state.get("fix_attempts", {})
                logger.info(f"Loaded CI monitor state: {len(self._processed_runs)} processed runs")
            except Exception as e:
                logger.error(f"Failed to load CI monitor state: {e}")

    def _save_state(self):
        """Save monitor state to file."""
        try:
            state = {
                "processed_runs": list(self._processed_runs)[-1000:],  # Keep last 1000
                "fix_attempts": self._fix_attempts,
                "last_save": datetime.utcnow().isoformat()
            }
            CI_MONITOR_STATE_FILE.write_text(json.dumps(state, indent=2))
        except Exception as e:
            logger.error(f"Failed to save CI monitor state: {e}")

    def add_repo(self, owner: str, repo: str):
        """Add a repository to monitor."""
        self._monitored_repos.add(f"{owner}/{repo}")
        logger.info(f"Added repo to CI monitor: {owner}/{repo}")

    def remove_repo(self, owner: str, repo: str):
        """Remove a repository from monitoring."""
        self._monitored_repos.discard(f"{owner}/{repo}")

    async def _get_workflow_runs(
        self,
        repo: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent workflow runs for a repository."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{GITHUB_API_BASE}/repos/{repo}/actions/runs",
                    headers=self.headers,
                    params={"per_page": limit}
                )
                if response.status_code == 200:
                    return response.json().get("workflow_runs", [])
                else:
                    logger.error(f"Failed to get workflow runs: {response.status_code}")
                    return []
            except Exception as e:
                logger.error(f"Error fetching workflow runs: {e}")
                return []

    async def _get_run_jobs(self, repo: str, run_id: int) -> List[Dict]:
        """Get jobs for a specific workflow run."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.get(
                    f"{GITHUB_API_BASE}/repos/{repo}/actions/runs/{run_id}/jobs",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.json().get("jobs", [])
                return []
            except Exception as e:
                logger.error(f"Error fetching run jobs: {e}")
                return []

    async def _get_job_logs(self, repo: str, job_id: int) -> Optional[str]:
        """Get logs for a specific job."""
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            try:
                response = await client.get(
                    f"{GITHUB_API_BASE}/repos/{repo}/actions/jobs/{job_id}/logs",
                    headers=self.headers
                )
                if response.status_code == 200:
                    return response.text
                return None
            except Exception as e:
                logger.error(f"Error fetching job logs: {e}")
                return None

    async def _analyze_failure(
        self,
        repo: str,
        run: Dict,
        job: Dict
    ) -> Optional[CIFailure]:
        """Analyze a failed job and classify the failure."""
        run_id = run["id"]
        job_id = job["id"]
        job_name = job["name"]

        # Find the failed step
        failed_step = None
        for step in job.get("steps", []):
            if step.get("conclusion") == "failure":
                failed_step = step["name"]
                break

        # Get job logs
        logs = await self._get_job_logs(repo, job_id)
        if not logs:
            return CIFailure(
                run_id=run_id,
                repo=repo,
                job_name=job_name,
                step_name=failed_step or "unknown",
                failure_type=CIFailureType.UNKNOWN,
                error_message="Could not retrieve logs"
            )

        # Classify failure
        failure_type, suggestion = CIFailureClassifier.classify(logs)
        error_files = CIFailureClassifier.extract_error_files(logs)

        # Extract relevant error message (last 500 chars of logs typically have the error)
        error_message = logs[-2000:] if len(logs) > 2000 else logs

        return CIFailure(
            run_id=run_id,
            repo=repo,
            job_name=job_name,
            step_name=failed_step or "unknown",
            failure_type=failure_type,
            error_message=error_message,
            error_files=error_files,
            suggested_fix=suggestion
        )

    async def _handle_failure(self, failure: CIFailure, project_name: str):
        """Handle a CI failure by attempting automatic remediation."""
        repo = failure.repo

        # Check if we've exceeded max auto-fix attempts
        attempts = self._fix_attempts.get(repo, 0)
        if attempts >= MAX_AUTO_FIX_ATTEMPTS:
            logger.warning(f"Max auto-fix attempts ({MAX_AUTO_FIX_ATTEMPTS}) reached for {repo}")
            if self.notification_callback:
                await self.notification_callback(
                    f"⚠️ *CI Auto-Fix Limit Reached*\n\n"
                    f"*Repo:* {repo}\n"
                    f"*Failure:* {failure.failure_type.value}\n\n"
                    f"Manual intervention required after {MAX_AUTO_FIX_ATTEMPTS} auto-fix attempts."
                )
            return

        # Send notification about failure
        if self.notification_callback:
            await self.notification_callback(
                f"🔴 *CI Failed - Auto-Fix Initiated*\n\n"
                f"*Repo:* {repo}\n"
                f"*Job:* {failure.job_name}\n"
                f"*Step:* {failure.step_name}\n"
                f"*Type:* {failure.failure_type.value}\n"
                f"*Attempt:* {attempts + 1}/{MAX_AUTO_FIX_ATTEMPTS}\n\n"
                f"*Suggested Fix:* {failure.suggested_fix or 'N/A'}"
            )

        # Create bug_fix job if callback available
        if self.create_job_callback and failure.failure_type != CIFailureType.UNKNOWN:
            task_description = self._generate_fix_task(failure)
            await self.create_job_callback(
                project_name=project_name,
                task_description=task_description,
                task_type="bug_fix",
                failure_context=failure.to_dict()
            )
            self._fix_attempts[repo] = attempts + 1
            self._save_state()

    def _generate_fix_task(self, failure: CIFailure) -> str:
        """Generate task description for bug fix job."""
        files_str = "\n".join(f"- {f}" for f in failure.error_files[:10])

        return f"""BUG FIX - CI Failure Auto-Remediation

FAILURE DETAILS:
- Type: {failure.failure_type.value}
- Job: {failure.job_name}
- Step: {failure.step_name}

SUGGESTED FIX:
{failure.suggested_fix or "Analyze the error and apply appropriate fix"}

AFFECTED FILES:
{files_str or "Unknown - analyze error logs"}

ERROR EXCERPT:
```
{failure.error_message[:1000]}
```

INSTRUCTIONS:
1. Read the error message carefully
2. Apply the suggested fix or investigate the root cause
3. Run local tests if applicable
4. Commit and push the fix
5. The CI will automatically re-run

IMPORTANT:
- Only fix the specific CI failure
- Do not make unrelated changes
- Update any affected tests
"""

    async def check_repos(self):
        """Check all monitored repositories for CI failures."""
        for repo in list(self._monitored_repos):
            try:
                runs = await self._get_workflow_runs(repo)
                for run in runs:
                    run_id = run["id"]

                    # Skip already processed runs
                    if run_id in self._processed_runs:
                        continue

                    # Only process completed runs
                    if run["status"] != "completed":
                        continue

                    self._processed_runs.add(run_id)

                    # Check if run failed
                    if run["conclusion"] == "failure":
                        jobs = await self._get_run_jobs(repo, run_id)
                        for job in jobs:
                            if job.get("conclusion") == "failure":
                                failure = await self._analyze_failure(repo, run, job)
                                if failure:
                                    # Extract project name from repo
                                    project_name = repo.split("/")[-1]
                                    await self._handle_failure(failure, project_name)
                                break  # Handle one failure per run

                    # Notify on success after previous failure
                    elif run["conclusion"] == "success":
                        if repo in self._fix_attempts and self._fix_attempts[repo] > 0:
                            # Reset fix attempts on success
                            self._fix_attempts[repo] = 0
                            self._save_state()

                            if self.notification_callback:
                                await self.notification_callback(
                                    f"✅ *CI Passed After Fix*\n\n"
                                    f"*Repo:* {repo}\n"
                                    f"*Workflow:* {run['name']}\n\n"
                                    f"Auto-fix successful! Ready for deployment."
                                )

            except Exception as e:
                logger.error(f"Error checking repo {repo}: {e}")

        self._save_state()

    async def start(self):
        """Start the CI monitor loop."""
        self._running = True
        logger.info(f"CI Monitor started, checking every {CI_MONITOR_INTERVAL}s")

        while self._running:
            try:
                await self.check_repos()
            except Exception as e:
                logger.error(f"CI Monitor error: {e}")

            await asyncio.sleep(CI_MONITOR_INTERVAL)

    def stop(self):
        """Stop the CI monitor."""
        self._running = False
        self._save_state()
        logger.info("CI Monitor stopped")


# Global instance
_ci_monitor: Optional[CIMonitor] = None


def get_ci_monitor(
    github_token: Optional[str] = None,
    notification_callback=None,
    create_job_callback=None
) -> Optional[CIMonitor]:
    """Get or create the CI monitor instance."""
    global _ci_monitor

    if _ci_monitor is None:
        token = github_token or os.getenv("GITHUB_TOKEN")
        if not token:
            logger.warning("No GitHub token available for CI monitor")
            return None

        _ci_monitor = CIMonitor(
            github_token=token,
            notification_callback=notification_callback,
            create_job_callback=create_job_callback
        )

    return _ci_monitor
