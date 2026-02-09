"""
Phase 26: Meta-Remediation (Rescue-the-Rescue Layer)

When a rescue job itself fails (e.g. Claude CLI exits non-zero, or the fix
doesn't deploy), this layer automatically creates a second-order fix job
with enhanced diagnostic context.

CONSTRAINTS:
- Max 3 meta-remediation attempts per project per rescue failure
- Deterministic classification (exit code 5 → RESCUE_JOB_FAILED)
- All state in Project.metadata (append-only JSONL for persistence)
- Fail-open: ImportError/exceptions must not crash the caller
- No ML, no heuristics, no direct SSH — fixes go through GitHub workflow
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger("meta_remediator")

META_REMEDIATION_MAX_ATTEMPTS = 3
META_STATE_FILE = Path("data/meta_rescue/meta_rescue_state.jsonl")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MetaRemediationStatus(str, Enum):
    PENDING = "pending"
    FIXING = "fixing"
    RETESTING = "retesting"
    RESOLVED = "resolved"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetaRemediationTask:
    """Immutable meta-remediation task for a failed rescue job."""
    id: str
    project: str
    module: Optional[str]
    original_failure: str       # FailurePattern value that triggered rescue
    rescue_job_id: str
    rescue_exit_code: int
    created_at: str
    status: str                 # MetaRemediationStatus value
    attempt: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project": self.project,
            "module": self.module,
            "original_failure": self.original_failure,
            "rescue_job_id": self.rescue_job_id,
            "rescue_exit_code": self.rescue_exit_code,
            "created_at": self.created_at,
            "status": self.status,
            "attempt": self.attempt,
        }


# ---------------------------------------------------------------------------
# MetaRemediator
# ---------------------------------------------------------------------------

class MetaRemediator:
    """
    Deterministic classifier and remediation task generator for failed rescue jobs.

    classify_rescue_failure() maps exit codes to failure patterns.
    No ML, no probabilistic inference.
    """

    def classify_rescue_failure(
        self,
        job_id: str,
        exit_code: int,
        project: str,
        module: Optional[str] = None,
    ) -> str:
        """
        Classify a rescue job failure into a FailurePattern.

        Args:
            job_id: The failed rescue job ID
            exit_code: Process exit code
            project: Project name
            module: Optional module that was being rescued

        Returns:
            FailurePattern value string
        """
        from controller.micro_remediator import FailurePattern

        # Exit code 5 = Claude CLI specific failure (tool execution error)
        if exit_code == 5:
            return FailurePattern.RESCUE_JOB_FAILED.value

        # Any non-zero exit code from a rescue job = rescue failed
        if exit_code != 0:
            return FailurePattern.RESCUE_JOB_FAILED.value

        return FailurePattern.UNKNOWN.value

    def create_meta_remediation_task(
        self,
        project: str,
        rescue_job_id: str,
        rescue_exit_code: int,
        original_failure: str = "unknown",
        module: Optional[str] = None,
    ) -> MetaRemediationTask:
        """
        Create a MetaRemediationTask and update registry metadata.

        Args:
            project: Project name
            rescue_job_id: ID of the failed rescue job
            rescue_exit_code: Exit code of the failed rescue
            original_failure: Original failure pattern that triggered rescue
            module: Optional module scope

        Returns:
            Frozen MetaRemediationTask, or None if blocked
        """
        # Phase 26.6: Runtime integrity enforcement — block task creation when FATAL
        try:
            from controller.runtime_enforcement import RuntimeIntegrityEnforcer, ActionType, RuntimeIntegrityBlockedError
            RuntimeIntegrityEnforcer.check_or_block(ActionType.META_REMEDIATE)
        except RuntimeIntegrityBlockedError:
            logger.error("Meta-remediation task creation blocked: runtime integrity FATAL")
            return None
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Runtime integrity check failed in meta-remediator (proceeding): {e}")

        state = self.get_meta_state(project)
        attempt = state.get("attempts", 0) + 1

        task = MetaRemediationTask(
            id=str(uuid.uuid4()),
            project=project,
            module=module,
            original_failure=original_failure,
            rescue_job_id=rescue_job_id,
            rescue_exit_code=rescue_exit_code,
            created_at=datetime.utcnow().isoformat(),
            status=MetaRemediationStatus.PENDING.value,
            attempt=attempt,
        )

        # Update registry metadata
        new_state = {
            "last_meta_failure": {
                "task_id": task.id,
                "rescue_job_id": rescue_job_id,
                "exit_code": rescue_exit_code,
                "original_failure": original_failure,
                "module": module,
                "created_at": task.created_at,
            },
            "meta_remediation_status": MetaRemediationStatus.PENDING.value,
            "attempts": attempt,
            "last_task_id": task.id,
        }
        self.update_meta_state(project, new_state)

        # Append to persistent JSONL
        self._append_record(task.to_dict())

        logger.info(
            f"Created meta-remediation task {task.id[:12]} for {project} "
            f"(attempt {attempt}/{META_REMEDIATION_MAX_ATTEMPTS})"
        )

        return task

    def generate_meta_fix_prompt(
        self,
        task: MetaRemediationTask,
        context: Dict[str, Any],
    ) -> str:
        """
        Generate a comprehensive Claude CLI fix prompt for meta-remediation.

        The prompt includes all available diagnostic context to give Claude
        the best chance of fixing the issue on the second attempt.

        Args:
            task: The MetaRemediationTask
            context: Dict with optional keys:
                - github_actions_url: URL to the failed workflow run
                - deployed_sha: SHA from DEPLOY_INFO.txt
                - repo_sha: Latest SHA in the repo
                - server_type: e.g. "CyberPanel/LiteSpeed"
                - failed_endpoints: List of failing URLs
                - validator_results: Dict of validation results
                - api_log_tail: Last 200 lines of api.log
                - uvicorn_running: Whether uvicorn is on 127.0.0.1:8022
                - litespeed_proxy_snippet: Proxy config excerpt

        Returns:
            Structured prompt string for create_claude_job()
        """
        github_url = context.get("github_actions_url", "Not available")
        deployed_sha = context.get("deployed_sha", "Unknown")
        repo_sha = context.get("repo_sha", "Unknown")
        server_type = context.get("server_type", "Unknown")
        failed_endpoints = context.get("failed_endpoints", [])
        validator_results = context.get("validator_results", {})
        api_log_tail = context.get("api_log_tail", "Not available")
        uvicorn_running = context.get("uvicorn_running", "Unknown")
        litespeed_snippet = context.get("litespeed_proxy_snippet", "Not available")

        endpoints_str = "\n".join(f"  - {ep}" for ep in failed_endpoints) if failed_endpoints else "  None recorded"
        validator_str = json.dumps(validator_results, indent=2, default=str)[:500] if validator_results else "  None"

        return (
            f"META-REMEDIATION JOB — Rescue-the-Rescue Fix\n"
            f"{'=' * 60}\n\n"
            f"A previous rescue job FAILED. This is a second-order fix attempt.\n\n"
            f"PROJECT: {task.project}\n"
            f"MODULE: {task.module or 'all'}\n"
            f"ATTEMPT: {task.attempt}/{META_REMEDIATION_MAX_ATTEMPTS}\n"
            f"ORIGINAL FAILURE: {task.original_failure}\n"
            f"RESCUE JOB ID: {task.rescue_job_id}\n"
            f"RESCUE EXIT CODE: {task.rescue_exit_code}\n\n"
            f"{'=' * 60}\n"
            f"DEPLOYMENT INFO\n"
            f"{'=' * 60}\n"
            f"GitHub Actions Run: {github_url}\n"
            f"Deployed SHA (DEPLOY_INFO.txt): {deployed_sha}\n"
            f"Latest Repo SHA: {repo_sha}\n"
            f"Server Type: {server_type}\n\n"
            f"{'=' * 60}\n"
            f"FAILED ENDPOINTS\n"
            f"{'=' * 60}\n"
            f"{endpoints_str}\n\n"
            f"{'=' * 60}\n"
            f"VALIDATOR RESULTS\n"
            f"{'=' * 60}\n"
            f"{validator_str}\n\n"
            f"{'=' * 60}\n"
            f"SERVER DIAGNOSTICS\n"
            f"{'=' * 60}\n"
            f"Uvicorn running on 127.0.0.1:8022: {uvicorn_running}\n"
            f"LiteSpeed proxy config:\n{litespeed_snippet}\n\n"
            f"{'=' * 60}\n"
            f"API LOG (last 200 lines)\n"
            f"{'=' * 60}\n"
            f"{api_log_tail[:3000]}\n\n"
            f"{'=' * 60}\n"
            f"INSTRUCTIONS\n"
            f"{'=' * 60}\n"
            f"1. Analyze WHY the previous rescue job (exit code {task.rescue_exit_code}) failed\n"
            f"2. Check the GitHub Actions workflow for errors\n"
            f"3. Verify the deployment pipeline end-to-end\n"
            f"4. Fix the root cause — not just the symptom\n"
            f"5. If the start_server.sh script is broken, fix it\n"
            f"6. If .env is misconfigured, fix the workflow that creates it\n"
            f"7. Commit and push: git push origin main\n\n"
            f"SCOPE: Only fix the deployment/rescue issue. Do NOT refactor.\n"
            f"VALIDATION WILL RE-RUN AUTOMATICALLY AFTER DEPLOYMENT.\n"
        )

    def handle_meta_remediation_completion(
        self,
        project: str,
        job_id: str,
        passed: bool,
    ) -> str:
        """
        Handle completion of a meta-remediation job.

        Args:
            project: Project name
            job_id: The meta-remediation job ID
            passed: Whether validation passed after the fix

        Returns:
            New MetaRemediationStatus value
        """
        state = self.get_meta_state(project)
        attempts = state.get("attempts", 0)

        if passed:
            new_status = MetaRemediationStatus.RESOLVED.value
            state["meta_remediation_status"] = new_status
            state["resolved_at"] = datetime.utcnow().isoformat()
            self.update_meta_state(project, state)
            logger.info(f"Meta-remediation RESOLVED for {project} after {attempts} attempts")
            return new_status

        if attempts >= META_REMEDIATION_MAX_ATTEMPTS:
            new_status = MetaRemediationStatus.FAILED.value
            state["meta_remediation_status"] = new_status
            state["failed_at"] = datetime.utcnow().isoformat()
            self.update_meta_state(project, state)
            logger.warning(
                f"Meta-remediation FAILED for {project} after {META_REMEDIATION_MAX_ATTEMPTS} attempts"
            )
            return new_status

        # Still has retries left — mark as retesting for next attempt
        new_status = MetaRemediationStatus.RETESTING.value
        state["meta_remediation_status"] = new_status
        self.update_meta_state(project, state)
        logger.info(f"Meta-remediation retesting for {project} (attempt {attempts})")
        return new_status

    # ------------------------------------------------------------------
    # State management (reads/writes Project.metadata)
    # ------------------------------------------------------------------

    @staticmethod
    def get_meta_state(project_name: str) -> Dict[str, Any]:
        """Read meta-remediation state from Project.metadata."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            project = registry.get_project(project_name)
            if project:
                return dict(project.metadata.get("meta_remediation", {}))
        except Exception as e:
            logger.warning(f"Failed to read meta state for {project_name}: {e}")
        return {}

    @staticmethod
    def update_meta_state(project_name: str, state: Dict[str, Any]) -> None:
        """Write meta-remediation state to Project.metadata."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            project = registry.get_project(project_name)
            if project:
                project.metadata["meta_remediation"] = state
                registry.update_project(project_name, {"metadata": project.metadata})
                logger.info(f"Updated meta state for {project_name}: status={state.get('meta_remediation_status')}")
        except Exception as e:
            logger.error(f"Failed to update meta state for {project_name}: {e}")

    @staticmethod
    def can_attempt_meta_remediation(project_name: str) -> tuple:
        """
        Check if meta-remediation can be attempted for a project.

        Returns:
            (can_attempt: bool, reason: str) tuple
        """
        state = MetaRemediator.get_meta_state(project_name)

        if not state:
            return True, "No previous meta-remediation attempts"

        status = state.get("meta_remediation_status")
        attempts = state.get("attempts", 0)

        if status == MetaRemediationStatus.RESOLVED.value:
            return True, "Previous meta-remediation was resolved"

        if status == MetaRemediationStatus.FAILED.value:
            return False, f"Meta-remediation failed after {attempts} attempts — manual intervention required"

        if status in (MetaRemediationStatus.FIXING.value, MetaRemediationStatus.RETESTING.value):
            return False, f"Meta-remediation already in progress (status={status})"

        if attempts >= META_REMEDIATION_MAX_ATTEMPTS:
            return False, f"Max meta-remediation attempts ({META_REMEDIATION_MAX_ATTEMPTS}) reached"

        return True, f"Attempt {attempts + 1} of {META_REMEDIATION_MAX_ATTEMPTS}"

    # ------------------------------------------------------------------
    # Append-only JSONL persistence
    # ------------------------------------------------------------------

    @staticmethod
    def _append_record(record: Dict[str, Any]) -> None:
        """Append record to JSONL file with fsync."""
        try:
            META_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(META_STATE_FILE, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                import os as _os
                _os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Failed to append meta-remediation record: {e}")

    @staticmethod
    def get_all_records(project: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read all records from JSONL, optionally filtered by project."""
        records = []
        try:
            if META_STATE_FILE.exists():
                with open(META_STATE_FILE) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            record = json.loads(line)
                            if project is None or record.get("project") == project:
                                records.append(record)
        except Exception as e:
            logger.error(f"Failed to read meta-remediation records: {e}")
        return records


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_meta_remediator: Optional[MetaRemediator] = None


def get_meta_remediator() -> MetaRemediator:
    global _meta_remediator
    if _meta_remediator is None:
        _meta_remediator = MetaRemediator()
    return _meta_remediator
