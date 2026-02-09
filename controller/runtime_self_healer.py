"""
Phase 26.7: Platform Self-Healing Layer

Automatically attempts safe, deterministic remediation when Runtime Integrity
is FATAL or DEGRADED. Restores platform to HEALTHY whenever possible without
human intervention. Escalates to Meta-Remediation (Phase 26) if healing fails.

CONSTRAINTS:
- Deterministic behaviour only (no ML, no heuristics)
- Read-only diagnosis before any modification
- Max 3 healing attempts per session
- Append-only immutable audit of all HealTask and HealResult records (fsync)
- Fail-closed: if audit write fails, stop healing
- No business logic or project code changes allowed
- No deployments, no lifecycle mutations, no Claude execution jobs
- Human-visible progress via Telegram
"""

import json
import logging
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("runtime_self_healer")

HEAL_AUDIT_PATH = Path("/home/aitesting.mybd.in/jobs/runtime_heal.jsonl")
HEAL_MAX_ATTEMPTS = 3
EXPECTED_VENV = Path("/home/aitesting.mybd.in/public_html/venv")
REQUIREMENTS_TXT = Path("/home/aitesting.mybd.in/public_html/requirements.txt")


# ---------------------------------------------------------------------------
# Enums (LOCKED)
# ---------------------------------------------------------------------------

class HealAction(str, Enum):
    """Safe, deterministic healing actions."""
    INSTALL_PYTHON_PACKAGE = "install_python_package"
    INSTALL_PLAYWRIGHT = "install_playwright"
    FIX_PERMISSIONS = "fix_permissions"
    RESTART_SERVICES = "restart_services"
    CLEAR_STALE_LOCKS = "clear_stale_locks"
    NO_OP = "no_op"


class HealStatus(str, Enum):
    """Status of a healing task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ESCALATED = "escalated"


# ---------------------------------------------------------------------------
# Frozen Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HealTask:
    """Immutable healing task."""
    id: str
    project_scope: Optional[str]  # None = platform-wide
    action: str  # HealAction value
    evidence: str  # What triggered this task
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "heal_task",
            "id": self.id,
            "project_scope": self.project_scope,
            "action": self.action,
            "evidence": self.evidence,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class HealResult:
    """Immutable result of a healing task execution."""
    task_id: str
    status: str  # HealStatus value
    before_integrity: str  # Status before heal (FATAL/DEGRADED/HEALTHY)
    after_integrity: str  # Status after heal
    error: Optional[str]
    completed_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "heal_result",
            "task_id": self.task_id,
            "status": self.status,
            "before_integrity": self.before_integrity,
            "after_integrity": self.after_integrity,
            "error": self.error,
            "completed_at": self.completed_at,
        }


# ---------------------------------------------------------------------------
# RuntimeSelfHealer
# ---------------------------------------------------------------------------

class RuntimeSelfHealer:
    """
    Deterministic self-healing for platform runtime integrity issues.

    Maps integrity failures to safe, predefined remediation actions.
    No ML, no heuristics, no project code changes.
    """

    def classify_issues(self, report) -> List[HealAction]:
        """
        Map integrity report failures to safe healing actions.

        Args:
            report: RuntimeIntegrityReport (frozen dataclass)

        Returns:
            List of HealAction values (deterministic, ordered by priority)
        """
        actions: List[HealAction] = []

        # Missing packages → install from requirements.txt
        if report.missing_packages:
            actions.append(HealAction.INSTALL_PYTHON_PACKAGE)

        # Playwright not OK → install playwright
        if not report.playwright_ok:
            actions.append(HealAction.INSTALL_PLAYWRIGHT)

        # Filesystem permissions broken → fix permissions
        if not report.fs_permissions_ok:
            actions.append(HealAction.FIX_PERMISSIONS)

        # Missing binaries → try clearing stale locks (often caused by
        # interrupted package managers or stale state)
        if report.missing_binaries:
            actions.append(HealAction.CLEAR_STALE_LOCKS)

        if not actions:
            actions.append(HealAction.NO_OP)

        return actions

    def create_heal_tasks(self, report) -> List[HealTask]:
        """
        Create immutable HealTask objects from an integrity report.

        Args:
            report: RuntimeIntegrityReport

        Returns:
            List of frozen HealTask objects
        """
        actions = self.classify_issues(report)
        now = datetime.now(timezone.utc).isoformat()
        tasks = []

        for action in actions:
            if action == HealAction.NO_OP:
                continue

            evidence = self._build_evidence(report, action)
            task = HealTask(
                id=str(uuid.uuid4()),
                project_scope=None,  # Platform-wide
                action=action.value,
                evidence=evidence,
                created_at=now,
            )
            tasks.append(task)

        return tasks

    def execute_task(self, task: HealTask) -> HealResult:
        """
        Execute a single healing task using predefined safe commands.

        Args:
            task: The HealTask to execute

        Returns:
            Frozen HealResult with before/after integrity status
        """
        # Get before-state
        before_status = self._get_current_status()
        error = None
        status = HealStatus.IN_PROGRESS

        try:
            action = HealAction(task.action)

            if action == HealAction.INSTALL_PYTHON_PACKAGE:
                error = self._run_install_packages()
            elif action == HealAction.INSTALL_PLAYWRIGHT:
                error = self._run_install_playwright()
            elif action == HealAction.FIX_PERMISSIONS:
                error = self._run_fix_permissions()
            elif action == HealAction.RESTART_SERVICES:
                error = self._run_restart_services()
            elif action == HealAction.CLEAR_STALE_LOCKS:
                error = self._run_clear_stale_locks()
            elif action == HealAction.NO_OP:
                error = None
            else:
                error = f"Unknown action: {task.action}"

            if error:
                status = HealStatus.FAILED
            else:
                status = HealStatus.SUCCESS

        except Exception as e:
            error = str(e)
            status = HealStatus.FAILED

        # Get after-state
        after_status = self._get_current_status()

        result = HealResult(
            task_id=task.id,
            status=status.value,
            before_integrity=before_status,
            after_integrity=after_status,
            error=error,
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        return result

    def attempt_heal(self, max_attempts: int = HEAL_MAX_ATTEMPTS) -> List[HealResult]:
        """
        Attempt healing up to max_attempts times.

        1. Run integrity check
        2. Classify issues → create tasks
        3. Execute tasks sequentially
        4. Re-check integrity after each round
        5. Stop on HEALTHY; escalate on repeated failure

        Returns:
            List of all HealResult objects from this session
        """
        all_results: List[HealResult] = []

        for attempt in range(1, max_attempts + 1):
            logger.info(f"Self-healing attempt {attempt}/{max_attempts}")

            # Run fresh integrity check
            report = self._run_fresh_integrity_check()
            if report is None:
                logger.error("Cannot run integrity check — aborting healing")
                break

            if report.status == "HEALTHY":
                logger.info("Platform is HEALTHY — no healing needed")
                break

            # Create heal tasks
            tasks = self.create_heal_tasks(report)
            if not tasks:
                logger.info("No actionable heal tasks — stopping")
                break

            # Execute each task
            for task in tasks:
                # Write task audit record (fail-closed)
                if not self._append_record(task.to_dict()):
                    logger.error("Audit write failed — stopping healing (fail-closed)")
                    return all_results

                result = self.execute_task(task)
                all_results.append(result)

                # Write result audit record (fail-closed)
                if not self._append_record(result.to_dict()):
                    logger.error("Audit write failed — stopping healing (fail-closed)")
                    return all_results

                logger.info(
                    f"Heal task {task.id[:12]} ({task.action}): "
                    f"{result.status} [{result.before_integrity} → {result.after_integrity}]"
                )

            # Re-check after this round
            post_report = self._run_fresh_integrity_check()
            if post_report and post_report.status == "HEALTHY":
                logger.info(f"Platform healed to HEALTHY after attempt {attempt}")
                return all_results

        # If we exhausted attempts and still not HEALTHY, escalate
        if all_results:
            last_result = all_results[-1]
            if last_result.after_integrity != "HEALTHY":
                self._escalate_to_meta_remediation(all_results)
                # Mark last result as escalated in a new record
                escalation_record = {
                    "type": "escalation",
                    "task_id": last_result.task_id,
                    "reason": f"Self-healing failed after {len(all_results)} tasks",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                self._append_record(escalation_record)

        return all_results

    def escalate_to_meta_remediation(self, task: HealTask, result: HealResult) -> None:
        """
        Public API for escalation. Creates a Phase 26 meta-remediation task.

        Args:
            task: The failed HealTask
            result: The HealResult showing failure
        """
        self._escalate_to_meta_remediation_single(task, result)

    # ------------------------------------------------------------------
    # Private: safe command runners
    # ------------------------------------------------------------------

    @staticmethod
    def _run_install_packages() -> Optional[str]:
        """pip install -r requirements.txt inside the venv."""
        pip_path = EXPECTED_VENV / "bin" / "pip"
        if not pip_path.exists():
            return f"pip not found at {pip_path}"

        req_path = REQUIREMENTS_TXT
        if not req_path.exists():
            return f"requirements.txt not found at {req_path}"

        try:
            result = subprocess.run(
                [str(pip_path), "install", "-r", str(req_path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(EXPECTED_VENV.parent),
            )
            if result.returncode != 0:
                return f"pip install failed (exit {result.returncode}): {result.stderr[:500]}"
            return None  # Success
        except subprocess.TimeoutExpired:
            return "pip install timed out (120s)"
        except Exception as e:
            return f"pip install error: {e}"

    @staticmethod
    def _run_install_playwright() -> Optional[str]:
        """playwright install inside the venv."""
        python_path = EXPECTED_VENV / "bin" / "python"
        if not python_path.exists():
            return f"python not found at {python_path}"

        try:
            result = subprocess.run(
                [str(python_path), "-m", "playwright", "install"],
                capture_output=True,
                text=True,
                timeout=180,
            )
            if result.returncode != 0:
                return f"playwright install failed (exit {result.returncode}): {result.stderr[:500]}"
            return None
        except subprocess.TimeoutExpired:
            return "playwright install timed out (180s)"
        except Exception as e:
            return f"playwright install error: {e}"

    @staticmethod
    def _run_fix_permissions() -> Optional[str]:
        """chown -R on the jobs directory."""
        jobs_dir = Path("/home/aitesting.mybd.in/jobs")
        if not jobs_dir.exists():
            return f"Jobs directory not found: {jobs_dir}"

        try:
            result = subprocess.run(
                ["chown", "-R", "aitesting.mybd.in:aitesting.mybd.in", str(jobs_dir)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return f"chown failed (exit {result.returncode}): {result.stderr[:300]}"

            # Also fix data directory
            data_dir = Path("/home/aitesting.mybd.in/public_html/data")
            if data_dir.exists():
                subprocess.run(
                    ["chown", "-R", "aitesting.mybd.in:aitesting.mybd.in", str(data_dir)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            return None
        except subprocess.TimeoutExpired:
            return "chown timed out"
        except Exception as e:
            return f"fix permissions error: {e}"

    @staticmethod
    def _run_restart_services() -> Optional[str]:
        """Restart ai-testing-controller and ai-telegram-bot services."""
        try:
            for svc in ("ai-testing-controller", "ai-telegram-bot"):
                result = subprocess.run(
                    ["systemctl", "restart", svc],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    return f"restart {svc} failed (exit {result.returncode}): {result.stderr[:200]}"
            return None
        except subprocess.TimeoutExpired:
            return "service restart timed out"
        except Exception as e:
            return f"service restart error: {e}"

    @staticmethod
    def _run_clear_stale_locks() -> Optional[str]:
        """Remove stale lock files in jobs directory."""
        jobs_dir = Path("/home/aitesting.mybd.in/jobs")
        if not jobs_dir.exists():
            return None  # Nothing to clear

        try:
            cleared = 0
            for lock_file in jobs_dir.rglob("*.lock"):
                try:
                    lock_file.unlink()
                    cleared += 1
                except OSError:
                    pass
            logger.info(f"Cleared {cleared} stale lock files")
            return None
        except Exception as e:
            return f"clear stale locks error: {e}"

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_status() -> str:
        """Get current integrity status (fail-closed)."""
        try:
            from controller.runtime_integrity import run_integrity_check
            report = run_integrity_check()
            return report.status
        except Exception as e:
            logger.error(f"Integrity check failed during healing: {e}")
            return "FATAL"

    @staticmethod
    def _run_fresh_integrity_check():
        """Run a fresh integrity check, return report or None."""
        try:
            from controller.runtime_integrity import run_integrity_check
            return run_integrity_check()
        except Exception as e:
            logger.error(f"Cannot run integrity check: {e}")
            return None

    @staticmethod
    def _build_evidence(report, action: HealAction) -> str:
        """Build a human-readable evidence string for a heal task."""
        if action == HealAction.INSTALL_PYTHON_PACKAGE:
            return f"Missing packages: {list(report.missing_packages)}"
        elif action == HealAction.INSTALL_PLAYWRIGHT:
            return f"Playwright not OK (playwright_ok={report.playwright_ok})"
        elif action == HealAction.FIX_PERMISSIONS:
            return f"Filesystem permissions broken (fs_permissions_ok={report.fs_permissions_ok})"
        elif action == HealAction.RESTART_SERVICES:
            return "Service restart requested"
        elif action == HealAction.CLEAR_STALE_LOCKS:
            return f"Missing binaries (may be lock-related): {list(report.missing_binaries)}"
        return "Unknown issue"

    @staticmethod
    def _append_record(record: Dict[str, Any]) -> bool:
        """
        Append record to JSONL audit file with fsync.

        Returns True on success, False on failure (fail-closed).
        """
        try:
            HEAL_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(HEAL_AUDIT_PATH, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())
            return True
        except Exception as e:
            logger.error(f"Failed to write heal audit record: {e}")
            return False

    def _escalate_to_meta_remediation(self, results: List[HealResult]) -> None:
        """Escalate to Phase 26 meta-remediation when healing fails."""
        try:
            from controller.meta_remediator import get_meta_remediator
            remediator = get_meta_remediator()

            # Use the last failed result for context
            last = results[-1] if results else None
            task_id = last.task_id if last else "unknown"
            error = last.error if last else "Self-healing exhausted"

            can_attempt, reason = remediator.can_attempt_meta_remediation("__platform__")
            if not can_attempt:
                logger.warning(f"Cannot escalate to meta-remediation: {reason}")
                return

            meta_task = remediator.create_meta_remediation_task(
                project="__platform__",
                rescue_job_id=f"self-heal-{task_id[:12]}",
                rescue_exit_code=1,
                original_failure="runtime_integrity_fatal",
                module="runtime_self_healer",
            )
            if meta_task:
                logger.warning(
                    f"Escalated to meta-remediation: task={meta_task.id[:12]}, "
                    f"reason=self-healing failed after {len(results)} tasks"
                )
        except ImportError:
            logger.warning("Meta-remediator not available for escalation")
        except Exception as e:
            logger.error(f"Escalation to meta-remediation failed: {e}")

    def _escalate_to_meta_remediation_single(
        self, task: HealTask, result: HealResult
    ) -> None:
        """Escalate a single failed task to meta-remediation."""
        try:
            from controller.meta_remediator import get_meta_remediator
            remediator = get_meta_remediator()

            can_attempt, reason = remediator.can_attempt_meta_remediation("__platform__")
            if not can_attempt:
                logger.warning(f"Cannot escalate single task: {reason}")
                return

            remediator.create_meta_remediation_task(
                project="__platform__",
                rescue_job_id=f"self-heal-{task.id[:12]}",
                rescue_exit_code=1,
                original_failure=f"heal_failed:{task.action}",
                module="runtime_self_healer",
            )
        except Exception as e:
            logger.error(f"Single task escalation failed: {e}")

    @staticmethod
    def get_all_records() -> List[Dict[str, Any]]:
        """Read all records from the heal audit JSONL file."""
        records = []
        try:
            if HEAL_AUDIT_PATH.exists():
                with open(HEAL_AUDIT_PATH) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read heal audit records: {e}")
        return records


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_healer: Optional[RuntimeSelfHealer] = None


def get_runtime_self_healer() -> RuntimeSelfHealer:
    """Get the singleton RuntimeSelfHealer."""
    global _healer
    if _healer is None:
        _healer = RuntimeSelfHealer()
    return _healer
