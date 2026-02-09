"""
Phase 26.6: Runtime Integrity Enforcement Layer

Enforces the runtime integrity policy by blocking actions when the platform
is in FATAL state. Writes an immutable audit log for every enforcement decision.

Fail-closed: If integrity check CANNOT run, treat as FATAL.
"""

import json
import logging
import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger("runtime_enforcement")

# Audit log path (append-only)
ENFORCEMENT_AUDIT_PATH = Path("/home/aitesting.mybd.in/jobs/integrity_enforcement.jsonl")


class ActionType(str, Enum):
    """Actions that can be blocked by runtime integrity enforcement."""
    CREATE_PROJECT = "create_project"
    RESCUE = "rescue"
    MICRO_REMEDIATE = "micro_remediate"
    META_REMEDIATE = "meta_remediate"
    DISPATCH_EXECUTION = "dispatch_execution"
    CLAUDE_JOB_CREATE = "claude_job_create"
    DEEP_E2E_VERIFY = "deep_e2e_verify"
    DEPLOY_TEST = "deploy_test"
    DEPLOY_PROD = "deploy_prod"


class RuntimeIntegrityBlockedError(Exception):
    """Raised when an action is blocked due to FATAL runtime integrity."""

    def __init__(self, action: ActionType, status: str, policy: str):
        self.action = action
        self.status = status
        self.policy = policy
        super().__init__(
            f"Action '{action.value}' blocked: runtime integrity is {status} "
            f"(policy: {policy})"
        )


class RuntimeIntegrityEnforcer:
    """
    Enforces runtime integrity policy.

    check_or_block() raises RuntimeIntegrityBlockedError if the action
    should be blocked based on the current integrity status and policy.
    """

    @staticmethod
    def check_or_block(action: ActionType) -> None:
        """
        Check runtime integrity and block if policy requires it.

        Raises RuntimeIntegrityBlockedError if blocked.
        Fail-closed: if integrity check cannot run, treat as FATAL.
        """
        from controller.runtime_integrity_policy import get_policy, RuntimeIntegrityPolicy

        policy = get_policy()

        # Get current integrity status (fail-closed)
        status = RuntimeIntegrityEnforcer._get_status_fail_closed()

        blocked = False
        reason = None

        if status == "FATAL" and policy in (
            RuntimeIntegrityPolicy.BLOCK_CRITICAL,
            RuntimeIntegrityPolicy.STRICT,
        ):
            blocked = True
            reason = f"RUNTIME_INTEGRITY_FATAL: status={status}, policy={policy.value}"
        elif status == "DEGRADED":
            logger.warning(
                f"Runtime integrity DEGRADED for action {action.value} "
                f"(policy={policy.value}, allowing)"
            )
        elif status == "FATAL" and policy == RuntimeIntegrityPolicy.WARN_ONLY:
            logger.error(
                f"Runtime integrity FATAL for action {action.value} "
                f"(policy=warn_only, allowing with warning)"
            )

        # Write audit record (always, regardless of blocked/allowed)
        RuntimeIntegrityEnforcer._write_audit(
            action=action,
            status=status,
            policy=policy.value,
            blocked=blocked,
            reason=reason,
        )

        if blocked:
            logger.error(
                f"ACTION BLOCKED by runtime integrity: action={action.value}, "
                f"status={status}, policy={policy.value}"
            )
            raise RuntimeIntegrityBlockedError(action, status, policy.value)

    @staticmethod
    def is_blocked(action: ActionType) -> bool:
        """
        Check if an action would be blocked (non-raising variant).

        Returns True if the action would be blocked.
        """
        try:
            RuntimeIntegrityEnforcer.check_or_block(action)
            return False
        except RuntimeIntegrityBlockedError:
            return True
        except Exception:
            # Fail-closed: if we can't check, assume blocked
            return True

    @staticmethod
    def _get_status_fail_closed() -> str:
        """
        Get the current runtime integrity status.

        Fail-closed: if check cannot run, return "FATAL".
        """
        try:
            from controller.runtime_integrity import get_last_integrity_report, run_integrity_check

            report = get_last_integrity_report()
            if report is None:
                # No cached report — run a fresh check
                report = run_integrity_check()
            return report.status
        except ImportError:
            logger.error("runtime_integrity module not available — treating as FATAL")
            return "FATAL"
        except Exception as e:
            logger.error(f"Integrity check failed — treating as FATAL: {e}")
            return "FATAL"

    @staticmethod
    def _write_audit(
        action: ActionType,
        status: str,
        policy: str,
        blocked: bool,
        reason: Optional[str],
    ) -> None:
        """Append an enforcement audit record (append-only, best-effort)."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action_attempted": action.value,
            "integrity_status": status,
            "policy": policy,
            "blocked": blocked,
            "reason": reason,
        }
        try:
            ENFORCEMENT_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(ENFORCEMENT_AUDIT_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            # Audit write failure is logged but does NOT block the action
            # (the blocking decision was already made above)
            logger.error(f"Failed to write enforcement audit: {e}")
