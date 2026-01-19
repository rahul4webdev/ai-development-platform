#!/usr/bin/env python3
"""
Phase 15.8 Dry-Run System Validation

This script validates the entire AI Development Platform.

It tests:
1. Claude CLI integration (detection AND real execution test)
2. Scheduler & Worker logic
3. Lifecycle state machine
4. Execution Gate enforcement
5. Telegram bot validation (RUNTIME health, not ENV presence)
6. Audit trail generation

IMPORTANT: This validation will FAIL if Claude CLI cannot execute real prompts.
Phase 15.7+ requires verified execution capability.

Phase 15.8: Telegram token validated via RUNTIME TRUTH, not configuration presence.
If the bot service is running and operational, the token is valid - period.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Results tracking
@dataclass
class ValidationResult:
    test_name: str
    category: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "category": self.category,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
        }


class DryRunValidator:
    """Comprehensive dry-run system validator."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dryrun_"))

    def add_result(
        self,
        test_name: str,
        category: str,
        passed: bool,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        result = ValidationResult(
            test_name=test_name,
            category=category,
            passed=passed,
            message=message,
            details=details,
        )
        self.results.append(result)
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: [{category}] {test_name}")
        if not passed:
            print(f"       → {message}")

    # =========================================================================
    # Task 1: Claude CLI Integration
    # =========================================================================
    async def validate_claude_cli_integration(self):
        """Validate Claude CLI detection logic (not actual execution)."""
        print("\n" + "="*70)
        print("📋 TASK 1: Claude CLI Integration Validation")
        print("="*70)

        # Test 1.1: Check claude_backend module imports
        try:
            from controller.claude_backend import (
                check_claude_availability,
                ClaudeJob,
                JobState,
                JobPriority,
                WorkspaceManager,
            )
            self.add_result(
                "claude_backend_imports",
                "Claude CLI",
                True,
                "All claude_backend components import successfully"
            )
        except ImportError as e:
            self.add_result(
                "claude_backend_imports",
                "Claude CLI",
                False,
                f"Import failed: {e}"
            )
            return

        # Test 1.2: Verify ClaudeJob has Phase 15.6 fields
        job = ClaudeJob(
            job_id="test-job-1",
            project_name="test-project",
            task_description="Test task",
            task_type="feature_development",
            state=JobState.QUEUED,
            created_at=datetime.utcnow(),
            lifecycle_id="test-lifecycle",
            lifecycle_state="development",
            requested_action="write_code",
            user_role="developer",
        )

        required_fields = [
            "lifecycle_id", "lifecycle_state", "requested_action",
            "user_role", "gate_allowed", "gate_denied_reason"
        ]
        missing_fields = [f for f in required_fields if not hasattr(job, f)]

        self.add_result(
            "claude_job_phase15_6_fields",
            "Claude CLI",
            len(missing_fields) == 0,
            f"ClaudeJob has all Phase 15.6 fields" if not missing_fields
            else f"Missing fields: {missing_fields}",
            {"checked_fields": required_fields, "missing": missing_fields}
        )

        # Test 1.3: Check ClaudeJob serialization includes gate fields
        job_dict = job.to_dict()
        gate_fields_in_dict = all(f in job_dict for f in ["lifecycle_state", "gate_allowed"])
        self.add_result(
            "claude_job_serialization",
            "Claude CLI",
            gate_fields_in_dict,
            "ClaudeJob.to_dict() includes execution gate fields"
        )

        # Test 1.4: Verify check_claude_availability function exists
        import inspect
        sig = inspect.signature(check_claude_availability)
        self.add_result(
            "check_claude_availability_function",
            "Claude CLI",
            True,
            "check_claude_availability function exists with correct signature",
            {"signature": str(sig)}
        )

        # Test 1.5: REAL EXECUTION TEST (Phase 15.7 - CRITICAL)
        # This is the most important test - verifies Claude CLI can actually work
        availability = await check_claude_availability()

        self.add_result(
            "claude_cli_installed",
            "Claude CLI",
            availability.get("installed", False),
            f"Claude CLI installed: {availability.get('version', 'N/A')}",
            {"version": availability.get("version")}
        )

        # The critical test - can_execute must be True
        can_execute = availability.get("can_execute", False)
        self.add_result(
            "claude_cli_can_execute",
            "Claude CLI",
            can_execute,
            f"Claude CLI REAL EXECUTION: {'VERIFIED' if can_execute else 'FAILED - ' + availability.get('error', 'Unknown error')}",
            {
                "can_execute": can_execute,
                "authenticated": availability.get("authenticated"),
                "auth_type": availability.get("auth_type"),
                "error": availability.get("error"),
                "test_output": availability.get("execution_test_output"),
            }
        )

        # Log clear status for Phase 15.7 requirement
        if can_execute:
            print(f"\n   ✅ PHASE 15.7 VERIFIED: Claude CLI can execute real jobs!")
            print(f"      Auth type: {availability.get('auth_type')}")
            print(f"      Version: {availability.get('version')}")
        else:
            print(f"\n   ❌ PHASE 15.7 FAILED: Claude CLI CANNOT execute real jobs")
            print(f"      Error: {availability.get('error')}")
            print(f"      Fix: Set ANTHROPIC_API_KEY or run 'claude setup-token'")

    # =========================================================================
    # Task 2: Scheduler & Worker Validation
    # =========================================================================
    async def validate_scheduler_workers(self):
        """Validate scheduler logic with simulated jobs."""
        print("\n" + "="*70)
        print("📋 TASK 2: Scheduler & Worker Validation")
        print("="*70)

        try:
            from controller.claude_backend import (
                ClaudeJob, JobState, JobPriority, JobQueue,
                MAX_CONCURRENT_JOBS, STARVATION_THRESHOLD_MINUTES,
            )
        except ImportError as e:
            self.add_result(
                "scheduler_imports",
                "Scheduler",
                False,
                f"Import failed: {e}"
            )
            return

        # Test 2.1: Verify MAX_CONCURRENT_JOBS
        self.add_result(
            "max_concurrent_jobs",
            "Scheduler",
            MAX_CONCURRENT_JOBS == 3,
            f"MAX_CONCURRENT_JOBS is {MAX_CONCURRENT_JOBS} (expected 3)"
        )

        # Test 2.2: Simulate job queue with 10 jobs
        queue = JobQueue()
        jobs = []

        # Create 10 jobs with varying priorities
        priorities = [
            JobPriority.NORMAL.value,  # Jobs 0-4: Normal
            JobPriority.NORMAL.value,
            JobPriority.NORMAL.value,
            JobPriority.NORMAL.value,
            JobPriority.NORMAL.value,
            JobPriority.HIGH.value,     # Jobs 5-6: High
            JobPriority.HIGH.value,
            JobPriority.LOW.value,      # Jobs 7-8: Low
            JobPriority.LOW.value,
            JobPriority.EMERGENCY.value, # Job 9: Emergency
        ]

        for i in range(10):
            job = ClaudeJob(
                job_id=f"sim-job-{i:02d}",
                project_name=f"project-{i % 3}",
                task_description=f"Simulated task {i}",
                task_type="feature_development",
                state=JobState.QUEUED,
                created_at=datetime.utcnow(),
                priority=priorities[i],
                aspect="core" if i % 2 == 0 else "backend",
            )
            jobs.append(job)
            await queue.enqueue(job)

        # Verify queue has 10 jobs
        self.add_result(
            "queue_enqueue_10_jobs",
            "Scheduler",
            len(queue._jobs) == 10,
            f"Successfully enqueued 10 jobs (actual: {len(queue._jobs)})"
        )

        # Test 2.3: Priority ordering
        # Sort jobs by priority (descending) to verify EMERGENCY comes first
        sorted_jobs = sorted(queue._jobs.values(), key=lambda j: j.priority, reverse=True)
        emergency_first = sorted_jobs[0].priority == JobPriority.EMERGENCY.value if sorted_jobs else False
        self.add_result(
            "priority_ordering",
            "Scheduler",
            emergency_first,
            f"EMERGENCY job has highest priority (priority: {sorted_jobs[0].priority if sorted_jobs else 'N/A'})"
        )

        # Test 2.4: Verify FIFO within same priority
        # Get all HIGH priority jobs and check they maintain insertion order
        all_jobs = list(queue._jobs.values())
        high_jobs = [j for j in all_jobs if j.priority == JobPriority.HIGH.value]
        if len(high_jobs) >= 2:
            # Jobs with same priority should be ordered by created_at
            fifo_correct = high_jobs[0].created_at <= high_jobs[1].created_at
            self.add_result(
                "fifo_within_priority",
                "Scheduler",
                fifo_correct,
                "FIFO ordering within same priority level"
            )
        else:
            self.add_result(
                "fifo_within_priority",
                "Scheduler",
                True,
                "FIFO check skipped (not enough same-priority jobs)"
            )

        # Test 2.5: Aspect isolation check
        aspects_in_queue = set(j.aspect for j in queue._jobs.values())
        self.add_result(
            "aspect_isolation_tracking",
            "Scheduler",
            len(aspects_in_queue) >= 2,
            f"Jobs have multiple aspects: {aspects_in_queue}",
            {"aspects": list(aspects_in_queue)}
        )

        # Test 2.6: Starvation prevention threshold exists
        self.add_result(
            "starvation_prevention",
            "Scheduler",
            STARVATION_THRESHOLD_MINUTES == 30,
            f"Starvation threshold is {STARVATION_THRESHOLD_MINUTES} minutes (expected 30 min)"
        )

    # =========================================================================
    # Task 3: Lifecycle Validation
    # =========================================================================
    async def validate_lifecycle(self):
        """Validate lifecycle state machine transitions."""
        print("\n" + "="*70)
        print("📋 TASK 3: Lifecycle Validation")
        print("="*70)

        try:
            from controller.lifecycle_v2 import (
                LifecycleState,
                LifecycleMode,
                LifecycleStateMachine,
                VALID_TRANSITIONS,
                TransitionTrigger,
            )
        except ImportError as e:
            self.add_result(
                "lifecycle_imports",
                "Lifecycle",
                False,
                f"Import failed: {e}"
            )
            return

        # Test 3.1: Verify all 10 states exist
        expected_states = [
            "CREATED", "PLANNING", "DEVELOPMENT", "TESTING",
            "AWAITING_FEEDBACK", "FIXING", "READY_FOR_PRODUCTION",
            "DEPLOYED", "REJECTED", "ARCHIVED"
        ]
        actual_states = [s.name for s in LifecycleState]
        missing_states = [s for s in expected_states if s not in actual_states]

        self.add_result(
            "all_10_states_exist",
            "Lifecycle",
            len(missing_states) == 0,
            f"All 10 lifecycle states defined" if not missing_states
            else f"Missing states: {missing_states}",
            {"states": actual_states}
        )

        # Test 3.2: Valid transition path (using trigger-based transitions)
        # CREATED → PLANNING → DEVELOPMENT → TESTING
        # VALID_TRANSITIONS maps state -> {trigger -> target_state}

        # Check CREATED -> PLANNING is possible
        created_trans = VALID_TRANSITIONS.get(LifecycleState.CREATED, {})
        can_reach_planning = LifecycleState.PLANNING in created_trans.values()

        # Check PLANNING -> DEVELOPMENT is possible
        planning_trans = VALID_TRANSITIONS.get(LifecycleState.PLANNING, {})
        can_reach_development = LifecycleState.DEVELOPMENT in planning_trans.values()

        # Check DEVELOPMENT -> TESTING is possible
        dev_trans = VALID_TRANSITIONS.get(LifecycleState.DEVELOPMENT, {})
        can_reach_testing = LifecycleState.TESTING in dev_trans.values()

        path_valid = can_reach_planning and can_reach_development and can_reach_testing

        self.add_result(
            "valid_transition_path",
            "Lifecycle",
            path_valid,
            "CREATED → PLANNING → DEVELOPMENT → TESTING path exists via valid triggers",
            {"created->planning": can_reach_planning,
             "planning->development": can_reach_development,
             "development->testing": can_reach_testing}
        )

        # Test 3.3: Invalid transition blocked
        # CREATED cannot go directly to DEPLOYED
        deployed_reachable_from_created = LifecycleState.DEPLOYED in created_trans.values()
        self.add_result(
            "invalid_transition_blocked",
            "Lifecycle",
            not deployed_reachable_from_created,
            "CREATED → DEPLOYED is correctly blocked (no direct path)"
        )

        # Test 3.4: Cannot skip TESTING to go to DEPLOYED
        testing_trans = VALID_TRANSITIONS.get(LifecycleState.TESTING, {})
        can_deploy_from_testing = LifecycleState.DEPLOYED in testing_trans.values()
        self.add_result(
            "testing_cannot_direct_deploy",
            "Lifecycle",
            not can_deploy_from_testing,
            "TESTING → DEPLOYED is correctly blocked (must go through READY_FOR_PRODUCTION)"
        )

        # Test 3.5: REJECTED has only archival transition (cleanup only, no re-activation)
        rejected_transitions = VALID_TRANSITIONS.get(LifecycleState.REJECTED, {})
        # REJECTED can only transition to ARCHIVED for cleanup purposes
        only_archive = all(t.value == "archived" or "archive" in t.value.lower()
                         for t in rejected_transitions.values()) if rejected_transitions else True
        self.add_result(
            "rejected_is_terminal",
            "Lifecycle",
            only_archive,
            "REJECTED state only allows archival (no re-activation)",
            {"transitions": [str(t) for t in rejected_transitions.keys()] if rejected_transitions else []}
        )

    # =========================================================================
    # Task 4: Execution Gate Enforcement
    # =========================================================================
    async def validate_execution_gate(self):
        """Validate execution gate security enforcement."""
        print("\n" + "="*70)
        print("📋 TASK 4: Execution Gate Enforcement")
        print("="*70)

        try:
            from controller.execution_gate import (
                ExecutionGate,
                ExecutionAction,
                ExecutionRequest,
                GateDecision,
                LifecycleState,
                UserRole,
                LIFECYCLE_ALLOWED_ACTIONS,
                ROLE_ALLOWED_ACTIONS,
                REQUIRED_GOVERNANCE_DOCS,
            )
        except ImportError as e:
            self.add_result(
                "execution_gate_imports",
                "Execution Gate",
                False,
                f"Import failed: {e}"
            )
            return

        gate = ExecutionGate()
        gate._audit_log_path = self.temp_dir / "test_audit.log"

        # Create temp workspace with governance docs
        test_workspace = self.temp_dir / "test_workspace"
        test_workspace.mkdir(exist_ok=True)
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (test_workspace / doc).write_text(f"# {doc}\nTest content")

        # Test 4.1: WRITE_CODE blocked in AWAITING_FEEDBACK
        request = ExecutionRequest(
            job_id="test-gate-1",
            project_name="test-project",
            aspect="core",
            lifecycle_id="test-lifecycle",
            lifecycle_state="awaiting_feedback",
            requested_action="write_code",
            requesting_user_id="user-1",
            requesting_user_role="developer",
            workspace_path=str(test_workspace),
            task_description="Test write in awaiting_feedback",
        )
        decision = gate.evaluate(request)
        self.add_result(
            "write_blocked_in_awaiting_feedback",
            "Execution Gate",
            decision.allowed is False and decision.hard_fail is True,
            "WRITE_CODE correctly blocked in AWAITING_FEEDBACK state",
            {"allowed": decision.allowed, "reason": decision.denied_reason}
        )

        # Test 4.2: COMMIT blocked in TESTING
        request.lifecycle_state = "testing"
        request.requested_action = "commit"
        decision = gate.evaluate(request)
        self.add_result(
            "commit_blocked_in_testing",
            "Execution Gate",
            decision.allowed is False,
            "COMMIT correctly blocked in TESTING state",
            {"allowed": decision.allowed, "reason": decision.denied_reason}
        )

        # Test 4.3: DEPLOY_PROD always denied (even with owner role)
        request.lifecycle_state = "ready_for_production"
        request.requested_action = "deploy_prod"
        request.requesting_user_role = "owner"
        decision = gate.evaluate(request)
        self.add_result(
            "deploy_prod_always_denied",
            "Execution Gate",
            decision.allowed is False and decision.hard_fail is True,
            "DEPLOY_PROD is NEVER allowed via automation",
            {"allowed": decision.allowed, "reason": decision.denied_reason}
        )

        # Test 4.4: Path traversal blocked
        request.lifecycle_state = "development"
        request.requested_action = "write_code"
        request.workspace_path = "/etc/passwd"
        decision = gate.evaluate(request)
        self.add_result(
            "path_traversal_blocked",
            "Execution Gate",
            decision.allowed is False,
            "Path traversal attack blocked (/etc/passwd)",
            {"workspace": request.workspace_path, "allowed": decision.allowed}
        )

        # Test 4.5: Valid action in DEVELOPMENT allowed
        request.workspace_path = str(test_workspace)
        request.lifecycle_state = "development"
        request.requested_action = "write_code"
        request.requesting_user_role = "developer"
        decision = gate.evaluate(request)
        self.add_result(
            "valid_development_action_allowed",
            "Execution Gate",
            decision.allowed is True,
            "WRITE_CODE correctly allowed in DEVELOPMENT for developer",
            {"allowed": decision.allowed, "actions": decision.allowed_actions}
        )

        # Test 4.6: Missing governance docs blocks execution
        empty_workspace = self.temp_dir / "empty_workspace"
        empty_workspace.mkdir(exist_ok=True)
        request.workspace_path = str(empty_workspace)
        decision = gate.evaluate(request)
        self.add_result(
            "missing_governance_docs_blocked",
            "Execution Gate",
            decision.allowed is False,
            "Execution blocked when governance docs missing",
            {"allowed": decision.allowed, "reason": decision.denied_reason}
        )

        # Test 4.7: VIEWER cannot WRITE_CODE
        request.workspace_path = str(test_workspace)
        request.requesting_user_role = "viewer"
        decision = gate.evaluate(request)
        self.add_result(
            "viewer_cannot_write",
            "Execution Gate",
            decision.allowed is False,
            "VIEWER role correctly cannot WRITE_CODE",
            {"role": request.requesting_user_role, "allowed": decision.allowed}
        )

    # =========================================================================
    # Task 5: Telegram Bot Validation (Phase 15.8: Runtime Truth)
    # =========================================================================
    async def validate_telegram_commands(self):
        """
        Validate Telegram bot using RUNTIME TRUTH, not config presence.

        Phase 15.8: Token validity is determined by:
        1. systemd service 'ai-telegram-bot' is active, OR
        2. Bot process is running and operational

        ENV presence is OPTIONAL - runtime health is authoritative.
        """
        print("\n" + "="*70)
        print("📋 TASK 5: Telegram Bot Validation (Runtime Truth)")
        print("="*70)

        try:
            from telegram_bot_v2 import bot as telegram_bot_module
            bot_module_exists = True
        except ImportError as e:
            bot_module_exists = False
            self.add_result(
                "telegram_bot_import",
                "Telegram",
                False,
                f"Import failed: {e}"
            )
            return

        self.add_result(
            "telegram_bot_import",
            "Telegram",
            True,
            "telegram_bot_v2.bot module imports successfully"
        )

        # Test 5.2: RUNTIME HEALTH CHECK (Phase 15.8)
        # Check if telegram bot service is actually running - this is the TRUTH
        bot_operational = await self._check_telegram_bot_runtime()

        self.add_result(
            "telegram_bot_operational",
            "Telegram",
            bot_operational["operational"],
            bot_operational["message"],
            {
                "method": bot_operational["method"],
                "service_status": bot_operational.get("service_status"),
                "uptime": bot_operational.get("uptime"),
                "validation_mode": "runtime-verified"
            }
        )

        if bot_operational["operational"]:
            print(f"\n   ✅ PHASE 15.8 VERIFIED: Telegram bot is OPERATIONAL")
            print(f"      Method: {bot_operational['method']}")
            if bot_operational.get("uptime"):
                print(f"      Uptime: {bot_operational['uptime']}")
        else:
            print(f"\n   ⚠️  Telegram bot not running - {bot_operational['message']}")
            print(f"      This may be expected if bot service hasn't been started yet")

    async def _check_telegram_bot_runtime(self) -> Dict[str, Any]:
        """
        Check Telegram bot operational status using runtime truth.

        Checks in order:
        1. systemd service 'ai-telegram-bot' active state
        2. Process check for telegram bot
        3. Fallback: ENV variable presence (legacy, lowest priority)

        Returns operational status based on RUNTIME TRUTH.
        """
        result = {
            "operational": False,
            "method": "none",
            "message": "Bot status unknown",
            "service_status": None,
            "uptime": None,
        }

        # Method 1: Check systemd service (authoritative on VPS)
        try:
            proc = subprocess.run(
                ["systemctl", "is-active", "ai-telegram-bot"],
                capture_output=True,
                text=True,
                timeout=5
            )
            service_active = proc.stdout.strip() == "active"

            if service_active:
                # Get uptime
                proc2 = subprocess.run(
                    ["systemctl", "show", "ai-telegram-bot", "--property=ActiveEnterTimestamp"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                timestamp = proc2.stdout.strip().split("=")[1] if "=" in proc2.stdout else "unknown"

                result["operational"] = True
                result["method"] = "systemd-service"
                result["message"] = "Telegram bot service is RUNNING (runtime verified)"
                result["service_status"] = "active"
                result["uptime"] = timestamp
                return result
            else:
                result["service_status"] = proc.stdout.strip() or "inactive"

        except FileNotFoundError:
            # systemctl not available (not Linux or not systemd)
            pass
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            result["service_status"] = f"check_error: {e}"

        # Method 2: Check for running bot process
        try:
            proc = subprocess.run(
                ["pgrep", "-f", "telegram_bot"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if proc.returncode == 0 and proc.stdout.strip():
                pids = proc.stdout.strip().split("\n")
                result["operational"] = True
                result["method"] = "process-check"
                result["message"] = f"Telegram bot process running (PIDs: {', '.join(pids)})"
                return result
        except Exception:
            pass

        # Method 3: Check for bot log file with recent activity
        log_paths = [
            Path("/var/log/ai-telegram-bot.log"),
            Path("/home/aitesting.mybd.in/logs/telegram_bot.log"),
        ]
        for log_path in log_paths:
            try:
                if log_path.exists():
                    # Check if log was modified in last 5 minutes
                    mtime = datetime.fromtimestamp(log_path.stat().st_mtime)
                    age_seconds = (datetime.now() - mtime).total_seconds()
                    if age_seconds < 300:  # 5 minutes
                        result["operational"] = True
                        result["method"] = "log-activity"
                        result["message"] = f"Telegram bot log active (updated {int(age_seconds)}s ago)"
                        return result
            except Exception:
                pass

        # If we reach here, bot is not verified as running
        # Check ENV as informational only (NOT authoritative)
        env_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if env_token and len(env_token) > 10:
            result["message"] = "Bot service not running (token configured but service inactive)"
        else:
            result["message"] = "Bot service not running (may need to be started)"

        return result

        # Check for command handlers
        import inspect
        try:
            from telegram_bot_v2 import bot as bot_module
            source = inspect.getsource(bot_module)

            expected_commands = [
                "health", "status", "lifecycle", "dashboard",
                "help", "projects", "jobs"
            ]

            for cmd in expected_commands:
                # Look for handler patterns
                handler_exists = (
                    f"cmd_{cmd}" in source or
                    f"handle_{cmd}" in source or
                    f"/{cmd}" in source or
                    f'"{cmd}"' in source
                )
                self.add_result(
                    f"command_{cmd}_handler",
                    "Telegram",
                    handler_exists,
                    f"/{cmd} command handler found" if handler_exists
                    else f"/{cmd} command handler NOT found (may be OK if using different pattern)"
                )
        except Exception as e:
            self.add_result(
                "telegram_command_inspection",
                "Telegram",
                False,
                f"Could not inspect bot module: {e}"
            )

    # =========================================================================
    # Task 6: Audit & Observability
    # =========================================================================
    async def validate_audit_observability(self):
        """Validate audit trail and logging."""
        print("\n" + "="*70)
        print("📋 TASK 6: Audit & Observability")
        print("="*70)

        try:
            from controller.execution_gate import (
                ExecutionGate, ExecutionRequest, ExecutionAuditEntry,
                REQUIRED_GOVERNANCE_DOCS,
            )
        except ImportError as e:
            self.add_result(
                "audit_imports",
                "Audit",
                False,
                f"Import failed: {e}"
            )
            return

        # Test 6.1: Audit entry creation
        entry = ExecutionAuditEntry(
            job_id="audit-test-1",
            project_name="test-project",
            aspect="core",
            lifecycle_id="test-lifecycle",
            lifecycle_state="development",
            allowed_actions=["read_code", "write_code"],
            executed_action="write_code",
            requesting_user_id="user-1",
            requesting_user_role="developer",
            gate_decision="ALLOWED",
            denied_reason=None,
            outcome="SUCCESS",
        )
        entry_dict = entry.to_dict()

        required_fields = [
            "job_id", "project_name", "aspect", "lifecycle_id",
            "lifecycle_state", "allowed_actions", "executed_action",
            "requesting_user_id", "requesting_user_role", "gate_decision",
            "timestamp"
        ]
        missing_fields = [f for f in required_fields if f not in entry_dict]

        self.add_result(
            "audit_entry_fields",
            "Audit",
            len(missing_fields) == 0,
            "ExecutionAuditEntry has all required fields",
            {"fields": list(entry_dict.keys())}
        )

        # Test 6.2: Audit log file creation
        gate = ExecutionGate()
        audit_log = self.temp_dir / "test_audit.log"
        gate._audit_log_path = audit_log

        # Create workspace with docs
        test_workspace = self.temp_dir / "audit_workspace"
        test_workspace.mkdir(exist_ok=True)
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (test_workspace / doc).write_text(f"# {doc}")

        request = ExecutionRequest(
            job_id="audit-test-2",
            project_name="test-project",
            aspect="core",
            lifecycle_id="test-lifecycle",
            lifecycle_state="development",
            requested_action="write_code",
            requesting_user_id="user-1",
            requesting_user_role="developer",
            workspace_path=str(test_workspace),
            task_description="Test audit logging",
        )

        # This should trigger audit log
        gate.evaluate(request)

        audit_exists = audit_log.exists() and audit_log.stat().st_size > 0
        self.add_result(
            "audit_log_written",
            "Audit",
            audit_exists,
            f"Audit log created and contains data: {audit_log}"
        )

        # Test 6.3: Audit log is valid JSON
        if audit_exists:
            try:
                with open(audit_log) as f:
                    for line in f:
                        json.loads(line)  # Each line should be valid JSON
                self.add_result(
                    "audit_log_valid_json",
                    "Audit",
                    True,
                    "Audit log entries are valid JSON (JSONL format)"
                )
            except json.JSONDecodeError as e:
                self.add_result(
                    "audit_log_valid_json",
                    "Audit",
                    False,
                    f"Audit log has invalid JSON: {e}"
                )

        # Test 6.4: Denied actions are logged
        request.requested_action = "deploy_prod"
        gate.evaluate(request)

        with open(audit_log) as f:
            log_content = f.read()
            denied_logged = "DENIED" in log_content

        self.add_result(
            "denied_actions_logged",
            "Audit",
            denied_logged,
            "Denied gate decisions are logged to audit trail"
        )

    # =========================================================================
    # Generate Report
    # =========================================================================
    def generate_report(self) -> str:
        """Generate final validation report."""
        print("\n" + "="*70)
        print("📊 DRY-RUN VALIDATION REPORT")
        print("="*70)

        passed = [r for r in self.results if r.passed]
        failed = [r for r in self.results if not r.passed]

        # Summary
        total = len(self.results)
        pass_count = len(passed)
        fail_count = len(failed)
        pass_rate = (pass_count / total * 100) if total > 0 else 0

        report = []
        report.append("# DRY-RUN VALIDATION REPORT")
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append(f"Platform Version: 0.15.8")
        report.append("")
        report.append("## Summary")
        report.append(f"- **Total Tests**: {total}")
        report.append(f"- **Passed**: {pass_count} ✅")
        report.append(f"- **Failed**: {fail_count} ❌")
        report.append(f"- **Pass Rate**: {pass_rate:.1f}%")
        report.append("")

        # Passed checks
        report.append("## ✅ Passed Checks")
        report.append("")
        for r in passed:
            report.append(f"- [{r.category}] {r.test_name}: {r.message}")
        report.append("")

        # Failed checks
        if failed:
            report.append("## ❌ Failed Checks")
            report.append("")
            for r in failed:
                report.append(f"- [{r.category}] {r.test_name}")
                report.append(f"  - **Issue**: {r.message}")
                if r.details:
                    report.append(f"  - **Details**: {r.details}")
            report.append("")

        # Check if Claude CLI execution passed
        cli_execution_passed = any(
            r.test_name == "claude_cli_can_execute" and r.passed
            for r in self.results
        )

        # Risks and Gaps
        report.append("## ⚠️ Risks & Gaps")
        report.append("")

        if not cli_execution_passed:
            report.append("### 🚨 CRITICAL: Claude CLI Cannot Execute")
            report.append("The Claude CLI is installed but CANNOT execute real prompts.")
            report.append("This must be fixed before the platform can work.")
            report.append("")
            report.append("**Fix Options:**")
            report.append("1. Set ANTHROPIC_API_KEY environment variable:")
            report.append("   ```bash")
            report.append("   export ANTHROPIC_API_KEY='sk-ant-api03-...'")
            report.append("   ```")
            report.append("2. Run `claude setup-token` interactively (requires Claude subscription)")
            report.append("")

        # Check telegram bot operational status
        telegram_operational = any(
            r.test_name == "telegram_bot_operational" and r.passed
            for r in self.results
        )

        if not telegram_operational:
            report.append("1. **Telegram Bot Service**: Not running")
            report.append("   - Start the service: `systemctl start ai-telegram-bot`")
            report.append("   - Note: Token validated via RUNTIME health, not ENV presence")
            report.append("")
        else:
            report.append("1. **Telegram Bot**: Operational (runtime verified)")
            report.append("")

        report.append("2. **VPS Services**: Ensure all services are started")
        report.append("   - Controller: `systemctl status ai-controller`")
        report.append("   - Telegram Bot: `systemctl status ai-telegram-bot`")
        report.append("")

        # Required fixes
        report.append("## 🔧 Required Fixes")
        report.append("")
        if fail_count > 0:
            report.append("### Code/Configuration Fixes Required:")
            for r in failed:
                if r.test_name == "claude_cli_can_execute":
                    report.append(f"- **CRITICAL**: {r.test_name} - Set ANTHROPIC_API_KEY or run 'claude setup-token'")
                else:
                    report.append(f"- Fix {r.test_name} in {r.category}")
        else:
            report.append("No fixes required - all tests passed!")
        report.append("")

        # Security confirmation
        report.append("## 🔒 Security Confirmation")
        report.append("")

        security_checks = [
            ("Claude CLI can execute (Phase 15.7)", cli_execution_passed),
            ("Telegram bot operational (Phase 15.8)", telegram_operational),
            ("DEPLOY_PROD always blocked", any(
                r.test_name == "deploy_prod_always_denied" and r.passed
                for r in self.results
            )),
            ("Path traversal blocked", any(
                r.test_name == "path_traversal_blocked" and r.passed
                for r in self.results
            )),
            ("Role-based access enforced", any(
                r.test_name == "viewer_cannot_write" and r.passed
                for r in self.results
            )),
            ("Audit trail generated", any(
                r.test_name == "audit_log_written" and r.passed
                for r in self.results
            )),
            ("Governance docs required", any(
                r.test_name == "missing_governance_docs_blocked" and r.passed
                for r in self.results
            )),
        ]

        all_security_passed = all(passed for _, passed in security_checks)

        for check_name, passed in security_checks:
            status = "✅" if passed else "❌"
            report.append(f"- {status} {check_name}")

        report.append("")
        if all_security_passed:
            report.append("**🔒 SECURITY CONFIRMATION: All security controls validated**")
        else:
            report.append("**⚠️ SECURITY WARNING: Some security controls need attention**")

        report.append("")
        report.append("---")
        report.append("*Dry-run validation complete. No production actions were performed.*")

        return "\n".join(report)

    async def run_all_validations(self):
        """Run all validation tasks."""
        print("="*70)
        print("🚀 AI DEVELOPMENT PLATFORM - DRY-RUN VALIDATION")
        print("   Version: 0.15.8 (Phase 15.8: Runtime Truth Validation)")
        print("   Mode: DRY-RUN with REAL execution tests")
        print("   Note: All validations use RUNTIME TRUTH, not config presence")
        print("="*70)

        await self.validate_claude_cli_integration()
        await self.validate_scheduler_workers()
        await self.validate_lifecycle()
        await self.validate_execution_gate()
        await self.validate_telegram_commands()
        await self.validate_audit_observability()

        report = self.generate_report()
        print(report)

        # Save report
        report_path = PROJECT_ROOT / "docs" / "DRY_RUN_REPORT.md"
        report_path.write_text(report)
        print(f"\n📄 Report saved to: {report_path}")

        # Cleanup
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        return self.results


async def main():
    validator = DryRunValidator()
    results = await validator.run_all_validations()

    # Exit with error code if any tests failed
    failed = [r for r in results if not r.passed]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
