"""
Phase 26.7: Platform Self-Healing Layer Tests

Tests:
- PART 1: HealAction and HealStatus enum validation
- PART 2: HealTask and HealResult frozen dataclass immutability
- PART 3: Deterministic classification rules
- PART 4: Task creation correctness
- PART 5: Execution paths for each HealAction (mocked commands)
- PART 6: Bounded attempts (max 3)
- PART 7: Escalation to meta-remediation on repeated failure
- PART 8: Audit persistence with fsync
- PART 9: Integration with runtime_enforcement blocking logic
- PART 10: Telegram /runtime_heal output formatting
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Helper: create a mock RuntimeIntegrityReport
def _make_report(
    venv_ok=True,
    missing_packages=(),
    playwright_ok=True,
    missing_binaries=(),
    fs_permissions_ok=True,
    status="HEALTHY",
):
    """Create a mock RuntimeIntegrityReport-like object."""
    report = MagicMock()
    report.venv_ok = venv_ok
    report.missing_packages = tuple(missing_packages)
    report.playwright_ok = playwright_ok
    report.missing_binaries = tuple(missing_binaries)
    report.fs_permissions_ok = fs_permissions_ok
    report.status = status
    return report


# ---------------------------------------------------------------------------
# PART 1: Enum Validation
# ---------------------------------------------------------------------------

class TestHealActionEnum:
    """HealAction enum has all required values."""

    def test_all_values_defined(self):
        from controller.runtime_self_healer import HealAction
        expected = [
            "install_python_package", "install_playwright", "fix_permissions",
            "restart_services", "clear_stale_locks", "no_op",
        ]
        actual = [a.value for a in HealAction]
        for val in expected:
            assert val in actual, f"{val} missing from HealAction"

    def test_exactly_six_values(self):
        from controller.runtime_self_healer import HealAction
        assert len(HealAction) == 6

    def test_is_string_enum(self):
        from controller.runtime_self_healer import HealAction
        assert isinstance(HealAction.NO_OP.value, str)
        assert HealAction.NO_OP == "no_op"


class TestHealStatusEnum:
    """HealStatus enum has all required values."""

    def test_all_values_defined(self):
        from controller.runtime_self_healer import HealStatus
        expected = ["pending", "in_progress", "success", "failed", "escalated"]
        actual = [s.value for s in HealStatus]
        for val in expected:
            assert val in actual, f"{val} missing from HealStatus"

    def test_exactly_five_values(self):
        from controller.runtime_self_healer import HealStatus
        assert len(HealStatus) == 5


# ---------------------------------------------------------------------------
# PART 2: Frozen Dataclass Immutability
# ---------------------------------------------------------------------------

class TestDataclassImmutability:
    """HealTask and HealResult are frozen (immutable)."""

    def test_heal_task_is_frozen(self):
        from controller.runtime_self_healer import HealTask
        task = HealTask(
            id="t-1", project_scope=None, action="install_python_package",
            evidence="Missing packages: ['aiohttp']",
            created_at="2026-02-09T00:00:00",
        )
        with pytest.raises(AttributeError):
            task.action = "no_op"

    def test_heal_result_is_frozen(self):
        from controller.runtime_self_healer import HealResult
        result = HealResult(
            task_id="t-1", status="success",
            before_integrity="FATAL", after_integrity="HEALTHY",
            error=None, completed_at="2026-02-09T00:00:00",
        )
        with pytest.raises(AttributeError):
            result.status = "failed"

    def test_heal_task_to_dict(self):
        from controller.runtime_self_healer import HealTask
        task = HealTask(
            id="t-1", project_scope=None, action="fix_permissions",
            evidence="fs broken", created_at="2026-02-09T00:00:00",
        )
        d = task.to_dict()
        assert d["type"] == "heal_task"
        assert d["id"] == "t-1"
        assert d["action"] == "fix_permissions"

    def test_heal_result_to_dict(self):
        from controller.runtime_self_healer import HealResult
        result = HealResult(
            task_id="t-1", status="failed",
            before_integrity="FATAL", after_integrity="FATAL",
            error="pip failed", completed_at="2026-02-09T00:00:00",
        )
        d = result.to_dict()
        assert d["type"] == "heal_result"
        assert d["task_id"] == "t-1"
        assert d["error"] == "pip failed"


# ---------------------------------------------------------------------------
# PART 3: Deterministic Classification Rules
# ---------------------------------------------------------------------------

class TestClassifyIssues:
    """classify_issues maps integrity report to correct HealActions."""

    def test_missing_packages_maps_to_install(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealAction
        healer = RuntimeSelfHealer()
        report = _make_report(missing_packages=["aiohttp"], status="FATAL")
        actions = healer.classify_issues(report)
        assert HealAction.INSTALL_PYTHON_PACKAGE in actions

    def test_playwright_not_ok_maps_to_install_playwright(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealAction
        healer = RuntimeSelfHealer()
        report = _make_report(playwright_ok=False, status="FATAL")
        actions = healer.classify_issues(report)
        assert HealAction.INSTALL_PLAYWRIGHT in actions

    def test_fs_permissions_maps_to_fix_permissions(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealAction
        healer = RuntimeSelfHealer()
        report = _make_report(fs_permissions_ok=False, status="FATAL")
        actions = healer.classify_issues(report)
        assert HealAction.FIX_PERMISSIONS in actions

    def test_missing_binaries_maps_to_clear_locks(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealAction
        healer = RuntimeSelfHealer()
        report = _make_report(missing_binaries=["curl"], status="DEGRADED")
        actions = healer.classify_issues(report)
        assert HealAction.CLEAR_STALE_LOCKS in actions

    def test_healthy_maps_to_no_op(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealAction
        healer = RuntimeSelfHealer()
        report = _make_report(status="HEALTHY")
        actions = healer.classify_issues(report)
        assert actions == [HealAction.NO_OP]

    def test_multiple_issues_multiple_actions(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealAction
        healer = RuntimeSelfHealer()
        report = _make_report(
            missing_packages=["aiohttp"],
            playwright_ok=False,
            fs_permissions_ok=False,
            status="FATAL",
        )
        actions = healer.classify_issues(report)
        assert HealAction.INSTALL_PYTHON_PACKAGE in actions
        assert HealAction.INSTALL_PLAYWRIGHT in actions
        assert HealAction.FIX_PERMISSIONS in actions
        assert len(actions) == 3


# ---------------------------------------------------------------------------
# PART 4: Task Creation Correctness
# ---------------------------------------------------------------------------

class TestCreateHealTasks:
    """create_heal_tasks creates correct frozen HealTask objects."""

    def test_creates_tasks_for_each_action(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask
        healer = RuntimeSelfHealer()
        report = _make_report(
            missing_packages=["requests"],
            playwright_ok=False,
            status="FATAL",
        )
        tasks = healer.create_heal_tasks(report)
        assert len(tasks) == 2
        actions = [t.action for t in tasks]
        assert "install_python_package" in actions
        assert "install_playwright" in actions

    def test_no_tasks_when_healthy(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        report = _make_report(status="HEALTHY")
        tasks = healer.create_heal_tasks(report)
        assert tasks == []

    def test_tasks_have_unique_ids(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        report = _make_report(
            missing_packages=["aiohttp"],
            fs_permissions_ok=False,
            status="FATAL",
        )
        tasks = healer.create_heal_tasks(report)
        ids = [t.id for t in tasks]
        assert len(ids) == len(set(ids))

    def test_task_evidence_contains_details(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        report = _make_report(missing_packages=["aiohttp", "pydantic"], status="FATAL")
        tasks = healer.create_heal_tasks(report)
        pkg_task = [t for t in tasks if t.action == "install_python_package"][0]
        assert "aiohttp" in pkg_task.evidence
        assert "pydantic" in pkg_task.evidence

    def test_task_project_scope_is_none(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        report = _make_report(missing_packages=["aiohttp"], status="FATAL")
        tasks = healer.create_heal_tasks(report)
        assert all(t.project_scope is None for t in tasks)


# ---------------------------------------------------------------------------
# PART 5: Execution Paths (Mocked Commands)
# ---------------------------------------------------------------------------

class TestExecuteTask:
    """execute_task runs correct commands for each HealAction."""

    def test_install_packages_success(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealStatus
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-1", project_scope=None, action="install_python_package",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_run_install_packages", return_value=None), \
             patch.object(healer, "_get_current_status", return_value="HEALTHY"):
            result = healer.execute_task(task)
        assert result.status == HealStatus.SUCCESS.value
        assert result.error is None

    def test_install_packages_failure(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealStatus
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-2", project_scope=None, action="install_python_package",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_run_install_packages", return_value="pip error"), \
             patch.object(healer, "_get_current_status", return_value="FATAL"):
            result = healer.execute_task(task)
        assert result.status == HealStatus.FAILED.value
        assert result.error == "pip error"

    def test_install_playwright_success(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealStatus
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-3", project_scope=None, action="install_playwright",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_run_install_playwright", return_value=None), \
             patch.object(healer, "_get_current_status", return_value="HEALTHY"):
            result = healer.execute_task(task)
        assert result.status == HealStatus.SUCCESS.value

    def test_fix_permissions_success(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealStatus
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-4", project_scope=None, action="fix_permissions",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_run_fix_permissions", return_value=None), \
             patch.object(healer, "_get_current_status", return_value="HEALTHY"):
            result = healer.execute_task(task)
        assert result.status == HealStatus.SUCCESS.value

    def test_restart_services_failure(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealStatus
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-5", project_scope=None, action="restart_services",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_run_restart_services", return_value="restart failed"), \
             patch.object(healer, "_get_current_status", return_value="FATAL"):
            result = healer.execute_task(task)
        assert result.status == HealStatus.FAILED.value

    def test_clear_stale_locks_success(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealStatus
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-6", project_scope=None, action="clear_stale_locks",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_run_clear_stale_locks", return_value=None), \
             patch.object(healer, "_get_current_status", return_value="DEGRADED"):
            result = healer.execute_task(task)
        assert result.status == HealStatus.SUCCESS.value

    def test_execute_captures_before_after(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-7", project_scope=None, action="install_python_package",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        statuses = iter(["FATAL", "HEALTHY"])
        with patch.object(healer, "_run_install_packages", return_value=None), \
             patch.object(healer, "_get_current_status", side_effect=statuses):
            result = healer.execute_task(task)
        assert result.before_integrity == "FATAL"
        assert result.after_integrity == "HEALTHY"


# ---------------------------------------------------------------------------
# PART 6: Bounded Attempts
# ---------------------------------------------------------------------------

class TestBoundedAttempts:
    """attempt_heal respects max_attempts limit."""

    def test_stops_when_healthy(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        healthy_report = _make_report(status="HEALTHY")

        with patch.object(healer, "_run_fresh_integrity_check", return_value=healthy_report):
            results = healer.attempt_heal(max_attempts=3)
        assert results == []

    def test_max_three_attempts(self):
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask
        healer = RuntimeSelfHealer()
        fatal_report = _make_report(missing_packages=["x"], status="FATAL")

        call_count = [0]

        def mock_fresh_check():
            call_count[0] += 1
            return fatal_report

        with patch.object(healer, "_run_fresh_integrity_check", side_effect=mock_fresh_check), \
             patch.object(healer, "_append_record", return_value=True), \
             patch.object(healer, "execute_task", return_value=MagicMock(
                 task_id="t", status="failed", before_integrity="FATAL",
                 after_integrity="FATAL", error="failed", completed_at="now",
                 to_dict=lambda: {},
             )), \
             patch.object(healer, "_escalate_to_meta_remediation"):
            results = healer.attempt_heal(max_attempts=3)

        # Should run fresh check: 3 attempts * 2 (before + after) = 6, but
        # first check is pre-tasks, second is post-round
        assert call_count[0] <= 7  # At most 3 pre + 3 post + 1

    def test_custom_max_attempts(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        fatal_report = _make_report(missing_packages=["x"], status="FATAL")

        with patch.object(healer, "_run_fresh_integrity_check", return_value=fatal_report), \
             patch.object(healer, "_append_record", return_value=True), \
             patch.object(healer, "execute_task", return_value=MagicMock(
                 task_id="t", status="failed", before_integrity="FATAL",
                 after_integrity="FATAL", error="failed", completed_at="now",
                 to_dict=lambda: {},
             )), \
             patch.object(healer, "_escalate_to_meta_remediation"):
            results = healer.attempt_heal(max_attempts=1)

        assert len(results) >= 1


# ---------------------------------------------------------------------------
# PART 7: Escalation to Meta-Remediation
# ---------------------------------------------------------------------------

class TestEscalation:
    """Escalation to meta-remediation on repeated failure."""

    def test_escalates_when_healing_fails(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        fatal_report = _make_report(missing_packages=["x"], status="FATAL")

        escalation_called = [False]

        def mock_escalate(results):
            escalation_called[0] = True

        with patch.object(healer, "_run_fresh_integrity_check", return_value=fatal_report), \
             patch.object(healer, "_append_record", return_value=True), \
             patch.object(healer, "execute_task", return_value=MagicMock(
                 task_id="t", status="failed", before_integrity="FATAL",
                 after_integrity="FATAL", error="failed", completed_at="now",
                 to_dict=lambda: {},
             )), \
             patch.object(healer, "_escalate_to_meta_remediation", side_effect=mock_escalate):
            healer.attempt_heal(max_attempts=2)

        assert escalation_called[0] is True

    def test_no_escalation_when_healed(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()

        checks = iter([
            _make_report(missing_packages=["x"], status="FATAL"),
            _make_report(status="HEALTHY"),
        ])

        with patch.object(healer, "_run_fresh_integrity_check", side_effect=checks), \
             patch.object(healer, "_append_record", return_value=True), \
             patch.object(healer, "execute_task", return_value=MagicMock(
                 task_id="t", status="success", before_integrity="FATAL",
                 after_integrity="HEALTHY", error=None, completed_at="now",
                 to_dict=lambda: {},
             )), \
             patch.object(healer, "_escalate_to_meta_remediation") as mock_esc:
            healer.attempt_heal(max_attempts=3)

        mock_esc.assert_not_called()

    def test_escalate_public_api(self):
        """escalate_to_meta_remediation(task, result) calls internal method."""
        from controller.runtime_self_healer import RuntimeSelfHealer, HealTask, HealResult
        healer = RuntimeSelfHealer()
        task = HealTask(
            id="t-1", project_scope=None, action="fix_permissions",
            evidence="test", created_at="2026-02-09T00:00:00",
        )
        result = HealResult(
            task_id="t-1", status="failed",
            before_integrity="FATAL", after_integrity="FATAL",
            error="chown failed", completed_at="2026-02-09T00:00:00",
        )
        with patch.object(healer, "_escalate_to_meta_remediation_single") as mock_esc:
            healer.escalate_to_meta_remediation(task, result)
        mock_esc.assert_called_once_with(task, result)


# ---------------------------------------------------------------------------
# PART 8: Audit Persistence
# ---------------------------------------------------------------------------

class TestAuditPersistence:
    """Append-only audit with fsync."""

    def test_append_record_writes_jsonl(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            audit_path = Path(f.name)

        try:
            with patch("controller.runtime_self_healer.HEAL_AUDIT_PATH", audit_path):
                success = healer._append_record({"action": "test", "status": "ok"})
            assert success is True
            content = audit_path.read_text().strip()
            record = json.loads(content)
            assert record["action"] == "test"
        finally:
            audit_path.unlink(missing_ok=True)

    def test_append_record_returns_false_on_error(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()

        with patch("controller.runtime_self_healer.HEAL_AUDIT_PATH",
                    Path("/nonexistent/dir/file.jsonl")):
            # Should fail gracefully and return False
            success = healer._append_record({"test": True})
        assert success is False

    def test_fail_closed_on_audit_failure(self):
        """If audit write fails, attempt_heal stops (fail-closed)."""
        from controller.runtime_self_healer import RuntimeSelfHealer
        healer = RuntimeSelfHealer()
        fatal_report = _make_report(missing_packages=["x"], status="FATAL")

        with patch.object(healer, "_run_fresh_integrity_check", return_value=fatal_report), \
             patch.object(healer, "_append_record", return_value=False):
            results = healer.attempt_heal(max_attempts=3)
        # Should stop immediately due to audit failure
        assert len(results) == 0

    def test_get_all_records(self):
        from controller.runtime_self_healer import RuntimeSelfHealer
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            f.write('{"a": 1}\n{"b": 2}\n')
            audit_path = Path(f.name)

        try:
            with patch("controller.runtime_self_healer.HEAL_AUDIT_PATH", audit_path):
                records = RuntimeSelfHealer.get_all_records()
            assert len(records) == 2
            assert records[0]["a"] == 1
        finally:
            audit_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# PART 9: Integration with runtime_enforcement
# ---------------------------------------------------------------------------

class TestEnforcementIntegration:
    """Self-healing is auto-triggered in runtime_enforcement before blocking."""

    def test_self_heal_called_when_fatal(self):
        """When status is FATAL, _attempt_self_heal is called."""
        from controller.runtime_enforcement import RuntimeIntegrityEnforcer, ActionType, RuntimeIntegrityBlockedError
        from controller.runtime_integrity_policy import RuntimeIntegrityPolicy, set_policy

        set_policy(RuntimeIntegrityPolicy.BLOCK_CRITICAL)

        with patch.object(
            RuntimeIntegrityEnforcer, "_get_status_fail_closed", return_value="FATAL"
        ), patch.object(
            RuntimeIntegrityEnforcer, "_attempt_self_heal", return_value="FATAL"
        ) as mock_heal, patch.object(
            RuntimeIntegrityEnforcer, "_write_audit"
        ):
            with pytest.raises(RuntimeIntegrityBlockedError):
                RuntimeIntegrityEnforcer.check_or_block(ActionType.CREATE_PROJECT)
            mock_heal.assert_called_once_with("FATAL")

    def test_self_heal_success_unblocks(self):
        """If self-heal restores HEALTHY, action is NOT blocked."""
        from controller.runtime_enforcement import RuntimeIntegrityEnforcer, ActionType
        from controller.runtime_integrity_policy import RuntimeIntegrityPolicy, set_policy

        set_policy(RuntimeIntegrityPolicy.BLOCK_CRITICAL)

        with patch.object(
            RuntimeIntegrityEnforcer, "_get_status_fail_closed", return_value="FATAL"
        ), patch.object(
            RuntimeIntegrityEnforcer, "_attempt_self_heal", return_value="HEALTHY"
        ), patch.object(
            RuntimeIntegrityEnforcer, "_write_audit"
        ):
            # Should NOT raise — self-heal restored HEALTHY
            RuntimeIntegrityEnforcer.check_or_block(ActionType.CREATE_PROJECT)

    def test_self_heal_not_called_when_healthy(self):
        """When status is HEALTHY, _attempt_self_heal is NOT called."""
        from controller.runtime_enforcement import RuntimeIntegrityEnforcer, ActionType
        from controller.runtime_integrity_policy import RuntimeIntegrityPolicy, set_policy

        set_policy(RuntimeIntegrityPolicy.BLOCK_CRITICAL)

        with patch.object(
            RuntimeIntegrityEnforcer, "_get_status_fail_closed", return_value="HEALTHY"
        ), patch.object(
            RuntimeIntegrityEnforcer, "_attempt_self_heal"
        ) as mock_heal, patch.object(
            RuntimeIntegrityEnforcer, "_write_audit"
        ):
            RuntimeIntegrityEnforcer.check_or_block(ActionType.RESCUE)
            mock_heal.assert_not_called()

    def test_self_heal_not_called_when_warn_only(self):
        """WARN_ONLY policy does not trigger self-heal."""
        from controller.runtime_enforcement import RuntimeIntegrityEnforcer, ActionType
        from controller.runtime_integrity_policy import RuntimeIntegrityPolicy, set_policy

        set_policy(RuntimeIntegrityPolicy.WARN_ONLY)

        with patch.object(
            RuntimeIntegrityEnforcer, "_get_status_fail_closed", return_value="FATAL"
        ), patch.object(
            RuntimeIntegrityEnforcer, "_attempt_self_heal"
        ) as mock_heal, patch.object(
            RuntimeIntegrityEnforcer, "_write_audit"
        ):
            RuntimeIntegrityEnforcer.check_or_block(ActionType.RESCUE)
            mock_heal.assert_not_called()


# ---------------------------------------------------------------------------
# PART 10: Telegram /runtime_heal Output
# ---------------------------------------------------------------------------

class TestTelegramRuntimeHeal:
    """Telegram /runtime_heal command formatting."""

    def test_runtime_heal_healthy_no_action(self):
        """When platform is HEALTHY, reports no healing needed."""
        import asyncio
        from telegram_bot_v2.bot import runtime_heal_command, UserRole

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_update.effective_user.id = 123

        healthy_report = _make_report(status="HEALTHY")

        loop = asyncio.new_event_loop()
        try:
            with patch(
                "telegram_bot_v2.bot.get_user_role",
                return_value=UserRole.OWNER,
            ), patch(
                "controller.runtime_integrity.run_integrity_check",
                return_value=healthy_report,
            ), patch(
                "telegram_bot_v2.bot.safe_reply_text",
                new_callable=AsyncMock,
            ) as mock_safe:
                loop.run_until_complete(
                    runtime_heal_command(mock_update, MagicMock())
                )
            # Should have called safe_reply_text with "HEALTHY" message
            mock_safe.assert_called_once()
            call_text = mock_safe.call_args[0][1]
            assert "HEALTHY" in call_text
        finally:
            loop.close()

    def test_runtime_heal_runs_healing(self):
        """When platform is FATAL, runs healing and shows results."""
        import asyncio
        from controller.runtime_self_healer import HealAction, HealStatus, HealResult
        from telegram_bot_v2.bot import runtime_heal_command, UserRole

        mock_update = MagicMock()
        mock_update.message.reply_text = AsyncMock()
        mock_update.effective_user.id = 123

        fatal_report = _make_report(missing_packages=["aiohttp"], status="FATAL")
        healthy_report = _make_report(status="HEALTHY")

        mock_healer = MagicMock()
        mock_healer.classify_issues.return_value = [HealAction.INSTALL_PYTHON_PACKAGE]
        mock_healer.attempt_heal.return_value = [
            HealResult(
                task_id="t-1", status=HealStatus.SUCCESS.value,
                before_integrity="FATAL", after_integrity="HEALTHY",
                error=None, completed_at="2026-02-09T00:00:00",
            ),
        ]

        loop = asyncio.new_event_loop()
        try:
            with patch(
                "telegram_bot_v2.bot.get_user_role",
                return_value=UserRole.OWNER,
            ), patch(
                "controller.runtime_integrity.run_integrity_check",
                side_effect=[fatal_report, healthy_report],
            ), patch(
                "controller.runtime_self_healer.get_runtime_self_healer",
                return_value=mock_healer,
            ), patch(
                "telegram_bot_v2.bot.safe_reply_text",
                new_callable=AsyncMock,
            ) as mock_safe:
                loop.run_until_complete(
                    runtime_heal_command(mock_update, MagicMock())
                )
            # Should show results (reply_text for "Running..." and action list)
            assert mock_update.message.reply_text.call_count >= 2
        finally:
            loop.close()
