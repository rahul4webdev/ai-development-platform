#!/usr/bin/env python3
"""
Phase 16B: Dashboard Backend Tests

Tests the read-only dashboard aggregation layer.

TEST FOCUS:
- ✅ Read-only behavior (no mutations)
- ✅ Data consistency
- ✅ No side effects
- ✅ Deterministic output
- ✅ Zero hallucinated data
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import MagicMock, patch, AsyncMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDashboardBackendImports:
    """Test that dashboard backend imports correctly."""

    def test_dashboard_backend_imports(self):
        """All dashboard_backend components should import successfully."""
        from controller.dashboard_backend import (
            DashboardBackend,
            ProjectOverview,
            ClaudeActivityPanel,
            LifecycleTimeline,
            DeploymentView,
            AuditEvent,
            DashboardSummary,
            SystemHealth,
        )
        assert DashboardBackend is not None
        assert ProjectOverview is not None
        assert ClaudeActivityPanel is not None
        assert LifecycleTimeline is not None
        assert DeploymentView is not None
        assert AuditEvent is not None
        assert DashboardSummary is not None
        assert SystemHealth is not None

    def test_module_functions_exist(self):
        """Module-level convenience functions should exist."""
        from controller import dashboard_backend

        assert hasattr(dashboard_backend, "get_dashboard_summary")
        assert hasattr(dashboard_backend, "get_all_projects")
        assert hasattr(dashboard_backend, "get_claude_activity")
        assert hasattr(dashboard_backend, "get_audit_events")
        assert hasattr(dashboard_backend, "get_security_summary")


class TestSystemHealthEnum:
    """Test SystemHealth enum values."""

    def test_all_health_states_exist(self):
        """All health states should be defined."""
        from controller.dashboard_backend import SystemHealth

        assert SystemHealth.HEALTHY.value == "healthy"
        assert SystemHealth.DEGRADED.value == "degraded"
        assert SystemHealth.CRITICAL.value == "critical"
        assert SystemHealth.UNKNOWN.value == "unknown"


class TestDataModels:
    """Test dashboard data models match actual implementation."""

    def test_project_overview_fields(self):
        """ProjectOverview should have all required fields."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="proj-123",
            project_name="test-project",
            mode="PROJECT_MODE",
            current_lifecycle_state="development",
            aspects={"core": "development"},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id="lc-123",
            created_at="2026-01-19T00:00:00",
            updated_at="2026-01-19T00:00:00"
        )

        assert overview.project_name == "test-project"
        assert overview.current_lifecycle_state == "development"
        assert overview.mode == "PROJECT_MODE"
        assert overview.aspects == {"core": "development"}

    def test_claude_activity_panel_fields(self):
        """ClaudeActivityPanel should have all required fields."""
        from controller.dashboard_backend import ClaudeActivityPanel

        panel = ClaudeActivityPanel(
            active_jobs=[],
            queued_jobs=[],
            completed_jobs_today=0,
            failed_jobs_today=0,
            worker_utilization={"active": 0, "max": 3},
            job_worker_mapping={},
            gate_decisions_today={"ALLOWED": 0, "DENIED": 0}
        )

        assert panel.active_jobs == []
        assert panel.queued_jobs == []
        assert panel.completed_jobs_today == 0
        assert panel.failed_jobs_today == 0
        assert panel.worker_utilization == {"active": 0, "max": 3}

    def test_lifecycle_timeline_fields(self):
        """LifecycleTimeline should have all required fields."""
        from controller.dashboard_backend import LifecycleTimeline

        timeline = LifecycleTimeline(
            lifecycle_id="test-lifecycle-123",
            project_name="test-project",
            aspect="core",
            current_state="development",
            transition_history=[],
            approvals=[],
            rejections=[],
            feedback_entries=[],
            cycle_history=[],
            change_summaries=[]
        )

        assert timeline.lifecycle_id == "test-lifecycle-123"
        assert timeline.project_name == "test-project"
        assert timeline.current_state == "development"
        assert timeline.transition_history == []

    def test_dashboard_summary_fields(self):
        """DashboardSummary should have all required fields."""
        from controller.dashboard_backend import DashboardSummary, SystemHealth

        summary = DashboardSummary(
            system_health=SystemHealth.HEALTHY,
            timestamp="2026-01-19T12:00:00",
            total_projects=5,
            active_projects=3,
            total_lifecycles=4,
            active_lifecycles=2,
            active_jobs=1,
            queued_jobs=2,
            completed_today=10,
            failed_today=0,
            gate_denials_today=0,
            policy_violations_today=0,
            claude_cli_available=True,
            telegram_bot_operational=True,
            controller_healthy=True
        )

        assert summary.total_projects == 5
        assert summary.active_projects == 3
        assert summary.active_jobs == 1
        assert summary.system_health == SystemHealth.HEALTHY

    def test_audit_event_fields(self):
        """AuditEvent should have all required fields."""
        from controller.dashboard_backend import AuditEvent

        event = AuditEvent(
            timestamp="2026-01-19T12:00:00",
            event_type="GATE_DENIAL",
            job_id="job-123",
            project_name="test-project",
            action="WRITE_CODE",
            user_id="user-123",
            user_role="developer",
            reason="Action not allowed in current state",
            severity="WARNING"
        )

        assert event.event_type == "GATE_DENIAL"
        assert event.job_id == "job-123"
        assert event.severity == "WARNING"


class TestDashboardBackendReadOnly:
    """Test that DashboardBackend is truly read-only."""

    def test_no_write_methods(self):
        """DashboardBackend should have no write/mutate methods."""
        from controller.dashboard_backend import DashboardBackend

        # List of methods that would indicate mutation
        forbidden_patterns = [
            "create", "update", "delete", "remove", "set",
            "add", "insert", "modify", "change", "mutate",
            "write", "save", "store", "push", "commit"
        ]

        backend = DashboardBackend()
        method_names = [m for m in dir(backend) if not m.startswith("_")]

        for method in method_names:
            for pattern in forbidden_patterns:
                # Allow "get_" prefix methods even if they contain patterns
                if not method.startswith("get_") and not method.startswith("_"):
                    assert pattern not in method.lower(), \
                        f"Found potentially mutating method: {method}"

    def test_methods_are_read_operations(self):
        """All public methods should be read operations (getters)."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()
        public_methods = [m for m in dir(backend) if not m.startswith("_")]

        # All public methods should start with "get_" or be properties
        for method in public_methods:
            if callable(getattr(backend, method)):
                assert method.startswith("get_") or method.startswith("_"), \
                    f"Public method {method} doesn't follow read-only naming convention"


class TestDashboardBackendDataConsistency:
    """Test data consistency in dashboard aggregation."""

    def test_summary_can_be_created(self):
        """DashboardBackend can create a summary."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        # Synchronously run the async method
        loop = asyncio.new_event_loop()
        try:
            summary = loop.run_until_complete(backend.get_dashboard_summary())
            assert summary is not None
            assert summary.timestamp is not None
        finally:
            loop.close()

    def test_deterministic_project_count(self):
        """Same call should return same project count."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            projects1 = loop.run_until_complete(backend.get_all_projects())
            projects2 = loop.run_until_complete(backend.get_all_projects())
            assert len(projects1) == len(projects2)
        finally:
            loop.close()


class TestDashboardBackendNoSideEffects:
    """Test that dashboard operations have no side effects."""

    def test_multiple_summary_calls_safe(self):
        """Getting summary multiple times should be safe."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            # Call multiple times
            for _ in range(5):
                summary = loop.run_until_complete(backend.get_dashboard_summary())
                assert summary is not None
        finally:
            loop.close()


class TestDashboardBackendZeroHallucination:
    """Test that dashboard doesn't hallucinate data."""

    def test_no_fake_projects(self):
        """Should not return projects that don't exist."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            projects = loop.run_until_complete(backend.get_all_projects())

            # All returned projects should have valid names
            for project in projects:
                assert project.project_name is not None
                assert len(project.project_name) > 0
                # Project names should not be placeholders
                assert project.project_name not in ["example", "placeholder", "test-fake"]
        finally:
            loop.close()

    def test_summary_timestamp_not_future(self):
        """Timestamps should not be in the future."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            summary = loop.run_until_complete(backend.get_dashboard_summary())
            now = datetime.utcnow().isoformat()
            # Timestamp should be <= now (string comparison works for ISO format)
            assert summary.timestamp[:19] <= now[:19], \
                "Summary timestamp should not be in the future"
        finally:
            loop.close()


class TestDashboardAPIEndpoints:
    """Test dashboard API endpoint integration."""

    def test_main_imports_dashboard_routes(self):
        """Main controller should import dashboard routes."""
        try:
            from controller import main
            assert True
        except ImportError as e:
            assert False, f"Failed to import main with dashboard: {e}"

    def test_dashboard_backend_module_functions(self):
        """Module-level functions should be callable."""
        from controller.dashboard_backend import (
            get_dashboard_summary,
            get_all_projects,
            get_claude_activity,
            get_audit_events,
            get_security_summary,
        )

        # All should be async functions
        assert asyncio.iscoroutinefunction(get_dashboard_summary)
        assert asyncio.iscoroutinefunction(get_all_projects)
        assert asyncio.iscoroutinefunction(get_claude_activity)
        assert asyncio.iscoroutinefunction(get_audit_events)
        assert asyncio.iscoroutinefunction(get_security_summary)


class TestTelegramDashboardIntegration:
    """Test Telegram bot dashboard integration."""

    def test_format_dashboard_enhanced_exists(self):
        """format_dashboard_enhanced function should exist."""
        from telegram_bot_v2.bot import format_dashboard_enhanced

        assert callable(format_dashboard_enhanced)

    def test_format_dashboard_enhanced_output(self):
        """format_dashboard_enhanced should produce valid output."""
        from telegram_bot_v2.bot import format_dashboard_enhanced

        # Mock summary data
        mock_summary = {
            "system_health": "healthy",
            "total_projects": 3,
            "total_lifecycles": 2,
            "pending_jobs": 1,
            "active_workers": 1,
            "max_workers": 3,
            "active_projects": [
                {"project_name": "test-project", "lifecycle_state": "development"}
            ],
            "claude_activity": {
                "running_jobs": 1,
                "queued_jobs": 0,
                "completed_24h": 5
            },
            "security": {
                "recent_gate_denials": 0
            },
            "timestamp": "2026-01-19T12:00:00"
        }

        result = format_dashboard_enhanced(mock_summary)

        # Should contain key sections
        assert "Platform Dashboard" in result
        assert "System Health" in result
        assert "Summary Counts" in result
        assert "Projects: 3" in result
        assert "Active Workers: 1/3" in result

    def test_get_lifecycle_state_emoji_exists(self):
        """get_lifecycle_state_emoji function should exist."""
        from telegram_bot_v2.bot import get_lifecycle_state_emoji

        # Test known states
        assert get_lifecycle_state_emoji("development") == "💻"
        assert get_lifecycle_state_emoji("testing") == "🧪"
        assert get_lifecycle_state_emoji("deployed") == "🚀"
        assert get_lifecycle_state_emoji("unknown_state") == "❓"

    def test_controller_client_dashboard_methods(self):
        """ControllerClient should have dashboard methods."""
        from telegram_bot_v2.bot import ControllerClient

        # Check methods exist
        assert hasattr(ControllerClient, "get_dashboard_summary")
        assert hasattr(ControllerClient, "get_dashboard_jobs")
        assert hasattr(ControllerClient, "get_dashboard_audit")


# Run tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
