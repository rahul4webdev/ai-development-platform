"""
Phase 23: Deep Functional E2E Enforcement Tests

Tests for:
- PART 1: DeepE2EVerifier checks (E2ELevel, DeepE2EResult, verifier.run)
- PART 2: ExecutionGate integration (deep_e2e_required flag)
- PART 6: Registry/Dashboard field propagation
- Edge cases and backward compatibility
"""

import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# PART 1: E2ELevel enum
# ---------------------------------------------------------------------------

class TestE2ELevelEnum:
    """E2ELevel enum values and behavior."""

    def test_enum_values(self):
        from controller.deep_e2e_verifier import E2ELevel
        assert E2ELevel.BASIC.value == "basic"
        assert E2ELevel.FUNCTIONAL.value == "functional"

    def test_enum_is_string(self):
        from controller.deep_e2e_verifier import E2ELevel
        assert isinstance(E2ELevel.BASIC, str)
        assert isinstance(E2ELevel.FUNCTIONAL, str)

    def test_all_check_types_defined(self):
        from controller.deep_e2e_verifier import E2ECheckType
        expected = [
            "api_schema_valid", "api_business_data", "cors_browser_origin",
            "frontend_pages", "auth_login_e2e", "commit_sha_match"
        ]
        actual = [ct.value for ct in E2ECheckType]
        for exp in expected:
            assert exp in actual, f"Missing check type: {exp}"


# ---------------------------------------------------------------------------
# PART 1: DeepE2EResult frozen dataclass
# ---------------------------------------------------------------------------

class TestDeepE2EResult:
    """DeepE2EResult is frozen (immutable) and has correct fields."""

    def test_result_is_frozen(self):
        from controller.deep_e2e_verifier import DeepE2EResult
        result = DeepE2EResult(
            project_name="test",
            level="functional",
            passed=True,
            total_checks=5,
            passed_checks=5,
            failed_checks=0,
            checks=(),
            summary="All checks passed",
        )
        with pytest.raises(AttributeError):
            result.passed = False

    def test_result_fields(self):
        from controller.deep_e2e_verifier import DeepE2EResult, E2ECheckResult
        check = E2ECheckResult(
            check_type="api_schema_valid",
            passed=True,
            message="Schema valid",
        )
        result = DeepE2EResult(
            project_name="test-project",
            level="functional",
            passed=True,
            total_checks=1,
            passed_checks=1,
            failed_checks=0,
            checks=(check,),
            summary="1/1 passed",
        )
        assert result.project_name == "test-project"
        assert result.level == "functional"
        assert len(result.checks) == 1
        assert result.checks[0].passed is True
        assert result.checks[0].check_type == "api_schema_valid"

    def test_check_result_is_frozen(self):
        from controller.deep_e2e_verifier import E2ECheckResult
        check = E2ECheckResult(
            check_type="test",
            passed=True,
            message="ok",
        )
        with pytest.raises(AttributeError):
            check.passed = False


# ---------------------------------------------------------------------------
# PART 1: DeepE2EVerifier.run()
# ---------------------------------------------------------------------------

class TestDeepE2EVerifier:
    """DeepE2EVerifier check execution."""

    def test_run_all_checks_pass(self):
        """When all HTTP calls succeed, all checks pass."""
        from controller.deep_e2e_verifier import DeepE2EVerifier, E2ELevel, DeepE2EResult

        async def _run():
            verifier = DeepE2EVerifier()

            last_registered_email = {}

            async def mock_request(method, url, **kwargs):
                if "/openapi.json" in url:
                    schema = {
                        "openapi": "3.0.0",
                        "paths": {
                            "/api/auth/login": {},
                            "/api/auth/register": {},
                            "/api/nutrition/food-log": {},
                        }
                    }
                    return 200, json.dumps(schema), {"content-type": "application/json"}
                elif "/api/auth/register" in url:
                    email = (kwargs.get("json_data") or {}).get("email", "test@x.com")
                    last_registered_email["email"] = email
                    return 201, json.dumps({"email": email, "id": 1, "name": "Test"}), {}
                elif "/api/auth/login" in url:
                    return 200, '{"access_token":"tok123","token_type":"bearer"}', {}
                elif "/api/auth/me" in url:
                    email = last_registered_email.get("email", "test@x.com")
                    return 200, json.dumps({"email": email}), {}
                elif "/api/nutrition/daily-summary" in url:
                    return 200, '{"total_calories":0,"entries_count":0,"date":"2026-01-01"}', {}
                elif "/health" in url and method == "OPTIONS":
                    return 200, "", {"access-control-allow-origin": "https://example.com"}
                elif "/login" in url or "/register" in url:
                    return 200, "<!DOCTYPE html><html><body>App</body></html>", {"content-type": "text/html"}
                return 200, '{"status":"ok"}', {"content-type": "text/html"}

            with patch.object(verifier, '_make_request', side_effect=mock_request):
                with patch(
                    "controller.deployment_validator.check_deploy_status",
                    new_callable=AsyncMock,
                    return_value=("deployed", "abc123", "abc123")
                ):
                    result = await verifier.run(
                        "test-project",
                        {"api": "https://api.example.com", "frontend": "https://example.com"},
                        github_repo="owner/repo",
                        level=E2ELevel.FUNCTIONAL,
                    )

            assert isinstance(result, DeepE2EResult)
            assert result.passed is True
            assert result.failed_checks == 0
            assert result.total_checks >= 5  # schema, auth, business, cors, frontend, commit
            assert result.duration_ms is not None

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_run_api_schema_fails(self):
        """When OpenAPI schema returns 404, the check fails."""
        from controller.deep_e2e_verifier import DeepE2EVerifier, E2ELevel

        async def _run():
            verifier = DeepE2EVerifier()

            async def mock_request(method, url, **kwargs):
                if "/openapi.json" in url:
                    return 404, "Not Found", {}
                elif "/api/auth/register" in url:
                    return 201, '{"email":"t@x.com","id":1}', {}
                elif "/api/auth/login" in url:
                    return 200, '{"access_token":"tok","token_type":"bearer"}', {}
                elif "/api/auth/me" in url:
                    return 200, '{"email":"t@x.com"}', {}
                elif "/api/nutrition/daily-summary" in url:
                    return 200, '{"total_calories":0,"entries_count":0}', {}
                return 200, '{"ok":true}', {}

            with patch.object(verifier, '_make_request', side_effect=mock_request):
                result = await verifier.run(
                    "test-project",
                    {"api": "https://api.example.com"},
                    level=E2ELevel.FUNCTIONAL,
                )

            assert result.passed is False
            assert result.failed_checks >= 1
            # Find the schema check
            schema_check = [c for c in result.checks if c.check_type == "api_schema_valid"]
            assert len(schema_check) == 1
            assert schema_check[0].passed is False

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_run_basic_level_fewer_checks(self):
        """BASIC level runs fewer checks than FUNCTIONAL."""
        from controller.deep_e2e_verifier import DeepE2EVerifier, E2ELevel

        async def _run():
            verifier = DeepE2EVerifier()

            async def mock_request(method, url, **kwargs):
                if "/openapi.json" in url:
                    return 200, '{"openapi":"3.0.0","paths":{"/test":{}}}', {}
                return 200, "<!DOCTYPE html><html></html>", {"content-type": "text/html"}

            with patch.object(verifier, '_make_request', side_effect=mock_request):
                basic_result = await verifier.run(
                    "test", {"api": "https://api.x", "frontend": "https://x"},
                    level=E2ELevel.BASIC,
                )
                functional_result = await verifier.run(
                    "test", {"api": "https://api.x", "frontend": "https://x"},
                    level=E2ELevel.FUNCTIONAL,
                )

            # Basic should have fewer checks
            assert basic_result.total_checks < functional_result.total_checks

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_run_no_urls_returns_empty_result(self):
        """Empty URLs dict produces zero checks and fails (no checks = not passed)."""
        from controller.deep_e2e_verifier import DeepE2EVerifier, E2ELevel

        async def _run():
            verifier = DeepE2EVerifier()
            result = await verifier.run("test", {}, level=E2ELevel.FUNCTIONAL)
            assert result.total_checks == 0
            assert result.passed is False  # No checks = not passed

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# PART 2: ExecutionGate integration
# ---------------------------------------------------------------------------

class TestExecutionGateIntegration:
    """ExecutionGate deep_e2e_required flag for DEPLOY_TEST."""

    def test_deploy_test_sets_deep_e2e_required(self, tmp_path):
        """DEPLOY_TEST action should set deep_e2e_required=True."""
        import os
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        gate = ExecutionGate()
        request = ExecutionRequest(
            job_id="test-job",
            project_name="test-project",
            aspect="core",
            lifecycle_id="lc-1",
            lifecycle_state=LifecycleState.READY_FOR_PRODUCTION.value,
            requested_action=ExecutionAction.DEPLOY_TEST.value,
            requesting_user_id="user-1",
            requesting_user_role=UserRole.OWNER.value,
            workspace_path=str(workspace),
            task_description="Deploy to testing",
        )

        decision = gate.evaluate(request)
        assert decision.allowed is True
        assert decision.deep_e2e_required is True

    def test_read_code_does_not_set_deep_e2e_required(self, tmp_path):
        """READ_CODE action should NOT set deep_e2e_required."""
        import os
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        gate = ExecutionGate()
        request = ExecutionRequest(
            job_id="test-job",
            project_name="test-project",
            aspect="core",
            lifecycle_id="lc-1",
            lifecycle_state=LifecycleState.DEVELOPMENT.value,
            requested_action=ExecutionAction.READ_CODE.value,
            requesting_user_id="user-1",
            requesting_user_role=UserRole.DEVELOPER.value,
            workspace_path=str(workspace),
            task_description="Read code",
        )

        decision = gate.evaluate(request)
        assert decision.allowed is True
        assert decision.deep_e2e_required is False

    def test_gate_decision_to_dict_includes_deep_e2e(self, tmp_path):
        """GateDecision.to_dict() includes all deep E2E fields."""
        import os
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole, REQUIRED_GOVERNANCE_DOCS,
        )

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")

        gate = ExecutionGate()
        request = ExecutionRequest(
            job_id="test-job",
            project_name="test-project",
            aspect="core",
            lifecycle_id="lc-1",
            lifecycle_state=LifecycleState.READY_FOR_PRODUCTION.value,
            requested_action=ExecutionAction.DEPLOY_TEST.value,
            requesting_user_id="user-1",
            requesting_user_role=UserRole.OWNER.value,
            workspace_path=str(workspace),
            task_description="Deploy to testing",
        )

        decision = gate.evaluate(request)
        d = decision.to_dict()

        assert "deep_e2e_required" in d
        assert "deep_e2e_checked" in d
        assert "deep_e2e_passed" in d
        assert "deep_e2e_block_reason" in d
        assert d["deep_e2e_required"] is True
        assert d["deep_e2e_checked"] is False
        assert d["deep_e2e_passed"] is False


# ---------------------------------------------------------------------------
# PART 6: Registry & Dashboard fields
# ---------------------------------------------------------------------------

class TestRegistryDashboardFields:
    """Registry metadata and dashboard field propagation."""

    def test_registry_metadata_stores_deep_e2e(self):
        """Project metadata dict accepts deep E2E fields."""
        from controller.project_registry import Project

        project = Project(
            project_id="test-id",
            name="test-project",
            description="Test",
            created_by="user-1",
            created_at="2026-01-01T00:00:00",
            updated_at="2026-01-01T00:00:00",
            metadata={
                "deep_e2e_level": "functional",
                "deep_e2e_passed": True,
                "deep_e2e_last_run": "2026-01-01T12:00:00",
                "deep_e2e_summary": "6/6 checks passed",
                "deep_e2e_checks_total": 6,
                "deep_e2e_checks_passed": 6,
            },
        )

        d = project.to_dict()
        assert d["metadata"]["deep_e2e_level"] == "functional"
        assert d["metadata"]["deep_e2e_passed"] is True
        assert d["metadata"]["deep_e2e_checks_total"] == 6

    def test_backward_compat_no_deep_e2e_fields(self):
        """Old projects without deep E2E metadata work fine."""
        from controller.project_registry import Project

        project = Project(
            project_id="old-id",
            name="old-project",
            description="Old project",
            created_by="user-1",
            created_at="2025-01-01T00:00:00",
            updated_at="2025-01-01T00:00:00",
        )

        d = project.to_dict()
        assert d["metadata"] == {}
        assert d["metadata"].get("deep_e2e_level") is None
        assert d["metadata"].get("deep_e2e_passed") is None

    def test_dashboard_project_overview_has_deep_e2e_fields(self):
        """ProjectOverview includes Phase 23 fields."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="test-id",
            project_name="test",
            mode="project_mode",
            current_lifecycle_state="development",
            aspects={"api": "development"},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id="lc-1",
            created_at="2026-01-01",
            updated_at="2026-01-01",
            deep_e2e_level="functional",
            deep_e2e_passed=True,
            deep_e2e_last_run="2026-01-01T12:00:00",
        )

        d = overview.to_dict()
        assert d["deep_e2e_level"] == "functional"
        assert d["deep_e2e_passed"] is True
        assert d["deep_e2e_last_run"] == "2026-01-01T12:00:00"

    def test_dashboard_overview_defaults_none(self):
        """ProjectOverview deep E2E fields default to None."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="test-id",
            project_name="test",
            mode="project_mode",
            current_lifecycle_state="development",
            aspects={},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id=None,
            created_at=None,
            updated_at=None,
        )

        d = overview.to_dict()
        assert d["deep_e2e_level"] is None
        assert d["deep_e2e_passed"] is None
        assert d["deep_e2e_last_run"] is None
