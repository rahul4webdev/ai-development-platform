"""
Phase 25: Adaptive E2E Test Governance & Reliability Layer Tests

Tests:
- PART 1: TestScope enum and TestContract derivation
- PART 2: TestGovernor (verify, confidence score, task prompt)
- PART 3: TestGapStore (append-only JSONL, gap confirmation)
- PART 4: ExecutionGate Step 15 (test governance blocking)
- PART 5: Dashboard confidence_score field
- PART 6: BlockReason enum
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# PART 1: TestScope Enum and Contract Derivation
# ---------------------------------------------------------------------------

class TestTestScopeEnum:
    """TestScope enum values and completeness."""

    def test_enum_values_exist(self):
        from controller.test_contract import TestScope

        assert TestScope.API == "api"
        assert TestScope.FRONTEND == "frontend"
        assert TestScope.ADMIN == "admin"
        assert TestScope.INTEGRATION == "integration"
        assert TestScope.DEPLOYMENT == "deployment"

    def test_all_scopes_defined(self):
        from controller.test_contract import TestScope

        assert len(TestScope) == 5


class TestContractDerivation:
    """derive_contract_from_chd deterministic rules."""

    def test_chd_with_api_domain(self):
        """CHD with api: section → API scope."""
        from controller.test_contract import derive_contract_from_chd, TestScope

        chd = """
testing_env:
  api:
    domain: healthapi.example.com
"""
        contract = derive_contract_from_chd("test-project", chd_text=chd)
        assert TestScope.API.value in contract.required_scopes
        assert TestScope.DEPLOYMENT.value in contract.required_scopes

    def test_chd_with_frontend_domain(self):
        """CHD with frontend_web: section → FRONTEND scope."""
        from controller.test_contract import derive_contract_from_chd, TestScope

        chd = """
testing_env:
  frontend_web:
    domain: health.example.com
"""
        contract = derive_contract_from_chd("test-project", chd_text=chd)
        assert TestScope.FRONTEND.value in contract.required_scopes
        assert contract.requires_playwright is True

    def test_empty_chd_minimal_contract(self):
        """Empty CHD → minimal contract with no scopes."""
        from controller.test_contract import derive_contract_from_chd

        contract = derive_contract_from_chd("test-project", chd_text="", domains={})
        assert len(contract.required_scopes) == 0
        assert contract.smoke_required is True
        assert contract.e2e_required is True

    def test_domains_dict_fallback(self):
        """Domains dict provides scopes when CHD is empty."""
        from controller.test_contract import derive_contract_from_chd, TestScope

        contract = derive_contract_from_chd(
            "test-project",
            chd_text="",
            domains={"api": "https://api.example.com", "frontend": "https://app.example.com"},
        )
        assert TestScope.API.value in contract.required_scopes
        assert TestScope.FRONTEND.value in contract.required_scopes
        assert TestScope.DEPLOYMENT.value in contract.required_scopes

    def test_contract_is_frozen(self):
        """TestContract is frozen (immutable)."""
        from controller.test_contract import derive_contract_from_chd

        contract = derive_contract_from_chd("test-project", chd_text="")
        with pytest.raises(AttributeError):
            contract.project = "changed"


# ---------------------------------------------------------------------------
# PART 2: TestGovernor
# ---------------------------------------------------------------------------

class TestTestGovernor:
    """TestGovernor verify, confidence, and task creation."""

    def test_verify_returns_missing_items(self):
        """verify_test_completeness returns missing items when tests absent."""
        from controller.test_governor import TestGovernor

        governor = TestGovernor()

        # Mock registry to return a project with API scope required but no tests
        mock_project = MagicMock()
        mock_project.requirements_raw = """
testing_env:
  api:
    domain: api.example.com
"""
        mock_project.domains = {"api": "https://api.example.com"}
        mock_project.tech_stack = {}
        mock_project.metadata = {"test_governance": {}}

        mock_registry = MagicMock()
        mock_registry.get_project.return_value = mock_project

        with patch("controller.project_registry.get_registry", return_value=mock_registry):
            missing = governor.verify_test_completeness("test-project")

        assert "MISSING_API_E2E" in missing
        assert "MISSING_SMOKE_TESTS" in missing

    def test_verify_returns_empty_when_complete(self):
        """verify_test_completeness returns empty list when all tests present."""
        from controller.test_governor import TestGovernor

        governor = TestGovernor()

        mock_project = MagicMock()
        mock_project.requirements_raw = """
testing_env:
  api:
    domain: api.example.com
"""
        mock_project.domains = {"api": "https://api.example.com"}
        mock_project.tech_stack = {}
        mock_project.metadata = {
            "test_governance": {
                "has_api_e2e": True,
                "has_frontend_e2e": True,
                "has_playwright": True,
                "has_smoke": True,
            }
        }

        mock_registry = MagicMock()
        mock_registry.get_project.return_value = mock_project

        with patch("controller.project_registry.get_registry", return_value=mock_registry):
            missing = governor.verify_test_completeness("test-project")

        assert missing == []

    def test_compute_confidence_base(self):
        """Base confidence score is 0.5 when no tests exist."""
        from controller.test_governor import TestGovernor

        mock_project = MagicMock()
        mock_project.metadata = {}

        mock_registry = MagicMock()
        mock_registry.get_project.return_value = mock_project

        with patch("controller.project_registry.get_registry", return_value=mock_registry):
            score = TestGovernor.compute_confidence_score("test-project")

        # Base 0.5 + 0.1 for no micro_remediation issues
        assert score == 0.6

    def test_compute_confidence_all_tests(self):
        """Confidence score is 1.0 when all tests present + no micro-remediation."""
        from controller.test_governor import TestGovernor

        mock_project = MagicMock()
        mock_project.metadata = {
            "test_governance": {
                "has_api_e2e": True,
                "has_frontend_e2e": True,
                "has_playwright": True,
                "has_smoke": True,
            },
            "micro_remediation": {},
        }

        mock_registry = MagicMock()
        mock_registry.get_project.return_value = mock_project

        with patch("controller.project_registry.get_registry", return_value=mock_registry):
            score = TestGovernor.compute_confidence_score("test-project")

        assert score == 1.0

    def test_create_test_task_contains_project_name(self):
        """create_test_task returns prompt containing project name."""
        from controller.test_governor import TestGovernor

        governor = TestGovernor()
        prompt = governor.create_test_task("health-tracker", ["MISSING_API_E2E"])

        assert "health-tracker" in prompt
        assert "MISSING_API_E2E" in prompt
        assert "TEST GOVERNANCE JOB" in prompt


# ---------------------------------------------------------------------------
# PART 3: TestGapStore
# ---------------------------------------------------------------------------

class TestTestGapStore:
    """TestGapStore append-only JSONL tracking and confirmation."""

    def test_record_first_gap_not_confirmed(self):
        """First occurrence of a gap → not confirmed (returns False)."""
        from controller.test_gap_store import TestGapStore

        with tempfile.TemporaryDirectory() as tmp:
            gap_file = Path(tmp) / "test_gaps.jsonl"
            store = TestGapStore(gap_file=gap_file)

            confirmed = store.record_gap("test-project", "MISSING_CORS_TEST")
            assert confirmed is False

    def test_record_second_gap_confirmed(self):
        """Second occurrence of same gap → confirmed (returns True)."""
        from controller.test_gap_store import TestGapStore

        with tempfile.TemporaryDirectory() as tmp:
            gap_file = Path(tmp) / "test_gaps.jsonl"
            store = TestGapStore(gap_file=gap_file)

            store.record_gap("test-project", "MISSING_CORS_TEST")
            confirmed = store.record_gap("test-project", "MISSING_CORS_TEST")
            assert confirmed is True

    def test_get_gaps_returns_all_for_project(self):
        """get_gaps returns all latest gaps for a project."""
        from controller.test_gap_store import TestGapStore

        with tempfile.TemporaryDirectory() as tmp:
            gap_file = Path(tmp) / "test_gaps.jsonl"
            store = TestGapStore(gap_file=gap_file)

            store.record_gap("proj-a", "MISSING_CORS_TEST")
            store.record_gap("proj-a", "MISSING_ROUTE_TESTS")
            store.record_gap("proj-b", "MISSING_CORS_TEST")

            gaps_a = store.get_gaps("proj-a")
            gaps_b = store.get_gaps("proj-b")

            assert len(gaps_a) == 2
            assert len(gaps_b) == 1

    def test_gap_types_enum(self):
        """GapType enum has all expected values."""
        from controller.test_gap_store import GapType

        assert GapType.MISSING_CORS_TEST == "MISSING_CORS_TEST"
        assert GapType.MISSING_ROUTE_TESTS == "MISSING_ROUTE_TESTS"
        assert GapType.MISSING_API_SEED_TESTS == "MISSING_API_SEED_TESTS"
        assert GapType.MISSING_DEPLOYMENT_SMOKE == "MISSING_DEPLOYMENT_SMOKE"
        assert GapType.MISSING_API_E2E == "MISSING_API_E2E"
        assert GapType.MISSING_FRONTEND_E2E == "MISSING_FRONTEND_E2E"
        assert GapType.MISSING_PLAYWRIGHT_CI == "MISSING_PLAYWRIGHT_CI"
        assert len(GapType) == 7


# ---------------------------------------------------------------------------
# PART 4: ExecutionGate Step 15
# ---------------------------------------------------------------------------

class TestExecutionGateStep15:
    """ExecutionGate Step 15 — test governance blocking."""

    def _make_workspace(self, tmp_path):
        from controller.execution_gate import REQUIRED_GOVERNANCE_DOCS
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for doc in REQUIRED_GOVERNANCE_DOCS:
            (workspace / doc).write_text(f"# {doc}")
        return workspace

    def test_blocked_when_tests_missing(self, tmp_path):
        """DEPLOY_TEST is blocked when verify_test_completeness returns items."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        mock_governor = MagicMock()
        mock_governor.verify_test_completeness.return_value = ["MISSING_API_E2E"]

        with patch("controller.test_governor.get_test_governor", return_value=mock_governor):
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

        assert decision.allowed is False
        assert decision.test_governance_blocks is True
        assert "INSUFFICIENT_TESTING" in (decision.denied_reason or "")

    def test_allowed_when_tests_complete(self, tmp_path):
        """DEPLOY_TEST is allowed when verify_test_completeness returns empty."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        mock_governor = MagicMock()
        mock_governor.verify_test_completeness.return_value = []

        with patch("controller.test_governor.get_test_governor", return_value=mock_governor):
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
        assert decision.test_governance_blocks is False

    def test_allowed_when_no_governor(self, tmp_path):
        """DEPLOY_TEST is allowed when test_governor import fails (fail-open)."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        with patch("controller.test_governor.get_test_governor", side_effect=ImportError("not available")):
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

    def test_read_code_unaffected(self, tmp_path):
        """READ_CODE action is not affected by test governance."""
        os.environ["EXECUTION_AUDIT_LOG"] = str(tmp_path / "audit.log")

        from controller.execution_gate import (
            ExecutionGate, ExecutionRequest, LifecycleState,
            ExecutionAction, UserRole,
        )

        workspace = self._make_workspace(tmp_path)

        gate = ExecutionGate()
        request = ExecutionRequest(
            job_id="test-job",
            project_name="test-project",
            aspect="core",
            lifecycle_id="lc-1",
            lifecycle_state=LifecycleState.DEVELOPMENT.value,
            requested_action=ExecutionAction.READ_CODE.value,
            requesting_user_id="user-1",
            requesting_user_role=UserRole.OWNER.value,
            workspace_path=str(workspace),
            task_description="Read project code",
        )
        decision = gate.evaluate(request)

        assert decision.allowed is True
        assert decision.test_governance_blocks is False


# ---------------------------------------------------------------------------
# PART 5: Dashboard Confidence Score
# ---------------------------------------------------------------------------

class TestDashboardConfidence:
    """Dashboard ProjectOverview confidence_score field."""

    def test_overview_has_confidence_score(self):
        """ProjectOverview includes confidence_score in to_dict."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="p-1",
            project_name="test-project",
            mode="project_mode",
            current_lifecycle_state="testing",
            aspects={"api": "testing"},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id="lc-1",
            created_at="2026-02-08T00:00:00",
            updated_at="2026-02-08T00:00:00",
            confidence_score=0.8,
        )
        d = overview.to_dict()
        assert d["confidence_score"] == 0.8

    def test_confidence_defaults_to_none(self):
        """confidence_score defaults to None when not set."""
        from controller.dashboard_backend import ProjectOverview

        overview = ProjectOverview(
            project_id="p-1",
            project_name="test-project",
            mode="project_mode",
            current_lifecycle_state="testing",
            aspects={},
            active_cycle_number=1,
            last_test_deployment=None,
            last_prod_deployment=None,
            current_claude_job=None,
            lifecycle_id="lc-1",
            created_at="2026-02-08T00:00:00",
            updated_at="2026-02-08T00:00:00",
        )
        d = overview.to_dict()
        assert d["confidence_score"] is None


# ---------------------------------------------------------------------------
# PART 6: BlockReason Enum
# ---------------------------------------------------------------------------

class TestBlockReasonEnum:
    """BlockReason includes INSUFFICIENT_TESTING."""

    def test_insufficient_testing_in_enum(self):
        from controller.execution_dispatcher import BlockReason

        assert BlockReason.INSUFFICIENT_TESTING == "insufficient_testing"
        assert BlockReason.INSUFFICIENT_TESTING.value == "insufficient_testing"
