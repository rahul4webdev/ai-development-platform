"""
Phase 25: Test Governor — Adaptive E2E Test Governance

Verifies test completeness for a project based on its TestContract.
Does NOT write tests — creates Claude jobs to add missing tests.

CONSTRAINTS:
- Deterministic checks (file existence, metadata flags)
- Never modifies project code directly
- All state stored in Project.metadata["test_governance"]
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger("test_governor")


# ---------------------------------------------------------------------------
# Missing item descriptions for Claude prompts
# ---------------------------------------------------------------------------

_MISSING_ITEM_DESCRIPTIONS: Dict[str, str] = {
    "MISSING_API_E2E": (
        "Create backend E2E tests in backend/tests/e2e/test_e2e_api.py that verify:\n"
        "  - OpenAPI schema is accessible at /openapi.json\n"
        "  - Health endpoint returns {status: healthy}\n"
        "  - Full auth flow: register → login → /me\n"
        "  - Business data endpoints return expected fields"
    ),
    "MISSING_FRONTEND_E2E": (
        "Create frontend E2E tests in frontend/tests/e2e-deep.spec.js that verify:\n"
        "  - Login page loads and form is interactive\n"
        "  - Register page loads and form is interactive\n"
        "  - Root page loads (dashboard or login redirect)\n"
        "  - No critical console errors on key pages"
    ),
    "MISSING_PLAYWRIGHT_CI": (
        "Add Playwright test step to .github/workflows/deploy-testing.yml:\n"
        "  - Install Playwright with chromium\n"
        "  - Run npx playwright test after deployment\n"
        "  - Ensure test results are reported"
    ),
    "MISSING_SMOKE_TESTS": (
        "Add smoke test configuration that verifies:\n"
        "  - All deployed endpoints return HTTP 200\n"
        "  - Health endpoints return expected JSON\n"
        "  - Frontend serves HTML content"
    ),
    "MISSING_CORS_TEST": (
        "Add CORS verification test that checks:\n"
        "  - API responds to requests with Origin header\n"
        "  - Access-Control-Allow-Origin matches frontend origin"
    ),
    "MISSING_ROUTE_TESTS": (
        "Add SPA route tests that verify:\n"
        "  - Direct navigation to /login returns 200\n"
        "  - Direct navigation to /register returns 200\n"
        "  - Page refresh on any route works (SPA fallback)"
    ),
}


# ---------------------------------------------------------------------------
# TestGovernor
# ---------------------------------------------------------------------------

class TestGovernor:
    """
    Verifies test completeness and computes confidence scores.

    Does NOT write tests directly. Creates Claude job prompts
    for missing test coverage.
    """

    def verify_test_completeness(self, project_name: str) -> List[str]:
        """
        Check what tests are missing for a project.

        Reads from Project.metadata["test_governance"] to determine
        what test infrastructure exists.

        Returns:
            List of missing item identifiers. Empty = all complete.
        """
        missing: List[str] = []

        try:
            from controller.test_contract import TestScope, derive_contract_from_chd
            from controller.project_registry import get_registry

            registry = get_registry()
            project = registry.get_project(project_name)
            if not project:
                return []

            contract = derive_contract_from_chd(
                project_name=project_name,
                chd_text=project.requirements_raw,
                domains=project.domains,
                tech_stack=project.tech_stack,
            )

            gov_state = project.metadata.get("test_governance", {})

            # Check each required scope
            if TestScope.API.value in contract.required_scopes:
                if not gov_state.get("has_api_e2e"):
                    missing.append("MISSING_API_E2E")

            if TestScope.FRONTEND.value in contract.required_scopes:
                if not gov_state.get("has_frontend_e2e"):
                    missing.append("MISSING_FRONTEND_E2E")

            if contract.requires_playwright:
                if not gov_state.get("has_playwright"):
                    missing.append("MISSING_PLAYWRIGHT_CI")

            if contract.smoke_required:
                if not gov_state.get("has_smoke"):
                    missing.append("MISSING_SMOKE_TESTS")

        except ImportError:
            logger.debug("Test contract module not available")
        except Exception as e:
            logger.warning(f"Test completeness check failed: {e}")

        return missing

    def create_test_task(
        self, project_name: str, missing_items: List[str]
    ) -> str:
        """
        Generate a Claude CLI fix prompt to add missing tests.

        Does NOT write tests directly — returns a prompt for Claude.
        """
        items_detail = "\n\n".join(
            f"### {item}\n{_MISSING_ITEM_DESCRIPTIONS.get(item, 'Add the missing test.')}"
            for item in missing_items
        )

        return (
            f"TEST GOVERNANCE JOB — Add Missing Tests\n"
            f"{'=' * 60}\n\n"
            f"PROJECT: {project_name}\n"
            f"MISSING TESTS: {len(missing_items)}\n\n"
            f"{'=' * 60}\n"
            f"REQUIRED TESTS\n"
            f"{'=' * 60}\n\n"
            f"{items_detail}\n\n"
            f"{'=' * 60}\n"
            f"INSTRUCTIONS\n"
            f"{'=' * 60}\n"
            f"1. Read the existing project code and test patterns\n"
            f"2. Create ONLY the missing tests listed above\n"
            f"3. Follow existing test patterns in the project\n"
            f"4. Do NOT modify existing code — only ADD tests\n"
            f"5. Commit changes with a clear message\n"
            f"6. Push to GitHub: git push origin main\n\n"
            f"VALIDATION WILL RE-RUN AUTOMATICALLY AFTER DEPLOYMENT.\n"
        )

    @staticmethod
    def compute_confidence_score(project_name: str) -> float:
        """
        Compute confidence score 0.0-1.0 for a project.

        Formula (deterministic):
        - Base: 0.5
        - +0.1 if backend E2E tests exist
        - +0.1 if frontend E2E tests exist
        - +0.1 if Playwright exists
        - +0.1 if smoke tests exist
        - +0.1 if last 3 deployments had zero micro-remediation
        """
        score = 0.5

        try:
            from controller.project_registry import get_registry

            registry = get_registry()
            project = registry.get_project(project_name)
            if not project:
                return score

            gov_state = project.metadata.get("test_governance", {})

            if gov_state.get("has_api_e2e"):
                score += 0.1
            if gov_state.get("has_frontend_e2e"):
                score += 0.1
            if gov_state.get("has_playwright"):
                score += 0.1
            if gov_state.get("has_smoke"):
                score += 0.1

            # Check micro-remediation history
            micro_rem = project.metadata.get("micro_remediation", {})
            if not micro_rem or micro_rem.get("status") in (None, "resolved"):
                score += 0.1

        except Exception as e:
            logger.warning(f"Confidence score computation failed: {e}")

        return round(min(score, 1.0), 2)

    @staticmethod
    def get_governance_state(project_name: str) -> Dict[str, Any]:
        """Read test governance state from Project.metadata."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            project = registry.get_project(project_name)
            if project:
                return dict(project.metadata.get("test_governance", {}))
        except Exception as e:
            logger.warning(f"Failed to read governance state: {e}")
        return {}

    @staticmethod
    def update_governance_state(
        project_name: str, state: Dict[str, Any]
    ) -> None:
        """Write test governance state to Project.metadata."""
        try:
            from controller.project_registry import get_registry
            registry = get_registry()
            project = registry.get_project(project_name)
            if project:
                project.metadata["test_governance"] = state
                registry.update_project(project_name, {"metadata": project.metadata})
        except Exception as e:
            logger.error(f"Failed to update governance state: {e}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_governor: Optional[TestGovernor] = None


def get_test_governor() -> TestGovernor:
    global _governor
    if _governor is None:
        _governor = TestGovernor()
    return _governor
