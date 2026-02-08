"""
Phase 25: Test Contract — Adaptive E2E Test Governance

Defines what tests a project MUST have based on its CHD (Claude Handoff Document),
domains, and tech stack. Deterministic derivation — no ML, no heuristics.

CONSTRAINTS:
- Contract is frozen (immutable once derived)
- Rules are deterministic based on CHD parsing
- No external network calls during derivation
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger("test_contract")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TestScope(str, Enum):
    API = "api"
    FRONTEND = "frontend"
    ADMIN = "admin"
    INTEGRATION = "integration"
    DEPLOYMENT = "deployment"


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TestContract:
    project: str
    required_scopes: tuple      # Tuple of TestScope values (strings)
    min_api_coverage: int       # Minimum API test coverage percentage
    min_frontend_coverage: int  # Minimum frontend test coverage percentage
    requires_playwright: bool
    smoke_required: bool
    e2e_required: bool
    derived_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# CHD parser helpers
# ---------------------------------------------------------------------------

def _parse_chd_testing_env(chd_text: str) -> Dict[str, str]:
    """
    Parse the testing_env section from CHD text.

    Returns dict mapping aspect -> domain, e.g.:
        {"api": "healthapi.example.com", "frontend": "health.example.com"}
    """
    if not chd_text:
        return {}

    result: Dict[str, str] = {}
    lines = chd_text.split("\n")
    in_testing_env = False
    current_aspect = None

    for line in lines:
        stripped = line.strip()

        if "testing_env:" in line:
            in_testing_env = True
            continue

        if "production_env:" in line:
            in_testing_env = False
            continue

        if not in_testing_env:
            continue

        # Detect aspect headers
        if stripped.endswith(":") and not stripped.startswith("-"):
            if "api:" in stripped:
                current_aspect = "api"
            elif "frontend_web:" in stripped or "frontend:" in stripped:
                current_aspect = "frontend"
            elif "admin_panel:" in stripped or "admin:" in stripped:
                current_aspect = "admin"
            else:
                current_aspect = None
            continue

        # Detect domain value
        if current_aspect and "domain:" in line:
            parts = line.split("domain:", 1)
            if len(parts) == 2:
                domain = parts[1].strip()
                if domain:
                    result[current_aspect] = domain

    return result


# ---------------------------------------------------------------------------
# Contract derivation
# ---------------------------------------------------------------------------

def derive_contract_from_chd(
    project_name: str,
    chd_text: Optional[str] = None,
    domains: Optional[Dict[str, str]] = None,
    tech_stack: Optional[Dict[str, Any]] = None,
) -> TestContract:
    """
    Derive a TestContract from project CHD, domains, and tech stack.

    Deterministic rules — first checks CHD text, then falls back to domains dict.

    Args:
        project_name: Project identifier
        chd_text: Raw CHD text (requirements_raw)
        domains: Domain mapping from registry (aspect -> URL)
        tech_stack: Tech stack info from registry

    Returns:
        Frozen TestContract
    """
    scopes: List[str] = []
    requires_playwright = False

    domains = domains or {}
    tech_stack = tech_stack or {}

    # Parse CHD for testing_env section
    chd_aspects = _parse_chd_testing_env(chd_text or "")

    # Rule 1: CHD testing_env aspects
    if "api" in chd_aspects:
        if TestScope.API.value not in scopes:
            scopes.append(TestScope.API.value)
    if "frontend" in chd_aspects:
        if TestScope.FRONTEND.value not in scopes:
            scopes.append(TestScope.FRONTEND.value)
    if "admin" in chd_aspects:
        if TestScope.ADMIN.value not in scopes:
            scopes.append(TestScope.ADMIN.value)

    # Rule 2: Domains dict fallback
    if "api" in domains and TestScope.API.value not in scopes:
        scopes.append(TestScope.API.value)
    if "frontend" in domains and TestScope.FRONTEND.value not in scopes:
        scopes.append(TestScope.FRONTEND.value)
    if "admin" in domains and TestScope.ADMIN.value not in scopes:
        scopes.append(TestScope.ADMIN.value)

    # Rule 3: Any deployment → DEPLOYMENT scope
    if domains or chd_aspects:
        if TestScope.DEPLOYMENT.value not in scopes:
            scopes.append(TestScope.DEPLOYMENT.value)

    # Rule 4: Check tech_stack for Playwright
    tech_str = str(tech_stack).lower()
    if "playwright" in tech_str or "e2e" in tech_str:
        requires_playwright = True

    # If frontend scope present, likely needs Playwright
    if TestScope.FRONTEND.value in scopes:
        requires_playwright = True

    return TestContract(
        project=project_name,
        required_scopes=tuple(scopes),
        min_api_coverage=70,
        min_frontend_coverage=60,
        requires_playwright=requires_playwright,
        smoke_required=True,
        e2e_required=True,
    )
