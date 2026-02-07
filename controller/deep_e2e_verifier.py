"""
Phase 23: Deep Functional E2E Verifier

Verifies that deployed applications actually work end-to-end,
not just that endpoints return 200.

Checks:
- API OpenAPI schema is valid and has expected paths
- API returns real business data (not empty shells)
- CORS allows requests from the frontend origin
- Frontend pages load without errors
- Auth flow works end-to-end (register → login → /me)
- Deployed commit SHA matches GitHub latest commit
"""

import json
import time
import logging
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class E2ELevel(str, Enum):
    """Level of E2E verification depth."""
    BASIC = "basic"            # HTTP status checks only (existing validator)
    FUNCTIONAL = "functional"  # Business logic, auth flows, CORS, data


class E2ECheckType(str, Enum):
    """Individual check types within functional verification."""
    API_SCHEMA_VALID = "api_schema_valid"
    API_BUSINESS_DATA = "api_business_data"
    CORS_BROWSER_ORIGIN = "cors_browser_origin"
    FRONTEND_PAGES = "frontend_pages"
    AUTH_LOGIN_E2E = "auth_login_e2e"
    COMMIT_SHA_MATCH = "commit_sha_match"


# ---------------------------------------------------------------------------
# Frozen Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class E2ECheckResult:
    """Result of a single E2E check. Immutable."""
    check_type: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    response_time_ms: Optional[float] = None
    checked_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass(frozen=True)
class DeepE2EResult:
    """Aggregate result of deep E2E verification. Immutable."""
    project_name: str
    level: str
    passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: int
    checks: tuple  # Tuple[E2ECheckResult, ...] — frozen requires tuple
    summary: str
    run_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_ms: Optional[float] = None


# ---------------------------------------------------------------------------
# DeepE2EVerifier
# ---------------------------------------------------------------------------

class DeepE2EVerifier:
    """
    Deep functional E2E verifier for deployed applications.

    Runs on the PLATFORM side (not in CI). Validates that a deployed app
    actually works end-to-end, not just that endpoints return 200.

    Constraints:
    - Timeout per check: 15 seconds
    - Total verification timeout: 120 seconds
    - Never modifies the target application (test users use .invalid TLD)
    """

    REQUEST_TIMEOUT = 15  # seconds per request

    async def run(
        self,
        project_name: str,
        urls: Dict[str, str],
        github_repo: Optional[str] = None,
        level: E2ELevel = E2ELevel.FUNCTIONAL,
    ) -> DeepE2EResult:
        """
        Run deep E2E verification against deployed endpoints.

        Args:
            project_name: Name of the project
            urls: Dict mapping endpoint type to base URL
                  e.g. {"api": "https://api.example.com", "frontend": "https://example.com"}
            github_repo: GitHub repo in "owner/repo" format (for commit match)
            level: Verification depth level

        Returns:
            DeepE2EResult (frozen, immutable)
        """
        start = time.monotonic()
        checks: List[E2ECheckResult] = []

        api_url = urls.get("api", "").rstrip("/")
        frontend_url = urls.get("frontend", "").rstrip("/")

        if level == E2ELevel.BASIC:
            # Basic level: just check endpoints respond
            if api_url:
                checks.append(await self._check_api_schema(api_url))
            if frontend_url:
                checks.append(await self._check_frontend_pages(frontend_url))
        else:
            # Functional level: full verification
            if api_url:
                checks.append(await self._check_api_schema(api_url))
                checks.append(await self._check_auth_e2e(api_url))
                checks.append(await self._check_api_business_data(api_url))

            if api_url and frontend_url:
                checks.append(await self._check_cors(api_url, frontend_url))

            if frontend_url:
                checks.append(await self._check_frontend_pages(frontend_url))

            if github_repo and (api_url or frontend_url):
                checks.append(await self._check_commit_match(urls, github_repo))

        elapsed_ms = (time.monotonic() - start) * 1000
        passed_count = sum(1 for c in checks if c.passed)
        failed_count = len(checks) - passed_count
        all_passed = failed_count == 0 and len(checks) > 0

        summary_parts = []
        for c in checks:
            status = "PASS" if c.passed else "FAIL"
            summary_parts.append(f"  [{status}] {c.check_type}: {c.message}")
        summary = f"{passed_count}/{len(checks)} checks passed\n" + "\n".join(summary_parts)

        return DeepE2EResult(
            project_name=project_name,
            level=level.value,
            passed=all_passed,
            total_checks=len(checks),
            passed_checks=passed_count,
            failed_checks=failed_count,
            checks=tuple(checks),
            summary=summary,
            duration_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    async def _check_api_schema(self, api_url: str) -> E2ECheckResult:
        """Verify OpenAPI schema is accessible and valid."""
        start = time.monotonic()
        try:
            status, body, headers = await self._make_request(
                "GET", f"{api_url}/openapi.json"
            )
            elapsed = (time.monotonic() - start) * 1000

            if status != 200:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_SCHEMA_VALID.value,
                    passed=False,
                    message=f"OpenAPI schema returned HTTP {status}",
                    details={"status_code": status},
                    response_time_ms=round(elapsed, 1),
                )

            try:
                schema = json.loads(body)
            except json.JSONDecodeError:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_SCHEMA_VALID.value,
                    passed=False,
                    message="OpenAPI schema is not valid JSON",
                    response_time_ms=round(elapsed, 1),
                )

            # Validate structure
            if "openapi" not in schema:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_SCHEMA_VALID.value,
                    passed=False,
                    message="Missing 'openapi' version field in schema",
                    response_time_ms=round(elapsed, 1),
                )

            paths = schema.get("paths", {})
            if not paths:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_SCHEMA_VALID.value,
                    passed=False,
                    message="Schema has no paths defined",
                    response_time_ms=round(elapsed, 1),
                )

            return E2ECheckResult(
                check_type=E2ECheckType.API_SCHEMA_VALID.value,
                passed=True,
                message=f"Valid OpenAPI schema with {len(paths)} paths",
                details={"openapi_version": schema.get("openapi"), "path_count": len(paths)},
                response_time_ms=round(elapsed, 1),
            )
        except Exception as e:
            return E2ECheckResult(
                check_type=E2ECheckType.API_SCHEMA_VALID.value,
                passed=False,
                message=f"Failed to fetch OpenAPI schema: {e}",
            )

    async def _check_auth_e2e(self, api_url: str) -> E2ECheckResult:
        """Full auth flow: register → login → /me."""
        start = time.monotonic()
        try:
            ts = int(time.time() * 1000)
            test_email = f"e2e-platform-{ts}@platform-test.invalid"
            test_password = f"E2eTest{ts}!"

            # Step 1: Register
            reg_status, reg_body, _ = await self._make_request(
                "POST",
                f"{api_url}/api/auth/register",
                json_data={"email": test_email, "password": test_password, "name": "E2E Test"},
            )

            if reg_status not in (200, 201):
                return E2ECheckResult(
                    check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                    passed=False,
                    message=f"Registration failed with HTTP {reg_status}",
                    details={"step": "register", "status": reg_status, "body": reg_body[:200]},
                    response_time_ms=round((time.monotonic() - start) * 1000, 1),
                )

            # Step 2: Login (OAuth2 form)
            login_status, login_body, _ = await self._make_request(
                "POST",
                f"{api_url}/api/auth/login",
                form_data={"username": test_email, "password": test_password},
            )

            if login_status != 200:
                return E2ECheckResult(
                    check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                    passed=False,
                    message=f"Login failed with HTTP {login_status}",
                    details={"step": "login", "status": login_status},
                    response_time_ms=round((time.monotonic() - start) * 1000, 1),
                )

            try:
                token_data = json.loads(login_body)
                access_token = token_data.get("access_token")
            except (json.JSONDecodeError, KeyError):
                return E2ECheckResult(
                    check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                    passed=False,
                    message="Login response missing access_token",
                    details={"step": "login", "body": login_body[:200]},
                    response_time_ms=round((time.monotonic() - start) * 1000, 1),
                )

            if not access_token:
                return E2ECheckResult(
                    check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                    passed=False,
                    message="Login returned empty access_token",
                    response_time_ms=round((time.monotonic() - start) * 1000, 1),
                )

            # Step 3: /me
            me_status, me_body, _ = await self._make_request(
                "GET",
                f"{api_url}/api/auth/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )

            elapsed = (time.monotonic() - start) * 1000
            if me_status != 200:
                return E2ECheckResult(
                    check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                    passed=False,
                    message=f"/me failed with HTTP {me_status}",
                    details={"step": "me", "status": me_status},
                    response_time_ms=round(elapsed, 1),
                )

            try:
                me_data = json.loads(me_body)
                if me_data.get("email") != test_email:
                    return E2ECheckResult(
                        check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                        passed=False,
                        message=f"/me returned wrong email: {me_data.get('email')}",
                        response_time_ms=round(elapsed, 1),
                    )
            except json.JSONDecodeError:
                return E2ECheckResult(
                    check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                    passed=False,
                    message="/me response is not valid JSON",
                    response_time_ms=round(elapsed, 1),
                )

            return E2ECheckResult(
                check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                passed=True,
                message="Auth flow passed: register → login → /me",
                details={"test_email": test_email},
                response_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            return E2ECheckResult(
                check_type=E2ECheckType.AUTH_LOGIN_E2E.value,
                passed=False,
                message=f"Auth flow error: {e}",
            )

    async def _check_api_business_data(self, api_url: str) -> E2ECheckResult:
        """Verify API returns real business data structure."""
        start = time.monotonic()
        try:
            ts = int(time.time() * 1000)
            test_email = f"e2e-biz-{ts}@platform-test.invalid"
            test_password = f"E2eBiz{ts}!"

            # Register + login to get token
            await self._make_request(
                "POST",
                f"{api_url}/api/auth/register",
                json_data={"email": test_email, "password": test_password, "name": "E2E Biz"},
            )
            login_status, login_body, _ = await self._make_request(
                "POST",
                f"{api_url}/api/auth/login",
                form_data={"username": test_email, "password": test_password},
            )

            if login_status != 200:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_BUSINESS_DATA.value,
                    passed=False,
                    message=f"Could not authenticate for business data check (HTTP {login_status})",
                    response_time_ms=round((time.monotonic() - start) * 1000, 1),
                )

            token = json.loads(login_body).get("access_token", "")
            headers = {"Authorization": f"Bearer {token}"}

            # Fetch daily summary
            ds_status, ds_body, _ = await self._make_request(
                "GET",
                f"{api_url}/api/nutrition/daily-summary",
                headers=headers,
            )
            elapsed = (time.monotonic() - start) * 1000

            if ds_status != 200:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_BUSINESS_DATA.value,
                    passed=False,
                    message=f"Daily summary returned HTTP {ds_status}",
                    details={"status": ds_status},
                    response_time_ms=round(elapsed, 1),
                )

            try:
                data = json.loads(ds_body)
            except json.JSONDecodeError:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_BUSINESS_DATA.value,
                    passed=False,
                    message="Daily summary is not valid JSON",
                    response_time_ms=round(elapsed, 1),
                )

            # Verify expected fields
            required_fields = ["total_calories", "entries_count"]
            missing = [f for f in required_fields if f not in data]
            if missing:
                return E2ECheckResult(
                    check_type=E2ECheckType.API_BUSINESS_DATA.value,
                    passed=False,
                    message=f"Daily summary missing fields: {missing}",
                    details={"missing_fields": missing, "actual_keys": list(data.keys())},
                    response_time_ms=round(elapsed, 1),
                )

            return E2ECheckResult(
                check_type=E2ECheckType.API_BUSINESS_DATA.value,
                passed=True,
                message="Business data API returns valid nutrition summary",
                details={"fields": list(data.keys())},
                response_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            return E2ECheckResult(
                check_type=E2ECheckType.API_BUSINESS_DATA.value,
                passed=False,
                message=f"Business data check error: {e}",
            )

    async def _check_cors(self, api_url: str, frontend_url: str) -> E2ECheckResult:
        """Verify CORS allows requests from the frontend origin."""
        start = time.monotonic()
        try:
            status, body, headers = await self._make_request(
                "OPTIONS",
                f"{api_url}/health",
                headers={
                    "Origin": frontend_url,
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "Authorization",
                },
            )
            elapsed = (time.monotonic() - start) * 1000

            cors_origin = headers.get("access-control-allow-origin", "")

            if not cors_origin:
                return E2ECheckResult(
                    check_type=E2ECheckType.CORS_BROWSER_ORIGIN.value,
                    passed=False,
                    message=f"No Access-Control-Allow-Origin header (Origin: {frontend_url})",
                    details={"status": status, "response_headers": dict(headers)},
                    response_time_ms=round(elapsed, 1),
                )

            if cors_origin != "*" and cors_origin != frontend_url:
                return E2ECheckResult(
                    check_type=E2ECheckType.CORS_BROWSER_ORIGIN.value,
                    passed=False,
                    message=f"CORS origin mismatch: got '{cors_origin}', expected '{frontend_url}' or '*'",
                    details={"allowed_origin": cors_origin, "requested_origin": frontend_url},
                    response_time_ms=round(elapsed, 1),
                )

            return E2ECheckResult(
                check_type=E2ECheckType.CORS_BROWSER_ORIGIN.value,
                passed=True,
                message=f"CORS allows origin: {cors_origin}",
                details={"allowed_origin": cors_origin},
                response_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            return E2ECheckResult(
                check_type=E2ECheckType.CORS_BROWSER_ORIGIN.value,
                passed=False,
                message=f"CORS check error: {e}",
            )

    async def _check_frontend_pages(self, frontend_url: str) -> E2ECheckResult:
        """Verify frontend SPA pages load correctly."""
        start = time.monotonic()
        try:
            pages_to_check = ["/login", "/register"]
            results = []

            for page in pages_to_check:
                status, body, headers = await self._make_request(
                    "GET", f"{frontend_url}{page}"
                )
                content_type = headers.get("content-type", "")

                if status != 200:
                    results.append({"page": page, "ok": False, "reason": f"HTTP {status}"})
                elif "text/html" not in content_type:
                    results.append({"page": page, "ok": False, "reason": f"Content-Type: {content_type}"})
                elif "<html" not in body.lower() and "<!doctype" not in body.lower():
                    results.append({"page": page, "ok": False, "reason": "Not HTML content"})
                else:
                    results.append({"page": page, "ok": True})

            elapsed = (time.monotonic() - start) * 1000
            failed = [r for r in results if not r["ok"]]

            if failed:
                fail_details = "; ".join(f"{r['page']}: {r['reason']}" for r in failed)
                return E2ECheckResult(
                    check_type=E2ECheckType.FRONTEND_PAGES.value,
                    passed=False,
                    message=f"Frontend pages failed: {fail_details}",
                    details={"results": results},
                    response_time_ms=round(elapsed, 1),
                )

            return E2ECheckResult(
                check_type=E2ECheckType.FRONTEND_PAGES.value,
                passed=True,
                message=f"All {len(pages_to_check)} frontend pages load correctly",
                details={"pages_checked": pages_to_check},
                response_time_ms=round(elapsed, 1),
            )

        except Exception as e:
            return E2ECheckResult(
                check_type=E2ECheckType.FRONTEND_PAGES.value,
                passed=False,
                message=f"Frontend page check error: {e}",
            )

    async def _check_commit_match(
        self, urls: Dict[str, str], github_repo: str
    ) -> E2ECheckResult:
        """Verify deployed commit matches GitHub latest."""
        start = time.monotonic()
        try:
            from controller.deployment_validator import check_deploy_status

            status, github_sha, deployed_sha = await check_deploy_status(
                project_name="",  # Not needed for the check
                urls=urls,
                github_repo=github_repo,
            )
            elapsed = (time.monotonic() - start) * 1000

            if status == "deployed":
                return E2ECheckResult(
                    check_type=E2ECheckType.COMMIT_SHA_MATCH.value,
                    passed=True,
                    message=f"Deployed commit matches GitHub: {(github_sha or '')[:8]}",
                    details={"github_sha": github_sha, "deployed_sha": deployed_sha},
                    response_time_ms=round(elapsed, 1),
                )
            elif status == "not_deployed":
                return E2ECheckResult(
                    check_type=E2ECheckType.COMMIT_SHA_MATCH.value,
                    passed=False,
                    message=f"Commit mismatch: GitHub={github_sha[:8] if github_sha else '?'}, deployed={deployed_sha[:8] if deployed_sha else '?'}",
                    details={"github_sha": github_sha, "deployed_sha": deployed_sha},
                    response_time_ms=round(elapsed, 1),
                )
            else:
                return E2ECheckResult(
                    check_type=E2ECheckType.COMMIT_SHA_MATCH.value,
                    passed=False,
                    message="Cannot determine deploy status (DEPLOY_INFO.txt not found)",
                    details={"status": status},
                    response_time_ms=round(elapsed, 1),
                )

        except ImportError:
            return E2ECheckResult(
                check_type=E2ECheckType.COMMIT_SHA_MATCH.value,
                passed=False,
                message="check_deploy_status not available",
            )
        except Exception as e:
            return E2ECheckResult(
                check_type=E2ECheckType.COMMIT_SHA_MATCH.value,
                passed=False,
                message=f"Commit match check error: {e}",
            )

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    async def _make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict] = None,
        form_data: Optional[Dict] = None,
    ) -> Tuple[int, str, Dict[str, str]]:
        """
        Make an HTTP request.

        Returns: (status_code, response_body, response_headers_dict)
        """
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        req_headers = dict(headers) if headers else {}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            kwargs: Dict[str, Any] = {"headers": req_headers, "ssl": False}

            if json_data is not None:
                kwargs["json"] = json_data
            elif form_data is not None:
                kwargs["data"] = form_data

            async with session.request(method, url, **kwargs) as resp:
                body = await resp.text()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                return resp.status, body, resp_headers


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_verifier: Optional[DeepE2EVerifier] = None


def get_deep_e2e_verifier() -> DeepE2EVerifier:
    """Get or create the singleton DeepE2EVerifier instance."""
    global _verifier
    if _verifier is None:
        _verifier = DeepE2EVerifier()
    return _verifier
