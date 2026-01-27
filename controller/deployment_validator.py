"""
Deployment Validator for AI Development Platform
Phase 22: Rescue & Recovery System

Validates deployed endpoints and classifies failures to enable
automatic rescue job creation.

Features:
- HTTP endpoint validation with retries
- Failure classification (404, 500, CORS, etc.)
- Server-aware fix suggestions
- Integration with rescue engine

CONSTRAINTS:
- Timeout: 10 seconds per request
- Total validation timeout: 60 seconds
- Max retries: 2 with backoff
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger("deployment_validator")

# Try to import httpx, fall back to aiohttp
try:
    import httpx
    HTTP_CLIENT = "httpx"
except ImportError:
    try:
        import aiohttp
        HTTP_CLIENT = "aiohttp"
    except ImportError:
        HTTP_CLIENT = None
        logger.warning("No async HTTP client available (httpx or aiohttp)")


class DeploymentFailureType(str, Enum):
    """
    Classification of deployment failures.

    LOCKED: Each type has specific remediation strategies.
    """
    HTTP_404 = "HTTP_404"              # Endpoint not found (routing issue)
    HTTP_500 = "HTTP_500"              # Server error (code/config issue)
    HTTP_502 = "HTTP_502"              # Bad gateway (proxy/upstream issue)
    HTTP_503 = "HTTP_503"              # Service unavailable (not started)
    HTTP_CORS = "HTTP_CORS"            # CORS blocked (config issue)
    CONNECTION_REFUSED = "CONNECTION_REFUSED"  # Service not running
    CONNECTION_TIMEOUT = "CONNECTION_TIMEOUT"  # Service slow/unresponsive
    SSL_ERROR = "SSL_ERROR"            # Certificate issue
    DNS_ERROR = "DNS_ERROR"            # Domain not resolving
    PARTIAL_FAILURE = "PARTIAL_FAILURE"  # Some endpoints work, others don't
    UNKNOWN = "UNKNOWN"                # Cannot classify


@dataclass
class EndpointValidation:
    """Result of validating a single endpoint."""
    url: str
    endpoint_type: str  # "api", "frontend", "admin", "docs"
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    is_healthy: bool = False
    cors_headers: Optional[Dict[str, str]] = None
    response_body_preview: Optional[str] = None  # First 500 chars
    checked_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "endpoint_type": self.endpoint_type,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "error": self.error,
            "is_healthy": self.is_healthy,
            "cors_headers": self.cors_headers,
            "response_body_preview": self.response_body_preview,
            "checked_at": self.checked_at.isoformat()
        }


@dataclass
class DeploymentFailure:
    """Represents a deployment validation failure."""
    project_name: str
    failure_type: DeploymentFailureType
    failed_urls: List[Dict[str, Any]]  # [{url, error, status_code, response_time}]
    successful_urls: List[str]
    deployment_job_id: str
    detected_at: datetime
    suggested_fixes: List[str]
    diagnostic_info: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "failure_type": self.failure_type.value,
            "failed_urls": self.failed_urls,
            "successful_urls": self.successful_urls,
            "deployment_job_id": self.deployment_job_id,
            "detected_at": self.detected_at.isoformat(),
            "suggested_fixes": self.suggested_fixes,
            "diagnostic_info": self.diagnostic_info
        }


# Validation configuration per endpoint type
VALIDATION_CONFIG = {
    "api": {
        "paths_to_check": [
            "/",           # Root (may redirect or return info)
            "/health",     # Standard health endpoint
            "/docs",       # FastAPI docs
        ],
        "expected_status_codes": [200, 301, 302, 307, 308],
        "timeout_seconds": 10,
        "check_cors": True,
        "required_headers": ["content-type"],
    },
    "frontend": {
        "paths_to_check": [
            "/",           # Main page
        ],
        "expected_status_codes": [200],
        "timeout_seconds": 10,
        "check_cors": False,
        "required_content": ["<html", "<!DOCTYPE", "<!doctype"],
    },
    "admin": {
        "paths_to_check": [
            "/",
        ],
        "expected_status_codes": [200, 301, 302, 307, 308],
        "timeout_seconds": 10,
        "check_cors": False,
    },
    "docs": {
        "paths_to_check": [
            "/",
        ],
        "expected_status_codes": [200],
        "timeout_seconds": 10,
        "check_cors": False,
        "required_content": ["swagger", "openapi", "redoc"],
    }
}

# Retry configuration
RETRY_CONFIG = {
    "max_retries": 2,
    "backoff_seconds": [2, 5],  # Wait 2s, then 5s
    "retry_on_status": [502, 503, 504],
    "retry_on_errors": ["timeout", "connection"],
}


class DeploymentFailureClassifier:
    """
    Classifies deployment failures based on HTTP responses.

    Rules are deterministic - no ML/probabilistic decisions.
    """

    @classmethod
    def classify_from_validations(
        cls,
        validations: List[EndpointValidation],
        server_config: Optional[Any] = None  # ServerConfig from server_detector
    ) -> Tuple[DeploymentFailureType, List[str]]:
        """
        Classify failure type and generate fix suggestions.

        Args:
            validations: List of endpoint validation results
            server_config: Optional server configuration for server-aware fixes

        Returns:
            (failure_type, suggested_fixes)
        """
        from controller.server_detector import get_server_detector, ServerConfig

        # Analyze errors
        failed = [v for v in validations if not v.is_healthy]
        if not failed:
            return DeploymentFailureType.UNKNOWN, []

        # Count error types
        error_counts: Dict[str, int] = {}
        status_codes: List[int] = []

        for v in failed:
            if v.status_code:
                status_codes.append(v.status_code)

            if v.error:
                error_lower = v.error.lower()
                if "refused" in error_lower or "connection refused" in error_lower:
                    error_counts["connection_refused"] = error_counts.get("connection_refused", 0) + 1
                elif "timeout" in error_lower or "timed out" in error_lower:
                    error_counts["timeout"] = error_counts.get("timeout", 0) + 1
                elif "ssl" in error_lower or "certificate" in error_lower:
                    error_counts["ssl"] = error_counts.get("ssl", 0) + 1
                elif "dns" in error_lower or "resolve" in error_lower or "getaddrinfo" in error_lower:
                    error_counts["dns"] = error_counts.get("dns", 0) + 1
                elif "cors" in error_lower:
                    error_counts["cors"] = error_counts.get("cors", 0) + 1

        # Determine primary failure type
        failure_type = DeploymentFailureType.UNKNOWN

        # Check connection errors first (more severe)
        if error_counts.get("connection_refused", 0) > 0:
            failure_type = DeploymentFailureType.CONNECTION_REFUSED
        elif error_counts.get("timeout", 0) > 0:
            failure_type = DeploymentFailureType.CONNECTION_TIMEOUT
        elif error_counts.get("ssl", 0) > 0:
            failure_type = DeploymentFailureType.SSL_ERROR
        elif error_counts.get("dns", 0) > 0:
            failure_type = DeploymentFailureType.DNS_ERROR
        elif error_counts.get("cors", 0) > 0:
            failure_type = DeploymentFailureType.HTTP_CORS
        # Check HTTP status codes
        elif status_codes:
            most_common = max(set(status_codes), key=status_codes.count)
            if most_common == 404:
                failure_type = DeploymentFailureType.HTTP_404
            elif most_common == 500:
                failure_type = DeploymentFailureType.HTTP_500
            elif most_common == 502:
                failure_type = DeploymentFailureType.HTTP_502
            elif most_common == 503:
                failure_type = DeploymentFailureType.HTTP_503
            elif most_common >= 400:
                failure_type = DeploymentFailureType.HTTP_500  # Generic server error

        # Check for partial failure
        successful = [v for v in validations if v.is_healthy]
        if successful and failed:
            # Some work, some don't - could indicate routing or config issue
            if failure_type == DeploymentFailureType.UNKNOWN:
                failure_type = DeploymentFailureType.PARTIAL_FAILURE

        # Get server-aware fix suggestions
        detector = get_server_detector()
        suggested_fixes = detector.get_rescue_instructions(
            failure_type.value,
            server_config
        )

        return failure_type, suggested_fixes


class DeploymentValidator:
    """
    Validates deployed endpoints with configurable checks.

    CONSTRAINTS:
    - Timeout per request: 10 seconds
    - Total validation timeout: 60 seconds
    - Retry count: 2 (with backoff)
    """

    def __init__(self):
        self._client = None

    async def validate_deployment(
        self,
        project_name: str,
        urls: Dict[str, str],  # {"api": "https://...", "frontend": "https://..."}
        deployment_job_id: str,
        server_config: Optional[Any] = None
    ) -> Tuple[bool, Optional[DeploymentFailure], List[EndpointValidation]]:
        """
        Validate all configured URLs for a deployment.

        Args:
            project_name: Name of the project
            urls: Dictionary of endpoint type to URL
            deployment_job_id: ID of the deployment job
            server_config: Optional server configuration

        Returns:
            (all_healthy, failure_info, validations)
        """
        validations: List[EndpointValidation] = []

        for endpoint_type, base_url in urls.items():
            if not base_url or base_url.startswith("Not configured"):
                continue

            # Get validation config for this endpoint type
            config = VALIDATION_CONFIG.get(endpoint_type, VALIDATION_CONFIG["api"])

            # Validate each path for this endpoint
            for path in config.get("paths_to_check", ["/"]):
                url = base_url.rstrip("/") + path
                validation = await self.validate_endpoint(
                    url=url,
                    endpoint_type=endpoint_type,
                    config=config
                )
                validations.append(validation)

        # Check if all healthy
        all_healthy = all(v.is_healthy for v in validations) if validations else False

        if all_healthy:
            return True, None, validations

        # Classify failure
        failure_type, suggested_fixes = DeploymentFailureClassifier.classify_from_validations(
            validations, server_config
        )

        # Build failure info
        failed_urls = [
            {
                "url": v.url,
                "error": v.error or f"HTTP {v.status_code}",
                "status_code": v.status_code,
                "response_time_ms": v.response_time_ms
            }
            for v in validations if not v.is_healthy
        ]

        successful_urls = [v.url for v in validations if v.is_healthy]

        failure = DeploymentFailure(
            project_name=project_name,
            failure_type=failure_type,
            failed_urls=failed_urls,
            successful_urls=successful_urls,
            deployment_job_id=deployment_job_id,
            detected_at=datetime.utcnow(),
            suggested_fixes=suggested_fixes,
            diagnostic_info={
                "total_endpoints": len(validations),
                "failed_count": len(failed_urls),
                "success_count": len(successful_urls),
                "urls_checked": urls
            }
        )

        return False, failure, validations

    async def validate_endpoint(
        self,
        url: str,
        endpoint_type: str,
        config: Optional[Dict] = None
    ) -> EndpointValidation:
        """
        Validate a single endpoint with retries.

        Args:
            url: Full URL to validate
            endpoint_type: Type of endpoint (api, frontend, etc.)
            config: Validation configuration

        Returns:
            EndpointValidation result
        """
        if config is None:
            config = VALIDATION_CONFIG.get(endpoint_type, VALIDATION_CONFIG["api"])

        timeout = config.get("timeout_seconds", 10)
        expected_codes = config.get("expected_status_codes", [200])

        # Retry loop
        last_error = None
        for attempt in range(RETRY_CONFIG["max_retries"] + 1):
            if attempt > 0:
                # Wait before retry
                wait_time = RETRY_CONFIG["backoff_seconds"][min(attempt - 1, len(RETRY_CONFIG["backoff_seconds"]) - 1)]
                await asyncio.sleep(wait_time)

            try:
                start_time = datetime.utcnow()
                status_code, response_body, headers, error = await self._make_request(url, timeout)
                elapsed_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

                if error:
                    last_error = error
                    # Check if we should retry
                    error_lower = error.lower()
                    should_retry = any(e in error_lower for e in RETRY_CONFIG["retry_on_errors"])
                    if not should_retry:
                        break
                    continue

                # Check status code
                is_healthy = status_code in expected_codes

                # Check required content if specified
                if is_healthy and "required_content" in config and response_body:
                    required = config["required_content"]
                    has_content = any(r.lower() in response_body.lower() for r in required)
                    is_healthy = is_healthy and has_content

                # Extract CORS headers if needed
                cors_headers = None
                if config.get("check_cors") and headers:
                    cors_headers = {
                        k: v for k, v in headers.items()
                        if k.lower().startswith("access-control")
                    }

                return EndpointValidation(
                    url=url,
                    endpoint_type=endpoint_type,
                    status_code=status_code,
                    response_time_ms=elapsed_ms,
                    error=None,
                    is_healthy=is_healthy,
                    cors_headers=cors_headers,
                    response_body_preview=response_body[:500] if response_body else None
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Validation attempt {attempt + 1} failed for {url}: {e}")

        # All retries failed
        return EndpointValidation(
            url=url,
            endpoint_type=endpoint_type,
            status_code=None,
            response_time_ms=None,
            error=last_error or "Unknown error",
            is_healthy=False,
            cors_headers=None,
            response_body_preview=None
        )

    async def _make_request(
        self,
        url: str,
        timeout: int
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], Optional[str]]:
        """
        Make HTTP request using available client.

        Returns:
            (status_code, response_body, headers, error)
        """
        if HTTP_CLIENT == "httpx":
            return await self._request_httpx(url, timeout)
        elif HTTP_CLIENT == "aiohttp":
            return await self._request_aiohttp(url, timeout)
        else:
            return None, None, None, "No HTTP client available"

    async def _request_httpx(
        self,
        url: str,
        timeout: int
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], Optional[str]]:
        """Make request using httpx."""
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=True,
                verify=True  # Verify SSL
            ) as client:
                response = await client.get(url)
                return (
                    response.status_code,
                    response.text[:2000] if response.text else None,
                    dict(response.headers),
                    None
                )
        except httpx.ConnectError as e:
            if "refused" in str(e).lower():
                return None, None, None, "Connection refused"
            return None, None, None, f"Connection error: {e}"
        except httpx.TimeoutException:
            return None, None, None, "Connection timeout"
        except httpx.HTTPStatusError as e:
            return e.response.status_code, None, None, str(e)
        except Exception as e:
            error_str = str(e).lower()
            if "ssl" in error_str or "certificate" in error_str:
                return None, None, None, f"SSL error: {e}"
            if "dns" in error_str or "resolve" in error_str:
                return None, None, None, f"DNS error: {e}"
            return None, None, None, str(e)

    async def _request_aiohttp(
        self,
        url: str,
        timeout: int
    ) -> Tuple[Optional[int], Optional[str], Optional[Dict], Optional[str]]:
        """Make request using aiohttp."""
        try:
            import aiohttp
            timeout_obj = aiohttp.ClientTimeout(total=timeout)
            async with aiohttp.ClientSession(timeout=timeout_obj) as session:
                async with session.get(url, ssl=True) as response:
                    body = await response.text()
                    return (
                        response.status,
                        body[:2000] if body else None,
                        dict(response.headers),
                        None
                    )
        except aiohttp.ClientConnectorError as e:
            if "refused" in str(e).lower():
                return None, None, None, "Connection refused"
            return None, None, None, f"Connection error: {e}"
        except asyncio.TimeoutError:
            return None, None, None, "Connection timeout"
        except Exception as e:
            error_str = str(e).lower()
            if "ssl" in error_str or "certificate" in error_str:
                return None, None, None, f"SSL error: {e}"
            return None, None, None, str(e)


# Singleton instance
_validator: Optional[DeploymentValidator] = None


def get_deployment_validator() -> DeploymentValidator:
    """Get singleton deployment validator instance."""
    global _validator
    if _validator is None:
        _validator = DeploymentValidator()
    return _validator
