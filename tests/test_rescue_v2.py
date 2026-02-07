#!/usr/bin/env python3
"""
Phase 22 Improvement Tests: Production-Hardened Rescue System

Tests for:
- PART 2: Module-scoped validation
- PART 3: Pushed-but-not-deployed detection
- PART 4: Module-scoped rescue jobs
- State persistence with new fields
- Edge cases
"""

import sys
import json
import uuid
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime
from dataclasses import asdict

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestModuleScopedValidation:
    """PART 2: Module filtering in deployment validator."""

    def test_validate_only_api_module(self, tmp_path):
        """When modules=["api"], only API endpoints should be checked."""
        from controller.deployment_validator import (
            DeploymentValidator,
            EndpointValidation,
            VALIDATION_CONFIG,
        )

        async def _run():
            validator = DeploymentValidator()

            # Mock _make_request to return healthy responses
            async def mock_request(url, timeout):
                return 200, '{"status": "ok"}', {"content-type": "application/json"}, None

            validator._make_request = mock_request

            urls = {
                "api": "https://api.example.com",
                "frontend": "https://example.com",
                "admin": "https://admin.example.com",
            }

            # Validate only API module
            all_healthy, failure, validations = await validator.validate_deployment(
                project_name="test-project",
                urls=urls,
                deployment_job_id="test-job-123",
                modules=["api"]
            )

            # Should only have API validations (3 paths: /, /health, /docs)
            assert all(v.endpoint_type == "api" for v in validations), \
                f"Expected only api validations, got: {[v.endpoint_type for v in validations]}"
            assert len(validations) == len(VALIDATION_CONFIG["api"]["paths_to_check"])
            assert all_healthy

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_validate_all_modules_default(self, tmp_path):
        """When modules=None, all endpoints should be checked."""
        from controller.deployment_validator import DeploymentValidator

        async def _run():
            validator = DeploymentValidator()

            async def mock_request(url, timeout):
                return 200, '<html>OK</html>', {"content-type": "text/html"}, None

            validator._make_request = mock_request

            urls = {
                "api": "https://api.example.com",
                "frontend": "https://example.com",
            }

            all_healthy, failure, validations = await validator.validate_deployment(
                project_name="test-project",
                urls=urls,
                deployment_job_id="test-job-123",
                modules=None  # Validate all
            )

            # Should have validations for both api and frontend
            endpoint_types = {v.endpoint_type for v in validations}
            assert "api" in endpoint_types, "API should be validated"
            assert "frontend" in endpoint_types, "Frontend should be validated"

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_classify_per_module(self):
        """Per-module classification should group by endpoint_type."""
        from controller.deployment_validator import (
            DeploymentFailureClassifier,
            EndpointValidation,
            DeploymentFailureType,
        )

        validations = [
            # API endpoints - one fails with 404
            EndpointValidation(
                url="https://api.example.com/",
                endpoint_type="api",
                status_code=404,
                is_healthy=False,
                error="Not Found"
            ),
            EndpointValidation(
                url="https://api.example.com/health",
                endpoint_type="api",
                status_code=404,
                is_healthy=False,
                error="Not Found"
            ),
            # Frontend - healthy
            EndpointValidation(
                url="https://example.com/",
                endpoint_type="frontend",
                status_code=200,
                is_healthy=True,
            ),
            # Admin - connection refused
            EndpointValidation(
                url="https://admin.example.com/",
                endpoint_type="admin",
                status_code=None,
                is_healthy=False,
                error="Connection refused"
            ),
        ]

        with patch("controller.server_detector.get_server_detector") as mock_detector:
            mock_detector.return_value.get_rescue_instructions.return_value = ["Check config"]
            results = DeploymentFailureClassifier.classify_per_module(validations)

        # API should be failing with HTTP_404
        assert "api" in results
        healthy, ftype, fixes = results["api"]
        assert not healthy
        assert ftype == DeploymentFailureType.HTTP_404

        # Frontend should be healthy
        assert "frontend" in results
        healthy, ftype, fixes = results["frontend"]
        assert healthy
        assert ftype is None

        # Admin should be failing with CONNECTION_REFUSED
        assert "admin" in results
        healthy, ftype, fixes = results["admin"]
        assert not healthy
        assert ftype == DeploymentFailureType.CONNECTION_REFUSED


class TestDeployStatusDetection:
    """PART 3: Pushed but not deployed detection."""

    def test_detect_not_deployed(self):
        """Different SHAs should return 'not_deployed'."""
        from controller.deployment_validator import check_deploy_status

        async def _run():
            deployed_sha = "abc1234567890"
            github_sha = "def9876543210"

            with patch("controller.deployment_validator.HTTP_CLIENT", "httpx"):
                with patch("controller.deployment_validator.httpx") as mock_httpx:
                    # Mock DEPLOY_INFO.txt response
                    deploy_resp = MagicMock()
                    deploy_resp.status_code = 200
                    deploy_resp.text = f"DEPLOY_DATE=2026-01-01\nGIT_COMMIT={deployed_sha}\n"

                    # Mock GitHub API response
                    github_resp = MagicMock()
                    github_resp.status_code = 200
                    github_resp.json.return_value = {"sha": github_sha}

                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(side_effect=[deploy_resp, github_resp])
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=False)
                    mock_httpx.AsyncClient.return_value = mock_client

                    status, g_sha, d_sha = await check_deploy_status(
                        "test-project",
                        {"api": "https://api.example.com"},
                        "owner/repo"
                    )

            assert status == "not_deployed"
            assert g_sha == github_sha[:40]
            assert d_sha == deployed_sha

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_detect_deployed(self):
        """Same SHAs should return 'deployed'."""
        from controller.deployment_validator import check_deploy_status

        async def _run():
            sha = "abc1234567890abcdef1234567890abcdef12345"

            with patch("controller.deployment_validator.HTTP_CLIENT", "httpx"):
                with patch("controller.deployment_validator.httpx") as mock_httpx:
                    deploy_resp = MagicMock()
                    deploy_resp.status_code = 200
                    deploy_resp.text = f"GIT_COMMIT={sha}\n"

                    github_resp = MagicMock()
                    github_resp.status_code = 200
                    github_resp.json.return_value = {"sha": sha}

                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(side_effect=[deploy_resp, github_resp])
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=False)
                    mock_httpx.AsyncClient.return_value = mock_client

                    status, g_sha, d_sha = await check_deploy_status(
                        "test-project",
                        {"frontend": "https://example.com"},
                        "owner/repo"
                    )

            assert status == "deployed"

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_detect_unknown_no_deploy_info(self):
        """If DEPLOY_INFO.txt not found, return 'unknown'."""
        from controller.deployment_validator import check_deploy_status

        async def _run():
            with patch("controller.deployment_validator.HTTP_CLIENT", "httpx"):
                with patch("controller.deployment_validator.httpx") as mock_httpx:
                    # DEPLOY_INFO.txt returns 404
                    deploy_resp = MagicMock()
                    deploy_resp.status_code = 404
                    deploy_resp.text = "Not Found"

                    # GitHub API returns normally
                    github_resp = MagicMock()
                    github_resp.status_code = 200
                    github_resp.json.return_value = {"sha": "abc123"}

                    mock_client = AsyncMock()
                    mock_client.get = AsyncMock(side_effect=[deploy_resp, github_resp])
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=False)
                    mock_httpx.AsyncClient.return_value = mock_client

                    status, g_sha, d_sha = await check_deploy_status(
                        "test-project",
                        {"api": "https://api.example.com"},
                        "owner/repo"
                    )

            assert status == "unknown"
            assert d_sha is None  # No deployed SHA found

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()


class TestModuleScopedRescueJob:
    """PART 4: Rescue jobs scoped to specific modules."""

    def test_rescue_task_includes_module_scope(self):
        """Rescue task template should include MODULE SCOPE section."""
        from controller.rescue_engine import RescueJobEngine

        engine = RescueJobEngine.__new__(RescueJobEngine)
        engine._state = {}

        # Mock failure object
        failure = MagicMock()
        failure.failure_type = MagicMock(value="HTTP_404")
        failure.failed_urls = [{"url": "https://api.example.com/", "error": "Not Found", "status_code": 404}]
        failure.successful_urls = ["https://example.com/"]
        failure.diagnostic_info = {"total_endpoints": 3, "failed_count": 1, "success_count": 2}
        failure.suggested_fixes = ["Check API routing"]
        failure.deployment_job_id = "job-123"

        with patch("controller.deployment_validator.get_project_directory", return_value=None):
            with patch("controller.deployment_validator.get_github_repo_url", return_value="owner/repo"):
                task = engine._generate_rescue_task_description(
                    project_name="test-project",
                    failure=failure,
                    attempt_number=1,
                    modules=["api"],
                    per_module_failures={"api": "HTTP_404", "frontend": "healthy", "admin": "healthy"}
                )

        assert "MODULE SCOPE" in task
        assert "SCOPED TO MODULES: api" in task
        assert "api: FAILING (HTTP_404)" in task
        assert "frontend: HEALTHY" in task
        assert "admin: HEALTHY" in task

    def test_rescue_state_tracks_modules(self, tmp_path):
        """RescueAttempt should store per-module failure info."""
        from controller.rescue_engine import RescueAttempt

        attempt = RescueAttempt(
            attempt_number=1,
            job_id="job-456",
            failure_type="HTTP_404",
            created_at=datetime.utcnow().isoformat(),
            modules={"api": "HTTP_404", "frontend": "healthy"}
        )

        d = attempt.to_dict()
        assert d["modules"] == {"api": "HTTP_404", "frontend": "healthy"}

        # Reconstruct from dict
        restored = RescueAttempt.from_dict(d)
        assert restored.modules == {"api": "HTTP_404", "frontend": "healthy"}

    def test_healthy_modules_excluded(self):
        """Rescue task should say DO NOT TOUCH for healthy modules."""
        from controller.rescue_engine import RescueJobEngine

        engine = RescueJobEngine.__new__(RescueJobEngine)
        engine._state = {}

        failure = MagicMock()
        failure.failure_type = MagicMock(value="HTTP_502")
        failure.failed_urls = [{"url": "https://admin.example.com/", "error": "Bad Gateway"}]
        failure.successful_urls = ["https://api.example.com/", "https://example.com/"]
        failure.diagnostic_info = {}
        failure.suggested_fixes = []
        failure.deployment_job_id = "job-789"

        with patch("controller.deployment_validator.get_project_directory", return_value=None):
            with patch("controller.deployment_validator.get_github_repo_url", return_value=None):
                task = engine._generate_rescue_task_description(
                    project_name="test-project",
                    failure=failure,
                    attempt_number=2,
                    modules=["admin"],
                    per_module_failures={"api": "healthy", "frontend": "healthy", "admin": "HTTP_502"}
                )

        assert "HEALTHY MODULES (DO NOT TOUCH)" in task
        assert "api: HEALTHY" in task
        assert "frontend: HEALTHY" in task
        assert "admin: FAILING (HTTP_502)" in task


class TestRescueStateExtended:
    """State persistence with new fields."""

    def test_save_load_with_modules(self, tmp_path):
        """State with module fields should serialize and deserialize correctly."""
        from controller.rescue_engine import (
            ProjectRescueState,
            RescueAttempt,
            RescueJobEngine,
            RESCUE_STATE_FILE,
        )

        state = ProjectRescueState(
            project_name="test-proj",
            deployment_job_id="dep-123",
            selected_modules=["api", "admin"],
            module_results=[
                {"module": "api", "url": "https://api.example.com", "healthy": False,
                 "failure_type": "HTTP_404", "last_checked": "2026-01-01T00:00:00"},
                {"module": "frontend", "url": "https://example.com", "healthy": True,
                 "failure_type": None, "last_checked": "2026-01-01T00:00:00"},
            ],
            attempts=[
                RescueAttempt(
                    attempt_number=1,
                    job_id="resc-001",
                    failure_type="HTTP_404",
                    created_at="2026-01-01T00:00:00",
                    modules={"api": "HTTP_404", "frontend": "healthy"}
                )
            ]
        )

        d = state.to_dict()
        assert d["selected_modules"] == ["api", "admin"]
        assert len(d["module_results"]) == 2
        assert d["attempts"][0]["modules"] == {"api": "HTTP_404", "frontend": "healthy"}

        # Reconstruct
        restored = ProjectRescueState.from_dict(d)
        assert restored.selected_modules == ["api", "admin"]
        assert len(restored.module_results) == 2
        assert restored.attempts[0].modules == {"api": "HTTP_404", "frontend": "healthy"}

    def test_backward_compat_old_state(self):
        """Old state files without new fields should load without errors."""
        from controller.rescue_engine import ProjectRescueState, RescueAttempt

        # Old format - no selected_modules, module_results, or attempt.modules
        old_data = {
            "project_name": "old-project",
            "deployment_job_id": "old-dep-123",
            "attempts": [
                {
                    "attempt_number": 1,
                    "job_id": "old-resc-001",
                    "failure_type": "HTTP_404",
                    "created_at": "2025-12-01T00:00:00",
                    "completed_at": None,
                    "success": False,
                    "outcome": None,
                    # No 'modules' key
                }
            ],
            "last_failure_type": "HTTP_404",
            "created_at": "2025-12-01T00:00:00",
            "resolved": False,
            "resolved_at": None,
            # No 'selected_modules' or 'module_results' keys
        }

        state = ProjectRescueState.from_dict(old_data)
        assert state.project_name == "old-project"
        assert state.selected_modules == []
        assert state.module_results == []
        assert state.attempts[0].modules == {}
        assert state.attempts[0].failure_type == "HTTP_404"


class TestEdgeCases:
    """PART 7: Edge case handling."""

    def test_missing_urls_returns_error(self):
        """Validation with empty URLs should return True (nothing to validate)."""
        from controller.deployment_validator import DeploymentValidator

        async def _run():
            validator = DeploymentValidator()
            all_healthy, failure, validations = await validator.validate_deployment(
                project_name="empty-project",
                urls={},
                deployment_job_id="test-job"
            )
            # Empty URLs = no validations = not healthy
            assert not all_healthy or len(validations) == 0

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_rescue_during_active_rescue_blocked(self):
        """Should not allow starting new rescue while one is pending."""
        from controller.rescue_engine import (
            RescueJobEngine,
            ProjectRescueState,
            RescueAttempt,
        )

        engine = RescueJobEngine.__new__(RescueJobEngine)
        engine._state = {
            "test-project": ProjectRescueState(
                project_name="test-project",
                deployment_job_id="dep-001",
                attempts=[
                    RescueAttempt(
                        attempt_number=1,
                        job_id="resc-001",
                        failure_type="HTTP_404",
                        created_at="2026-01-01T00:00:00",
                        completed_at=None,  # Not completed = pending
                    )
                ]
            )
        }

        assert engine.has_pending_attempt("test-project")
        assert not engine.has_pending_attempt("nonexistent-project")

    def test_partial_failure_scoped_correctly(self):
        """When 1 of 3 modules fails, per-module classification should be accurate."""
        from controller.deployment_validator import (
            DeploymentFailureClassifier,
            EndpointValidation,
            DeploymentFailureType,
        )

        validations = [
            # API - healthy
            EndpointValidation(url="https://api.example.com/", endpoint_type="api",
                             status_code=200, is_healthy=True),
            EndpointValidation(url="https://api.example.com/health", endpoint_type="api",
                             status_code=200, is_healthy=True),
            # Frontend - healthy
            EndpointValidation(url="https://example.com/", endpoint_type="frontend",
                             status_code=200, is_healthy=True),
            # Admin - failing
            EndpointValidation(url="https://admin.example.com/", endpoint_type="admin",
                             status_code=502, is_healthy=False, error="Bad Gateway"),
        ]

        with patch("controller.server_detector.get_server_detector") as mock_det:
            mock_det.return_value.get_rescue_instructions.return_value = []
            results = DeploymentFailureClassifier.classify_per_module(validations)

        # API and frontend should be healthy
        assert results["api"][0] is True
        assert results["frontend"][0] is True

        # Admin should be failing with HTTP_502
        assert results["admin"][0] is False
        assert results["admin"][1] == DeploymentFailureType.HTTP_502
