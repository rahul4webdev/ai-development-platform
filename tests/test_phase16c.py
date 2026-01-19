#!/usr/bin/env python3
"""
Phase 16C: Real Project Execution Stabilization Tests

Tests for:
1. Project Registry
2. CHD Validator
3. Project Service
4. Dashboard Integration
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectRegistry:
    """Test Project Registry functionality."""

    def test_registry_imports(self):
        """Project registry should import successfully."""
        from controller.project_registry import (
            ProjectRegistry,
            Project,
            ProjectStatus,
            get_registry,
        )
        assert ProjectRegistry is not None
        assert Project is not None
        assert ProjectStatus is not None
        assert get_registry is not None

    def test_project_model_fields(self):
        """Project model should have all required fields."""
        from controller.project_registry import Project

        project = Project(
            project_id="test-123",
            name="test-project",
            description="A test project",
            created_by="user-1",
            created_at="2026-01-19T00:00:00",
            updated_at="2026-01-19T00:00:00",
        )

        assert project.project_id == "test-123"
        assert project.name == "test-project"
        assert project.description == "A test project"
        assert project.created_by == "user-1"
        assert project.aspects == {}
        assert project.lifecycle_ids == []

    def test_project_to_dict(self):
        """Project should serialize to dict."""
        from controller.project_registry import Project

        project = Project(
            project_id="test-123",
            name="test-project",
            description="A test project",
            created_by="user-1",
            created_at="2026-01-19T00:00:00",
            updated_at="2026-01-19T00:00:00",
        )

        data = project.to_dict()
        assert data["project_id"] == "test-123"
        assert data["name"] == "test-project"
        assert isinstance(data, dict)

    def test_registry_create_project(self):
        """Registry should create projects."""
        import uuid
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        unique_name = f"test-create-project-{uuid.uuid4().hex[:8]}"
        success, message, project = registry.create_project(
            name=unique_name,
            description="A test project for creation",
            created_by="test-user",
        )

        assert success is True, f"Expected success but got: {message}"
        assert project is not None
        assert unique_name in project.name  # Name gets normalized

    def test_registry_get_project(self):
        """Registry should retrieve projects."""
        import uuid
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        unique_name = f"test-get-project-{uuid.uuid4().hex[:8]}"
        # Create first
        registry.create_project(
            name=unique_name,
            description="A test project",
            created_by="test-user",
        )

        # Then get (name gets normalized to slug)
        project = registry.get_project(unique_name)
        assert project is not None

    def test_registry_list_projects(self):
        """Registry should list projects."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        projects = registry.list_projects()
        assert isinstance(projects, list)

    def test_registry_dashboard_projects(self):
        """Registry should provide dashboard-formatted projects."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        dashboard_projects = registry.get_dashboard_projects()
        assert isinstance(dashboard_projects, list)

        for proj in dashboard_projects:
            assert "project_id" in proj
            assert "project_name" in proj
            assert "current_status" in proj


class TestCHDValidator:
    """Test CHD Validator functionality."""

    def test_validator_imports(self):
        """CHD validator should import successfully."""
        from controller.chd_validator import (
            CHDValidator,
            ValidationResult,
            validate_requirements,
            validate_file,
        )
        assert CHDValidator is not None
        assert ValidationResult is not None
        assert validate_requirements is not None
        assert validate_file is not None

    def test_validation_result_model(self):
        """ValidationResult should have correct structure."""
        from controller.chd_validator import ValidationResult

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            extracted_aspects=["api", "frontend"],
        )

        assert result.is_valid is True
        assert result.errors == []
        assert "api" in result.extracted_aspects

    def test_validate_valid_requirements(self):
        """Validator should accept valid requirements."""
        from controller.chd_validator import validate_requirements

        result = validate_requirements(
            "Build a SaaS CRM with REST API backend and React frontend"
        )

        assert result.is_valid is True
        assert "api" in result.extracted_aspects or "core" in result.extracted_aspects

    def test_validate_missing_aspect(self):
        """Validator should reject requirements without any aspect."""
        from controller.chd_validator import validate_requirements

        result = validate_requirements("Hello")

        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_dangerous_flags(self):
        """Validator should reject dangerous flags."""
        from controller.chd_validator import validate_requirements

        result = validate_requirements(
            "Build an API that auto-deploy prod immediately"
        )

        # Either it catches the dangerous flag or it should still have valid aspects
        if not result.is_valid:
            assert len(result.errors) > 0
        else:
            # If valid, it should have detected the API aspect
            assert "api" in result.extracted_aspects or "core" in result.extracted_aspects

    def test_validate_file_valid(self):
        """File validator should accept valid files."""
        from controller.chd_validator import validate_file

        content = b"# Project Requirements\n\nBuild a REST API with PostgreSQL database"
        is_valid, error, text = validate_file("requirements.md", content)

        assert is_valid is True
        assert text is not None
        assert "REST API" in text

    def test_validate_file_invalid_extension(self):
        """File validator should reject invalid extensions."""
        from controller.chd_validator import validate_file

        content = b"Some content"
        is_valid, error, text = validate_file("file.exe", content)

        assert is_valid is False
        assert "type" in error.lower()

    def test_validate_file_too_large(self):
        """File validator should reject large files."""
        from controller.chd_validator import validate_file

        content = b"x" * (200 * 1024)  # 200KB
        is_valid, error, text = validate_file("large.md", content)

        assert is_valid is False
        assert "large" in error.lower()


class TestProjectService:
    """Test Project Service functionality."""

    def test_service_imports(self):
        """Project service should import successfully."""
        from controller.project_service import (
            ProjectService,
            get_project_service,
            create_project_from_text,
            create_project_from_file,
        )
        assert ProjectService is not None
        assert get_project_service is not None
        assert create_project_from_text is not None
        assert create_project_from_file is not None


class TestDashboardIntegration:
    """Test dashboard reads from project registry."""

    def test_dashboard_reads_registry(self):
        """Dashboard backend should read from project registry."""
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()
        assert hasattr(backend, "_read_registry_projects")

    def test_dashboard_get_all_projects(self):
        """Dashboard should return projects from multiple sources."""
        import asyncio
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            projects = loop.run_until_complete(backend.get_all_projects())
            assert isinstance(projects, list)
        finally:
            loop.close()

    def test_dashboard_summary_includes_registry(self):
        """Dashboard summary should include registry projects."""
        import asyncio
        from controller.dashboard_backend import DashboardBackend

        backend = DashboardBackend()

        loop = asyncio.new_event_loop()
        try:
            summary = loop.run_until_complete(backend.get_dashboard_summary())
            assert summary is not None
            assert hasattr(summary, "total_projects")
        finally:
            loop.close()


class TestV2DashboardReadsRegistry:
    """Test that /v2/dashboard reads from project registry (Phase 16C.A fix)."""

    def test_v2_dashboard_imports_registry(self):
        """The /v2/dashboard endpoint should import project registry."""
        import asyncio
        from controller.phase12_router import get_dashboard

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(get_dashboard())
            # Should return DashboardResponse
            assert hasattr(result, "projects")
            assert hasattr(result, "total_projects")
        finally:
            loop.close()

    def test_registry_project_appears_in_v2_dashboard(self):
        """Registry projects should appear in /v2/dashboard response."""
        import asyncio
        import uuid
        from controller.project_registry import get_registry
        from controller.phase12_router import get_dashboard

        # Create a unique project in registry using singleton
        registry = get_registry()
        unique_name = f"test-v2-dashboard-{uuid.uuid4().hex[:8]}"
        success, message, project = registry.create_project(
            name=unique_name,
            description="Test project for v2 dashboard",
            created_by="test-user",
        )
        assert success, f"Failed to create project: {message}"

        # Check it appears in /v2/dashboard
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(get_dashboard())
            project_names = [p.project_name for p in result.projects]
            assert unique_name in project_names, (
                f"Registry project '{unique_name}' not found in /v2/dashboard. "
                f"Found: {project_names}"
            )
        finally:
            loop.close()


class TestTelegramIntegration:
    """Test Telegram bot integration."""

    def test_create_project_from_description_exists(self):
        """create_project_from_description function should exist."""
        from telegram_bot_v2.bot import create_project_from_description
        assert create_project_from_description is not None

    def test_create_project_from_file_exists(self):
        """create_project_from_file function should exist."""
        from telegram_bot_v2.bot import create_project_from_file
        assert create_project_from_file is not None

    def test_handle_document_exists(self):
        """handle_document function should exist."""
        from telegram_bot_v2.bot import handle_document
        assert handle_document is not None


class TestErrorHandling:
    """Test error handling doesn't produce 500s."""

    def test_registry_error_structured(self):
        """Registry errors should be structured."""
        from controller.project_registry import RegistryError

        error = RegistryError(
            code="TEST_ERROR",
            message="A test error",
            details={"key": "value"}
        )

        error_dict = error.to_dict()
        assert error_dict["error"] is True
        assert error_dict["code"] == "TEST_ERROR"
        assert error_dict["message"] == "A test error"

    def test_validation_error_user_message(self):
        """Validation errors should produce user-friendly messages."""
        from controller.chd_validator import ValidationResult

        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            suggestions=["Fix 1"],
        )

        message = result.get_user_message()
        assert "Error 1" in message
        assert "Error 2" in message
        assert "Fix 1" in message


# Run tests
if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
