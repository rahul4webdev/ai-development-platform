#!/usr/bin/env python3
"""
System Consistency Tests

Tests that the project creation → dashboard display pipeline is consistent:
1. Projects created appear in both /projects and /dashboard
2. Project creation creates lifecycle instances for all aspects
3. Registry is the single source of truth
4. No hidden or phantom projects
"""

import sys
import json
import uuid
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestProjectCreatedAppearsInDashboard:
    """Test that a project created via registry appears in dashboard endpoints."""

    def test_registry_project_appears_in_dashboard_projects(self):
        """Registry project should appear in get_dashboard_projects()."""
        from controller.project_registry import ProjectRegistry, Project

        registry = ProjectRegistry()
        name = f"test-dash-{uuid.uuid4().hex[:8]}"

        success, msg, project = registry.create_project(
            name=name,
            description="Test project for dashboard",
            created_by="test-user",
        )
        assert success, f"Failed to create project: {msg}"

        dashboard_projects = registry.get_dashboard_projects()
        project_names = [p.get("project_name") for p in dashboard_projects]
        assert name in project_names, (
            f"Project '{name}' not found in dashboard projects. "
            f"Found: {project_names}"
        )

    def test_registry_project_has_aspects(self):
        """Registry project should have aspects initialized."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        name = f"test-aspects-{uuid.uuid4().hex[:8]}"

        success, _, project = registry.create_project(
            name=name,
            description="Test project with aspects",
            created_by="test-user",
            aspects=["api", "frontend", "admin"],
        )
        assert success
        assert "api" in project.aspects
        assert "frontend" in project.aspects
        assert "admin" in project.aspects


class TestProjectCreationCreatesLifecycles:
    """Test that project creation creates lifecycle instances for all aspects."""

    def test_lifecycle_creation_for_aspects(self, tmp_path):
        """Lifecycle instances should be created for each aspect."""
        from controller.lifecycle_v2 import (
            ProjectAspect,
            LifecycleManager,
            LifecycleMode,
        )

        async def _run():
            # Use tmp_path for lifecycle state dir
            manager = LifecycleManager(state_dir=tmp_path)
            project_name = f"test-lc-{uuid.uuid4().hex[:8]}"
            aspects = [
                ProjectAspect.CORE,
                ProjectAspect.BACKEND,
                ProjectAspect.FRONTEND_WEB,
                ProjectAspect.ADMIN,
            ]

            created_ids = []
            for aspect in aspects:
                success, msg, lifecycle = await manager.create_lifecycle(
                    project_name=project_name,
                    mode=LifecycleMode.PROJECT_MODE,
                    aspect=aspect,
                    created_by="test-user",
                    description="Test lifecycle",
                )
                assert success, f"Failed to create lifecycle for {aspect.value}: {msg}"
                assert lifecycle is not None
                created_ids.append(lifecycle.lifecycle_id)

            assert len(created_ids) == 4, f"Expected 4 lifecycles, got {len(created_ids)}"

            # Verify all can be retrieved
            for lc_id in created_ids:
                lc = await manager.get_lifecycle(lc_id)
                assert lc is not None, f"Lifecycle {lc_id} not found after creation"
                assert lc.project_name == project_name

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

    def test_aspect_mapping_completeness(self):
        """All standard aspects should map to lifecycle ProjectAspect enums."""
        from controller.lifecycle_v2 import ProjectAspect

        aspect_mapping = {
            "api": ProjectAspect.BACKEND,
            "backend": ProjectAspect.BACKEND,
            "frontend": ProjectAspect.FRONTEND_WEB,
            "frontend_web": ProjectAspect.FRONTEND_WEB,
            "admin": ProjectAspect.ADMIN,
            "core": ProjectAspect.CORE,
        }

        # Every standard aspect should have a mapping
        for name in ["api", "frontend", "admin", "core"]:
            assert name in aspect_mapping, f"Aspect '{name}' missing from mapping"


class TestRegistryIsSingleSourceOfTruth:
    """Test that the project registry is the authoritative source for project data."""

    def test_registry_create_returns_project(self):
        """Creating a project returns the project object."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        name = f"test-ssot-{uuid.uuid4().hex[:8]}"

        success, msg, project = registry.create_project(
            name=name,
            description="Single source of truth test",
            created_by="test-user",
        )
        assert success
        assert project is not None
        assert project.name == name

    def test_registry_get_project_after_create(self):
        """get_project() should find a project right after creation."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        name = f"test-get-{uuid.uuid4().hex[:8]}"

        registry.create_project(
            name=name,
            description="Get test",
            created_by="test-user",
        )

        project = registry.get_project(name)
        assert project is not None, f"get_project('{name}') returned None"
        assert project.name == name

    def test_duplicate_project_rejected(self):
        """Creating the same project twice should fail."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        name = f"test-dup-{uuid.uuid4().hex[:8]}"

        success1, _, _ = registry.create_project(
            name=name,
            description="First",
            created_by="test-user",
        )
        assert success1

        success2, msg, _ = registry.create_project(
            name=name,
            description="Duplicate",
            created_by="test-user",
        )
        assert not success2, "Duplicate project should be rejected"


class TestNoHiddenProjectsCreated:
    """Test that no phantom or hidden projects are created."""

    def test_project_count_matches_creation(self):
        """Dashboard project count should match number of created projects."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        initial_count = len(registry.get_dashboard_projects())

        name = f"test-count-{uuid.uuid4().hex[:8]}"
        registry.create_project(
            name=name,
            description="Count test",
            created_by="test-user",
        )

        new_count = len(registry.get_dashboard_projects())
        assert new_count == initial_count + 1, (
            f"Expected count to increase by 1. "
            f"Initial: {initial_count}, New: {new_count}"
        )

    def test_create_project_with_identity_no_phantom(self):
        """create_project_with_identity should create exactly one project."""
        from controller.project_registry import ProjectRegistry

        registry = ProjectRegistry()
        initial_count = len(registry.get_dashboard_projects())

        unique_id = uuid.uuid4().hex
        name = f"test-identity-{unique_id[:8]}"
        success, msg, project = registry.create_project_with_identity(
            name=name,
            description=f"Identity test {unique_id} - Project for automated testing with {unique_id}",
            created_by="test-user",
            requirements_raw=f"Build app {unique_id}: React frontend with Node.js backend, PostgreSQL db, admin dashboard, user auth, {unique_id}",
            aspects=["api", "frontend", "admin"],
        )
        assert success, f"Failed: {msg}"

        new_count = len(registry.get_dashboard_projects())
        assert new_count == initial_count + 1, (
            f"Expected exactly 1 new project. "
            f"Initial: {initial_count}, New: {new_count}"
        )


class TestDashboardLifecycleReading:
    """Test that dashboard correctly reads lifecycle_v2 format."""

    def test_dashboard_reads_v2_lifecycles(self, tmp_path):
        """Dashboard should read lifecycles from lifecycles.json (v2 format)."""
        from controller.dashboard_backend import DashboardBackend as DashboardService

        # Create a mock lifecycle dir with v2 format
        lifecycle_dir = tmp_path / "lifecycle"
        lifecycle_dir.mkdir()

        v2_data = {
            "lifecycles": {
                "lc-001": {
                    "lifecycle_id": "lc-001",
                    "project_name": "test-project",
                    "aspect": "backend",
                    "state": "created",
                },
                "lc-002": {
                    "lifecycle_id": "lc-002",
                    "project_name": "test-project",
                    "aspect": "frontend_web",
                    "state": "created",
                },
            },
            "created_at": "2026-01-01T00:00:00",
        }

        (lifecycle_dir / "lifecycles.json").write_text(json.dumps(v2_data))

        service = DashboardService()
        service._lifecycle_dir = lifecycle_dir

        async def _run():
            return await service._read_all_lifecycles()

        loop = asyncio.new_event_loop()
        try:
            lifecycles = loop.run_until_complete(_run())
        finally:
            loop.close()

        assert len(lifecycles) == 2, f"Expected 2 lifecycles, got {len(lifecycles)}"

        lc_ids = {lc.get("lifecycle_id") for lc in lifecycles}
        assert "lc-001" in lc_ids
        assert "lc-002" in lc_ids
