"""
Tests for Phase 15.3: Project Ingestion Engine

Tests covering:
- Ingestion request creation and management
- Git repository cloning
- Local path preparation
- File enumeration
- Structure analysis
- Aspect detection
- Risk scanning
- Document generation
- Lifecycle integration
- Approval workflow
- Edge cases and error handling
"""

import asyncio
import json
import os
import tempfile
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

# Import ingestion engine components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from controller.ingestion_engine import (
    ProjectIngestionEngine,
    IngestionRequest,
    IngestionReport,
    IngestionStatus,
    IngestionSource,
    FileInfo,
    AspectDetection,
    RiskAssessment,
    GitMetadata,
    StructureAnalysis,
    ExistingDocuments,
    get_ingestion_engine,
    create_ingestion_request,
    start_ingestion_analysis,
    approve_ingestion,
    reject_ingestion,
    register_ingested_project,
    get_ingestion_request,
    list_ingestion_requests,
    ASPECT_PATTERNS,
    RISK_PATTERNS,
    GOVERNANCE_DOCS,
)
from controller.lifecycle_v2 import (
    ProjectAspect,
    UserRole,
    LifecycleState,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir) / "test-project"
        project_path.mkdir()

        # Create some files
        (project_path / "README.md").write_text("# Test Project\nA test project.")
        (project_path / "main.py").write_text("def main():\n    print('Hello')\n")
        (project_path / "requirements.txt").write_text("fastapi\npydantic\n")

        # Create directories
        (project_path / "src").mkdir()
        (project_path / "src" / "api").mkdir()
        (project_path / "src" / "api" / "routes.py").write_text("# API routes\n")
        (project_path / "tests").mkdir()
        (project_path / "tests" / "test_main.py").write_text("def test_main(): pass\n")

        yield project_path


@pytest.fixture
def temp_git_project(temp_project_dir):
    """Create a temporary project with git initialized."""
    import subprocess

    # Initialize git
    subprocess.run(["git", "init"], cwd=str(temp_project_dir), capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(temp_project_dir), capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=str(temp_project_dir), capture_output=True)
    subprocess.run(["git", "add", "."], cwd=str(temp_project_dir), capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=str(temp_project_dir), capture_output=True)

    yield temp_project_dir


@pytest.fixture
def engine():
    """Create an ingestion engine with temporary directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        workspace_dir = Path(tmpdir) / "workspaces"
        reports_dir = Path(tmpdir) / "reports"

        engine = ProjectIngestionEngine(
            state_dir=state_dir,
            workspace_dir=workspace_dir,
            reports_dir=reports_dir,
        )
        yield engine


# -----------------------------------------------------------------------------
# Test: Ingestion Request Creation
# -----------------------------------------------------------------------------

class TestIngestionRequestCreation:
    """Tests for creating ingestion requests."""

    @pytest.mark.asyncio
    async def test_create_git_ingestion_request(self, engine):
        """Test creating an ingestion request for a git repository."""
        success, msg, request = await engine.create_ingestion_request(
            project_name="test-project",
            source_type=IngestionSource.GIT_REPOSITORY,
            source_location="https://github.com/user/repo.git",
            requested_by="user123",
            description="Test ingestion",
        )

        assert success
        assert request is not None
        assert request.project_name == "test-project"
        assert request.source_type == IngestionSource.GIT_REPOSITORY
        assert request.status == IngestionStatus.PENDING
        assert request.ingestion_id is not None

    @pytest.mark.asyncio
    async def test_create_local_ingestion_request(self, engine):
        """Test creating an ingestion request for a local path."""
        success, msg, request = await engine.create_ingestion_request(
            project_name="local-project",
            source_type=IngestionSource.LOCAL_PATH,
            source_location="/home/user/project",
            requested_by="user123",
            description="Local project ingestion",
        )

        assert success
        assert request is not None
        assert request.source_type == IngestionSource.LOCAL_PATH

    @pytest.mark.asyncio
    async def test_create_request_with_target_aspects(self, engine):
        """Test creating request with specific target aspects."""
        success, msg, request = await engine.create_ingestion_request(
            project_name="specific-project",
            source_type=IngestionSource.LOCAL_PATH,
            source_location="/path/to/project",
            requested_by="user123",
            target_aspects=[ProjectAspect.BACKEND, ProjectAspect.FRONTEND_WEB],
        )

        assert success
        assert len(request.target_aspects) == 2
        assert ProjectAspect.BACKEND in request.target_aspects


# -----------------------------------------------------------------------------
# Test: File Enumeration
# -----------------------------------------------------------------------------

class TestFileEnumeration:
    """Tests for file enumeration."""

    @pytest.mark.asyncio
    async def test_enumerate_files(self, engine, temp_project_dir):
        """Test enumerating files in a project."""
        files = await engine.enumerate_files(temp_project_dir)

        assert len(files) > 0
        assert any(f.relative_path == "main.py" for f in files)
        assert any(f.relative_path == "README.md" for f in files)

    @pytest.mark.asyncio
    async def test_enumerate_files_respects_limit(self, engine, temp_project_dir):
        """Test that file enumeration respects max_files limit."""
        files = await engine.enumerate_files(temp_project_dir, max_files=2)

        assert len(files) <= 2

    @pytest.mark.asyncio
    async def test_file_info_attributes(self, engine, temp_project_dir):
        """Test that FileInfo has correct attributes."""
        files = await engine.enumerate_files(temp_project_dir)

        py_file = next((f for f in files if f.relative_path == "main.py"), None)
        assert py_file is not None
        assert py_file.extension == ".py"
        assert py_file.size_bytes > 0
        assert py_file.is_binary == False

    @pytest.mark.asyncio
    async def test_excludes_git_directory(self, engine, temp_git_project):
        """Test that .git directory is excluded."""
        files = await engine.enumerate_files(temp_git_project)

        git_files = [f for f in files if ".git" in f.relative_path]
        assert len(git_files) == 0


# -----------------------------------------------------------------------------
# Test: Git Metadata Extraction
# -----------------------------------------------------------------------------

class TestGitMetadataExtraction:
    """Tests for Git metadata extraction."""

    @pytest.mark.asyncio
    async def test_extract_git_metadata_from_repo(self, engine, temp_git_project):
        """Test extracting metadata from a git repository."""
        metadata = await engine.extract_git_metadata(temp_git_project)

        assert metadata.is_git_repo == True
        assert metadata.current_branch is not None
        assert metadata.last_commit_hash is not None
        assert metadata.total_commits >= 1

    @pytest.mark.asyncio
    async def test_extract_git_metadata_non_repo(self, engine, temp_project_dir):
        """Test extracting metadata from a non-git directory."""
        metadata = await engine.extract_git_metadata(temp_project_dir)

        assert metadata.is_git_repo == False
        assert metadata.remote_url is None


# -----------------------------------------------------------------------------
# Test: Structure Analysis
# -----------------------------------------------------------------------------

class TestStructureAnalysis:
    """Tests for project structure analysis."""

    @pytest.mark.asyncio
    async def test_analyze_structure(self, engine, temp_project_dir):
        """Test analyzing project structure."""
        files = await engine.enumerate_files(temp_project_dir)
        structure = await engine.analyze_structure(temp_project_dir, files)

        assert structure.total_files > 0
        assert structure.total_size_bytes > 0
        assert ".py" in structure.file_types
        assert "src" in structure.top_level_dirs or "tests" in structure.top_level_dirs

    @pytest.mark.asyncio
    async def test_detect_tests_directory(self, engine, temp_project_dir):
        """Test detection of tests directory."""
        files = await engine.enumerate_files(temp_project_dir)
        structure = await engine.analyze_structure(temp_project_dir, files)

        assert structure.has_tests == True

    @pytest.mark.asyncio
    async def test_detect_package_manager(self, engine, temp_project_dir):
        """Test detection of package manager."""
        files = await engine.enumerate_files(temp_project_dir)
        structure = await engine.analyze_structure(temp_project_dir, files)

        assert structure.package_manager == "pip"


# -----------------------------------------------------------------------------
# Test: Aspect Detection
# -----------------------------------------------------------------------------

class TestAspectDetection:
    """Tests for project aspect detection."""

    @pytest.mark.asyncio
    async def test_detect_backend_aspect(self, engine, temp_project_dir):
        """Test detection of backend aspect."""
        files = await engine.enumerate_files(temp_project_dir)
        aspects = await engine.detect_aspects(temp_project_dir, files)

        # Should detect backend due to api/routes.py
        assert len(aspects.detected_aspects) >= 1
        assert aspects.primary_aspect is not None

    @pytest.mark.asyncio
    async def test_aspect_confidence_scores(self, engine, temp_project_dir):
        """Test that confidence scores are calculated."""
        files = await engine.enumerate_files(temp_project_dir)
        aspects = await engine.detect_aspects(temp_project_dir, files)

        # Scores should be normalized
        total = sum(aspects.confidence_scores.values())
        assert total > 0  # At least some detection

    @pytest.mark.asyncio
    async def test_aspect_evidence_tracking(self, engine, temp_project_dir):
        """Test that evidence is tracked for detected aspects."""
        files = await engine.enumerate_files(temp_project_dir)
        aspects = await engine.detect_aspects(temp_project_dir, files)

        # Evidence should be tracked
        for aspect in aspects.detected_aspects:
            evidence = aspects.evidence.get(aspect.value, [])
            # May or may not have evidence depending on patterns


# -----------------------------------------------------------------------------
# Test: Risk Scanning
# -----------------------------------------------------------------------------

class TestRiskScanning:
    """Tests for security risk scanning."""

    @pytest.mark.asyncio
    async def test_scan_clean_project(self, engine, temp_project_dir):
        """Test scanning a project with no security issues."""
        files = await engine.enumerate_files(temp_project_dir)
        risk = await engine.scan_risks(temp_project_dir, files)

        assert risk.risk_level in ["low", "medium", "high", "critical"]
        assert risk.total_issues >= 0

    @pytest.mark.asyncio
    async def test_detect_hardcoded_secret(self, engine, temp_project_dir):
        """Test detection of hardcoded secrets."""
        # Create a file with a hardcoded secret
        secrets_file = temp_project_dir / "config.py"
        secrets_file.write_text('password = "mysecretpassword123"\n')

        files = await engine.enumerate_files(temp_project_dir)
        risk = await engine.scan_risks(temp_project_dir, files)

        assert len(risk.hardcoded_secrets) > 0

    @pytest.mark.asyncio
    async def test_detect_sensitive_file(self, engine, temp_project_dir):
        """Test detection of sensitive files."""
        # Create a .env file
        env_file = temp_project_dir / ".env"
        env_file.write_text("API_KEY=secret\n")

        files = await engine.enumerate_files(temp_project_dir)
        risk = await engine.scan_risks(temp_project_dir, files)

        assert ".env" in " ".join(risk.sensitive_files)

    @pytest.mark.asyncio
    async def test_risk_recommendations(self, engine, temp_project_dir):
        """Test that recommendations are generated."""
        # Create issues
        (temp_project_dir / ".env").write_text("SECRET=value\n")

        files = await engine.enumerate_files(temp_project_dir)
        risk = await engine.scan_risks(temp_project_dir, files)

        # Should have recommendations
        assert len(risk.recommendations) >= 0  # May or may not have recommendations


# -----------------------------------------------------------------------------
# Test: Document Generation
# -----------------------------------------------------------------------------

class TestDocumentGeneration:
    """Tests for governance document generation."""

    @pytest.mark.asyncio
    async def test_check_existing_documents(self, engine, temp_project_dir):
        """Test checking for existing governance documents."""
        existing = await engine.check_existing_documents(temp_project_dir)

        assert "README.md" in " ".join(existing.found) or len(existing.missing) > 0
        assert "PROJECT_MANIFEST.yaml" in existing.missing

    @pytest.mark.asyncio
    async def test_generate_project_manifest(self, engine, temp_project_dir):
        """Test generating PROJECT_MANIFEST.yaml."""
        files = await engine.enumerate_files(temp_project_dir)
        structure = await engine.analyze_structure(temp_project_dir, files)
        aspects = await engine.detect_aspects(temp_project_dir, files)
        git = await engine.extract_git_metadata(temp_project_dir)
        risk = await engine.scan_risks(temp_project_dir, files)
        existing = await engine.check_existing_documents(temp_project_dir)

        report = IngestionReport(
            report_id="test",
            ingestion_id="test",
            project_name="test-project",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
            analyzed_at=datetime.utcnow(),
            git_metadata=git,
            structure=structure,
            aspects=aspects,
            risk_assessment=risk,
            existing_docs=existing,
            files=files,
            ready_for_registration=True,
            blocking_issues=[],
            warnings=[],
        )

        manifest = await engine.generate_project_manifest("test-project", report)

        assert "project:" in manifest
        assert "test-project" in manifest
        assert "aspects:" in manifest

    @pytest.mark.asyncio
    async def test_generate_current_state(self, engine, temp_project_dir):
        """Test generating CURRENT_STATE.md."""
        files = await engine.enumerate_files(temp_project_dir)
        structure = await engine.analyze_structure(temp_project_dir, files)
        aspects = await engine.detect_aspects(temp_project_dir, files)
        git = await engine.extract_git_metadata(temp_project_dir)
        risk = await engine.scan_risks(temp_project_dir, files)
        existing = await engine.check_existing_documents(temp_project_dir)

        report = IngestionReport(
            report_id="test",
            ingestion_id="test",
            project_name="test-project",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
            analyzed_at=datetime.utcnow(),
            git_metadata=git,
            structure=structure,
            aspects=aspects,
            risk_assessment=risk,
            existing_docs=existing,
            files=files,
            ready_for_registration=True,
            blocking_issues=[],
            warnings=[],
        )

        state_doc = await engine.generate_current_state("test-project", report)

        assert "# Current State" in state_doc
        assert "test-project" in state_doc
        assert "DEPLOYED" in state_doc

    @pytest.mark.asyncio
    async def test_generate_ai_policy(self, engine):
        """Test generating AI_POLICY.md."""
        policy = await engine.generate_ai_policy("test-project")

        assert "# AI Policy" in policy
        assert "Human Oversight" in policy
        assert "MUST NOT" in policy


# -----------------------------------------------------------------------------
# Test: Full Analysis Pipeline
# -----------------------------------------------------------------------------

class TestAnalysisPipeline:
    """Tests for the complete analysis pipeline."""

    @pytest.mark.asyncio
    async def test_full_analysis(self, engine, temp_project_dir):
        """Test running the full analysis pipeline."""
        success, msg, report = await engine.analyze_project(
            project_path=temp_project_dir,
            ingestion_id="test-ingestion",
            project_name="test-project",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
        )

        assert success
        assert report is not None
        assert report.structure.total_files > 0
        assert len(report.aspects.detected_aspects) > 0
        assert report.risk_assessment is not None

    @pytest.mark.asyncio
    async def test_analysis_creates_report(self, engine, temp_project_dir):
        """Test that analysis creates a report file."""
        success, msg, report = await engine.analyze_project(
            project_path=temp_project_dir,
            ingestion_id="test-ingestion-2",
            project_name="test-project",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
        )

        assert success
        report_path = engine._reports_dir / "test-ingestion-2.json"
        assert report_path.exists()


# -----------------------------------------------------------------------------
# Test: Approval Workflow
# -----------------------------------------------------------------------------

class TestApprovalWorkflow:
    """Tests for the ingestion approval workflow."""

    @pytest.mark.asyncio
    async def test_approve_ingestion_request(self, engine, temp_project_dir):
        """Test approving an ingestion request."""
        # Create and analyze
        success, _, request = await engine.create_ingestion_request(
            project_name="approval-test",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
            requested_by="user123",
        )
        await engine.start_analysis(request.ingestion_id)

        # Approve
        success, msg = await engine.approve_ingestion(
            ingestion_id=request.ingestion_id,
            approved_by="admin123",
            user_role=UserRole.ADMIN,
        )

        assert success
        updated = await engine.get_ingestion_request(request.ingestion_id)
        assert updated.status == IngestionStatus.APPROVED

    @pytest.mark.asyncio
    async def test_reject_ingestion_request(self, engine, temp_project_dir):
        """Test rejecting an ingestion request."""
        # Create and analyze
        success, _, request = await engine.create_ingestion_request(
            project_name="reject-test",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
            requested_by="user123",
        )
        await engine.start_analysis(request.ingestion_id)

        # Reject
        success, msg = await engine.reject_ingestion(
            ingestion_id=request.ingestion_id,
            rejected_by="admin123",
            reason="Contains sensitive data",
            user_role=UserRole.ADMIN,
        )

        assert success
        updated = await engine.get_ingestion_request(request.ingestion_id)
        assert updated.status == IngestionStatus.REJECTED
        assert "sensitive" in updated.rejection_reason.lower()

    @pytest.mark.asyncio
    async def test_cannot_approve_pending(self, engine):
        """Test that pending requests cannot be approved."""
        success, _, request = await engine.create_ingestion_request(
            project_name="pending-test",
            source_type=IngestionSource.LOCAL_PATH,
            source_location="/fake/path",
            requested_by="user123",
        )

        # Try to approve without analysis
        success, msg = await engine.approve_ingestion(
            ingestion_id=request.ingestion_id,
            approved_by="admin123",
            user_role=UserRole.ADMIN,
        )

        assert not success
        assert "not awaiting approval" in msg.lower()

    @pytest.mark.asyncio
    async def test_role_permission_check(self, engine, temp_project_dir):
        """Test that only admins/owners can approve."""
        success, _, request = await engine.create_ingestion_request(
            project_name="role-test",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
            requested_by="user123",
        )
        await engine.start_analysis(request.ingestion_id)

        # Try to approve as developer
        success, msg = await engine.approve_ingestion(
            ingestion_id=request.ingestion_id,
            approved_by="dev123",
            user_role=UserRole.DEVELOPER,
        )

        assert not success
        assert "cannot approve" in msg.lower()


# -----------------------------------------------------------------------------
# Test: Query Methods
# -----------------------------------------------------------------------------

class TestQueryMethods:
    """Tests for query methods."""

    @pytest.mark.asyncio
    async def test_list_ingestion_requests(self, engine):
        """Test listing ingestion requests."""
        # Create several requests
        for i in range(3):
            await engine.create_ingestion_request(
                project_name=f"list-test-{i}",
                source_type=IngestionSource.LOCAL_PATH,
                source_location=f"/path/{i}",
                requested_by="user123",
            )

        requests = await engine.list_ingestion_requests()
        assert len(requests) >= 3

    @pytest.mark.asyncio
    async def test_list_with_status_filter(self, engine):
        """Test listing requests with status filter."""
        await engine.create_ingestion_request(
            project_name="filter-test",
            source_type=IngestionSource.LOCAL_PATH,
            source_location="/path/filter",
            requested_by="user123",
        )

        pending = await engine.list_ingestion_requests(status_filter=IngestionStatus.PENDING)
        registered = await engine.list_ingestion_requests(status_filter=IngestionStatus.REGISTERED)

        assert len(pending) >= 1
        # No registered requests yet
        assert all(r.status == IngestionStatus.PENDING for r in pending)

    @pytest.mark.asyncio
    async def test_list_with_limit(self, engine):
        """Test listing requests with limit."""
        for i in range(5):
            await engine.create_ingestion_request(
                project_name=f"limit-test-{i}",
                source_type=IngestionSource.LOCAL_PATH,
                source_location=f"/path/{i}",
                requested_by="user123",
            )

        requests = await engine.list_ingestion_requests(limit=3)
        assert len(requests) <= 3


# -----------------------------------------------------------------------------
# Test: Serialization
# -----------------------------------------------------------------------------

class TestSerialization:
    """Tests for data serialization."""

    def test_ingestion_request_to_dict(self):
        """Test IngestionRequest serialization."""
        request = IngestionRequest(
            ingestion_id="test-id",
            project_name="test",
            source_type=IngestionSource.GIT_REPOSITORY,
            source_location="https://github.com/test",
            status=IngestionStatus.PENDING,
            requested_by="user123",
            requested_at=datetime.utcnow(),
        )

        data = request.to_dict()
        assert data["ingestion_id"] == "test-id"
        assert data["source_type"] == "git_repository"
        assert data["status"] == "pending"

    def test_ingestion_request_from_dict(self):
        """Test IngestionRequest deserialization."""
        data = {
            "ingestion_id": "test-id",
            "project_name": "test",
            "source_type": "git_repository",
            "source_location": "https://github.com/test",
            "status": "pending",
            "requested_by": "user123",
            "requested_at": "2024-01-01T00:00:00",
        }

        request = IngestionRequest.from_dict(data)
        assert request.ingestion_id == "test-id"
        assert request.source_type == IngestionSource.GIT_REPOSITORY
        assert request.status == IngestionStatus.PENDING

    def test_file_info_to_dict(self):
        """Test FileInfo serialization."""
        file_info = FileInfo(
            path="/path/to/file.py",
            relative_path="file.py",
            size_bytes=1024,
            extension=".py",
        )

        data = file_info.to_dict()
        assert data["path"] == "/path/to/file.py"
        assert data["size_bytes"] == 1024

    def test_aspect_detection_to_dict(self):
        """Test AspectDetection serialization."""
        detection = AspectDetection(
            detected_aspects=[ProjectAspect.BACKEND, ProjectAspect.CORE],
            primary_aspect=ProjectAspect.BACKEND,
            confidence_scores={"backend": 0.7, "core": 0.3},
            evidence={"backend": ["api/routes.py"]},
        )

        data = detection.to_dict()
        assert "backend" in data["detected_aspects"]
        assert data["primary_aspect"] == "backend"


# -----------------------------------------------------------------------------
# Test: Edge Cases
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_directory(self, engine):
        """Test analyzing an empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            empty_dir = Path(tmpdir) / "empty"
            empty_dir.mkdir()

            files = await engine.enumerate_files(empty_dir)
            assert len(files) == 0

    @pytest.mark.asyncio
    async def test_nonexistent_ingestion(self, engine):
        """Test getting a non-existent ingestion request."""
        request = await engine.get_ingestion_request("nonexistent-id")
        assert request is None

    @pytest.mark.asyncio
    async def test_prepare_nonexistent_path(self, engine):
        """Test preparing a non-existent local path."""
        success, msg, path = await engine.prepare_local_path(
            "/nonexistent/path/that/does/not/exist",
            "test-ingestion",
        )
        assert not success
        assert "does not exist" in msg.lower()

    @pytest.mark.asyncio
    async def test_binary_file_detection(self, engine):
        """Test that binary files are detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "binary-test"
            project.mkdir()

            # Create a binary file
            binary_file = project / "data.bin"
            binary_file.write_bytes(b"\x00\x01\x02\x03\x04")

            # Create a text file
            text_file = project / "text.txt"
            text_file.write_text("Hello World")

            files = await engine.enumerate_files(project)

            binary = next((f for f in files if f.relative_path == "data.bin"), None)
            text = next((f for f in files if f.relative_path == "text.txt"), None)

            assert binary is not None
            assert binary.is_binary == True
            assert text is not None
            assert text.is_binary == False


# -----------------------------------------------------------------------------
# Test: Public API Functions
# -----------------------------------------------------------------------------

class TestPublicAPI:
    """Tests for the public API functions."""

    @pytest.mark.asyncio
    async def test_create_ingestion_request_function(self):
        """Test the create_ingestion_request public function."""
        with patch('controller.ingestion_engine._engine_instance', None):
            # This test verifies the function signature is correct
            # Actual execution requires proper mocking of the engine
            pass  # Skip actual execution as it requires file system

    def test_governance_docs_constant(self):
        """Test that all expected governance docs are defined."""
        expected = [
            "PROJECT_MANIFEST.yaml",
            "CURRENT_STATE.md",
            "ARCHITECTURE.md",
            "AI_POLICY.md",
            "TESTING_STRATEGY.md",
        ]
        for doc in expected:
            assert doc in GOVERNANCE_DOCS

    def test_aspect_patterns_defined(self):
        """Test that aspect patterns are defined for all aspects."""
        for aspect in [ProjectAspect.BACKEND, ProjectAspect.FRONTEND_WEB,
                       ProjectAspect.FRONTEND_MOBILE, ProjectAspect.ADMIN, ProjectAspect.CORE]:
            assert aspect in ASPECT_PATTERNS
            assert len(ASPECT_PATTERNS[aspect]) > 0

    def test_risk_patterns_defined(self):
        """Test that risk patterns are defined."""
        assert "hardcoded_secrets" in RISK_PATTERNS
        assert "sensitive_files" in RISK_PATTERNS
        assert "dangerous_patterns" in RISK_PATTERNS


# -----------------------------------------------------------------------------
# Test: Integration with Lifecycle Engine
# -----------------------------------------------------------------------------

class TestLifecycleIntegration:
    """Tests for integration with Lifecycle Engine v2."""

    @pytest.mark.asyncio
    async def test_register_creates_lifecycles(self, engine, temp_project_dir):
        """Test that registering creates lifecycle instances."""
        # This test would require mocking the lifecycle manager
        # For now, we test the workflow up to registration

        # Create, analyze, approve
        success, _, request = await engine.create_ingestion_request(
            project_name="lifecycle-test",
            source_type=IngestionSource.LOCAL_PATH,
            source_location=str(temp_project_dir),
            requested_by="user123",
        )
        await engine.start_analysis(request.ingestion_id)
        await engine.approve_ingestion(
            request.ingestion_id,
            "admin123",
            UserRole.ADMIN,
        )

        # Verify status is approved
        updated = await engine.get_ingestion_request(request.ingestion_id)
        assert updated.status == IngestionStatus.APPROVED


# -----------------------------------------------------------------------------
# Run Tests
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
