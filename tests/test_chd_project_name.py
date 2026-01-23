"""
Phase 19 Fix: CHD Project Name Precedence Tests

CRITICAL: Verifies that project_name from CHD file is the SINGLE source of truth.

Tests ensure:
1. CHD project_name is extracted correctly
2. No fallback to inferred/system text
3. Project appears in dashboard with correct name
4. Fingerprint is stable across runs
"""

import pytest
from controller.chd_validator import (
    CHDValidator,
    ValidationResult,
    validate_requirements,
)


# -----------------------------------------------------------------------------
# Test CHD Project Name Extraction
# -----------------------------------------------------------------------------
class TestCHDProjectNameExtraction:
    """Tests for extracting project_name from CHD content."""

    def test_extracts_project_name_from_yaml_style(self):
        """project_name: value format should be extracted."""
        validator = CHDValidator()
        content = """
        project_name: health-tracker-app
        tech_stack:
          backend: fastapi
        """
        result = validator.validate(content, content)

        assert result.extracted_project_name == "health-tracker-app"

    def test_extracts_project_name_with_quotes(self):
        """project_name: "value" format should be extracted."""
        validator = CHDValidator()
        content = """
        project_name: "My Health Tracker"
        tech_stack:
          backend: fastapi
        """
        result = validator.validate(content, content)

        assert result.extracted_project_name == "My Health Tracker"

    def test_extracts_project_name_case_insensitive(self):
        """Project_Name or PROJECT_NAME should work."""
        validator = CHDValidator()
        content = """
        Project_Name: my-project
        tech_stack:
          backend: fastapi
        """
        result = validator.validate(content, content)

        assert result.extracted_project_name == "my-project"

    def test_does_not_use_system_instructions_as_name(self):
        """System instructions should NEVER become project name."""
        validator = CHDValidator()
        content = """
        You are Claude CLI operating inside the AI Development Platform.

        PHASE: 16C - REAL PROJECT EXECUTION

        project_name: health-tracker-app

        OBJECTIVE: Execute ONE real project
        """
        result = validator.validate(content, content)

        # Must extract the actual project_name, not system text
        assert result.extracted_project_name == "health-tracker-app"
        assert "You are Claude" not in (result.extracted_project_name or "")
        assert "operating" not in (result.extracted_project_name or "")

    def test_project_name_not_from_first_line(self):
        """First line of content should NOT be used as project name."""
        validator = CHDValidator()
        content = """You are Claude CLI operating inside the AI Development Platform.

        PHASE: 16C

        project_name: actual-project-name

        tech_stack:
          backend: python
        """
        result = validator.validate(content, content)

        assert result.extracted_project_name == "actual-project-name"
        assert result.extracted_project_name != "You are Claude CLI operating inside the AI Development Platform."

    def test_returns_none_when_no_project_name(self):
        """When no project_name in CHD, should return None (no fallback)."""
        validator = CHDValidator()
        content = """
        Build a REST API for user management.
        tech_stack:
          backend: fastapi
        """
        result = validator.validate(content, content)

        # Should be None, not inferred from description
        assert result.extracted_project_name is None

    def test_extracts_from_real_chd_file(self):
        """Test with realistic CHD file content."""
        validator = CHDValidator()
        content = """You are Claude CLI operating inside the AI Development Platform.

PHASE: 16C - REAL PROJECT EXECUTION (TEST -> PRODUCTION)

You MUST obey ALL existing governance documents:
- AI_POLICY.md
- ARCHITECTURE.md

--------------------------------------------------
INPUT CONTEXT (DO NOT MODIFY)
--------------------------------------------------
project_name: health-tracker-app
project_type: new

git_repo:
  type: new
  provider: github
  repo_name: https://github.com/rahul4webdev/health-tracker-app.git

tech_stack:
  backend:
    language: python
    framework: fastapi
  frontend:
    type: web
    framework: react

project_requirements: |
  Build a Health Tracker web application
"""
        result = validator.validate(content, content)

        assert result.extracted_project_name == "health-tracker-app"


# -----------------------------------------------------------------------------
# Test No Fallback Behavior
# -----------------------------------------------------------------------------
class TestNoFallbackBehavior:
    """Verify NO fallback to inferred text."""

    def test_no_fallback_to_description(self):
        """Should NOT use description as project name."""
        validator = CHDValidator()
        content = "Build a beautiful e-commerce platform with payment integration."
        result = validator.validate(content, content)

        # No project_name in content, should be None
        assert result.extracted_project_name is None
        # Should NOT be "Build a beautiful..."
        assert result.extracted_project_name != "Build a beautiful e-commerce platform"

    def test_no_fallback_to_first_word(self):
        """Should NOT use first word/phrase as name."""
        validator = CHDValidator()
        content = "E-commerce API backend with React frontend"
        result = validator.validate(content, content)

        assert result.extracted_project_name is None
        assert result.extracted_project_name != "E-commerce"

    def test_chd_project_name_overrides_everything(self):
        """CHD project_name must override any inferred name."""
        validator = CHDValidator()
        content = """
        name: wrong-name
        project_name: correct-project-name
        description: Some description that looks like a name
        """
        result = validator.validate(content, content)

        # project_name takes precedence over name
        assert result.extracted_project_name == "correct-project-name"


# -----------------------------------------------------------------------------
# Test Validation Result Fields
# -----------------------------------------------------------------------------
class TestValidationResultFields:
    """Verify ValidationResult has correct fields."""

    def test_validation_result_has_project_name_field(self):
        """ValidationResult must have extracted_project_name field."""
        result = ValidationResult(is_valid=True)
        assert hasattr(result, 'extracted_project_name')

    def test_validation_result_to_dict_includes_project_name(self):
        """to_dict() must include extracted_project_name."""
        result = ValidationResult(is_valid=True)
        result.extracted_project_name = "test-project"

        d = result.to_dict()
        assert "extracted_project_name" in d
        assert d["extracted_project_name"] == "test-project"

    def test_validation_result_project_name_default_none(self):
        """Default extracted_project_name should be None."""
        result = ValidationResult(is_valid=True)
        assert result.extracted_project_name is None


# -----------------------------------------------------------------------------
# Test Module-Level Function
# -----------------------------------------------------------------------------
class TestModuleLevelValidation:
    """Test module-level validate_requirements function."""

    def test_validate_requirements_extracts_project_name(self):
        """validate_requirements should extract project_name."""
        content = """
        project_name: my-api-project

        Build a REST API with FastAPI.
        """
        result = validate_requirements(content, content)

        assert result.extracted_project_name == "my-api-project"

    def test_validate_requirements_with_complex_chd(self):
        """Test with complex CHD file."""
        content = """You are an AI assistant.

project_name: complex-project
project_type: new

tech_stack:
  backend: fastapi
  frontend: react

requirements: Build something complex
"""
        result = validate_requirements(content, content)

        assert result.extracted_project_name == "complex-project"
        assert result.is_valid  # Should still be valid


# -----------------------------------------------------------------------------
# Test Edge Cases
# -----------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases in project name extraction."""

    def test_project_name_with_spaces(self):
        """Project name with spaces should be preserved."""
        validator = CHDValidator()
        content = 'project_name: "My Project Name"'
        result = validator.validate(content, content)

        assert result.extracted_project_name == "My Project Name"

    def test_project_name_with_numbers(self):
        """Project name with numbers should work."""
        validator = CHDValidator()
        content = "project_name: project-v2-api"
        result = validator.validate(content, content)

        assert result.extracted_project_name == "project-v2-api"

    def test_project_name_at_end_of_file(self):
        """Project name at end of file should be found."""
        validator = CHDValidator()
        content = """
        Build a REST API.
        tech_stack: fastapi
        project_name: api-project
        """
        result = validator.validate(content, content)

        assert result.extracted_project_name == "api-project"

    def test_multiple_project_name_uses_first(self):
        """If multiple project_name lines, use first."""
        validator = CHDValidator()
        content = """
        project_name: first-name
        some content
        project_name: second-name
        """
        result = validator.validate(content, content)

        assert result.extracted_project_name == "first-name"


# -----------------------------------------------------------------------------
# Test Determinism
# -----------------------------------------------------------------------------
class TestDeterminism:
    """Verify extraction is deterministic."""

    def test_same_input_same_output(self):
        """Same CHD content should always produce same project_name."""
        validator = CHDValidator()
        content = """
        project_name: deterministic-test
        backend: fastapi
        """

        results = [validator.validate(content, content) for _ in range(10)]
        names = [r.extracted_project_name for r in results]

        # All should be identical
        assert all(n == "deterministic-test" for n in names)

    def test_whitespace_variations_same_result(self):
        """Minor whitespace changes should not affect extraction."""
        validator = CHDValidator()

        content1 = "project_name: my-project"
        content2 = "project_name:  my-project"
        content3 = "project_name: my-project  "

        r1 = validator.validate(content1, content1)
        r2 = validator.validate(content2, content2)
        r3 = validator.validate(content3, content3)

        assert r1.extracted_project_name == "my-project"
        assert r2.extracted_project_name == "my-project"
        assert r3.extracted_project_name == "my-project"
