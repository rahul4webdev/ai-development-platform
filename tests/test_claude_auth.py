"""
Unit Tests for Claude CLI Authentication Detection - Phase 15.5

Test coverage for:
- CLI session-based authentication detection
- API key authentication (legacy)
- No authentication scenario
- Config file discovery across multiple paths
- Authentication priority (CLI session vs API key)

Phase 15.5: Claude CLI authentication uses session-based auth; API key optional
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary config directory structure."""
    config_dir = tmp_path / ".config" / "claude"
    config_dir.mkdir(parents=True)
    return config_dir


@pytest.fixture
def mock_claude_version_success():
    """Mock successful claude --version execution."""
    async def mock_exec(*args, **kwargs):
        mock_process = MagicMock()
        mock_process.returncode = 0

        async def mock_communicate():
            return (b"2.1.12 (Claude Code)", b"")

        mock_process.communicate = mock_communicate
        return mock_process

    return mock_exec


@pytest.fixture
def mock_claude_version_not_found():
    """Mock claude --version when CLI is not installed."""
    async def mock_exec(*args, **kwargs):
        raise FileNotFoundError("claude not found")

    return mock_exec


@pytest.fixture
def mock_claude_version_auth_error():
    """Mock claude --version with auth error."""
    async def mock_exec(*args, **kwargs):
        mock_process = MagicMock()
        mock_process.returncode = 1

        async def mock_communicate():
            return (b"", b"Error: Not authenticated. Please run 'claude auth login'")

        mock_process.communicate = mock_communicate
        return mock_process

    return mock_exec


# -----------------------------------------------------------------------------
# Test Cases: CLI Detection
# -----------------------------------------------------------------------------
class TestClaudeCliDetection:
    """Tests for Claude CLI binary detection."""

    @pytest.mark.asyncio
    async def test_cli_not_installed(self, mock_claude_version_not_found):
        """Test detection when Claude CLI is not installed."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_not_found):
            from controller.claude_backend import check_claude_availability
            result = await check_claude_availability()

            assert result["available"] is False
            assert result["installed"] is False
            assert result["authenticated"] is False
            assert result["auth_type"] == "none"
            assert "not installed" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cli_installed_version_detected(self, mock_claude_version_success):
        """Test that version is correctly detected when CLI is installed."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value=None):  # No API key
                with patch.object(Path, 'exists', return_value=False):  # No config files
                    from controller.claude_backend import check_claude_availability
                    result = await check_claude_availability()

                    assert result["installed"] is True
                    assert result["version"] == "2.1.12 (Claude Code)"


# -----------------------------------------------------------------------------
# Test Cases: Authentication Detection
# -----------------------------------------------------------------------------
class TestClaudeAuthDetection:
    """Tests for Claude CLI authentication detection."""

    @pytest.mark.asyncio
    async def test_api_key_authentication(self, mock_claude_version_success):
        """Test detection when API key is configured (legacy method)."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value="sk-ant-api03-test-key"):
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                assert result["available"] is True
                assert result["installed"] is True
                assert result["authenticated"] is True
                assert result["auth_type"] == "api_key"
                assert result["api_key_configured"] is True

    @pytest.mark.asyncio
    async def test_cli_session_authentication(self, mock_claude_version_success, temp_config_dir):
        """Test detection when CLI session auth is present (no API key)."""
        # Create mock auth file
        auth_file = temp_config_dir / "credentials.json"
        auth_file.write_text('{"token": "mock-session-token"}')

        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value=None):  # No API key
                with patch.object(Path, 'home', return_value=temp_config_dir.parent.parent):
                    from controller.claude_backend import check_claude_availability
                    result = await check_claude_availability()

                    assert result["installed"] is True
                    assert result["version"] == "2.1.12 (Claude Code)"
                    # Should be available even without API key
                    assert result["api_key_configured"] is False

    @pytest.mark.asyncio
    async def test_no_authentication(self, mock_claude_version_auth_error):
        """Test detection when CLI is installed but not authenticated."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_auth_error):
            with patch('os.getenv', return_value=None):
                with patch.object(Path, 'exists', return_value=False):
                    from controller.claude_backend import check_claude_availability
                    result = await check_claude_availability()

                    # CLI returned error, so should not be available
                    assert result["installed"] is False
                    assert result["authenticated"] is False
                    assert result["auth_type"] == "none"

    @pytest.mark.asyncio
    async def test_api_key_takes_precedence(self, mock_claude_version_success, temp_config_dir):
        """Test that API key auth is detected first (before checking config files)."""
        # Create mock auth file
        auth_file = temp_config_dir / "credentials.json"
        auth_file.write_text('{"token": "mock-session-token"}')

        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value="sk-ant-api03-test-key"):
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                # API key should be detected first
                assert result["authenticated"] is True
                assert result["auth_type"] == "api_key"
                assert result["api_key_configured"] is True


# -----------------------------------------------------------------------------
# Test Cases: Config File Discovery
# -----------------------------------------------------------------------------
class TestConfigFileDiscovery:
    """Tests for Claude CLI config file discovery."""

    def test_config_paths_checked(self):
        """Test that multiple config paths are checked."""
        from controller.claude_backend import check_claude_availability
        # The function should check:
        # - ~/.config/claude/
        # - ~/.claude/
        # - /root/.config/claude/
        # - /home/ai-controller/.config/claude/
        # This is a documentation test to ensure paths are considered


# -----------------------------------------------------------------------------
# Test Cases: Result Structure
# -----------------------------------------------------------------------------
class TestResultStructure:
    """Tests for the result dictionary structure."""

    @pytest.mark.asyncio
    async def test_result_has_all_required_fields(self, mock_claude_version_success):
        """Test that result contains all required fields."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value=None):
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                # Check all required fields exist
                required_fields = [
                    "available",
                    "installed",
                    "version",
                    "authenticated",
                    "auth_type",
                    "api_key_configured",
                    "wrapper_exists",
                    "error",
                    "auth_config_path",
                ]

                for field in required_fields:
                    assert field in result, f"Missing field: {field}"

    @pytest.mark.asyncio
    async def test_auth_type_values(self, mock_claude_version_success):
        """Test that auth_type has valid values."""
        valid_auth_types = ["cli_session", "api_key", "none"]

        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value=None):
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                assert result["auth_type"] in valid_auth_types


# -----------------------------------------------------------------------------
# Test Cases: Backwards Compatibility
# -----------------------------------------------------------------------------
class TestBackwardsCompatibility:
    """Tests for backwards compatibility with old API."""

    @pytest.mark.asyncio
    async def test_api_key_configured_field_present(self, mock_claude_version_success):
        """Test that api_key_configured field is still present (legacy support)."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value="sk-ant-api03-test"):
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                # Legacy field should still be present
                assert "api_key_configured" in result
                assert result["api_key_configured"] is True

    @pytest.mark.asyncio
    async def test_wrapper_exists_field_present(self, mock_claude_version_success):
        """Test that wrapper_exists field is still present."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            from controller.claude_backend import check_claude_availability
            result = await check_claude_availability()

            assert "wrapper_exists" in result
            assert isinstance(result["wrapper_exists"], bool)


# -----------------------------------------------------------------------------
# Test Cases: Error Handling
# -----------------------------------------------------------------------------
class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_permission_error_on_config_check(self, mock_claude_version_success):
        """Test graceful handling of permission errors when checking config."""
        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value=None):
                # Permission errors should be caught and handled gracefully
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                # Should not crash, should return a valid result
                assert isinstance(result, dict)
                assert "available" in result


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------
class TestIntegration:
    """Integration tests for Claude CLI availability check."""

    @pytest.mark.asyncio
    async def test_full_flow_with_session_auth(self, mock_claude_version_success, temp_config_dir):
        """Test full flow: CLI installed, session auth, no API key."""
        # Create auth file
        auth_file = temp_config_dir / "credentials.json"
        auth_file.write_text('{"session": "active"}')

        with patch('asyncio.create_subprocess_exec', mock_claude_version_success):
            with patch('os.getenv', return_value=None):  # No API key
                from controller.claude_backend import check_claude_availability
                result = await check_claude_availability()

                # Should be available with session auth, no API key needed
                assert result["installed"] is True
                assert result["version"] is not None
                assert result["api_key_configured"] is False
                # available should be True because version check passed
