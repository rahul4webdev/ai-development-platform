"""
Pytest configuration for AI Development Platform tests.

This module provides:
1. Async test support without pytest-asyncio
2. Common fixtures for all tests
3. Test session configuration
"""

import asyncio
import functools
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock


# -----------------------------------------------------------------------------
# Async Test Support
# -----------------------------------------------------------------------------
def async_test(func):
    """
    Decorator to run async tests without pytest-asyncio.

    Usage:
        @async_test
        async def test_something(self):
            result = await some_async_function()
            assert result is not None
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
    return wrapper


# -----------------------------------------------------------------------------
# Common Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_scheduler():
    """Create a mock multi-worker scheduler."""
    scheduler = MagicMock()
    scheduler.running = True
    scheduler.enqueue_job = AsyncMock()
    scheduler.get_queue_status = MagicMock(return_value={
        "pending_jobs": 0,
        "running_jobs": 0,
        "workers": 3
    })
    return scheduler


@pytest.fixture
def mock_stopped_scheduler():
    """Create a mock scheduler that is stopped."""
    scheduler = MagicMock()
    scheduler.running = False
    return scheduler


@pytest.fixture
def sample_chd_content():
    """Return sample CHD content for testing."""
    return """# Project: health-tracker

## Overview
Build a comprehensive health tracking application.

## Requirements
- User authentication
- Track daily steps
- Data visualization

## Technical Stack
- Backend: FastAPI with Python
- Frontend: React with TypeScript
- Database: PostgreSQL

## Constraints
- Must be HIPAA-compliant
"""


@pytest.fixture
def sample_chd_file(tmp_path, sample_chd_content) -> Path:
    """Create a sample CHD file for testing."""
    chd_path = tmp_path / "health-tracker.md"
    chd_path.write_text(sample_chd_content)
    return chd_path


# -----------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------
TEST_PROJECT_NAME = "e2e-test-project"
TEST_USER_ID = "test-user-123"
TEST_TASK_DESCRIPTION = "Build a health tracker application"


# -----------------------------------------------------------------------------
# Session Configuration
# -----------------------------------------------------------------------------
def pytest_configure(config):
    """Configure pytest session."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async (custom implementation)"
    )
