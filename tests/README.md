# Tests - AI Development Platform

This directory contains the test suite for the AI Development Platform. All tests follow the platform's [TESTING_STRATEGY.md](../docs/TESTING_STRATEGY.md).

## Test Structure

```
tests/
├── README.md                           # This file
├── __init__.py                         # Test package initialization
├── test_e2e_platform.py               # End-to-end platform tests (Phase 19)
├── test_job_execution.py              # Claude job execution tests
├── test_claude_auth.py                # Claude CLI authentication tests
├── test_controller.py                 # Task controller tests
├── test_telegram_bot.py               # Telegram bot tests
├── test_project_registry.py           # Project registry tests (if exists)
├── test_chd_project_name.py           # CHD project name extraction tests
├── test_lifecycle_engine.py           # Lifecycle engine tests
├── test_dashboard_backend.py          # Dashboard backend tests
├── test_priority_scheduling.py        # Priority scheduling tests
└── test_phase*.py                     # Phase-specific tests
```

## Running Tests

### Run All Tests
```bash
# From project root
pytest

# With verbose output and logging
pytest -v --log-cli-level=INFO

# With coverage
pytest --cov=controller --cov-report=html
```

### Run Specific Test Categories

```bash
# End-to-end tests only
pytest tests/test_e2e_platform.py -v

# Job execution tests only
pytest tests/test_job_execution.py -v

# Authentication tests only
pytest tests/test_claude_auth.py -v

# Skip slow/integration tests
pytest -m "not integration"
```

### Run with Flow Tracing

The E2E test suite includes detailed flow tracing for debugging:

```bash
# Run with full flow trace output
pytest tests/test_e2e_platform.py -v -s --log-cli-level=DEBUG
```

## Test Categories

### Layer 1: Unit Tests (Mandatory)
- **Coverage Threshold**: Minimum 70%
- **Execution**: On every commit
- **Blocking**: Merge blocked if tests fail

Files:
- `test_controller.py`
- `test_lifecycle_engine.py`
- `test_project_registry.py`
- `test_chd_project_name.py`

### Layer 3: Integration Tests
- **Execution**: On PR merge
- **Scope**: Component interactions, API contracts

Files:
- `test_e2e_platform.py`
- `test_dashboard_backend.py`
- `test_job_execution.py`

## Test Naming Convention

All tests follow the pattern: `test_{feature}_{scenario}`

Examples:
- `test_job_initial_state_is_queued`
- `test_script_uses_correct_permission_mode`
- `test_create_project_from_text`

## Key Test Fixtures

### `flow_tracer`
Creates a FlowTracer instance for detailed execution tracing:
```python
def test_example(self, flow_tracer):
    flow_tracer.step("step_name", {"key": "value"})
    # ... test code ...
    flow_tracer.complete(True, result)
```

### `sample_chd_file`
Creates a temporary CHD (Claude-Human Document) file:
```python
def test_example(self, sample_chd_file):
    content = sample_chd_file.read_text()
    # ... test with CHD content ...
```

### `mock_scheduler`
Provides a mock multi-worker scheduler:
```python
@pytest.mark.asyncio
async def test_example(self, mock_scheduler):
    with patch('controller.claude_backend.multi_scheduler', mock_scheduler):
        # ... test with mocked scheduler ...
```

## Critical Tests

### Permission Mode Test (CRITICAL)
The test `test_script_uses_correct_permission_mode` in `test_job_execution.py` verifies:
- Script uses `--permission-mode acceptEdits` (required for automation)
- Script does NOT use `--dangerously-skip-permissions` (causes core dumps)

This test MUST pass before any deployment.

### Job Scheduling Test
Tests in `test_e2e_platform.py` verify:
- Job is created when project is created
- Job is enqueued to scheduler
- Job state transitions correctly

## Adding New Tests

1. Create test file following naming convention: `test_{feature}.py`
2. Import necessary fixtures from conftest or define locally
3. Use FlowTracer for tracing complex flows
4. Add appropriate pytest markers (`@pytest.mark.asyncio`, etc.)
5. Update this README with new test category/file

## Test Failure Protocol

From [AI_POLICY.md](../docs/AI_POLICY.md):

1. AI agent identifies failing tests
2. AI agent fixes code or test
3. Re-run tests
4. Repeat until passing
5. Update CURRENT_STATE.md

## Environment Variables

Tests may require these environment variables (set via `.env` or export):
- `CLAUDE_JOBS_DIR`: Jobs directory path
- `CLAUDE_DOCS_DIR`: Documentation directory path
- `ANTHROPIC_API_KEY`: Claude API key (for live tests)

## Continuous Integration

Tests are run automatically:
- On every push to feature branches
- On every PR to main/develop
- Before deployment to testing environment

See `.github/workflows/ci.yml` for CI configuration.
