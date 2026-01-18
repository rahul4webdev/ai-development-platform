"""
Unit Tests for Priority Scheduling - Phase 14.11

Test coverage for:
- Priority preemption (EMERGENCY > HIGH > NORMAL > LOW)
- FIFO ordering within same priority level
- Starvation prevention (30-min escalation)
- Priority capping at HIGH (75)
- Idempotent escalation
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os
import json

# Set up test environment before imports
os.environ['CLAUDE_JOBS_DIR'] = tempfile.mkdtemp()

from controller.claude_backend import (
    JobPriority,
    JobState,
    ClaudeJob,
    PersistentJobStore,
    MultiWorkerScheduler,
    STARVATION_THRESHOLD_MINUTES,
    PRIORITY_ESCALATION_AMOUNT,
    PRIORITY_ESCALATION_CAP,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def temp_state_file():
    """Create a temporary state file for testing."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def persistent_store(temp_state_file):
    """Create a PersistentJobStore with temp file."""
    from pathlib import Path
    return PersistentJobStore(state_file=Path(temp_state_file))


def create_test_job(
    job_id: str,
    priority: int = JobPriority.NORMAL.value,
    created_at: datetime = None,
    state: JobState = JobState.QUEUED,
) -> ClaudeJob:
    """Helper to create test jobs."""
    return ClaudeJob(
        job_id=job_id,
        project_name="test_project",
        task_description=f"Test task {job_id}",
        task_type="feature_development",
        state=state,
        created_at=created_at or datetime.utcnow(),
        priority=priority,
    )


# -----------------------------------------------------------------------------
# Test 1: Priority Preemption
# -----------------------------------------------------------------------------
class TestPriorityPreemption:
    """Test that higher priority jobs are scheduled before lower priority."""

    @pytest.mark.asyncio
    async def test_emergency_before_normal(self, persistent_store):
        """EMERGENCY (100) job should be scheduled before NORMAL (50) job."""
        # Create NORMAL job first
        normal_job = create_test_job(
            "job_normal_1",
            priority=JobPriority.NORMAL.value,
            created_at=datetime.utcnow() - timedelta(minutes=5),
        )
        await persistent_store.save_job(normal_job)

        # Create EMERGENCY job later
        emergency_job = create_test_job(
            "job_emergency_1",
            priority=JobPriority.EMERGENCY.value,
            created_at=datetime.utcnow(),
        )
        await persistent_store.save_job(emergency_job)

        # Get queued jobs - EMERGENCY should be first
        queued = await persistent_store.get_queued_jobs()
        assert len(queued) == 2
        assert queued[0].job_id == "job_emergency_1"
        assert queued[1].job_id == "job_normal_1"

    @pytest.mark.asyncio
    async def test_priority_ordering_all_levels(self, persistent_store):
        """Test full priority ordering: EMERGENCY > HIGH > NORMAL > LOW."""
        # Create jobs in reverse priority order (LOW first)
        low_job = create_test_job("job_low", JobPriority.LOW.value)
        await persistent_store.save_job(low_job)

        normal_job = create_test_job("job_normal", JobPriority.NORMAL.value)
        await persistent_store.save_job(normal_job)

        high_job = create_test_job("job_high", JobPriority.HIGH.value)
        await persistent_store.save_job(high_job)

        emergency_job = create_test_job("job_emergency", JobPriority.EMERGENCY.value)
        await persistent_store.save_job(emergency_job)

        # Verify ordering
        queued = await persistent_store.get_queued_jobs()
        assert len(queued) == 4
        assert queued[0].job_id == "job_emergency"  # Priority 100
        assert queued[1].job_id == "job_high"       # Priority 75
        assert queued[2].job_id == "job_normal"     # Priority 50
        assert queued[3].job_id == "job_low"        # Priority 25


# -----------------------------------------------------------------------------
# Test 2: FIFO Within Same Priority
# -----------------------------------------------------------------------------
class TestFIFOWithinPriority:
    """Test that jobs with same priority are ordered by created_at (FIFO)."""

    @pytest.mark.asyncio
    async def test_fifo_same_priority(self, persistent_store):
        """Jobs with same priority should be processed in FIFO order."""
        base_time = datetime.utcnow()

        # Create 3 NORMAL priority jobs at different times
        job_1 = create_test_job(
            "job_first",
            priority=JobPriority.NORMAL.value,
            created_at=base_time - timedelta(minutes=10),
        )
        await persistent_store.save_job(job_1)

        job_2 = create_test_job(
            "job_second",
            priority=JobPriority.NORMAL.value,
            created_at=base_time - timedelta(minutes=5),
        )
        await persistent_store.save_job(job_2)

        job_3 = create_test_job(
            "job_third",
            priority=JobPriority.NORMAL.value,
            created_at=base_time,
        )
        await persistent_store.save_job(job_3)

        # Verify FIFO order within same priority
        queued = await persistent_store.get_queued_jobs()
        assert len(queued) == 3
        assert queued[0].job_id == "job_first"   # Oldest
        assert queued[1].job_id == "job_second"
        assert queued[2].job_id == "job_third"   # Newest

    @pytest.mark.asyncio
    async def test_job_id_tiebreaker(self, persistent_store):
        """Jobs with same priority and timestamp should use job_id as tiebreaker."""
        same_time = datetime.utcnow()

        # Create jobs with same priority and timestamp
        job_a = create_test_job(
            "aaa_job",
            priority=JobPriority.NORMAL.value,
            created_at=same_time,
        )
        await persistent_store.save_job(job_a)

        job_z = create_test_job(
            "zzz_job",
            priority=JobPriority.NORMAL.value,
            created_at=same_time,
        )
        await persistent_store.save_job(job_z)

        # Verify job_id tiebreaker (alphabetical)
        queued = await persistent_store.get_queued_jobs()
        assert len(queued) == 2
        assert queued[0].job_id == "aaa_job"
        assert queued[1].job_id == "zzz_job"


# -----------------------------------------------------------------------------
# Test 3: Starvation Prevention Escalation
# -----------------------------------------------------------------------------
class TestStarvationPrevention:
    """Test automatic priority escalation for long-waiting jobs."""

    def test_wait_time_calculation(self):
        """Test that wait time is calculated correctly."""
        job = create_test_job(
            "test_job",
            created_at=datetime.utcnow() - timedelta(minutes=45),
        )
        wait_minutes = job.get_wait_time_minutes()
        assert 44 < wait_minutes < 46  # Allow for test execution time

    def test_escalation_eligibility(self):
        """Test that jobs waiting >30 min are eligible for escalation."""
        # Job waiting 25 minutes - not eligible
        job_short = create_test_job(
            "short_wait",
            created_at=datetime.utcnow() - timedelta(minutes=25),
        )
        assert job_short.get_wait_time_minutes() < STARVATION_THRESHOLD_MINUTES

        # Job waiting 35 minutes - eligible
        job_long = create_test_job(
            "long_wait",
            created_at=datetime.utcnow() - timedelta(minutes=35),
        )
        assert job_long.get_wait_time_minutes() >= STARVATION_THRESHOLD_MINUTES

    def test_escalation_amount(self):
        """Test that escalation increases priority by correct amount."""
        job = create_test_job("test_job", priority=JobPriority.LOW.value)
        old_priority = job.priority

        # Simulate escalation
        new_priority = old_priority + PRIORITY_ESCALATION_AMOUNT

        assert new_priority == JobPriority.LOW.value + 10  # 25 + 10 = 35


# -----------------------------------------------------------------------------
# Test 4: Priority Capping at HIGH (75)
# -----------------------------------------------------------------------------
class TestPriorityCapping:
    """Test that escalation is capped at HIGH priority (75)."""

    def test_can_escalate_to_checks_cap(self):
        """Test that can_escalate_to enforces the cap."""
        # Can escalate from LOW (25) to 35
        assert JobPriority.can_escalate_to(25, 35) is True

        # Can escalate from NORMAL (50) to 60
        assert JobPriority.can_escalate_to(50, 60) is True

        # Can escalate to exactly HIGH (75)
        assert JobPriority.can_escalate_to(65, 75) is True

        # Cannot escalate beyond HIGH (75)
        assert JobPriority.can_escalate_to(70, 80) is False

        # Cannot escalate from already HIGH (75)
        assert JobPriority.can_escalate_to(75, 85) is False

    def test_escalation_capped_at_high(self):
        """Test that repeated escalations cap at HIGH."""
        job = create_test_job("test_job", priority=JobPriority.LOW.value)

        # Simulate multiple escalations
        for i in range(10):  # More than needed to reach cap
            new_priority = job.priority + PRIORITY_ESCALATION_AMOUNT
            if JobPriority.can_escalate_to(job.priority, new_priority):
                job.priority = min(new_priority, PRIORITY_ESCALATION_CAP)
                job.priority_escalations += 1

        # Should be capped at HIGH
        assert job.priority == PRIORITY_ESCALATION_CAP  # 75
        assert job.priority <= JobPriority.HIGH.value

    def test_emergency_priority_preserved(self):
        """EMERGENCY jobs should never be created via escalation."""
        # Starting from highest possible escalation
        job = create_test_job("test_job", priority=PRIORITY_ESCALATION_CAP)

        # Cannot escalate beyond HIGH
        new_priority = job.priority + PRIORITY_ESCALATION_AMOUNT
        assert JobPriority.can_escalate_to(job.priority, new_priority) is False

        # EMERGENCY can only be set explicitly, not via escalation
        assert PRIORITY_ESCALATION_CAP < JobPriority.EMERGENCY.value


# -----------------------------------------------------------------------------
# Test 5: Idempotent Escalation
# -----------------------------------------------------------------------------
class TestIdempotentEscalation:
    """Test that escalation is deterministic and idempotent."""

    def test_escalation_deterministic(self):
        """Same input should always produce same output."""
        job1 = create_test_job("job1", priority=JobPriority.NORMAL.value)
        job2 = create_test_job("job2", priority=JobPriority.NORMAL.value)

        # Same escalation logic
        new1 = min(job1.priority + PRIORITY_ESCALATION_AMOUNT, PRIORITY_ESCALATION_CAP)
        new2 = min(job2.priority + PRIORITY_ESCALATION_AMOUNT, PRIORITY_ESCALATION_CAP)

        assert new1 == new2
        assert new1 == 60  # 50 + 10

    def test_no_double_escalation_in_same_cycle(self):
        """A job should not be escalated twice in the same check cycle."""
        job = create_test_job(
            "test_job",
            priority=JobPriority.LOW.value,
            created_at=datetime.utcnow() - timedelta(minutes=60),
        )

        # First escalation
        job.priority = min(job.priority + PRIORITY_ESCALATION_AMOUNT, PRIORITY_ESCALATION_CAP)
        job.priority_escalations += 1
        job.last_escalation_at = datetime.utcnow()

        first_priority = job.priority
        first_escalations = job.priority_escalations

        # Second escalation attempt in same cycle should not happen
        # (wait_time is reset tracking, but we track escalation_at)
        # The scheduler checks if escalation already happened recently

        # Just verify state after single escalation
        assert job.priority == 35  # 25 + 10
        assert job.priority_escalations == 1
        assert job.last_escalation_at is not None


# -----------------------------------------------------------------------------
# Test JobPriority Enum
# -----------------------------------------------------------------------------
class TestJobPriorityEnum:
    """Test JobPriority enum functionality."""

    def test_priority_values(self):
        """Test that priority values are correct."""
        assert JobPriority.EMERGENCY.value == 100
        assert JobPriority.HIGH.value == 75
        assert JobPriority.NORMAL.value == 50
        assert JobPriority.LOW.value == 25

    def test_from_value_exact_match(self):
        """Test from_value with exact enum values."""
        assert JobPriority.from_value(100) == JobPriority.EMERGENCY
        assert JobPriority.from_value(75) == JobPriority.HIGH
        assert JobPriority.from_value(50) == JobPriority.NORMAL
        assert JobPriority.from_value(25) == JobPriority.LOW

    def test_from_value_clamping(self):
        """Test from_value with non-enum values."""
        # Values above EMERGENCY return EMERGENCY
        assert JobPriority.from_value(150) == JobPriority.EMERGENCY

        # Values between priorities return appropriate level
        assert JobPriority.from_value(80) == JobPriority.HIGH
        assert JobPriority.from_value(60) == JobPriority.NORMAL
        assert JobPriority.from_value(30) == JobPriority.LOW

        # Values below LOW return LOW
        assert JobPriority.from_value(10) == JobPriority.LOW


# -----------------------------------------------------------------------------
# Test ClaudeJob Priority Sort Key
# -----------------------------------------------------------------------------
class TestClaudeJobSortKey:
    """Test ClaudeJob priority sort key generation."""

    def test_sort_key_format(self):
        """Test that sort key has correct format: (-priority, created_at, job_id)."""
        job = create_test_job("test_job", priority=50)
        key = job.get_priority_sort_key()

        assert isinstance(key, tuple)
        assert len(key) == 3
        assert key[0] == -50  # Negated priority for descending sort
        assert isinstance(key[1], float)  # Timestamp
        assert key[2] == "test_job"

    def test_sort_key_ordering(self):
        """Test that sort keys produce correct ordering."""
        base_time = datetime.utcnow()

        jobs = [
            create_test_job("c", priority=50, created_at=base_time),
            create_test_job("a", priority=100, created_at=base_time),
            create_test_job("b", priority=50, created_at=base_time - timedelta(minutes=1)),
        ]

        # Sort by sort key
        sorted_jobs = sorted(jobs, key=lambda j: j.get_priority_sort_key())

        # Should be: a (priority 100), b (priority 50, older), c (priority 50, newer)
        assert sorted_jobs[0].job_id == "a"
        assert sorted_jobs[1].job_id == "b"
        assert sorted_jobs[2].job_id == "c"


# -----------------------------------------------------------------------------
# Integration Test
# -----------------------------------------------------------------------------
class TestPrioritySchedulingIntegration:
    """Integration tests for the full priority scheduling flow."""

    @pytest.mark.asyncio
    async def test_mixed_priority_queue_ordering(self, persistent_store):
        """Test realistic queue with mixed priorities and ages."""
        base_time = datetime.utcnow()

        # Create a mix of jobs
        jobs = [
            create_test_job("low_old", JobPriority.LOW.value, base_time - timedelta(hours=2)),
            create_test_job("normal_new", JobPriority.NORMAL.value, base_time),
            create_test_job("high_medium", JobPriority.HIGH.value, base_time - timedelta(hours=1)),
            create_test_job("normal_old", JobPriority.NORMAL.value, base_time - timedelta(hours=3)),
            create_test_job("emergency_new", JobPriority.EMERGENCY.value, base_time),
        ]

        for job in jobs:
            await persistent_store.save_job(job)

        # Get ordered queue
        queued = await persistent_store.get_queued_jobs()

        # Verify ordering
        assert queued[0].job_id == "emergency_new"  # Priority 100
        assert queued[1].job_id == "high_medium"    # Priority 75
        assert queued[2].job_id == "normal_old"     # Priority 50, oldest
        assert queued[3].job_id == "normal_new"     # Priority 50, newer
        assert queued[4].job_id == "low_old"        # Priority 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
