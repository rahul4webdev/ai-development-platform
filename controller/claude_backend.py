"""
Claude CLI Execution Backend
Phase 14.3: Controller Integration
Phase 14.10: Multi-Claude Worker Scheduler
Phase 14.11: Priority & Fair Scheduling

This module integrates Claude CLI as an execution backend for the controller.
It manages job workspaces, invokes the secure wrapper, and tracks job state.

Phase 14.10 adds:
- Multi-worker scheduler (MAX_CONCURRENT_JOBS=3)
- Persistent job state
- Resource limits (CPU nice, memory)
- Crash recovery
- Graceful failure handling

Phase 14.11 adds:
- Priority-aware scheduling (EMERGENCY > HIGH > NORMAL > LOW)
- Starvation prevention (auto-escalation after 30 minutes)
- Priority audit logging
- Deterministic ordering (priority, then created_at, then job_id)

IMPORTANT: This module does NOT replace existing controller functionality.
It ADDS Claude CLI as a new execution backend alongside the existing flow.
"""

import asyncio
import json
import logging
import os
import resource
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Set

import yaml

logger = logging.getLogger("claude_backend")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
JOBS_BASE_DIR = Path(os.getenv("CLAUDE_JOBS_DIR", "/home/aitesting.mybd.in/jobs"))
DOCS_DIR = Path(os.getenv("CLAUDE_DOCS_DIR", "/home/aitesting.mybd.in/public_html/docs"))
WRAPPER_SCRIPT = Path("/home/aitesting.mybd.in/public_html/scripts/run_claude_job.sh")
JOB_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))

# Phase 14.10: Multi-worker configuration
MAX_CONCURRENT_JOBS = 3  # Maximum concurrent Claude CLI jobs
WORKER_NICE_VALUE = 10   # CPU nice value for worker processes (lower priority)
WORKER_MEMORY_LIMIT_MB = 2048  # Memory limit per worker in MB
JOB_STATE_FILE = JOBS_BASE_DIR / "job_state.json"  # Persistent job state
JOB_CLEANUP_AFTER_HOURS = 24  # Auto-cleanup completed jobs after this many hours

# Phase 14.11: Priority & Fairness configuration
STARVATION_THRESHOLD_MINUTES = 30  # Time before priority escalation
PRIORITY_ESCALATION_AMOUNT = 10  # Priority increase per escalation
PRIORITY_ESCALATION_CAP = 75  # Maximum priority from escalation (HIGH)
PRIORITY_AUDIT_LOG = JOBS_BASE_DIR / "priority_audit.log"  # Priority escalation audit log

# Required policy documents (must exist in docs/)
REQUIRED_DOCS = [
    "AI_POLICY.md",
    "ARCHITECTURE.md",
    "CURRENT_STATE.md",
    "DEPLOYMENT.md",
    "PROJECT_CONTEXT.md",
    "PROJECT_MANIFEST.yaml",
    "TESTING_STRATEGY.md",
]

# Phase 19 fix: Claude availability cache to prevent sporadic timeout issues
# The execution test can occasionally timeout due to CLI initialization delays
# Cache successful results for 5 minutes to avoid repeated slow checks
_claude_availability_cache: Optional[Dict[str, Any]] = None
_claude_availability_cache_time: Optional[datetime] = None
CLAUDE_AVAILABILITY_CACHE_TTL_SECONDS = 300  # 5 minutes


# -----------------------------------------------------------------------------
# Job State Management (Phase 14.10 Extended)
# -----------------------------------------------------------------------------
class JobState(str, Enum):
    """
    Job execution states.

    State machine:
    QUEUED → RUNNING → AWAITING_APPROVAL → DEPLOYED → COMPLETED
                   ↓           ↓              ↓
                 FAILED      FAILED        FAILED
                   ↓
                TIMEOUT
                   ↓
               CANCELLED
    """
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    AWAITING_APPROVAL = "awaiting_approval"  # Phase 14.10: waiting for human approval
    DEPLOYED = "deployed"  # Phase 14.10: deployed to environment
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    @classmethod
    def terminal_states(cls) -> Set["JobState"]:
        """Return states that indicate job completion."""
        return {cls.COMPLETED, cls.FAILED, cls.TIMEOUT, cls.CANCELLED}

    @classmethod
    def active_states(cls) -> Set["JobState"]:
        """Return states where a job is actively being processed."""
        return {cls.PREPARING, cls.RUNNING}


# -----------------------------------------------------------------------------
# Phase 14.11: Job Priority
# -----------------------------------------------------------------------------
class JobPriority(int, Enum):
    """
    Job priority levels for scheduling.

    Phase 14.11: Priority is immutable once assigned, except for starvation escalation.
    Priority is set ONLY by controller logic - never by Claude CLI or Telegram bot.

    Ordering: EMERGENCY (100) > HIGH (75) > NORMAL (50) > LOW (25)
    """
    EMERGENCY = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25

    @classmethod
    def from_value(cls, value: int) -> "JobPriority":
        """Get priority enum from integer value, clamping to valid range."""
        for priority in cls:
            if priority.value == value:
                return priority
        # Return closest valid priority
        if value >= cls.EMERGENCY.value:
            return cls.EMERGENCY
        elif value >= cls.HIGH.value:
            return cls.HIGH
        elif value >= cls.NORMAL.value:
            return cls.NORMAL
        return cls.LOW

    @classmethod
    def can_escalate_to(cls, current: int, target: int) -> bool:
        """Check if escalation from current to target is allowed."""
        # Cannot exceed HIGH (75) via escalation
        return target <= PRIORITY_ESCALATION_CAP and target > current


@dataclass
class ClaudeJob:
    """Represents a Claude CLI execution job."""
    job_id: str
    project_name: str
    task_description: str
    task_type: str
    state: JobState
    created_at: datetime
    # Phase 14.11: Priority scheduling
    priority: int = JobPriority.NORMAL.value  # Default to NORMAL (50)
    priority_escalations: int = 0  # Number of times priority was escalated
    last_escalation_at: Optional[datetime] = None  # Last escalation timestamp
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    error_message: Optional[str] = None
    workspace_dir: Optional[Path] = None
    created_by: Optional[str] = None
    # Execution metadata
    stdout_file: Optional[Path] = None
    stderr_file: Optional[Path] = None
    result_summary: Optional[str] = None
    # Phase 15.2: Aspect isolation
    aspect: str = "core"  # Project aspect for scheduler isolation
    # Phase 15.6: Execution Gate integration
    lifecycle_id: Optional[str] = None  # Associated lifecycle ID
    lifecycle_state: str = "development"  # Current lifecycle state for gate check
    requested_action: str = "write_code"  # Action being requested
    user_role: str = "developer"  # Role of requesting user
    gate_allowed: Optional[bool] = None  # Result of execution gate check
    gate_denied_reason: Optional[str] = None  # Reason if gate denied

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "project_name": self.project_name,
            "task_description": self.task_description,
            "task_type": self.task_type,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            # Phase 14.11: Priority fields
            "priority": self.priority,
            "priority_escalations": self.priority_escalations,
            "last_escalation_at": self.last_escalation_at.isoformat() if self.last_escalation_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir else None,
            "created_by": self.created_by,
            "result_summary": self.result_summary,
            # Phase 15.2: Aspect isolation
            "aspect": self.aspect,
            # Phase 15.6: Execution Gate fields
            "lifecycle_id": self.lifecycle_id,
            "lifecycle_state": self.lifecycle_state,
            "requested_action": self.requested_action,
            "user_role": self.user_role,
            "gate_allowed": self.gate_allowed,
            "gate_denied_reason": self.gate_denied_reason,
        }

    def get_wait_time_seconds(self) -> float:
        """Get time spent waiting in QUEUED state."""
        if self.state != JobState.QUEUED:
            return 0
        return (datetime.utcnow() - self.created_at).total_seconds()

    def get_wait_time_minutes(self) -> float:
        """Get time spent waiting in QUEUED state in minutes."""
        return self.get_wait_time_seconds() / 60

    def get_priority_sort_key(self) -> tuple:
        """
        Get sort key for priority queue ordering.

        Phase 14.11: Ordering is:
        1. Highest priority first (-priority for descending)
        2. Oldest job first (created_at timestamp)
        3. Job ID as tie-breaker (for stability)
        """
        return (-self.priority, self.created_at.timestamp(), self.job_id)


# -----------------------------------------------------------------------------
# Job Queue (In-Memory)
# -----------------------------------------------------------------------------
class JobQueue:
    """
    In-memory job queue for Claude CLI execution.
    Jobs are persisted to disk for durability.
    """

    def __init__(self):
        self._jobs: Dict[str, ClaudeJob] = {}
        self._queue: List[str] = []  # Job IDs in queue order
        self._active_job: Optional[str] = None
        self._lock = asyncio.Lock()

    async def enqueue(self, job: ClaudeJob) -> str:
        """Add a job to the queue."""
        async with self._lock:
            self._jobs[job.job_id] = job
            self._queue.append(job.job_id)
            logger.info(f"Job {job.job_id} enqueued (queue size: {len(self._queue)})")
            return job.job_id

    async def get_job(self, job_id: str) -> Optional[ClaudeJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def get_next(self) -> Optional[ClaudeJob]:
        """Get the next job in queue."""
        async with self._lock:
            while self._queue:
                job_id = self._queue[0]
                job = self._jobs.get(job_id)
                if job and job.state == JobState.QUEUED:
                    return job
                self._queue.pop(0)
            return None

    async def update_job(self, job_id: str, **kwargs) -> bool:
        """Update job attributes."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        return True

    async def set_active(self, job_id: Optional[str]) -> None:
        """Set the currently active job."""
        async with self._lock:
            self._active_job = job_id

    async def get_active(self) -> Optional[ClaudeJob]:
        """Get the currently active job."""
        if self._active_job:
            return self._jobs.get(self._active_job)
        return None

    async def list_jobs(
        self,
        state: Optional[JobState] = None,
        project: Optional[str] = None,
        limit: int = 50
    ) -> List[ClaudeJob]:
        """List jobs with optional filtering."""
        jobs = list(self._jobs.values())
        if state:
            jobs = [j for j in jobs if j.state == state]
        if project:
            jobs = [j for j in jobs if j.project_name == project]
        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    async def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status summary."""
        queued = sum(1 for j in self._jobs.values() if j.state == JobState.QUEUED)
        running = sum(1 for j in self._jobs.values() if j.state == JobState.RUNNING)
        completed = sum(1 for j in self._jobs.values() if j.state == JobState.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.state == JobState.FAILED)
        return {
            "queued": queued,
            "running": running,
            "completed": completed,
            "failed": failed,
            "total": len(self._jobs),
            "active_job": self._active_job,
        }


# Global job queue instance
job_queue = JobQueue()


# -----------------------------------------------------------------------------
# Job Workspace Management
# -----------------------------------------------------------------------------
class WorkspaceManager:
    """Manages job workspaces for Claude CLI execution."""

    @staticmethod
    def create_workspace(job_id: str, project_name: str) -> Path:
        """
        Create an isolated workspace for a job.

        Structure:
        /jobs/job-<uuid>/
          AI_POLICY.md
          ARCHITECTURE.md
          CURRENT_STATE.md
          DEPLOYMENT.md
          PROJECT_CONTEXT.md
          PROJECT_MANIFEST.yaml
          TESTING_STRATEGY.md
          TASK.md
          logs/
        """
        workspace = JOBS_BASE_DIR / f"job-{job_id}"
        workspace.mkdir(parents=True, exist_ok=True)

        # Create logs directory
        (workspace / "logs").mkdir(exist_ok=True)

        # Copy required documents
        for doc in REQUIRED_DOCS:
            src = DOCS_DIR / doc
            dst = workspace / doc
            if src.exists():
                shutil.copy2(src, dst)
            else:
                logger.warning(f"Required document not found: {src}")

        logger.info(f"Created workspace: {workspace}")
        return workspace

    @staticmethod
    def create_task_file(workspace: Path, task_description: str, task_type: str, metadata: Dict[str, Any]) -> Path:
        """Create the TASK.md file with instructions for Claude."""
        task_file = workspace / "TASK.md"

        content = f"""# Task Instructions

## Task Type
{task_type}

## Description
{task_description}

## Metadata
- Created: {metadata.get('created_at', datetime.utcnow().isoformat())}
- Created By: {metadata.get('created_by', 'system')}
- Project: {metadata.get('project_name', 'unknown')}
- Job ID: {metadata.get('job_id', 'unknown')}

## Requirements

1. Read all policy documents BEFORE starting work
2. Follow AI_POLICY.md rules strictly
3. Respect ARCHITECTURE.md constraints
4. Update CURRENT_STATE.md after completing changes
5. Document your work in logs/EXECUTION_LOG.md

## Task Details

{task_description}

## Expected Deliverables

Based on task type "{task_type}":
"""

        if task_type == "feature_development":
            content += """
- Implementation code
- Unit tests
- Updated documentation
- Git commit (atomic, well-documented)
"""
        elif task_type == "bug_fix":
            content += """
- Bug fix implementation
- Regression test
- Root cause analysis in logs/
"""
        elif task_type == "refactoring":
            content += """
- Refactored code
- Preserved functionality (no behavior changes)
- Updated tests if needed
"""
        elif task_type == "deployment":
            content += """
- Deployment preparation
- Environment verification
- Rollback plan documentation
"""
        else:
            content += """
- Task completion
- Documentation of changes
- Any blocking issues noted
"""

        content += """
## Completion Checklist

- [ ] Policy documents read and followed
- [ ] Task completed successfully
- [ ] CURRENT_STATE.md updated
- [ ] logs/EXECUTION_LOG.md created
- [ ] No blocking issues (or documented in logs/BLOCKERS.md)
"""

        task_file.write_text(content)
        logger.info(f"Created task file: {task_file}")
        return task_file

    @staticmethod
    def cleanup_workspace(workspace: Path, keep_logs: bool = True) -> None:
        """Clean up a job workspace."""
        if not workspace.exists():
            return

        if keep_logs:
            # Archive logs before cleanup
            logs_dir = workspace / "logs"
            if logs_dir.exists():
                archive_dir = JOBS_BASE_DIR / "archives" / workspace.name
                archive_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(logs_dir, archive_dir / "logs", dirs_exist_ok=True)

        shutil.rmtree(workspace)
        logger.info(f"Cleaned up workspace: {workspace}")


# -----------------------------------------------------------------------------
# Claude CLI Executor
# -----------------------------------------------------------------------------
class ClaudeExecutor:
    """
    Executes Claude CLI jobs using the secure wrapper.
    """

    def __init__(self):
        self._process: Optional[asyncio.subprocess.Process] = None

    async def execute(self, job: ClaudeJob) -> ClaudeJob:
        """
        Execute a Claude CLI job.

        Returns the updated job with execution results.
        """
        logger.info(f"Executing job {job.job_id}")

        # Validate wrapper script exists
        if not WRAPPER_SCRIPT.exists():
            job.state = JobState.FAILED
            job.error_message = f"Wrapper script not found: {WRAPPER_SCRIPT}"
            job.completed_at = datetime.utcnow()
            return job

        # Create workspace if not exists
        if not job.workspace_dir or not job.workspace_dir.exists():
            job.workspace_dir = WorkspaceManager.create_workspace(
                job.job_id, job.project_name
            )

        # Create task file
        task_file = WorkspaceManager.create_task_file(
            job.workspace_dir,
            job.task_description,
            job.task_type,
            {
                "job_id": job.job_id,
                "project_name": job.project_name,
                "created_at": job.created_at.isoformat(),
                "created_by": job.created_by,
            }
        )

        # Update job state
        job.state = JobState.RUNNING
        job.started_at = datetime.utcnow()
        await job_queue.set_active(job.job_id)

        try:
            # Execute wrapper script
            process = await asyncio.create_subprocess_exec(
                str(WRAPPER_SCRIPT),
                job.job_id,
                str(job.workspace_dir),
                str(task_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(job.workspace_dir),
            )

            self._process = process

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=JOB_TIMEOUT
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                job.state = JobState.TIMEOUT
                job.error_message = f"Job timed out after {JOB_TIMEOUT}s"
                job.completed_at = datetime.utcnow()
                return job

            job.exit_code = process.returncode

            # Store output files
            job.stdout_file = job.workspace_dir / "logs" / "stdout.log"
            job.stderr_file = job.workspace_dir / "logs" / "stderr.log"

            if job.exit_code == 0:
                job.state = JobState.COMPLETED
                # Try to read result summary
                result_file = job.workspace_dir / "logs" / "result.txt"
                if result_file.exists():
                    job.result_summary = result_file.read_text()[:1000]  # Limit size
            else:
                job.state = JobState.FAILED
                job.error_message = f"Exit code: {job.exit_code}"
                if stderr:
                    job.error_message += f"\nStderr: {stderr.decode()[:500]}"

        except Exception as e:
            logger.error(f"Job execution error: {e}")
            job.state = JobState.FAILED
            job.error_message = str(e)

        finally:
            job.completed_at = datetime.utcnow()
            await job_queue.set_active(None)
            self._process = None

        return job

    async def cancel(self) -> bool:
        """Cancel the currently running job."""
        if self._process:
            self._process.kill()
            return True
        return False


# Global executor instance
executor = ClaudeExecutor()


# -----------------------------------------------------------------------------
# Job Scheduler
# -----------------------------------------------------------------------------
class JobScheduler:
    """
    Schedules and runs jobs from the queue.
    Runs as a background task.
    """

    def __init__(self):
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the job scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Job scheduler started")

    async def stop(self) -> None:
        """Stop the job scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Job scheduler stopped")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for queued jobs
                job = await job_queue.get_next()
                if job:
                    logger.info(f"Processing job {job.job_id}")
                    job = await executor.execute(job)
                    await job_queue.update_job(
                        job.job_id,
                        state=job.state,
                        started_at=job.started_at,
                        completed_at=job.completed_at,
                        exit_code=job.exit_code,
                        error_message=job.error_message,
                        result_summary=job.result_summary,
                    )
                    logger.info(f"Job {job.job_id} finished: {job.state.value}")
                else:
                    # No jobs, wait before checking again
                    await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(10)


# Global scheduler instance (legacy - kept for compatibility)
scheduler = JobScheduler()


# -----------------------------------------------------------------------------
# Phase 14.10: Persistent Job Store
# -----------------------------------------------------------------------------
class PersistentJobStore:
    """
    Persistent job storage for crash recovery.

    Stores job state to JSON file, enabling:
    - State recovery after scheduler restart
    - Audit trail of all jobs
    - Queue reconstruction
    """

    def __init__(self, state_file: Path = JOB_STATE_FILE):
        self._state_file = state_file
        self._lock = asyncio.Lock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """Ensure state file directory exists."""
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # May fail on non-VPS environments, which is OK
            logger.debug(f"Could not create state file directory: {e}")

    async def save_job(self, job: ClaudeJob) -> None:
        """Save or update a job in persistent storage."""
        async with self._lock:
            logger.debug(f"    save_job: Loading state...")
            state = await self._load_state()
            logger.debug(f"    save_job: State loaded, type(state['jobs'])={type(state.get('jobs', 'MISSING'))}")

            # Ensure jobs is a dict, not a list (fix corrupted state)
            if not isinstance(state.get("jobs"), dict):
                logger.warning(f"    save_job: FIXING corrupted state - jobs was {type(state.get('jobs'))}, resetting to dict")
                state["jobs"] = {}

            state["jobs"][job.job_id] = job.to_dict()
            state["last_updated"] = datetime.utcnow().isoformat()
            logger.debug(f"    save_job: Saving state...")
            await self._save_state(state)
            logger.debug(f"Persisted job {job.job_id} state: {job.state.value}")

    async def load_jobs(self) -> List[ClaudeJob]:
        """Load all jobs from persistent storage."""
        state = await self._load_state()
        jobs = []
        for job_data in state.get("jobs", {}).values():
            try:
                job = self._deserialize_job(job_data)
                jobs.append(job)
            except Exception as e:
                logger.warning(f"Failed to deserialize job: {e}")
        return jobs

    async def get_job(self, job_id: str) -> Optional[ClaudeJob]:
        """Load a specific job from persistent storage."""
        state = await self._load_state()
        job_data = state.get("jobs", {}).get(job_id)
        if job_data:
            return self._deserialize_job(job_data)
        return None

    async def remove_job(self, job_id: str) -> bool:
        """Remove a job from persistent storage."""
        async with self._lock:
            state = await self._load_state()
            if job_id in state.get("jobs", {}):
                del state["jobs"][job_id]
                await self._save_state(state)
                return True
            return False

    async def get_queued_jobs(self) -> List[ClaudeJob]:
        """
        Get all jobs in QUEUED state, ordered by priority.

        Phase 14.11: Changed from FIFO to priority queue ordering.
        Order: highest priority first, then oldest, then job_id for stability.
        """
        jobs = await self.load_jobs()
        queued = [j for j in jobs if j.state == JobState.QUEUED]
        # Phase 14.11: Priority ordering instead of FIFO
        queued.sort(key=lambda j: j.get_priority_sort_key())
        return queued

    async def get_active_jobs(self) -> List[ClaudeJob]:
        """Get all jobs in active states (PREPARING, RUNNING)."""
        jobs = await self.load_jobs()
        return [j for j in jobs if j.state in JobState.active_states()]

    async def _load_state(self) -> Dict[str, Any]:
        """Load state from file."""
        default_state = {"jobs": {}, "created_at": datetime.utcnow().isoformat()}

        if not self._state_file.exists():
            return default_state
        try:
            state = json.loads(self._state_file.read_text())

            # Ensure state has correct structure
            if not isinstance(state, dict):
                logger.warning(f"State file is not a dict (was {type(state)}), resetting")
                return default_state

            # Ensure jobs is a dict, not a list
            if "jobs" not in state or not isinstance(state["jobs"], dict):
                logger.warning(f"State jobs field missing or invalid (was {type(state.get('jobs'))}), resetting jobs")
                state["jobs"] = {}

            return state
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load state file: {e}")
            return default_state

    async def _save_state(self, state: Dict[str, Any]) -> None:
        """Save state to file atomically."""
        temp_file = self._state_file.with_suffix(".tmp")
        try:
            temp_file.write_text(json.dumps(state, indent=2, default=str))
            temp_file.replace(self._state_file)
        except IOError as e:
            logger.error(f"Failed to save state file: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def _deserialize_job(self, data: Dict[str, Any]) -> ClaudeJob:
        """Deserialize a job from dict."""
        return ClaudeJob(
            job_id=data["job_id"],
            project_name=data["project_name"],
            task_description=data["task_description"],
            task_type=data["task_type"],
            state=JobState(data["state"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            # Phase 14.11: Priority fields with defaults for backwards compatibility
            priority=data.get("priority", JobPriority.NORMAL.value),
            priority_escalations=data.get("priority_escalations", 0),
            last_escalation_at=datetime.fromisoformat(data["last_escalation_at"]) if data.get("last_escalation_at") else None,
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            exit_code=data.get("exit_code"),
            error_message=data.get("error_message"),
            workspace_dir=Path(data["workspace_dir"]) if data.get("workspace_dir") else None,
            created_by=data.get("created_by"),
            result_summary=data.get("result_summary"),
            # Phase 15.2: Aspect isolation with default for backwards compatibility
            aspect=data.get("aspect", "core"),
        )


# Global persistent store instance
persistent_store = PersistentJobStore()


# -----------------------------------------------------------------------------
# Phase 14.10: Claude Worker with Process Isolation
# -----------------------------------------------------------------------------
class ClaudeWorker:
    """
    Individual worker for Claude CLI execution.

    Features:
    - Process isolation via subprocess
    - Resource limits (CPU nice, memory)
    - Graceful shutdown
    - Execution logging
    """

    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self._process: Optional[asyncio.subprocess.Process] = None
        self._current_job: Optional[ClaudeJob] = None
        self._running = False

    @property
    def is_busy(self) -> bool:
        """Check if worker is currently executing a job."""
        return self._current_job is not None

    @property
    def current_job_id(self) -> Optional[str]:
        """Get ID of current job being executed."""
        return self._current_job.job_id if self._current_job else None

    async def execute(self, job: ClaudeJob) -> ClaudeJob:
        """
        Execute a Claude CLI job with resource limits.

        Phase 15.6: Now includes ExecutionGate check before execution.
        Execution will HARD FAIL if gate denies permission.

        Returns the updated job with execution results.
        """
        self._current_job = job
        self._running = True
        logger.info(f"Worker {self.worker_id}: Executing job {job.job_id}")

        # Import execution gate (lazy to avoid circular imports)
        try:
            from .execution_gate import (
                execution_gate,
                ExecutionRequest,
                get_execution_constraints_for_job,
            )
            gate_available = True
        except ImportError as e:
            logger.warning(f"ExecutionGate not available: {e}")
            gate_available = False

        try:
            # Validate wrapper script
            if not WRAPPER_SCRIPT.exists():
                job.state = JobState.FAILED
                job.error_message = f"Wrapper script not found: {WRAPPER_SCRIPT}"
                job.completed_at = datetime.utcnow()
                return job

            # Ensure workspace exists
            if not job.workspace_dir or not job.workspace_dir.exists():
                job.workspace_dir = WorkspaceManager.create_workspace(
                    job.job_id, job.project_name
                )

            # Phase 15.6: EXECUTION GATE CHECK
            if gate_available:
                request = ExecutionRequest(
                    job_id=job.job_id,
                    project_name=job.project_name,
                    aspect=job.aspect,
                    lifecycle_id=job.lifecycle_id or f"auto-{job.job_id}",
                    lifecycle_state=job.lifecycle_state,
                    requested_action=job.requested_action,
                    requesting_user_id=job.created_by or "system",
                    requesting_user_role=job.user_role,
                    workspace_path=str(job.workspace_dir),
                    task_description=job.task_description,
                )

                decision = execution_gate.evaluate(request)
                job.gate_allowed = decision.allowed
                job.gate_denied_reason = decision.denied_reason

                if not decision.allowed:
                    # HARD FAIL - execution gate denied
                    job.state = JobState.FAILED
                    job.error_message = f"EXECUTION GATE DENIED: {decision.denied_reason}"
                    job.completed_at = datetime.utcnow()
                    logger.error(
                        f"Worker {self.worker_id}: Job {job.job_id} GATE DENIED - "
                        f"{decision.denied_reason}"
                    )
                    await persistent_store.save_job(job)
                    return job

                logger.info(
                    f"Worker {self.worker_id}: Job {job.job_id} GATE ALLOWED - "
                    f"action={job.requested_action}, state={job.lifecycle_state}"
                )

                # Get execution constraints to pass to Claude
                execution_constraints = get_execution_constraints_for_job(
                    job.lifecycle_state,
                    job.user_role,
                )
            else:
                execution_constraints = None

            # Create task file with execution constraints
            task_metadata = {
                "job_id": job.job_id,
                "project_name": job.project_name,
                "created_at": job.created_at.isoformat(),
                "created_by": job.created_by,
                "worker_id": self.worker_id,
            }

            # Phase 15.6: Add execution constraints to metadata
            if execution_constraints:
                task_metadata["execution_constraints"] = execution_constraints

            task_file = WorkspaceManager.create_task_file(
                job.workspace_dir,
                job.task_description,
                job.task_type,
                task_metadata,
            )

            # Update job state
            job.state = JobState.RUNNING
            job.started_at = datetime.utcnow()

            # Persist state
            await persistent_store.save_job(job)

            # Build command with resource limits
            # Use nice for CPU priority and ulimit for memory
            cmd = self._build_command(job, task_file)

            # Execute
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(job.workspace_dir),
                preexec_fn=self._setup_resource_limits,
            )

            self._process = process

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=JOB_TIMEOUT
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                job.state = JobState.TIMEOUT
                job.error_message = f"Job timed out after {JOB_TIMEOUT}s"
                job.completed_at = datetime.utcnow()
                logger.warning(f"Worker {self.worker_id}: Job {job.job_id} timed out")
                # Log outcome to execution gate audit
                if gate_available:
                    execution_gate.log_execution_outcome(request, "TIMEOUT", job.error_message)
                return job

            job.exit_code = process.returncode

            # Store output files
            job.stdout_file = job.workspace_dir / "logs" / "stdout.log"
            job.stderr_file = job.workspace_dir / "logs" / "stderr.log"

            if job.exit_code == 0:
                job.state = JobState.COMPLETED
                # Try to read result summary
                result_file = job.workspace_dir / "logs" / "result.txt"
                if result_file.exists():
                    job.result_summary = result_file.read_text()[:1000]
                logger.info(f"Worker {self.worker_id}: Job {job.job_id} completed successfully")
                # Log success outcome
                if gate_available:
                    execution_gate.log_execution_outcome(request, "SUCCESS")
            else:
                job.state = JobState.FAILED
                job.error_message = f"Exit code: {job.exit_code}"
                if stderr:
                    job.error_message += f"\nStderr: {stderr.decode()[:500]}"
                logger.error(f"Worker {self.worker_id}: Job {job.job_id} failed: {job.error_message}")
                # Log failure outcome
                if gate_available:
                    execution_gate.log_execution_outcome(request, "FAILURE", job.error_message)

        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Job execution error: {e}")
            job.state = JobState.FAILED
            job.error_message = str(e)
            # Log exception outcome
            if gate_available:
                try:
                    execution_gate.log_execution_outcome(request, "FAILURE", str(e))
                except:
                    pass

        finally:
            job.completed_at = datetime.utcnow()
            self._process = None
            self._current_job = None
            self._running = False

            # Persist final state
            await persistent_store.save_job(job)

        return job

    def _build_command(self, job: ClaudeJob, task_file: Path) -> str:
        """Build shell command with nice prefix for CPU priority."""
        # Use nice to lower process priority
        return f"nice -n {WORKER_NICE_VALUE} {WRAPPER_SCRIPT} {job.job_id} {job.workspace_dir} {task_file}"

    def _setup_resource_limits(self) -> None:
        """
        Set up resource limits for the subprocess.
        Called in the child process before exec.
        """
        try:
            # Set memory limit (soft and hard)
            memory_bytes = WORKER_MEMORY_LIMIT_MB * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        except (ValueError, resource.error) as e:
            # Log but don't fail - resource limits may not be available
            pass

    async def cancel(self) -> bool:
        """Cancel the currently running job."""
        if self._process and self._current_job:
            logger.info(f"Worker {self.worker_id}: Cancelling job {self._current_job.job_id}")
            try:
                self._process.terminate()
                await asyncio.sleep(2)  # Grace period
                if self._process.returncode is None:
                    self._process.kill()
                return True
            except ProcessLookupError:
                pass
        return False

    async def shutdown(self) -> None:
        """Gracefully shutdown the worker."""
        await self.cancel()
        self._running = False


# -----------------------------------------------------------------------------
# Phase 14.10: Multi-Worker Scheduler
# -----------------------------------------------------------------------------
class MultiWorkerScheduler:
    """
    Multi-worker job scheduler with concurrency control.

    Features:
    - Maximum concurrent jobs enforcement (MAX_CONCURRENT_JOBS=3)
    - Priority-aware scheduling (Phase 14.11)
    - Starvation prevention with auto-escalation (Phase 14.11)
    - Process isolation per worker
    - Crash recovery from persistent state
    - Graceful shutdown
    - Auto-cleanup of old completed jobs
    """

    def __init__(self, max_workers: int = MAX_CONCURRENT_JOBS):
        self.max_workers = max_workers
        self._workers: List[ClaudeWorker] = []
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._starvation_task: Optional[asyncio.Task] = None  # Phase 14.11
        self._lock = asyncio.Lock()

        # Initialize workers
        for i in range(max_workers):
            self._workers.append(ClaudeWorker(worker_id=i))

        logger.info(f"MultiWorkerScheduler initialized with {max_workers} workers")

    async def start(self) -> None:
        """Start the scheduler and recover state."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True

        # Recover from crash - handle interrupted jobs
        await self._recover_state()

        # Start scheduler loop
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        # Start cleanup task
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        # Phase 14.11: Start starvation prevention task
        self._starvation_task = asyncio.create_task(self._starvation_prevention_loop())

        logger.info("MultiWorkerScheduler started")

    async def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self._running = False

        # Cancel scheduler task
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Phase 14.11: Cancel starvation prevention task
        if self._starvation_task:
            self._starvation_task.cancel()
            try:
                await self._starvation_task
            except asyncio.CancelledError:
                pass

        # Shutdown all workers
        for worker in self._workers:
            await worker.shutdown()

        logger.info("MultiWorkerScheduler stopped")

    async def enqueue_job(self, job: ClaudeJob) -> str:
        """Add a job to the queue."""
        try:
            # Step 1: Persist immediately
            logger.debug(f"  enqueue_job Step 1: Persisting job {job.job_id}...")
            await persistent_store.save_job(job)
            logger.debug(f"  enqueue_job Step 1: DONE")

            # Step 2: Also add to in-memory queue for legacy compatibility
            logger.debug(f"  enqueue_job Step 2: Adding to in-memory queue...")
            await job_queue.enqueue(job)
            logger.debug(f"  enqueue_job Step 2: DONE")

            # Step 3: Get queue position for logging
            logger.debug(f"  enqueue_job Step 3: Getting queue position...")
            position = await self._get_queue_position(job.job_id)
            logger.debug(f"  enqueue_job Step 3: DONE (position={position})")

            logger.info(f"Job {job.job_id} enqueued (position: {position})")
            return job.job_id
        except Exception as e:
            logger.error(f"  enqueue_job FAILED at step: {e.__class__.__name__}: {e}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            raise

    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive scheduler status.

        Phase 14.11: Enhanced to include priority and wait_time for queued jobs.
        """
        queued_jobs = await persistent_store.get_queued_jobs()
        all_jobs = await persistent_store.load_jobs()

        active_workers = [w for w in self._workers if w.is_busy]

        # Phase 14.11: Build queue details with priority, wait_time, state
        queue_details = [
            {
                "job_id": job.job_id,
                "project": job.project_name,
                "priority": job.priority,
                "priority_label": JobPriority.from_value(job.priority).name,
                "wait_time_minutes": round(job.get_wait_time_minutes(), 2),
                "state": job.state.value,
                "escalations": job.priority_escalations,
            }
            for job in queued_jobs
        ]

        return {
            "running": self._running,
            "max_workers": self.max_workers,
            "active_workers": len(active_workers),
            "available_workers": self.max_workers - len(active_workers),
            "queued_jobs": len(queued_jobs),
            "queue": queue_details,  # Phase 14.11: Detailed queue info
            "active_jobs": [w.current_job_id for w in active_workers if w.current_job_id],
            "total_jobs": len(all_jobs),
            "completed_jobs": sum(1 for j in all_jobs if j.state == JobState.COMPLETED),
            "failed_jobs": sum(1 for j in all_jobs if j.state == JobState.FAILED),
            "workers": [
                {
                    "id": w.worker_id,
                    "busy": w.is_busy,
                    "current_job": w.current_job_id,
                }
                for w in self._workers
            ],
        }

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a job (queued or running)."""
        # Check if job is running
        for worker in self._workers:
            if worker.current_job_id == job_id:
                success = await worker.cancel()
                if success:
                    job = await persistent_store.get_job(job_id)
                    if job:
                        job.state = JobState.CANCELLED
                        job.completed_at = datetime.utcnow()
                        job.error_message = "Cancelled by user"
                        await persistent_store.save_job(job)
                return success

        # Check if job is queued
        job = await persistent_store.get_job(job_id)
        if job and job.state == JobState.QUEUED:
            job.state = JobState.CANCELLED
            job.completed_at = datetime.utcnow()
            job.error_message = "Cancelled before execution"
            await persistent_store.save_job(job)
            return True

        return False

    async def _scheduler_loop(self) -> None:
        """Main scheduling loop - assigns jobs to available workers."""
        while self._running:
            try:
                async with self._lock:
                    # Find available workers
                    available_workers = [w for w in self._workers if not w.is_busy]

                    if available_workers:
                        # Get queued jobs
                        queued_jobs = await persistent_store.get_queued_jobs()

                        # Phase 15.2: Get currently active jobs to enforce aspect isolation
                        active_jobs = await persistent_store.get_active_jobs()
                        active_aspects = {
                            (job.project_name, job.aspect)
                            for job in active_jobs
                        }

                        # Filter queued jobs to enforce one active job per project+aspect
                        eligible_jobs = []
                        for job in queued_jobs:
                            job_key = (job.project_name, job.aspect)
                            if job_key not in active_aspects:
                                eligible_jobs.append(job)
                                # Mark as "will be active" to prevent double-scheduling same aspect
                                active_aspects.add(job_key)
                            else:
                                logger.debug(
                                    f"Job {job.job_id} blocked - active job exists for "
                                    f"{job.project_name}/{job.aspect}"
                                )

                        # Assign eligible jobs to workers (up to available capacity)
                        for worker, job in zip(available_workers, eligible_jobs):
                            # Mark as preparing to prevent double-assignment
                            job.state = JobState.PREPARING
                            await persistent_store.save_job(job)

                            # Execute asynchronously
                            asyncio.create_task(self._execute_job(worker, job))

                            logger.info(
                                f"Assigned job {job.job_id} to worker {worker.worker_id} "
                                f"(project: {job.project_name}, aspect: {job.aspect})"
                            )

                # Wait before next scheduling cycle
                await asyncio.sleep(2)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(5)

    async def _execute_job(self, worker: ClaudeWorker, job: ClaudeJob) -> None:
        """Execute a job on a worker and update state."""
        try:
            updated_job = await worker.execute(job)

            # Update in-memory queue too
            await job_queue.update_job(
                updated_job.job_id,
                state=updated_job.state,
                started_at=updated_job.started_at,
                completed_at=updated_job.completed_at,
                exit_code=updated_job.exit_code,
                error_message=updated_job.error_message,
                result_summary=updated_job.result_summary,
            )

            logger.info(f"Job {updated_job.job_id} finished: {updated_job.state.value}")

        except Exception as e:
            logger.error(f"Error executing job {job.job_id}: {e}")
            job.state = JobState.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            await persistent_store.save_job(job)

    async def _recover_state(self) -> None:
        """Recover state after restart - handle interrupted jobs."""
        logger.info("Recovering scheduler state...")

        # Get jobs that were in active states when we crashed
        active_jobs = await persistent_store.get_active_jobs()

        for job in active_jobs:
            logger.warning(f"Found interrupted job {job.job_id} in state {job.state.value}")

            # Mark as failed with recovery message
            job.state = JobState.FAILED
            job.error_message = "Job interrupted by scheduler restart"
            job.completed_at = datetime.utcnow()
            await persistent_store.save_job(job)

            # Clean up workspace (keep logs)
            if job.workspace_dir and job.workspace_dir.exists():
                WorkspaceManager.cleanup_workspace(job.workspace_dir, keep_logs=True)

        if active_jobs:
            logger.info(f"Recovered {len(active_jobs)} interrupted jobs")

        # Load queued jobs into memory queue for legacy compatibility
        queued_jobs = await persistent_store.get_queued_jobs()
        for job in queued_jobs:
            await job_queue.enqueue(job)

        logger.info(f"Loaded {len(queued_jobs)} queued jobs")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up old completed jobs."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour

                all_jobs = await persistent_store.load_jobs()
                cutoff = datetime.utcnow() - timedelta(hours=JOB_CLEANUP_AFTER_HOURS)

                cleaned = 0
                for job in all_jobs:
                    if job.state in JobState.terminal_states() and job.completed_at and job.completed_at < cutoff:
                        # Clean up workspace
                        if job.workspace_dir and job.workspace_dir.exists():
                            WorkspaceManager.cleanup_workspace(job.workspace_dir, keep_logs=True)

                        # Remove from persistent store
                        await persistent_store.remove_job(job.job_id)
                        cleaned += 1

                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} old completed jobs")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _starvation_prevention_loop(self) -> None:
        """
        Phase 14.11: Periodically check for starving jobs and escalate priority.

        If a job waits longer than STARVATION_THRESHOLD_MINUTES (30 min) in QUEUED state:
        - Increase priority by PRIORITY_ESCALATION_AMOUNT (+10)
        - Log escalation in audit trail
        - Cap escalation at PRIORITY_ESCALATION_CAP (HIGH=75)

        This ensures lower-priority jobs eventually get processed.
        """
        while self._running:
            try:
                # Check every 5 minutes
                await asyncio.sleep(300)

                queued_jobs = await persistent_store.get_queued_jobs()
                now = datetime.utcnow()

                for job in queued_jobs:
                    wait_minutes = job.get_wait_time_minutes()

                    # Check if job has been waiting too long
                    if wait_minutes >= STARVATION_THRESHOLD_MINUTES:
                        # Calculate new priority
                        new_priority = job.priority + PRIORITY_ESCALATION_AMOUNT

                        # Check if escalation is allowed (cap at HIGH=75)
                        if JobPriority.can_escalate_to(job.priority, new_priority):
                            old_priority = job.priority
                            job.priority = min(new_priority, PRIORITY_ESCALATION_CAP)
                            job.priority_escalations += 1
                            job.last_escalation_at = now

                            # Persist the change
                            await persistent_store.save_job(job)

                            # Log the escalation to audit trail
                            await self._log_priority_escalation(
                                job_id=job.job_id,
                                old_priority=old_priority,
                                new_priority=job.priority,
                                wait_minutes=wait_minutes,
                                escalation_count=job.priority_escalations,
                            )

                            logger.info(
                                f"Starvation prevention: Job {job.job_id} escalated "
                                f"from {old_priority} to {job.priority} "
                                f"(waited {wait_minutes:.1f} min, escalation #{job.priority_escalations})"
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Starvation prevention loop error: {e}")

    async def _log_priority_escalation(
        self,
        job_id: str,
        old_priority: int,
        new_priority: int,
        wait_minutes: float,
        escalation_count: int,
    ) -> None:
        """
        Phase 14.11: Log priority escalation to audit trail.

        Writes to PRIORITY_AUDIT_LOG in append-only format.
        """
        try:
            PRIORITY_AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.utcnow().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "event": "priority_escalation",
                "job_id": job_id,
                "old_priority": old_priority,
                "new_priority": new_priority,
                "wait_minutes": round(wait_minutes, 2),
                "escalation_count": escalation_count,
            }

            # Append to audit log
            with open(PRIORITY_AUDIT_LOG, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        except IOError as e:
            logger.warning(f"Failed to write priority audit log: {e}")

    async def _get_queue_position(self, job_id: str) -> int:
        """Get position of a job in the queue (1-indexed)."""
        queued_jobs = await persistent_store.get_queued_jobs()
        for i, job in enumerate(queued_jobs):
            if job.job_id == job_id:
                return i + 1
        return -1


# Global multi-worker scheduler instance
multi_scheduler = MultiWorkerScheduler()


# -----------------------------------------------------------------------------
# Public API (Phase 14.10: Updated for Multi-Worker Scheduler)
# -----------------------------------------------------------------------------
async def create_job(
    project_name: str,
    task_description: str,
    task_type: str = "feature_development",
    created_by: Optional[str] = None,
    priority: int = JobPriority.NORMAL.value,
    aspect: str = "core",
    # Phase 15.6: Execution Gate parameters
    lifecycle_id: Optional[str] = None,
    lifecycle_state: str = "development",
    requested_action: str = "write_code",
    user_role: str = "developer",
) -> ClaudeJob:
    """
    Create and enqueue a new Claude CLI job.

    Phase 14.10: Now uses multi-worker scheduler with persistent state.
    Phase 14.11: Added priority parameter for scheduling.
    Phase 15.2: Added aspect parameter for scheduler isolation.
    Phase 15.6: Added lifecycle and execution gate parameters.

    Args:
        project_name: Name of the project
        task_description: Description of the task
        task_type: Type of task (feature_development, bug_fix, refactoring, deployment)
        created_by: User ID or identifier of creator
        priority: Job priority (default: NORMAL=50). Use JobPriority enum values.
        aspect: Project aspect for scheduler isolation (core, backend, frontend_web, etc.)
        lifecycle_id: Associated lifecycle ID for the project/aspect
        lifecycle_state: Current lifecycle state (development, testing, etc.)
        requested_action: Action being requested (read_code, write_code, run_tests, etc.)
        user_role: Role of requesting user (owner, admin, developer, tester, viewer)

    Returns:
        The created ClaudeJob
    """
    logger.info("=" * 50)
    logger.info("CREATE_JOB - START")
    logger.info(f"  project_name: {project_name}")
    logger.info(f"  task_type: {task_type}")
    logger.info(f"  task_description length: {len(task_description)} chars")
    logger.info(f"  created_by: {created_by}")
    logger.info(f"  priority: {priority}")
    logger.info(f"  aspect: {aspect}")
    logger.info(f"  lifecycle_state: {lifecycle_state}")
    logger.info(f"  requested_action: {requested_action}")
    logger.info(f"  user_role: {user_role}")

    # Phase 14.11: Validate and normalize priority
    validated_priority = max(JobPriority.LOW.value, min(priority, JobPriority.EMERGENCY.value))
    logger.info(f"  Validated priority: {validated_priority}")

    # Phase 15.2: Validate aspect
    valid_aspects = ["core", "backend", "frontend_web", "frontend_mobile", "admin", "custom"]
    validated_aspect = aspect if aspect in valid_aspects else "core"
    logger.info(f"  Validated aspect: {validated_aspect}")

    # Phase 15.6: Validate lifecycle state and action
    valid_lifecycle_states = [
        "created", "planning", "development", "testing", "awaiting_feedback",
        "fixing", "ready_for_production", "deployed", "rejected", "archived"
    ]
    validated_lifecycle_state = lifecycle_state if lifecycle_state in valid_lifecycle_states else "development"

    valid_actions = ["read_code", "write_code", "run_tests", "commit", "push", "deploy_test", "deploy_prod"]
    validated_action = requested_action if requested_action in valid_actions else "write_code"

    valid_roles = ["owner", "admin", "developer", "tester", "viewer"]
    validated_role = user_role if user_role in valid_roles else "developer"

    logger.info("  Creating ClaudeJob object...")
    job = ClaudeJob(
        job_id=str(uuid.uuid4()),
        project_name=project_name,
        task_description=task_description,
        task_type=task_type,
        state=JobState.QUEUED,
        created_at=datetime.utcnow(),
        created_by=created_by,
        priority=validated_priority,
        aspect=validated_aspect,
        # Phase 15.6: Execution Gate fields
        lifecycle_id=lifecycle_id,
        lifecycle_state=validated_lifecycle_state,
        requested_action=validated_action,
        user_role=validated_role,
    )
    logger.info(f"  Job created: job_id={job.job_id}")

    # Create workspace early so it's ready
    logger.info("  Creating workspace...")
    try:
        job.workspace_dir = WorkspaceManager.create_workspace(job.job_id, project_name)
        logger.info(f"  Workspace created: {job.workspace_dir}")
    except Exception as e:
        logger.error(f"  Workspace creation FAILED: {e}", exc_info=True)
        raise

    # Use multi-worker scheduler (Phase 14.10)
    logger.info("  Checking scheduler status...")
    if not multi_scheduler._running:
        logger.error("  SCHEDULER NOT RUNNING - cannot enqueue job")
        raise RuntimeError("Job scheduler is not running")

    logger.info("  Enqueuing job to scheduler...")
    try:
        await multi_scheduler.enqueue_job(job)
        logger.info(f"  Job enqueued successfully")
    except Exception as e:
        logger.error(f"  Job enqueue FAILED: {e}", exc_info=True)
        raise

    logger.info("CREATE_JOB - SUCCESS")
    logger.info(f"  Job ID: {job.job_id}")
    logger.info(f"  State: {job.state.value}")
    logger.info("=" * 50)
    return job


async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a job."""
    # Try persistent store first (Phase 14.10)
    job = await persistent_store.get_job(job_id)
    if job:
        return job.to_dict()

    # Fallback to in-memory queue
    job = await job_queue.get_job(job_id)
    if job:
        return job.to_dict()
    return None


async def list_jobs(
    state: Optional[str] = None,
    project: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """List jobs with optional filtering."""
    # Use persistent store (Phase 14.10)
    jobs = await persistent_store.load_jobs()

    # Apply filters
    if state:
        job_state = JobState(state)
        jobs = [j for j in jobs if j.state == job_state]
    if project:
        jobs = [j for j in jobs if j.project_name == project]

    # Sort by created_at descending
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return [j.to_dict() for j in jobs[:limit]]


async def get_queue_status() -> Dict[str, Any]:
    """
    Get current queue status.

    Phase 14.10: Returns enhanced status from multi-worker scheduler.
    """
    return await multi_scheduler.get_status()


async def cancel_job(job_id: str) -> bool:
    """
    Cancel a running or queued job.

    Phase 14.10: Uses multi-worker scheduler for proper worker management.
    """
    return await multi_scheduler.cancel_job(job_id)


async def get_scheduler_status() -> Dict[str, Any]:
    """
    Get detailed multi-worker scheduler status.

    Phase 14.10: New API for monitoring concurrent job execution.

    Returns:
        Dict with scheduler state including:
        - running: bool
        - max_workers: int (3)
        - active_workers: int (0-3)
        - available_workers: int
        - queued_jobs: int
        - active_jobs: list of job IDs currently running
        - workers: detailed worker state
    """
    return await multi_scheduler.get_status()


async def start_scheduler() -> None:
    """
    Start the multi-worker scheduler.

    Phase 14.10: Should be called during controller startup.
    Handles state recovery from persistent storage.
    """
    await multi_scheduler.start()
    logger.info("Multi-worker scheduler started")


async def stop_scheduler() -> None:
    """
    Stop the multi-worker scheduler gracefully.

    Phase 14.10: Should be called during controller shutdown.
    Allows running jobs to complete before stopping.
    """
    await multi_scheduler.stop()
    logger.info("Multi-worker scheduler stopped")


async def check_claude_availability() -> Dict[str, Any]:
    """
    Check if Claude CLI is available, authenticated, AND can execute prompts.

    Phase 15.7: Updated to perform REAL execution test.
    Phase 19 fix: Added caching to prevent sporadic timeout issues.

    CRITICAL DISTINCTION:
    - `claude --version` only checks if CLI binary is installed
    - Actual prompt execution requires valid authentication (API key or setup-token)
    - OAuth session from `~/.claude.json` does NOT work for `--print` mode
    - This function now performs a REAL execution test to verify end-to-end functionality

    Authentication States:
    1. NOT_INSTALLED: CLI binary not found
    2. INSTALLED_NOT_AUTHENTICATED: CLI installed, version works, but execution fails
    3. AUTHENTICATED_FOR_AUTOMATION: CLI can execute prompts with --print

    Returns status dict with:
    - available: bool (True ONLY if CLI can execute real prompts)
    - installed: bool (True if CLI binary exists)
    - version: str or None
    - authenticated: bool (True ONLY if real execution test passes)
    - can_execute: bool (CRITICAL: True only if real prompt execution works)
    - auth_type: str ('api_key', 'setup_token', 'none')
    - api_key_configured: bool
    - wrapper_exists: bool
    - error: str or None
    - execution_test_output: str or None (output from real test)
    """
    global _claude_availability_cache, _claude_availability_cache_time

    # Phase 19 fix: Return cached result if available and still valid
    # This prevents sporadic timeout issues from causing /health to show CLI as unavailable
    if _claude_availability_cache and _claude_availability_cache_time:
        cache_age = (datetime.utcnow() - _claude_availability_cache_time).total_seconds()
        if cache_age < CLAUDE_AVAILABILITY_CACHE_TTL_SECONDS:
            # Only return cached result if it was successful
            # If cached result shows unavailable, we should recheck
            if _claude_availability_cache.get("available"):
                logger.debug(f"Returning cached Claude availability (age: {cache_age:.1f}s)")
                return _claude_availability_cache.copy()

    result = {
        "available": False,
        "installed": False,
        "version": None,
        "authenticated": False,
        "can_execute": False,  # Phase 15.7: Critical new field
        "auth_type": "none",
        "api_key_configured": False,
        "wrapper_exists": False,
        "error": None,
        "execution_test_output": None,
    }

    # Check wrapper script
    result["wrapper_exists"] = WRAPPER_SCRIPT.exists()

    # Step 1: Check if Claude CLI binary exists and get version
    try:
        process = await asyncio.create_subprocess_exec(
            "claude", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=15  # Phase 19 fix: Increased from 5s to 15s to handle CLI initialization
            )
        except asyncio.TimeoutError:
            # Kill subprocess on timeout
            try:
                process.kill()
                await process.wait()
            except Exception:
                pass
            result["error"] = "Claude CLI version check timed out"
            logger.warning("Claude CLI --version timed out after 15s - subprocess killed")
            return result

        if process.returncode == 0:
            result["installed"] = True
            result["version"] = stdout.decode().strip()
            logger.info(f"Claude CLI detected: {result['version']}")
        else:
            error_msg = stderr.decode().strip()
            result["error"] = f"Claude CLI error: {error_msg}"
            logger.warning(f"Claude CLI returned error: {error_msg}")
            return result

    except FileNotFoundError:
        result["error"] = "Claude CLI not installed"
        logger.warning("Claude CLI not found in PATH")
        return result
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"Error checking Claude CLI: {e}")
        return result

    # Step 2: Check for API key (environment variable)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key and len(api_key) > 10:
        result["api_key_configured"] = True
        logger.info("ANTHROPIC_API_KEY environment variable is set")

    # Step 3: REAL EXECUTION TEST (Phase 15.7)
    # This is the ONLY way to know if Claude CLI can actually work
    # `--version` passing means nothing for actual execution
    try:
        logger.info("Running real execution test...")

        # Use a simple prompt that should return a predictable response
        test_prompt = "respond with exactly: CLAUDE_CLI_OK"

        process = await asyncio.create_subprocess_exec(
            "claude", "--print", test_prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=30  # 30 second timeout for test
            )
        except asyncio.TimeoutError:
            # Kill the subprocess on timeout to prevent zombie processes
            try:
                process.kill()
                await process.wait()  # Ensure process is fully terminated
            except Exception:
                pass  # Process may already be dead
            result["error"] = "Execution test timed out after 30 seconds"
            logger.warning("Claude CLI execution test timed out - subprocess killed")
            return result

        stdout_text = stdout.decode().strip()
        stderr_text = stderr.decode().strip()

        result["execution_test_output"] = stdout_text[:200] if stdout_text else stderr_text[:200]

        if process.returncode == 0 and "CLAUDE_CLI_OK" in stdout_text:
            # SUCCESS! Claude CLI can execute prompts
            result["can_execute"] = True
            result["authenticated"] = True
            result["available"] = True

            # Determine auth type
            if result["api_key_configured"]:
                result["auth_type"] = "api_key"
            else:
                result["auth_type"] = "setup_token"  # Must be setup-token auth

            logger.info(
                f"Claude CLI VERIFIED EXECUTABLE: version={result['version']}, "
                f"auth_type={result['auth_type']}"
            )

            # Phase 19 fix: Cache successful result to prevent sporadic timeout issues
            _claude_availability_cache = result.copy()
            _claude_availability_cache_time = datetime.utcnow()
            logger.debug("Cached Claude availability result")

            return result

        # Execution failed - parse the error
        if "Invalid API key" in stderr_text or "Invalid API key" in stdout_text:
            result["error"] = (
                "Claude CLI NOT AUTHENTICATED for automation. "
                "Options: 1) Set ANTHROPIC_API_KEY environment variable, "
                "2) Run 'claude setup-token' interactively"
            )
            logger.warning("Claude CLI installed but not authenticated for automation")
        elif "Please run /login" in stderr_text or "Please run /login" in stdout_text:
            result["error"] = (
                "Claude CLI requires authentication. "
                "Run 'claude setup-token' to configure long-lived auth for automation."
            )
            logger.warning("Claude CLI requires login for automation")
        else:
            result["error"] = f"Execution test failed: exit_code={process.returncode}, stderr={stderr_text[:200]}"
            logger.warning(f"Claude CLI execution test failed: {result['error']}")

    except Exception as e:
        result["error"] = f"Execution test error: {str(e)}"
        logger.error(f"Error during execution test: {e}")

    return result
