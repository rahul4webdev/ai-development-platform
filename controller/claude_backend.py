"""
Claude CLI Execution Backend
Phase 14.3: Controller Integration

This module integrates Claude CLI as an execution backend for the controller.
It manages job workspaces, invokes the secure wrapper, and tracks job state.

IMPORTANT: This module does NOT replace existing controller functionality.
It ADDS Claude CLI as a new execution backend alongside the existing flow.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml

logger = logging.getLogger("claude_backend")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
JOBS_BASE_DIR = Path(os.getenv("CLAUDE_JOBS_DIR", "/home/aitesting.mybd.in/jobs"))
DOCS_DIR = Path(os.getenv("CLAUDE_DOCS_DIR", "/home/aitesting.mybd.in/public_html/docs"))
WRAPPER_SCRIPT = Path("/home/aitesting.mybd.in/public_html/scripts/run_claude_job.sh")
JOB_TIMEOUT = int(os.getenv("CLAUDE_TIMEOUT", "600"))

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


# -----------------------------------------------------------------------------
# Job State Management
# -----------------------------------------------------------------------------
class JobState(str, Enum):
    """Job execution states."""
    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ClaudeJob:
    """Represents a Claude CLI execution job."""
    job_id: str
    project_name: str
    task_description: str
    task_type: str
    state: JobState
    created_at: datetime
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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "job_id": self.job_id,
            "project_name": self.project_name,
            "task_description": self.task_description,
            "task_type": self.task_type,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "exit_code": self.exit_code,
            "error_message": self.error_message,
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir else None,
            "created_by": self.created_by,
            "result_summary": self.result_summary,
        }


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


# Global scheduler instance
scheduler = JobScheduler()


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
async def create_job(
    project_name: str,
    task_description: str,
    task_type: str = "feature_development",
    created_by: Optional[str] = None
) -> ClaudeJob:
    """
    Create and enqueue a new Claude CLI job.

    Args:
        project_name: Name of the project
        task_description: Description of the task
        task_type: Type of task (feature_development, bug_fix, refactoring, deployment)
        created_by: User ID or identifier of creator

    Returns:
        The created ClaudeJob
    """
    job = ClaudeJob(
        job_id=str(uuid.uuid4()),
        project_name=project_name,
        task_description=task_description,
        task_type=task_type,
        state=JobState.QUEUED,
        created_at=datetime.utcnow(),
        created_by=created_by,
    )

    # Create workspace early so it's ready
    job.workspace_dir = WorkspaceManager.create_workspace(job.job_id, project_name)

    await job_queue.enqueue(job)

    logger.info(f"Created job {job.job_id} for project {project_name}")
    return job


async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get the status of a job."""
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
    job_state = JobState(state) if state else None
    jobs = await job_queue.list_jobs(state=job_state, project=project, limit=limit)
    return [j.to_dict() for j in jobs]


async def get_queue_status() -> Dict[str, Any]:
    """Get current queue status."""
    return await job_queue.get_queue_status()


async def cancel_job(job_id: str) -> bool:
    """Cancel a running job."""
    job = await job_queue.get_job(job_id)
    if not job:
        return False

    if job.state == JobState.RUNNING:
        success = await executor.cancel()
        if success:
            await job_queue.update_job(
                job_id,
                state=JobState.CANCELLED,
                completed_at=datetime.utcnow(),
                error_message="Cancelled by user"
            )
        return success
    elif job.state == JobState.QUEUED:
        await job_queue.update_job(
            job_id,
            state=JobState.CANCELLED,
            completed_at=datetime.utcnow(),
            error_message="Cancelled before execution"
        )
        return True

    return False


async def check_claude_availability() -> Dict[str, Any]:
    """
    Check if Claude CLI is available and configured.

    Returns status dict with:
    - available: bool
    - version: str or None
    - api_key_configured: bool
    - error: str or None
    """
    result = {
        "available": False,
        "version": None,
        "api_key_configured": False,
        "wrapper_exists": False,
        "error": None,
    }

    # Check wrapper script
    result["wrapper_exists"] = WRAPPER_SCRIPT.exists()

    # Check Claude CLI
    try:
        process = await asyncio.create_subprocess_exec(
            "claude", "--version",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()

        if process.returncode == 0:
            result["available"] = True
            result["version"] = stdout.decode().strip()
        else:
            result["error"] = f"Claude CLI error: {stderr.decode()}"

    except FileNotFoundError:
        result["error"] = "Claude CLI not installed"
    except Exception as e:
        result["error"] = str(e)

    # Check API key (from environment)
    result["api_key_configured"] = bool(os.getenv("ANTHROPIC_API_KEY"))

    return result
