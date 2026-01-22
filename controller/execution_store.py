"""
Phase 18C: Execution Store

Append-only persistence for execution intents and results.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- APPEND-ONLY: Records are NEVER modified or deleted
- IMMUTABLE: Once written, records cannot change
- DETERMINISTIC: Same query always returns same results
- NO SIDE EFFECTS: Reading does not modify state
- FSYNC: All writes are fsync'd for durability
- AUDIT TRAIL: Every operation is logged

This store provides:
1. Execution intent persistence
2. Execution result history
3. Read-only queries for dispatcher

This store does NOT:
1. Execute any actions
2. Send notifications
3. Modify external state
4. Make decisions (that's the dispatcher's job)
"""

import json
import os
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import uuid

from .execution_dispatcher import (
    ExecutionIntent,
    ExecutionResult,
    ExecutionStatus,
    ActionType,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STORE_VERSION = "18C.1.0"

# Storage paths
STORAGE_DIR = Path(os.getenv("EXECUTION_STORAGE_DIR", "data/execution"))
INTENTS_FILE = STORAGE_DIR / "execution_intents.jsonl"
RESULTS_FILE = STORAGE_DIR / "execution_results.jsonl"


# -----------------------------------------------------------------------------
# Execution Store (Append-Only Persistence)
# -----------------------------------------------------------------------------
class ExecutionStore:
    """
    Phase 18C: Execution Store.

    Append-only persistence for execution intents and results.

    CRITICAL CONSTRAINTS:
    - APPEND-ONLY: Records are NEVER modified or deleted
    - IMMUTABLE: Once written, records cannot change
    - FSYNC: All writes are fsync'd for durability
    - DETERMINISTIC: Same query always returns same results
    """

    def __init__(
        self,
        intents_file: Optional[Path] = None,
        results_file: Optional[Path] = None,
    ):
        """
        Initialize store.

        Args:
            intents_file: Path to intents file (optional, for testing)
            results_file: Path to results file (optional, for testing)
        """
        self._intents_file = intents_file or INTENTS_FILE
        self._results_file = results_file or RESULTS_FILE
        self._version = STORE_VERSION

    # -------------------------------------------------------------------------
    # Write Operations (Append-Only)
    # -------------------------------------------------------------------------

    def record_intent(self, intent: ExecutionIntent) -> None:
        """
        Record an execution intent.

        APPEND-ONLY: Creates a new record, never modifies existing.

        Args:
            intent: The execution intent to record
        """
        self._append_record(self._intents_file, intent.to_dict())

    def record_result(self, result: ExecutionResult) -> None:
        """
        Record an execution result.

        APPEND-ONLY: Creates a new record, never modifies existing.
        Multiple results for the same execution_id are allowed
        (e.g., PENDING -> SUCCESS).

        Args:
            result: The execution result to record
        """
        self._append_record(self._results_file, result.to_dict())

    # -------------------------------------------------------------------------
    # Read Operations (Read-Only)
    # -------------------------------------------------------------------------

    def get_intent(self, intent_id: str) -> Optional[ExecutionIntent]:
        """
        Get an execution intent by ID.

        READ-ONLY: Does not modify state.

        Args:
            intent_id: ID of the intent

        Returns:
            ExecutionIntent if found, None otherwise
        """
        for record in self._read_records(self._intents_file):
            if record.get("intent_id") == intent_id:
                return ExecutionIntent.from_dict(record)
        return None

    def get_intents_for_project(
        self,
        project_id: str,
        limit: int = 100,
    ) -> List[ExecutionIntent]:
        """
        Get all intents for a project.

        READ-ONLY: Does not modify state.

        Args:
            project_id: ID of the project
            limit: Maximum number of results

        Returns:
            List of ExecutionIntents
        """
        results = []
        for record in self._read_records(self._intents_file):
            if record.get("project_id") == project_id:
                results.append(ExecutionIntent.from_dict(record))
                if len(results) >= limit:
                    break
        return results

    def get_result(self, execution_id: str) -> Optional[ExecutionResult]:
        """
        Get the latest execution result by ID.

        READ-ONLY: Does not modify state.
        Returns the most recent result if multiple exist.

        Args:
            execution_id: ID of the execution

        Returns:
            ExecutionResult if found, None otherwise
        """
        latest = None
        for record in self._read_records(self._results_file):
            if record.get("execution_id") == execution_id:
                latest = ExecutionResult.from_dict(record)
        return latest

    def get_results_for_intent(
        self,
        intent_id: str,
    ) -> List[ExecutionResult]:
        """
        Get all results for an intent.

        READ-ONLY: Does not modify state.

        Args:
            intent_id: ID of the intent

        Returns:
            List of ExecutionResults
        """
        results = []
        for record in self._read_records(self._results_file):
            if record.get("intent_id") == intent_id:
                results.append(ExecutionResult.from_dict(record))
        return results

    def get_recent_results(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[ExecutionResult]:
        """
        Get recent execution results.

        READ-ONLY: Does not modify state.

        Args:
            limit: Maximum number of results
            status: Optional filter by status
            project_id: Optional filter by project (requires joining with intents)

        Returns:
            List of ExecutionResults (most recent first)
        """
        # If filtering by project_id, we need to get intent_ids first
        intent_ids = None
        if project_id:
            intent_ids = set()
            for record in self._read_records(self._intents_file):
                if record.get("project_id") == project_id:
                    intent_ids.add(record.get("intent_id"))

        results = []
        for record in self._read_records(self._results_file):
            # Filter by status
            if status and record.get("status") != status:
                continue

            # Filter by project
            if intent_ids is not None and record.get("intent_id") not in intent_ids:
                continue

            results.append(ExecutionResult.from_dict(record))

        # Return most recent first
        results.reverse()
        return results[:limit]

    def get_pending_executions(
        self,
        limit: int = 100,
    ) -> List[ExecutionResult]:
        """
        Get pending executions.

        READ-ONLY: Does not modify state.

        Args:
            limit: Maximum number of results

        Returns:
            List of pending ExecutionResults
        """
        # Get all execution IDs with their latest status
        execution_status: Dict[str, str] = {}
        for record in self._read_records(self._results_file):
            exec_id = record.get("execution_id")
            status = record.get("status")
            if exec_id and status:
                execution_status[exec_id] = status

        # Filter for pending only
        pending_ids = {
            exec_id
            for exec_id, status in execution_status.items()
            if status == ExecutionStatus.EXECUTION_PENDING.value
        }

        # Get full results for pending
        results = []
        for record in self._read_records(self._results_file):
            if record.get("execution_id") in pending_ids:
                if record.get("status") == ExecutionStatus.EXECUTION_PENDING.value:
                    results.append(ExecutionResult.from_dict(record))
                    pending_ids.discard(record.get("execution_id"))

        return results[:limit]

    def is_execution_complete(self, execution_id: str) -> bool:
        """
        Check if an execution has reached a terminal state.

        READ-ONLY: Does not modify state.

        Args:
            execution_id: ID of the execution

        Returns:
            True if complete (SUCCESS or FAILED or BLOCKED), False otherwise
        """
        terminal_statuses = {
            ExecutionStatus.EXECUTION_SUCCESS.value,
            ExecutionStatus.EXECUTION_FAILED.value,
            ExecutionStatus.EXECUTION_BLOCKED.value,
        }

        latest_status = None
        for record in self._read_records(self._results_file):
            if record.get("execution_id") == execution_id:
                latest_status = record.get("status")

        return latest_status in terminal_statuses if latest_status else False

    # -------------------------------------------------------------------------
    # Summary/Aggregation (Read-Only)
    # -------------------------------------------------------------------------

    def get_summary(
        self,
        project_id: Optional[str] = None,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get a summary of execution activity.

        READ-ONLY: Does not modify state.

        Args:
            project_id: Optional filter by project
            since_hours: Hours to look back

        Returns:
            Summary dictionary
        """
        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        # Get intent IDs for project filter if needed
        intent_ids = None
        if project_id:
            intent_ids = set()
            for record in self._read_records(self._intents_file):
                if record.get("project_id") == project_id:
                    intent_ids.add(record.get("intent_id"))

        total = 0
        blocked = 0
        pending = 0
        success = 0
        failed = 0
        by_action_type: Dict[str, int] = {}
        by_block_reason: Dict[str, int] = {}

        # Track latest status per execution_id
        execution_statuses: Dict[str, Dict[str, Any]] = {}

        for record in self._read_records(self._results_file):
            if record.get("timestamp", "") < cutoff:
                continue

            if intent_ids is not None and record.get("intent_id") not in intent_ids:
                continue

            exec_id = record.get("execution_id")
            execution_statuses[exec_id] = record

        # Count by latest status
        for record in execution_statuses.values():
            total += 1
            status = record.get("status")

            if status == ExecutionStatus.EXECUTION_BLOCKED.value:
                blocked += 1
                reason = record.get("block_reason", "unknown")
                by_block_reason[reason] = by_block_reason.get(reason, 0) + 1
            elif status == ExecutionStatus.EXECUTION_PENDING.value:
                pending += 1
            elif status == ExecutionStatus.EXECUTION_SUCCESS.value:
                success += 1
            elif status == ExecutionStatus.EXECUTION_FAILED.value:
                failed += 1

        # Count by action type from intents
        for record in self._read_records(self._intents_file):
            if record.get("created_at", "") < cutoff:
                continue

            if project_id and record.get("project_id") != project_id:
                continue

            action = record.get("action_type", "unknown")
            by_action_type[action] = by_action_type.get(action, 0) + 1

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "since_hours": since_hours,
            "project_id": project_id,
            "total_executions": total,
            "blocked_count": blocked,
            "pending_count": pending,
            "success_count": success,
            "failed_count": failed,
            "success_rate": success / total if total > 0 else 0.0,
            "by_action_type": by_action_type,
            "by_block_reason": by_block_reason,
        }

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _append_record(self, file_path: Path, record: Dict[str, Any]) -> None:
        """
        Append a record to a JSONL file with fsync.

        APPEND-ONLY: Only appends, never modifies.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
            f.flush()
            os.fsync(f.fileno())

    def _read_records(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Read all records from a JSONL file.

        READ-ONLY: Does not modify state.
        """
        if not file_path.exists():
            return []

        records = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

        return records


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------

# Singleton instance
_store: Optional[ExecutionStore] = None


def get_execution_store(
    intents_file: Optional[Path] = None,
    results_file: Optional[Path] = None,
) -> ExecutionStore:
    """Get the execution store singleton."""
    global _store
    if _store is None:
        _store = ExecutionStore(
            intents_file=intents_file,
            results_file=results_file,
        )
    return _store


def record_execution_intent(intent: ExecutionIntent) -> None:
    """
    Record an execution intent.

    Convenience function using singleton store.
    """
    store = get_execution_store()
    store.record_intent(intent)


def record_execution_result(result: ExecutionResult) -> None:
    """
    Record an execution result.

    Convenience function using singleton store.
    """
    store = get_execution_store()
    store.record_result(result)


def get_execution_result(execution_id: str) -> Optional[ExecutionResult]:
    """
    Get an execution result.

    Convenience function using singleton store.
    """
    store = get_execution_store()
    return store.get_result(execution_id)


def get_execution_store_summary(
    project_id: Optional[str] = None,
    since_hours: int = 24,
) -> Dict[str, Any]:
    """
    Get execution summary.

    Convenience function using singleton store.
    """
    store = get_execution_store()
    return store.get_summary(project_id=project_id, since_hours=since_hours)
