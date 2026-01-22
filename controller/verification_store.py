"""
Phase 18D: Verification Store

Append-only persistence for verification results and violations.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- APPEND-ONLY: Records are NEVER modified or deleted
- IMMUTABLE: Once written, records cannot change
- DETERMINISTIC: Same query always returns same results
- NO SIDE EFFECTS: Reading does not modify state
- FSYNC: All writes are fsync'd for durability
- NO MUTATION: Never triggers any actions

This store provides:
1. Verification result persistence
2. Violation history
3. Read-only queries for engine

This store does NOT:
1. Execute any actions
2. Trigger rollbacks
3. Send notifications
4. Recommend fixes
5. Modify external state
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from .verification_model import (
    VerificationStatus,
    ViolationSeverity,
    ViolationType,
    ExecutionVerificationResult,
    InvariantViolation,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STORE_VERSION = "18D.1.0"

# Storage paths
STORAGE_DIR = Path(os.getenv("VERIFICATION_STORAGE_DIR", "data/verification"))
RESULTS_FILE = STORAGE_DIR / "verification_results.jsonl"
VIOLATIONS_FILE = STORAGE_DIR / "violations.jsonl"


# -----------------------------------------------------------------------------
# Verification Store (Append-Only Persistence)
# -----------------------------------------------------------------------------
class VerificationStore:
    """
    Phase 18D: Verification Store.

    Append-only persistence for verification results and violations.

    CRITICAL CONSTRAINTS:
    - APPEND-ONLY: Records are NEVER modified or deleted
    - IMMUTABLE: Once written, records cannot change
    - FSYNC: All writes are fsync'd for durability
    - DETERMINISTIC: Same query always returns same results
    - NO MUTATION: Never triggers any actions
    """

    def __init__(
        self,
        results_file: Optional[Path] = None,
        violations_file: Optional[Path] = None,
    ):
        """
        Initialize store.

        Args:
            results_file: Path to results file (optional, for testing)
            violations_file: Path to violations file (optional, for testing)
        """
        self._results_file = results_file or RESULTS_FILE
        self._violations_file = violations_file or VIOLATIONS_FILE
        self._version = STORE_VERSION

    # -------------------------------------------------------------------------
    # Write Operations (Append-Only)
    # -------------------------------------------------------------------------

    def record_verification(self, result: ExecutionVerificationResult) -> None:
        """
        Record a verification result.

        APPEND-ONLY: Creates a new record, never modifies existing.

        Args:
            result: The verification result to record
        """
        self._append_record(self._results_file, result.to_dict())

        # Also record individual violations for easier querying
        for violation in result.violations:
            violation_record = {
                **violation.to_dict(),
                "verification_id": result.verification_id,
                "execution_id": result.execution_id,
                "verification_status": result.verification_status,
            }
            self._append_record(self._violations_file, violation_record)

    # -------------------------------------------------------------------------
    # Read Operations (Read-Only)
    # -------------------------------------------------------------------------

    def get_verification(self, execution_id: str) -> Optional[ExecutionVerificationResult]:
        """
        Get verification result for an execution.

        READ-ONLY: Does not modify state.
        Returns the most recent result if multiple exist.

        Args:
            execution_id: ID of the execution

        Returns:
            ExecutionVerificationResult if found, None otherwise
        """
        latest = None
        for record in self._read_records(self._results_file):
            if record.get("execution_id") == execution_id:
                latest = ExecutionVerificationResult.from_dict(record)
        return latest

    def get_verification_by_id(self, verification_id: str) -> Optional[ExecutionVerificationResult]:
        """
        Get verification result by verification ID.

        READ-ONLY: Does not modify state.

        Args:
            verification_id: ID of the verification

        Returns:
            ExecutionVerificationResult if found, None otherwise
        """
        for record in self._read_records(self._results_file):
            if record.get("verification_id") == verification_id:
                return ExecutionVerificationResult.from_dict(record)
        return None

    def get_recent_verifications(
        self,
        limit: int = 100,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[ExecutionVerificationResult]:
        """
        Get recent verification results.

        READ-ONLY: Does not modify state.

        Args:
            limit: Maximum number of results
            status: Optional filter by status (passed, failed, unknown)
            project_id: Optional filter by project

        Returns:
            List of ExecutionVerificationResults (most recent first)
        """
        results = []
        for record in self._read_records(self._results_file):
            # Filter by status
            if status and record.get("verification_status") != status:
                continue

            results.append(ExecutionVerificationResult.from_dict(record))

        # Return most recent first
        results.reverse()
        return results[:limit]

    def get_violations_for_execution(
        self,
        execution_id: str,
    ) -> List[InvariantViolation]:
        """
        Get all violations for an execution.

        READ-ONLY: Does not modify state.

        Args:
            execution_id: ID of the execution

        Returns:
            List of InvariantViolations
        """
        violations = []
        for record in self._read_records(self._violations_file):
            if record.get("execution_id") == execution_id:
                violations.append(InvariantViolation.from_dict(record))
        return violations

    def get_violations_by_type(
        self,
        violation_type: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get violations by type.

        READ-ONLY: Does not modify state.

        Args:
            violation_type: Type of violation to filter by
            limit: Maximum number of results

        Returns:
            List of violation records with execution context
        """
        results = []
        for record in self._read_records(self._violations_file):
            if record.get("violation_type") == violation_type:
                results.append(record)
                if len(results) >= limit:
                    break
        return results

    def get_violations_by_severity(
        self,
        severity: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get violations by severity.

        READ-ONLY: Does not modify state.

        Args:
            severity: Severity to filter by (info, low, medium, high)
            limit: Maximum number of results

        Returns:
            List of violation records with execution context
        """
        results = []
        for record in self._read_records(self._violations_file):
            if record.get("severity") == severity:
                results.append(record)
                if len(results) >= limit:
                    break
        return results

    def is_execution_verified(self, execution_id: str) -> bool:
        """
        Check if an execution has been verified.

        READ-ONLY: Does not modify state.

        Args:
            execution_id: ID of the execution

        Returns:
            True if verified (regardless of status), False otherwise
        """
        for record in self._read_records(self._results_file):
            if record.get("execution_id") == execution_id:
                return True
        return False

    def get_verification_status(self, execution_id: str) -> Optional[str]:
        """
        Get verification status for an execution.

        READ-ONLY: Does not modify state.

        Args:
            execution_id: ID of the execution

        Returns:
            VerificationStatus value if verified, None otherwise
        """
        result = self.get_verification(execution_id)
        return result.verification_status if result else None

    # -------------------------------------------------------------------------
    # Summary/Aggregation (Read-Only)
    # -------------------------------------------------------------------------

    def get_summary(
        self,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get a summary of verification activity.

        READ-ONLY: Does not modify state.

        Args:
            since_hours: Hours to look back

        Returns:
            Summary dictionary
        """
        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        total = 0
        passed = 0
        failed = 0
        unknown = 0
        total_violations = 0
        high_severity = 0
        by_violation_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}

        for record in self._read_records(self._results_file):
            if record.get("checked_at", "") < cutoff:
                continue

            total += 1
            status = record.get("verification_status")

            if status == VerificationStatus.PASSED.value:
                passed += 1
            elif status == VerificationStatus.FAILED.value:
                failed += 1
                total_violations += record.get("violation_count", 0)
                high_severity += record.get("high_severity_count", 0)
            elif status == VerificationStatus.UNKNOWN.value:
                unknown += 1

        # Count by violation type and severity
        for record in self._read_records(self._violations_file):
            if record.get("detected_at", "") < cutoff:
                continue

            v_type = record.get("violation_type", "unknown")
            severity = record.get("severity", "unknown")

            by_violation_type[v_type] = by_violation_type.get(v_type, 0) + 1
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "since_hours": since_hours,
            "total_verifications": total,
            "passed_count": passed,
            "failed_count": failed,
            "unknown_count": unknown,
            "pass_rate": passed / total if total > 0 else 0.0,
            "total_violations": total_violations,
            "high_severity_violations": high_severity,
            "by_violation_type": by_violation_type,
            "by_severity": by_severity,
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
_store: Optional[VerificationStore] = None


def get_verification_store(
    results_file: Optional[Path] = None,
    violations_file: Optional[Path] = None,
) -> VerificationStore:
    """Get the verification store singleton."""
    global _store
    if _store is None:
        _store = VerificationStore(
            results_file=results_file,
            violations_file=violations_file,
        )
    return _store


def record_verification(result: ExecutionVerificationResult) -> None:
    """
    Record a verification result.

    Convenience function using singleton store.
    """
    store = get_verification_store()
    store.record_verification(result)


def get_verification_for_execution(execution_id: str) -> Optional[ExecutionVerificationResult]:
    """
    Get verification for an execution.

    Convenience function using singleton store.
    """
    store = get_verification_store()
    return store.get_verification(execution_id)


def get_violations_for_execution(execution_id: str) -> List[InvariantViolation]:
    """
    Get violations for an execution.

    Convenience function using singleton store.
    """
    store = get_verification_store()
    return store.get_violations_for_execution(execution_id)


def get_verification_store_summary(since_hours: int = 24) -> Dict[str, Any]:
    """
    Get verification summary.

    Convenience function using singleton store.
    """
    store = get_verification_store()
    return store.get_summary(since_hours=since_hours)
