"""
Phase 25: Test Gap Store — Append-Only JSONL Gap Tracking

Tracks test gaps (missing tests) across deployments. When a gap
is seen twice, it is confirmed and triggers a Claude task.

CONSTRAINTS:
- Append-only JSONL file (never modifies existing records)
- Deterministic confirmation: 2 occurrences = confirmed
- Thread-safe via file-level writes with fsync
"""

import json
import os
import logging
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger("test_gap_store")


# ---------------------------------------------------------------------------
# Gap types
# ---------------------------------------------------------------------------

class GapType(str, Enum):
    MISSING_CORS_TEST = "MISSING_CORS_TEST"
    MISSING_ROUTE_TESTS = "MISSING_ROUTE_TESTS"
    MISSING_API_SEED_TESTS = "MISSING_API_SEED_TESTS"
    MISSING_DEPLOYMENT_SMOKE = "MISSING_DEPLOYMENT_SMOKE"
    MISSING_API_E2E = "MISSING_API_E2E"
    MISSING_FRONTEND_E2E = "MISSING_FRONTEND_E2E"
    MISSING_PLAYWRIGHT_CI = "MISSING_PLAYWRIGHT_CI"


# ---------------------------------------------------------------------------
# TestGapStore
# ---------------------------------------------------------------------------

class TestGapStore:
    """
    Append-only JSONL store for test gap tracking.

    Each record represents a gap observation. When the same
    (project, gap_type) pair is seen 2+ times, it's confirmed.
    """

    DEFAULT_GAP_FILE = Path("data/test_gaps/test_gaps.jsonl")

    def __init__(self, gap_file: Optional[Path] = None):
        self._gap_file = gap_file or self.DEFAULT_GAP_FILE

    def record_gap(self, project: str, gap_type: str) -> bool:
        """
        Record a gap occurrence.

        Returns True if the gap is now confirmed (2+ occurrences).
        """
        now = datetime.utcnow().isoformat()
        existing = self._find_latest_gap(project, gap_type)

        if existing:
            occurrences = existing.get("occurrences", 1) + 1
            confirmed = occurrences >= 2
            record = {
                "project": project,
                "gap_type": gap_type,
                "first_seen_at": existing.get("first_seen_at", now),
                "occurrences": occurrences,
                "confirmed": confirmed,
                "last_seen_at": now,
            }
            self._append_record(record)
            if confirmed:
                logger.info(f"Gap confirmed: {project}/{gap_type} (seen {occurrences} times)")
            return confirmed
        else:
            record = {
                "project": project,
                "gap_type": gap_type,
                "first_seen_at": now,
                "occurrences": 1,
                "confirmed": False,
                "last_seen_at": now,
            }
            self._append_record(record)
            return False

    def get_gaps(self, project: str) -> List[Dict[str, Any]]:
        """
        Get all gaps for a project (latest record per gap_type).
        """
        all_records = self._read_all()
        latest: Dict[str, Dict[str, Any]] = {}

        for record in all_records:
            if record.get("project") == project:
                gap_type = record.get("gap_type", "")
                latest[gap_type] = record

        return list(latest.values())

    def is_confirmed(self, project: str, gap_type: str) -> bool:
        """Check if a gap has been confirmed (seen 2+ times)."""
        record = self._find_latest_gap(project, gap_type)
        if record:
            return record.get("confirmed", False)
        return False

    def _find_latest_gap(
        self, project: str, gap_type: str
    ) -> Optional[Dict[str, Any]]:
        """Find the latest record for a specific (project, gap_type) pair."""
        all_records = self._read_all()
        latest = None
        for record in all_records:
            if (record.get("project") == project and
                    record.get("gap_type") == gap_type):
                latest = record
        return latest

    def _read_all(self) -> List[Dict[str, Any]]:
        """Read all records from the JSONL file."""
        records: List[Dict[str, Any]] = []
        if not self._gap_file.exists():
            return records

        try:
            with open(self._gap_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Failed to read gap store: {e}")

        return records

    def _append_record(self, record: Dict[str, Any]) -> None:
        """Append a record to the JSONL file with fsync."""
        try:
            self._gap_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._gap_file, "a") as f:
                f.write(json.dumps(record) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as e:
            logger.error(f"Failed to write gap record: {e}")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: Optional[TestGapStore] = None


def get_test_gap_store() -> TestGapStore:
    global _store
    if _store is None:
        _store = TestGapStore()
    return _store
