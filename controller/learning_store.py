"""
Phase 19: Learning Store

Append-only persistence for learning data, patterns, and memory.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- APPEND-ONLY: Records are NEVER modified or deleted
- IMMUTABLE: Once written, records cannot change
- DETERMINISTIC: Same query always returns same results
- NO SIDE EFFECTS: Reading does not modify state
- FSYNC: All writes are fsync'd for durability
- NO BEHAVIORAL COUPLING: Never influences other phases

This store provides:
1. Pattern persistence
2. Trend history
3. Memory storage
4. Aggregate records

This store does NOT:
1. Trigger any actions
2. Modify thresholds
3. Influence decisions
4. Execute automation
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from .learning_model import (
    PatternType,
    TrendDirection,
    ConfidenceLevel,
    AggregateType,
    MemoryEntryType,
    ObservedPattern,
    HistoricalAggregate,
    TrendObservation,
    MemoryEntry,
    LearningSummary,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STORE_VERSION = "19.1.0"

# Storage paths
STORAGE_DIR = Path(os.getenv("LEARNING_STORAGE_DIR", "data/learning"))
PATTERNS_FILE = STORAGE_DIR / "patterns.jsonl"
TRENDS_FILE = STORAGE_DIR / "trends.jsonl"
AGGREGATES_FILE = STORAGE_DIR / "aggregates.jsonl"
MEMORY_FILE = STORAGE_DIR / "memory.jsonl"
SUMMARIES_FILE = STORAGE_DIR / "summaries.jsonl"


# -----------------------------------------------------------------------------
# Learning Store (Append-Only Persistence)
# -----------------------------------------------------------------------------
class LearningStore:
    """
    Phase 19: Learning Store.

    Append-only persistence for learning data.

    CRITICAL CONSTRAINTS:
    - APPEND-ONLY: Records are NEVER modified or deleted
    - IMMUTABLE: Once written, records cannot change
    - FSYNC: All writes are fsync'd for durability
    - DETERMINISTIC: Same query always returns same results
    - NO BEHAVIORAL COUPLING: Never influences other phases
    """

    def __init__(
        self,
        patterns_file: Optional[Path] = None,
        trends_file: Optional[Path] = None,
        aggregates_file: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        summaries_file: Optional[Path] = None,
    ):
        """
        Initialize store.

        Args:
            patterns_file: Path to patterns file (optional, for testing)
            trends_file: Path to trends file (optional, for testing)
            aggregates_file: Path to aggregates file (optional, for testing)
            memory_file: Path to memory file (optional, for testing)
            summaries_file: Path to summaries file (optional, for testing)
        """
        self._patterns_file = patterns_file or PATTERNS_FILE
        self._trends_file = trends_file or TRENDS_FILE
        self._aggregates_file = aggregates_file or AGGREGATES_FILE
        self._memory_file = memory_file or MEMORY_FILE
        self._summaries_file = summaries_file or SUMMARIES_FILE
        self._version = STORE_VERSION

    # -------------------------------------------------------------------------
    # Write Operations (Append-Only)
    # -------------------------------------------------------------------------

    def record_pattern(self, pattern: ObservedPattern) -> None:
        """
        Record an observed pattern.

        APPEND-ONLY: Creates a new record, never modifies existing.
        """
        self._append_record(self._patterns_file, pattern.to_dict())

    def record_trend(self, trend: TrendObservation) -> None:
        """
        Record a trend observation.

        APPEND-ONLY: Creates a new record, never modifies existing.
        """
        self._append_record(self._trends_file, trend.to_dict())

    def record_aggregate(self, aggregate: HistoricalAggregate) -> None:
        """
        Record a historical aggregate.

        APPEND-ONLY: Creates a new record, never modifies existing.
        """
        self._append_record(self._aggregates_file, aggregate.to_dict())

    def record_memory(self, entry: MemoryEntry) -> None:
        """
        Record a memory entry.

        APPEND-ONLY: Creates a new record, never modifies existing.
        """
        self._append_record(self._memory_file, entry.to_dict())

    def record_summary(self, summary: LearningSummary) -> None:
        """
        Record a learning summary.

        APPEND-ONLY: Creates a new record, never modifies existing.
        """
        self._append_record(self._summaries_file, summary.to_dict())

    # -------------------------------------------------------------------------
    # Read Operations (Read-Only)
    # -------------------------------------------------------------------------

    def get_pattern(self, pattern_id: str) -> Optional[ObservedPattern]:
        """
        Get pattern by ID.

        READ-ONLY: Does not modify state.
        """
        for record in self._read_records(self._patterns_file):
            if record.get("pattern_id") == pattern_id:
                return ObservedPattern.from_dict(record)
        return None

    def get_recent_patterns(
        self,
        limit: int = 100,
        pattern_type: Optional[str] = None,
        confidence: Optional[str] = None,
    ) -> List[ObservedPattern]:
        """
        Get recent patterns.

        READ-ONLY: Does not modify state.
        """
        patterns = []
        for record in self._read_records(self._patterns_file):
            if pattern_type and record.get("pattern_type") != pattern_type:
                continue
            if confidence and record.get("confidence") != confidence:
                continue
            patterns.append(ObservedPattern.from_dict(record))
        # Most recent first
        patterns.reverse()
        return patterns[:limit]

    def get_trend(self, trend_id: str) -> Optional[TrendObservation]:
        """
        Get trend by ID.

        READ-ONLY: Does not modify state.
        """
        for record in self._read_records(self._trends_file):
            if record.get("trend_id") == trend_id:
                return TrendObservation.from_dict(record)
        return None

    def get_recent_trends(
        self,
        limit: int = 100,
        metric_name: Optional[str] = None,
        direction: Optional[str] = None,
    ) -> List[TrendObservation]:
        """
        Get recent trends.

        READ-ONLY: Does not modify state.
        """
        trends = []
        for record in self._read_records(self._trends_file):
            if metric_name and record.get("metric_name") != metric_name:
                continue
            if direction and record.get("direction") != direction:
                continue
            trends.append(TrendObservation.from_dict(record))
        trends.reverse()
        return trends[:limit]

    def get_aggregate(self, aggregate_id: str) -> Optional[HistoricalAggregate]:
        """
        Get aggregate by ID.

        READ-ONLY: Does not modify state.
        """
        for record in self._read_records(self._aggregates_file):
            if record.get("aggregate_id") == aggregate_id:
                return HistoricalAggregate.from_dict(record)
        return None

    def get_recent_aggregates(
        self,
        limit: int = 100,
        aggregate_type: Optional[str] = None,
    ) -> List[HistoricalAggregate]:
        """
        Get recent aggregates.

        READ-ONLY: Does not modify state.
        """
        aggregates = []
        for record in self._read_records(self._aggregates_file):
            if aggregate_type and record.get("aggregate_type") != aggregate_type:
                continue
            aggregates.append(HistoricalAggregate.from_dict(record))
        aggregates.reverse()
        return aggregates[:limit]

    def get_memory_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get memory entry by ID.

        READ-ONLY: Does not modify state.
        """
        for record in self._read_records(self._memory_file):
            if record.get("entry_id") == entry_id:
                return MemoryEntry.from_dict(record)
        return None

    def get_recent_memory(
        self,
        limit: int = 100,
        entry_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """
        Get recent memory entries.

        READ-ONLY: Does not modify state.
        """
        entries = []
        for record in self._read_records(self._memory_file):
            if entry_type and record.get("entry_type") != entry_type:
                continue
            if project_id and record.get("project_id") != project_id:
                continue
            entries.append(MemoryEntry.from_dict(record))
        entries.reverse()
        return entries[:limit]

    def get_summary(self, summary_id: str) -> Optional[LearningSummary]:
        """
        Get summary by ID.

        READ-ONLY: Does not modify state.
        """
        for record in self._read_records(self._summaries_file):
            if record.get("summary_id") == summary_id:
                return LearningSummary.from_dict(record)
        return None

    def get_recent_summaries(self, limit: int = 10) -> List[LearningSummary]:
        """
        Get recent summaries.

        READ-ONLY: Does not modify state.
        """
        summaries = []
        for record in self._read_records(self._summaries_file):
            summaries.append(LearningSummary.from_dict(record))
        summaries.reverse()
        return summaries[:limit]

    def get_latest_summary(self) -> Optional[LearningSummary]:
        """
        Get the most recent summary.

        READ-ONLY: Does not modify state.
        """
        summaries = self.get_recent_summaries(limit=1)
        return summaries[0] if summaries else None

    # -------------------------------------------------------------------------
    # Summary/Aggregation (Read-Only)
    # -------------------------------------------------------------------------

    def get_statistics(
        self,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get learning statistics.

        READ-ONLY: Does not modify state.
        """
        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        pattern_count = 0
        trend_count = 0
        memory_count = 0
        aggregate_count = 0
        summary_count = 0

        by_pattern_type: Dict[str, int] = {}
        by_trend_direction: Dict[str, int] = {}
        by_memory_type: Dict[str, int] = {}

        # Count patterns
        for record in self._read_records(self._patterns_file):
            if record.get("observed_at", "") >= cutoff:
                pattern_count += 1
                p_type = record.get("pattern_type", "unknown")
                by_pattern_type[p_type] = by_pattern_type.get(p_type, 0) + 1

        # Count trends
        for record in self._read_records(self._trends_file):
            if record.get("observed_at", "") >= cutoff:
                trend_count += 1
                direction = record.get("direction", "unknown")
                by_trend_direction[direction] = by_trend_direction.get(direction, 0) + 1

        # Count memory entries
        for record in self._read_records(self._memory_file):
            if record.get("recorded_at", "") >= cutoff:
                memory_count += 1
                m_type = record.get("entry_type", "unknown")
                by_memory_type[m_type] = by_memory_type.get(m_type, 0) + 1

        # Count aggregates
        for record in self._read_records(self._aggregates_file):
            if record.get("computed_at", "") >= cutoff:
                aggregate_count += 1

        # Count summaries
        for record in self._read_records(self._summaries_file):
            if record.get("generated_at", "") >= cutoff:
                summary_count += 1

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "since_hours": since_hours,
            "pattern_count": pattern_count,
            "trend_count": trend_count,
            "memory_count": memory_count,
            "aggregate_count": aggregate_count,
            "summary_count": summary_count,
            "by_pattern_type": by_pattern_type,
            "by_trend_direction": by_trend_direction,
            "by_memory_type": by_memory_type,
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
_store: Optional[LearningStore] = None


def get_learning_store(
    patterns_file: Optional[Path] = None,
    trends_file: Optional[Path] = None,
    aggregates_file: Optional[Path] = None,
    memory_file: Optional[Path] = None,
    summaries_file: Optional[Path] = None,
) -> LearningStore:
    """Get the learning store singleton."""
    global _store
    if _store is None:
        _store = LearningStore(
            patterns_file=patterns_file,
            trends_file=trends_file,
            aggregates_file=aggregates_file,
            memory_file=memory_file,
            summaries_file=summaries_file,
        )
    return _store


def record_pattern(pattern: ObservedPattern) -> None:
    """Record a pattern. Convenience function using singleton store."""
    store = get_learning_store()
    store.record_pattern(pattern)


def record_trend(trend: TrendObservation) -> None:
    """Record a trend. Convenience function using singleton store."""
    store = get_learning_store()
    store.record_trend(trend)


def record_memory(entry: MemoryEntry) -> None:
    """Record a memory entry. Convenience function using singleton store."""
    store = get_learning_store()
    store.record_memory(entry)


def get_recent_patterns(
    limit: int = 100,
    pattern_type: Optional[str] = None,
) -> List[ObservedPattern]:
    """Get recent patterns. Convenience function using singleton store."""
    store = get_learning_store()
    return store.get_recent_patterns(limit=limit, pattern_type=pattern_type)


def get_recent_trends(
    limit: int = 100,
    metric_name: Optional[str] = None,
) -> List[TrendObservation]:
    """Get recent trends. Convenience function using singleton store."""
    store = get_learning_store()
    return store.get_recent_trends(limit=limit, metric_name=metric_name)


def get_learning_statistics(since_hours: int = 24) -> Dict[str, Any]:
    """Get learning statistics. Convenience function using singleton store."""
    store = get_learning_store()
    return store.get_statistics(since_hours=since_hours)
