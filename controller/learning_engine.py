"""
Phase 19: Learning, Memory & System Intelligence - Engine

Pattern detection and historical aggregation engine.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- NO BEHAVIORAL COUPLING: Never influences eligibility, approval, execution, or recommendations
- NO THRESHOLD MODIFICATION: Never changes system thresholds or limits
- NO ML INFERENCE: No machine learning, no optimization, no prediction
- NO AUTOMATION: Never triggers any automated actions
- 100% DETERMINISTIC: Same inputs = same aggregates
- READ-ONLY ANALYSIS: Observes history, never changes it
- APPEND-ONLY OUTPUT: Memory is written, never modified

This engine provides INSIGHT, not ACTION.
It is MEMORY, not INTELLIGENCE.
Patterns are RECORDED, not ACTED UPON.
"""

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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
    LearningInput,
)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
ENGINE_VERSION = "19.1.0"

# Storage paths
LEARNING_DIR = Path(os.getenv("LEARNING_DIR", "data/learning"))
PATTERNS_FILE = LEARNING_DIR / "patterns.jsonl"
AGGREGATES_FILE = LEARNING_DIR / "aggregates.jsonl"
TRENDS_FILE = LEARNING_DIR / "trends.jsonl"
MEMORY_FILE = LEARNING_DIR / "memory.jsonl"
SUMMARIES_FILE = LEARNING_DIR / "summaries.jsonl"

# Confidence thresholds (STATISTICAL, not ML)
CONFIDENCE_HIGH_MIN_N = 100
CONFIDENCE_HIGH_MIN_CONSISTENCY = 0.90
CONFIDENCE_MEDIUM_MIN_N = 30
CONFIDENCE_MEDIUM_MIN_CONSISTENCY = 0.70
CONFIDENCE_LOW_MIN_N = 10
CONFIDENCE_LOW_MIN_CONSISTENCY = 0.50


# -----------------------------------------------------------------------------
# Learning Engine (READ-ONLY ANALYSIS)
# -----------------------------------------------------------------------------
class LearningEngine:
    """
    Phase 19: Learning, Memory & System Intelligence Engine.

    READ-ONLY analysis engine that observes patterns and computes aggregates.

    CRITICAL CONSTRAINTS:
    - NO behavioral coupling to other phases
    - NO threshold modification
    - NO ML inference
    - NO automation triggers
    - DETERMINISTIC: Same inputs = same outputs
    - APPEND-ONLY: All outputs are immutable records

    Patterns are RECORDED for human insight, not for automation.
    """

    def __init__(
        self,
        patterns_file: Optional[Path] = None,
        aggregates_file: Optional[Path] = None,
        trends_file: Optional[Path] = None,
        memory_file: Optional[Path] = None,
        summaries_file: Optional[Path] = None,
    ):
        """Initialize engine with storage paths."""
        self._patterns_file = patterns_file or PATTERNS_FILE
        self._aggregates_file = aggregates_file or AGGREGATES_FILE
        self._trends_file = trends_file or TRENDS_FILE
        self._memory_file = memory_file or MEMORY_FILE
        self._summaries_file = summaries_file or SUMMARIES_FILE
        self._version = ENGINE_VERSION

    # -------------------------------------------------------------------------
    # Main Analysis Entry Point
    # -------------------------------------------------------------------------

    def analyze(self, learning_input: LearningInput) -> LearningSummary:
        """
        Analyze historical data and generate learning summary.

        This is the main entry point for learning analysis.
        It:
        1. Records memory entries
        2. Computes aggregates
        3. Detects patterns
        4. Observes trends
        5. Generates summary

        All outputs are APPEND-ONLY and for HUMAN INSIGHT only.

        Args:
            learning_input: Complete input from historical data

        Returns:
            LearningSummary with observations
        """
        generated_at = datetime.utcnow().isoformat()
        summary_id = f"lsum-{generated_at.replace(':', '-').replace('.', '-')}-{uuid.uuid4().hex[:8]}"

        # Step 1: Record memory entries
        self._record_memory_entries(learning_input, generated_at)

        # Step 2: Compute aggregates
        aggregates = self._compute_aggregates(learning_input, generated_at)

        # Step 3: Detect patterns
        patterns = self._detect_patterns(learning_input, generated_at)

        # Step 4: Observe trends
        trends = self._observe_trends(learning_input, generated_at)

        # Step 5: Compute summary metrics
        execution_success_rate = self._compute_execution_success_rate(learning_input)
        verification_pass_rate = self._compute_verification_pass_rate(learning_input)
        approval_grant_rate = self._compute_approval_grant_rate(learning_input)

        # Step 6: Create summary
        summary = LearningSummary(
            summary_id=summary_id,
            period_start=learning_input.period_start,
            period_end=learning_input.period_end,
            generated_at=generated_at,
            total_executions=len(learning_input.execution_results),
            total_verifications=len(learning_input.verification_results),
            total_approvals=len(learning_input.approval_outcomes),
            total_incidents=len(learning_input.incident_summaries),
            execution_success_rate=execution_success_rate,
            verification_pass_rate=verification_pass_rate,
            approval_grant_rate=approval_grant_rate,
            pattern_count=len(patterns),
            trend_count=len(trends),
            top_patterns=tuple(p.pattern_id for p in patterns[:5]),
            top_trends=tuple(t.trend_id for t in trends[:5]),
            engine_version=self._version,
        )

        # Record summary
        self._append_record(self._summaries_file, summary.to_dict())

        return summary

    # -------------------------------------------------------------------------
    # Memory Recording (APPEND-ONLY)
    # -------------------------------------------------------------------------

    def _record_memory_entries(self, learning_input: LearningInput, recorded_at: str) -> None:
        """
        Record memory entries from input data.

        APPEND-ONLY: Only appends, never modifies.
        """
        # Record execution outcomes
        for exec_data in learning_input.execution_results:
            entry = MemoryEntry(
                entry_id=f"mem-{uuid.uuid4().hex[:12]}",
                entry_type=MemoryEntryType.EXECUTION_OUTCOME.value,
                source_id=exec_data.get("execution_id", "unknown"),
                source_type="execution",
                timestamp=exec_data.get("timestamp", recorded_at),
                recorded_at=recorded_at,
                summary=f"Execution {exec_data.get('status', 'unknown')}",
                outcome=exec_data.get("status", "unknown"),
                project_id=exec_data.get("project_id"),
                details=tuple([
                    ("action_type", str(exec_data.get("action_type", ""))),
                    ("block_reason", str(exec_data.get("block_reason", ""))),
                ]),
            )
            self._append_record(self._memory_file, entry.to_dict())

        # Record verification results
        for ver_data in learning_input.verification_results:
            entry = MemoryEntry(
                entry_id=f"mem-{uuid.uuid4().hex[:12]}",
                entry_type=MemoryEntryType.VERIFICATION_RESULT.value,
                source_id=ver_data.get("verification_id", "unknown"),
                source_type="verification",
                timestamp=ver_data.get("checked_at", recorded_at),
                recorded_at=recorded_at,
                summary=f"Verification {ver_data.get('verification_status', 'unknown')}",
                outcome=ver_data.get("verification_status", "unknown"),
                project_id=ver_data.get("project_id"),
                details=tuple([
                    ("violation_count", str(ver_data.get("violation_count", 0))),
                    ("high_severity_count", str(ver_data.get("high_severity_count", 0))),
                ]),
            )
            self._append_record(self._memory_file, entry.to_dict())

        # Record approval decisions
        for appr_data in learning_input.approval_outcomes:
            entry = MemoryEntry(
                entry_id=f"mem-{uuid.uuid4().hex[:12]}",
                entry_type=MemoryEntryType.APPROVAL_DECISION.value,
                source_id=appr_data.get("approval_id", "unknown"),
                source_type="approval",
                timestamp=appr_data.get("timestamp", recorded_at),
                recorded_at=recorded_at,
                summary=f"Approval {appr_data.get('status', 'unknown')}",
                outcome=appr_data.get("status", "unknown"),
                project_id=appr_data.get("project_id"),
                details=tuple([
                    ("requester_id", str(appr_data.get("requester_id", ""))),
                ]),
            )
            self._append_record(self._memory_file, entry.to_dict())

        # Record incidents
        for inc_data in learning_input.incident_summaries:
            entry = MemoryEntry(
                entry_id=f"mem-{uuid.uuid4().hex[:12]}",
                entry_type=MemoryEntryType.INCIDENT_OCCURRENCE.value,
                source_id=inc_data.get("incident_id", "unknown"),
                source_type="incident",
                timestamp=inc_data.get("timestamp", recorded_at),
                recorded_at=recorded_at,
                summary=f"Incident {inc_data.get('type', 'unknown')} - {inc_data.get('severity', 'unknown')}",
                outcome=inc_data.get("severity", "unknown"),
                project_id=inc_data.get("project_id"),
                details=tuple([
                    ("incident_type", str(inc_data.get("type", ""))),
                    ("scope", str(inc_data.get("scope", ""))),
                ]),
            )
            self._append_record(self._memory_file, entry.to_dict())

    # -------------------------------------------------------------------------
    # Aggregate Computation (DETERMINISTIC)
    # -------------------------------------------------------------------------

    def _compute_aggregates(
        self,
        learning_input: LearningInput,
        computed_at: str,
    ) -> List[HistoricalAggregate]:
        """
        Compute historical aggregates from input data.

        DETERMINISTIC: Same inputs = same outputs.
        """
        aggregates = []

        # Failure rate aggregate
        if learning_input.execution_results:
            failure_rate = self._compute_execution_failure_rate(learning_input)
            agg = HistoricalAggregate(
                aggregate_id=f"agg-{uuid.uuid4().hex[:12]}",
                aggregate_type=AggregateType.FAILURE_RATE.value,
                value=failure_rate,
                sample_size=len(learning_input.execution_results),
                period_start=learning_input.period_start,
                period_end=learning_input.period_end,
                computed_at=computed_at,
                project_id=None,
                breakdown=self._compute_failure_breakdown(learning_input),
            )
            aggregates.append(agg)
            self._append_record(self._aggregates_file, agg.to_dict())

        # Violation frequency aggregate
        if learning_input.verification_results:
            violation_freq = self._compute_violation_frequency(learning_input)
            agg = HistoricalAggregate(
                aggregate_id=f"agg-{uuid.uuid4().hex[:12]}",
                aggregate_type=AggregateType.VIOLATION_FREQUENCY.value,
                value=violation_freq,
                sample_size=len(learning_input.verification_results),
                period_start=learning_input.period_start,
                period_end=learning_input.period_end,
                computed_at=computed_at,
                project_id=None,
                breakdown=self._compute_violation_breakdown(learning_input),
            )
            aggregates.append(agg)
            self._append_record(self._aggregates_file, agg.to_dict())

        # Approval rejection rate aggregate
        if learning_input.approval_outcomes:
            rejection_rate = self._compute_approval_rejection_rate(learning_input)
            agg = HistoricalAggregate(
                aggregate_id=f"agg-{uuid.uuid4().hex[:12]}",
                aggregate_type=AggregateType.APPROVAL_REJECTION_RATE.value,
                value=rejection_rate,
                sample_size=len(learning_input.approval_outcomes),
                period_start=learning_input.period_start,
                period_end=learning_input.period_end,
                computed_at=computed_at,
                project_id=None,
                breakdown=(),
            )
            aggregates.append(agg)
            self._append_record(self._aggregates_file, agg.to_dict())

        # Incident frequency aggregate
        if learning_input.incident_summaries:
            incident_freq = len(learning_input.incident_summaries)
            agg = HistoricalAggregate(
                aggregate_id=f"agg-{uuid.uuid4().hex[:12]}",
                aggregate_type=AggregateType.INCIDENT_FREQUENCY.value,
                value=float(incident_freq),
                sample_size=len(learning_input.incident_summaries),
                period_start=learning_input.period_start,
                period_end=learning_input.period_end,
                computed_at=computed_at,
                project_id=None,
                breakdown=self._compute_incident_breakdown(learning_input),
            )
            aggregates.append(agg)
            self._append_record(self._aggregates_file, agg.to_dict())

        return aggregates

    def _compute_execution_failure_rate(self, learning_input: LearningInput) -> float:
        """Compute execution failure rate."""
        if not learning_input.execution_results:
            return 0.0
        failed = sum(
            1 for e in learning_input.execution_results
            if e.get("status") in ("execution_failed", "execution_blocked")
        )
        return failed / len(learning_input.execution_results)

    def _compute_execution_success_rate(self, learning_input: LearningInput) -> float:
        """Compute execution success rate."""
        if not learning_input.execution_results:
            return 0.0
        success = sum(
            1 for e in learning_input.execution_results
            if e.get("status") == "execution_success"
        )
        return success / len(learning_input.execution_results)

    def _compute_verification_pass_rate(self, learning_input: LearningInput) -> float:
        """Compute verification pass rate."""
        if not learning_input.verification_results:
            return 0.0
        passed = sum(
            1 for v in learning_input.verification_results
            if v.get("verification_status") == "passed"
        )
        return passed / len(learning_input.verification_results)

    def _compute_approval_grant_rate(self, learning_input: LearningInput) -> float:
        """Compute approval grant rate."""
        if not learning_input.approval_outcomes:
            return 0.0
        granted = sum(
            1 for a in learning_input.approval_outcomes
            if a.get("status") == "approval_granted"
        )
        return granted / len(learning_input.approval_outcomes)

    def _compute_approval_rejection_rate(self, learning_input: LearningInput) -> float:
        """Compute approval rejection rate."""
        if not learning_input.approval_outcomes:
            return 0.0
        denied = sum(
            1 for a in learning_input.approval_outcomes
            if a.get("status") == "approval_denied"
        )
        return denied / len(learning_input.approval_outcomes)

    def _compute_violation_frequency(self, learning_input: LearningInput) -> float:
        """Compute average violations per verification."""
        if not learning_input.verification_results:
            return 0.0
        total_violations = sum(
            v.get("violation_count", 0) for v in learning_input.verification_results
        )
        return total_violations / len(learning_input.verification_results)

    def _compute_failure_breakdown(self, learning_input: LearningInput) -> Tuple[Tuple[str, float], ...]:
        """Compute breakdown of failure reasons."""
        reasons: Dict[str, int] = defaultdict(int)
        for e in learning_input.execution_results:
            if e.get("status") in ("execution_failed", "execution_blocked"):
                reason = e.get("block_reason") or e.get("failure_reason") or "unknown"
                reasons[reason] += 1
        total = sum(reasons.values()) or 1
        return tuple((r, c / total) for r, c in reasons.items())

    def _compute_violation_breakdown(self, learning_input: LearningInput) -> Tuple[Tuple[str, float], ...]:
        """Compute breakdown of violation types."""
        types: Dict[str, int] = defaultdict(int)
        for v in learning_input.verification_results:
            for violation in v.get("violations", []):
                v_type = violation.get("violation_type", "unknown")
                types[v_type] += 1
        total = sum(types.values()) or 1
        return tuple((t, c / total) for t, c in types.items())

    def _compute_incident_breakdown(self, learning_input: LearningInput) -> Tuple[Tuple[str, float], ...]:
        """Compute breakdown of incident types."""
        types: Dict[str, int] = defaultdict(int)
        for i in learning_input.incident_summaries:
            i_type = i.get("type", "unknown")
            types[i_type] += 1
        total = sum(types.values()) or 1
        return tuple((t, c / total) for t, c in types.items())

    # -------------------------------------------------------------------------
    # Pattern Detection (DETERMINISTIC, NO ML)
    # -------------------------------------------------------------------------

    def _detect_patterns(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> List[ObservedPattern]:
        """
        Detect patterns in historical data.

        DETERMINISTIC: Same inputs = same patterns.
        NO ML: Uses simple statistical rules only.
        """
        patterns = []

        # Detect execution failure clusters
        failure_pattern = self._detect_failure_cluster(learning_input, observed_at)
        if failure_pattern:
            patterns.append(failure_pattern)
            self._append_record(self._patterns_file, failure_pattern.to_dict())

        # Detect recurring violations
        violation_pattern = self._detect_recurring_violations(learning_input, observed_at)
        if violation_pattern:
            patterns.append(violation_pattern)
            self._append_record(self._patterns_file, violation_pattern.to_dict())

        # Detect approval rejection trends
        rejection_pattern = self._detect_rejection_trend(learning_input, observed_at)
        if rejection_pattern:
            patterns.append(rejection_pattern)
            self._append_record(self._patterns_file, rejection_pattern.to_dict())

        # Detect incident recurrence
        incident_pattern = self._detect_incident_recurrence(learning_input, observed_at)
        if incident_pattern:
            patterns.append(incident_pattern)
            self._append_record(self._patterns_file, incident_pattern.to_dict())

        return patterns

    def _detect_failure_cluster(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[ObservedPattern]:
        """Detect if failures are clustered."""
        failures = [
            e for e in learning_input.execution_results
            if e.get("status") in ("execution_failed", "execution_blocked")
        ]

        if len(failures) < 3:
            return None

        # Simple clustering: if failure rate > 30%, it's a cluster
        rate = len(failures) / len(learning_input.execution_results) if learning_input.execution_results else 0
        if rate < 0.3:
            return None

        confidence = self._compute_statistical_confidence(len(failures), rate)

        return ObservedPattern(
            pattern_id=f"pat-{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.EXECUTION_FAILURE_CLUSTER.value,
            description=f"Execution failure cluster detected: {rate*100:.1f}% failure rate",
            frequency=len(failures),
            confidence=confidence,
            evidence_ids=tuple(e.get("execution_id", "") for e in failures[:10]),
            first_observed=failures[0].get("timestamp", observed_at) if failures else observed_at,
            last_observed=failures[-1].get("timestamp", observed_at) if failures else observed_at,
            observed_at=observed_at,
            project_id=None,
            metadata=tuple([("failure_rate", f"{rate:.4f}")]),
        )

    def _detect_recurring_violations(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[ObservedPattern]:
        """Detect recurring violation types."""
        violation_counts: Dict[str, int] = defaultdict(int)
        evidence: Dict[str, List[str]] = defaultdict(list)

        for v in learning_input.verification_results:
            for violation in v.get("violations", []):
                v_type = violation.get("violation_type", "unknown")
                violation_counts[v_type] += 1
                evidence[v_type].append(v.get("verification_id", ""))

        # Find most common violation type
        if not violation_counts:
            return None

        most_common = max(violation_counts.items(), key=lambda x: x[1])
        if most_common[1] < 3:
            return None

        total = sum(violation_counts.values())
        consistency = most_common[1] / total if total > 0 else 0
        confidence = self._compute_statistical_confidence(most_common[1], consistency)

        return ObservedPattern(
            pattern_id=f"pat-{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.VERIFICATION_VIOLATION_RECURRING.value,
            description=f"Recurring violation: {most_common[0]} ({most_common[1]} occurrences)",
            frequency=most_common[1],
            confidence=confidence,
            evidence_ids=tuple(evidence[most_common[0]][:10]),
            first_observed=learning_input.period_start,
            last_observed=learning_input.period_end,
            observed_at=observed_at,
            project_id=None,
            metadata=tuple([("violation_type", most_common[0])]),
        )

    def _detect_rejection_trend(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[ObservedPattern]:
        """Detect if approvals are frequently rejected."""
        rejections = [
            a for a in learning_input.approval_outcomes
            if a.get("status") == "approval_denied"
        ]

        if len(rejections) < 3:
            return None

        rate = len(rejections) / len(learning_input.approval_outcomes) if learning_input.approval_outcomes else 0
        if rate < 0.2:  # Less than 20% rejection is not a trend
            return None

        confidence = self._compute_statistical_confidence(len(rejections), rate)

        return ObservedPattern(
            pattern_id=f"pat-{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.APPROVAL_REJECTION_TREND.value,
            description=f"Approval rejection trend: {rate*100:.1f}% rejection rate",
            frequency=len(rejections),
            confidence=confidence,
            evidence_ids=tuple(a.get("approval_id", "") for a in rejections[:10]),
            first_observed=learning_input.period_start,
            last_observed=learning_input.period_end,
            observed_at=observed_at,
            project_id=None,
            metadata=tuple([("rejection_rate", f"{rate:.4f}")]),
        )

    def _detect_incident_recurrence(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[ObservedPattern]:
        """Detect recurring incidents."""
        type_counts: Dict[str, int] = defaultdict(int)
        evidence: Dict[str, List[str]] = defaultdict(list)

        for i in learning_input.incident_summaries:
            i_type = i.get("type", "unknown")
            type_counts[i_type] += 1
            evidence[i_type].append(i.get("incident_id", ""))

        if not type_counts:
            return None

        most_common = max(type_counts.items(), key=lambda x: x[1])
        if most_common[1] < 2:
            return None

        total = sum(type_counts.values())
        consistency = most_common[1] / total if total > 0 else 0
        confidence = self._compute_statistical_confidence(most_common[1], consistency)

        return ObservedPattern(
            pattern_id=f"pat-{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.INCIDENT_RECURRENCE.value,
            description=f"Recurring incident: {most_common[0]} ({most_common[1]} occurrences)",
            frequency=most_common[1],
            confidence=confidence,
            evidence_ids=tuple(evidence[most_common[0]][:10]),
            first_observed=learning_input.period_start,
            last_observed=learning_input.period_end,
            observed_at=observed_at,
            project_id=None,
            metadata=tuple([("incident_type", most_common[0])]),
        )

    def _compute_statistical_confidence(self, sample_size: int, consistency: float) -> str:
        """
        Compute statistical confidence level.

        This is STATISTICAL confidence, NOT ML confidence.
        Based on sample size and consistency only.
        """
        if sample_size >= CONFIDENCE_HIGH_MIN_N and consistency >= CONFIDENCE_HIGH_MIN_CONSISTENCY:
            return ConfidenceLevel.HIGH.value
        elif sample_size >= CONFIDENCE_MEDIUM_MIN_N and consistency >= CONFIDENCE_MEDIUM_MIN_CONSISTENCY:
            return ConfidenceLevel.MEDIUM.value
        elif sample_size >= CONFIDENCE_LOW_MIN_N and consistency >= CONFIDENCE_LOW_MIN_CONSISTENCY:
            return ConfidenceLevel.LOW.value
        else:
            return ConfidenceLevel.INSUFFICIENT.value

    # -------------------------------------------------------------------------
    # Trend Observation (DETERMINISTIC, NO PREDICTION)
    # -------------------------------------------------------------------------

    def _observe_trends(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> List[TrendObservation]:
        """
        Observe trends in historical data.

        DETERMINISTIC: Same inputs = same trends.
        NO PREDICTION: Only observes past, never predicts future.
        """
        trends = []

        # Execution success rate trend
        exec_trend = self._observe_execution_trend(learning_input, observed_at)
        if exec_trend:
            trends.append(exec_trend)
            self._append_record(self._trends_file, exec_trend.to_dict())

        # Verification pass rate trend
        ver_trend = self._observe_verification_trend(learning_input, observed_at)
        if ver_trend:
            trends.append(ver_trend)
            self._append_record(self._trends_file, ver_trend.to_dict())

        # Incident frequency trend
        inc_trend = self._observe_incident_trend(learning_input, observed_at)
        if inc_trend:
            trends.append(inc_trend)
            self._append_record(self._trends_file, inc_trend.to_dict())

        return trends

    def _observe_execution_trend(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[TrendObservation]:
        """Observe execution success rate trend."""
        if len(learning_input.execution_results) < 10:
            return None

        # Split into two halves
        mid = len(learning_input.execution_results) // 2
        first_half = learning_input.execution_results[:mid]
        second_half = learning_input.execution_results[mid:]

        first_rate = sum(1 for e in first_half if e.get("status") == "execution_success") / len(first_half)
        second_rate = sum(1 for e in second_half if e.get("status") == "execution_success") / len(second_half)

        change = second_rate - first_rate
        if abs(change) < 0.05:  # Less than 5% change is stable
            direction = TrendDirection.STABLE.value
        elif change > 0:
            direction = TrendDirection.INCREASING.value
        else:
            direction = TrendDirection.DECREASING.value

        confidence = self._compute_statistical_confidence(
            len(learning_input.execution_results),
            abs(change) if abs(change) > 0.1 else 0.5,
        )

        return TrendObservation(
            trend_id=f"trend-{uuid.uuid4().hex[:12]}",
            metric_name="execution_success_rate",
            direction=direction,
            change_rate=change * 100,  # Percentage
            period_count=2,
            period_unit="half_period",
            start_value=first_rate,
            end_value=second_rate,
            observed_at=observed_at,
            confidence=confidence,
            project_id=None,
        )

    def _observe_verification_trend(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[TrendObservation]:
        """Observe verification pass rate trend."""
        if len(learning_input.verification_results) < 10:
            return None

        mid = len(learning_input.verification_results) // 2
        first_half = learning_input.verification_results[:mid]
        second_half = learning_input.verification_results[mid:]

        first_rate = sum(1 for v in first_half if v.get("verification_status") == "passed") / len(first_half)
        second_rate = sum(1 for v in second_half if v.get("verification_status") == "passed") / len(second_half)

        change = second_rate - first_rate
        if abs(change) < 0.05:
            direction = TrendDirection.STABLE.value
        elif change > 0:
            direction = TrendDirection.INCREASING.value
        else:
            direction = TrendDirection.DECREASING.value

        confidence = self._compute_statistical_confidence(
            len(learning_input.verification_results),
            abs(change) if abs(change) > 0.1 else 0.5,
        )

        return TrendObservation(
            trend_id=f"trend-{uuid.uuid4().hex[:12]}",
            metric_name="verification_pass_rate",
            direction=direction,
            change_rate=change * 100,
            period_count=2,
            period_unit="half_period",
            start_value=first_rate,
            end_value=second_rate,
            observed_at=observed_at,
            confidence=confidence,
            project_id=None,
        )

    def _observe_incident_trend(
        self,
        learning_input: LearningInput,
        observed_at: str,
    ) -> Optional[TrendObservation]:
        """Observe incident frequency trend."""
        if len(learning_input.incident_summaries) < 4:
            return None

        mid = len(learning_input.incident_summaries) // 2
        first_count = mid
        second_count = len(learning_input.incident_summaries) - mid

        if first_count == 0:
            return None

        change_rate = (second_count - first_count) / first_count
        if abs(change_rate) < 0.1:
            direction = TrendDirection.STABLE.value
        elif change_rate > 0:
            direction = TrendDirection.INCREASING.value
        else:
            direction = TrendDirection.DECREASING.value

        confidence = self._compute_statistical_confidence(
            len(learning_input.incident_summaries),
            abs(change_rate) if abs(change_rate) > 0.2 else 0.5,
        )

        return TrendObservation(
            trend_id=f"trend-{uuid.uuid4().hex[:12]}",
            metric_name="incident_frequency",
            direction=direction,
            change_rate=change_rate * 100,
            period_count=2,
            period_unit="half_period",
            start_value=float(first_count),
            end_value=float(second_count),
            observed_at=observed_at,
            confidence=confidence,
            project_id=None,
        )

    # -------------------------------------------------------------------------
    # Read Operations (READ-ONLY)
    # -------------------------------------------------------------------------

    def get_patterns(
        self,
        limit: int = 100,
        pattern_type: Optional[str] = None,
    ) -> List[ObservedPattern]:
        """Get observed patterns."""
        patterns = []
        for record in self._read_records(self._patterns_file):
            if pattern_type and record.get("pattern_type") != pattern_type:
                continue
            patterns.append(ObservedPattern.from_dict(record))
        # Most recent first
        patterns.reverse()
        return patterns[:limit]

    def get_trends(
        self,
        limit: int = 100,
        metric_name: Optional[str] = None,
    ) -> List[TrendObservation]:
        """Get observed trends."""
        trends = []
        for record in self._read_records(self._trends_file):
            if metric_name and record.get("metric_name") != metric_name:
                continue
            trends.append(TrendObservation.from_dict(record))
        trends.reverse()
        return trends[:limit]

    def get_history(
        self,
        limit: int = 100,
        entry_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Get memory history."""
        entries = []
        for record in self._read_records(self._memory_file):
            if entry_type and record.get("entry_type") != entry_type:
                continue
            if project_id and record.get("project_id") != project_id:
                continue
            entries.append(MemoryEntry.from_dict(record))
        entries.reverse()
        return entries[:limit]

    def get_aggregates(
        self,
        limit: int = 100,
        aggregate_type: Optional[str] = None,
    ) -> List[HistoricalAggregate]:
        """Get historical aggregates."""
        aggregates = []
        for record in self._read_records(self._aggregates_file):
            if aggregate_type and record.get("aggregate_type") != aggregate_type:
                continue
            aggregates.append(HistoricalAggregate.from_dict(record))
        aggregates.reverse()
        return aggregates[:limit]

    def get_summaries(self, limit: int = 10) -> List[LearningSummary]:
        """Get learning summaries."""
        summaries = []
        for record in self._read_records(self._summaries_file):
            summaries.append(LearningSummary.from_dict(record))
        summaries.reverse()
        return summaries[:limit]

    def get_latest_summary(self) -> Optional[LearningSummary]:
        """Get the most recent learning summary."""
        summaries = self.get_summaries(limit=1)
        return summaries[0] if summaries else None

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _append_record(self, file_path: Path, record: Dict[str, Any]) -> None:
        """Append a record to a JSONL file with fsync."""
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'a') as f:
                f.write(json.dumps(record) + '\n')
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass  # Logging failure doesn't affect learning

    def _read_records(self, file_path: Path) -> List[Dict[str, Any]]:
        """Read all records from a JSONL file."""
        if not file_path.exists():
            return []

        records = []
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass

        return records


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------

# Singleton instance
_engine: Optional[LearningEngine] = None


def get_learning_engine(
    patterns_file: Optional[Path] = None,
    aggregates_file: Optional[Path] = None,
    trends_file: Optional[Path] = None,
    memory_file: Optional[Path] = None,
    summaries_file: Optional[Path] = None,
) -> LearningEngine:
    """Get the learning engine singleton."""
    global _engine
    if _engine is None:
        _engine = LearningEngine(
            patterns_file=patterns_file,
            aggregates_file=aggregates_file,
            trends_file=trends_file,
            memory_file=memory_file,
            summaries_file=summaries_file,
        )
    return _engine


def analyze_history(learning_input: LearningInput) -> LearningSummary:
    """
    Analyze historical data.

    Convenience function using singleton engine.
    """
    engine = get_learning_engine()
    return engine.analyze(learning_input)


def get_learning_patterns(
    limit: int = 100,
    pattern_type: Optional[str] = None,
) -> List[ObservedPattern]:
    """Get observed patterns."""
    engine = get_learning_engine()
    return engine.get_patterns(limit=limit, pattern_type=pattern_type)


def get_learning_trends(
    limit: int = 100,
    metric_name: Optional[str] = None,
) -> List[TrendObservation]:
    """Get observed trends."""
    engine = get_learning_engine()
    return engine.get_trends(limit=limit, metric_name=metric_name)


def get_learning_history(
    limit: int = 100,
    entry_type: Optional[str] = None,
    project_id: Optional[str] = None,
) -> List[MemoryEntry]:
    """Get memory history."""
    engine = get_learning_engine()
    return engine.get_history(limit=limit, entry_type=entry_type, project_id=project_id)


def get_learning_summary() -> Optional[LearningSummary]:
    """Get latest learning summary."""
    engine = get_learning_engine()
    return engine.get_latest_summary()
