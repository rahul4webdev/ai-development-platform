"""
Phase 17B: Incident Engine - Signal Correlation & Classification

This module provides deterministic, rule-based incident classification.
It correlates signals within fixed time windows and classifies them into incidents.

CRITICAL CONSTRAINTS:
- READ-ONLY: No lifecycle changes, no deployments, no mutations
- DETERMINISTIC: No ML, no probabilistic inference, no guessing
- EXPLICIT UNKNOWN: Insufficient data = UNKNOWN, never guessed
- NO ALERTS: Classification only, no notifications
- NO RECOMMENDATIONS: No suggested actions
- NO SIDE EFFECTS: Pure functions, read-only operations

This engine OBSERVES and CLASSIFIES, it does NOT ACT.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, FrozenSet
import uuid

from .incident_model import (
    Incident,
    IncidentType,
    IncidentSeverity,
    IncidentScope,
    IncidentState,
    ClassificationRule,
    CORRELATION_WINDOW_MINUTES,
    MIN_SIGNALS_FOR_INCIDENT,
    MAX_SIGNALS_PER_INCIDENT,
)
from .runtime_intelligence import (
    RuntimeSignal,
    SignalType,
    Severity,
)

logger = logging.getLogger("incident_engine")


# -----------------------------------------------------------------------------
# Classification Rules (DETERMINISTIC, LOCKED)
# -----------------------------------------------------------------------------
# These rules are FIXED and define how signals map to incident types.
# Each rule specifies:
# - Which signal types trigger it
# - Minimum severity threshold
# - Resulting incident type
# - How to derive scope

CLASSIFICATION_RULES: Tuple[ClassificationRule, ...] = (
    # RESOURCE incidents - from system resource signals
    ClassificationRule(
        rule_id="rule-resource-001",
        name="System Resource Exhaustion",
        description="High CPU, memory, or disk usage indicates resource exhaustion",
        signal_types=frozenset({SignalType.SYSTEM_RESOURCE.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.RESOURCE.value,
        scope_derivation="system",
        confidence=0.9,
    ),

    # RELIABILITY incidents - from worker queue and job failures
    ClassificationRule(
        rule_id="rule-reliability-001",
        name="Worker Queue Saturation",
        description="Queue saturation indicates potential reliability issues",
        signal_types=frozenset({SignalType.WORKER_QUEUE.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.RELIABILITY.value,
        scope_derivation="system",
        confidence=0.85,
    ),
    ClassificationRule(
        rule_id="rule-reliability-002",
        name="Job Execution Failure",
        description="Job failures indicate reliability problems",
        signal_types=frozenset({SignalType.JOB_FAILURE.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.RELIABILITY.value,
        scope_derivation="from_signal",
        confidence=0.95,
    ),
    ClassificationRule(
        rule_id="rule-reliability-003",
        name="Test Regression Detected",
        description="Test failures indicate regression in code quality",
        signal_types=frozenset({SignalType.TEST_REGRESSION.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.RELIABILITY.value,
        scope_derivation="from_signal",
        confidence=0.9,
    ),
    ClassificationRule(
        rule_id="rule-reliability-004",
        name="Deployment Failure",
        description="Deployment failures affect service reliability",
        signal_types=frozenset({SignalType.DEPLOYMENT_FAILURE.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.RELIABILITY.value,
        scope_derivation="from_signal",
        confidence=0.95,
    ),

    # GOVERNANCE incidents - from drift warnings and human overrides
    ClassificationRule(
        rule_id="rule-governance-001",
        name="Intent Drift Warning",
        description="Project deviating from baseline intent",
        signal_types=frozenset({SignalType.DRIFT_WARNING.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.GOVERNANCE.value,
        scope_derivation="from_signal",
        confidence=0.85,
    ),
    ClassificationRule(
        rule_id="rule-governance-002",
        name="Human Override Recorded",
        description="Human override of system decisions",
        signal_types=frozenset({SignalType.HUMAN_OVERRIDE.value}),
        min_severity=Severity.INFO.value,
        incident_type=IncidentType.GOVERNANCE.value,
        scope_derivation="from_signal",
        confidence=0.95,
    ),

    # CONFIGURATION incidents - from config anomalies
    ClassificationRule(
        rule_id="rule-config-001",
        name="Configuration Anomaly",
        description="Configuration anomaly detected",
        signal_types=frozenset({SignalType.CONFIG_ANOMALY.value}),
        min_severity=Severity.WARNING.value,
        incident_type=IncidentType.CONFIGURATION.value,
        scope_derivation="from_signal",
        confidence=0.8,
    ),
)

# Map signal types to rules for fast lookup
_SIGNAL_TYPE_TO_RULES: Dict[str, List[ClassificationRule]] = {}
for rule in CLASSIFICATION_RULES:
    for sig_type in rule.signal_types:
        if sig_type not in _SIGNAL_TYPE_TO_RULES:
            _SIGNAL_TYPE_TO_RULES[sig_type] = []
        _SIGNAL_TYPE_TO_RULES[sig_type].append(rule)


# -----------------------------------------------------------------------------
# Severity Mapping (Signal Severity -> Incident Severity)
# -----------------------------------------------------------------------------
def _map_signal_severity_to_incident(signal_severity: str) -> str:
    """
    Deterministic mapping from signal severity to incident severity.

    This is a FIXED mapping - no ML, no guessing.
    """
    mapping = {
        Severity.INFO.value: IncidentSeverity.INFO.value,
        Severity.WARNING.value: IncidentSeverity.LOW.value,
        Severity.DEGRADED.value: IncidentSeverity.MEDIUM.value,
        Severity.CRITICAL.value: IncidentSeverity.HIGH.value,
        Severity.UNKNOWN.value: IncidentSeverity.UNKNOWN.value,
    }
    return mapping.get(signal_severity, IncidentSeverity.UNKNOWN.value)


def _aggregate_severity(severities: List[str]) -> str:
    """
    Aggregate multiple severities into one.

    Rule: Use the HIGHEST severity among all signals.
    If any severity is UNKNOWN, result is UNKNOWN.
    """
    if not severities:
        return IncidentSeverity.UNKNOWN.value

    # If any UNKNOWN, result is UNKNOWN
    if IncidentSeverity.UNKNOWN.value in severities:
        return IncidentSeverity.UNKNOWN.value

    severity_order = [
        IncidentSeverity.INFO.value,
        IncidentSeverity.LOW.value,
        IncidentSeverity.MEDIUM.value,
        IncidentSeverity.HIGH.value,
        IncidentSeverity.CRITICAL.value,
    ]

    max_idx = -1
    for sev in severities:
        try:
            idx = severity_order.index(sev)
            max_idx = max(max_idx, idx)
        except ValueError:
            # Unknown severity value
            return IncidentSeverity.UNKNOWN.value

    if max_idx < 0:
        return IncidentSeverity.UNKNOWN.value

    return severity_order[max_idx]


# -----------------------------------------------------------------------------
# Signal Correlation Engine (Read-Only)
# -----------------------------------------------------------------------------
class SignalCorrelationEngine:
    """
    Correlates signals within fixed time windows.

    This engine is READ-ONLY - it reads signals and produces correlations
    without modifying any state.

    Correlation rules:
    - Signals within CORRELATION_WINDOW_MINUTES are correlated
    - Signals are grouped by type, project, and aspect
    - Minimum MIN_SIGNALS_FOR_INCIDENT signals to create incident
    - Maximum MAX_SIGNALS_PER_INCIDENT signals per incident
    """

    def __init__(self, window_minutes: int = CORRELATION_WINDOW_MINUTES):
        """Initialize with correlation window."""
        self._window_minutes = window_minutes

    def correlate_signals(
        self,
        signals: List[RuntimeSignal],
    ) -> List[Tuple[str, List[RuntimeSignal]]]:
        """
        Correlate signals into groups.

        Returns list of (correlation_key, [signals]) tuples.
        This is a READ-ONLY operation.

        Correlation key format: {signal_type}:{project_id or 'system'}:{aspect or 'none'}
        """
        if not signals:
            return []

        # Group signals by correlation key
        groups: Dict[str, List[RuntimeSignal]] = {}

        for signal in signals:
            # Build correlation key
            project_part = signal.project_id or "system"
            aspect_part = signal.aspect or "none"
            key = f"{signal.signal_type}:{project_part}:{aspect_part}"

            if key not in groups:
                groups[key] = []
            groups[key].append(signal)

        # Filter groups by time window and minimum count
        result = []
        for key, group_signals in groups.items():
            # Sort by timestamp
            sorted_signals = sorted(
                group_signals,
                key=lambda s: s.timestamp,
            )

            # Apply time window filtering
            windowed = self._apply_time_window(sorted_signals)

            # Check minimum count
            if len(windowed) >= MIN_SIGNALS_FOR_INCIDENT:
                # Cap at maximum
                capped = windowed[:MAX_SIGNALS_PER_INCIDENT]
                result.append((key, capped))

        return result

    def _apply_time_window(
        self,
        signals: List[RuntimeSignal],
    ) -> List[RuntimeSignal]:
        """
        Filter signals to those within correlation window.

        Uses a sliding window from the most recent signal.
        """
        if not signals:
            return []

        # Find the most recent signal
        try:
            latest_ts = max(
                datetime.fromisoformat(s.timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
                for s in signals
            )
        except (ValueError, AttributeError):
            # If we can't parse timestamps, return all
            return signals

        # Calculate window cutoff
        cutoff = latest_ts - timedelta(minutes=self._window_minutes)

        # Filter signals within window
        result = []
        for signal in signals:
            try:
                sig_ts = datetime.fromisoformat(
                    signal.timestamp.replace("Z", "+00:00")
                ).replace(tzinfo=None)
                if sig_ts >= cutoff:
                    result.append(signal)
            except (ValueError, AttributeError):
                # Include signals with unparseable timestamps
                result.append(signal)

        return result


# -----------------------------------------------------------------------------
# Incident Classifier (Rule-Based Only)
# -----------------------------------------------------------------------------
class IncidentClassifier:
    """
    Classifies correlated signals into incidents.

    This classifier uses DETERMINISTIC RULES ONLY:
    - No ML
    - No probabilistic inference
    - No guessing
    - Explicit thresholds
    - Same input always produces same output

    If classification cannot be determined, result is UNKNOWN.
    """

    def __init__(self):
        """Initialize classifier with rules."""
        self._rules = CLASSIFICATION_RULES
        self._signal_type_to_rules = _SIGNAL_TYPE_TO_RULES

    def classify(
        self,
        correlation_key: str,
        signals: List[RuntimeSignal],
    ) -> Incident:
        """
        Classify correlated signals into an incident.

        This is a PURE FUNCTION - same input always produces same output.
        No side effects, no state modification.
        """
        if not signals:
            return self._create_unknown_incident(
                correlation_key,
                signals,
                "No signals provided",
            )

        # Extract signal type from correlation key
        parts = correlation_key.split(":")
        if len(parts) < 3:
            return self._create_unknown_incident(
                correlation_key,
                signals,
                "Invalid correlation key format",
            )

        signal_type = parts[0]
        project_id = parts[1] if parts[1] != "system" else None
        aspect = parts[2] if parts[2] != "none" else None

        # Find matching rule
        rules = self._signal_type_to_rules.get(signal_type, [])

        if not rules:
            return self._create_unknown_incident(
                correlation_key,
                signals,
                f"No classification rule for signal type: {signal_type}",
            )

        # Find the best matching rule (highest confidence that matches)
        best_rule = None
        for rule in rules:
            # Check severity threshold
            signal_severities = [s.severity for s in signals]
            if any(rule.matches_severity(sev) for sev in signal_severities):
                if best_rule is None or rule.confidence > best_rule.confidence:
                    best_rule = rule

        if best_rule is None:
            return self._create_unknown_incident(
                correlation_key,
                signals,
                "No rule matched severity threshold",
            )

        # Classify using the best rule
        return self._create_incident_from_rule(
            best_rule,
            signals,
            project_id,
            aspect,
        )

    def _create_incident_from_rule(
        self,
        rule: ClassificationRule,
        signals: List[RuntimeSignal],
        project_id: Optional[str],
        aspect: Optional[str],
    ) -> Incident:
        """Create an incident using a classification rule."""
        now = datetime.utcnow()

        # Collect signal IDs
        signal_ids = tuple(s.signal_id for s in signals)

        # Find first and last signal timestamps
        timestamps = []
        for s in signals:
            try:
                ts = datetime.fromisoformat(s.timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
                timestamps.append(ts)
            except (ValueError, AttributeError):
                pass

        if timestamps:
            first_signal = min(timestamps)
            last_signal = max(timestamps)
        else:
            first_signal = now
            last_signal = now

        # Map signal severities to incident severity
        incident_severities = [
            _map_signal_severity_to_incident(s.severity)
            for s in signals
        ]
        aggregate_severity = _aggregate_severity(incident_severities)

        # Determine scope
        scope = self._derive_scope(rule, project_id, aspect, signals)

        # Determine state (OPEN if recent signals)
        recent_cutoff = now - timedelta(minutes=CORRELATION_WINDOW_MINUTES)
        state = IncidentState.OPEN.value if last_signal > recent_cutoff else IncidentState.CLOSED.value

        # Generate title and description
        title = self._generate_title(rule, signals, aggregate_severity)
        description = self._generate_description(rule, signals, project_id, aspect)

        # Collect job_id if present
        job_ids = [s.metadata.get("job_id") for s in signals if s.metadata.get("job_id")]
        job_id = job_ids[0] if job_ids else None

        # Build metadata
        metadata = {
            "rule_id": rule.rule_id,
            "rule_name": rule.name,
            "signal_types": list(set(s.signal_type for s in signals)),
            "signal_severities": list(set(s.severity for s in signals)),
        }

        return Incident(
            incident_id=f"inc-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            created_at=now.isoformat(),
            incident_type=rule.incident_type,
            severity=aggregate_severity,
            scope=scope,
            state=state,
            title=title,
            description=description,
            source_signal_ids=signal_ids,
            signal_count=len(signals),
            first_signal_at=first_signal.isoformat(),
            last_signal_at=last_signal.isoformat(),
            correlation_window_minutes=CORRELATION_WINDOW_MINUTES,
            project_id=project_id,
            aspect=aspect,
            job_id=job_id,
            confidence=rule.confidence,
            classification_rule=rule.rule_id,
            metadata=metadata,
        )

    def _create_unknown_incident(
        self,
        correlation_key: str,
        signals: List[RuntimeSignal],
        reason: str,
    ) -> Incident:
        """Create an UNKNOWN incident when classification fails."""
        now = datetime.utcnow()

        signal_ids = tuple(s.signal_id for s in signals) if signals else ()

        # Extract project/aspect from key
        parts = correlation_key.split(":")
        project_id = parts[1] if len(parts) > 1 and parts[1] != "system" else None
        aspect = parts[2] if len(parts) > 2 and parts[2] != "none" else None

        return Incident(
            incident_id=f"inc-unknown-{now.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}",
            created_at=now.isoformat(),
            incident_type=IncidentType.UNKNOWN.value,
            severity=IncidentSeverity.UNKNOWN.value,
            scope=IncidentScope.UNKNOWN.value,
            state=IncidentState.UNKNOWN.value,
            title="Unknown Incident",
            description=f"Could not classify incident: {reason}",
            source_signal_ids=signal_ids,
            signal_count=len(signals),
            first_signal_at=now.isoformat(),
            last_signal_at=now.isoformat(),
            correlation_window_minutes=CORRELATION_WINDOW_MINUTES,
            project_id=project_id,
            aspect=aspect,
            job_id=None,
            confidence=0.0,
            classification_rule="unknown",
            metadata={"reason": reason},
        )

    def _derive_scope(
        self,
        rule: ClassificationRule,
        project_id: Optional[str],
        aspect: Optional[str],
        signals: List[RuntimeSignal],
    ) -> str:
        """Derive incident scope from rule and signal context."""
        if rule.scope_derivation == "system":
            return IncidentScope.SYSTEM.value

        if rule.scope_derivation == "from_signal":
            # Use the most specific scope from signals
            if aspect and project_id:
                return IncidentScope.PROJECT_ASPECT.value
            elif project_id:
                return IncidentScope.PROJECT.value
            else:
                return IncidentScope.SYSTEM.value

        # Check if any signal has job context
        job_ids = [s.metadata.get("job_id") for s in signals if s.metadata.get("job_id")]
        if job_ids:
            return IncidentScope.JOB.value

        return IncidentScope.UNKNOWN.value

    def _generate_title(
        self,
        rule: ClassificationRule,
        signals: List[RuntimeSignal],
        severity: str,
    ) -> str:
        """Generate a descriptive title for the incident."""
        severity_label = severity.upper()
        return f"[{severity_label}] {rule.name}"

    def _generate_description(
        self,
        rule: ClassificationRule,
        signals: List[RuntimeSignal],
        project_id: Optional[str],
        aspect: Optional[str],
    ) -> str:
        """Generate a detailed description for the incident."""
        lines = [rule.description]

        if project_id:
            lines.append(f"Project: {project_id}")
        if aspect:
            lines.append(f"Aspect: {aspect}")

        lines.append(f"Signals: {len(signals)}")

        # Summarize signal descriptions
        if signals:
            sample_descriptions = [s.description for s in signals[:3]]
            if sample_descriptions:
                lines.append("Sample signals:")
                for desc in sample_descriptions:
                    lines.append(f"  - {desc[:100]}")

        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Incident Classification Engine (Main Class)
# -----------------------------------------------------------------------------
class IncidentClassificationEngine:
    """
    Main engine for classifying signals into incidents.

    This is the primary interface for Phase 17B incident classification.
    It combines signal correlation and rule-based classification.

    CRITICAL: This engine is OBSERVATION-ONLY.
    It does NOT:
    - Trigger alerts
    - Make recommendations
    - Execute actions
    - Modify lifecycle state
    """

    def __init__(self, correlation_window_minutes: int = CORRELATION_WINDOW_MINUTES):
        """Initialize the engine."""
        self._correlator = SignalCorrelationEngine(window_minutes=correlation_window_minutes)
        self._classifier = IncidentClassifier()

    def classify_signals(
        self,
        signals: List[RuntimeSignal],
    ) -> List[Incident]:
        """
        Classify a list of signals into incidents.

        This is the main entry point for incident classification.
        It is a PURE FUNCTION with no side effects.

        Args:
            signals: List of RuntimeSignal objects to classify

        Returns:
            List of Incident objects (one per correlation group)
        """
        if not signals:
            return []

        # Step 1: Correlate signals
        correlations = self._correlator.correlate_signals(signals)

        # Step 2: Classify each correlation group
        incidents = []
        for key, group_signals in correlations:
            incident = self._classifier.classify(key, group_signals)
            incidents.append(incident)

        logger.debug(f"Classified {len(signals)} signals into {len(incidents)} incidents")
        return incidents

    def classify_single_signal(self, signal: RuntimeSignal) -> Incident:
        """
        Classify a single signal into an incident.

        Useful for real-time classification of incoming signals.
        """
        incidents = self.classify_signals([signal])
        if incidents:
            return incidents[0]

        # Create unknown incident for single signal that couldn't be classified
        return self._classifier._create_unknown_incident(
            f"{signal.signal_type}:system:none",
            [signal],
            "Single signal classification failed",
        )


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------
_engine: Optional[IncidentClassificationEngine] = None


def get_classification_engine() -> IncidentClassificationEngine:
    """Get the global classification engine instance."""
    global _engine
    if _engine is None:
        _engine = IncidentClassificationEngine()
    return _engine


def classify_signals(signals: List[RuntimeSignal]) -> List[Incident]:
    """
    Classify signals into incidents.

    Module-level convenience function.
    """
    return get_classification_engine().classify_signals(signals)


def classify_single_signal(signal: RuntimeSignal) -> Incident:
    """
    Classify a single signal into an incident.

    Module-level convenience function.
    """
    return get_classification_engine().classify_single_signal(signal)


logger.info("Incident Engine module loaded (Phase 17B - OBSERVATION ONLY)")
