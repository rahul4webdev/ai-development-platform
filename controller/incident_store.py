"""
Phase 17B: Incident Store - Append-Only Persistence

This module provides immutable, append-only storage for incidents.
Once persisted, incidents are NEVER modified or deleted.

CRITICAL CONSTRAINTS:
- APPEND-ONLY: Incidents are never edited or deleted
- IMMUTABLE: No update or delete methods
- FSYNC: All writes are durable (fsync on each persist)
- AUDIT TRAIL: Corrections create new incidents, not edits
- READ-ONLY QUERIES: All query methods are pure reads

To "correct" an incident, create a new incident that references the original.
"""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .incident_model import (
    Incident,
    IncidentSummary,
    IncidentType,
    IncidentSeverity,
    IncidentScope,
    IncidentState,
)

logger = logging.getLogger("incident_store")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INCIDENTS_DIR = Path(os.getenv(
    "INCIDENTS_DIR",
    "/home/aitesting.mybd.in/jobs/incidents"
))

# Fallback for local development/testing
try:
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    INCIDENTS_DIR = Path("/tmp/incidents")
    INCIDENTS_DIR.mkdir(parents=True, exist_ok=True)

INCIDENTS_FILE = INCIDENTS_DIR / "incidents.jsonl"
INCIDENTS_AUDIT_FILE = INCIDENTS_DIR / "incident_audit.log"


# -----------------------------------------------------------------------------
# Incident Store (Append-Only)
# -----------------------------------------------------------------------------
class IncidentStore:
    """
    Append-only storage for incidents.

    CRITICAL: This store has NO methods for editing or deleting incidents.
    Once an incident is persisted, it is IMMUTABLE.

    To "correct" an incident:
    1. Create a new incident
    2. Reference the original incident_id in metadata
    3. Persist the new incident

    This ensures a complete audit trail.
    """

    def __init__(self, incidents_file: Optional[Path] = None):
        """Initialize the store."""
        self._incidents_file = incidents_file or INCIDENTS_FILE
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # WRITE Operations (Append-Only)
    # -------------------------------------------------------------------------

    def persist(self, incidents: List[Incident]) -> int:
        """
        Persist incidents to storage.

        This is an APPEND-ONLY operation.
        Incidents are added to the file, never overwritten.

        Returns: Number of incidents persisted
        """
        if not incidents:
            return 0

        with self._lock:
            try:
                # Ensure directory exists
                self._incidents_file.parent.mkdir(parents=True, exist_ok=True)

                # Append incidents (never overwrite)
                with open(self._incidents_file, "a") as f:
                    for incident in incidents:
                        f.write(json.dumps(incident.to_dict()) + "\n")
                    # fsync for durability
                    f.flush()
                    os.fsync(f.fileno())

                # Log to audit trail
                self._log_audit("INCIDENTS_PERSISTED", {
                    "count": len(incidents),
                    "incident_ids": [i.incident_id for i in incidents],
                })

                logger.debug(f"Persisted {len(incidents)} incidents")
                return len(incidents)

            except Exception as e:
                logger.error(f"Incident persistence failed: {e}")
                return 0

    def persist_single(self, incident: Incident) -> bool:
        """
        Persist a single incident.

        Returns: True if successful, False otherwise
        """
        return self.persist([incident]) == 1

    # -------------------------------------------------------------------------
    # READ Operations
    # -------------------------------------------------------------------------

    def read_incidents(
        self,
        since: Optional[datetime] = None,
        incident_type: Optional[IncidentType] = None,
        severity: Optional[IncidentSeverity] = None,
        scope: Optional[IncidentScope] = None,
        state: Optional[IncidentState] = None,
        project_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Incident]:
        """
        Read incidents from storage with optional filters.

        This is a READ-ONLY operation.
        """
        incidents = []

        if not self._incidents_file.exists():
            return incidents

        try:
            with open(self._incidents_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        incident = Incident.from_dict(data)

                        # Apply filters
                        if since:
                            try:
                                incident_ts = datetime.fromisoformat(
                                    incident.created_at.replace("Z", "+00:00")
                                ).replace(tzinfo=None)
                                if incident_ts < since:
                                    continue
                            except (ValueError, AttributeError):
                                pass

                        if incident_type and incident.incident_type != incident_type.value:
                            continue

                        if severity and incident.severity != severity.value:
                            continue

                        if scope and incident.scope != scope.value:
                            continue

                        if state and incident.state != state.value:
                            continue

                        if project_id and incident.project_id != project_id:
                            continue

                        incidents.append(incident)

                        if len(incidents) >= limit:
                            break

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed incident: {e}")
                        continue

        except Exception as e:
            logger.error(f"Incident read failed: {e}")

        return incidents

    def read_recent(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> List[Incident]:
        """
        Read recent incidents.

        Convenience method for getting incidents from the last N hours.
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.read_incidents(since=since, limit=limit)

    def get_by_id(self, incident_id: str) -> Optional[Incident]:
        """
        Get a specific incident by ID.

        Returns None if not found.
        """
        if not self._incidents_file.exists():
            return None

        try:
            with open(self._incidents_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("incident_id") == incident_id:
                            return Incident.from_dict(data)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Incident lookup failed: {e}")

        return None

    def get_by_signal_id(self, signal_id: str) -> List[Incident]:
        """
        Get all incidents that reference a specific signal ID.

        Useful for tracing which incidents were created from a signal.
        """
        incidents = []

        if not self._incidents_file.exists():
            return incidents

        try:
            with open(self._incidents_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        signal_ids = data.get("source_signal_ids", [])
                        if signal_id in signal_ids:
                            incidents.append(Incident.from_dict(data))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Incident signal lookup failed: {e}")

        return incidents

    # -------------------------------------------------------------------------
    # SUMMARY Operations (Read-Only Aggregation)
    # -------------------------------------------------------------------------

    def get_summary(
        self,
        since: Optional[datetime] = None,
        limit_recent: int = 10,
    ) -> IncidentSummary:
        """
        Generate a summary of incidents.

        This is a READ-ONLY aggregation.
        """
        now = datetime.utcnow()
        if since is None:
            since = now - timedelta(hours=24)

        incidents = self.read_incidents(since=since, limit=10000)

        # Aggregate counts
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_scope: Dict[str, int] = {}
        by_state: Dict[str, int] = {}
        unknown_count = 0
        open_count = 0

        for incident in incidents:
            # By severity
            by_severity[incident.severity] = by_severity.get(incident.severity, 0) + 1

            # By type
            by_type[incident.incident_type] = by_type.get(incident.incident_type, 0) + 1

            # By scope
            by_scope[incident.scope] = by_scope.get(incident.scope, 0) + 1

            # By state
            by_state[incident.state] = by_state.get(incident.state, 0) + 1

            # Unknown count
            if (incident.severity == IncidentSeverity.UNKNOWN.value or
                incident.incident_type == IncidentType.UNKNOWN.value):
                unknown_count += 1

            # Open count
            if incident.state == IncidentState.OPEN.value:
                open_count += 1

        # Get recent incidents for summary
        recent = sorted(
            incidents,
            key=lambda i: i.created_at,
            reverse=True
        )[:limit_recent]

        recent_dicts = [
            {
                "incident_id": i.incident_id,
                "created_at": i.created_at,
                "incident_type": i.incident_type,
                "severity": i.severity,
                "title": i.title,
                "state": i.state,
            }
            for i in recent
        ]

        return IncidentSummary(
            generated_at=now.isoformat(),
            time_window_start=since.isoformat(),
            time_window_end=now.isoformat(),
            total_incidents=len(incidents),
            by_severity=by_severity,
            by_type=by_type,
            by_scope=by_scope,
            by_state=by_state,
            unknown_count=unknown_count,
            open_count=open_count,
            recent_incidents=recent_dicts,
        )

    # -------------------------------------------------------------------------
    # Storage Statistics (Read-Only)
    # -------------------------------------------------------------------------

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Read-only operation for observability.
        """
        stats = {
            "file_path": str(self._incidents_file),
            "file_exists": self._incidents_file.exists(),
            "file_size_bytes": 0,
            "total_incidents": 0,
        }

        if self._incidents_file.exists():
            stats["file_size_bytes"] = self._incidents_file.stat().st_size

            # Count incidents (line count)
            try:
                with open(self._incidents_file) as f:
                    stats["total_incidents"] = sum(1 for _ in f)
            except Exception:
                pass

        return stats

    # -------------------------------------------------------------------------
    # Audit Trail
    # -------------------------------------------------------------------------

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log to audit trail."""
        try:
            INCIDENTS_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                **details,
            }
            with open(INCIDENTS_AUDIT_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Incident audit log failed: {e}")


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------
_store: Optional[IncidentStore] = None


def get_incident_store() -> IncidentStore:
    """Get the global incident store instance."""
    global _store
    if _store is None:
        _store = IncidentStore()
    return _store


def persist_incidents(incidents: List[Incident]) -> int:
    """Persist incidents to storage."""
    return get_incident_store().persist(incidents)


def read_incidents(
    since: Optional[datetime] = None,
    incident_type: Optional[IncidentType] = None,
    severity: Optional[IncidentSeverity] = None,
    limit: int = 1000,
) -> List[Incident]:
    """Read incidents from storage."""
    return get_incident_store().read_incidents(
        since=since,
        incident_type=incident_type,
        severity=severity,
        limit=limit,
    )


def read_recent_incidents(hours: int = 24, limit: int = 50) -> List[Incident]:
    """Read recent incidents."""
    return get_incident_store().read_recent(hours=hours, limit=limit)


def get_incident_by_id(incident_id: str) -> Optional[Incident]:
    """Get incident by ID."""
    return get_incident_store().get_by_id(incident_id)


def get_incident_summary(since_hours: int = 24) -> IncidentSummary:
    """Get incident summary."""
    since = datetime.utcnow() - timedelta(hours=since_hours)
    return get_incident_store().get_summary(since=since)


logger.info("Incident Store module loaded (Phase 17B - APPEND-ONLY)")
