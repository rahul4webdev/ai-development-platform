"""
Phase 17C: Recommendation Store - Append-Only Persistence

This module provides immutable, append-only storage for recommendations.
Once persisted, recommendations are NEVER modified or deleted.

CRITICAL CONSTRAINTS:
- APPEND-ONLY: Recommendations are never edited or deleted
- IMMUTABLE: No update or delete methods
- FSYNC: All writes are durable (fsync on each persist)
- AUDIT TRAIL: Status changes create new records, not edits
- READ-ONLY QUERIES: All query methods are pure reads
- SEPARATE APPROVAL LOG: Approvals/dismissals logged separately

To "update" a recommendation status, use the approval log.
"""

import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

from .recommendation_model import (
    Recommendation,
    RecommendationSummary,
    ApprovalRecord,
    RecommendationType,
    RecommendationSeverity,
    RecommendationApproval,
    RecommendationStatus,
)

logger = logging.getLogger("recommendation_store")


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
RECOMMENDATIONS_DIR = Path(os.getenv(
    "RECOMMENDATIONS_DIR",
    "/home/aitesting.mybd.in/jobs/recommendations"
))

# Fallback for local development/testing
try:
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    RECOMMENDATIONS_DIR = Path("/tmp/recommendations")
    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)

RECOMMENDATIONS_FILE = RECOMMENDATIONS_DIR / "recommendations.jsonl"
APPROVALS_FILE = RECOMMENDATIONS_DIR / "approvals.jsonl"
RECOMMENDATIONS_AUDIT_FILE = RECOMMENDATIONS_DIR / "recommendation_audit.log"


# -----------------------------------------------------------------------------
# Recommendation Store (Append-Only)
# -----------------------------------------------------------------------------
class RecommendationStore:
    """
    Append-only storage for recommendations.

    CRITICAL: This store has NO methods for editing or deleting recommendations.
    Once a recommendation is persisted, it is IMMUTABLE.

    To "update" a recommendation status:
    1. Create an ApprovalRecord (approve/dismiss)
    2. Persist the ApprovalRecord to the approvals log
    3. Query combines recommendation + approval status

    This ensures a complete audit trail.
    """

    def __init__(
        self,
        recommendations_file: Optional[Path] = None,
        approvals_file: Optional[Path] = None,
    ):
        """Initialize the store."""
        self._recommendations_file = recommendations_file or RECOMMENDATIONS_FILE
        self._approvals_file = approvals_file or APPROVALS_FILE
        self._lock = threading.Lock()

    # -------------------------------------------------------------------------
    # WRITE Operations (Append-Only)
    # -------------------------------------------------------------------------

    def persist(self, recommendations: List[Recommendation]) -> int:
        """
        Persist recommendations to storage.

        This is an APPEND-ONLY operation.
        Recommendations are added to the file, never overwritten.

        Returns: Number of recommendations persisted
        """
        if not recommendations:
            return 0

        with self._lock:
            try:
                # Ensure directory exists
                self._recommendations_file.parent.mkdir(parents=True, exist_ok=True)

                # Append recommendations (never overwrite)
                with open(self._recommendations_file, "a") as f:
                    for rec in recommendations:
                        f.write(json.dumps(rec.to_dict()) + "\n")
                    # fsync for durability
                    f.flush()
                    os.fsync(f.fileno())

                # Log to audit trail
                self._log_audit("RECOMMENDATIONS_PERSISTED", {
                    "count": len(recommendations),
                    "recommendation_ids": [r.recommendation_id for r in recommendations],
                })

                logger.debug(f"Persisted {len(recommendations)} recommendations")
                return len(recommendations)

            except Exception as e:
                logger.error(f"Recommendation persistence failed: {e}")
                return 0

    def persist_single(self, recommendation: Recommendation) -> bool:
        """
        Persist a single recommendation.

        Returns: True if successful, False otherwise
        """
        return self.persist([recommendation]) == 1

    def persist_approval(self, record: ApprovalRecord) -> bool:
        """
        Persist an approval/dismissal record.

        This is an APPEND-ONLY operation to the approvals log.
        Does NOT modify the original recommendation.

        Returns: True if successful, False otherwise
        """
        with self._lock:
            try:
                # Ensure directory exists
                self._approvals_file.parent.mkdir(parents=True, exist_ok=True)

                # Append approval record
                with open(self._approvals_file, "a") as f:
                    f.write(json.dumps(record.to_dict()) + "\n")
                    f.flush()
                    os.fsync(f.fileno())

                # Log to audit trail
                self._log_audit(f"RECOMMENDATION_{record.action.upper()}", {
                    "recommendation_id": record.recommendation_id,
                    "user_id": record.user_id,
                    "reason": record.reason,
                })

                logger.debug(f"Persisted approval record for {record.recommendation_id}")
                return True

            except Exception as e:
                logger.error(f"Approval persistence failed: {e}")
                return False

    # -------------------------------------------------------------------------
    # READ Operations
    # -------------------------------------------------------------------------

    def read_recommendations(
        self,
        since: Optional[datetime] = None,
        recommendation_type: Optional[RecommendationType] = None,
        severity: Optional[RecommendationSeverity] = None,
        status: Optional[RecommendationStatus] = None,
        project_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Recommendation]:
        """
        Read recommendations from storage with optional filters.

        This is a READ-ONLY operation.
        """
        recommendations = []

        if not self._recommendations_file.exists():
            return recommendations

        # Load approval records to determine current status
        approvals = self._load_approvals()

        try:
            with open(self._recommendations_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        rec = Recommendation.from_dict(data)

                        # Apply approval status
                        rec_with_status = self._apply_approval_status(rec, approvals)

                        # Apply filters
                        if since:
                            try:
                                rec_ts = datetime.fromisoformat(
                                    rec_with_status.created_at.replace("Z", "+00:00")
                                ).replace(tzinfo=None)
                                if rec_ts < since:
                                    continue
                            except (ValueError, AttributeError):
                                pass

                        if recommendation_type and rec_with_status.recommendation_type != recommendation_type.value:
                            continue

                        if severity and rec_with_status.severity != severity.value:
                            continue

                        if status and rec_with_status.status != status.value:
                            continue

                        if project_id and rec_with_status.project_id != project_id:
                            continue

                        recommendations.append(rec_with_status)

                        if len(recommendations) >= limit:
                            break

                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed recommendation: {e}")
                        continue

        except Exception as e:
            logger.error(f"Recommendation read failed: {e}")

        return recommendations

    def read_recent(
        self,
        hours: int = 24,
        limit: int = 50,
    ) -> List[Recommendation]:
        """
        Read recent recommendations.

        Convenience method for getting recommendations from the last N hours.
        """
        since = datetime.utcnow() - timedelta(hours=hours)
        return self.read_recommendations(since=since, limit=limit)

    def read_pending(
        self,
        limit: int = 50,
    ) -> List[Recommendation]:
        """
        Read pending recommendations (awaiting human action).

        Returns recommendations that haven't been approved or dismissed.
        """
        return self.read_recommendations(
            status=RecommendationStatus.PENDING,
            limit=limit,
        )

    def read_pending_approvals(
        self,
        limit: int = 50,
    ) -> List[Recommendation]:
        """
        Read recommendations requiring explicit approval.

        Returns pending recommendations with EXPLICIT_APPROVAL_REQUIRED.
        """
        pending = self.read_pending(limit=limit * 2)  # Get more to filter
        return [
            r for r in pending
            if r.approval_required == RecommendationApproval.EXPLICIT_APPROVAL_REQUIRED.value
        ][:limit]

    def get_by_id(self, recommendation_id: str) -> Optional[Recommendation]:
        """
        Get a specific recommendation by ID.

        Returns None if not found.
        """
        if not self._recommendations_file.exists():
            return None

        approvals = self._load_approvals()

        try:
            with open(self._recommendations_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("recommendation_id") == recommendation_id:
                            rec = Recommendation.from_dict(data)
                            return self._apply_approval_status(rec, approvals)
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Recommendation lookup failed: {e}")

        return None

    def get_by_incident_id(self, incident_id: str) -> List[Recommendation]:
        """
        Get all recommendations that reference a specific incident ID.

        Useful for tracing which recommendations were created from an incident.
        """
        recommendations = []

        if not self._recommendations_file.exists():
            return recommendations

        approvals = self._load_approvals()

        try:
            with open(self._recommendations_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        incident_ids = data.get("source_incident_ids", [])
                        if incident_id in incident_ids:
                            rec = Recommendation.from_dict(data)
                            recommendations.append(
                                self._apply_approval_status(rec, approvals)
                            )
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Recommendation incident lookup failed: {e}")

        return recommendations

    def get_approval_history(
        self,
        recommendation_id: str,
    ) -> List[ApprovalRecord]:
        """
        Get approval history for a recommendation.

        Returns all approval/dismissal records for the given recommendation.
        """
        records = []

        if not self._approvals_file.exists():
            return records

        try:
            with open(self._approvals_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("recommendation_id") == recommendation_id:
                            records.append(ApprovalRecord.from_dict(data))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.error(f"Approval history lookup failed: {e}")

        return records

    # -------------------------------------------------------------------------
    # Approval Operations (Create New Records)
    # -------------------------------------------------------------------------

    def approve(
        self,
        recommendation_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> Optional[ApprovalRecord]:
        """
        Approve a recommendation.

        This creates a NEW approval record, it does NOT modify the original.

        Returns: ApprovalRecord if successful, None otherwise
        """
        # Verify recommendation exists
        rec = self.get_by_id(recommendation_id)
        if rec is None:
            logger.warning(f"Cannot approve: recommendation {recommendation_id} not found")
            return None

        # Check if already approved/dismissed
        if rec.status != RecommendationStatus.PENDING.value:
            logger.warning(f"Cannot approve: recommendation {recommendation_id} is {rec.status}")
            return None

        # Create approval record
        record = ApprovalRecord(
            record_id=f"approval-{uuid.uuid4().hex[:12]}",
            recommendation_id=recommendation_id,
            action="approved",
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            reason=reason,
            metadata={},
        )

        if self.persist_approval(record):
            return record
        return None

    def dismiss(
        self,
        recommendation_id: str,
        user_id: str,
        reason: Optional[str] = None,
    ) -> Optional[ApprovalRecord]:
        """
        Dismiss a recommendation.

        This creates a NEW approval record, it does NOT modify the original.

        Returns: ApprovalRecord if successful, None otherwise
        """
        # Verify recommendation exists
        rec = self.get_by_id(recommendation_id)
        if rec is None:
            logger.warning(f"Cannot dismiss: recommendation {recommendation_id} not found")
            return None

        # Check if already approved/dismissed
        if rec.status != RecommendationStatus.PENDING.value:
            logger.warning(f"Cannot dismiss: recommendation {recommendation_id} is {rec.status}")
            return None

        # Create dismissal record
        record = ApprovalRecord(
            record_id=f"dismissal-{uuid.uuid4().hex[:12]}",
            recommendation_id=recommendation_id,
            action="dismissed",
            user_id=user_id,
            timestamp=datetime.utcnow().isoformat(),
            reason=reason,
            metadata={},
        )

        if self.persist_approval(record):
            return record
        return None

    # -------------------------------------------------------------------------
    # SUMMARY Operations (Read-Only Aggregation)
    # -------------------------------------------------------------------------

    def get_summary(
        self,
        since: Optional[datetime] = None,
        limit_recent: int = 10,
    ) -> RecommendationSummary:
        """
        Generate a summary of recommendations.

        This is a READ-ONLY aggregation.
        """
        now = datetime.utcnow()
        if since is None:
            since = now - timedelta(hours=24)

        recommendations = self.read_recommendations(since=since, limit=10000)

        # Aggregate counts
        by_severity: Dict[str, int] = {}
        by_type: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_approval: Dict[str, int] = {}
        pending_count = 0
        pending_approval_count = 0
        unknown_count = 0

        for rec in recommendations:
            # By severity
            by_severity[rec.severity] = by_severity.get(rec.severity, 0) + 1

            # By type
            by_type[rec.recommendation_type] = by_type.get(rec.recommendation_type, 0) + 1

            # By status
            by_status[rec.status] = by_status.get(rec.status, 0) + 1

            # By approval
            by_approval[rec.approval_required] = by_approval.get(rec.approval_required, 0) + 1

            # Pending count
            if rec.status == RecommendationStatus.PENDING.value:
                pending_count += 1
                if rec.approval_required == RecommendationApproval.EXPLICIT_APPROVAL_REQUIRED.value:
                    pending_approval_count += 1

            # Unknown count
            if (rec.severity == RecommendationSeverity.UNKNOWN.value or
                rec.recommendation_type == RecommendationType.NO_ACTION.value):
                unknown_count += 1

        # Get recent recommendations for summary
        recent = sorted(
            recommendations,
            key=lambda r: r.created_at,
            reverse=True
        )[:limit_recent]

        recent_dicts = [
            {
                "recommendation_id": r.recommendation_id,
                "created_at": r.created_at,
                "recommendation_type": r.recommendation_type,
                "severity": r.severity,
                "title": r.title,
                "status": r.status,
                "approval_required": r.approval_required,
            }
            for r in recent
        ]

        return RecommendationSummary(
            generated_at=now.isoformat(),
            time_window_start=since.isoformat(),
            time_window_end=now.isoformat(),
            total_recommendations=len(recommendations),
            by_severity=by_severity,
            by_type=by_type,
            by_status=by_status,
            by_approval=by_approval,
            pending_count=pending_count,
            pending_approval_count=pending_approval_count,
            unknown_count=unknown_count,
            recent_recommendations=recent_dicts,
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
            "recommendations_file": str(self._recommendations_file),
            "approvals_file": str(self._approvals_file),
            "recommendations_exists": self._recommendations_file.exists(),
            "approvals_exists": self._approvals_file.exists(),
            "recommendations_size_bytes": 0,
            "approvals_size_bytes": 0,
            "total_recommendations": 0,
            "total_approvals": 0,
        }

        if self._recommendations_file.exists():
            stats["recommendations_size_bytes"] = self._recommendations_file.stat().st_size
            try:
                with open(self._recommendations_file) as f:
                    stats["total_recommendations"] = sum(1 for _ in f)
            except Exception:
                pass

        if self._approvals_file.exists():
            stats["approvals_size_bytes"] = self._approvals_file.stat().st_size
            try:
                with open(self._approvals_file) as f:
                    stats["total_approvals"] = sum(1 for _ in f)
            except Exception:
                pass

        return stats

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _load_approvals(self) -> Dict[str, ApprovalRecord]:
        """Load all approval records indexed by recommendation_id."""
        approvals: Dict[str, ApprovalRecord] = {}

        if not self._approvals_file.exists():
            return approvals

        try:
            with open(self._approvals_file) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        record = ApprovalRecord.from_dict(data)
                        # Latest approval wins (overwrites previous)
                        approvals[record.recommendation_id] = record
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except Exception as e:
            logger.warning(f"Failed to load approvals: {e}")

        return approvals

    def _apply_approval_status(
        self,
        recommendation: Recommendation,
        approvals: Dict[str, ApprovalRecord],
    ) -> Recommendation:
        """Apply approval status to a recommendation (creates new instance)."""
        approval = approvals.get(recommendation.recommendation_id)

        if approval is None:
            # Check if expired
            if recommendation.expires_at:
                try:
                    expires = datetime.fromisoformat(
                        recommendation.expires_at.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if datetime.utcnow() > expires:
                        # Return new recommendation with EXPIRED status
                        return Recommendation(
                            recommendation_id=recommendation.recommendation_id,
                            created_at=recommendation.created_at,
                            recommendation_type=recommendation.recommendation_type,
                            severity=recommendation.severity,
                            approval_required=recommendation.approval_required,
                            status=RecommendationStatus.EXPIRED.value,
                            title=recommendation.title,
                            description=recommendation.description,
                            rationale=recommendation.rationale,
                            suggested_actions=recommendation.suggested_actions,
                            source_incident_ids=recommendation.source_incident_ids,
                            incident_count=recommendation.incident_count,
                            project_id=recommendation.project_id,
                            aspect=recommendation.aspect,
                            confidence=recommendation.confidence,
                            classification_rule=recommendation.classification_rule,
                            expires_at=recommendation.expires_at,
                            metadata=recommendation.metadata,
                        )
                except (ValueError, AttributeError):
                    pass
            return recommendation

        # Map approval action to status
        new_status = (
            RecommendationStatus.APPROVED.value
            if approval.action == "approved"
            else RecommendationStatus.DISMISSED.value
        )

        # Return new recommendation with updated status
        return Recommendation(
            recommendation_id=recommendation.recommendation_id,
            created_at=recommendation.created_at,
            recommendation_type=recommendation.recommendation_type,
            severity=recommendation.severity,
            approval_required=recommendation.approval_required,
            status=new_status,
            title=recommendation.title,
            description=recommendation.description,
            rationale=recommendation.rationale,
            suggested_actions=recommendation.suggested_actions,
            source_incident_ids=recommendation.source_incident_ids,
            incident_count=recommendation.incident_count,
            project_id=recommendation.project_id,
            aspect=recommendation.aspect,
            confidence=recommendation.confidence,
            classification_rule=recommendation.classification_rule,
            expires_at=recommendation.expires_at,
            metadata=recommendation.metadata,
        )

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log to audit trail."""
        try:
            RECOMMENDATIONS_AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "action": action,
                **details,
            }
            with open(RECOMMENDATIONS_AUDIT_FILE, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Recommendation audit log failed: {e}")


# -----------------------------------------------------------------------------
# Module-Level Functions
# -----------------------------------------------------------------------------
_store: Optional[RecommendationStore] = None


def get_recommendation_store() -> RecommendationStore:
    """Get the global recommendation store instance."""
    global _store
    if _store is None:
        _store = RecommendationStore()
    return _store


def persist_recommendations(recommendations: List[Recommendation]) -> int:
    """Persist recommendations to storage."""
    return get_recommendation_store().persist(recommendations)


def read_recommendations(
    since: Optional[datetime] = None,
    recommendation_type: Optional[RecommendationType] = None,
    severity: Optional[RecommendationSeverity] = None,
    limit: int = 1000,
) -> List[Recommendation]:
    """Read recommendations from storage."""
    return get_recommendation_store().read_recommendations(
        since=since,
        recommendation_type=recommendation_type,
        severity=severity,
        limit=limit,
    )


def read_recent_recommendations(hours: int = 24, limit: int = 50) -> List[Recommendation]:
    """Read recent recommendations."""
    return get_recommendation_store().read_recent(hours=hours, limit=limit)


def read_pending_recommendations(limit: int = 50) -> List[Recommendation]:
    """Read pending recommendations."""
    return get_recommendation_store().read_pending(limit=limit)


def get_recommendation_by_id(recommendation_id: str) -> Optional[Recommendation]:
    """Get recommendation by ID."""
    return get_recommendation_store().get_by_id(recommendation_id)


def get_recommendation_summary(since_hours: int = 24) -> RecommendationSummary:
    """Get recommendation summary."""
    since = datetime.utcnow() - timedelta(hours=since_hours)
    return get_recommendation_store().get_summary(since=since)


def approve_recommendation(
    recommendation_id: str,
    user_id: str,
    reason: Optional[str] = None,
) -> Optional[ApprovalRecord]:
    """Approve a recommendation."""
    return get_recommendation_store().approve(recommendation_id, user_id, reason)


def dismiss_recommendation(
    recommendation_id: str,
    user_id: str,
    reason: Optional[str] = None,
) -> Optional[ApprovalRecord]:
    """Dismiss a recommendation."""
    return get_recommendation_store().dismiss(recommendation_id, user_id, reason)


logger.info("Recommendation Store module loaded (Phase 17C - APPEND-ONLY)")
