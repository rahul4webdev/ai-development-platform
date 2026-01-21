"""
Phase 18B: Approval Store

Append-only persistence for approval requests and decisions.

CRITICAL CONSTRAINTS (NON-NEGOTIABLE):
- APPEND-ONLY: Records are NEVER modified or deleted
- IMMUTABLE: Once written, records cannot change
- DETERMINISTIC: Same query always returns same results
- NO SIDE EFFECTS: Reading does not modify state
- FSYNC: All writes are fsync'd for durability
- AUDIT TRAIL: Every operation is logged

This store provides:
1. Approval request persistence
2. Approval decision history
3. Read-only queries for orchestrator

This store does NOT:
1. Execute any actions
2. Send notifications
3. Modify external state
4. Make decisions (that's the orchestrator's job)
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
STORE_VERSION = "18B.1.0"

# Storage paths
STORAGE_DIR = Path(os.getenv("APPROVAL_STORAGE_DIR", "data/approvals"))
REQUESTS_FILE = STORAGE_DIR / "approval_requests.jsonl"
DECISIONS_FILE = STORAGE_DIR / "approval_decisions.jsonl"
APPROVER_ACTIONS_FILE = STORAGE_DIR / "approver_actions.jsonl"

# Default expiration (24 hours)
DEFAULT_EXPIRY_HOURS = 24


# -----------------------------------------------------------------------------
# Request Status Enum (LOCKED)
# -----------------------------------------------------------------------------
class RequestStatus(str, Enum):
    """
    Status of an approval request.

    This enum is LOCKED - do not add values without explicit approval.
    """
    OPEN = "open"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


# -----------------------------------------------------------------------------
# Approver Action Enum (LOCKED)
# -----------------------------------------------------------------------------
class ApproverAction(str, Enum):
    """
    Actions an approver can take.

    This enum is LOCKED - do not add values without explicit approval.
    """
    APPROVE = "approve"
    DENY = "deny"
    ABSTAIN = "abstain"


# -----------------------------------------------------------------------------
# Approval Request Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ApprovalRequestRecord:
    """
    Immutable record of an approval request.

    Once created, this record CANNOT be modified.
    Status changes create NEW decision records.
    """
    request_id: str
    created_at: str  # ISO format
    expires_at: str  # ISO format
    requester_id: str
    requester_role: str
    recommendation_id: str
    project_id: Optional[str]
    approval_type: str  # ApprovalType value
    required_approver_count: int
    request_reason: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.required_approver_count < 0:
            raise ValueError(f"required_approver_count cannot be negative: {self.required_approver_count}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApprovalRequestRecord":
        return cls(
            request_id=data["request_id"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            requester_id=data["requester_id"],
            requester_role=data["requester_role"],
            recommendation_id=data["recommendation_id"],
            project_id=data.get("project_id"),
            approval_type=data["approval_type"],
            required_approver_count=data["required_approver_count"],
            request_reason=data.get("request_reason"),
            metadata=data.get("metadata", {}),
        )


# -----------------------------------------------------------------------------
# Approver Action Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class ApproverActionRecord:
    """
    Immutable record of an approver's action.

    Append-only. Each action creates a new record.
    """
    action_id: str
    request_id: str
    approver_id: str
    approver_role: str
    action: str  # ApproverAction value
    action_timestamp: str  # ISO format
    reason: Optional[str]

    def __post_init__(self):
        if self.action not in [a.value for a in ApproverAction]:
            raise ValueError(f"Invalid action: {self.action}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApproverActionRecord":
        return cls(
            action_id=data["action_id"],
            request_id=data["request_id"],
            approver_id=data["approver_id"],
            approver_role=data["approver_role"],
            action=data["action"],
            action_timestamp=data["action_timestamp"],
            reason=data.get("reason"),
        )


# -----------------------------------------------------------------------------
# Decision Record (Frozen - Immutable)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class DecisionRecord:
    """
    Immutable record of a final decision on a request.

    This records the outcome of the approval process.
    Append-only - decisions are never modified.
    """
    decision_id: str
    request_id: str
    decision_timestamp: str  # ISO format
    status: str  # RequestStatus value (approved/denied/expired/cancelled)
    decided_by: Optional[str]  # User who triggered final decision, or "system" for expiry
    approver_count: int
    required_count: int
    reason: Optional[str]

    def __post_init__(self):
        if self.status not in [s.value for s in RequestStatus]:
            raise ValueError(f"Invalid status: {self.status}")
        if self.approver_count < 0:
            raise ValueError(f"approver_count cannot be negative: {self.approver_count}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionRecord":
        return cls(
            decision_id=data["decision_id"],
            request_id=data["request_id"],
            decision_timestamp=data["decision_timestamp"],
            status=data["status"],
            decided_by=data.get("decided_by"),
            approver_count=data["approver_count"],
            required_count=data["required_count"],
            reason=data.get("reason"),
        )


# -----------------------------------------------------------------------------
# Approval Store (Append-Only Persistence)
# -----------------------------------------------------------------------------
class ApprovalStore:
    """
    Phase 18B: Approval Store.

    Append-only persistence for approval requests and decisions.

    CRITICAL CONSTRAINTS:
    - APPEND-ONLY: Records are NEVER modified or deleted
    - IMMUTABLE: Once written, records cannot change
    - FSYNC: All writes are fsync'd for durability
    - DETERMINISTIC: Same query always returns same results
    """

    def __init__(
        self,
        requests_file: Optional[Path] = None,
        decisions_file: Optional[Path] = None,
        actions_file: Optional[Path] = None,
    ):
        """
        Initialize store.

        Args:
            requests_file: Path to requests file (optional, for testing)
            decisions_file: Path to decisions file (optional, for testing)
            actions_file: Path to approver actions file (optional, for testing)
        """
        self._requests_file = requests_file or REQUESTS_FILE
        self._decisions_file = decisions_file or DECISIONS_FILE
        self._actions_file = actions_file or APPROVER_ACTIONS_FILE
        self._version = STORE_VERSION

    # -------------------------------------------------------------------------
    # Write Operations (Append-Only)
    # -------------------------------------------------------------------------

    def create_request(
        self,
        requester_id: str,
        requester_role: str,
        recommendation_id: str,
        approval_type: str,
        required_approver_count: int,
        project_id: Optional[str] = None,
        request_reason: Optional[str] = None,
        expiry_hours: int = DEFAULT_EXPIRY_HOURS,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ApprovalRequestRecord:
        """
        Create a new approval request.

        APPEND-ONLY: Creates a new record, never modifies existing.

        Args:
            requester_id: ID of the requester
            requester_role: Role of the requester
            recommendation_id: ID of the recommendation being approved
            approval_type: Type of approval required
            required_approver_count: Number of approvers needed
            project_id: Optional project ID
            request_reason: Optional reason for request
            expiry_hours: Hours until request expires
            metadata: Optional additional metadata

        Returns:
            Created ApprovalRequestRecord
        """
        timestamp = datetime.utcnow()
        request_id = f"req-{timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

        record = ApprovalRequestRecord(
            request_id=request_id,
            created_at=timestamp.isoformat(),
            expires_at=(timestamp + timedelta(hours=expiry_hours)).isoformat(),
            requester_id=requester_id,
            requester_role=requester_role,
            recommendation_id=recommendation_id,
            project_id=project_id,
            approval_type=approval_type,
            required_approver_count=required_approver_count,
            request_reason=request_reason,
            metadata=metadata or {},
        )

        self._append_record(self._requests_file, record.to_dict())
        return record

    def record_approver_action(
        self,
        request_id: str,
        approver_id: str,
        approver_role: str,
        action: str,
        reason: Optional[str] = None,
    ) -> ApproverActionRecord:
        """
        Record an approver's action.

        APPEND-ONLY: Creates a new record, never modifies existing.

        Args:
            request_id: ID of the approval request
            approver_id: ID of the approver
            approver_role: Role of the approver
            action: Action taken (approve/deny/abstain)
            reason: Optional reason for action

        Returns:
            Created ApproverActionRecord
        """
        timestamp = datetime.utcnow()
        action_id = f"act-{timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

        record = ApproverActionRecord(
            action_id=action_id,
            request_id=request_id,
            approver_id=approver_id,
            approver_role=approver_role,
            action=action,
            action_timestamp=timestamp.isoformat(),
            reason=reason,
        )

        self._append_record(self._actions_file, record.to_dict())
        return record

    def record_decision(
        self,
        request_id: str,
        status: str,
        approver_count: int,
        required_count: int,
        decided_by: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> DecisionRecord:
        """
        Record a final decision on a request.

        APPEND-ONLY: Creates a new record, never modifies existing.

        Args:
            request_id: ID of the approval request
            status: Final status (approved/denied/expired/cancelled)
            approver_count: Number of approvers who approved
            required_count: Number of approvers required
            decided_by: Who/what triggered the decision
            reason: Optional reason for decision

        Returns:
            Created DecisionRecord
        """
        timestamp = datetime.utcnow()
        decision_id = f"dec-{timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"

        record = DecisionRecord(
            decision_id=decision_id,
            request_id=request_id,
            decision_timestamp=timestamp.isoformat(),
            status=status,
            decided_by=decided_by,
            approver_count=approver_count,
            required_count=required_count,
            reason=reason,
        )

        self._append_record(self._decisions_file, record.to_dict())
        return record

    # -------------------------------------------------------------------------
    # Read Operations (Read-Only)
    # -------------------------------------------------------------------------

    def get_request(self, request_id: str) -> Optional[ApprovalRequestRecord]:
        """
        Get an approval request by ID.

        READ-ONLY: Does not modify state.

        Args:
            request_id: ID of the request

        Returns:
            ApprovalRequestRecord if found, None otherwise
        """
        for record in self._read_records(self._requests_file):
            if record.get("request_id") == request_id:
                return ApprovalRequestRecord.from_dict(record)
        return None

    def get_requests_for_recommendation(
        self,
        recommendation_id: str,
    ) -> List[ApprovalRequestRecord]:
        """
        Get all requests for a recommendation.

        READ-ONLY: Does not modify state.

        Args:
            recommendation_id: ID of the recommendation

        Returns:
            List of ApprovalRequestRecords
        """
        results = []
        for record in self._read_records(self._requests_file):
            if record.get("recommendation_id") == recommendation_id:
                results.append(ApprovalRequestRecord.from_dict(record))
        return results

    def get_open_requests(
        self,
        project_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ApprovalRequestRecord]:
        """
        Get open (non-decided) requests.

        READ-ONLY: Does not modify state.

        Args:
            project_id: Optional filter by project
            limit: Maximum number of results

        Returns:
            List of open ApprovalRequestRecords
        """
        # Get all decided request IDs
        decided_ids = set()
        for record in self._read_records(self._decisions_file):
            decided_ids.add(record.get("request_id"))

        # Find open requests
        results = []
        for record in self._read_records(self._requests_file):
            if record.get("request_id") in decided_ids:
                continue

            if project_id and record.get("project_id") != project_id:
                continue

            results.append(ApprovalRequestRecord.from_dict(record))

            if len(results) >= limit:
                break

        return results

    def get_approver_actions(self, request_id: str) -> List[ApproverActionRecord]:
        """
        Get all approver actions for a request.

        READ-ONLY: Does not modify state.

        Args:
            request_id: ID of the request

        Returns:
            List of ApproverActionRecords
        """
        results = []
        for record in self._read_records(self._actions_file):
            if record.get("request_id") == request_id:
                results.append(ApproverActionRecord.from_dict(record))
        return results

    def get_approval_count(self, request_id: str) -> int:
        """
        Count approvals for a request.

        READ-ONLY: Does not modify state.

        Args:
            request_id: ID of the request

        Returns:
            Number of approvals
        """
        count = 0
        seen_approvers = set()

        for record in self._read_records(self._actions_file):
            if record.get("request_id") != request_id:
                continue

            approver_id = record.get("approver_id")
            if approver_id in seen_approvers:
                continue  # Count each approver only once

            if record.get("action") == ApproverAction.APPROVE.value:
                count += 1
                seen_approvers.add(approver_id)

        return count

    def get_decision(self, request_id: str) -> Optional[DecisionRecord]:
        """
        Get the decision for a request.

        READ-ONLY: Does not modify state.
        Returns the most recent decision if multiple exist.

        Args:
            request_id: ID of the request

        Returns:
            DecisionRecord if found, None otherwise
        """
        latest = None
        for record in self._read_records(self._decisions_file):
            if record.get("request_id") == request_id:
                latest = DecisionRecord.from_dict(record)
        return latest

    def is_request_decided(self, request_id: str) -> bool:
        """
        Check if a request has been decided.

        READ-ONLY: Does not modify state.

        Args:
            request_id: ID of the request

        Returns:
            True if decided, False otherwise
        """
        for record in self._read_records(self._decisions_file):
            if record.get("request_id") == request_id:
                return True
        return False

    def has_approver_acted(self, request_id: str, approver_id: str) -> bool:
        """
        Check if an approver has already acted on a request.

        READ-ONLY: Does not modify state.

        Args:
            request_id: ID of the request
            approver_id: ID of the approver

        Returns:
            True if approver has acted, False otherwise
        """
        for record in self._read_records(self._actions_file):
            if (record.get("request_id") == request_id and
                record.get("approver_id") == approver_id):
                return True
        return False

    # -------------------------------------------------------------------------
    # Summary/Aggregation (Read-Only)
    # -------------------------------------------------------------------------

    def get_summary(
        self,
        project_id: Optional[str] = None,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get a summary of approval activity.

        READ-ONLY: Does not modify state.

        Args:
            project_id: Optional filter by project
            since_hours: Hours to look back

        Returns:
            Summary dictionary
        """
        cutoff = (datetime.utcnow() - timedelta(hours=since_hours)).isoformat()

        total_requests = 0
        open_requests = 0
        approved_count = 0
        denied_count = 0
        expired_count = 0
        by_approval_type: Dict[str, int] = {}

        # Get decided request IDs
        decided_ids: Dict[str, str] = {}  # request_id -> status
        for record in self._read_records(self._decisions_file):
            decided_ids[record.get("request_id")] = record.get("status")
            status = record.get("status")
            if status == RequestStatus.APPROVED.value:
                approved_count += 1
            elif status == RequestStatus.DENIED.value:
                denied_count += 1
            elif status == RequestStatus.EXPIRED.value:
                expired_count += 1

        # Count requests
        for record in self._read_records(self._requests_file):
            if record.get("created_at", "") < cutoff:
                continue

            if project_id and record.get("project_id") != project_id:
                continue

            total_requests += 1

            approval_type = record.get("approval_type", "unknown")
            by_approval_type[approval_type] = by_approval_type.get(approval_type, 0) + 1

            if record.get("request_id") not in decided_ids:
                open_requests += 1

        return {
            "generated_at": datetime.utcnow().isoformat(),
            "since_hours": since_hours,
            "project_id": project_id,
            "total_requests": total_requests,
            "open_requests": open_requests,
            "approved_count": approved_count,
            "denied_count": denied_count,
            "expired_count": expired_count,
            "by_approval_type": by_approval_type,
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
_store: Optional[ApprovalStore] = None


def get_approval_store(
    requests_file: Optional[Path] = None,
    decisions_file: Optional[Path] = None,
    actions_file: Optional[Path] = None,
) -> ApprovalStore:
    """Get the approval store singleton."""
    global _store
    if _store is None:
        _store = ApprovalStore(
            requests_file=requests_file,
            decisions_file=decisions_file,
            actions_file=actions_file,
        )
    return _store


def create_approval_request(
    requester_id: str,
    requester_role: str,
    recommendation_id: str,
    approval_type: str,
    required_approver_count: int = 1,
    project_id: Optional[str] = None,
    request_reason: Optional[str] = None,
) -> ApprovalRequestRecord:
    """
    Create a new approval request.

    Convenience function using singleton store.
    """
    store = get_approval_store()
    return store.create_request(
        requester_id=requester_id,
        requester_role=requester_role,
        recommendation_id=recommendation_id,
        approval_type=approval_type,
        required_approver_count=required_approver_count,
        project_id=project_id,
        request_reason=request_reason,
    )


def record_approval(
    request_id: str,
    approver_id: str,
    approver_role: str,
    reason: Optional[str] = None,
) -> ApproverActionRecord:
    """
    Record an approval action.

    Convenience function using singleton store.
    """
    store = get_approval_store()
    return store.record_approver_action(
        request_id=request_id,
        approver_id=approver_id,
        approver_role=approver_role,
        action=ApproverAction.APPROVE.value,
        reason=reason,
    )


def record_denial(
    request_id: str,
    approver_id: str,
    approver_role: str,
    reason: Optional[str] = None,
) -> ApproverActionRecord:
    """
    Record a denial action.

    Convenience function using singleton store.
    """
    store = get_approval_store()
    return store.record_approver_action(
        request_id=request_id,
        approver_id=approver_id,
        approver_role=approver_role,
        action=ApproverAction.DENY.value,
        reason=reason,
    )


def get_approval_summary(
    project_id: Optional[str] = None,
    since_hours: int = 24,
) -> Dict[str, Any]:
    """
    Get approval summary.

    Convenience function using singleton store.
    """
    store = get_approval_store()
    return store.get_summary(project_id=project_id, since_hours=since_hours)
