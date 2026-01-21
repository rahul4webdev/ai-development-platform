"""
Phase 16F: Intent Baseline Manager

Immutable baseline storage for project intent.

The Intent Baseline is the "contract" between humans and Claude about
what a project IS and what it is supposed to DO.

HARD CONSTRAINTS:
- Baseline is IMMUTABLE once created
- Baseline can only be replaced via explicit REBASELINE approval
- All baselines are versioned and audited
- Baseline changes require human confirmation
- NO auto-rebaselining allowed

LIFECYCLE:
1. Project created -> Initial baseline captured
2. Project evolves -> Current intent may drift
3. Drift detected -> Block if severe, warn otherwise
4. Rebaseline requested -> Requires explicit approval
5. Approved -> New baseline replaces old (old preserved in history)
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("intent_baseline")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASELINE_DIR = Path(os.getenv(
    "INTENT_BASELINE_DIR",
    "/home/aitesting.mybd.in/jobs/intent_baselines"
))

# Fallback for local development
if not BASELINE_DIR.exists():
    BASELINE_DIR = Path("/tmp/intent_baselines")


# -----------------------------------------------------------------------------
# Baseline Status
# -----------------------------------------------------------------------------
class BaselineStatus(str, Enum):
    """Status of an intent baseline."""
    ACTIVE = "active"           # Currently enforced baseline
    SUPERSEDED = "superseded"   # Replaced by newer baseline
    PENDING = "pending"         # Awaiting approval (for rebaseline)
    REJECTED = "rejected"       # Rebaseline was rejected


class RebaselineReason(str, Enum):
    """Valid reasons for rebaselining."""
    SCOPE_EXPANSION = "scope_expansion"         # Intentional scope increase
    PIVOT = "pivot"                             # Project direction change
    ARCHITECTURE_UPGRADE = "architecture_upgrade"  # Planned architecture change
    REQUIREMENTS_CLARIFICATION = "requirements_clarification"  # Initial intent was unclear
    MERGER = "merger"                           # Merging with another project
    SPLIT = "split"                             # Splitting from larger project
    CORRECTION = "correction"                   # Baseline was incorrect
    OTHER = "other"                             # Other (requires explanation)


# -----------------------------------------------------------------------------
# Intent Baseline Model
# -----------------------------------------------------------------------------
@dataclass
class IntentBaseline:
    """
    Immutable baseline of project intent.

    Once created, a baseline CANNOT be modified.
    It can only be superseded by a new baseline through explicit approval.
    """
    baseline_id: str
    project_id: str
    project_name: str
    version: int  # Baseline version (1, 2, 3...)
    status: str  # BaselineStatus value

    # The actual intent snapshot
    normalized_intent: Dict[str, Any]  # NormalizedIntent as dict
    fingerprint: str  # Fingerprint at baseline time

    # Creation metadata
    created_at: str
    created_by: str
    creation_reason: str  # "initial" or RebaselineReason value

    # Supersession info (if superseded)
    superseded_at: Optional[str] = None
    superseded_by: Optional[str] = None  # baseline_id of replacement
    supersession_approved_by: Optional[str] = None

    # Optional approval info (for rebaselines)
    approved_at: Optional[str] = None
    approved_by: Optional[str] = None
    approval_justification: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentBaseline":
        """Create from dictionary."""
        return cls(**data)

    def is_active(self) -> bool:
        """Check if this baseline is currently active."""
        return self.status == BaselineStatus.ACTIVE.value


@dataclass
class RebaselineRequest:
    """Request to create a new baseline."""
    request_id: str
    project_id: str
    project_name: str
    current_baseline_id: str
    proposed_intent: Dict[str, Any]  # New NormalizedIntent as dict
    proposed_fingerprint: str
    reason: str  # RebaselineReason value
    justification: str  # Human explanation
    requested_by: str
    requested_at: str
    status: str = "pending"  # pending, approved, rejected
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RebaselineRequest":
        """Create from dictionary."""
        return cls(**data)


# -----------------------------------------------------------------------------
# Baseline Manager
# -----------------------------------------------------------------------------
class IntentBaselineManager:
    """
    Manages intent baselines for projects.

    SECURITY-CRITICAL: This class enforces baseline immutability.
    All operations are logged to an append-only audit trail.
    """

    def __init__(self, baseline_dir: Optional[Path] = None):
        self._baseline_dir = baseline_dir or BASELINE_DIR
        self._baselines_file = self._baseline_dir / "baselines.json"
        self._requests_file = self._baseline_dir / "rebaseline_requests.json"
        self._audit_file = self._baseline_dir / "baseline_audit.log"

        # Ensure directory exists
        self._baseline_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        self._baselines: Dict[str, IntentBaseline] = {}  # baseline_id -> baseline
        self._project_baselines: Dict[str, List[str]] = {}  # project_id -> [baseline_ids]
        self._requests: Dict[str, RebaselineRequest] = {}  # request_id -> request
        self._load_data()

    def _load_data(self) -> None:
        """Load baselines and requests from storage."""
        # Load baselines
        if self._baselines_file.exists():
            try:
                with open(self._baselines_file) as f:
                    data = json.load(f)
                    for item in data.get("baselines", []):
                        baseline = IntentBaseline.from_dict(item)
                        self._baselines[baseline.baseline_id] = baseline
                        if baseline.project_id not in self._project_baselines:
                            self._project_baselines[baseline.project_id] = []
                        self._project_baselines[baseline.project_id].append(baseline.baseline_id)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load baselines: {e}")

        # Load requests
        if self._requests_file.exists():
            try:
                with open(self._requests_file) as f:
                    data = json.load(f)
                    for item in data.get("requests", []):
                        request = RebaselineRequest.from_dict(item)
                        self._requests[request.request_id] = request
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load rebaseline requests: {e}")

    def _save_baselines(self) -> None:
        """Persist baselines to storage."""
        data = {
            "baselines": [b.to_dict() for b in self._baselines.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        temp_file = self._baselines_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self._baselines_file)

    def _save_requests(self) -> None:
        """Persist rebaseline requests to storage."""
        data = {
            "requests": [r.to_dict() for r in self._requests.values()],
            "updated_at": datetime.utcnow().isoformat(),
        }
        temp_file = self._requests_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(data, f, indent=2)
        temp_file.replace(self._requests_file)

    def _log_audit(self, action: str, details: Dict[str, Any]) -> None:
        """Append to audit log (immutable)."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            **details,
        }
        with open(self._audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # -------------------------------------------------------------------------
    # Baseline Creation
    # -------------------------------------------------------------------------

    def create_initial_baseline(
        self,
        project_id: str,
        project_name: str,
        normalized_intent: Dict[str, Any],
        fingerprint: str,
        created_by: str,
    ) -> Tuple[bool, str, Optional[IntentBaseline]]:
        """
        Create the initial baseline for a new project.

        This is called ONCE when a project is first created.
        Returns: (success, message, baseline)
        """
        # Check if project already has a baseline
        existing = self.get_active_baseline(project_id)
        if existing:
            return False, f"Project '{project_name}' already has an active baseline", None

        baseline_id = f"baseline-{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()

        baseline = IntentBaseline(
            baseline_id=baseline_id,
            project_id=project_id,
            project_name=project_name,
            version=1,
            status=BaselineStatus.ACTIVE.value,
            normalized_intent=normalized_intent,
            fingerprint=fingerprint,
            created_at=now,
            created_by=created_by,
            creation_reason="initial",
        )

        # Save
        self._baselines[baseline_id] = baseline
        if project_id not in self._project_baselines:
            self._project_baselines[project_id] = []
        self._project_baselines[project_id].append(baseline_id)
        self._save_baselines()

        # Audit
        self._log_audit("BASELINE_CREATED", {
            "baseline_id": baseline_id,
            "project_id": project_id,
            "project_name": project_name,
            "version": 1,
            "created_by": created_by,
            "fingerprint": fingerprint[:16] + "...",
        })

        logger.info(f"Created initial baseline for {project_name}: {baseline_id}")
        return True, f"Initial baseline created: {baseline_id}", baseline

    # -------------------------------------------------------------------------
    # Baseline Retrieval
    # -------------------------------------------------------------------------

    def get_active_baseline(self, project_id: str) -> Optional[IntentBaseline]:
        """Get the currently active baseline for a project."""
        baseline_ids = self._project_baselines.get(project_id, [])
        for bid in reversed(baseline_ids):  # Most recent first
            baseline = self._baselines.get(bid)
            if baseline and baseline.is_active():
                return baseline
        return None

    def get_baseline_by_id(self, baseline_id: str) -> Optional[IntentBaseline]:
        """Get a specific baseline by ID."""
        return self._baselines.get(baseline_id)

    def get_baseline_history(self, project_id: str) -> List[IntentBaseline]:
        """Get all baselines for a project (history)."""
        baseline_ids = self._project_baselines.get(project_id, [])
        return [self._baselines[bid] for bid in baseline_ids if bid in self._baselines]

    # -------------------------------------------------------------------------
    # Rebaseline Workflow
    # -------------------------------------------------------------------------

    def request_rebaseline(
        self,
        project_id: str,
        project_name: str,
        proposed_intent: Dict[str, Any],
        proposed_fingerprint: str,
        reason: str,
        justification: str,
        requested_by: str,
    ) -> Tuple[bool, str, Optional[RebaselineRequest]]:
        """
        Request a new baseline for a project.

        This creates a PENDING request that must be approved.
        Returns: (success, message, request)
        """
        # Validate reason
        try:
            RebaselineReason(reason)
        except ValueError:
            return False, f"Invalid rebaseline reason: {reason}", None

        # Get current baseline
        current = self.get_active_baseline(project_id)
        if not current:
            return False, f"No active baseline found for project {project_id}", None

        # Check for existing pending request
        for req in self._requests.values():
            if req.project_id == project_id and req.status == "pending":
                return False, f"Pending rebaseline request already exists: {req.request_id}", None

        request_id = f"rebase-{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()

        request = RebaselineRequest(
            request_id=request_id,
            project_id=project_id,
            project_name=project_name,
            current_baseline_id=current.baseline_id,
            proposed_intent=proposed_intent,
            proposed_fingerprint=proposed_fingerprint,
            reason=reason,
            justification=justification,
            requested_by=requested_by,
            requested_at=now,
        )

        # Save
        self._requests[request_id] = request
        self._save_requests()

        # Audit
        self._log_audit("REBASELINE_REQUESTED", {
            "request_id": request_id,
            "project_id": project_id,
            "project_name": project_name,
            "current_baseline": current.baseline_id,
            "reason": reason,
            "requested_by": requested_by,
        })

        logger.info(f"Rebaseline requested for {project_name}: {request_id}")
        return True, f"Rebaseline request created: {request_id}", request

    def approve_rebaseline(
        self,
        request_id: str,
        approved_by: str,
        notes: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[IntentBaseline]]:
        """
        Approve a rebaseline request and create new baseline.

        Returns: (success, message, new_baseline)
        """
        request = self._requests.get(request_id)
        if not request:
            return False, f"Rebaseline request not found: {request_id}", None

        if request.status != "pending":
            return False, f"Request is not pending (status: {request.status})", None

        # Self-approval check (optional - can be enforced)
        # if request.requested_by == approved_by:
        #     return False, "Cannot approve your own rebaseline request", None

        # Get current baseline
        current = self.get_baseline_by_id(request.current_baseline_id)
        if not current:
            return False, f"Current baseline not found: {request.current_baseline_id}", None

        now = datetime.utcnow().isoformat()

        # Create new baseline
        new_baseline_id = f"baseline-{uuid.uuid4().hex[:12]}"
        new_version = current.version + 1

        new_baseline = IntentBaseline(
            baseline_id=new_baseline_id,
            project_id=request.project_id,
            project_name=request.project_name,
            version=new_version,
            status=BaselineStatus.ACTIVE.value,
            normalized_intent=request.proposed_intent,
            fingerprint=request.proposed_fingerprint,
            created_at=now,
            created_by=request.requested_by,
            creation_reason=request.reason,
            approved_at=now,
            approved_by=approved_by,
            approval_justification=request.justification,
        )

        # Supersede old baseline
        current.status = BaselineStatus.SUPERSEDED.value
        current.superseded_at = now
        current.superseded_by = new_baseline_id
        current.supersession_approved_by = approved_by

        # Update request
        request.status = "approved"
        request.reviewed_by = approved_by
        request.reviewed_at = now
        request.review_notes = notes

        # Save all
        self._baselines[new_baseline_id] = new_baseline
        self._project_baselines[request.project_id].append(new_baseline_id)
        self._save_baselines()
        self._save_requests()

        # Audit
        self._log_audit("REBASELINE_APPROVED", {
            "request_id": request_id,
            "old_baseline": current.baseline_id,
            "new_baseline": new_baseline_id,
            "new_version": new_version,
            "approved_by": approved_by,
            "project_id": request.project_id,
        })

        logger.info(f"Rebaseline approved: {request_id} -> {new_baseline_id}")
        return True, f"Rebaseline approved. New baseline: {new_baseline_id}", new_baseline

    def reject_rebaseline(
        self,
        request_id: str,
        rejected_by: str,
        notes: str,
    ) -> Tuple[bool, str]:
        """
        Reject a rebaseline request.

        Returns: (success, message)
        """
        request = self._requests.get(request_id)
        if not request:
            return False, f"Rebaseline request not found: {request_id}"

        if request.status != "pending":
            return False, f"Request is not pending (status: {request.status})"

        now = datetime.utcnow().isoformat()

        # Update request
        request.status = "rejected"
        request.reviewed_by = rejected_by
        request.reviewed_at = now
        request.review_notes = notes

        self._save_requests()

        # Audit
        self._log_audit("REBASELINE_REJECTED", {
            "request_id": request_id,
            "project_id": request.project_id,
            "rejected_by": rejected_by,
            "notes": notes,
        })

        logger.info(f"Rebaseline rejected: {request_id}")
        return True, f"Rebaseline request rejected: {request_id}"

    def get_pending_requests(self, project_id: Optional[str] = None) -> List[RebaselineRequest]:
        """Get pending rebaseline requests."""
        pending = [r for r in self._requests.values() if r.status == "pending"]
        if project_id:
            pending = [r for r in pending if r.project_id == project_id]
        return pending


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------
_manager: Optional[IntentBaselineManager] = None


def get_baseline_manager() -> IntentBaselineManager:
    """Get the global baseline manager instance."""
    global _manager
    if _manager is None:
        _manager = IntentBaselineManager()
    return _manager


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
def create_initial_baseline(
    project_id: str,
    project_name: str,
    normalized_intent: Dict[str, Any],
    fingerprint: str,
    created_by: str,
) -> Tuple[bool, str, Optional[IntentBaseline]]:
    """Create initial baseline for a new project."""
    return get_baseline_manager().create_initial_baseline(
        project_id=project_id,
        project_name=project_name,
        normalized_intent=normalized_intent,
        fingerprint=fingerprint,
        created_by=created_by,
    )


def get_active_baseline(project_id: str) -> Optional[IntentBaseline]:
    """Get the active baseline for a project."""
    return get_baseline_manager().get_active_baseline(project_id)


def request_rebaseline(
    project_id: str,
    project_name: str,
    proposed_intent: Dict[str, Any],
    proposed_fingerprint: str,
    reason: str,
    justification: str,
    requested_by: str,
) -> Tuple[bool, str, Optional[RebaselineRequest]]:
    """Request a new baseline."""
    return get_baseline_manager().request_rebaseline(
        project_id=project_id,
        project_name=project_name,
        proposed_intent=proposed_intent,
        proposed_fingerprint=proposed_fingerprint,
        reason=reason,
        justification=justification,
        requested_by=requested_by,
    )


def approve_rebaseline(
    request_id: str,
    approved_by: str,
    notes: Optional[str] = None,
) -> Tuple[bool, str, Optional[IntentBaseline]]:
    """Approve a rebaseline request."""
    return get_baseline_manager().approve_rebaseline(
        request_id=request_id,
        approved_by=approved_by,
        notes=notes,
    )
