"""
Phase 12: Autonomous Multi-Aspect Project Orchestration

This module provides:
- Multi-aspect project model (Core/Backend/Frontend)
- Internal Project Contract (IPC)
- Aspect lifecycle management
- Approval workflow with structured feedback
- CI trigger logic
- Notification system
- Dashboard API models

Core Principles:
- Users speak naturally; structure is internal
- No schemas required from humans
- Claude normalizes all inputs
- Multi-aspect projects are first-class citizens
- Autonomous execution with explicit human gates
"""

import logging
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("phase12")

# -----------------------------------------------------------------------------
# Enums - Multi-Aspect Project Model
# -----------------------------------------------------------------------------

class ProjectAspect(str, Enum):
    """Project aspects for decomposition."""
    CORE = "core"           # APIs, shared services, mobile/frontend APIs
    BACKEND = "backend"     # Admin panels, management tools, internal dashboards
    FRONTEND = "frontend"   # Website, web application, mobile application


class AspectPhase(str, Enum):
    """Lifecycle phases for each aspect."""
    NOT_STARTED = "not_started"
    PLANNING = "planning"
    DEVELOPMENT = "development"
    UNIT_TESTING = "unit_testing"
    INTEGRATION = "integration"
    CODE_REVIEW = "code_review"
    CI_RUNNING = "ci_running"
    CI_PASSED = "ci_passed"
    CI_FAILED = "ci_failed"
    READY_FOR_TESTING = "ready_for_testing"
    DEPLOYED_TESTING = "deployed_testing"
    AWAITING_FEEDBACK = "awaiting_feedback"
    BUG_FIXING = "bug_fixing"
    IMPROVEMENTS = "improvements"
    READY_FOR_PRODUCTION = "ready_for_production"
    DEPLOYED_PRODUCTION = "deployed_production"
    COMPLETED = "completed"


class FeedbackType(str, Enum):
    """Types of feedback from testers."""
    APPROVE = "approve"
    BUG = "bug"
    IMPROVEMENTS = "improvements"
    REJECT = "reject"


class CITriggerReason(str, Enum):
    """Valid reasons for CI trigger."""
    PHASE_COMPLETE = "phase_complete"
    MODULE_COMPLETE = "module_complete"
    UNIT_TESTS_PASS = "unit_tests_pass"
    BUG_FIX_COMPLETE = "bug_fix_complete"
    PRE_TESTING_DEPLOY = "pre_testing_deploy"
    PRE_PRODUCTION_DEPLOY = "pre_production_deploy"


class NotificationType(str, Enum):
    """Types of notifications."""
    PHASE_TRANSITION = "phase_transition"
    TESTING_READY = "testing_ready"
    FEEDBACK_REQUIRED = "feedback_required"
    CI_RESULT = "ci_result"
    DEPLOYMENT_COMPLETE = "deployment_complete"
    ERROR = "error"
    APPROVAL_REQUIRED = "approval_required"


# -----------------------------------------------------------------------------
# Tech Stack Model
# -----------------------------------------------------------------------------

class TechStackConfig(BaseModel):
    """Technology stack configuration for a project."""
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    deployment_targets: List[str] = Field(default_factory=list)
    ci_cd_platform: Optional[str] = None
    additional_tools: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Aspect Models
# -----------------------------------------------------------------------------

class AspectConfig(BaseModel):
    """Configuration for a single project aspect."""
    aspect_type: ProjectAspect
    enabled: bool = True
    description: str = ""
    tech_stack: TechStackConfig = Field(default_factory=TechStackConfig)
    deploy_target: Optional[str] = None
    testing_url: Optional[str] = None
    production_url: Optional[str] = None


class AspectState(BaseModel):
    """Current state of a project aspect."""
    aspect_type: ProjectAspect
    current_phase: AspectPhase = AspectPhase.NOT_STARTED
    phase_started_at: Optional[datetime] = None
    last_ci_run_id: Optional[str] = None
    last_ci_status: Optional[str] = None
    last_ci_at: Optional[datetime] = None
    test_coverage_percent: Optional[float] = None
    unit_tests_passed: int = 0
    unit_tests_failed: int = 0
    integration_tests_passed: int = 0
    integration_tests_failed: int = 0
    last_deploy_testing_at: Optional[datetime] = None
    last_deploy_production_at: Optional[datetime] = None
    feedback_pending: bool = False
    feedback_history: List[Dict[str, Any]] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Internal Project Contract (IPC)
# -----------------------------------------------------------------------------

class InternalProjectContract(BaseModel):
    """
    Internal Project Contract (IPC) - Claude-owned structured definition.

    Derived from natural language human input.
    Contains all normalized project information.
    """
    # Identity
    contract_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None

    # Human Input (preserved)
    original_description: str = ""
    original_requirements: List[str] = Field(default_factory=list)
    attached_documents: List[str] = Field(default_factory=list)
    repo_urls: List[str] = Field(default_factory=list)
    reference_urls: List[str] = Field(default_factory=list)

    # Normalized Structure
    tech_stack: TechStackConfig = Field(default_factory=TechStackConfig)

    # Aspects
    aspects: Dict[ProjectAspect, AspectConfig] = Field(default_factory=dict)
    aspect_states: Dict[ProjectAspect, AspectState] = Field(default_factory=dict)

    # Deployment Configuration
    testing_domain: Optional[str] = None
    production_domain: Optional[str] = None

    # Constraints
    constraints: List[str] = Field(default_factory=list)

    # Status
    overall_status: str = "initializing"

    def initialize_aspects(self) -> None:
        """Initialize default aspects if not set."""
        for aspect in ProjectAspect:
            if aspect not in self.aspects:
                self.aspects[aspect] = AspectConfig(
                    aspect_type=aspect,
                    enabled=True,
                    description=f"Auto-generated {aspect.value} aspect"
                )
            if aspect not in self.aspect_states:
                self.aspect_states[aspect] = AspectState(aspect_type=aspect)


# -----------------------------------------------------------------------------
# Feedback Models
# -----------------------------------------------------------------------------

class TestingFeedback(BaseModel):
    """Structured feedback from testing phase."""
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    aspect: ProjectAspect
    feedback_type: FeedbackType
    submitted_by: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    # Required for non-approve feedback
    explanation: Optional[str] = None

    # Optional details
    affected_features: List[str] = Field(default_factory=list)
    reproduction_steps: Optional[str] = None
    expected_behavior: Optional[str] = None
    actual_behavior: Optional[str] = None
    priority: Optional[str] = None
    screenshots: List[str] = Field(default_factory=list)


class ProductionApproval(BaseModel):
    """Production deployment approval."""
    approval_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    aspect: ProjectAspect
    approved: bool
    approved_by: str
    approved_at: datetime = Field(default_factory=datetime.utcnow)
    justification: str
    risk_acknowledged: bool = False
    rollback_plan: Optional[str] = None


# -----------------------------------------------------------------------------
# Notification Models
# -----------------------------------------------------------------------------

class Notification(BaseModel):
    """Notification to be sent to stakeholders."""
    notification_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    notification_type: NotificationType
    project_name: str
    aspect: Optional[ProjectAspect] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Content
    title: str
    summary: str
    details: Dict[str, Any] = Field(default_factory=dict)

    # URLs and actions
    environment_url: Optional[str] = None
    action_required: bool = False
    next_action: Optional[str] = None

    # Test summary (for testing ready notifications)
    test_coverage_summary: Optional[str] = None
    features_completed: List[str] = Field(default_factory=list)
    known_limitations: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# CI Models
# -----------------------------------------------------------------------------

class CITrigger(BaseModel):
    """CI trigger event."""
    trigger_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    aspect: ProjectAspect
    reason: CITriggerReason
    triggered_at: datetime = Field(default_factory=datetime.utcnow)
    triggered_by: str = "system"

    # Context
    phase_completed: Optional[str] = None
    module_completed: Optional[str] = None
    commit_hash: Optional[str] = None

    # Status
    ci_run_id: Optional[str] = None
    status: str = "pending"


# -----------------------------------------------------------------------------
# Dashboard Models
# -----------------------------------------------------------------------------

class AspectProgress(BaseModel):
    """Progress summary for a single aspect."""
    aspect: ProjectAspect
    current_phase: AspectPhase
    phase_progress_percent: float = 0.0
    ci_status: Optional[str] = None
    last_ci_at: Optional[datetime] = None
    test_summary: Optional[str] = None
    deploy_status: Optional[str] = None
    last_deploy_at: Optional[datetime] = None
    health: str = "unknown"  # healthy, warning, error, unknown


class ProjectDashboard(BaseModel):
    """Read-only dashboard view of a project."""
    project_name: str
    overall_status: str
    created_at: datetime
    updated_at: datetime

    # Aspect progress
    aspects: Dict[ProjectAspect, AspectProgress] = Field(default_factory=dict)

    # Aggregated status
    total_phases_completed: int = 0
    total_phases_remaining: int = 0

    # CI summary
    last_ci_result: Optional[str] = None
    last_ci_at: Optional[datetime] = None
    ci_pass_rate: Optional[float] = None

    # Test summary
    total_tests_passed: int = 0
    total_tests_failed: int = 0
    overall_coverage: Optional[float] = None

    # Deployment status
    testing_deployed: bool = False
    testing_url: Optional[str] = None
    production_deployed: bool = False
    production_url: Optional[str] = None

    # Health indicators
    system_health: str = "unknown"
    last_error: Optional[str] = None
    pending_actions: List[str] = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Execution Plan Models
# -----------------------------------------------------------------------------

class ExecutionStep(BaseModel):
    """Single step in an execution plan."""
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_number: int
    aspect: ProjectAspect
    action: str
    description: str
    requires_approval: bool = False
    auto_retry: bool = True
    max_retries: int = 3
    timeout_seconds: int = 300

    # Status
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class ExecutionPlan(BaseModel):
    """Complete execution plan for a project or aspect."""
    plan_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    project_name: str
    aspect: Optional[ProjectAspect] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Steps
    steps: List[ExecutionStep] = Field(default_factory=list)
    current_step: int = 0

    # Status
    status: str = "pending"  # pending, running, completed, failed, paused
    paused_at_gate: Optional[str] = None

    # Results
    steps_completed: int = 0
    steps_failed: int = 0


# -----------------------------------------------------------------------------
# State Ledger Entry
# -----------------------------------------------------------------------------

class LedgerEntry(BaseModel):
    """Immutable entry in the state ledger."""
    entry_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    project_name: str
    aspect: Optional[ProjectAspect] = None

    # Action
    action_type: str
    action_details: Dict[str, Any] = Field(default_factory=dict)

    # Actor
    actor: str  # "system", "claude", or user_id

    # State change
    previous_state: Optional[str] = None
    new_state: Optional[str] = None

    # Immutability hash (for verification)
    content_hash: Optional[str] = None


# -----------------------------------------------------------------------------
# Request/Response Models for API
# -----------------------------------------------------------------------------

class CreateProjectRequest(BaseModel):
    """Natural language project creation request."""
    description: str = Field(..., description="Natural language project description")
    requirements: List[str] = Field(default_factory=list)
    repo_url: Optional[str] = None
    reference_urls: List[str] = Field(default_factory=list)
    user_id: str


class CreateProjectResponse(BaseModel):
    """Response for project creation."""
    project_name: str
    contract_id: str
    aspects_initialized: List[str]
    next_steps: List[str]
    message: str


class SubmitFeedbackRequest(BaseModel):
    """Request to submit testing feedback."""
    project_name: str
    aspect: ProjectAspect
    feedback_type: FeedbackType
    explanation: Optional[str] = None
    affected_features: List[str] = Field(default_factory=list)
    user_id: str


class SubmitFeedbackResponse(BaseModel):
    """Response for feedback submission."""
    feedback_id: str
    project_name: str
    aspect: ProjectAspect
    feedback_type: FeedbackType
    action_taken: str
    next_steps: List[str]
    message: str


class ApproveProductionRequest(BaseModel):
    """Request to approve production deployment."""
    project_name: str
    aspect: ProjectAspect
    justification: str
    risk_acknowledged: bool
    rollback_plan: str
    user_id: str


class ApproveProductionResponse(BaseModel):
    """Response for production approval."""
    approval_id: str
    project_name: str
    aspect: ProjectAspect
    approved: bool
    deployment_status: str
    production_url: Optional[str]
    message: str


class DashboardResponse(BaseModel):
    """Response containing dashboard data."""
    projects: List[ProjectDashboard]
    total_projects: int
    system_health: str


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def create_default_ipc(
    project_name: str,
    description: str,
    user_id: str,
    repo_url: Optional[str] = None
) -> InternalProjectContract:
    """Create a default IPC with all aspects initialized."""
    ipc = InternalProjectContract(
        project_name=project_name,
        original_description=description,
        created_by=user_id,
        repo_urls=[repo_url] if repo_url else []
    )
    ipc.initialize_aspects()
    return ipc


def should_trigger_ci(
    aspect_state: AspectState,
    reason: CITriggerReason
) -> bool:
    """
    Determine if CI should be triggered based on current state and reason.

    CI triggers ONLY when:
    - A phase completes
    - A major module completes
    - Unit tests pass
    - Bug-fix loop completes
    - Pre-testing deploy
    - Pre-production deploy

    CI does NOT trigger on feedback alone.
    """
    valid_trigger_phases = {
        CITriggerReason.PHASE_COMPLETE: [
            AspectPhase.DEVELOPMENT,
            AspectPhase.UNIT_TESTING,
            AspectPhase.INTEGRATION,
            AspectPhase.CODE_REVIEW
        ],
        CITriggerReason.MODULE_COMPLETE: [
            AspectPhase.DEVELOPMENT,
            AspectPhase.INTEGRATION
        ],
        CITriggerReason.UNIT_TESTS_PASS: [
            AspectPhase.UNIT_TESTING
        ],
        CITriggerReason.BUG_FIX_COMPLETE: [
            AspectPhase.BUG_FIXING
        ],
        CITriggerReason.PRE_TESTING_DEPLOY: [
            AspectPhase.READY_FOR_TESTING
        ],
        CITriggerReason.PRE_PRODUCTION_DEPLOY: [
            AspectPhase.READY_FOR_PRODUCTION
        ]
    }

    allowed_phases = valid_trigger_phases.get(reason, [])
    return aspect_state.current_phase in allowed_phases


def create_testing_notification(
    project_name: str,
    aspect: ProjectAspect,
    testing_url: str,
    features: List[str],
    test_summary: str,
    limitations: List[str] = None
) -> Notification:
    """Create a notification for testing readiness."""
    return Notification(
        notification_type=NotificationType.TESTING_READY,
        project_name=project_name,
        aspect=aspect,
        title=f"Testing Ready: {project_name} - {aspect.value}",
        summary=f"The {aspect.value} aspect is ready for testing",
        environment_url=testing_url,
        action_required=True,
        next_action="Please test and provide feedback",
        test_coverage_summary=test_summary,
        features_completed=features,
        known_limitations=limitations or []
    )


logger.info("Phase 12 module loaded successfully")
