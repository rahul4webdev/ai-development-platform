"""
Phase 12: API Router for Autonomous Multi-Aspect Project Orchestration

This module provides FastAPI routes for:
- Project creation from natural language
- Multi-aspect project management
- Approval workflow with structured feedback
- Dashboard API
- Notification endpoints
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any
import yaml
import json

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .phase12 import (
    ProjectAspect,
    AspectPhase,
    FeedbackType,
    InternalProjectContract,
    TestingFeedback,
    ProductionApproval,
    TechStackConfig,
    AspectConfig,
    AspectState,
    AspectProgress,
    ProjectDashboard,
    Notification,
    create_default_ipc,
    CreateProjectRequest,
    CreateProjectResponse,
    SubmitFeedbackRequest,
    SubmitFeedbackResponse,
    ApproveProductionRequest,
    ApproveProductionResponse,
    DashboardResponse
)

from .lifecycle_engine import get_lifecycle_engine, LifecycleEngine

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("phase12_router")

# -----------------------------------------------------------------------------
# Router Setup
# -----------------------------------------------------------------------------
router = APIRouter(prefix="/v2", tags=["Phase 12 - Multi-Aspect Projects"])

# -----------------------------------------------------------------------------
# Storage (In-memory for now, will be persisted)
# -----------------------------------------------------------------------------
_ipc_store: Dict[str, InternalProjectContract] = {}

# Projects directory (will be set during integration)
PROJECTS_DIR = Path(__file__).parent.parent / "projects"


def _get_engine() -> LifecycleEngine:
    """Get the lifecycle engine instance."""
    return get_lifecycle_engine(PROJECTS_DIR)


def _save_ipc(ipc: InternalProjectContract) -> None:
    """Save IPC to filesystem."""
    project_dir = PROJECTS_DIR / ipc.project_name
    project_dir.mkdir(parents=True, exist_ok=True)

    ipc_path = project_dir / "INTERNAL_PROJECT_CONTRACT.yaml"

    # Convert to dict for YAML serialization
    ipc_dict = {
        "contract_id": ipc.contract_id,
        "project_name": ipc.project_name,
        "created_at": ipc.created_at.isoformat(),
        "updated_at": ipc.updated_at.isoformat(),
        "created_by": ipc.created_by,
        "original_description": ipc.original_description,
        "original_requirements": ipc.original_requirements,
        "repo_urls": ipc.repo_urls,
        "reference_urls": ipc.reference_urls,
        "testing_domain": ipc.testing_domain,
        "production_domain": ipc.production_domain,
        "constraints": ipc.constraints,
        "overall_status": ipc.overall_status,
        "tech_stack": ipc.tech_stack.model_dump(),
        "aspects": {k.value: v.model_dump() for k, v in ipc.aspects.items()},
        "aspect_states": {k.value: {
            **v.model_dump(),
            "phase_started_at": v.phase_started_at.isoformat() if v.phase_started_at else None,
            "last_ci_at": v.last_ci_at.isoformat() if v.last_ci_at else None,
            "last_deploy_testing_at": v.last_deploy_testing_at.isoformat() if v.last_deploy_testing_at else None,
            "last_deploy_production_at": v.last_deploy_production_at.isoformat() if v.last_deploy_production_at else None
        } for k, v in ipc.aspect_states.items()}
    }

    with open(ipc_path, "w") as f:
        yaml.dump(ipc_dict, f, default_flow_style=False, allow_unicode=True)

    # Also keep in memory
    _ipc_store[ipc.project_name] = ipc

    logger.info(f"Saved IPC for project: {ipc.project_name}")


def _load_ipc(project_name: str) -> Optional[InternalProjectContract]:
    """Load IPC from filesystem or memory."""
    # Check memory first
    if project_name in _ipc_store:
        return _ipc_store[project_name]

    # Try filesystem
    ipc_path = PROJECTS_DIR / project_name / "INTERNAL_PROJECT_CONTRACT.yaml"
    if not ipc_path.exists():
        return None

    try:
        with open(ipc_path, "r") as f:
            data = yaml.safe_load(f)

        # Reconstruct IPC
        ipc = InternalProjectContract(
            contract_id=data.get("contract_id"),
            project_name=data["project_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            created_by=data.get("created_by"),
            original_description=data.get("original_description", ""),
            original_requirements=data.get("original_requirements", []),
            repo_urls=data.get("repo_urls", []),
            reference_urls=data.get("reference_urls", []),
            testing_domain=data.get("testing_domain"),
            production_domain=data.get("production_domain"),
            constraints=data.get("constraints", []),
            overall_status=data.get("overall_status", "initializing"),
            tech_stack=TechStackConfig(**data.get("tech_stack", {}))
        )

        # Reconstruct aspects
        for aspect_key, aspect_data in data.get("aspects", {}).items():
            aspect = ProjectAspect(aspect_key)
            ipc.aspects[aspect] = AspectConfig(**aspect_data)

        # Reconstruct aspect states
        for aspect_key, state_data in data.get("aspect_states", {}).items():
            aspect = ProjectAspect(aspect_key)
            # Handle datetime fields
            if state_data.get("phase_started_at"):
                state_data["phase_started_at"] = datetime.fromisoformat(state_data["phase_started_at"])
            if state_data.get("last_ci_at"):
                state_data["last_ci_at"] = datetime.fromisoformat(state_data["last_ci_at"])
            if state_data.get("last_deploy_testing_at"):
                state_data["last_deploy_testing_at"] = datetime.fromisoformat(state_data["last_deploy_testing_at"])
            if state_data.get("last_deploy_production_at"):
                state_data["last_deploy_production_at"] = datetime.fromisoformat(state_data["last_deploy_production_at"])
            ipc.aspect_states[aspect] = AspectState(**state_data)

        _ipc_store[project_name] = ipc
        return ipc

    except Exception as e:
        logger.error(f"Failed to load IPC for {project_name}: {e}")
        return None


# -----------------------------------------------------------------------------
# Project Creation Endpoints
# -----------------------------------------------------------------------------

@router.post("/project/create", response_model=CreateProjectResponse)
async def create_project_from_natural_language(
    request: CreateProjectRequest
):
    """
    Create a new multi-aspect project from natural language description.

    Claude will:
    1. Interpret the natural language description
    2. Generate an Internal Project Contract (IPC)
    3. Initialize all aspects (Core, Backend, Frontend)
    4. Begin autonomous planning phase
    """
    # Normalize project name from description
    project_name = _normalize_project_name(request.description)

    # Check if project already exists
    if _load_ipc(project_name):
        raise HTTPException(
            status_code=400,
            detail=f"Project '{project_name}' already exists"
        )

    # Create IPC
    ipc = create_default_ipc(
        project_name=project_name,
        description=request.description,
        user_id=request.user_id,
        repo_url=request.repo_url
    )

    # Add requirements
    ipc.original_requirements = request.requirements
    ipc.reference_urls = request.reference_urls

    # Save IPC
    _save_ipc(ipc)

    # Get engine and start planning for each aspect
    engine = _get_engine()
    for aspect in ProjectAspect:
        engine.transition_phase(
            ipc=ipc,
            aspect=aspect,
            target_phase=AspectPhase.PLANNING,
            actor=request.user_id,
            reason="Project created"
        )

    _save_ipc(ipc)

    logger.info(f"Created project: {project_name} with all aspects initialized")

    return CreateProjectResponse(
        project_name=project_name,
        contract_id=ipc.contract_id,
        aspects_initialized=[a.value for a in ProjectAspect],
        next_steps=[
            "Autonomous planning will begin",
            "Development will proceed automatically",
            "You will be notified when testing is ready",
            "Approval required before production deployment"
        ],
        message=f"Project '{project_name}' created successfully. Autonomous development starting."
    )


def _normalize_project_name(description: str) -> str:
    """Normalize description to a valid project name."""
    # Take first few words and convert to slug
    words = description.lower().split()[:3]
    name = "-".join(words)
    # Remove invalid characters
    name = "".join(c if c.isalnum() or c == "-" else "" for c in name)
    # Ensure it starts with a letter
    if name and not name[0].isalpha():
        name = "project-" + name
    return name or "new-project"


# -----------------------------------------------------------------------------
# Project Status Endpoints
# -----------------------------------------------------------------------------

@router.get("/project/{project_name}")
async def get_project_details(project_name: str):
    """Get full project details including all aspects."""
    ipc = _load_ipc(project_name)
    if not ipc:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    return {
        "project_name": ipc.project_name,
        "contract_id": ipc.contract_id,
        "created_at": ipc.created_at.isoformat(),
        "updated_at": ipc.updated_at.isoformat(),
        "created_by": ipc.created_by,
        "original_description": ipc.original_description,
        "overall_status": ipc.overall_status,
        "aspects": {
            aspect.value: {
                "enabled": config.enabled,
                "description": config.description,
                "current_phase": ipc.aspect_states[aspect].current_phase.value if aspect in ipc.aspect_states else "not_started",
                "testing_url": config.testing_url,
                "production_url": config.production_url
            }
            for aspect, config in ipc.aspects.items()
        }
    }


@router.get("/project/{project_name}/aspect/{aspect}")
async def get_aspect_details(
    project_name: str,
    aspect: ProjectAspect
):
    """Get details for a specific project aspect."""
    ipc = _load_ipc(project_name)
    if not ipc:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    if aspect not in ipc.aspect_states:
        raise HTTPException(status_code=404, detail=f"Aspect '{aspect.value}' not found")

    state = ipc.aspect_states[aspect]
    config = ipc.aspects.get(aspect)

    return {
        "project_name": project_name,
        "aspect": aspect.value,
        "enabled": config.enabled if config else True,
        "current_phase": state.current_phase.value,
        "phase_started_at": state.phase_started_at.isoformat() if state.phase_started_at else None,
        "last_ci_status": state.last_ci_status,
        "last_ci_at": state.last_ci_at.isoformat() if state.last_ci_at else None,
        "test_coverage": state.test_coverage_percent,
        "unit_tests": {
            "passed": state.unit_tests_passed,
            "failed": state.unit_tests_failed
        },
        "integration_tests": {
            "passed": state.integration_tests_passed,
            "failed": state.integration_tests_failed
        },
        "feedback_pending": state.feedback_pending,
        "feedback_history": state.feedback_history
    }


# -----------------------------------------------------------------------------
# Feedback Endpoints
# -----------------------------------------------------------------------------

@router.post("/project/{project_name}/aspect/{aspect}/feedback", response_model=SubmitFeedbackResponse)
async def submit_feedback(
    project_name: str,
    aspect: ProjectAspect,
    request: SubmitFeedbackRequest
):
    """
    Submit testing feedback for an aspect.

    Feedback types:
    - approve: Approve for production
    - bug: Report a bug (explanation required)
    - improvements: Request improvements (explanation required)
    - reject: Reject completely (explanation required)
    """
    ipc = _load_ipc(project_name)
    if not ipc:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Validate explanation requirement
    if request.feedback_type != FeedbackType.APPROVE:
        if not request.explanation or len(request.explanation.strip()) < 10:
            raise HTTPException(
                status_code=400,
                detail=f"{request.feedback_type.value} feedback requires explanation (min 10 characters)"
            )

    # Create feedback object
    feedback = TestingFeedback(
        project_name=project_name,
        aspect=aspect,
        feedback_type=request.feedback_type,
        submitted_by=request.user_id,
        explanation=request.explanation,
        affected_features=request.affected_features
    )

    # Process feedback
    engine = _get_engine()
    success, message, next_phase = engine.process_feedback(ipc, aspect, feedback)

    if not success:
        raise HTTPException(status_code=400, detail=message)

    # Save updated IPC
    _save_ipc(ipc)

    # Determine action taken
    action_map = {
        FeedbackType.APPROVE: "Approved for production. Awaiting final approval.",
        FeedbackType.BUG: "Bug report received. Starting autonomous bug-fix loop.",
        FeedbackType.IMPROVEMENTS: "Improvement request received. Starting enhancement work.",
        FeedbackType.REJECT: "Rejected. Returning to development phase."
    }

    # Determine next steps
    next_steps_map = {
        FeedbackType.APPROVE: ["Production approval required", "Will deploy after approval"],
        FeedbackType.BUG: ["Claude will analyze and fix the bug", "CI will run after fix", "You will be notified when ready"],
        FeedbackType.IMPROVEMENTS: ["Claude will implement improvements", "CI will run after changes", "You will be notified when ready"],
        FeedbackType.REJECT: ["Claude will reassess requirements", "Development will restart", "You will be notified when ready"]
    }

    return SubmitFeedbackResponse(
        feedback_id=feedback.feedback_id,
        project_name=project_name,
        aspect=aspect,
        feedback_type=request.feedback_type,
        action_taken=action_map[request.feedback_type],
        next_steps=next_steps_map[request.feedback_type],
        message=f"Feedback processed. Next phase: {next_phase.value}"
    )


# -----------------------------------------------------------------------------
# Production Approval Endpoints
# -----------------------------------------------------------------------------

@router.post("/project/{project_name}/aspect/{aspect}/approve-production", response_model=ApproveProductionResponse)
async def approve_production_deployment(
    project_name: str,
    aspect: ProjectAspect,
    request: ApproveProductionRequest
):
    """
    Approve production deployment for an aspect.

    This is the ONLY way to deploy to production.
    No auto-promotion is allowed.
    """
    ipc = _load_ipc(project_name)
    if not ipc:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    # Validate aspect is ready for production
    if aspect not in ipc.aspect_states:
        raise HTTPException(status_code=404, detail=f"Aspect '{aspect.value}' not found")

    state = ipc.aspect_states[aspect]
    if state.current_phase != AspectPhase.READY_FOR_PRODUCTION:
        raise HTTPException(
            status_code=400,
            detail=f"Aspect is not ready for production. Current phase: {state.current_phase.value}"
        )

    if not request.risk_acknowledged:
        raise HTTPException(
            status_code=400,
            detail="You must acknowledge production risk"
        )

    # Create approval
    approval = ProductionApproval(
        project_name=project_name,
        aspect=aspect,
        approved=True,
        approved_by=request.user_id,
        justification=request.justification,
        risk_acknowledged=request.risk_acknowledged,
        rollback_plan=request.rollback_plan
    )

    # Deploy to production
    engine = _get_engine()
    success, message, production_url = engine.deploy_to_production(
        ipc=ipc,
        aspect=aspect,
        approved_by=request.user_id,
        justification=request.justification
    )

    if not success:
        raise HTTPException(status_code=400, detail=message)

    # Save updated IPC
    _save_ipc(ipc)

    return ApproveProductionResponse(
        approval_id=approval.approval_id,
        project_name=project_name,
        aspect=aspect,
        approved=True,
        deployment_status="deployed",
        production_url=production_url,
        message=f"Successfully deployed {aspect.value} to production"
    )


# -----------------------------------------------------------------------------
# Dashboard Endpoints
# -----------------------------------------------------------------------------

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard():
    """
    Get read-only dashboard view of all projects.

    Shows:
    - Project list
    - Aspect-wise progress (Core / Backend / Frontend)
    - Current phase per aspect
    - CI results
    - Internal test summary
    - Last deployment status
    - Health indicators
    """
    dashboards = []

    # Get all projects from directory
    if PROJECTS_DIR.exists():
        for project_dir in PROJECTS_DIR.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                ipc = _load_ipc(project_dir.name)
                if ipc:
                    dashboard = _build_dashboard(ipc)
                    dashboards.append(dashboard)

    return DashboardResponse(
        projects=dashboards,
        total_projects=len(dashboards),
        system_health="healthy" if dashboards else "no_projects"
    )


@router.get("/dashboard/{project_name}", response_model=ProjectDashboard)
async def get_project_dashboard(project_name: str):
    """Get dashboard view for a specific project."""
    ipc = _load_ipc(project_name)
    if not ipc:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    return _build_dashboard(ipc)


def _build_dashboard(ipc: InternalProjectContract) -> ProjectDashboard:
    """Build dashboard data from IPC."""
    aspects_progress = {}
    total_tests_passed = 0
    total_tests_failed = 0

    for aspect, state in ipc.aspect_states.items():
        config = ipc.aspects.get(aspect)

        # Calculate phase progress
        phase_order = list(AspectPhase)
        current_index = phase_order.index(state.current_phase)
        progress = (current_index / len(phase_order)) * 100

        # Determine health
        health = "unknown"
        if state.current_phase == AspectPhase.CI_FAILED:
            health = "error"
        elif state.feedback_pending:
            health = "warning"
        elif state.current_phase in [AspectPhase.DEPLOYED_TESTING, AspectPhase.DEPLOYED_PRODUCTION]:
            health = "healthy"

        aspects_progress[aspect] = AspectProgress(
            aspect=aspect,
            current_phase=state.current_phase,
            phase_progress_percent=progress,
            ci_status=state.last_ci_status,
            last_ci_at=state.last_ci_at,
            test_summary=f"Unit: {state.unit_tests_passed}/{state.unit_tests_passed + state.unit_tests_failed}",
            deploy_status="deployed" if state.last_deploy_testing_at else None,
            last_deploy_at=state.last_deploy_testing_at,
            health=health
        )

        total_tests_passed += state.unit_tests_passed + state.integration_tests_passed
        total_tests_failed += state.unit_tests_failed + state.integration_tests_failed

    # Determine overall health
    healths = [p.health for p in aspects_progress.values()]
    if "error" in healths:
        system_health = "error"
    elif "warning" in healths:
        system_health = "warning"
    elif "healthy" in healths:
        system_health = "healthy"
    else:
        system_health = "unknown"

    # Check deployment status
    testing_deployed = any(
        s.last_deploy_testing_at is not None for s in ipc.aspect_states.values()
    )
    production_deployed = any(
        s.last_deploy_production_at is not None for s in ipc.aspect_states.values()
    )

    # Build pending actions
    pending_actions = []
    for aspect, state in ipc.aspect_states.items():
        if state.current_phase == AspectPhase.AWAITING_FEEDBACK:
            pending_actions.append(f"Provide feedback for {aspect.value}")
        elif state.current_phase == AspectPhase.READY_FOR_PRODUCTION:
            pending_actions.append(f"Approve production for {aspect.value}")

    return ProjectDashboard(
        project_name=ipc.project_name,
        overall_status=ipc.overall_status,
        created_at=ipc.created_at,
        updated_at=ipc.updated_at,
        aspects=aspects_progress,
        total_tests_passed=total_tests_passed,
        total_tests_failed=total_tests_failed,
        testing_deployed=testing_deployed,
        testing_url=ipc.testing_domain,
        production_deployed=production_deployed,
        production_url=ipc.production_domain,
        system_health=system_health,
        pending_actions=pending_actions
    )


# -----------------------------------------------------------------------------
# Notification Endpoints
# -----------------------------------------------------------------------------

@router.get("/notifications")
async def get_notifications(
    project_name: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200)
):
    """Get pending notifications."""
    engine = _get_engine()
    notifications = engine.get_pending_notifications()

    if project_name:
        notifications = [n for n in notifications if n.project_name == project_name]

    return {
        "notifications": [n.model_dump() for n in notifications[:limit]],
        "count": len(notifications)
    }


# -----------------------------------------------------------------------------
# Ledger Endpoints (Audit Trail)
# -----------------------------------------------------------------------------

@router.get("/ledger/{project_name}")
async def get_project_ledger(
    project_name: str,
    aspect: Optional[ProjectAspect] = Query(None),
    limit: int = Query(100, ge=1, le=1000)
):
    """Get immutable ledger entries for a project."""
    ipc = _load_ipc(project_name)
    if not ipc:
        raise HTTPException(status_code=404, detail=f"Project '{project_name}' not found")

    engine = _get_engine()
    entries = engine.get_ledger_entries(project_name, aspect, limit)

    return {
        "project_name": project_name,
        "entries": [
            {
                "entry_id": e.entry_id,
                "timestamp": e.timestamp.isoformat(),
                "aspect": e.aspect.value if e.aspect else None,
                "action_type": e.action_type,
                "action_details": e.action_details,
                "actor": e.actor,
                "previous_state": e.previous_state,
                "new_state": e.new_state
            }
            for e in entries
        ],
        "count": len(entries)
    }


logger.info("Phase 12 Router loaded successfully")
