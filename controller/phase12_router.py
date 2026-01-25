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

# Phase 19 fix: Import Claude backend for autonomous planning job scheduling
try:
    from .claude_backend import (
        create_job as create_claude_job,
        JobPriority,
    )
    CLAUDE_BACKEND_AVAILABLE = True
except ImportError:
    CLAUDE_BACKEND_AVAILABLE = False

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

    Phase 16E Enhanced:
    1. Validate requirements (CHD layer)
    2. Run decision engine for conflict detection
    3. If conflict: return for user resolution
    4. Register project in Project Registry with identity
    5. Generate Internal Project Contract (IPC)
    6. Initialize all aspects (Core, Backend, Frontend)
    7. Begin autonomous planning phase
    """
    logger.info("=" * 60)
    logger.info("PROJECT CREATION - START")
    logger.info(f"  User ID: {request.user_id}")
    logger.info(f"  Description length: {len(request.description)} chars")
    logger.info(f"  Requirements count: {len(request.requirements) if request.requirements else 0}")
    logger.info("=" * 60)

    try:
        # STEP 1: Validate requirements (CHD layer)
        logger.info("[STEP 1/7] CHD Validation - Starting...")
        validation = None
        try:
            from controller.chd_validator import validate_requirements
            # Validate with both description and requirements for full CHD extraction
            requirements_text = "\n".join(request.requirements) if request.requirements else None
            logger.info(f"  Calling validate_requirements with desc={len(request.description)} chars, req={len(requirements_text) if requirements_text else 0} chars")
            validation = validate_requirements(request.description, requirements_text)
            logger.info(f"  Validation result: is_valid={validation.is_valid}, errors={len(validation.errors)}, extracted_name={validation.extracted_project_name}")
            if not validation.is_valid:
                logger.warning(f"  CHD validation FAILED: {validation.errors}")
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "Validation failed",
                        "errors": validation.errors,
                        "suggestions": validation.suggestions,
                    }
                )
            logger.info("[STEP 1/7] CHD Validation - PASSED")
        except ImportError as e:
            logger.warning(f"[STEP 1/7] CHD validator not available: {e}, skipping validation")

        # STEP 2: Extract project name
        logger.info("[STEP 2/7] Project Name Extraction - Starting...")
        if validation and validation.extracted_project_name:
            project_name = validation.extracted_project_name
            logger.info(f"  Using CHD-extracted project_name: {project_name}")
        else:
            project_name = _normalize_project_name(request.description)
            logger.info(f"  Using normalized project_name: {project_name}")
        logger.info("[STEP 2/7] Project Name Extraction - DONE")

        # STEP 3: Run decision engine for conflict detection
        logger.info("[STEP 3/7] Decision Engine - Starting...")
        try:
            from controller.project_registry import get_registry
            from controller.project_decision_engine import (
                evaluate_project_creation,
                DecisionType,
            )

            registry = get_registry()
            existing_identities = registry.get_all_identities()
            logger.info(f"  Found {len(existing_identities)} existing project identities")

            # Evaluate for conflicts
            requirements_raw = "\n".join(request.requirements) if request.requirements else None
            decision = evaluate_project_creation(
                description=request.description,
                requirements=requirements_raw,
                tech_stack=None,
                aspects=None,
                existing_identities=existing_identities,
            )
            logger.info(f"  Decision: {decision.decision.value}, requires_confirmation={decision.requires_user_confirmation}")
            logger.info(f"  Confidence: {decision.confidence}, conflict_type={decision.conflict_type.value}")

            # If conflict detected, return for user resolution
            if decision.requires_user_confirmation:
                logger.warning(f"  CONFLICT DETECTED: {decision.explanation}")
                logger.info("[STEP 3/7] Decision Engine - CONFLICT, returning to user")
                return CreateProjectResponse(
                    success=False,
                    project_name=project_name,
                    contract_id=None,
                    aspects_initialized=[],
                    next_steps=[],
                    message="Conflict detected - user confirmation required",
                    metadata={
                        "conflict_detected": True,
                        "decision": decision.to_dict(),
                    }
                )

            # If decision is CHANGE_MODE
            if decision.decision == DecisionType.CHANGE_MODE:
                logger.info(f"  CHANGE_MODE: Redirecting to existing project {decision.existing_project_name}")
                return CreateProjectResponse(
                    success=False,
                    project_name=decision.existing_project_name,
                    contract_id=None,
                    aspects_initialized=[],
                    next_steps=["Use CHANGE_MODE to modify existing project"],
                    message=decision.explanation,
                    metadata={
                        "redirect_to_change_mode": True,
                        "existing_project": decision.existing_project_name,
                    }
                )

            # If decision is NEW_VERSION
            if decision.decision == DecisionType.NEW_VERSION:
                logger.info(f"  NEW_VERSION: Creating new version of {decision.existing_project_name}")
                existing_project = registry.get_project(decision.existing_project_name)
                if existing_project:
                    success, message, project = registry.create_project_version(
                        parent_project=existing_project,
                        description=request.description,
                        created_by=request.user_id,
                        requirements_raw=requirements_raw,
                    )
                    if success:
                        project_name = project.name
                        logger.info(f"  Created new version: {project_name}")
                    else:
                        logger.error(f"  Failed to create version: {message}")
                        raise HTTPException(status_code=400, detail=message)

            logger.info("[STEP 3/7] Decision Engine - PASSED (NEW_PROJECT)")

        except ImportError as e:
            logger.warning(f"[STEP 3/7] Decision engine not available: {e}")

        # STEP 4: Legacy IPC check
        logger.info("[STEP 4/7] Legacy IPC Check - Starting...")
        existing_ipc = _load_ipc(project_name)
        if existing_ipc:
            logger.error(f"  Project '{project_name}' already exists as IPC file")
            raise HTTPException(
                status_code=400,
                detail=f"Project '{project_name}' already exists"
            )
        logger.info("[STEP 4/7] Legacy IPC Check - PASSED (no existing IPC)")

        # STEP 5: Register in Project Registry with identity
        logger.info("[STEP 5/7] Registry Registration - Starting...")
        try:
            from controller.project_registry import get_registry
            registry = get_registry()

            # Use create_project_with_identity for fingerprint tracking
            logger.info(f"  Calling registry.create_project_with_identity for '{project_name}'")
            success, message, project = registry.create_project_with_identity(
                name=project_name,
                description=request.description,
                created_by=request.user_id,
                requirements_raw="\n".join(request.requirements) if request.requirements else None,
                requirements_source="api",
            )
            if not success:
                logger.error(f"  Registry registration FAILED: {message}")
                raise HTTPException(status_code=400, detail=message)
            logger.info(f"  Registry registration SUCCESS: project_id={project.project_id}")
            logger.info("[STEP 5/7] Registry Registration - PASSED")
        except ImportError as e:
            logger.warning(f"[STEP 5/7] Project registry not available: {e}")

        # STEP 6: Create Internal Project Contract (IPC)
        logger.info("[STEP 6/7] IPC Creation - Starting...")
        ipc = create_default_ipc(
            project_name=project_name,
            description=request.description,
            user_id=request.user_id,
            repo_url=request.repo_url
        )
        logger.info(f"  Created IPC: contract_id={ipc.contract_id}")

        # Add requirements
        ipc.original_requirements = request.requirements
        ipc.reference_urls = request.reference_urls

        # Save IPC
        _save_ipc(ipc)
        logger.info(f"  Saved IPC to filesystem")

        # Link IPC to registry
        try:
            registry.update_project(project_name, {"ipc_contract_id": ipc.contract_id})
            logger.info(f"  Linked IPC to registry")
        except Exception as e:
            logger.warning(f"  Failed to link IPC to registry: {e}")

        # Initialize aspects via lifecycle engine
        logger.info("  Initializing aspects...")
        engine = _get_engine()
        for aspect in ProjectAspect:
            engine.transition_phase(
                ipc=ipc,
                aspect=aspect,
                target_phase=AspectPhase.PLANNING,
                actor=request.user_id,
                reason="Project created"
            )
            logger.info(f"    {aspect.value}: transitioned to PLANNING")

        _save_ipc(ipc)
        logger.info("[STEP 6/7] IPC Creation - PASSED")

        # STEP 7: Schedule Claude planning job
        logger.info("[STEP 7/7] Job Scheduling - Starting...")
        job_id = None
        if CLAUDE_BACKEND_AVAILABLE:
            logger.info("  Claude backend is available")
            try:
                requirements_raw = "\n".join(request.requirements) if request.requirements else request.description
                task_description = f"""PLANNING PHASE - Analyze Requirements and Create Implementation Plan

PROJECT: {project_name}
ASPECTS: {', '.join([a.value for a in ProjectAspect])}

REQUIREMENTS:
{requirements_raw[:3000]}

YOUR TASK:
1. Analyze the project requirements thoroughly
2. Identify key features, components, and dependencies
3. Create a detailed implementation plan for each aspect
4. Estimate complexity and potential risks
5. Define the file structure and architecture
6. Output a structured plan in YAML format

IMPORTANT:
- Read all governance documents (AI_POLICY.md, ARCHITECTURE.md)
- Follow the project's coding standards
- Identify any clarifications needed before development
- Be conservative with scope - focus on MVP features first

OUTPUT FORMAT:
Create a PLANNING_OUTPUT.yaml file with:
- features: list of features to implement
- architecture: proposed architecture decisions
- file_structure: proposed file/folder structure
- risks: identified risks and mitigations
- questions: any questions for clarification
- estimated_phases: breakdown of development phases
"""
                logger.info(f"  Creating planning job for project '{project_name}'...")
                job = await create_claude_job(
                    project_name=project_name,
                    task_description=task_description,
                    task_type="planning",
                    created_by=request.user_id,
                    priority=JobPriority.HIGH.value,
                    aspect="core",
                    lifecycle_state="planning",
                    requested_action="write_code",
                    user_role="owner",
                )
                job_id = job.job_id
                logger.info(f"  Job created: job_id={job_id}, state={job.state.value}")
                logger.info("[STEP 7/7] Job Scheduling - PASSED")
            except Exception as e:
                logger.error(f"  Job scheduling FAILED: {e}", exc_info=True)
                logger.info("[STEP 7/7] Job Scheduling - FAILED (but project created)")
        else:
            logger.warning("  Claude backend NOT available - skipping job scheduling")
            logger.info("[STEP 7/7] Job Scheduling - SKIPPED")

        # Build next_steps based on whether job was scheduled
        if job_id:
            next_steps = [
                f"Claude planning job scheduled (job ID: {job_id[:8]}...)",
                "Planning phase has begun automatically",
                "You will be notified when review is needed",
                "Use /project_status to check progress"
            ]
        else:
            next_steps = [
                "Project created but Claude backend unavailable",
                "Manual planning may be required",
                "Use /project_status to check progress"
            ]

        logger.info("=" * 60)
        logger.info("PROJECT CREATION - SUCCESS")
        logger.info(f"  Project: {project_name}")
        logger.info(f"  Contract ID: {ipc.contract_id}")
        logger.info(f"  Job ID: {job_id or 'None'}")
        logger.info("=" * 60)

        return CreateProjectResponse(
            success=True,
            project_name=project_name,
            contract_id=ipc.contract_id,
            aspects_initialized=[a.value for a in ProjectAspect],
            next_steps=next_steps,
            message=f"Project '{project_name}' created successfully. Autonomous development starting."
        )

    except HTTPException as he:
        logger.error(f"PROJECT CREATION - HTTP ERROR: {he.detail}")
        raise
    except Exception as e:
        logger.error("=" * 60)
        logger.error("PROJECT CREATION - FAILED")
        logger.error(f"  Error: {e}", exc_info=True)
        logger.error("=" * 60)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error creating project: {str(e)}"
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

    Data sources (Phase 16C):
    1. Project Registry (primary source for new projects)
    2. IPC files (legacy source for older projects)
    """
    dashboards = []
    seen_projects = set()

    # Source 1: Project Registry (Phase 16C - primary source)
    try:
        from controller.project_registry import get_registry
        registry = get_registry()
        registry_projects = registry.get_dashboard_projects()

        for proj in registry_projects:
            project_name = proj.get("project_name", "")
            if project_name:
                seen_projects.add(project_name)
                # Convert registry format to dashboard format
                aspects_progress = {}
                for aspect_name, aspect_status in proj.get("aspects", {}).items():
                    # Map registry aspect names to ProjectAspect enum
                    aspect_mapping = {
                        "api": ProjectAspect.CORE,
                        "core": ProjectAspect.CORE,
                        "backend": ProjectAspect.BACKEND,
                        "admin": ProjectAspect.BACKEND,
                        "frontend": ProjectAspect.FRONTEND,
                        "frontend_web": ProjectAspect.FRONTEND,
                    }
                    mapped_aspect = aspect_mapping.get(aspect_name.lower(), ProjectAspect.CORE)

                    # Map registry status to AspectPhase
                    phase_mapping = {
                        "created": AspectPhase.NOT_STARTED,
                        "planning": AspectPhase.PLANNING,
                        "development": AspectPhase.DEVELOPMENT,
                        "testing": AspectPhase.UNIT_TESTING,
                        "awaiting_feedback": AspectPhase.AWAITING_FEEDBACK,
                        "ready_for_production": AspectPhase.READY_FOR_PRODUCTION,
                        "deployed": AspectPhase.DEPLOYED_PRODUCTION,
                    }
                    current_phase = phase_mapping.get(aspect_status, AspectPhase.NOT_STARTED)

                    # Use mapped aspect as key (ProjectAspect enum)
                    aspects_progress[mapped_aspect] = AspectProgress(
                        aspect=mapped_aspect,
                        current_phase=current_phase,
                        phase_progress_percent=10.0 if aspect_status == "planning" else 0.0,
                        ci_status=None,
                        last_ci_at=None,
                        test_summary="Pending",
                        deploy_status=None,
                        last_deploy_at=None,
                        health="unknown"
                    )

                # Parse timestamps - handle ISO format from registry
                created_at_str = proj.get("created_at", "")
                updated_at_str = proj.get("updated_at", "")
                try:
                    created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now()
                    updated_at = datetime.fromisoformat(updated_at_str) if updated_at_str else datetime.now()
                except ValueError:
                    created_at = datetime.now()
                    updated_at = datetime.now()

                dashboard = ProjectDashboard(
                    project_name=project_name,
                    overall_status=proj.get("current_status", "created"),
                    created_at=created_at,
                    updated_at=updated_at,
                    aspects=aspects_progress,
                    total_tests_passed=0,
                    total_tests_failed=0,
                    system_health="healthy",
                    last_activity=updated_at,
                    notifications_count=0
                )
                dashboards.append(dashboard)

        logger.debug(f"Loaded {len(registry_projects)} projects from registry")
    except ImportError:
        logger.debug("Project registry not available")
    except Exception as e:
        logger.warning(f"Failed to load registry projects: {e}")

    # Source 2: IPC files (legacy - for projects not in registry)
    if PROJECTS_DIR.exists():
        for project_dir in PROJECTS_DIR.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith("."):
                if project_dir.name in seen_projects:
                    continue  # Already loaded from registry
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
