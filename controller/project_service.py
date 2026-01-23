"""
Phase 16C: Project Service

Unified service for project creation that:
1. Validates requirements (CHD layer)
2. Creates project in registry
3. Creates lifecycle instances
4. Triggers Claude planning job
5. Provides progress callbacks

This is the single entry point for project creation.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Awaitable

from controller.project_registry import (
    get_registry,
    Project,
    ProjectStatus,
    RegistryError,
)
from controller.chd_validator import (
    validate_requirements,
    validate_file,
    ValidationResult,
)

logger = logging.getLogger("project_service")

# Try to import lifecycle module
try:
    from controller.lifecycle_v2 import (
        LifecycleEngine,
        LifecycleMode,
        ProjectAspect,
        TransitionTrigger,
        UserRole,
    )
    LIFECYCLE_AVAILABLE = True
except ImportError:
    LIFECYCLE_AVAILABLE = False
    logger.warning("lifecycle_v2 module not available")

# Try to import Claude backend for job scheduling (Phase 19 fix)
try:
    from controller.claude_backend import (
        create_job as create_claude_job,
        JobPriority,
    )
    CLAUDE_BACKEND_AVAILABLE = True
except ImportError:
    CLAUDE_BACKEND_AVAILABLE = False
    logger.warning("claude_backend module not available - autonomous planning disabled")


# -----------------------------------------------------------------------------
# Progress Callback Types
# -----------------------------------------------------------------------------
ProgressCallback = Callable[[str, str, Optional[Dict[str, Any]]], Awaitable[None]]


@dataclass
class ProjectCreationProgress:
    """Progress state for project creation."""
    step: str
    status: str  # "pending", "in_progress", "completed", "failed"
    message: str
    details: Optional[Dict[str, Any]] = None


# Progress steps
PROGRESS_STEPS = [
    ("input_received", "Input received"),
    ("parsing", "Parsing requirements"),
    ("validating", "Validating execution plan"),
    ("creating_project", "Creating project"),
    ("creating_lifecycles", "Initializing aspects"),
    ("scheduling_planning", "Scheduling planning job"),
    ("completed", "Project created successfully"),
]


# -----------------------------------------------------------------------------
# Project Service
# -----------------------------------------------------------------------------
class ProjectService:
    """
    Unified service for project creation.

    Handles the full flow:
    1. Input validation (text or file)
    2. Requirements parsing
    3. CHD validation
    4. Project registry creation
    5. Lifecycle instance creation
    6. Claude job scheduling
    """

    def __init__(self):
        self._registry = get_registry()
        self._lifecycle_engine: Optional["LifecycleEngine"] = None

    def _get_lifecycle_engine(self) -> Optional["LifecycleEngine"]:
        """Get or create lifecycle engine."""
        if not LIFECYCLE_AVAILABLE:
            return None

        if self._lifecycle_engine is None:
            try:
                self._lifecycle_engine = LifecycleEngine()
            except Exception as e:
                logger.error(f"Failed to create lifecycle engine: {e}")
                return None

        return self._lifecycle_engine

    # -------------------------------------------------------------------------
    # Main Entry Points
    # -------------------------------------------------------------------------

    async def create_project_from_text(
        self,
        description: str,
        user_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Create project from text description.

        Returns structured result with success/error.
        """
        return await self._create_project(
            description=description,
            requirements_raw=description,
            requirements_source="text",
            user_id=user_id,
            progress_callback=progress_callback,
        )

    async def create_project_from_file(
        self,
        filename: str,
        file_content: bytes,
        user_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Create project from uploaded file.

        Returns structured result with success/error.
        """
        # Step 1: Validate and parse file
        if progress_callback:
            await progress_callback("input_received", "in_progress", {"filename": filename})

        is_valid, error, content = validate_file(filename, file_content)
        if not is_valid:
            return {
                "success": False,
                "error": error,
                "code": "FILE_VALIDATION_FAILED",
            }

        if progress_callback:
            await progress_callback("input_received", "completed", {"filename": filename, "size": len(content)})

        return await self._create_project(
            description=content[:500],  # First 500 chars as description
            requirements_raw=content,
            requirements_source="file",
            user_id=user_id,
            progress_callback=progress_callback,
        )

    # -------------------------------------------------------------------------
    # Core Implementation
    # -------------------------------------------------------------------------

    async def _create_project(
        self,
        description: str,
        requirements_raw: str,
        requirements_source: str,
        user_id: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Core project creation logic.

        Steps:
        1. Parse and validate requirements
        2. Create project in registry
        3. Create lifecycle instances
        4. Schedule planning job

        CRITICAL (Phase 19 fix): project_name from CHD takes ABSOLUTE precedence.
        NO fallback to inferred text, system prompts, or descriptions.
        """
        result = {
            "success": False,
            "project_name": None,
            "project_id": None,
            "aspects": [],
            "lifecycle_ids": [],
            "next_steps": [],
            "error": None,
            "code": None,
            "validation_result": None,
        }

        try:
            # Step 1: Parse requirements
            if progress_callback:
                await progress_callback("parsing", "in_progress", None)

            # Step 2: Validate requirements
            if progress_callback:
                await progress_callback("validating", "in_progress", None)

            validation = validate_requirements(description, requirements_raw)
            result["validation_result"] = validation.to_dict()

            if not validation.is_valid:
                if progress_callback:
                    await progress_callback("validating", "failed", {"errors": validation.errors})
                result["error"] = validation.get_user_message()
                result["code"] = "VALIDATION_FAILED"
                return result

            if progress_callback:
                await progress_callback("validating", "completed", {
                    "aspects": validation.extracted_aspects,
                    "tech": validation.extracted_tech,
                })

            # Step 3: Create project in registry
            if progress_callback:
                await progress_callback("creating_project", "in_progress", None)

            # CRITICAL (Phase 19 fix): Use CHD project_name if available
            # For file-based projects, CHD project_name is the SINGLE source of truth
            # For text-based projects, generate a safe name from description
            if requirements_source == "file" and validation.extracted_project_name:
                # CHD project_name takes ABSOLUTE precedence
                project_name = validation.extracted_project_name
                project_description = validation.extracted_description or description[:500]
                logger.info(f"Using CHD project_name: {project_name}")
            else:
                # Text-based input: generate safe name from description
                project_name = self._generate_safe_name(description)
                project_description = description[:500]
                logger.info(f"Generated project name from description: {project_name}")

            success, message, project = self._registry.create_project(
                name=project_name,
                description=project_description,
                created_by=user_id,
                requirements_raw=requirements_raw,
                requirements_source=requirements_source,
                tech_stack=validation.extracted_tech,
                aspects=validation.extracted_aspects,
            )

            if not success:
                if progress_callback:
                    await progress_callback("creating_project", "failed", {"message": message})
                result["error"] = message
                result["code"] = "PROJECT_CREATION_FAILED"
                return result

            result["project_name"] = project.name
            result["project_id"] = project.project_id
            result["aspects"] = list(project.aspects.keys())

            if progress_callback:
                await progress_callback("creating_project", "completed", {
                    "project_name": project.name,
                    "project_id": project.project_id,
                })

            # Step 4: Create lifecycle instances
            if progress_callback:
                await progress_callback("creating_lifecycles", "in_progress", None)

            lifecycle_ids = await self._create_lifecycles(
                project=project,
                aspects=validation.extracted_aspects,
                user_id=user_id,
            )
            result["lifecycle_ids"] = lifecycle_ids

            # Update project with lifecycle IDs
            for lc_id in lifecycle_ids:
                self._registry.add_lifecycle_id(project.name, lc_id)

            if progress_callback:
                await progress_callback("creating_lifecycles", "completed", {
                    "lifecycle_ids": lifecycle_ids,
                })

            # Step 5: Update aspect statuses to PLANNING
            for aspect in project.aspects.keys():
                self._registry.update_aspect_status(
                    project.name,
                    aspect,
                    ProjectStatus.PLANNING.value,
                )

            # Step 6: Schedule Claude planning job (Phase 19 fix - THIS WAS MISSING!)
            if progress_callback:
                await progress_callback("scheduling_planning", "in_progress", None)

            job_ids = await self._schedule_planning_jobs(
                project=project,
                requirements_raw=requirements_raw,
                user_id=user_id,
            )
            result["job_ids"] = job_ids

            if progress_callback:
                await progress_callback("scheduling_planning", "completed", {
                    "job_ids": job_ids,
                })

            # Step 7: Completion
            if progress_callback:
                await progress_callback("completed", "completed", {
                    "project_name": project.name,
                    "aspects": list(project.aspects.keys()),
                    "jobs_scheduled": len(job_ids),
                })

            result["success"] = True
            if job_ids:
                result["next_steps"] = [
                    f"Claude planning job scheduled (job ID: {job_ids[0][:8]}...)",
                    "Planning phase has begun automatically",
                    "You will be notified when review is needed",
                    "Use /project_status to check progress",
                ]
            else:
                result["next_steps"] = [
                    "Project created but Claude backend unavailable",
                    "Manual planning may be required",
                    "Use /project_status to check progress",
                ]

            logger.info(f"Project created successfully: {project.name}")
            return result

        except RegistryError as e:
            logger.error(f"Registry error creating project: {e}")
            result["error"] = e.message
            result["code"] = e.code
            if progress_callback:
                await progress_callback("creating_project", "failed", {"error": str(e)})
            return result

        except Exception as e:
            logger.error(f"Unexpected error creating project: {e}", exc_info=True)
            result["error"] = f"Unexpected error: {str(e)}"
            result["code"] = "INTERNAL_ERROR"
            if progress_callback:
                await progress_callback("creating_project", "failed", {"error": str(e)})
            return result

    async def _create_lifecycles(
        self,
        project: Project,
        aspects: List[str],
        user_id: str,
    ) -> List[str]:
        """Create lifecycle instances for each aspect."""
        lifecycle_ids = []

        engine = self._get_lifecycle_engine()
        if not engine:
            logger.warning("Lifecycle engine not available, skipping lifecycle creation")
            return lifecycle_ids

        for aspect_name in aspects:
            try:
                # Map string to enum
                aspect_enum = self._map_aspect_to_enum(aspect_name)
                if not aspect_enum:
                    continue

                success, message, lifecycle = await engine.create_lifecycle(
                    project_name=project.name,
                    mode=LifecycleMode.PROJECT_MODE,
                    aspect=aspect_enum,
                    created_by=user_id,
                    description=project.description[:200],
                )

                if success and lifecycle:
                    lifecycle_ids.append(lifecycle.lifecycle_id)

                    # Transition to PLANNING
                    await engine.transition_lifecycle(
                        lifecycle_id=lifecycle.lifecycle_id,
                        trigger=TransitionTrigger.SYSTEM_INIT,
                        triggered_by=user_id,
                        user_role=UserRole.OWNER,
                        reason="Initial project planning",
                    )

                    logger.info(f"Created lifecycle {lifecycle.lifecycle_id} for {project.name}/{aspect_name}")
                else:
                    logger.warning(f"Failed to create lifecycle for {aspect_name}: {message}")

            except Exception as e:
                logger.error(f"Error creating lifecycle for {aspect_name}: {e}")

        return lifecycle_ids

    def _map_aspect_to_enum(self, aspect_name: str) -> Optional["ProjectAspect"]:
        """Map aspect name string to ProjectAspect enum."""
        if not LIFECYCLE_AVAILABLE:
            return None

        mapping = {
            "api": ProjectAspect.CORE,  # API goes to core
            "backend": ProjectAspect.BACKEND,
            "frontend": ProjectAspect.FRONTEND_WEB,
            "admin": ProjectAspect.ADMIN,
            "core": ProjectAspect.CORE,
        }

        return mapping.get(aspect_name.lower())

    async def _schedule_planning_jobs(
        self,
        project: Project,
        requirements_raw: str,
        user_id: str,
    ) -> List[str]:
        """
        Schedule Claude planning job(s) for the project.

        Phase 19 fix: THIS WAS THE MISSING PIECE!
        Previously, project creation only set state to PLANNING but never
        actually scheduled a Claude job to do the planning work.

        Creates a planning job with HIGH priority to analyze requirements
        and generate the implementation plan.
        """
        job_ids = []

        if not CLAUDE_BACKEND_AVAILABLE:
            logger.warning("Claude backend not available - cannot schedule planning jobs")
            return job_ids

        try:
            # Create planning job description from requirements
            task_description = f"""PLANNING PHASE - Analyze Requirements and Create Implementation Plan

PROJECT: {project.name}
ASPECTS: {', '.join(project.aspects.keys())}
TECH STACK: {project.tech_stack}

REQUIREMENTS:
{requirements_raw[:2000]}

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

            # Create job with HIGH priority (planning is critical path)
            job = await create_claude_job(
                project_name=project.name,
                task_description=task_description,
                task_type="planning",
                created_by=user_id,
                priority=JobPriority.HIGH.value,
                aspect="core",  # Planning always goes to core
                lifecycle_state="planning",
                requested_action="write_code",  # Planning outputs files
                user_role="owner",  # Owner-level permissions for planning
            )

            job_ids.append(job.job_id)
            logger.info(f"Scheduled planning job {job.job_id} for project {project.name}")

        except Exception as e:
            logger.error(f"Error scheduling planning job: {e}", exc_info=True)

        return job_ids

    def _generate_safe_name(self, description: str) -> str:
        """
        Generate a safe project name from description.

        Used ONLY for text-based input where no CHD project_name exists.
        For file-based input, CHD project_name takes absolute precedence.
        """
        import re

        # Extract first meaningful line or phrase
        first_line = description.strip().split('\n')[0][:100]

        # Remove common prefixes that aren't project names
        prefixes_to_remove = [
            r'^build\s+(a\s+)?',
            r'^create\s+(a\s+)?',
            r'^make\s+(a\s+)?',
            r'^develop\s+(a\s+)?',
            r'^i\s+want\s+(a\s+)?',
            r'^i\s+need\s+(a\s+)?',
        ]
        cleaned = first_line.lower()
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, '', cleaned, flags=re.IGNORECASE)

        # Extract words and create slug
        words = re.findall(r'[a-zA-Z0-9]+', cleaned)
        if not words:
            return f"project-{uuid.uuid4().hex[:8]}"

        # Take first 4 meaningful words
        name_words = words[:4]
        name = '-'.join(name_words)

        # Ensure reasonable length
        if len(name) > 50:
            name = name[:50].rsplit('-', 1)[0]

        return name or f"project-{uuid.uuid4().hex[:8]}"


# -----------------------------------------------------------------------------
# Global Service Instance
# -----------------------------------------------------------------------------
_service: Optional[ProjectService] = None


def get_project_service() -> ProjectService:
    """Get the global project service instance."""
    global _service
    if _service is None:
        _service = ProjectService()
    return _service


# -----------------------------------------------------------------------------
# Module-Level Convenience Functions
# -----------------------------------------------------------------------------
async def create_project_from_text(
    description: str,
    user_id: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Create project from text description."""
    return await get_project_service().create_project_from_text(
        description=description,
        user_id=user_id,
        progress_callback=progress_callback,
    )


async def create_project_from_file(
    filename: str,
    file_content: bytes,
    user_id: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Create project from uploaded file."""
    return await get_project_service().create_project_from_file(
        filename=filename,
        file_content=file_content,
        user_id=user_id,
        progress_callback=progress_callback,
    )
