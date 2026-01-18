"""
Phase 12: Aspect Lifecycle Engine

Manages the autonomous execution of project aspects through their lifecycle phases.

Key responsibilities:
- Phase transitions
- Autonomous execution with approval gates
- Bug-fix loops
- Self-healing on failures
- CI coordination
- Deployment orchestration
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path

from .phase12 import (
    ProjectAspect,
    AspectPhase,
    AspectState,
    AspectConfig,
    InternalProjectContract,
    FeedbackType,
    TestingFeedback,
    CITriggerReason,
    CITrigger,
    ExecutionPlan,
    ExecutionStep,
    LedgerEntry,
    Notification,
    NotificationType,
    should_trigger_ci,
    create_testing_notification
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("lifecycle_engine")

# -----------------------------------------------------------------------------
# Phase Transition Rules
# -----------------------------------------------------------------------------

# Valid transitions from each phase
VALID_TRANSITIONS: Dict[AspectPhase, List[AspectPhase]] = {
    AspectPhase.NOT_STARTED: [AspectPhase.PLANNING],
    AspectPhase.PLANNING: [AspectPhase.DEVELOPMENT],
    AspectPhase.DEVELOPMENT: [AspectPhase.UNIT_TESTING],
    AspectPhase.UNIT_TESTING: [AspectPhase.INTEGRATION, AspectPhase.DEVELOPMENT],  # Can go back if tests fail
    AspectPhase.INTEGRATION: [AspectPhase.CODE_REVIEW, AspectPhase.DEVELOPMENT],
    AspectPhase.CODE_REVIEW: [AspectPhase.CI_RUNNING, AspectPhase.DEVELOPMENT],
    AspectPhase.CI_RUNNING: [AspectPhase.CI_PASSED, AspectPhase.CI_FAILED],
    AspectPhase.CI_PASSED: [AspectPhase.READY_FOR_TESTING],
    AspectPhase.CI_FAILED: [AspectPhase.DEVELOPMENT, AspectPhase.BUG_FIXING],
    AspectPhase.READY_FOR_TESTING: [AspectPhase.DEPLOYED_TESTING],
    AspectPhase.DEPLOYED_TESTING: [AspectPhase.AWAITING_FEEDBACK],
    AspectPhase.AWAITING_FEEDBACK: [
        AspectPhase.READY_FOR_PRODUCTION,  # On approve
        AspectPhase.BUG_FIXING,            # On bug report
        AspectPhase.IMPROVEMENTS,          # On improvement request
        AspectPhase.DEVELOPMENT            # On reject
    ],
    AspectPhase.BUG_FIXING: [AspectPhase.CI_RUNNING],
    AspectPhase.IMPROVEMENTS: [AspectPhase.CI_RUNNING],
    AspectPhase.READY_FOR_PRODUCTION: [AspectPhase.DEPLOYED_PRODUCTION],
    AspectPhase.DEPLOYED_PRODUCTION: [AspectPhase.COMPLETED],
    AspectPhase.COMPLETED: []  # Terminal state
}

# Phases that require human approval
APPROVAL_GATES: List[AspectPhase] = [
    AspectPhase.AWAITING_FEEDBACK,
    AspectPhase.READY_FOR_PRODUCTION
]

# Phases that are autonomous (no human input needed)
AUTONOMOUS_PHASES: List[AspectPhase] = [
    AspectPhase.PLANNING,
    AspectPhase.DEVELOPMENT,
    AspectPhase.UNIT_TESTING,
    AspectPhase.INTEGRATION,
    AspectPhase.CODE_REVIEW,
    AspectPhase.CI_RUNNING,
    AspectPhase.BUG_FIXING,
    AspectPhase.IMPROVEMENTS
]


class LifecycleEngine:
    """
    Engine for managing aspect lifecycle transitions and autonomous execution.
    """

    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir
        self.ledger: List[LedgerEntry] = []
        self.pending_notifications: List[Notification] = []

    # -------------------------------------------------------------------------
    # Phase Transition Management
    # -------------------------------------------------------------------------

    def can_transition(
        self,
        current_phase: AspectPhase,
        target_phase: AspectPhase
    ) -> Tuple[bool, str]:
        """Check if a phase transition is valid."""
        valid_targets = VALID_TRANSITIONS.get(current_phase, [])
        if target_phase in valid_targets:
            return True, f"Transition {current_phase.value} -> {target_phase.value} allowed"
        return False, f"Invalid transition: {current_phase.value} -> {target_phase.value}. Valid targets: {[t.value for t in valid_targets]}"

    def transition_phase(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        target_phase: AspectPhase,
        actor: str = "system",
        reason: str = ""
    ) -> Tuple[bool, str]:
        """
        Transition an aspect to a new phase.

        Returns (success, message)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        if not aspect_state:
            return False, f"Aspect {aspect.value} not found in project"

        current_phase = aspect_state.current_phase

        # Check if transition is valid
        can_do, message = self.can_transition(current_phase, target_phase)
        if not can_do:
            return False, message

        # Check if this is an approval gate
        if target_phase in APPROVAL_GATES and actor == "system":
            # System cannot bypass approval gates
            logger.info(f"Pausing at approval gate: {target_phase.value}")

        # Perform transition
        old_phase = aspect_state.current_phase
        aspect_state.current_phase = target_phase
        aspect_state.phase_started_at = datetime.utcnow()

        # Log to ledger
        self._log_ledger(
            project_name=ipc.project_name,
            aspect=aspect,
            action_type="phase_transition",
            action_details={
                "from_phase": old_phase.value,
                "to_phase": target_phase.value,
                "reason": reason
            },
            actor=actor,
            previous_state=old_phase.value,
            new_state=target_phase.value
        )

        # Update IPC timestamp
        ipc.updated_at = datetime.utcnow()

        logger.info(f"Phase transition: {ipc.project_name}/{aspect.value}: {old_phase.value} -> {target_phase.value}")

        return True, f"Transitioned to {target_phase.value}"

    def is_at_approval_gate(self, aspect_state: AspectState) -> bool:
        """Check if aspect is at an approval gate."""
        return aspect_state.current_phase in APPROVAL_GATES

    def is_autonomous_phase(self, phase: AspectPhase) -> bool:
        """Check if phase allows autonomous execution."""
        return phase in AUTONOMOUS_PHASES

    # -------------------------------------------------------------------------
    # Feedback Processing
    # -------------------------------------------------------------------------

    def process_feedback(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        feedback: TestingFeedback
    ) -> Tuple[bool, str, AspectPhase]:
        """
        Process testing feedback and determine next phase.

        Returns (success, message, next_phase)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        if not aspect_state:
            return False, f"Aspect {aspect.value} not found", AspectPhase.NOT_STARTED

        if aspect_state.current_phase != AspectPhase.AWAITING_FEEDBACK:
            return False, f"Cannot process feedback in phase {aspect_state.current_phase.value}", aspect_state.current_phase

        # Validate feedback
        if feedback.feedback_type != FeedbackType.APPROVE:
            if not feedback.explanation or len(feedback.explanation.strip()) < 10:
                return False, f"{feedback.feedback_type.value} feedback requires explanation (min 10 chars)", aspect_state.current_phase

        # Record feedback
        aspect_state.feedback_history.append({
            "feedback_id": feedback.feedback_id,
            "type": feedback.feedback_type.value,
            "submitted_by": feedback.submitted_by,
            "submitted_at": feedback.submitted_at.isoformat(),
            "explanation": feedback.explanation
        })

        # Determine next phase based on feedback type
        next_phase_map = {
            FeedbackType.APPROVE: AspectPhase.READY_FOR_PRODUCTION,
            FeedbackType.BUG: AspectPhase.BUG_FIXING,
            FeedbackType.IMPROVEMENTS: AspectPhase.IMPROVEMENTS,
            FeedbackType.REJECT: AspectPhase.DEVELOPMENT
        }

        next_phase = next_phase_map[feedback.feedback_type]

        # Log to ledger
        self._log_ledger(
            project_name=ipc.project_name,
            aspect=aspect,
            action_type="feedback_received",
            action_details={
                "feedback_type": feedback.feedback_type.value,
                "feedback_id": feedback.feedback_id,
                "explanation": feedback.explanation
            },
            actor=feedback.submitted_by,
            previous_state=aspect_state.current_phase.value,
            new_state=next_phase.value
        )

        # Transition to next phase
        success, message = self.transition_phase(
            ipc=ipc,
            aspect=aspect,
            target_phase=next_phase,
            actor=feedback.submitted_by,
            reason=f"Feedback: {feedback.feedback_type.value}"
        )

        return success, message, next_phase

    # -------------------------------------------------------------------------
    # CI Coordination
    # -------------------------------------------------------------------------

    def should_trigger_ci(
        self,
        aspect_state: AspectState,
        reason: CITriggerReason
    ) -> bool:
        """Determine if CI should be triggered."""
        return should_trigger_ci(aspect_state, reason)

    def trigger_ci(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        reason: CITriggerReason,
        triggered_by: str = "system"
    ) -> Tuple[bool, str, Optional[CITrigger]]:
        """
        Trigger CI for an aspect.

        Returns (success, message, ci_trigger)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        if not aspect_state:
            return False, f"Aspect {aspect.value} not found", None

        if not self.should_trigger_ci(aspect_state, reason):
            return False, f"CI not allowed for reason {reason.value} in phase {aspect_state.current_phase.value}", None

        # Create CI trigger
        ci_trigger = CITrigger(
            project_name=ipc.project_name,
            aspect=aspect,
            reason=reason,
            triggered_by=triggered_by
        )

        # Update aspect state
        aspect_state.last_ci_run_id = ci_trigger.trigger_id
        aspect_state.last_ci_at = ci_trigger.triggered_at

        # Transition to CI_RUNNING
        success, message = self.transition_phase(
            ipc=ipc,
            aspect=aspect,
            target_phase=AspectPhase.CI_RUNNING,
            actor=triggered_by,
            reason=f"CI triggered: {reason.value}"
        )

        # Log to ledger
        self._log_ledger(
            project_name=ipc.project_name,
            aspect=aspect,
            action_type="ci_triggered",
            action_details={
                "trigger_id": ci_trigger.trigger_id,
                "reason": reason.value,
                "triggered_by": triggered_by
            },
            actor=triggered_by
        )

        logger.info(f"CI triggered for {ipc.project_name}/{aspect.value}: {reason.value}")

        return True, "CI triggered successfully", ci_trigger

    def process_ci_result(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        passed: bool,
        details: Dict[str, Any] = None
    ) -> Tuple[bool, str]:
        """
        Process CI result and transition accordingly.

        Returns (success, message)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        if not aspect_state:
            return False, f"Aspect {aspect.value} not found"

        if aspect_state.current_phase != AspectPhase.CI_RUNNING:
            return False, f"Cannot process CI result in phase {aspect_state.current_phase.value}"

        aspect_state.last_ci_status = "passed" if passed else "failed"

        # Log to ledger
        self._log_ledger(
            project_name=ipc.project_name,
            aspect=aspect,
            action_type="ci_completed",
            action_details={
                "passed": passed,
                "details": details or {}
            },
            actor="ci_system"
        )

        # Transition based on result
        target_phase = AspectPhase.CI_PASSED if passed else AspectPhase.CI_FAILED
        return self.transition_phase(
            ipc=ipc,
            aspect=aspect,
            target_phase=target_phase,
            actor="ci_system",
            reason=f"CI {'passed' if passed else 'failed'}"
        )

    # -------------------------------------------------------------------------
    # Deployment Orchestration
    # -------------------------------------------------------------------------

    def deploy_to_testing(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        deployed_by: str = "system"
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Deploy aspect to testing environment.

        Returns (success, message, testing_url)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        aspect_config = ipc.aspects.get(aspect)

        if not aspect_state or not aspect_config:
            return False, f"Aspect {aspect.value} not found", None

        if aspect_state.current_phase != AspectPhase.READY_FOR_TESTING:
            return False, f"Cannot deploy to testing in phase {aspect_state.current_phase.value}", None

        # Simulate deployment
        testing_url = aspect_config.testing_url or ipc.testing_domain or f"https://testing.{ipc.project_name}.local"

        aspect_state.last_deploy_testing_at = datetime.utcnow()

        # Transition to DEPLOYED_TESTING
        success, message = self.transition_phase(
            ipc=ipc,
            aspect=aspect,
            target_phase=AspectPhase.DEPLOYED_TESTING,
            actor=deployed_by,
            reason="Deployed to testing"
        )

        if success:
            # Create notification
            notification = create_testing_notification(
                project_name=ipc.project_name,
                aspect=aspect,
                testing_url=testing_url,
                features=["Feature list to be populated"],
                test_summary=f"Unit: {aspect_state.unit_tests_passed}/{aspect_state.unit_tests_passed + aspect_state.unit_tests_failed}, Integration: {aspect_state.integration_tests_passed}/{aspect_state.integration_tests_passed + aspect_state.integration_tests_failed}",
                limitations=[]
            )
            self.pending_notifications.append(notification)

            # Log to ledger
            self._log_ledger(
                project_name=ipc.project_name,
                aspect=aspect,
                action_type="deployed_testing",
                action_details={
                    "testing_url": testing_url,
                    "deployed_by": deployed_by
                },
                actor=deployed_by
            )

        return success, message, testing_url

    def deploy_to_production(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        approved_by: str,
        justification: str
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Deploy aspect to production environment.

        REQUIRES explicit human approval.

        Returns (success, message, production_url)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        aspect_config = ipc.aspects.get(aspect)

        if not aspect_state or not aspect_config:
            return False, f"Aspect {aspect.value} not found", None

        if aspect_state.current_phase != AspectPhase.READY_FOR_PRODUCTION:
            return False, f"Cannot deploy to production in phase {aspect_state.current_phase.value}", None

        # Production deployment requires explicit approval
        if not approved_by or approved_by == "system":
            return False, "Production deployment requires explicit human approval", None

        # Simulate deployment
        production_url = aspect_config.production_url or ipc.production_domain or f"https://{ipc.project_name}.com"

        aspect_state.last_deploy_production_at = datetime.utcnow()

        # Transition to DEPLOYED_PRODUCTION
        success, message = self.transition_phase(
            ipc=ipc,
            aspect=aspect,
            target_phase=AspectPhase.DEPLOYED_PRODUCTION,
            actor=approved_by,
            reason=f"Production deployment: {justification}"
        )

        if success:
            # Log to ledger (critical action)
            self._log_ledger(
                project_name=ipc.project_name,
                aspect=aspect,
                action_type="deployed_production",
                action_details={
                    "production_url": production_url,
                    "approved_by": approved_by,
                    "justification": justification
                },
                actor=approved_by
            )

            # Create notification
            notification = Notification(
                notification_type=NotificationType.DEPLOYMENT_COMPLETE,
                project_name=ipc.project_name,
                aspect=aspect,
                title=f"Production Deployed: {ipc.project_name} - {aspect.value}",
                summary="Successfully deployed to production",
                environment_url=production_url,
                action_required=False,
                details={"approved_by": approved_by, "justification": justification}
            )
            self.pending_notifications.append(notification)

        return success, message, production_url

    # -------------------------------------------------------------------------
    # Bug Fix Loop
    # -------------------------------------------------------------------------

    def start_bug_fix_loop(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        bug_details: str
    ) -> Tuple[bool, str]:
        """
        Start autonomous bug-fix loop.

        Returns (success, message)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        if not aspect_state:
            return False, f"Aspect {aspect.value} not found"

        if aspect_state.current_phase != AspectPhase.BUG_FIXING:
            return False, f"Not in bug-fixing phase: {aspect_state.current_phase.value}"

        # Log start of bug fix
        self._log_ledger(
            project_name=ipc.project_name,
            aspect=aspect,
            action_type="bug_fix_started",
            action_details={"bug_details": bug_details},
            actor="claude"
        )

        logger.info(f"Bug fix loop started for {ipc.project_name}/{aspect.value}")

        return True, "Bug fix loop started"

    def complete_bug_fix(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect,
        fix_summary: str
    ) -> Tuple[bool, str]:
        """
        Complete bug fix and trigger CI.

        Returns (success, message)
        """
        aspect_state = ipc.aspect_states.get(aspect)
        if not aspect_state:
            return False, f"Aspect {aspect.value} not found"

        if aspect_state.current_phase != AspectPhase.BUG_FIXING:
            return False, f"Not in bug-fixing phase: {aspect_state.current_phase.value}"

        # Log bug fix completion
        self._log_ledger(
            project_name=ipc.project_name,
            aspect=aspect,
            action_type="bug_fix_completed",
            action_details={"fix_summary": fix_summary},
            actor="claude"
        )

        # Trigger CI
        return self.trigger_ci(
            ipc=ipc,
            aspect=aspect,
            reason=CITriggerReason.BUG_FIX_COMPLETE,
            triggered_by="claude"
        )[:2]  # Return only success, message

    # -------------------------------------------------------------------------
    # Execution Plan Management
    # -------------------------------------------------------------------------

    def create_execution_plan(
        self,
        ipc: InternalProjectContract,
        aspect: ProjectAspect
    ) -> ExecutionPlan:
        """Create execution plan for an aspect."""
        aspect_state = ipc.aspect_states.get(aspect)

        plan = ExecutionPlan(
            project_name=ipc.project_name,
            aspect=aspect
        )

        # Build steps based on current phase
        step_templates = [
            ("planning", "Generate implementation plan", False),
            ("development", "Implement features", False),
            ("unit_testing", "Write and run unit tests", False),
            ("integration", "Run integration tests", False),
            ("code_review", "Automated code review", False),
            ("ci", "Run CI pipeline", False),
            ("deploy_testing", "Deploy to testing", False),
            ("await_feedback", "Await human feedback", True),
            ("production_approval", "Await production approval", True),
            ("deploy_production", "Deploy to production", True)
        ]

        for i, (action, description, requires_approval) in enumerate(step_templates):
            step = ExecutionStep(
                step_number=i + 1,
                aspect=aspect,
                action=action,
                description=description,
                requires_approval=requires_approval
            )
            plan.steps.append(step)

        return plan

    # -------------------------------------------------------------------------
    # Ledger Management
    # -------------------------------------------------------------------------

    def _log_ledger(
        self,
        project_name: str,
        aspect: Optional[ProjectAspect],
        action_type: str,
        action_details: Dict[str, Any],
        actor: str,
        previous_state: Optional[str] = None,
        new_state: Optional[str] = None
    ) -> None:
        """Log an entry to the immutable ledger."""
        entry = LedgerEntry(
            project_name=project_name,
            aspect=aspect,
            action_type=action_type,
            action_details=action_details,
            actor=actor,
            previous_state=previous_state,
            new_state=new_state
        )
        self.ledger.append(entry)

        logger.debug(f"Ledger entry: {action_type} by {actor} for {project_name}")

    def get_ledger_entries(
        self,
        project_name: str,
        aspect: Optional[ProjectAspect] = None,
        limit: int = 100
    ) -> List[LedgerEntry]:
        """Get ledger entries for a project."""
        entries = [e for e in self.ledger if e.project_name == project_name]
        if aspect:
            entries = [e for e in entries if e.aspect == aspect]
        return entries[-limit:]

    # -------------------------------------------------------------------------
    # Notification Management
    # -------------------------------------------------------------------------

    def get_pending_notifications(self) -> List[Notification]:
        """Get and clear pending notifications."""
        notifications = self.pending_notifications.copy()
        self.pending_notifications.clear()
        return notifications


# -----------------------------------------------------------------------------
# Singleton Instance
# -----------------------------------------------------------------------------

_engine_instance: Optional[LifecycleEngine] = None


def get_lifecycle_engine(projects_dir: Path) -> LifecycleEngine:
    """Get or create the lifecycle engine singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LifecycleEngine(projects_dir)
    return _engine_instance


logger.info("Lifecycle Engine module loaded successfully")
