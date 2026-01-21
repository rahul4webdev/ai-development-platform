"""
Task Controller Module

Central orchestrator for the AI-driven autonomous development platform.
Handles task management, project state tracking, and coordination between
the Telegram bot and Claude CLI agent.

Phase 15: Autonomous Lifecycle Engine with Continuous Change Cycles
- Phase 12 features: Multi-aspect projects, IPC, lifecycle management
- Phase 13: Telegram bot with RBAC, health monitoring, project management
- Phase 14: Claude CLI as execution backend, job workspaces, autonomous development
- Phase 15.1: Deterministic lifecycle state machine with event-driven transitions
- Phase 15.2: Continuous change cycles with DEPLOYED -> AWAITING_FEEDBACK loops
- Phase 15.3: Existing project ingestion & adoption engine
- Phase 15.4: Roadmap intelligence with epics and milestones
- Phase 15.5: Claude CLI session-based auth support (API key optional)
- Phase 15.6: Execution Gate Model for lifecycle-based permission enforcement
- Phase 15.7: Real Execution Verification - Claude CLI must execute, not just exist
- Phase 15.8: Runtime Truth Validation - Telegram bot validated via runtime health

Phase 16: Claude Execution & Observability
- Phase 16A: Claude Execution Smoke Test - Real end-to-end job execution proof
- Phase 16B: Platform Dashboard & Observability Layer - Read-only control plane
- Phase 16C: Real Project Execution Stabilization - Project Registry, CHD validation, file upload
- Phase 16E: Project Identity, Fingerprinting & Conflict Resolution Engine
- Phase 16F: Intent Drift, Regression & Contract Enforcement

Phase 17: Runtime Intelligence & Decision Support
- Phase 17A: Runtime Intelligence & Signal Collection Layer - OBSERVATION-ONLY
  * Runtime signals with LOCKED enums (SignalType, Severity)
  * Deterministic severity classification (no ML, no guessing)
  * UNKNOWN severity for missing data (never guess)
  * Append-only JSONL persistence with fsync
  * API endpoints: GET /runtime/signals, /runtime/summary, /runtime/status
  * Telegram commands: /signals, /signals_recent, /runtime_status
  * Dashboard observability health indicators

- Phase 17B: Signal Interpretation & Incident Classification Layer - OBSERVATION-ONLY
  * Incident classification with LOCKED enums (IncidentType, IncidentSeverity, IncidentScope)
  * Deterministic rule-based classification (no ML, no guessing)
  * UNKNOWN for missing data (never assumed)
  * Frozen dataclass Incident (immutable after creation)
  * Append-only JSONL persistence with fsync
  * API endpoints: GET /incidents, /incidents/recent, /incidents/summary, /incidents/{id}
  * Telegram commands: /incidents, /incidents_recent, /incidents_summary
  * Dashboard incident health indicators

- Phase 17C: Recommendation & Human-in-the-Loop Reasoning Layer - ADVISORY ONLY
  * Recommendations with LOCKED enums (RecommendationType, RecommendationSeverity, RecommendationApproval)
  * Rule-based recommendation generation (NO ML, NO probabilistic inference)
  * UNKNOWN propagates from incidents (never guessed)
  * Frozen dataclass Recommendation (immutable after creation)
  * Append-only JSONL persistence with separate approval log
  * API endpoints: GET/POST /recommendations, /recommendations/{id}/approve, /recommendations/{id}/dismiss
  * Telegram commands: /recommendations, /recommendation, /rec_approve, /rec_dismiss
  * Dashboard recommendation health indicators
  * CRITICAL: ADVISORY ONLY - recommendations suggest, NEVER execute
  * Human approval required for any action
  * NO automation, NO lifecycle mutation, NO deployment

Phase 18: Automation Eligibility & Controlled Execution
- Phase 18A: Automation Eligibility Engine - DECISION-ONLY
  * LOCKED enum EligibilityDecision (EXACTLY 3 values):
    - AUTOMATION_FORBIDDEN
    - AUTOMATION_ALLOWED_WITH_APPROVAL
    - AUTOMATION_ALLOWED_LIMITED (RUN_TESTS, UPDATE_DOCS only)
  * Frozen dataclass inputs (all 7 mandatory):
    - RecommendationInput, DriftEvaluationInput, IncidentSummaryInput
    - LifecycleStateInput, ExecutionGateInput, IntentBaselineInput, RuntimeIntelligenceInput
  * Hard-stop rules (ANY match → FORBIDDEN):
    - Drift: HIGH/CRITICAL, architecture/database change, invalid baseline
    - Incidents: CRITICAL severity, SECURITY type, UNKNOWN state
    - Signals: UNKNOWN severity, missing runtime window
    - Environment: PRODUCTION + (MEDIUM drift OR MEDIUM incident)
    - Governance: ExecutionGate denied, audit unavailable
  * 100% deterministic: Same inputs = same output (NO ML, NO heuristics)
  * NO execution, NO scheduling, NO triggering, NO mutation
  * Mandatory audit: Every evaluation emits immutable audit record
  * If audit write fails → AUTOMATION_FORBIDDEN

- Phase 18B: Human Approval Orchestration - DECISION-ONLY, HUMAN-GOVERNED
  * LOCKED enum ApprovalStatus (EXACTLY 3 values):
    - APPROVAL_GRANTED
    - APPROVAL_DENIED
    - APPROVAL_PENDING
  * Frozen dataclass inputs (all mandatory):
    - EligibilityResult, RecommendationInput, LifecycleStateInput
    - ExecutionGateInput, ApprovalRequesterInput, ApprovalStateInput
  * Immediate denial rules (ANY match → DENIED):
    - Missing input, eligibility FORBIDDEN, gate denied
    - Approval expired, self-approval attempt
  * Approval grant rules:
    - No approval required → GRANTED
    - Sufficient approvers → GRANTED
  * Pending state:
    - AWAITING_APPROVAL, AWAITING_CONFIRMATION, AWAITING_DUAL_APPROVAL
  * 100% deterministic: Same inputs = same output
  * NO execution, NO notifications, NO automation
  * Mandatory audit: Every evaluation emits immutable audit record
  * If audit write fails → APPROVAL_DENIED
  * Approval store: Append-only JSONL persistence with fsync
"""

__version__ = "0.18.1"

# Single source of truth for phase metadata
CURRENT_PHASE = "18B"
CURRENT_PHASE_NAME = "Human Approval Orchestration"
CURRENT_PHASE_FULL = f"Phase {CURRENT_PHASE}: {CURRENT_PHASE_NAME}"
