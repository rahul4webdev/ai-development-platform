# Current State (Living System State)

This file is updated after every task. It is machine-first, not prose.
The AI agent MUST update this file after completing any task.

---

## Last Updated
- **Timestamp**: 2026-01-21
- **Task**: Phase 18B Human Approval Orchestration - COMPLETED
- **Status**: Complete - DECISION-ONLY orchestrator, LOCKED enum ApprovalStatus (EXACTLY 3 values), frozen dataclass inputs, immediate denial rules, approval grant rules, pending states, 100% deterministic, mandatory audit, NO execution/notifications/automation, approval store with append-only JSONL, 43 tests

---

## Current Phase
```
Phase: PHASE_18B_COMPLETE
Mode: development
Version: 0.18.1
```

## Phase 16A: Claude Execution Smoke Test - VERIFIED

**End-to-end Claude CLI job execution has been proven.**

### Smoke Test Results
- **Job ID**: smoke-test-668030e5
- **Workspace**: /home/aitesting.mybd.in/jobs/job-smoke-test-668030e5
- **Duration**: 9.08 seconds
- **Exit Code**: 0
- **Gate Passed**: ✅
- **README.md Created**: ✅
- **Content Exact Match**: ✅
- **Audit Logged**: ✅

### Key Learnings
- `--permission-mode acceptEdits` required for file writes in automation
- `--dangerously-skip-permissions` cannot be used with root (security restriction)
- Claude CLI execution works end-to-end when properly configured

See [SMOKE_TEST_REPORT.md](SMOKE_TEST_REPORT.md) for full details.

---

## Phase 16B: Platform Dashboard & Observability Layer - VERIFIED

**READ-ONLY control plane for comprehensive system visibility.**

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| DashboardBackend | controller/dashboard_backend.py | Read-only aggregation engine |
| ProjectOverview | controller/dashboard_backend.py | Project state view model |
| ClaudeActivityPanel | controller/dashboard_backend.py | Job activity view model |
| LifecycleTimeline | controller/dashboard_backend.py | Lifecycle history view model |
| DeploymentView | controller/dashboard_backend.py | Deployment status view model |
| AuditEvent | controller/dashboard_backend.py | Audit event view model |
| DashboardSummary | controller/dashboard_backend.py | Top-level summary view model |
| SystemHealth | controller/dashboard_backend.py | Health status enum |

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /dashboard | GET | Get comprehensive dashboard summary |
| /dashboard/projects | GET | List all projects with states |
| /dashboard/project/{name} | GET | Get specific project details |
| /dashboard/jobs | GET | Get Claude job activity |
| /dashboard/lifecycles | GET | List all lifecycles |
| /dashboard/lifecycle/{id} | GET | Get lifecycle timeline |
| /dashboard/audit | GET | Get audit events |

### Telegram Integration

| Component | Description |
|-----------|-------------|
| /dashboard | Enhanced command showing summary, active projects, jobs, health |
| format_dashboard_enhanced() | New formatter for rich dashboard output |
| get_lifecycle_state_emoji() | Emoji mapping for lifecycle states |
| ControllerClient.get_dashboard_summary() | API client method |
| ControllerClient.get_dashboard_jobs() | API client method |
| ControllerClient.get_dashboard_audit() | API client method |

### Hard Constraints (Enforced)

| Constraint | Status |
|------------|--------|
| ❌ No business logic duplication | ✅ Enforced |
| ❌ No lifecycle mutation | ✅ Enforced |
| ❌ No Claude execution | ✅ Enforced |
| ✅ Read-only aggregation only | ✅ Verified |
| ✅ Deterministic output | ✅ Verified |
| ✅ Zero hallucinated data | ✅ Verified |

### Test Coverage (tests/test_dashboard_backend.py)

| Test Class | Coverage |
|------------|----------|
| TestDashboardBackendImports | Module imports, function existence |
| TestSystemHealthEnum | Health state enum values |
| TestDataModels | All data model field validation |
| TestDashboardBackendReadOnly | No write methods, read-only naming |
| TestDashboardBackendDataConsistency | Summary creation, deterministic counts |
| TestDashboardBackendNoSideEffects | Multiple call safety |
| TestDashboardBackendZeroHallucination | No fake data, valid timestamps |
| TestDashboardAPIEndpoints | Main imports, module functions |
| TestTelegramDashboardIntegration | Format functions, client methods |

**All 21 tests passing.**

---

## Phase 16C: Real Project Execution Stabilization - VERIFIED

**Fixed all blockers preventing real project creation and dashboard visibility.**

### Root Cause Identified
Projects were being created as IPC files in `projects/` directory, but dashboard read from lifecycle files in `LIFECYCLE_STATE_DIR`. This caused "No projects found" despite successful creation.

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| ProjectRegistry | controller/project_registry.py | Canonical project storage with persistence |
| Project Model | controller/project_registry.py | Unified project data model |
| CHDValidator | controller/chd_validator.py | Requirements validation before Claude execution |
| FileContentValidator | controller/chd_validator.py | File upload validation (.md, .txt) |
| ProjectService | controller/project_service.py | Unified project creation flow |
| Progress Callbacks | controller/project_service.py | Real-time creation progress |

### Fixes Applied

| Issue | Solution |
|-------|----------|
| /dashboard shows "No projects found" | Dashboard now reads from Project Registry first |
| 500 errors on project creation | All exceptions caught and returned as structured errors |
| No file upload support | Added document handler in Telegram bot |
| Missing validation | CHD validation layer catches invalid requirements early |
| No progress feedback | Progress callback system shows real-time status |

### File Upload Support (Telegram)

- Supports: `.md`, `.txt` files
- Max size: 100KB
- Automatic requirements parsing
- Progress feedback during creation

### Validation Rules (CHD Layer)

| Rule | Description |
|------|-------------|
| Aspect Required | At least one backend OR frontend must be defined |
| No Auto-Deploy | `auto-deploy prod` pattern blocked |
| No Skip Tests | `skip tests` pattern blocked |
| Min Content | At least 10 characters required |
| Database Hint | Warns if persistence mentioned but no DB specified |

### API Changes

| Endpoint | Change |
|----------|--------|
| POST /v2/project/create | Now validates with CHD layer and registers in Project Registry |
| GET /dashboard | Now reads from Project Registry + Lifecycle files |
| GET /dashboard/projects | Now reads from Project Registry + Lifecycle files |

### Telegram Changes

| Feature | Description |
|---------|-------------|
| File Handler | New `handle_document()` for `.md`/`.txt` uploads |
| Progress Feedback | Real-time step-by-step progress messages |
| Error Messages | Structured errors with suggestions |
| create_project_from_file | New function for file-based creation |

### Test Coverage (tests/test_phase16c.py)

| Test Class | Tests |
|------------|-------|
| TestProjectRegistry | 7 tests |
| TestCHDValidator | 7 tests |
| TestProjectService | 1 test |
| TestDashboardIntegration | 3 tests |
| TestTelegramIntegration | 3 tests |
| TestErrorHandling | 2 tests |

**All 24 tests passing.**

---

## Phase 16E: Project Identity, Fingerprinting & Conflict Resolution - VERIFIED

**Enterprise-grade system to prevent duplicate projects, ambiguous intent, and silent overwrites.**

### Problem Solved

| Issue | Impact | Solution |
|-------|--------|----------|
| Duplicate Projects | Same project created multiple times with different names | Fingerprint-based deduplication |
| Ambiguous Intent | "Build CRM" vs "Build CRM with API" - unclear if same project | Normalized intent extraction |
| Silent Overwrites | Creating similar project overwrites work | Conflict detection with user choices |
| No Version History | No tracking of project evolution | Parent-child version linking |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| ProjectIdentity | controller/project_identity.py | Frozen/immutable project identity |
| NormalizedIntent | controller/project_identity.py | Semantic project representation |
| IntentExtractor | controller/project_identity.py | Extracts purpose, modules, DB, architecture |
| FingerprintGenerator | controller/project_identity.py | SHA-256 deterministic fingerprinting |
| ProjectIdentityManager | controller/project_identity.py | Identity creation and comparison |
| DecisionResult | controller/project_decision_engine.py | Decision with explanation and confidence |
| ProjectDecisionEngine | controller/project_decision_engine.py | Locked decision matrix |
| ArchitectureChangeDetector | controller/project_decision_engine.py | Breaking change detection |
| ScopeChangeDetector | controller/project_decision_engine.py | Module change detection |

### Fingerprinting Rules (Deterministic)

Fingerprint is SHA-256 hash of canonicalized:
- `purpose_keywords` (sorted, lowercase)
- `functional_modules` (sorted, lowercase)
- `domain_topology` (sorted, lowercase)
- `database_type` (string)
- `architecture_class` (string)
- `target_users` (sorted, lowercase)

**NOT included**: repo URLs, file paths, timestamps, secrets, user IDs.

### Decision Matrix (Locked)

| Condition | Decision | User Confirmation |
|-----------|----------|-------------------|
| No existing projects | NEW_PROJECT | No |
| Exact fingerprint match | CONFLICT_DETECTED | **Yes** |
| Similarity >= 85% | CONFLICT_DETECTED | **Yes** |
| Architecture change | NEW_VERSION | **Yes** |
| Scope change only | CHANGE_MODE | **Yes** |
| Similarity < 50% | NEW_PROJECT | No |

### Similarity Scoring

| Component | Weight |
|-----------|--------|
| Purpose Keywords | 0.30 |
| Functional Modules | 0.30 |
| Domain Topology | 0.15 |
| Database Type | 0.10 |
| Architecture Class | 0.10 |
| Target Users | 0.05 |

### Project Registry v2 Changes

| New Field | Type | Purpose |
|-----------|------|---------|
| fingerprint | str | SHA-256 identity fingerprint |
| normalized_intent | dict | Semantic intent representation |
| version | str | v1, v2, v3... |
| change_history | list | Change records |
| parent_project_id | str | Link to parent version |

| New Method | Purpose |
|------------|---------|
| get_all_identities() | Return all (ProjectIdentity, name) tuples |
| get_project_by_fingerprint() | Lookup by fingerprint |
| find_similar_projects() | Find projects above similarity threshold |
| create_project_with_identity() | Create with fingerprint |
| create_project_version() | Create new version of existing project |
| add_change_record() | Track change history |
| get_dashboard_projects_grouped() | Group by fingerprint |

### Telegram Bot Conflict UX

When conflict detected, bot shows:
```
⚠️ Conflict Detected

A similar project already exists:
Existing: `crm-saas-v1`

Similarity: 87%
Reason: High similarity in purpose and modules

Choose an action:
[1️⃣ Improve existing project]
[2️⃣ Add new module]
[3️⃣ Create new version]
[4️⃣ Cancel]
```

### Dashboard Identity Grouping

| Method | Purpose |
|--------|---------|
| get_projects_grouped_by_identity() | Group projects by fingerprint family |
| ProjectOverview.fingerprint | New field in project overview |
| ProjectOverview.version | New field in project overview |
| ProjectOverview.parent_project_id | New field in project overview |

### API Changes

| Endpoint | Change |
|----------|--------|
| POST /v2/project/create | Now runs decision engine, returns conflict if detected |
| GET /dashboard | Now includes fingerprint, version, parent in projects |

### Test Coverage (tests/test_phase16e.py)

| Test Class | Tests |
|------------|-------|
| TestProjectIdentityEngine | 8 tests |
| TestProjectDecisionEngine | 9 tests |
| TestProjectRegistryV2 | 6 tests |
| TestPhase16EIntegration | 4 tests |
| TestEdgeCases | 3 tests |

**30 tests total.**

---

## Phase 16F: Intent Drift, Regression & Contract Enforcement - VERIFIED

**Prevents silent project evolution through immutable baselines, drift detection, and contract enforcement.**

### Problem Solved

| Issue | Impact | Solution |
|-------|--------|----------|
| Silent Project Evolution | Project scope creeps without human awareness | Drift detection on every execution |
| Architecture Changes | Breaking changes introduced silently | Hard block on architecture/DB changes |
| Purpose Creep | Project expands beyond original intent | Purpose drift tracking and blocking |
| No Accountability | Changes happen without approval | Immutable audit trail, rebaseline workflow |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| IntentBaseline | controller/intent_baseline.py | Immutable baseline snapshot |
| RebaselineRequest | controller/intent_baseline.py | Workflow for updating baselines |
| IntentBaselineManager | controller/intent_baseline.py | Baseline lifecycle management |
| DriftLevel | controller/intent_drift_engine.py | NONE/LOW/MEDIUM/HIGH/CRITICAL classification |
| DriftDimension | controller/intent_drift_engine.py | 6 drift axes (purpose, module, arch, db, surface, nonfunc) |
| IntentDriftEngine | controller/intent_drift_engine.py | Deterministic drift analysis |
| ContractType | controller/intent_contract.py | SOFT/CONFIRMATION_REQUIRED/HARD_BLOCK |
| ContractViolation | controller/intent_contract.py | Violation record with severity |
| IntentContractEnforcer | controller/intent_contract.py | Contract evaluation and enforcement |
| PendingConfirmation | controller/intent_contract.py | Confirmation workflow for MEDIUM drift |

### Drift Classification Levels (Locked)

| Level | Score Range | Action |
|-------|-------------|--------|
| NONE | 0-5 | Allow silently |
| LOW | 6-20 | Log warning, proceed |
| MEDIUM | 21-50 | Require user confirmation |
| HIGH | 51-80 | Hard block until approved |
| CRITICAL | 81-100 | Freeze project |

### Drift Dimensions (All 6 Required)

| Dimension | Weight | Breaking Triggers |
|-----------|--------|-------------------|
| Purpose | 0.30 | Purpose keywords change significantly |
| Architecture | 0.25 | monolith→microservices, api_only→fullstack |
| Module | 0.20 | Major module addition/removal |
| Database | 0.10 | postgresql→mongodb, none→postgresql |
| Surface Area | 0.10 | New domains/APIs added |
| Non-Functional | 0.05 | Target users change |

### Contract Types

| Type | Behavior |
|------|----------|
| SOFT | Claude may proceed, warning logged |
| CONFIRMATION_REQUIRED | Claude must stop and ask user |
| HARD_BLOCK | Claude is forbidden from proceeding |

### Violation Rules

| Violation Type | Default Contract |
|----------------|------------------|
| ARCHITECTURE_CHANGE | HARD_BLOCK |
| DATABASE_CHANGE | HARD_BLOCK |
| PURPOSE_EXPANSION | CONFIRMATION_REQUIRED |
| DOMAIN_ADDITION | CONFIRMATION_REQUIRED |
| MODULE_ADDITION | SOFT |
| USER_TYPE_CHANGE | SOFT |
| DRIFT_THRESHOLD_EXCEEDED | HARD_BLOCK |

### ExecutionGate Integration

| Enhancement | Description |
|-------------|-------------|
| drift_checked | Boolean indicating drift analysis was performed |
| drift_blocks_execution | True if drift BLOCKS execution |
| drift_requires_confirmation | True if drift REQUIRES confirmation |
| drift_evaluation | Full contract evaluation result |
| Phase 16F constraints | Added to execution constraints |

### Rebaseline Workflow

| Step | Description |
|------|-------------|
| 1. Drift detected | ExecutionGate blocks with HIGH/CRITICAL drift |
| 2. Request rebaseline | User requests baseline update with justification |
| 3. Review request | Admin reviews proposed new baseline |
| 4. Approve/Reject | Admin approves (creates new baseline) or rejects |
| 5. Old superseded | Previous baseline marked SUPERSEDED |
| 6. Execution allowed | New baseline becomes active |

### Audit Trail

| File | Purpose |
|------|---------|
| /home/aitesting.mybd.in/jobs/intent_baselines/baseline_audit.log | Baseline operations |
| /home/aitesting.mybd.in/jobs/intent_contracts/contract_audit.log | Contract evaluations |

### Claude Restrictions (Enforced)

- NEVER change architecture class without approval
- NEVER change database type without approval
- NEVER expand project purpose beyond baseline
- NEVER add new production domains silently
- NEVER bypass drift checks during execution

### Test Coverage (tests/test_phase16f.py)

| Test Class | Tests |
|------------|-------|
| TestIntentBaselineManager | 8 tests |
| TestIntentDriftEngine | 10 tests |
| TestIntentContractEnforcer | 10 tests |
| TestIntegration | 5 tests |

**33 tests total.**

---

## Phase 17A: Runtime Intelligence & Signal Collection Layer - VERIFIED

**OBSERVATION-ONLY system for collecting, classifying, and persisting runtime signals.**

### Critical Constraints (Enforced)

| Constraint | Status |
|------------|--------|
| ❌ No lifecycle transitions | ✅ Enforced |
| ❌ No deployment actions | ✅ Enforced |
| ❌ No intent mutation | ✅ Enforced |
| ❌ No auto-healing | ✅ Enforced |
| ✅ Read-only aggregation only | ✅ Verified |
| ✅ Deterministic classification | ✅ Verified |
| ✅ UNKNOWN when data missing | ✅ Verified |
| ✅ Append-only persistence | ✅ Verified |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| SignalType Enum | controller/runtime_intelligence.py | 8 LOCKED signal types |
| Severity Enum | controller/runtime_intelligence.py | 5 LOCKED severity levels (includes UNKNOWN) |
| RuntimeSignal | controller/runtime_intelligence.py | Immutable (frozen) signal dataclass |
| SignalCollector | controller/runtime_intelligence.py | Read-only signal collection from system, workers, lifecycle |
| SignalPersister | controller/runtime_intelligence.py | Append-only JSONL persistence with fsync |
| RuntimeIntelligenceEngine | controller/runtime_intelligence.py | Polling engine for signal collection |
| SignalSummary | controller/runtime_intelligence.py | Read-only aggregation model |

### Signal Types (LOCKED)

| Type | Description |
|------|-------------|
| SYSTEM_RESOURCE | CPU, memory, disk metrics |
| WORKER_QUEUE | Job queue saturation, worker status |
| JOB_FAILURE | Claude job failures |
| TEST_REGRESSION | Test failure patterns |
| DEPLOYMENT_FAILURE | Deployment failures |
| DRIFT_WARNING | Intent drift detected |
| HUMAN_OVERRIDE | Human intervention signals |
| CONFIG_ANOMALY | Configuration anomalies |

### Severity Levels (LOCKED)

| Level | Description | Action |
|-------|-------------|--------|
| INFO | Normal operation | Log only |
| WARNING | Potential issue | Monitor |
| DEGRADED | Service degradation | Alert |
| CRITICAL | Service impairment | Immediate attention |
| UNKNOWN | Data unavailable | Never guess, always explicit |

### Deterministic Classification Thresholds

| Metric | INFO | WARNING | DEGRADED | CRITICAL |
|--------|------|---------|----------|----------|
| CPU % | <70 | 70-85 | 85-95 | ≥95 |
| Memory % | <70 | 70-85 | 85-95 | ≥95 |
| Disk % | <75 | 75-85 | 85-95 | ≥95 |
| Queue Jobs | <3 | 3-5 | 5-10 | ≥10 |
| Failures/hr | 0 | 1-2 | 3-4 | ≥5 |

### API Endpoints (READ-ONLY)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /runtime/signals | GET | Get signals with filters (project, severity, type, since) |
| /runtime/summary | GET | Get signal summary for time window |
| /runtime/status | GET | Get collection engine status |
| /runtime/poll | POST | Trigger manual poll cycle |

### Telegram Commands (READ-ONLY)

| Command | Purpose |
|---------|---------|
| /signals [project] | View signals summary with severity breakdown |
| /signals_recent [hours] [limit] | View recent signals list |
| /runtime_status | View collection status |

### Dashboard Integration

| Component | Description |
|-----------|-------------|
| ObservabilityHealth | New dataclass for observability status |
| DashboardSummary.observability | New field with signal counts |
| _get_observability_health() | Method to aggregate signal health |
| _determine_system_health() | Updated to consider observability |

### Persistence

| File | Format | Purpose |
|------|--------|---------|
| signals.jsonl | JSONL (append-only) | Signal storage with fsync |
| poll_audit.log | JSONL (append-only) | Poll cycle audit trail |

### Safety Guarantees

- **IMMUTABLE_SIGNALS**: RuntimeSignal is frozen, cannot be modified after creation
- **APPEND_ONLY_PERSISTENCE**: Signals are never deleted or modified in storage
- **FSYNC_DURABILITY**: Every persist operation calls fsync for crash safety
- **UNKNOWN_NOT_GUESSED**: Missing data ALWAYS produces UNKNOWN severity, never guessed
- **NO_MUTATION_METHODS**: No lifecycle transition, deployment, or fix methods exist
- **DETERMINISTIC_CLASSIFICATION**: Same input always produces same severity output
- **CONFIDENCE_INDICATOR**: Every signal has 0.0-1.0 confidence score

### Test Coverage (tests/test_phase17a_runtime_intelligence.py)

| Test Class | Tests |
|------------|-------|
| TestSignalTypeEnum | 3 tests (LOCKED values, count) |
| TestSeverityEnum | 3 tests (LOCKED values, UNKNOWN exists) |
| TestSignalSourceEnum | 1 test (source values) |
| TestRuntimeSignal | 9 tests (creation, immutability, validation, serialization) |
| TestSignalCollector | 9 tests (collection, classification, UNKNOWN on failure) |
| TestSignalPersister | 12 tests (persist, append-only, filters, summary) |
| TestRuntimeIntelligenceEngine | 11 tests (polling, status, start/stop) |
| TestObservationOnlyBehavior | 5 tests (no mutation methods) |
| TestUnknownSeverityBehavior | 3 tests (UNKNOWN on missing data) |
| TestSignalSummary | 3 tests (summary generation, observability status) |

**58 tests total.**

---

## Phase 17B: Signal Interpretation & Incident Classification Layer - VERIFIED

**OBSERVATION-ONLY system for correlating signals and classifying incidents.**

### Critical Constraints (Enforced)

| Constraint | Status |
|------------|--------|
| ❌ No lifecycle transitions | ✅ Enforced |
| ❌ No deployment actions | ✅ Enforced |
| ❌ No Claude execution | ✅ Enforced |
| ❌ No alerts/notifications | ✅ Enforced |
| ❌ No recommendations | ✅ Enforced |
| ✅ Read-only aggregation only | ✅ Verified |
| ✅ Deterministic classification | ✅ Verified |
| ✅ UNKNOWN when data missing | ✅ Verified |
| ✅ Append-only persistence | ✅ Verified |
| ✅ Immutable incidents | ✅ Verified |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| IncidentType Enum | controller/incident_model.py | 7 LOCKED incident types |
| IncidentSeverity Enum | controller/incident_model.py | 6 LOCKED severity levels (includes UNKNOWN) |
| IncidentScope Enum | controller/incident_model.py | 5 LOCKED scope values |
| IncidentState Enum | controller/incident_model.py | 3 LOCKED state values |
| Incident | controller/incident_model.py | Frozen (immutable) incident dataclass |
| IncidentSummary | controller/incident_model.py | Read-only aggregation model |
| ClassificationRule | controller/incident_model.py | Frozen classification rule |
| SignalCorrelationEngine | controller/incident_engine.py | Signal correlation within time windows |
| IncidentClassifier | controller/incident_engine.py | Rule-based incident classification |
| IncidentClassificationEngine | controller/incident_engine.py | Main engine combining correlation + classification |
| IncidentStore | controller/incident_store.py | Append-only JSONL persistence |

### Incident Types (LOCKED)

| Type | Description |
|------|-------------|
| PERFORMANCE | System slowdowns, high latency, resource contention |
| RELIABILITY | Service failures, job failures, test failures |
| SECURITY | Security-related signals, gate denials |
| GOVERNANCE | Drift violations, contract breaches, human overrides |
| RESOURCE | Resource exhaustion, disk/memory/CPU issues |
| CONFIGURATION | Config anomalies, misconfigurations |
| UNKNOWN | Cannot classify - data insufficient |

### Incident Severity (LOCKED)

| Level | Description |
|-------|-------------|
| INFO | Informational, no impact |
| LOW | Minor impact, no immediate action needed |
| MEDIUM | Moderate impact, should be investigated |
| HIGH | Significant impact, requires attention |
| CRITICAL | Severe impact, urgent attention needed |
| UNKNOWN | MANDATORY when data is insufficient |

### Classification Rules (Deterministic)

| Rule | Signal Types | Incident Type | Scope |
|------|--------------|---------------|-------|
| rule-resource-001 | SYSTEM_RESOURCE | RESOURCE | system |
| rule-reliability-001 | JOB_FAILURE, TEST_REGRESSION, DEPLOYMENT_FAILURE | RELIABILITY | from_signal |
| rule-governance-001 | DRIFT_WARNING, HUMAN_OVERRIDE | GOVERNANCE | from_signal |
| rule-config-001 | CONFIG_ANOMALY | CONFIGURATION | from_signal |

### API Endpoints (READ-ONLY)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /incidents | GET | Get incidents with filters (project, type, severity, since) |
| /incidents/recent | GET | Get recent incidents |
| /incidents/summary | GET | Get incident summary for time window |
| /incidents/{id} | GET | Get specific incident by ID |
| /incidents/classify | POST | Classify signals (no persistence) |

### Telegram Commands (READ-ONLY)

| Command | Purpose |
|---------|---------|
| /incidents [project] | View incidents summary with severity/type breakdown |
| /incidents_recent [hours] [limit] | View recent incidents list |
| /incidents_summary | View detailed incident statistics |

### Dashboard Integration

| Component | Description |
|-----------|-------------|
| IncidentHealth | New dataclass for incident status |
| DashboardSummary.incidents | New field with incident counts |
| _get_incident_health() | Method to aggregate incident health |
| _determine_system_health() | Updated to consider incidents |

### Persistence

| File | Format | Purpose |
|------|--------|---------|
| incidents.jsonl | JSONL (append-only) | Incident storage with fsync |
| incident_audit.log | JSONL (append-only) | Persist audit trail |

### Safety Guarantees

- **IMMUTABLE_INCIDENTS**: Incident is frozen, cannot be modified after creation
- **APPEND_ONLY_PERSISTENCE**: Incidents are never deleted or modified in storage
- **FSYNC_DURABILITY**: Every persist operation calls fsync for crash safety
- **UNKNOWN_NOT_GUESSED**: Missing data ALWAYS produces UNKNOWN, never guessed
- **NO_MUTATION_METHODS**: No delete, update, edit, resolve, close methods exist
- **DETERMINISTIC_CLASSIFICATION**: Same input always produces same incident
- **TUPLE_SIGNAL_IDS**: source_signal_ids is tuple (immutable), not list
- **CONFIDENCE_INDICATOR**: Every incident has 0.0-1.0 confidence score

### Test Coverage (tests/test_phase17b_incident_classification.py)

| Test Class | Tests |
|------------|-------|
| TestIncidentImmutability | 10 tests (frozen dataclass, tuple validation) |
| TestClassificationDeterminism | 10 tests (same input = same output) |
| TestUnknownHandling | 8 tests (UNKNOWN on missing data) |
| TestAppendOnlyStore | 8 tests (no delete, no update, append-only) |
| TestNoSideEffects | 6 tests (no lifecycle, no execute methods) |
| TestEnumValidation | 5 tests (invalid values rejected) |
| TestSerialization | 5 tests (to_dict/from_dict roundtrip) |

**52 tests total.**

---

## Phase 17C: Recommendation & Human-in-the-Loop Reasoning Layer - VERIFIED

**ADVISORY-ONLY system for generating recommendations from incidents with human approval workflow.**

### Critical Constraints (Enforced)

| Constraint | Status |
|------------|--------|
| ❌ No automatic execution | ✅ Enforced |
| ❌ No lifecycle mutation | ✅ Enforced |
| ❌ No deployment actions | ✅ Enforced |
| ❌ No Claude execution | ✅ Enforced |
| ❌ No alerts/notifications | ✅ Enforced |
| ✅ Advisory suggestions only | ✅ Verified |
| ✅ Human approval required | ✅ Verified |
| ✅ Deterministic rule-based | ✅ Verified |
| ✅ UNKNOWN propagates from incidents | ✅ Verified |
| ✅ Append-only persistence | ✅ Verified |
| ✅ Separate approval log | ✅ Verified |
| ✅ Immutable recommendations | ✅ Verified |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| RecommendationType Enum | controller/recommendation_model.py | 6 LOCKED recommendation types |
| RecommendationSeverity Enum | controller/recommendation_model.py | 6 LOCKED severity levels (includes UNKNOWN) |
| RecommendationApproval Enum | controller/recommendation_model.py | 3 LOCKED approval requirements |
| RecommendationStatus Enum | controller/recommendation_model.py | 5 LOCKED status values |
| Recommendation | controller/recommendation_model.py | Frozen (immutable) recommendation dataclass |
| ApprovalRecord | controller/recommendation_model.py | Frozen approval/dismissal record |
| RecommendationSummary | controller/recommendation_model.py | Read-only aggregation model |
| RecommendationRule | controller/recommendation_model.py | Frozen classification rule |
| RecommendationGenerator | controller/recommendation_engine.py | Single incident -> recommendation |
| RecommendationEngine | controller/recommendation_engine.py | Main engine for batch generation |
| RecommendationStore | controller/recommendation_store.py | Append-only JSONL persistence |

### Recommendation Types (LOCKED)

| Type | Description |
|------|-------------|
| INVESTIGATE | Needs human investigation |
| MITIGATE | Suggests mitigation steps |
| IMPROVE | Suggests improvement actions |
| REFACTOR | Suggests code/config refactoring |
| DOCUMENT | Suggests documentation updates |
| NO_ACTION | No action recommended (informational) |

### Recommendation Severity (LOCKED)

| Level | Description |
|-------|-------------|
| INFO | Informational, low priority |
| LOW | Minor priority |
| MEDIUM | Moderate priority |
| HIGH | High priority, needs attention |
| CRITICAL | Critical priority, urgent |
| UNKNOWN | MANDATORY when data is insufficient |

### Approval Requirements (LOCKED)

| Level | Description |
|-------|-------------|
| NONE_REQUIRED | Info only, no approval needed |
| CONFIRMATION_REQUIRED | Simple confirmation |
| EXPLICIT_APPROVAL_REQUIRED | Detailed approval with reason |

### API Endpoints (ADVISORY-ONLY)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /recommendations | GET | Get recommendations with filters |
| /recommendations/recent | GET | Get recent recommendations |
| /recommendations/summary | GET | Get recommendation summary |
| /recommendations/{id} | GET | Get specific recommendation |
| /recommendations/{id}/approve | POST | Approve recommendation (creates ApprovalRecord) |
| /recommendations/{id}/dismiss | POST | Dismiss recommendation (creates ApprovalRecord) |
| /recommendations/generate | POST | Generate recommendations from incidents |

### Telegram Commands (ADVISORY-ONLY)

| Command | Purpose |
|---------|---------|
| /recommendations [status] [limit] | View recommendations list |
| /recommendation <id> | View specific recommendation details |
| /rec_approve <id> [reason] | Approve recommendation (ADVISORY-ONLY) |
| /rec_dismiss <id> [reason] | Dismiss recommendation |

### Dashboard Integration

| Component | Description |
|-----------|-------------|
| RecommendationHealth | New dataclass for recommendation status |
| DashboardSummary.recommendations | New field with recommendation counts |
| _get_recommendation_health() | Method to aggregate recommendation health |

### Persistence

| File | Format | Purpose |
|------|--------|---------|
| recommendations.jsonl | JSONL (append-only) | Recommendation storage with fsync |
| approvals.jsonl | JSONL (append-only) | Separate approval/dismissal log |

### Safety Guarantees

- **IMMUTABLE_RECOMMENDATIONS**: Recommendation is frozen, cannot be modified after creation
- **APPEND_ONLY_PERSISTENCE**: Recommendations are never deleted or modified in storage
- **SEPARATE_APPROVAL_LOG**: Approvals create NEW records, don't modify originals
- **FSYNC_DURABILITY**: Every persist operation calls fsync for crash safety
- **UNKNOWN_PROPAGATES**: UNKNOWN incidents produce UNKNOWN recommendations
- **NO_EXECUTE_METHODS**: No execute, apply, trigger, run methods exist
- **DETERMINISTIC_GENERATION**: Same incident always produces same recommendation type
- **TUPLE_IMMUTABLE**: source_incident_ids and suggested_actions are tuples (immutable)
- **ADVISORY_ONLY**: Approval does NOT trigger any automatic action

### Test Coverage (tests/test_phase17c_recommendations.py)

| Test Class | Tests |
|------------|-------|
| TestRecommendationImmutability | 8 tests (frozen dataclass, tuple validation) |
| TestApprovalRecordImmutability | 4 tests (frozen record, action validation) |
| TestEnumValidation | 8 tests (LOCKED enum values, invalid rejected) |
| TestDeterminism | 5 tests (same input = same output) |
| TestUnknownHandling | 5 tests (UNKNOWN propagation) |
| TestAppendOnlyStore | 6 tests (no delete, no update, separate approvals) |
| TestAdvisoryOnly | 5 tests (no execute methods, approval doesn't trigger) |
| TestSerialization | 4 tests (to_dict/from_dict roundtrip) |
| TestConfidenceValidation | 3 tests (0.0-1.0 range enforced) |
| TestIncidentCountValidation | 2 tests (non-negative count) |

**50 tests total.**

---

## Phase 18A: Automation Eligibility Engine - VERIFIED

**DECISION-ONLY engine that answers: "Is automation allowed in this situation?"**

### Critical Constraints (Enforced)

| Constraint | Status |
|------------|--------|
| ❌ No execution | ✅ Enforced |
| ❌ No scheduling | ✅ Enforced |
| ❌ No triggering | ✅ Enforced |
| ❌ No mutation | ✅ Enforced |
| ❌ No recommendations | ✅ Enforced |
| ❌ No planning | ✅ Enforced |
| ✅ Decision-only | ✅ Verified |
| ✅ 100% deterministic | ✅ Verified |
| ✅ Mandatory audit | ✅ Verified |
| ✅ All inputs mandatory | ✅ Verified |

### Eligibility Decision Enum (LOCKED - EXACTLY 3 VALUES)

| Decision | Meaning |
|----------|---------|
| AUTOMATION_FORBIDDEN | Automation must NEVER proceed |
| AUTOMATION_ALLOWED_WITH_APPROVAL | Requires explicit human approval (Phase 18C) |
| AUTOMATION_ALLOWED_LIMITED | ONLY RUN_TESTS and UPDATE_DOCS (no code writes) |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| EligibilityDecision Enum | controller/automation_eligibility.py | 3 LOCKED decision values |
| LimitedAction Enum | controller/automation_eligibility.py | 2 allowed limited actions |
| HardStopRule Enum | controller/automation_eligibility.py | All hard-stop rule IDs |
| RecommendationInput | controller/automation_eligibility.py | Frozen recommendation snapshot |
| DriftEvaluationInput | controller/automation_eligibility.py | Frozen drift snapshot |
| IncidentSummaryInput | controller/automation_eligibility.py | Frozen incident snapshot |
| LifecycleStateInput | controller/automation_eligibility.py | Frozen lifecycle snapshot |
| ExecutionGateInput | controller/automation_eligibility.py | Frozen gate snapshot |
| IntentBaselineInput | controller/automation_eligibility.py | Frozen baseline snapshot |
| RuntimeIntelligenceInput | controller/automation_eligibility.py | Frozen runtime snapshot |
| EligibilityInput | controller/automation_eligibility.py | Combined input (all 7 mandatory) |
| EligibilityResult | controller/automation_eligibility.py | Frozen result with decision |
| EligibilityAuditRecord | controller/automation_eligibility.py | Frozen audit record |
| AutomationEligibilityEngine | controller/automation_eligibility.py | Pure decision logic |

### Required Inputs (ALL MANDATORY)

| Input | Source | If Missing |
|-------|--------|------------|
| Recommendation | Phase 17C | → FORBIDDEN |
| Drift Evaluation | Phase 16F | → FORBIDDEN |
| Incident Summary | Phase 17B | → FORBIDDEN |
| Lifecycle State | Lifecycle Engine | → FORBIDDEN |
| Execution Gate | Phase 15.6 | → FORBIDDEN |
| Environment | Context | → FORBIDDEN |
| Intent Baseline | IntentBaselineManager | → FORBIDDEN |
| Runtime Intelligence | Phase 17A | → FORBIDDEN |

### Hard-Stop Rules (ANY Match → FORBIDDEN)

| Category | Rules |
|----------|-------|
| Drift | HIGH/CRITICAL level, architecture change, database change |
| Baseline | Missing or invalid intent baseline |
| Incidents | CRITICAL severity, SECURITY type, UNKNOWN state |
| Signals | UNKNOWN severity, missing runtime window |
| Environment | PRODUCTION + (MEDIUM drift OR MEDIUM incident) |
| Governance | ExecutionGate denied, audit unavailable |

### Safety Guarantees

- **DECISION_ONLY**: Returns decision, NEVER executes anything
- **NO_SIDE_EFFECTS**: No disk writes (except audit), no state changes
- **DETERMINISTIC**: Same inputs ALWAYS produce same output
- **ALL_INPUTS_MANDATORY**: Missing any input → FORBIDDEN
- **HARD_STOP_PRIORITY**: Hard-stop rules evaluated FIRST
- **AUDIT_REQUIRED**: Every evaluation emits immutable audit record
- **AUDIT_FAILURE_FORBIDDEN**: If audit write fails → FORBIDDEN
- **NO_BYPASS**: Cannot be bypassed by Claude, workers, jobs, APIs

### Audit Record Structure

| Field | Description |
|-------|-------------|
| audit_id | Unique identifier |
| input_hash | SHA-256 hash of all inputs |
| decision | EligibilityDecision value |
| matched_rules | List of triggered HardStopRule values |
| timestamp | ISO timestamp |
| engine_version | Engine version (18A.1.0) |
| environment | TEST or PRODUCTION |
| project_id | Project identifier if available |

### Test Coverage (tests/test_phase18a_automation_eligibility.py)

| Test Class | Tests |
|------------|-------|
| TestEnumValidation | 4 tests (LOCKED enum values) |
| TestImmutability | 5 tests (frozen dataclasses) |
| TestMissingInputs | 8 tests (all inputs mandatory) |
| TestDriftHardStops | 4 tests (HIGH/CRITICAL/arch/db) |
| TestIncidentHardStops | 3 tests (CRITICAL/SECURITY/UNKNOWN) |
| TestProductionEnvironment | 3 tests (stricter production rules) |
| TestGovernanceHardStops | 2 tests (gate denied, invalid baseline) |
| TestDeterminism | 3 tests (same input = same output) |
| TestAllowedDecisions | 3 tests (LIMITED, WITH_APPROVAL) |
| TestAudit | 2 tests (audit creation, required fields) |
| TestNoSideEffects | 2 tests (no execute methods) |

**39 tests total.**

---

## Phase 18B: Human Approval Orchestration - VERIFIED

**DECISION-ONLY orchestrator that answers: "What is the approval status?"**

### Critical Constraints (Enforced)

| Constraint | Status |
|------------|--------|
| ❌ No execution | ✅ Enforced |
| ❌ No notifications | ✅ Enforced |
| ❌ No automation | ✅ Enforced |
| ❌ No lifecycle mutation | ✅ Enforced |
| ✅ Decision-only | ✅ Verified |
| ✅ Human-governed | ✅ Verified |
| ✅ 100% deterministic | ✅ Verified |
| ✅ Mandatory audit | ✅ Verified |

### Approval Status Enum (LOCKED - EXACTLY 3 VALUES)

| Status | Meaning |
|--------|---------|
| APPROVAL_GRANTED | Human approval obtained, action may proceed |
| APPROVAL_DENIED | Approval denied or conditions not met |
| APPROVAL_PENDING | Awaiting human approval |

### Denial Reasons (LOCKED)

| Category | Reasons |
|----------|---------|
| Missing Input | MISSING_ELIGIBILITY, MISSING_RECOMMENDATION, MISSING_LIFECYCLE_STATE, MISSING_EXECUTION_GATE, MISSING_REQUESTER |
| Eligibility | ELIGIBILITY_FORBIDDEN |
| Process | APPROVAL_EXPIRED, APPROVAL_REVOKED, APPROVER_SAME_AS_REQUESTER, APPROVER_UNAUTHORIZED, INSUFFICIENT_APPROVERS |
| Governance | EXECUTION_GATE_DENIED |
| Audit | AUDIT_WRITE_FAILED |

### Pending Reasons (LOCKED)

| Reason | Meaning |
|--------|---------|
| AWAITING_APPROVAL | Waiting for explicit approval |
| AWAITING_CONFIRMATION | Waiting for simple confirmation |
| AWAITING_DUAL_APPROVAL | Waiting for multiple approvers |

### Components Implemented

| Component | File | Purpose |
|-----------|------|---------|
| ApprovalStatus Enum | controller/approval_orchestrator.py | 3 LOCKED status values |
| DenialReason Enum | controller/approval_orchestrator.py | All denial reason IDs |
| PendingReason Enum | controller/approval_orchestrator.py | All pending reason IDs |
| ApprovalType Enum | controller/approval_orchestrator.py | 4 approval type values |
| ApprovalRequesterInput | controller/approval_orchestrator.py | Frozen requester snapshot |
| ApproverInput | controller/approval_orchestrator.py | Frozen approver snapshot |
| ApprovalStateInput | controller/approval_orchestrator.py | Frozen approval state |
| OrchestrationInput | controller/approval_orchestrator.py | Combined input (all mandatory) |
| OrchestrationResult | controller/approval_orchestrator.py | Frozen result with status |
| ApprovalAuditRecord | controller/approval_orchestrator.py | Frozen audit record |
| HumanApprovalOrchestrator | controller/approval_orchestrator.py | Pure orchestration logic |
| ApprovalStore | controller/approval_store.py | Append-only JSONL persistence |
| ApprovalRequestRecord | controller/approval_store.py | Frozen request record |
| ApproverActionRecord | controller/approval_store.py | Frozen action record |
| DecisionRecord | controller/approval_store.py | Frozen decision record |

### Required Inputs (All Mandatory)

| Input | Source | If Missing |
|-------|--------|------------|
| Eligibility Result | Phase 18A | → DENIED |
| Recommendation | Phase 17C | → DENIED |
| Lifecycle State | Lifecycle Engine | → DENIED |
| Execution Gate | Phase 15.6 | → DENIED |
| Requester | Request Context | → DENIED |
| Approval State | Approval Store | → PENDING (if fresh) |

### Immediate Denial Rules (ANY Match → DENIED)

| Rule | Condition |
|------|-----------|
| Eligibility Forbidden | eligibility_result.decision = FORBIDDEN |
| Gate Denied | execution_gate.gate_allows_action = false |
| Approval Expired | expires_at < current_timestamp |
| Self-Approval | approver_id = requester_id |

### Approval Grant Rules

| Rule | Condition |
|------|-----------|
| No Approval Required | approval_required = "none_required" |
| Sufficient Approvers | approver_count >= required_approver_count |

### Symmetry Guarantee Table

| Input Condition | Output Status | Output Reason |
|-----------------|---------------|---------------|
| eligibility_result=None | APPROVAL_DENIED | MISSING_ELIGIBILITY |
| recommendation=None | APPROVAL_DENIED | MISSING_RECOMMENDATION |
| lifecycle_state=None | APPROVAL_DENIED | MISSING_LIFECYCLE_STATE |
| execution_gate=None | APPROVAL_DENIED | MISSING_EXECUTION_GATE |
| requester=None | APPROVAL_DENIED | MISSING_REQUESTER |
| eligibility=FORBIDDEN | APPROVAL_DENIED | ELIGIBILITY_FORBIDDEN |
| gate_allows_action=False | APPROVAL_DENIED | EXECUTION_GATE_DENIED |
| expires_at < current | APPROVAL_DENIED | APPROVAL_EXPIRED |
| approver=requester | APPROVAL_DENIED | APPROVER_SAME_AS_REQUESTER |
| approval_required=none | APPROVAL_GRANTED | None |
| approvers >= required | APPROVAL_GRANTED | None |
| confirmation, 0 approvers | APPROVAL_PENDING | AWAITING_CONFIRMATION |
| explicit, 0 approvers | APPROVAL_PENDING | AWAITING_APPROVAL |
| dual, < required | APPROVAL_PENDING | AWAITING_DUAL_APPROVAL |
| audit write fails | APPROVAL_DENIED | AUDIT_WRITE_FAILED |

### Safety Guarantees

- **DECISION_ONLY**: Returns status, NEVER executes anything
- **NO_NOTIFICATIONS**: Does not send alerts, emails, or messages
- **NO_AUTOMATION**: Does not trigger any automated actions
- **DETERMINISTIC**: Same inputs ALWAYS produce same output
- **HUMAN_GOVERNED**: Humans approve, system tracks
- **SELF_APPROVAL_BLOCKED**: Requester cannot approve own request
- **EXPIRY_ENFORCED**: Expired approvals are DENIED
- **AUDIT_REQUIRED**: Every evaluation emits immutable audit record
- **AUDIT_FAILURE_DENIED**: If audit write fails → DENIED

### Approval Store

| Feature | Description |
|---------|-------------|
| Persistence | Append-only JSONL with fsync |
| Requests File | approval_requests.jsonl |
| Decisions File | approval_decisions.jsonl |
| Actions File | approver_actions.jsonl |
| Immutability | Records never modified or deleted |

### Test Coverage (tests/test_phase18b_approval_orchestration.py)

| Test Class | Tests |
|------------|-------|
| TestEnumValidation | 5 tests (LOCKED enum values) |
| TestImmutability | 5 tests (frozen dataclasses) |
| TestMissingInputs | 6 tests (all inputs mandatory) |
| TestImmediateDenial | 5 tests (denial conditions) |
| TestApprovalGrant | 5 tests (grant conditions) |
| TestPendingState | 4 tests (pending reasons) |
| TestDeterminism | 3 tests (same input = same output) |
| TestAudit | 2 tests (audit creation, failure) |
| TestApprovalStore | 6 tests (store operations) |
| TestIntegration | 2 tests (full workflow) |

**43 tests total.**

---

## Important: Runtime Truth Validation (Phase 15.8)

**All system validations now use RUNTIME TRUTH, not configuration presence.**

### Principle
If a service is running and operational, it is valid - period.
Configuration presence is informational, not authoritative.

### Claude CLI (Phase 15.7)
- `claude --version` only checks if binary exists
- Real execution test (`claude --print`) determines actual capability
- OAuth session does NOT work for `--print` mode
- Only ANTHROPIC_API_KEY or `claude setup-token` work for automation

### Telegram Bot (Phase 15.8)
- Token validity determined by RUNTIME health, not ENV presence
- If `ai-telegram-bot` service is running → token is VALID
- Checks: systemd status, process check, log activity

### Validation Priority Order
1. **systemd service status** (authoritative on VPS)
2. **Process check** (fallback)
3. **Log activity** (recent log updates)
4. **ENV presence** (informational only, NOT authoritative)

See [CLAUDE_CLI_EXECUTION_MODEL.md](CLAUDE_CLI_EXECUTION_MODEL.md) for CLI details.

---

## Implemented Components

| Component | Status | File(s) | Notes |
|-----------|--------|---------|-------|
| Task Controller | Implemented | controller/main.py | FastAPI, full task lifecycle, execution engine, CI/release gates, production deployment |
| Telegram Bot | Implemented | bots/telegram_bot.py | Multi-user, multi-project, execution commands, CI commands, production commands |
| CI/CD Pipeline | Skeleton | workflows/ci.yml | Lint + test + deploy placeholder |
| Unit Tests | Complete | tests/*.py | pytest, comprehensive Phase 6 tests (dual approval, audit logs) |
| Documentation | Updated | docs/*.md | Architecture diagrams, execution safety model, CI gates, production gates |
| Project Bootstrap | Implemented | controller/main.py | Creates project dirs, manifest, state files, diffs/, backups/, audit/, releases/ |
| Plan Generation | Implemented | controller/main.py | Markdown plans with risk/rollback |
| Approval Gates | Implemented | controller/main.py | Approve/reject with reason validation |
| Diff Generation | Implemented | controller/main.py | Unified diff format |
| Dry-Run | Implemented | controller/main.py | Phase 4: Simulation without file modification |
| Apply | Implemented | controller/main.py | Phase 4: Requires confirmation, creates backup |
| Rollback | Implemented | controller/main.py | Phase 4: Restore from backup |
| Backup Strategy | Implemented | controller/main.py | Phase 4: Automatic backup before apply |
| Failure Handling | Implemented | controller/main.py | Phase 4: Auto-restore on apply failure |
| Commit Preparation | Implemented | controller/main.py | Phase 5: Requires confirmation, local only |
| CI Trigger | Implemented | controller/main.py | Phase 5: Human-initiated CI |
| CI Result Ingestion | Implemented | controller/main.py | Phase 5: Passed/failed status |
| Testing Deployment | Implemented | controller/main.py | Phase 5: Requires CI pass + confirmation |
| Production Request | Implemented | controller/main.py | Phase 6: Risk acknowledgment, justification required |
| Production Approval | Implemented | controller/main.py | Phase 6: DUAL APPROVAL (different user required) |
| Production Apply | Implemented | controller/main.py | Phase 6: Final confirmation, creates release manifest |
| Production Rollback | Implemented | controller/main.py | Phase 6: Break-glass (no dual approval for speed) |
| Audit Trail | Implemented | controller/main.py | Phase 6: Immutable, append-only production.log |
| Policy Hooks | Implemented | controller/main.py | Includes execution, CI, and production policy hooks |
| Rate Limiting | Implemented | telegram_bot_v2/bot.py | Phase 13.10: Per-user, per-command rate limits |
| Timeouts & Retries | Implemented | telegram_bot_v2/bot.py | Phase 13.11: Exponential backoff, no retries for destructive actions |
| Degraded Mode | Implemented | telegram_bot_v2/bot.py | Phase 13.12: NORMAL/DEGRADED/CRITICAL modes, owner override |
| Claude CLI | Installed | /usr/local/bin/claude | Phase 14.1: v2.1.12 on VPS |
| Execution Wrapper | Implemented | scripts/run_claude_job.sh | Phase 14.2: Secure, sandboxed execution |
| Claude Backend | Implemented | controller/claude_backend.py | Phase 14.3: Job queue, workspace manager, executor |
| Job Workspace Model | Implemented | controller/claude_backend.py | Phase 14.4: Isolated workspaces with policy docs |
| Claude API Endpoints | Implemented | controller/main.py | Phase 14.3: /claude/job, /claude/jobs, /claude/queue, /claude/status |
| Bot Claude Integration | Implemented | telegram_bot_v2/bot.py | Phase 14.6: /health shows Claude status |
| Multi-Worker Scheduler | Implemented | controller/claude_backend.py | Phase 14.10: MAX_CONCURRENT_JOBS=3, FIFO queue |
| Persistent Job Store | Implemented | controller/claude_backend.py | Phase 14.10: JSON-based state persistence for crash recovery |
| Worker Process Isolation | Implemented | controller/claude_backend.py | Phase 14.10: CPU nice, memory limits, isolated subprocesses |
| Scheduler Status Endpoint | Implemented | controller/main.py | Phase 14.10: /claude/scheduler for worker monitoring |
| Priority Scheduling | Implemented | controller/claude_backend.py | Phase 14.11: EMERGENCY > HIGH > NORMAL > LOW |
| Starvation Prevention | Implemented | controller/claude_backend.py | Phase 14.11: Auto-escalation after 30min wait |
| Priority Audit Logging | Implemented | controller/claude_backend.py | Phase 14.11: Append-only escalation audit trail |
| Priority Tests | Implemented | tests/test_priority_scheduling.py | Phase 14.11: 20+ tests for priority scheduling |
| Lifecycle State Machine | Implemented | controller/lifecycle_v2.py | Phase 15.1: Deterministic 10-state lifecycle |
| PROJECT_MODE/CHANGE_MODE | Implemented | controller/lifecycle_v2.py | Phase 15.1: Two modes of work |
| Multi-Aspect Isolation | Implemented | controller/lifecycle_v2.py | Phase 15.1: core/backend/frontend isolation |
| Event-Driven Transitions | Implemented | controller/lifecycle_v2.py | Phase 15.1: Claude/test/feedback/approval triggers |
| Lifecycle API Endpoints | Implemented | controller/main.py | Phase 15.1: GET/POST /lifecycle endpoints |
| Telegram Lifecycle Commands | Implemented | telegram_bot_v2/bot.py | Phase 15.1: lifecycle_status/approve/reject/feedback |
| Lifecycle Tests | Implemented | tests/test_lifecycle_engine.py | Phase 15.1: 50+ tests for lifecycle engine |
| Continuous Change Cycles | Implemented | controller/lifecycle_v2.py | Phase 15.2: DEPLOYED -> AWAITING_FEEDBACK loops |
| SECURITY Change Type | Implemented | controller/lifecycle_v2.py | Phase 15.2: Added security fix change type |
| Cycle Tracking | Implemented | controller/lifecycle_v2.py | Phase 15.2: cycle_number, change_summary, previous_deployment_id |
| Change History & Lineage | Implemented | controller/lifecycle_v2.py | Phase 15.2: get_cycle_history, get_change_lineage |
| Deployment Summaries | Implemented | controller/lifecycle_v2.py | Phase 15.2: generate_deployment_summary |
| Change API Endpoints | Implemented | controller/main.py | Phase 15.2: /lifecycle/{id}/change, /cycles, /lineage, /summary |
| Project Changes API | Implemented | controller/main.py | Phase 15.2: /project/{id}/changes, /aspects/{aspect}/history |
| Telegram Change Commands | Implemented | telegram_bot_v2/bot.py | Phase 15.2: new_feature/report_bug/improve/refactor/security_fix |
| Scheduler Aspect Isolation | Implemented | controller/claude_backend.py | Phase 15.2: One active job per project+aspect |
| Phase 15.2 Tests | Implemented | tests/test_lifecycle_engine.py | Phase 15.2: 30+ additional tests |
| Ingestion Engine | Implemented | controller/ingestion_engine.py | Phase 15.3: External project ingestion & adoption |
| Git Repository Analysis | Implemented | controller/ingestion_engine.py | Phase 15.3: Clone, analyze git metadata |
| Local Path Analysis | Implemented | controller/ingestion_engine.py | Phase 15.3: Analyze local directories |
| File Enumeration | Implemented | controller/ingestion_engine.py | Phase 15.3: Enumerate files, detect binaries |
| Structure Analysis | Implemented | controller/ingestion_engine.py | Phase 15.3: Detect build system, package manager |
| Aspect Detection | Implemented | controller/ingestion_engine.py | Phase 15.3: Auto-detect project aspects |
| Risk Scanning | Implemented | controller/ingestion_engine.py | Phase 15.3: Detect secrets, sensitive files |
| Document Generation | Implemented | controller/ingestion_engine.py | Phase 15.3: Generate PROJECT_MANIFEST, CURRENT_STATE, etc |
| Ingestion Workflow | Implemented | controller/ingestion_engine.py | Phase 15.3: Create, analyze, approve, register |
| Ingestion API Endpoints | Implemented | controller/main.py | Phase 15.3: /ingestion/* endpoints |
| Telegram Ingestion Commands | Implemented | telegram_bot_v2/bot.py | Phase 15.3: ingest_git/ingest_local/approve/register |
| Phase 15.3 Tests | Implemented | tests/test_ingestion_engine.py | Phase 15.3: 30+ tests |
| Claude CLI Session Auth | Implemented | controller/claude_backend.py | Phase 15.5: Session-based auth, API key optional |
| Auth Detection Tests | Implemented | tests/test_claude_auth.py | Phase 15.5: Unit tests for auth detection |
| Execution Gate Model | Implemented | controller/execution_gate.py | Phase 15.6: Lifecycle-state/role/aspect permission enforcement |
| Lifecycle-Action Mapping | Implemented | controller/execution_gate.py | Phase 15.6: LIFECYCLE_ALLOWED_ACTIONS, ROLE_ALLOWED_ACTIONS |
| Workspace Isolation | Implemented | controller/execution_gate.py | Phase 15.6: Job workspace validation, path traversal prevention |
| Governance Doc Enforcement | Implemented | controller/execution_gate.py | Phase 15.6: Required docs must exist before execution |
| Execution Audit Trail | Implemented | controller/execution_gate.py | Phase 15.6: Immutable, append-only audit log |
| Hard Fail Conditions | Implemented | controller/execution_gate.py | Phase 15.6: Security violations terminate execution |
| Claude Invocation Contract | Implemented | controller/claude_backend.py | Phase 15.6: ExecutionGate check before execution |
| Execution Gate Tests | Implemented | tests/test_execution_gate.py | Phase 15.6: 50+ security tests |
| Real Execution Check | Implemented | controller/claude_backend.py | Phase 15.7: check_claude_availability tests real prompt execution |
| Execution Model Docs | Implemented | docs/CLAUDE_CLI_EXECUTION_MODEL.md | Phase 15.7: Auth states, setup guide, troubleshooting |
| Dry-Run Validation | Updated | scripts/dry_run_validation.py | Phase 15.7: Includes real Claude CLI execution test |
| Runtime Truth Validation | Implemented | scripts/dry_run_validation.py | Phase 15.8: Telegram bot validated via runtime health |
| Telegram Runtime Check | Implemented | scripts/dry_run_validation.py | Phase 15.8: systemd/process/log-based verification |
| Claude Smoke Test | Verified | scripts/claude_smoke_test.py | Phase 16A: Real end-to-end job execution proven |
| Smoke Test Report | Generated | docs/SMOKE_TEST_REPORT.md | Phase 16A: Execution results and validation |
| Dashboard Backend | Implemented | controller/dashboard_backend.py | Phase 16B: Read-only observability layer |
| Dashboard API | Implemented | controller/main.py | Phase 16B: /dashboard/* endpoints |
| Telegram Dashboard | Enhanced | telegram_bot_v2/bot.py | Phase 16B: Enhanced /dashboard command |
| Dashboard Tests | Implemented | tests/test_dashboard_backend.py | Phase 16B: 21 tests for read-only behavior
| Project Registry | Implemented | controller/project_registry.py | Phase 16C: Canonical project storage, crash recovery |
| CHD Validator | Implemented | controller/chd_validator.py | Phase 16C: Requirements validation before job creation |
| Project Service | Implemented | controller/project_service.py | Phase 16C: Unified project creation with progress callbacks |
| Telegram File Handler | Implemented | telegram_bot_v2/bot.py | Phase 16C: .md/.txt file upload for project creation |
| Phase 16C Tests | Implemented | tests/test_phase16c.py | Phase 16C: 24 tests for registry, validator, service |
| Project Identity Engine | Implemented | controller/project_identity.py | Phase 16E: Deterministic fingerprinting, normalized intent |
| Decision Engine | Implemented | controller/project_decision_engine.py | Phase 16E: Locked decision matrix, conflict detection |
| Project Registry v2 | Upgraded | controller/project_registry.py | Phase 16E: Identity fields, version tracking, fingerprint lookup |
| Conflict Resolution UX | Implemented | telegram_bot_v2/bot.py | Phase 16E: User choice buttons for conflict resolution |
| Dashboard Identity Grouping | Implemented | controller/dashboard_backend.py | Phase 16E: Group projects by fingerprint family |
| Phase12 Router Integration | Updated | controller/phase12_router.py | Phase 16E: Decision engine before project creation |
| Phase 16E Tests | Implemented | tests/test_phase16e.py | Phase 16E: 30 tests for identity, decision, integration |
| Intent Baseline Manager | Implemented | controller/intent_baseline.py | Phase 16F: Immutable baseline storage, rebaseline workflow |
| Intent Drift Engine | Implemented | controller/intent_drift_engine.py | Phase 16F: Drift detection, classification, scoring |
| Intent Contract Enforcer | Implemented | controller/intent_contract.py | Phase 16F: Contract enforcement, confirmation workflow |
| ExecutionGate + Drift | Updated | controller/execution_gate.py | Phase 16F: Drift checks integrated into execution gate |
| Phase 16F Tests | Implemented | tests/test_phase16f.py | Phase 16F: 33 tests for baseline, drift, contract |
| Runtime Intelligence Engine | Implemented | controller/runtime_intelligence.py | Phase 17A: OBSERVATION-ONLY signal collection and persistence |
| RuntimeSignal Model | Implemented | controller/runtime_intelligence.py | Phase 17A: Frozen/immutable signal dataclass |
| SignalCollector | Implemented | controller/runtime_intelligence.py | Phase 17A: Read-only system/worker/lifecycle signal collection |
| SignalPersister | Implemented | controller/runtime_intelligence.py | Phase 17A: Append-only JSONL persistence with fsync |
| Runtime API Endpoints | Implemented | controller/main.py | Phase 17A: GET /runtime/signals, /summary, /status |
| Dashboard Observability | Updated | controller/dashboard_backend.py | Phase 17A: ObservabilityHealth, signal counts in summary |
| Telegram Signal Commands | Implemented | telegram_bot_v2/bot.py | Phase 17A: /signals, /signals_recent, /runtime_status |
| Phase 17A Tests | Implemented | tests/test_phase17a_runtime_intelligence.py | Phase 17A: 58 tests for observation-only behavior |

---

## Deployments

### Development
- **Status**: Not deployed
- **Version**: N/A
- **URL**: localhost:8000 (when running locally)
- **Last Deploy**: N/A

### Testing
- **Status**: Not deployed
- **Version**: N/A
- **URL**: https://aitesting.mybd.in (configured, not deployed)
- **Last Deploy**: N/A

### Production
- **Status**: Not deployed
- **Version**: N/A
- **URL**: https://ai.mybd.in (configured, not deployed)
- **Last Deploy**: N/A

---

## Test Results

| Test Suite | Status | Coverage | Last Run |
|------------|--------|----------|----------|
| Unit Tests | Scaffold ready | N/A | Not run yet |
| Integration Tests | Not created | N/A | N/A |
| Smoke Tests | Not created | N/A | N/A |

---

## Known Issues

| Issue ID | Description | Severity | Status | Created |
|----------|-------------|----------|--------|---------|
| - | No known issues | - | - | - |

---

## Blocking Issues

```
None
```

---

## Next Planned Tasks

1. **[HIGH]** Push code to GitHub repository
2. **[HIGH]** Create Telegram bot token
3. **[HIGH]** Connect Telegram bot to actual Telegram API
4. **[MEDIUM]** Implement deployment scripts for VPS
5. **[MEDIUM]** Phase 7: Claude CLI integration (actual AI code generation)
6. **[LOW]** Add persistent session storage (database)
7. **[LOW]** Add authentication/authorization

---

## Recent Activity Log

| Timestamp | Task | Status | Details |
|-----------|------|--------|---------|
| 2026-01-21 | Phase 18B complete | Completed | Human Approval Orchestration: DECISION-ONLY, HUMAN-GOVERNED, LOCKED enum ApprovalStatus (EXACTLY 3 values: GRANTED/DENIED/PENDING), frozen inputs, immediate denial rules, approval grant rules, pending states, self-approval blocked, expiry enforced, approval store (append-only JSONL), 100% deterministic, mandatory audit, NO execution/notifications/automation, 43 tests |
| 2026-01-20 | Phase 18A complete | Completed | Automation Eligibility Engine: DECISION-ONLY, LOCKED enum EligibilityDecision (EXACTLY 3 values), frozen inputs (all 7 mandatory), hard-stop rules (drift/incidents/signals/environment/governance), 100% deterministic, mandatory audit, NO execution/scheduling/mutation, 39 tests |
| 2026-01-20 | Phase 17C complete | Completed | Recommendation & Human-in-the-Loop Reasoning Layer: ADVISORY-ONLY recommendations, RecommendationType/Severity/Approval/Status LOCKED enums, frozen Recommendation dataclass, rule-based generation, UNKNOWN propagation, append-only JSONL with separate approval log, API endpoints (approve/dismiss), Telegram commands, dashboard integration, 50 tests |
| 2026-01-20 | Phase 17B complete | Completed | Signal Interpretation & Incident Classification Layer: OBSERVATION-ONLY incidents, IncidentType/Severity/Scope LOCKED enums, frozen Incident dataclass, deterministic rule-based classification, UNKNOWN for missing data, append-only JSONL persistence, API endpoints, Telegram commands, dashboard incident integration, 52 tests |
| 2026-01-20 | Phase 17A complete | Completed | Runtime Intelligence & Signal Collection Layer: OBSERVATION-ONLY signals, SignalType/Severity LOCKED enums, deterministic classification, UNKNOWN for missing data, append-only JSONL persistence, API endpoints, Telegram commands, dashboard observability integration, 58 tests |
| 2026-01-20 | Phase 16F complete | Completed | Intent Drift, Regression & Contract Enforcement: Immutable baselines, drift detection (6 dimensions), contract enforcement (SOFT/CONFIRM/BLOCK), ExecutionGate integration, 33 tests |
| 2026-01-20 | Phase 16E complete | Completed | Project Identity, Fingerprinting & Conflict Resolution: Deterministic fingerprinting, decision engine, conflict resolution UX |
| 2026-01-19 | Phase 16C complete | Completed | Real Project Execution Stabilization: Project Registry, CHD Validator, Project Service, Telegram file uploads, 24 tests |
| 2026-01-19 | Phase 16B complete | Completed | Platform Dashboard & Observability Layer: Read-only aggregation, API endpoints, Telegram integration, 21 tests |
| 2026-01-19 | Phase 16A complete | Completed | Claude Execution Smoke Test: Real end-to-end job execution proven, README.md created with exact content |
| 2026-01-19 | Phase 15.6 complete | Completed | Execution Gate Model: ExecutionGate, lifecycle-state/role/aspect permissions, audit trail, hard fail conditions, 50+ security tests |
| 2026-01-19 | Phase 15.4 complete | Completed | Roadmap Intelligence: ROADMAP.md, EPICS.yaml (24 epics), MILESTONES.yaml (46 milestones), lifecycle integration notes |
| 2026-01-19 | Phase 15.3 complete | Completed | Project Ingestion Engine: external project adoption, git/local analysis, aspect detection, risk scanning, document generation |
| 2026-01-19 | Phase 15.2 complete | Completed | Continuous Change Cycles: DEPLOYED loops, SECURITY type, cycle tracking, change history/lineage |
| 2026-01-19 | Phase 15.1 complete | Completed | Autonomous Lifecycle Engine: 10-state machine, PROJECT/CHANGE modes, aspect isolation, event triggers |
| 2026-01-19 | Phase 14.11 complete | Completed | Priority & Fair Scheduling: EMERGENCY/HIGH/NORMAL/LOW, starvation prevention (30min), audit logging |
| 2026-01-19 | Phase 14.10 complete | Completed | Multi-worker scheduler: MAX_CONCURRENT_JOBS=3, persistent state, resource limits, crash recovery |
| 2026-01-18 | Phase 14.0-14.6 complete | Completed | Claude CLI installation, execution wrapper, job backend, workspace model, API endpoints |
| 2026-01-18 | Phase 13.10-13.12 complete | Completed | Rate limiting, timeouts/retries, degraded mode for Telegram bot |
| 2026-01-16 | Phase 6 complete | Completed | Production request, approval (dual), apply, rollback (break-glass), audit trail |
| 2026-01-16 | Phase 6 tests | Completed | Dual approval tests, audit log tests, state transition tests, policy tests |
| 2026-01-16 | Phase 5 complete | Completed | Commit, CI trigger, CI result, testing deploy with gates |
| 2026-01-16 | Phase 5 tests | Completed | Commit tests, CI tests, deploy tests, full lifecycle tests |
| 2026-01-16 | Phase 4 complete | Completed | Dry-run, apply with confirmation, rollback, backup strategy, failure handling |
| 2026-01-16 | Phase 4 tests | Completed | Execution tests, policy enforcement tests, full lifecycle tests |
| 2026-01-16 | Phase 3 complete | Completed | Diff generation endpoint, DIFF_GENERATED state, policy hooks, bot command |
| 2026-01-16 | Phase 3 tests | Completed | Diff generation tests, policy enforcement tests, lifecycle tests |
| 2026-01-16 | Phase 2 complete | Completed | Project bootstrap, task lifecycle, plans, approval gates, policy hooks |
| 2026-01-16 | Phase 2 tests | Completed | 50+ unit tests for controller and bot |
| 2026-01-16 | Phase 1 skeleton | Completed | Created controller/, bots/, tests/, workflows/, moved docs/, created README files |
| 2026-01-16 | Bootstrap config | Completed | Updated all files with confirmed repository, domains, hosting, and tech stack details |
| 2026-01-16 | Bootstrap docs | Completed | Created all foundational files |

---

## Phase 15.6 Deliverables

### Execution Gate Model (controller/execution_gate.py)

| Component | Description |
|-----------|-------------|
| ExecutionGate | Single point of authorization for Claude CLI execution |
| ExecutionAction | Enum of permitted actions: READ_CODE, WRITE_CODE, RUN_TESTS, COMMIT, PUSH, DEPLOY_TEST, DEPLOY_PROD |
| LifecycleState | Enum of 10 lifecycle states |
| UserRole | Enum of 5 user roles: OWNER, ADMIN, DEVELOPER, TESTER, VIEWER |
| ExecutionRequest | Request data class for gate evaluation |
| GateDecision | Decision result with allowed/denied and reason |
| ExecutionAuditEntry | Immutable audit log entry |
| LIFECYCLE_ALLOWED_ACTIONS | Mapping of state -> permitted actions |
| ROLE_ALLOWED_ACTIONS | Mapping of role -> permitted actions |

### Lifecycle State Permissions

| State | Allowed Actions |
|-------|-----------------|
| CREATED | None |
| PLANNING | READ_CODE |
| DEVELOPMENT | READ_CODE, WRITE_CODE, RUN_TESTS, COMMIT |
| TESTING | READ_CODE, RUN_TESTS |
| AWAITING_FEEDBACK | READ_CODE |
| FIXING | READ_CODE, WRITE_CODE, RUN_TESTS, COMMIT |
| READY_FOR_PRODUCTION | READ_CODE, PUSH, DEPLOY_TEST |
| DEPLOYED | READ_CODE |
| REJECTED | None |
| ARCHIVED | None |

### Role Permissions

| Role | Allowed Actions |
|------|-----------------|
| OWNER | READ_CODE, WRITE_CODE, RUN_TESTS, COMMIT, PUSH, DEPLOY_TEST |
| ADMIN | READ_CODE, WRITE_CODE, RUN_TESTS, COMMIT, PUSH, DEPLOY_TEST |
| DEVELOPER | READ_CODE, WRITE_CODE, RUN_TESTS, COMMIT |
| TESTER | READ_CODE, RUN_TESTS |
| VIEWER | READ_CODE |

### Hard Fail Conditions

| Condition | Description |
|-----------|-------------|
| Invalid lifecycle state | Execution terminates, audit logged |
| Invalid action | Execution terminates, audit logged |
| Invalid user role | Execution terminates, audit logged |
| Action not permitted in state | Execution terminates, audit logged |
| Role cannot perform action | Execution terminates, audit logged |
| Workspace outside /home/aitesting.mybd.in/jobs/ | Execution terminates (path traversal prevention) |
| Governance documents missing | Execution terminates (policy enforcement) |
| DEPLOY_PROD via automation | Always blocked (requires human dual-approval) |

### Security Tests (tests/test_execution_gate.py)

| Test Class | Coverage |
|------------|----------|
| TestLifecycleStatePermissions | State permission enforcement (10 states) |
| TestDeployRestrictions | Deploy blocked from wrong states, prod always blocked |
| TestCommitRestrictions | Commit blocked in AWAITING_FEEDBACK, TESTING, DEPLOYED |
| TestWorkspaceIsolation | Path traversal prevention, jobs directory enforcement |
| TestGovernanceDocuments | Required docs enforcement |
| TestRoleBasedAccessControl | Role permission enforcement |
| TestAuditTrail | Audit logging for all decisions |
| TestInvalidInputs | Invalid state/action/role handling |

---

## Phase 15.3 Deliverables

### Project Ingestion Engine (controller/ingestion_engine.py)

| Component | Description |
|-----------|-------------|
| ProjectIngestionEngine | Main class for analyzing and ingesting external projects |
| IngestionRequest | Data class for tracking ingestion requests |
| IngestionReport | Complete analysis report for a project |
| FileInfo | Information about individual files |
| AspectDetection | Detected project aspects with confidence scores |
| RiskAssessment | Security risk analysis results |
| GitMetadata | Git repository metadata |
| StructureAnalysis | Project structure analysis |

### Ingestion Workflow

| Step | Description |
|------|-------------|
| 1. Create Request | /ingestion POST - creates pending request |
| 2. Start Analysis | /ingestion/{id}/analyze POST - clones/prepares and analyzes |
| 3. Review Report | Analysis results available with risk assessment |
| 4. Approve/Reject | /ingestion/{id}/approve or /reject POST |
| 5. Register | /ingestion/{id}/register POST - creates lifecycle instances |

### Ingestion Status Values

| Status | Description |
|--------|-------------|
| pending | Request created, not yet analyzed |
| analyzing | Analysis in progress |
| awaiting_approval | Analysis complete, waiting for human approval |
| approved | Approved by admin/owner |
| rejected | Rejected with reason |
| registered | Successfully registered as project(s) |
| failed | Analysis or registration failed |

### Ingestion Source Types

| Type | Description |
|------|-------------|
| git_repository | Clone from Git URL (https or git@) |
| local_path | Analyze local filesystem directory |

### Analysis Pipeline

| Step | Description |
|------|-------------|
| Repository Inspection | Clone git repo or prepare local path |
| Git Metadata | Extract remote URL, branch, commits, contributors |
| File Enumeration | Enumerate all files (excludes .git, node_modules, etc) |
| Structure Analysis | Detect build system, package manager, tests, CI |
| Aspect Detection | Classify aspects (backend, frontend, core, etc) |
| Risk Scanning | Detect hardcoded secrets, sensitive files, dangerous patterns |
| Document Check | Find existing governance documents |

### Risk Levels

| Level | Criteria |
|-------|----------|
| low | No issues found |
| medium | Some issues found (>5 total) |
| high | Dangerous patterns or >10 issues |
| critical | Hardcoded secrets found (>5) |

### Generated Governance Documents

| Document | Description |
|----------|-------------|
| PROJECT_MANIFEST.yaml | Project metadata and configuration |
| CURRENT_STATE.md | Current project state documentation |
| ARCHITECTURE.md | Architecture and structure overview |
| AI_POLICY.md | AI collaboration policy and boundaries |
| TESTING_STRATEGY.md | Testing strategy and requirements |

### API Endpoints (Phase 15.3)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /ingestion | POST | Create ingestion request |
| /ingestion/{id}/analyze | POST | Start analysis |
| /ingestion/{id}/approve | POST | Approve ingestion |
| /ingestion/{id}/reject | POST | Reject ingestion |
| /ingestion/{id}/register | POST | Register as project |
| /ingestion/{id} | GET | Get ingestion by ID |
| /ingestion | GET | List ingestions |
| /ingestion/status | GET | Get engine status |

### Telegram Bot Commands (Phase 15.3)

| Command | Purpose |
|---------|---------|
| /ingest_git <name> <url> | Ingest from Git repository |
| /ingest_local <name> <path> | Ingest from local path |
| /approve_ingestion <id> | Approve ingestion |
| /reject_ingestion <id> <reason> | Reject ingestion |
| /register_ingestion <id> | Register approved ingestion |
| /ingestion_status [id] | Check ingestion status |

### Safety Guarantees

- **READ_ONLY_ANALYSIS**: Analysis never modifies the source project
- **HUMAN_APPROVAL_REQUIRED**: Registration requires explicit approval
- **RISK_VISIBILITY**: Security issues clearly reported before approval
- **DOCUMENT_GENERATION**: Missing governance docs auto-generated
- **LIFECYCLE_INTEGRATION**: Registered projects start in DEPLOYED state
- **AUDIT_TRAIL**: All ingestion actions logged

### Files

| Path | Purpose |
|------|---------|
| /home/aitesting.mybd.in/jobs/ingestion/ingestion_state.json | Persistent ingestion state |
| /home/aitesting.mybd.in/jobs/ingestion/workspaces/{id}/ | Cloned/prepared project |
| /home/aitesting.mybd.in/jobs/ingestion/reports/{id}.json | Analysis reports |

### Test Coverage (tests/test_ingestion_engine.py)

| Test Class | Coverage |
|------------|----------|
| TestIngestionRequestCreation | Create git/local/targeted requests |
| TestFileEnumeration | File listing, limits, exclusions |
| TestGitMetadataExtraction | Git repo detection, metadata |
| TestStructureAnalysis | Build system, package manager detection |
| TestAspectDetection | Aspect classification, confidence |
| TestRiskScanning | Secret/sensitive file detection |
| TestDocumentGeneration | Governance doc generation |
| TestAnalysisPipeline | Full pipeline execution |
| TestApprovalWorkflow | Approve/reject/permission flow |
| TestQueryMethods | List/filter ingestion requests |
| TestSerialization | Data serialization/deserialization |
| TestEdgeCases | Empty dirs, missing paths, binaries |
| TestPublicAPI | Public function signatures |
| TestLifecycleIntegration | Lifecycle registration flow |

---

## Phase 15.1 Deliverables

### Lifecycle State Machine (controller/lifecycle_v2.py)

| State | Description |
|-------|-------------|
| CREATED | Lifecycle instance created, awaiting initialization |
| PLANNING | Claude is generating implementation plan |
| DEVELOPMENT | Claude is implementing feature/fix |
| TESTING | Automated tests are running |
| AWAITING_FEEDBACK | Human review and feedback required |
| FIXING | Claude is addressing feedback/bugs |
| READY_FOR_PRODUCTION | Final approval needed for production |
| DEPLOYED | Successfully deployed to production |
| REJECTED | Lifecycle rejected (terminal) |
| ARCHIVED | Lifecycle archived (terminal) |

### Lifecycle Modes

| Mode | Description |
|------|-------------|
| PROJECT_MODE | New project development (no change_reference required) |
| CHANGE_MODE | Feature/bug/improvement on existing project (requires change_reference) |

### Change Types (CHANGE_MODE)

| Type | Description |
|------|-------------|
| bug | Bug fix |
| feature | New feature |
| improvement | Enhancement to existing functionality |
| refactor | Code refactoring |

### Project Aspects (Multi-Aspect Isolation)

| Aspect | Description |
|--------|-------------|
| core | APIs, shared services |
| backend | Admin panels, internal tools |
| frontend_web | Web application |
| frontend_mobile | Mobile application |
| admin | Administrative interfaces |
| custom | Custom aspect type |

### Transition Triggers

| Trigger | Description |
|---------|-------------|
| claude_job_completed | Claude CLI job finished |
| test_passed | Automated tests passed |
| test_failed | Automated tests failed |
| telegram_feedback | User feedback from Telegram |
| human_approval | Human approved transition |
| human_rejection | Human rejected |
| system_init | System initialization |
| manual_archive | Manual archive request |

### User Roles & Permissions

| Role | Can Trigger |
|------|-------------|
| owner | All triggers |
| admin | All triggers except system_init |
| developer | claude_job_completed, test_*, system_init |
| tester | test_passed, test_failed, telegram_feedback |
| viewer | None (read-only) |

### API Endpoints (Phase 15.1)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /lifecycle | POST | Create PROJECT_MODE lifecycle |
| /lifecycle/change | POST | Create CHANGE_MODE lifecycle |
| /lifecycle/{id} | GET | Get lifecycle by ID |
| /lifecycle | GET | List lifecycles with filters |
| /lifecycle/{id}/transition | POST | Trigger state transition |
| /lifecycle/{id}/guidance | GET | Get next steps guidance |

### Telegram Bot Commands (Phase 15.1)

| Command | Purpose |
|---------|---------|
| /lifecycle_status [id] | View lifecycle state and guidance |
| /lifecycle_approve <id> [reason] | Approve lifecycle transition |
| /lifecycle_reject <id> <reason> | Reject lifecycle |
| /lifecycle_feedback <id> <text> | Submit feedback |
| /lifecycle_prod_approve <id> | Final production approval |

### Safety Guarantees

- **NO_STATE_SKIPPING**: All transitions must follow valid paths
- **NO_IMPLICIT_APPROVALS**: Human approval required at gates
- **NO_CROSS_ASPECT_EFFECTS**: Aspects are fully isolated
- **NO_SILENT_FAILURES**: Invalid actions rejected with explanation
- **IMMUTABLE_AUDIT**: All transitions logged to audit trail
- **ROLE_ENFORCED**: Permissions checked for every transition

### Files

| Path | Purpose |
|------|---------|
| /home/aitesting.mybd.in/jobs/lifecycle/lifecycles.json | Persistent lifecycle state |
| /home/aitesting.mybd.in/jobs/lifecycle/lifecycle_audit.log | Immutable audit trail |

### Test Coverage (tests/test_lifecycle_engine.py)

| Test Class | Coverage |
|------------|----------|
| TestValidTransitions | Full lifecycle flow, state-to-state |
| TestInvalidTransitions | State skipping, wrong triggers |
| TestAspectIsolation | Multi-aspect creation, filtering |
| TestLifecycleModes | PROJECT_MODE vs CHANGE_MODE |
| TestEventDrivenTransitions | All trigger types |
| TestRolePermissions | Role-based access control |
| TestPersistenceRecovery | State persistence, crash recovery |
| TestStateMachineConfig | Configuration validation |
| TestGuidanceSystem | Next steps guidance |
| TestSerialization | Data roundtrip |

---

## Phase 14.11 Deliverables

### Priority Scheduling (controller/claude_backend.py)

| Component | Description |
|-----------|-------------|
| JobPriority Enum | EMERGENCY (100) > HIGH (75) > NORMAL (50) > LOW (25) |
| Priority Queue | Higher priority jobs scheduled first |
| FIFO Within Priority | Same priority jobs processed in order received |
| Job ID Tiebreaker | Deterministic ordering when priority and time match |

### Starvation Prevention

| Configuration | Value | Description |
|--------------|-------|-------------|
| STARVATION_THRESHOLD_MINUTES | 30 | Time before escalation |
| PRIORITY_ESCALATION_AMOUNT | 10 | Priority increase per escalation |
| PRIORITY_ESCALATION_CAP | 75 | Maximum priority (HIGH) via escalation |

### Escalation Rules

- Jobs waiting >30 minutes in QUEUED state get +10 priority
- Maximum priority reachable via escalation is HIGH (75)
- EMERGENCY (100) can only be set explicitly, not via escalation
- Escalation is deterministic and idempotent
- All escalations logged to audit trail

### Enhanced Queue Response (/claude/queue)

| Field | Description |
|-------|-------------|
| job_id | Unique job identifier |
| project | Project name |
| priority | Current priority value |
| priority_label | Human-readable priority (EMERGENCY/HIGH/NORMAL/LOW) |
| wait_time_minutes | Time spent waiting in queue |
| state | Current job state |
| escalations | Number of priority escalations |

### Priority Audit Log

| Path | Purpose |
|------|---------|
| /home/aitesting.mybd.in/jobs/priority_audit.log | Append-only escalation audit trail |

### Safety Guarantees

- **PRIORITY_IMMUTABLE**: Priority only changed by controller (starvation prevention)
- **ESCALATION_CAPPED**: Cannot exceed HIGH (75) via auto-escalation
- **DETERMINISTIC_ORDER**: Same input always produces same order
- **AUDIT_TRAIL**: All escalations logged for governance
- **NO_STARVATION**: All jobs eventually get processed

### Test Coverage (tests/test_priority_scheduling.py)

| Test Class | Coverage |
|------------|----------|
| TestPriorityPreemption | EMERGENCY > HIGH > NORMAL > LOW ordering |
| TestFIFOWithinPriority | FIFO for same priority, job_id tiebreaker |
| TestStarvationPrevention | 30-min threshold, escalation eligibility |
| TestPriorityCapping | Cap at HIGH (75), EMERGENCY preserved |
| TestIdempotentEscalation | Deterministic, no double escalation |
| TestJobPriorityEnum | Enum values, from_value clamping |
| TestClaudeJobSortKey | Sort key format and ordering |
| TestPrioritySchedulingIntegration | Mixed priority queue ordering |

---

## Phase 14.10 Deliverables

### Multi-Worker Scheduler (controller/claude_backend.py)

| Component | Description |
|-----------|-------------|
| MAX_CONCURRENT_JOBS | 3 concurrent Claude CLI jobs maximum |
| WORKER_NICE_VALUE | 10 (lower CPU priority for worker processes) |
| WORKER_MEMORY_LIMIT_MB | 2048 MB per worker process |
| JOB_CLEANUP_AFTER_HOURS | 24 hours (auto-cleanup completed jobs) |

### New Classes

| Class | Purpose |
|-------|---------|
| PersistentJobStore | JSON-based job state persistence for crash recovery |
| ClaudeWorker | Individual worker with process isolation and resource limits |
| MultiWorkerScheduler | Manages worker pool, enforces concurrency, handles recovery |

### Extended JobState Enum

| State | Description |
|-------|-------------|
| QUEUED | Job in queue, waiting for worker |
| PREPARING | Worker assigned, preparing workspace |
| RUNNING | Claude CLI executing |
| AWAITING_APPROVAL | Completed, waiting for human approval |
| DEPLOYED | Deployed to environment |
| COMPLETED | Successfully completed |
| FAILED | Execution failed |
| TIMEOUT | Execution timed out |
| CANCELLED | Cancelled by user |

### New API Functions

| Function | Purpose |
|----------|---------|
| get_scheduler_status() | Get multi-worker scheduler status |
| start_scheduler() | Start scheduler with crash recovery |
| stop_scheduler() | Gracefully stop scheduler |

### New Endpoints (controller/main.py)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /claude/scheduler | GET | Detailed multi-worker scheduler status |

### Safety Guarantees

- **CONCURRENCY_LIMIT_ENFORCED**: Maximum 3 concurrent jobs (4th waits in FIFO queue)
- **PROCESS_ISOLATION**: Each worker runs in isolated subprocess
- **RESOURCE_LIMITS**: CPU nice value (10) and memory limits (512MB) per worker
- **CRASH_RECOVERY**: Persistent state enables restart without losing queue
- **GRACEFUL_SHUTDOWN**: Workers can complete current job before stopping
- **AUTO_CLEANUP**: Old completed jobs cleaned up after 24 hours
- **FIFO_ORDERING**: Jobs processed in order received
- **STATE_PERSISTENCE**: All job state changes persisted to JSON file

### Files

| Path | Purpose |
|------|---------|
| /home/aitesting.mybd.in/jobs/job_state.json | Persistent job state |
| /home/aitesting.mybd.in/jobs/job-{uuid}/ | Individual job workspace |
| /home/aitesting.mybd.in/jobs/archives/ | Archived job logs |

---

## Phase 6 Deliverables

### New Endpoints (controller/main.py)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /task/{id}/deploy/production/request | POST | Request production deployment (DEPLOYED_TESTING → PROD_DEPLOY_REQUESTED) |
| /task/{id}/deploy/production/approve | POST | Approve production (PROD_DEPLOY_REQUESTED → PROD_APPROVED) |
| /task/{id}/deploy/production/apply | POST | Execute production deploy (PROD_APPROVED → DEPLOYED_PRODUCTION) |
| /task/{id}/deploy/production/rollback | POST | Rollback production (DEPLOYED_PRODUCTION → PROD_ROLLED_BACK) |

### New Bot Commands (bots/telegram_bot.py)

| Command | Purpose |
|---------|---------|
| /prod_request <task-id> | Request production deployment (REQUIRES justification, rollback plan, risk acknowledgment) |
| /prod_approve <task-id> | Approve production (DIFFERENT user required, REQUIRES reviewed_changes, reviewed_rollback) |
| /prod_apply <task-id> confirm | Execute production deployment (REQUIRES confirm keyword) |
| /prod_rollback <task-id> | Rollback production immediately (break-glass, REQUIRES reason) |

### New Task States

| State | Description |
|-------|-------------|
| PROD_DEPLOY_REQUESTED | Production deployment requested, awaiting approval |
| PROD_APPROVED | Approved by DIFFERENT user, ready for deploy |
| DEPLOYED_PRODUCTION | Deployed to production |
| PROD_ROLLED_BACK | Production rollback executed (break-glass) |

### New Policy Hooks

| Hook | Purpose |
|------|---------|
| can_request_prod_deploy(risk_acknowledged) | Gate production request (REQUIRES user_id, risk acknowledgment) |
| can_approve_prod_deploy(approver, requester, reviewed) | Gate production approval (ENFORCES dual approval) |
| can_apply_prod_deploy(confirmed) | Gate production apply (REQUIRES confirmed=True) |
| can_rollback_prod() | Gate production rollback (break-glass, no dual approval) |

### New Folders/Files

| Path | Purpose |
|------|---------|
| projects/{name}/audit/ | Directory for audit logs |
| projects/{name}/audit/production.log | Immutable, append-only production audit trail |
| projects/{name}/releases/ | Directory for release manifests |
| projects/{name}/releases/{task_id}/RELEASE_MANIFEST.yaml | Release metadata (requester, approver, applier) |
| projects/{name}/releases/{task_id}/DEPLOY_LOG.md | Deployment log with actions |

### Safety Guarantees

- **DUAL_APPROVAL_REQUIRED**: Production deploy requires requester AND approver (different users)
- **NO_SELF_APPROVAL**: Same user CANNOT approve their own request
- **CONFIRMATION_REQUIRED**: Production apply requires explicit `confirm=true`
- **TESTING_FIRST**: Must deploy to testing before production
- **AUDIT_TRAIL**: All production actions logged to immutable audit trail
- **BREAK_GLASS_ROLLBACK**: Production rollback does NOT require dual approval (speed > ceremony)
- **RISK_ACKNOWLEDGMENT**: Requester must acknowledge production risk
- **JUSTIFICATION_REQUIRED**: Requester must provide justification (min 20 chars)
- **ROLLBACK_PLAN_REQUIRED**: Requester must provide rollback plan

---

## Phase 5 Deliverables

### New Endpoints (controller/main.py)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /task/{id}/commit | POST | Create git commit locally (APPLIED → COMMITTED) |
| /task/{id}/ci/run | POST | Trigger CI pipeline (COMMITTED → CI_RUNNING) |
| /task/{id}/ci/result | POST | Ingest CI result (CI_RUNNING → CI_PASSED/CI_FAILED) |
| /task/{id}/deploy/testing | POST | Deploy to testing (CI_PASSED → DEPLOYED_TESTING) |

### New Bot Commands (bots/telegram_bot.py)

| Command | Purpose |
|---------|---------|
| /commit <task-id> confirm | Create git commit (REQUIRES confirm keyword) |
| /ci_run [task-id\|last] | Trigger CI pipeline |
| /ci_result <task-id> passed\|failed [logs_url] | Report CI result |
| /deploy_testing <task-id> confirm | Deploy to testing (REQUIRES confirm keyword) |

### New Task States

| State | Description |
|-------|-------------|
| COMMITTED | Git commit created locally (NOT pushed) |
| CI_RUNNING | CI pipeline running |
| CI_PASSED | CI pipeline passed, ready for deploy |
| CI_FAILED | CI pipeline failed, blocks deployment |
| DEPLOYED_TESTING | Deployed to testing environment |

### New Policy Hooks

| Hook | Purpose |
|------|---------|
| can_commit(confirmed) | Gate commit (REQUIRES confirmed=True) |
| can_trigger_ci() | Gate CI trigger |
| can_deploy_testing(confirmed) | Gate testing deploy (REQUIRES confirmed=True) |

### Safety Guarantees

- **CONFIRMATION_REQUIRED**: Commit and deploy_testing require explicit `confirm=true`
- **CI_GATE_ENFORCED**: CI must pass before testing deployment
- **NO_AUTO_PUSH**: Commits are local only, NOT pushed automatically
- **NO_AUTO_MERGE**: No automatic merges ever
- **NO_CI_WITHOUT_INTENT**: CI must be explicitly triggered
- **NO_BYPASS_FAILURES**: CI failure blocks all promotion
- **NO_PRODUCTION_DEPLOYMENT**: Still blocked by policy

---

## Phase 4 Deliverables

### New Endpoints (controller/main.py)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /task/{id}/dry-run | POST | Simulate diff application (DIFF_GENERATED → READY_TO_APPLY) |
| /task/{id}/apply | POST | Apply diff with backup (READY_TO_APPLY → APPLIED) |
| /task/{id}/rollback | POST | Restore from backup (APPLIED → ROLLED_BACK) |

### New Bot Commands (bots/telegram_bot.py)

| Command | Purpose |
|---------|---------|
| /dry_run [task-id\|last] | Simulate diff application |
| /apply <task-id> confirm | Apply diff (REQUIRES confirm keyword) |
| /rollback [task-id\|last] | Restore from backup |

### New Task States

| State | Description |
|-------|-------------|
| READY_TO_APPLY | Dry-run successful, ready for apply |
| APPLYING | Apply in progress (transient) |
| APPLIED | Diff applied successfully |
| ROLLED_BACK | Restored from backup |
| EXECUTION_FAILED | Apply failed (auto-restored from backup) |

### New Policy Hooks

| Hook | Purpose |
|------|---------|
| can_dry_run() | Gate dry-run execution |
| can_apply(confirmed) | Gate apply (REQUIRES confirmed=True) |
| can_rollback() | Gate rollback execution |

### New Folders/Files

| Path | Purpose |
|------|---------|
| projects/{name}/backups/ | Directory for backup snapshots |
| projects/{name}/backups/{task_id}/ | Per-task backup directory |
| projects/{name}/backups/{task_id}/BACKUP_MANIFEST.yaml | Backup metadata |

### Safety Guarantees

- **CONFIRMATION_REQUIRED**: Apply requires explicit `confirm=true`
- **BACKUP_MANDATORY**: Files are backed up BEFORE any modification
- **ROLLBACK_GUARANTEED**: Any applied changes can be reversed
- **AUTOMATIC_RESTORE**: On failure, backup is restored automatically
- **NO_AUTONOMOUS_EXECUTION**: Human triggers every step
- **NO_PRODUCTION_DEPLOYMENT**: Still blocked by policy

---

## Phase 3 Deliverables

### New Endpoints (controller/main.py)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /task/{id}/generate-diff | POST | Generate diff for approved task (APPROVED → DIFF_GENERATED) |
| /task/{id}/diff | GET | Retrieve generated diff content |

### New Bot Commands (bots/telegram_bot.py)

| Command | Purpose |
|---------|---------|
| /generate_diff [task-id\|last] | Generate code diff for approved task |

### New Task State

| State | Description |
|-------|-------------|
| DIFF_GENERATED | Diff generated, awaiting human review (NOT applied) |

### New Policy Hooks

| Hook | Purpose |
|------|---------|
| can_generate_diff() | Gate diff generation |
| diff_within_scope() | Validate diff stays in project scope |
| diff_file_limit_ok() | Enforce max 10 files per diff |

### New Folders/Files

| Path | Purpose |
|------|---------|
| projects/{name}/diffs/ | Directory for generated diff files |
| projects/{name}/diffs/{task_id}.diff | Individual diff artifacts |

### Safety Model

- **Diffs are NEVER applied automatically**
- **Diffs are NEVER committed to git**
- **Human MUST review before any application**
- **All diffs include DISCLAIMER header**
- **Max 10 files per diff enforced**

---

## Phase 2 Deliverables

### New Endpoints (controller/main.py)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| /project/bootstrap | POST | Create new project with manifest/state |
| /task/{id}/validate | POST | Validate task (RECEIVED → VALIDATED) |
| /task/{id}/plan | POST | Generate plan (VALIDATED → AWAITING_APPROVAL) |
| /task/{id}/approve | POST | Approve plan (AWAITING_APPROVAL → APPROVED) |
| /task/{id}/reject | POST | Reject with reason (→ REJECTED) |

### New Bot Commands (bots/telegram_bot.py)

| Command | Purpose |
|---------|---------|
| /bootstrap <name> <url> [tech] | Create new project |
| /validate [task-id\|last] | Validate task |
| /plan [task-id\|last] | Generate plan |
| /approve [task-id\|last] | Approve plan |
| /reject <id> <reason> | Reject with reason (min 10 chars) |

### Task States (NO EXECUTED STATE)

| State | Description |
|-------|-------------|
| RECEIVED | Task created, awaiting validation |
| VALIDATED | Task validated, ready for planning |
| PLANNED | Plan generated |
| AWAITING_APPROVAL | Plan ready for human review |
| APPROVED | Plan approved (Phase 2 stops here) |
| REJECTED | Plan rejected with reason |
| ARCHIVED | Task archived |

### Policy Hooks (Stubs)

| Hook | Purpose |
|------|---------|
| can_create_project() | Gate project creation |
| can_submit_task() | Gate task submission |
| can_validate_task() | Gate validation |
| can_plan_task() | Gate plan generation |
| can_approve_task() | Gate approval |
| can_reject_task() | Gate rejection |

---

## Phase 1 Deliverables

### Files Created

| File | Purpose |
|------|---------|
| controller/__init__.py | Module init |
| controller/main.py | FastAPI Task Controller |
| bots/__init__.py | Module init |
| bots/telegram_bot.py | Telegram Bot |
| workflows/ci.yml | GitHub Actions CI pipeline |
| tests/__init__.py | Test module init |
| tests/test_controller.py | Controller unit tests |
| tests/test_telegram_bot.py | Bot unit tests |
| projects/README.md | Projects directory documentation |
| utils/__init__.py | Utils module init |
| utils/README.md | Utils documentation |
| requirements.txt | Python dependencies |
| pytest.ini | Test configuration |
| README.md | Project overview |

### Task Controller Endpoints (Phase 1)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | Service info |
| /health | GET | Detailed health status |
| /task | POST | Create new task |
| /status/{project} | GET | Get project status |
| /deploy | POST | Trigger deployment |
| /projects | GET | List all projects |

### Telegram Bot Commands (Phase 1)

| Command | Purpose |
|---------|---------|
| /start | Welcome message |
| /help | Show help |
| /project <name> | Select project |
| /task <desc> | Create task |
| /deploy testing | Deploy to testing |
| /status | Show project status |
| /list | List projects |

---

## Resolved Questions

| Question | Answer |
|----------|--------|
| Repository | github.com/rahul4webdev/ai-development-platform |
| Testing Domain | aitesting.mybd.in |
| Production Domain | ai.mybd.in |
| Hosting Provider | VPS - AlmaLinux 9 + CyberPanel + OpenLiteSpeed |
| VPS IP Address | 82.25.110.109 |
| VPS SSH Access | ssh root@82.25.110.109 |
| VPS Web Root | /home/aitesting.mybd.in/public_html |
| Tech Stack | Python + FastAPI (confirmed) |
| Telegram Bot | Token to be created later |

---

## Pending Items (Requiring Human Input)

1. **Server Credentials**: Full server path and deployment credentials
2. **Telegram Bot Token**: To be created when ready
3. **GitHub Push**: Code ready to push to repository

---

## What is NOT Implemented (Intentionally)

Per Phase 6 constraints (production hardening):

- **NO AUTONOMOUS PRODUCTION DEPLOYMENT**: Every production action requires human
- **NO SINGLE-ACTOR PRODUCTION APPROVAL**: Dual approval mandatory
- **NO SILENT PRODUCTION CHANGES**: All actions logged to audit trail
- **NO AUTOMATIC COMMITS**: Commits require explicit confirmation
- **NO AUTOMATIC MERGES**: No automatic merges ever
- **NO AUTOMATIC PUSH**: Commits are local only
- **NO CI WITHOUT INTENT**: CI must be explicitly triggered
- **NO BYPASS TEST FAILURES**: CI failure blocks promotion
- **NO BYPASS TESTING ENVIRONMENT**: Must deploy to testing before production
- Claude CLI integration (actual AI code generation) - Phase 7
- Telegram API connection (no token yet)
- Database/persistent storage (file-based only)
- Authentication/authorization (policy hooks are stubs)
- Rate limiting

---

## Autonomy Status

### ENABLED (Phase 6)
- Project bootstrap (create directories, manifests, state files, diffs/, backups/, audit/, releases/)
- Task creation with type inference
- Task validation (state transition)
- Plan generation (markdown templates)
- Approval/rejection workflow
- Diff generation (unified diff format)
- Dry-run simulation (no file modification)
- Apply with confirmation (creates backup first)
- Rollback from backup (guaranteed reversibility)
- Automatic restore on failure
- **Commit with confirmation** (local only, not pushed)
- **CI trigger** (human-initiated)
- **CI result ingestion** (passed/failed)
- **Testing deployment with confirmation** (requires CI pass)
- **Production request with risk acknowledgment** (justification required)
- **Production approval by DIFFERENT user** (dual approval enforced)
- **Production apply with final confirmation** (creates release manifest)
- **Production rollback (break-glass)** (no dual approval for speed)
- **Immutable audit trail** (all production actions logged)
- Multi-user session management
- Multi-project support

### STILL BLOCKED (Requires Phase 7+)
- Claude CLI integration (actual AI code generation)
- Git push (manual only)
- Database operations
- External API calls (except controller)

---

## Phase 15.4 Deliverables: Roadmap Intelligence

### Generated Artifacts

| File | Purpose |
|------|---------|
| docs/ROADMAP.md | Technical roadmap with vision, phases, constraints |
| docs/EPICS.yaml | 20 structured epics across 6 roadmap phases |
| docs/MILESTONES.yaml | 47 milestones with acceptance criteria |

### Roadmap Phases

| Phase | Name | Epic Count | Milestone Count |
|-------|------|------------|-----------------|
| A | Operational Foundation | 4 | 12 |
| B | Claude Execution Activation | 3 | 8 |
| C | End-to-End Workflow Validation | 4 | 13 |
| D | Production Hardening | 6 | 10 |
| E | Production Launch | 3 | 6 |
| F | Operational Maturity | 4 | 4 |

### Lifecycle Integration Notes

#### How Milestones Map to Lifecycle Transitions

| Milestone Pattern | Lifecycle Transition | Trigger |
|-------------------|---------------------|---------|
| Initial creation | CREATED | SYSTEM_INIT |
| Planning milestones | PLANNING | SYSTEM_INIT |
| Development milestones | DEVELOPMENT | CLAUDE_JOB_COMPLETED |
| Testing milestones | TESTING | CLAUDE_JOB_COMPLETED |
| Approval milestones | AWAITING_FEEDBACK | TEST_PASSED |
| Ready milestones | READY_FOR_PRODUCTION | HUMAN_APPROVAL |
| Deploy milestones | DEPLOYED | HUMAN_APPROVAL |

#### Steps Requiring Human Approval

1. **Telegram Bot Token Creation** (EPIC-A1, MS-A1-01)
   - Cannot be automated, requires BotFather interaction

2. **VPS Access Verification** (EPIC-A2, MS-A2-01)
   - Human must confirm credentials work

3. **GitHub Secrets Configuration** (EPIC-A3, MS-A3-03)
   - Human must enter secrets in GitHub UI

4. **ANTHROPIC_API_KEY Provision** (EPIC-B1, MS-B1-01)
   - Human must provide API key

5. **All Lifecycle Transitions to AWAITING_FEEDBACK**
   - Requires HUMAN_APPROVAL trigger

6. **All Lifecycle Transitions to READY_FOR_PRODUCTION**
   - Requires HUMAN_APPROVAL trigger

7. **All Production Deployments**
   - Requires DUAL_APPROVAL (two different users)

8. **Security Audit Sign-off** (EPIC-D1)
   - Human must review and accept findings

#### Steps That Trigger Claude Jobs (Phase 15.5+)

| Epic | Milestone | Claude Job Type |
|------|-----------|-----------------|
| EPIC-C1 | MS-C1-03 | Development implementation |
| EPIC-C1 | MS-C1-04 | Test execution |
| EPIC-C2 | MS-C2-01 | Bug fix implementation |
| EPIC-C2 | MS-C2-02 | Feature implementation |
| EPIC-C2 | MS-C2-03 | Security fix implementation |
| EPIC-C3 | MS-C3-01 | Project analysis |
| EPIC-D6 | MS-D6-01 | Test generation |
| EPIC-D6 | MS-D6-02 | Integration test creation |

#### Blocking Dependencies

Critical path for production deployment:

```
EPIC-A1 (Bot) + EPIC-A2 (VPS) + EPIC-A3 (CI/CD)
         ↓
    EPIC-A4 (Monitoring)
         ↓
    EPIC-B1 (Claude API) → EPIC-B2 (Workspace) → EPIC-B3 (Scheduler)
         ↓
    EPIC-C1 (PROJECT_MODE) → EPIC-C2 (CHANGE_MODE)
         ↓
    EPIC-D1 (Security) + EPIC-D4 (Rollback) + EPIC-D5 (Dual Approval) + EPIC-D6 (Coverage)
         ↓
    EPIC-E1 (Production Deploy)
         ↓
    EPIC-F1-F4 (Operational Maturity)
```

### Current Blockers

| Blocker | Required For | Status |
|---------|--------------|--------|
| Telegram bot token | EPIC-A1 | NOT PROVIDED |
| ANTHROPIC_API_KEY | EPIC-B1 | NOT PROVIDED |
| Second user for dual approval | EPIC-D5, EPIC-E1 | NOT CONFIGURED |

### Next Steps for Human

1. **Immediate**: Create Telegram bot via BotFather
2. **Immediate**: Verify VPS access at 82.25.110.109
3. **Before Phase B**: Obtain ANTHROPIC_API_KEY from Anthropic
4. **Before Phase D**: Configure second user for dual approval testing
5. **Before Phase E**: Approve production deployment via Telegram

---

*This file is the living system state. Update after every task completion.*
