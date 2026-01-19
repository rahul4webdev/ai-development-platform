# Current State (Living System State)

This file is updated after every task. It is machine-first, not prose.
The AI agent MUST update this file after completing any task.

---

## Last Updated
- **Timestamp**: 2026-01-19
- **Task**: Phase 16C Real Project Execution Stabilization - COMPLETED
- **Status**: Complete - Project Registry, CHD validation, file upload support, no more 500 errors

---

## Current Phase
```
Phase: PHASE_16C_COMPLETE
Mode: development
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
