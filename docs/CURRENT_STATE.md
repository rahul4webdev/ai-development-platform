# Current State (Living System State)

This file is updated after every task. It is machine-first, not prose.
The AI agent MUST update this file after completing any task.

---

## Last Updated
- **Timestamp**: 2026-01-19
- **Task**: Phase 15.1 Autonomous Lifecycle Engine - COMPLETED
- **Status**: Complete (ANTHROPIC_API_KEY still required for execution)

---

## Current Phase
```
Phase: PHASE_15.1_COMPLETE
Mode: development
```

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

*This file is the living system state. Update after every task completion.*
