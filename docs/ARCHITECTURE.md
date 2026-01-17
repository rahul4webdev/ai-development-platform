# Architecture

This file defines the technical architecture, folder structure, coding standards, and design patterns.
The AI agent MUST follow this architecture to prevent drift.

---

## System Overview

```
User (Mobile / Desktop Chat)
        ↓
Chat Bot (Telegram)
        ↓
Task Controller (API / Service)
        ↓
Claude CLI Agent (VPS)
        ↓
Git Repository + Filesystem
        ↓
CI/CD Pipeline
        ↓
Dev → Test → Production Environments
```

---

## Component Breakdown

### 1. Chat Bot (Telegram)
- **Purpose**: User-facing interface for commands and notifications
- **Responsibilities**:
  - Accept user instructions
  - Upload project handoff documents
  - Display progress summaries
  - Notify on testing required, deployment complete, or blocking errors
  - Accept feedback for fixes or improvements
- **Boundary**: Never executes code directly; only communicates with Task Controller

### 2. Task Controller
- **Purpose**: Central orchestrator between chat and agent
- **Responsibilities**:
  - Receive chat messages
  - Map messages to task types
  - Attach project context
  - Invoke Claude CLI
  - Stream structured output back to chat
  - Track project states
  - Enforce task boundaries
- **Task Types**:
  - `project_bootstrap`
  - `feature_development`
  - `bug_fix`
  - `refactoring`
  - `deployment`
  - `maintenance`

### 3. Claude CLI Agent
- **Purpose**: Execute development, testing, and deployment tasks
- **Responsibilities**:
  - Parse handoff documents
  - Generate project scaffolding
  - Implement features
  - Write and run tests
  - Deploy to environments
  - Update project state files
- **Boundary**: Stateless; relies on filesystem for all state

### 4. CI/CD Pipeline
- **Purpose**: Enforce quality gates and deployment rules
- **Responsibilities**:
  - Run tests
  - Check coverage thresholds
  - Lint code
  - Validate policy compliance
  - Gate deployments

---

## Directory Structure (IMPLEMENTED)

```
ai-development-platform/
├── controller/                  # Task Controller (FastAPI) [SKELETON]
│   ├── __init__.py
│   └── main.py                  # API endpoints, task management
├── bots/                        # Chat bot implementations [SKELETON]
│   ├── __init__.py
│   └── telegram_bot.py          # Multi-user, multi-project support
├── projects/                    # Managed project directories
│   └── README.md                # Project structure documentation
├── workflows/                   # GitHub Actions CI/CD
│   └── ci.yml                   # Lint, test, deploy pipeline
├── docs/                        # Platform documentation
│   ├── AI_POLICY.md
│   ├── PROJECT_CONTEXT.md
│   ├── ARCHITECTURE.md
│   ├── DEPLOYMENT.md
│   ├── TESTING_STRATEGY.md
│   ├── PROJECT_MANIFEST.yaml
│   └── CURRENT_STATE.md
├── utils/                       # Shared utilities
│   ├── __init__.py
│   └── README.md
├── tests/                       # Test suite (pytest)
│   ├── __init__.py
│   ├── test_controller.py
│   └── test_telegram_bot.py
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Test configuration
└── README.md                    # Project overview
```

### Per-Project Structure (Future)

```
projects/project-name/
├── PROJECT_MANIFEST.yaml        # Single source of truth
├── AI_POLICY.md                 # Project-specific policy
├── PROJECT_CONTEXT.md           # Business memory
├── ARCHITECTURE.md              # Technical architecture
├── DEPLOYMENT.md                # Deployment configuration
├── TESTING_STRATEGY.md          # Testing approach
├── CURRENT_STATE.md             # Living system state
└── repo/                        # Actual project code (Git)
```

---

## Tech Stack (CONFIRMED)

| Component        | Technology                          | Status    |
|------------------|-------------------------------------|-----------|
| Chat Bot         | Python (python-telegram-bot)        | Confirmed |
| Task Controller  | Python + FastAPI                    | Confirmed |
| Claude CLI Agent | Claude CLI (Anthropic)              | Confirmed |
| CI/CD            | GitHub Actions                      | Confirmed |
| Repository       | GitHub (rahul4webdev)               | Confirmed |

### Hosting Infrastructure (CONFIRMED)

| Component        | Technology                          |
|------------------|-------------------------------------|
| VPS OS           | AlmaLinux 9                         |
| Control Panel    | CyberPanel                          |
| Web Server       | OpenLiteSpeed                       |
| Testing Domain   | aitesting.mybd.in                   |
| Production Domain| ai.mybd.in                          |

---

## Coding Standards

### General
- Use consistent code formatting (Prettier/Black/etc. based on language)
- All code must pass linting before merge
- Write meaningful commit messages
- Keep functions small and focused
- Document public APIs

### Python
- Follow PEP 8
- Use type hints
- Use `black` for formatting
- Use `flake8` or `ruff` for linting

### JavaScript/TypeScript
- Use ESLint with standard config
- Use Prettier for formatting
- Prefer TypeScript over JavaScript
- Use async/await over callbacks

---

## API Conventions

- RESTful endpoints for Task Controller
- Use JSON for request/response bodies
- Standard HTTP status codes
- Consistent error response format:
  ```json
  {
    "error": {
      "code": "ERROR_CODE",
      "message": "Human readable message",
      "details": {}
    }
  }
  ```

---

## Design Patterns

1. **File-Based State**: All persistent state lives in files, not databases (for simplicity)
2. **Event-Driven Notifications**: Chat bot receives events, not polling
3. **Immutable Deployments**: Each deployment is a new build, not in-place updates
4. **Environment Isolation**: Strict separation between dev, test, and production

---

## Confirmed Decisions

- Python for bot and controller (confirmed)
- File-based state; can migrate to database if needed
- Single VPS deployment for MVP; can scale horizontally later
- AlmaLinux 9 + CyberPanel + OpenLiteSpeed for hosting

## Pending Items

- Telegram bot token (to be created later)
- Server path and credentials (to be provided per project)

---

## Implementation Status

### Phase 1: Safe Skeleton Setup - COMPLETE

| Component | Status | File(s) |
|-----------|--------|---------|
| Task Controller | Skeleton | controller/main.py |
| Telegram Bot | Skeleton | bots/telegram_bot.py |
| CI/CD Pipeline | Skeleton | workflows/ci.yml |
| Unit Tests | Scaffold | tests/*.py |
| Documentation | Updated | docs/*.md |

### Phase 2: Autonomous Project Bootstrap Engine - COMPLETE

| Component | Status | File(s) |
|-----------|--------|---------|
| Project Bootstrap | Implemented | controller/main.py |
| Task Lifecycle | Implemented | controller/main.py |
| Plan Generation | Implemented | controller/main.py |
| Approval Gates | Implemented | controller/main.py |
| Telegram Bot v2 | Extended | bots/telegram_bot.py |
| Policy Hooks | Implemented | controller/main.py |
| Unit Tests | Complete | tests/*.py |

### Phase 3: Controlled Code Generation & Diff Engine - COMPLETE

| Component | Status | File(s) |
|-----------|--------|---------|
| DIFF_GENERATED State | Implemented | controller/main.py |
| Diff Generation Endpoint | Implemented | controller/main.py |
| Diff Content Templates | Implemented | controller/main.py |
| Diff Policy Hooks | Implemented | controller/main.py |
| Telegram Bot v3 | Extended | bots/telegram_bot.py |
| Unit Tests | Complete | tests/*.py |

### Phase 4: Human-Supervised Execution Engine - COMPLETE

| Component | Status | File(s) |
|-----------|--------|---------|
| Execution States | Implemented | controller/main.py |
| Dry-Run Endpoint | Implemented | controller/main.py |
| Apply Endpoint | Implemented | controller/main.py |
| Rollback Endpoint | Implemented | controller/main.py |
| Backup Strategy | Implemented | controller/main.py |
| Failure Handling | Implemented | controller/main.py |
| Telegram Bot v4 | Extended | bots/telegram_bot.py |
| Execution Policy Hooks | Implemented | controller/main.py |
| Unit Tests | Complete | tests/*.py |

---

## Phase 4 Architecture

### Task Lifecycle State Machine (Phase 4 - Complete)

```
                    ┌─────────────────────────────────────────────────────────────────────────────────────┐
                    │                                                                                     │
                    ▼                                                                                     │
┌──────────┐    ┌──────────┐    ┌────────┐    ┌─────────────────┐    ┌──────────┐    ┌────────────────┐   │
│ RECEIVED │───▶│ VALIDATED│───▶│ PLANNED│───▶│AWAITING_APPROVAL│───▶│ APPROVED │───▶│ DIFF_GENERATED │   │
└──────────┘    └──────────┘    └────────┘    └─────────────────┘    └──────────┘    └────────────────┘   │
     │               │                              │                      │                  │           │
     │               │                              │                      │                  ▼           │
     │               │                              │                      │         ┌────────────────┐   │
     │               │                              │                      │         │ READY_TO_APPLY │   │
     │               │                              │                      │         └────────────────┘   │
     │               │                              │                      │                  │           │
     │               │                              │                      │                  ▼           │
     │               │                              │                      │          ┌────────────┐      │
     │               │                              │                      │          │  APPLYING  │      │
     │               │                              │                      │          └────────────┘      │
     │               │                              │                      │             │      │         │
     │               │                              │                      │    success  │      │ failure │
     │               │                              │                      │             ▼      ▼         │
     │               │                              │                      │       ┌─────────┐ ┌─────────────────┐
     │               │                              │                      │       │ APPLIED │ │EXECUTION_FAILED │
     │               │                              │                      │       └─────────┘ └─────────────────┘
     │               │                              │                      │             │                │
     │               │                              │                      │    /rollback│                │
     │               │                              │                      │             ▼                │
     │               │                              │                      │       ┌─────────────┐        │
     │               │                              │                      │       │ ROLLED_BACK │        │
     │               │                              │                      │       └─────────────┘        │
     └───────────────┴──────────────────────────────┴──────────────────────┴──────────────────────────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ REJECTED │
                              └──────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ ARCHIVED │
                              └──────────┘
```

### Human-Supervised Execution Model

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ DIFF_GENERATED  │───▶│ /dry_run        │───▶│ READY_TO_APPLY  │
│ (human reviewed)│    │ (simulation)    │    │ (preview OK)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────────┐
                                            │ /apply CONFIRM      │
                                            │ (explicit keyword)  │
                                            └─────────────────────┘
                                                      │
                                      ┌───────────────┴───────────────┐
                                      │                               │
                                      ▼                               ▼
                              ┌─────────────┐               ┌─────────────────┐
                              │ Backup      │               │ (on failure)    │
                              │ Created     │               │ Auto-Restore    │
                              └─────────────┘               └─────────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │ Files       │
                              │ Modified    │
                              └─────────────┘
                                      │
                                      ▼
                              ┌─────────────┐
                              │ APPLIED     │
                              └─────────────┘
                                      │
                           (optional) │ /rollback
                                      ▼
                              ┌─────────────┐
                              │ ROLLED_BACK │
                              └─────────────┘
```

**CRITICAL SAFETY CONSTRAINTS (Phase 4)**:
- **CONFIRMATION REQUIRED**: Apply requires explicit `confirm=true` parameter
- **BACKUP MANDATORY**: Files are backed up BEFORE any modification
- **ROLLBACK GUARANTEED**: Any applied changes can be reversed
- **AUTOMATIC RESTORE ON FAILURE**: If apply fails, backup is restored automatically
- **NO AUTONOMOUS EXECUTION**: Human triggers every execution step
- **NO PRODUCTION DEPLOYMENT**: Still blocked by policy

### Per-Project Structure (Phase 4 Extended)

```
projects/{project-name}/
├── PROJECT_MANIFEST.yaml   # Project metadata (repo, tech stack, phase)
├── CURRENT_STATE.md        # Living state file
├── tasks/                  # Task YAML files
│   └── {task-id}.yaml      # Task record (state, description, timestamps)
├── plans/                  # Generated plan files
│   └── {task-id}_plan.md   # Implementation plan (markdown)
├── diffs/                  # Phase 3: Generated diff files
│   └── {task-id}.diff      # Unified diff format
└── backups/                # Phase 4: Backup snapshots
    └── {task-id}/          # Per-task backup directory
        ├── BACKUP_MANIFEST.yaml  # Backup metadata
        └── {relative_path}       # Original file copies
```

### Backup Manifest Structure (Phase 4)

```yaml
task_id: <uuid>
project: <project_name>
created_at: <iso_timestamp>
files:
  - path: <relative_path>
    original_size: <bytes>
    backed_up: true
```

### API Endpoints (Phase 4 Extended)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | Service info |
| /health | GET | Health check with capabilities & constraints |
| /project/bootstrap | POST | Create new project |
| /task | POST | Create task (state: RECEIVED) |
| /task/{id}/validate | POST | Validate task (RECEIVED → VALIDATED) |
| /task/{id}/plan | POST | Generate plan (VALIDATED → AWAITING_APPROVAL) |
| /task/{id}/approve | POST | Approve plan (AWAITING_APPROVAL → APPROVED) |
| /task/{id}/reject | POST | Reject plan (→ REJECTED) |
| /task/{id}/generate-diff | POST | Generate diff (APPROVED → DIFF_GENERATED) |
| /task/{id}/diff | GET | Retrieve diff content |
| /task/{id}/dry-run | POST | Simulate diff application (DIFF_GENERATED → READY_TO_APPLY) **Phase 4** |
| /task/{id}/apply | POST | Apply diff with backup (READY_TO_APPLY → APPLIED) **Phase 4** |
| /task/{id}/rollback | POST | Restore from backup (APPLIED → ROLLED_BACK) **Phase 4** |
| /status/{project} | GET | Project status with task counts |
| /projects | GET | List all projects |
| /deploy | POST | Deploy (testing only, production BLOCKED) |

### Telegram Bot Commands (Phase 4 Extended)

| Command | Purpose |
|---------|---------|
| /start | Welcome with Phase 4 info |
| /help | Show all commands |
| /bootstrap <name> <url> [tech] | Create new project |
| /project <name> | Switch project context |
| /task <description> | Create task |
| /validate [task-id\|last] | Validate task |
| /plan [task-id\|last] | Generate plan |
| /approve [task-id\|last] | Approve plan |
| /reject <task-id> <reason> | Reject plan (min 10 chars) |
| /generate_diff [task-id\|last] | Generate diff (NOT applied) |
| /dry_run [task-id\|last] | Simulate diff application **Phase 4** |
| /apply <task-id> confirm | Apply diff (REQUIRES confirm keyword) **Phase 4** |
| /rollback [task-id\|last] | Restore from backup **Phase 4** |
| /status | Show project status |
| /list | List all projects |
| /deploy testing | Deploy to testing (production BLOCKED) |

### Policy Enforcement Hooks (Phase 4 Extended)

| Hook | Purpose |
|------|---------|
| can_create_project() | Gate project creation |
| can_submit_task() | Gate task submission |
| can_validate_task() | Gate task validation |
| can_plan_task() | Gate plan generation |
| can_approve_task() | Gate task approval |
| can_reject_task() | Gate task rejection |
| can_generate_diff() | Gate diff generation |
| diff_within_scope() | Verify diff files match plan |
| diff_file_limit_ok() | Enforce max 10 files per diff |
| can_dry_run() | Gate dry-run execution **Phase 4** |
| can_apply(confirmed) | Gate apply (REQUIRES confirmed=True) **Phase 4** |
| can_rollback() | Gate rollback execution **Phase 4** |

---

## Phase 3 Architecture

### Task Lifecycle State Machine (Updated)

```
                    ┌─────────────────────────────────────────────────────┐
                    │                                                     │
                    ▼                                                     │
┌──────────┐    ┌──────────┐    ┌────────┐    ┌─────────────────┐    ┌──────────┐    ┌────────────────┐
│ RECEIVED │───▶│ VALIDATED│───▶│ PLANNED│───▶│AWAITING_APPROVAL│───▶│ APPROVED │───▶│ DIFF_GENERATED │
└──────────┘    └──────────┘    └────────┘    └─────────────────┘    └──────────┘    └────────────────┘
     │               │                              │                      │                  │
     │               │                              │                      │                  │
     └───────────────┴──────────────────────────────┴──────────────────────┴──────────────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ REJECTED │
                              └──────────┘
                                    │
                                    ▼
                              ┌──────────┐
                              │ ARCHIVED │
                              └──────────┘
```

**IMPORTANT**: There is still NO `EXECUTED` state. Phase 3 generates diffs but does NOT apply them.

### Diff-Only Safety Model

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ APPROVED Task   │───▶│ Generate Diff   │───▶│ Diff File       │
│ (human approved)│    │ (no execution)  │    │ (for review)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────────┐
                                            │ Human Review        │
                                            │ (mandatory)         │
                                            └─────────────────────┘
                                                      │
                                                      ▼
                                            ┌─────────────────────┐
                                            │ Future Phase:       │
                                            │ Manual Apply        │
                                            │ (not in Phase 3)    │
                                            └─────────────────────┘
```

**SAFETY CONSTRAINTS**:
- Diffs are NEVER applied automatically
- Human review is MANDATORY
- No code execution
- No git commits
- No file modifications

### Data Flow

```
User (Telegram)
      │
      ▼
┌─────────────┐     HTTP/REST      ┌─────────────────┐
│ Telegram Bot │ ◀──────────────▶ │ Task Controller │
└─────────────┘                    └─────────────────┘
      │                                    │
      │                                    ▼
      │                            ┌─────────────────┐
      │                            │ Filesystem      │
      │                            │ - projects/     │
      │                            │ - tasks/        │
      │                            │ - plans/        │
      │                            │ - diffs/        │
      │                            └─────────────────┘
      │
      ▼
  Notifications
  (task created, plan ready, approval needed)
```

### Per-Project Structure (Phase 3 Extended)

```
projects/{project-name}/
├── PROJECT_MANIFEST.yaml   # Project metadata (repo, tech stack, phase)
├── CURRENT_STATE.md        # Living state file
├── tasks/                  # Task YAML files
│   └── {task-id}.yaml      # Task record (state, description, timestamps)
├── plans/                  # Generated plan files
│   └── {task-id}_plan.md   # Implementation plan (markdown)
└── diffs/                  # Phase 3: Generated diff files
    └── {task-id}.diff      # Unified diff format (NOT APPLIED)
```

### Diff File Structure (Phase 3)

Each diff file contains:
```
# TASK_ID: <uuid>
# PROJECT: <project_name>
# GENERATED_AT: <iso_timestamp>
# PLAN_REF: plans/<task_id>_plan.md
# DISCLAIMER: NOT APPLIED. FOR HUMAN REVIEW ONLY.

--- a/src/file.py
+++ b/src/file.py
@@ -1,5 +1,10 @@
 # Changes here...
```

### API Endpoints (Phase 3)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| / | GET | Service info |
| /health | GET | Health check with capabilities & constraints |
| /project/bootstrap | POST | Create new project |
| /task | POST | Create task (state: RECEIVED) |
| /task/{id}/validate | POST | Validate task (RECEIVED → VALIDATED) |
| /task/{id}/plan | POST | Generate plan (VALIDATED → AWAITING_APPROVAL) |
| /task/{id}/approve | POST | Approve plan (AWAITING_APPROVAL → APPROVED) |
| /task/{id}/reject | POST | Reject plan (→ REJECTED) |
| /task/{id}/generate-diff | POST | Generate diff (APPROVED → DIFF_GENERATED) **Phase 3** |
| /task/{id}/diff | GET | Retrieve diff content **Phase 3** |
| /status/{project} | GET | Project status with task counts |
| /projects | GET | List all projects |
| /deploy | POST | Deploy (testing only, production BLOCKED) |

### Telegram Bot Commands (Phase 3)

| Command | Purpose |
|---------|---------|
| /start | Welcome with Phase 3 info |
| /help | Show all commands |
| /bootstrap <name> <url> [tech] | Create new project |
| /project <name> | Switch project context |
| /task <description> | Create task |
| /validate [task-id\|last] | Validate task |
| /plan [task-id\|last] | Generate plan |
| /approve [task-id\|last] | Approve plan |
| /reject <task-id> <reason> | Reject plan (min 10 chars) |
| /generate_diff [task-id\|last] | Generate diff (NOT applied) **Phase 3** |
| /status | Show project status |
| /list | List all projects |
| /deploy testing | Deploy to testing (production BLOCKED) |

### Policy Enforcement Hooks (Phase 3 Extended)

All policy hooks are implemented as stubs that:
1. Log the decision being made
2. Return `(True, "reason")` for now
3. Can be extended to enforce real policies later

| Hook | Purpose |
|------|---------|
| can_create_project() | Gate project creation |
| can_submit_task() | Gate task submission |
| can_validate_task() | Gate task validation |
| can_plan_task() | Gate plan generation |
| can_approve_task() | Gate task approval |
| can_reject_task() | Gate task rejection |
| can_generate_diff() | Gate diff generation **Phase 3** |
| diff_within_scope() | Verify diff files match plan **Phase 3** |
| diff_file_limit_ok() | Enforce max 10 files per diff **Phase 3** |

---

### What is READY (Phase 4)

- Project bootstrap with manifest and state files
- Task lifecycle management (create, validate, plan, approve, reject)
- Plan generation (markdown templates with risk analysis, rollback)
- Approval gates with reason requirement for rejection
- Diff generation (unified diff format)
- Diff retrieval endpoint
- **Dry-run simulation (preview changes without modifying files)**
- **Apply with confirmation (REQUIRES explicit confirm=true)**
- **Backup creation before any apply**
- **Rollback from backup (guaranteed reversibility)**
- **Automatic restore on apply failure**
- Extended Telegram bot with task lifecycle + execution commands
- Policy enforcement hooks (stubs, logs decisions)
- Comprehensive test coverage

### What is NOT Implemented (Intentionally)

- **NO AUTONOMOUS EXECUTION**: Human must trigger every execution step
- **NO PRODUCTION DEPLOYMENT**: BLOCKED by policy
- **NO GIT COMMITS**: Applied changes are NOT automatically committed
- Claude CLI integration (future phase)
- Telegram API connection (no token yet)
- Database/persistent storage (file-based only)
- Authentication/authorization (policy hooks are stubs)
- Rate limiting

---

*This file prevents architectural drift. Update only through explicit architecture decisions.*
