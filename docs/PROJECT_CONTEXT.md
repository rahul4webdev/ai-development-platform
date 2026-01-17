# Project Context (Business Memory)

This file contains persistent project knowledge that the AI agent MUST read at the start of every session.

---

## Project Purpose

Build an **AI-Driven Autonomous Development Platform** that enables near-fully autonomous software development where:
- A comprehensive project handoff document is provided once
- An AI agent (Claude CLI) performs development, testing, deployment, and iteration
- Human involvement is limited to high-level feedback, validation, and production promotion

---

## Target Users

1. **Software Project Owners**: Individuals or teams who want to minimize hands-on development effort
2. **Solo Developers**: Developers managing multiple projects who need automation
3. **Technical Managers**: Those who want to oversee development progress from mobile/desktop chat

---

## Business Goals

1. Enable project management entirely from mobile or desktop chat interface
2. Minimize human micromanagement of development tasks
3. Avoid repeated instructions by persisting project memory in files
4. Support multiple concurrent projects with full isolation
5. Reduce human effort to validation-only for the development process
6. Move features from idea to production with minimal friction

---

## Key Capabilities

1. **Chat Interface**: Telegram bot (primary) or Discord bot for user interaction
2. **Task Controller**: Central orchestrator that maps chat messages to tasks
3. **Claude CLI Agent**: Executes development, testing, and deployment tasks
4. **Multi-Project Support**: Isolated project directories with independent state
5. **Environment Separation**: Dev → Test → Production pipeline
6. **Persistent Memory**: File-based state that survives agent restarts
7. **Task Lifecycle**: Full state machine from task receipt to approval
8. **Plan Generation**: AI generates implementation plans for human review
9. **Approval Gates**: Human must approve plans before execution
10. **Diff Generation**: AI generates code diffs (unified format) for human review
11. **Human-Supervised Execution**: Dry-run, apply (with confirmation), and rollback
12. **Guaranteed Rollback**: Backup before apply, restore on demand or failure

---

## Task Lifecycle (Phase 4)

The system implements a strict task lifecycle with human approval gates and supervised execution:

```
User creates task via Telegram
         │
         ▼
   ┌──────────┐
   │ RECEIVED │  Task is logged, assigned ID
   └──────────┘
         │ /validate
         ▼
   ┌──────────┐
   │ VALIDATED│  Task description is valid
   └──────────┘
         │ /plan
         ▼
   ┌────────┐
   │ PLANNED│  Implementation plan generated
   └────────┘
         │
         ▼
┌─────────────────┐
│AWAITING_APPROVAL│  Human reviews plan
└─────────────────┘
    │           │
    │ /approve  │ /reject
    ▼           ▼
┌──────────┐ ┌──────────┐
│ APPROVED │ │ REJECTED │
└──────────┘ └──────────┘
    │
    │ /generate_diff
    ▼
┌────────────────┐
│ DIFF_GENERATED │  Code diff created
└────────────────┘
    │
    │ /dry_run (Phase 4)
    ▼
┌────────────────┐
│ READY_TO_APPLY │  Simulation successful
└────────────────┘
    │
    │ /apply confirm (Phase 4)
    ▼
┌────────────────┐
│    APPLIED     │  Files modified (backup created)
└────────────────┘
    │
    │ /rollback (optional, Phase 4)
    ▼
┌────────────────┐
│  ROLLED_BACK   │  Original files restored
└────────────────┘
    │
    ▼
┌──────────┐
│ ARCHIVED │  Task complete
└──────────┘
```

**Phase 4 Execution Flow**:
1. `/dry_run` - Simulate diff application (no files modified)
2. `/apply confirm` - Apply diff with explicit confirmation (backup created first)
3. `/rollback` - Restore from backup if needed (always available)

### Approval Flow

1. **Task Created**: User submits task via `/task` command
2. **Validation**: System validates task is well-formed (`/validate`)
3. **Planning**: AI generates implementation plan (`/plan`)
4. **Review**: Human reviews plan (markdown file in `plans/` directory)
5. **Decision**: Human approves (`/approve`) or rejects (`/reject <reason>`)
6. **Archive**: Completed or rejected tasks can be archived

### Rejection Requirements

- Rejections MUST include a reason (minimum 10 characters)
- Reason is stored with the task record
- Rejected tasks can be revised and resubmitted

---

## Human-Supervised Execution Model (Phase 4)

Phase 4 implements a "human-supervised execution" safety model where:

1. **Plans become Diffs**: Approved plans are converted to unified diff format
2. **Dry-Run First**: Human runs simulation before any modification
3. **Explicit Confirmation**: Apply REQUIRES the `confirm` keyword
4. **Backup Before Apply**: Files are backed up BEFORE any modification
5. **Rollback Guaranteed**: Any applied changes can be reversed
6. **Automatic Restore on Failure**: If apply fails, backup is restored automatically
7. **No Git Commits**: Applied changes are NOT automatically committed
8. **No Autonomous Execution**: Human must trigger every execution step

### Execution Artifact Structure

```
projects/<project_name>/
├── diffs/<task_id>.diff       # Generated diff (unified format)
└── backups/<task_id>/          # Backup directory
    ├── BACKUP_MANIFEST.yaml    # Backup metadata
    └── <relative_path>         # Original file copies
```

### Execution Workflow (Phase 4)

1. **Review Diff**: Human reviews diff in `projects/{name}/diffs/{task_id}.diff`
2. **Dry-Run**: Human runs `/dry_run` to simulate changes (no files modified)
3. **Apply with Confirm**: Human runs `/apply <task_id> confirm` (explicit confirmation)
4. **Backup Created**: System creates backup BEFORE modifying files
5. **Files Modified**: Diff is applied to project files
6. **Verify**: Human verifies changes
7. **Rollback (if needed)**: Human runs `/rollback` to restore original files
8. **Commit (manual)**: Human commits changes manually (not automated)

### Safety Guarantees

- **CONFIRMATION_REQUIRED**: Cannot apply without explicit `confirm` keyword
- **BACKUP_REQUIRED**: Every apply creates backup first
- **ROLLBACK_GUARANTEED**: Any applied change can be reversed
- **AUTOMATIC_RESTORE**: On failure, backup is restored automatically
- **NO_AUTONOMOUS_EXECUTION**: Human triggers every step
- **NO_PRODUCTION_DEPLOYMENT**: Still blocked by policy

---

## Constraints

- Chat interface never directly executes code (communicates only with Task Controller)
- Production deployment requires explicit human trigger
- All state must persist in files, not in chat sessions
- CI/CD cannot be bypassed by the agent
- Secrets must never be stored in the repository
- Agent must be stateless; system state lives in files
- **Phase 4**: Apply REQUIRES explicit confirmation keyword
- **Phase 4**: Backup REQUIRED before any file modification
- **Phase 4**: Rollback ALWAYS available
- **Phase 4**: No autonomous execution (human triggers every step)
- **Phase 4**: Max 10 files per diff (enforced by policy)

---

## Non-Goals

This system is NOT:
- A fully unsupervised AI CTO (human oversight required for business decisions)
- A replacement for business/product decisions
- A blind production automation tool (human validation gates exist)
- A system that requires constant human supervision

---

## Success Criteria

The system is successful when:
- [ ] Projects can be managed entirely from mobile chat
- [ ] Claude works without repeated instructions
- [ ] Features move from idea → production with minimal friction
- [ ] Human effort is reduced to validation only
- [ ] Multiple projects run concurrently without interference

---

## Assumptions

- [ASSUMPTION] Primary chat interface is Telegram Bot
- [ASSUMPTION] VPS hosting for Claude CLI agent (provider TBD)
- [ASSUMPTION] Git repository hosting is GitHub (can be changed)
- [ASSUMPTION] Target projects are web applications (initial focus)

---

*This file is the business memory. Update when project scope or goals change.*
