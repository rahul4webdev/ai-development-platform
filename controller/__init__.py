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
"""

__version__ = "0.16.4"

# Single source of truth for phase metadata
CURRENT_PHASE = "16E"
CURRENT_PHASE_NAME = "Project Identity, Fingerprinting & Conflict Resolution"
CURRENT_PHASE_FULL = f"Phase {CURRENT_PHASE}: {CURRENT_PHASE_NAME}"
