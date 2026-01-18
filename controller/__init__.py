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
"""

__version__ = "0.15.2"

# Single source of truth for phase metadata
CURRENT_PHASE = "15.2"
CURRENT_PHASE_NAME = "Continuous Change Cycles"
CURRENT_PHASE_FULL = f"Phase {CURRENT_PHASE}: {CURRENT_PHASE_NAME}"
