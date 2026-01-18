"""
Task Controller Module

Central orchestrator for the AI-driven autonomous development platform.
Handles task management, project state tracking, and coordination between
the Telegram bot and Claude CLI agent.

Phase 14: Claude CLI Bootstrap & Full Integration
- Phase 12 features: Multi-aspect projects, IPC, lifecycle management
- Phase 13: Telegram bot with RBAC, health monitoring, project management
- Phase 14: Claude CLI as execution backend, job workspaces, autonomous development
"""

__version__ = "0.14.0"

# Single source of truth for phase metadata
CURRENT_PHASE = "14.0"
CURRENT_PHASE_NAME = "Claude CLI Integration"
CURRENT_PHASE_FULL = f"Phase {CURRENT_PHASE}: {CURRENT_PHASE_NAME}"
