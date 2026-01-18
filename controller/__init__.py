"""
Task Controller Module

Central orchestrator for the AI-driven autonomous development platform.
Handles task management, project state tracking, and coordination between
the Telegram bot and Claude CLI agent.

Phase 13.9: Telegram Bot Integration Complete
- Phase 12 features: Multi-aspect projects, IPC, lifecycle management
- Phase 13: Telegram bot with RBAC, health monitoring, project management
"""

__version__ = "0.13.9"

# Single source of truth for phase metadata
CURRENT_PHASE = "13.9"
CURRENT_PHASE_NAME = "Telegram Bot Integration"
CURRENT_PHASE_FULL = f"Phase {CURRENT_PHASE}: {CURRENT_PHASE_NAME}"
