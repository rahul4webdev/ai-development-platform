"""
Telegram Bot - Phase 13.3-16E

Production Control Plane for AI Development Platform.

Features:
- Health awareness (13.3)
- Role-aware command model (13.4)
- Human approval workflow (13.5)
- Deployment notifications (13.6)
- CI trigger rules (13.7)
- Dashboard API integration (13.8)
- Multi-aspect project support (13.9)
- Rate limiting (13.10)
- Timeouts & retries (13.11)
- Degraded mode handling (13.12)
- Claude CLI job management (14.0)
- Autonomous lifecycle engine (15.1)
- Continuous change cycles (15.2)
- Project ingestion & adoption (15.3)
- Project identity & fingerprinting (16E)
- Conflict detection & resolution UX (16E)

Safety:
- Bot cannot trigger prod deploy directly
- Bot cannot skip CI
- Bot cannot self-approve
- All actions logged
- Dual approval rules enforced via controller
- Rate limiting prevents abuse
- Degraded mode protects system integrity
- Claude jobs run in isolated workspaces
"""

import asyncio
import logging
import os
import sys
import subprocess
import resource
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Optional, Dict, Any, List, Callable, Tuple

import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------
LOG_FILE = os.getenv("TELEGRAM_BOT_LOG", "/var/log/ai-telegram-bot.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logger = logging.getLogger("telegram_bot")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
except PermissionError:
    pass

console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(console_handler)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CONTROLLER_BASE_URL = os.getenv("CONTROLLER_URL", "http://127.0.0.1:8001")
HTTP_TIMEOUT = 30.0
BOT_VERSION = "0.19.0"
BOT_START_TIME = datetime.utcnow()

# -----------------------------------------------------------------------------
# Phase 13.11: Timeout & Retry Configuration
# -----------------------------------------------------------------------------
API_TIMEOUT_DEFAULT = 30.0  # seconds
API_TIMEOUT_CI_CHECK = 60.0  # CI checks may take longer
API_MAX_RETRIES = 3
API_RETRY_BACKOFF_BASE = 1.0  # seconds, doubles each retry
# Actions that should NOT be retried (destructive/stateful)
NO_RETRY_ACTIONS = frozenset([
    "approve_production",
    "submit_feedback",
    "create_project",
    "deploy",
])


def escape_markdown(text: str) -> str:
    """
    Escape special characters for Telegram Markdown (v1) parsing.

    Characters that need escaping: _ * ` [ ] ( )
    Parentheses and brackets can cause issues with link parsing.
    """
    if not text:
        return text
    # Escape backslash first, then other special chars
    for char in ['\\', '_', '*', '`', '[', ']', '(', ')']:
        text = text.replace(char, f'\\{char}')
    return text


def extract_api_error(result: dict) -> str:
    """
    Extract a human-readable error message from API response.

    API returns errors in format:
    {
        "error": {
            "code": "HTTP_400",
            "message": {"error": "...", "errors": [...], "suggestions": [...]}
        }
    }
    or
    {
        "error": "simple string message"
    }
    or for conflict detection (Phase 16E):
    {
        "success": false,
        "message": "Conflict detected...",
        "metadata": {
            "conflict_detected": true,
            "decision": {"explanation": "...", ...}
        }
    }
    """
    if not result:
        return "Unknown error"

    # Phase 16E: Check for conflict detection response (no error field)
    metadata = result.get("metadata", {})
    if metadata.get("conflict_detected"):
        decision = metadata.get("decision", {})
        explanation = decision.get("explanation", "")
        if explanation:
            return explanation
        # Fallback to message field
        message = result.get("message", "")
        if message:
            return message

    # Check for redirect to change mode
    if metadata.get("redirect_to_change_mode"):
        existing = metadata.get("existing_project", "")
        message = result.get("message", "")
        if message:
            return f"{message} (existing project: {existing})"

    error = result.get("error")
    if not error:
        # No error field - check for message field
        message = result.get("message", "")
        if message and "conflict" in message.lower():
            return message
        return "Unknown error"

    # If error is a simple string
    if isinstance(error, str):
        return error

    # If error is a dict (structured API response)
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
        if isinstance(message, dict):
            # Extract the main error and any sub-errors
            main_error = message.get("error", "")
            errors = message.get("errors", [])
            if errors:
                return f"{main_error}: {'; '.join(errors)}"
            return main_error or "Unknown error"
        return str(error.get("code", "Unknown error"))

    return str(error)


# -----------------------------------------------------------------------------
# Phase 13.4: Role System
# -----------------------------------------------------------------------------
class UserRole(str, Enum):
    """User roles for access control."""
    OWNER = "owner"
    ADMIN = "admin"
    DEVELOPER = "developer"  # Phase 15.2: Added for change request commands
    TESTER = "tester"
    VIEWER = "viewer"


# Role configuration - loaded from environment or config
# Format: ROLE_OWNERS=123456,789012 ROLE_ADMINS=111111,222222
def load_role_config() -> Dict[UserRole, List[int]]:
    """Load role mappings from environment."""
    config = {
        UserRole.OWNER: [],
        UserRole.ADMIN: [],
        UserRole.DEVELOPER: [],
        UserRole.TESTER: [],
        UserRole.VIEWER: [],
    }

    for role in UserRole:
        env_key = f"ROLE_{role.value.upper()}S"
        ids_str = os.getenv(env_key, "")
        if ids_str:
            try:
                config[role] = [int(uid.strip()) for uid in ids_str.split(",") if uid.strip()]
            except ValueError as e:
                logger.warning(f"Invalid user ID in {env_key}: {e}")

    return config


ROLE_CONFIG = load_role_config()


def get_user_role(user_id: int) -> UserRole:
    """Get the highest role for a user."""
    for role in [UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER]:
        if user_id in ROLE_CONFIG.get(role, []):
            return role
    return UserRole.VIEWER


def role_required(*allowed_roles: UserRole):
    """Decorator to enforce role-based access control."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            try:
                user_id = update.effective_user.id
                user_role = get_user_role(user_id)

                if user_role not in allowed_roles:
                    await update.message.reply_text(
                        f"Access denied. Required role: " + ", ".join(r.value for r in allowed_roles) + "\n"
                        f"Your role: {user_role.value}"
                    )
                    logger.warning(f"Access denied for user {user_id} (role: {user_role}) to {func.__name__}")
                    return

                return await func(update, context, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}")
                if update and update.message:
                    await update.message.reply_text(f"Error: {str(e)}")
        return wrapper
    return decorator


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class FeedbackType(str, Enum):
    """Types of feedback from testers."""
    APPROVE = "approve"
    BUG = "bug"
    IMPROVEMENTS = "improvements"
    REJECT = "reject"


class RejectionReason(str, Enum):
    """Structured rejection reasons."""
    BUG = "bug"
    IMPROVEMENT = "improvement"
    INVALID_REQUIREMENT = "invalid_requirement"
    OTHER = "other"


class DeploymentState(str, Enum):
    """Deployment state machine."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED_TESTING = "deployed_testing"
    DEPLOYED_PRODUCTION = "deployed_production"
    FEEDBACK_REQUIRED = "feedback_required"
    ROLLED_BACK = "rolled_back"


class CallbackAction(str, Enum):
    """Callback query actions."""
    FEEDBACK_APPROVE = "fb_approve"
    FEEDBACK_BUG = "fb_bug"
    FEEDBACK_IMPROVEMENTS = "fb_improve"
    FEEDBACK_REJECT = "fb_reject"
    PROD_APPROVE = "prod_approve"
    SELECT_PROJECT = "sel_proj"
    SELECT_ASPECT = "sel_aspect"
    REJECT_BUG = "rej_bug"
    REJECT_IMPROVEMENT = "rej_improve"
    REJECT_INVALID = "rej_invalid"
    REJECT_OTHER = "rej_other"
    # Phase 16E: Conflict Resolution Actions
    CONFLICT_IMPROVE = "conf_improve"      # Improve existing project
    CONFLICT_ADD_MODULE = "conf_addmod"    # Add new module to existing
    CONFLICT_NEW_VERSION = "conf_newver"   # Create new version
    CONFLICT_CANCEL = "conf_cancel"        # Cancel project creation


# -----------------------------------------------------------------------------
# User State Management
# -----------------------------------------------------------------------------
class UserState:
    """Temporary state for multi-step interactions."""
    def __init__(self):
        self.pending_feedback: Dict[int, Dict[str, Any]] = {}
        self.pending_project_creation: Dict[int, str] = {}
        self.awaiting_input: Dict[int, str] = {}
        self.approval_history: Dict[str, List[Dict[str, Any]]] = {}  # deployment_id -> approvals
        # Phase 16E: Conflict resolution state
        self.pending_conflicts: Dict[int, Dict[str, Any]] = {}  # user_id -> conflict details


user_state = UserState()


# -----------------------------------------------------------------------------
# Safety Guardrails
# -----------------------------------------------------------------------------
class SafetyGuardrails:
    """Safety constraints enforced by the Telegram bot."""

    @staticmethod
    def log_action(user_id: int, action: str, details: Dict[str, Any]) -> None:
        """Log all user actions for audit trail."""
        logger.info(f"ACTION: user={user_id} action={action} details={details}")

    @staticmethod
    def is_production_action(action: str) -> bool:
        """Check if action is production-related."""
        prod_actions = ["prod_approve", "prod_apply", "prod_deploy"]
        return action in prod_actions

    @staticmethod
    def validate_feedback_has_explanation(
        feedback_type: str,
        explanation: Optional[str]
    ) -> Tuple[bool, str]:
        """Ensure bug/improvements/reject feedback has explanation."""
        if feedback_type in ["bug", "improvements", "reject"]:
            if not explanation or len(explanation.strip()) < 10:
                return False, f"{feedback_type.title()} feedback requires explanation (min 10 chars)"
        return True, ""

    @staticmethod
    def check_dual_approval(deployment_id: str, user_id: int) -> Tuple[bool, str]:
        """Check if dual approval rules are satisfied."""
        approvals = user_state.approval_history.get(deployment_id, [])

        # Check if user already approved
        for approval in approvals:
            if approval.get("user_id") == user_id:
                return False, "You have already approved this deployment. Dual approval requires different users."

        return True, ""


safety = SafetyGuardrails()


# -----------------------------------------------------------------------------
# Phase 13.10: Rate Limiting
# -----------------------------------------------------------------------------
class RateLimiter:
    """
    Per-user, per-command rate limiting with role-based thresholds.
    Uses in-memory TTL structure with automatic cleanup.
    """

    # Role-based rate limits: (requests, window_seconds)
    ROLE_LIMITS: Dict[UserRole, Tuple[int, int]] = {
        UserRole.OWNER: (100, 60),    # 100 requests per minute
        UserRole.ADMIN: (60, 60),     # 60 requests per minute
        UserRole.TESTER: (30, 60),    # 30 requests per minute
        UserRole.VIEWER: (15, 60),    # 15 requests per minute
    }

    # Per-command limits (overrides role limit if stricter)
    COMMAND_LIMITS: Dict[str, Tuple[int, int]] = {
        "new_project": (5, 300),       # 5 per 5 minutes
        "approve_production": (3, 300), # 3 per 5 minutes
        "submit_feedback": (10, 60),    # 10 per minute
        "create_project": (5, 300),     # 5 per 5 minutes
    }

    def __init__(self):
        self._requests: Dict[int, List[datetime]] = defaultdict(list)
        self._command_requests: Dict[Tuple[int, str], List[datetime]] = defaultdict(list)
        self._lock = Lock()
        self._last_cleanup = datetime.utcnow()
        self._cleanup_interval = timedelta(minutes=5)

    def _cleanup_old_entries(self) -> None:
        """Remove expired entries to prevent memory growth."""
        now = datetime.utcnow()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        cutoff = now - timedelta(minutes=10)

        # Clean user requests
        for user_id in list(self._requests.keys()):
            self._requests[user_id] = [
                t for t in self._requests[user_id] if t > cutoff
            ]
            if not self._requests[user_id]:
                del self._requests[user_id]

        # Clean command requests
        for key in list(self._command_requests.keys()):
            self._command_requests[key] = [
                t for t in self._command_requests[key] if t > cutoff
            ]
            if not self._command_requests[key]:
                del self._command_requests[key]

        self._last_cleanup = now

    def check_rate_limit(
        self,
        user_id: int,
        user_role: UserRole,
        command: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if user is within rate limits.
        Returns (allowed, error_message).
        """
        with self._lock:
            self._cleanup_old_entries()
            now = datetime.utcnow()

            # Check role-based limit
            max_requests, window_seconds = self.ROLE_LIMITS.get(
                user_role, (15, 60)
            )
            window_start = now - timedelta(seconds=window_seconds)

            # Filter to requests within window
            self._requests[user_id] = [
                t for t in self._requests[user_id] if t > window_start
            ]

            if len(self._requests[user_id]) >= max_requests:
                safety.log_action(user_id, "rate_limit_exceeded", {
                    "type": "role_limit",
                    "role": user_role.value,
                    "limit": max_requests,
                    "window": window_seconds
                })
                return False, (
                    f"Rate limit exceeded. "
                    f"Please wait before making more requests. "
                    f"(Limit: {max_requests} per {window_seconds}s)"
                )

            # Check command-specific limit if applicable
            if command and command in self.COMMAND_LIMITS:
                cmd_max, cmd_window = self.COMMAND_LIMITS[command]
                cmd_start = now - timedelta(seconds=cmd_window)
                key = (user_id, command)

                self._command_requests[key] = [
                    t for t in self._command_requests[key] if t > cmd_start
                ]

                if len(self._command_requests[key]) >= cmd_max:
                    safety.log_action(user_id, "rate_limit_exceeded", {
                        "type": "command_limit",
                        "command": command,
                        "limit": cmd_max,
                        "window": cmd_window
                    })
                    return False, (
                        f"Command rate limit exceeded for /{command}. "
                        f"Please wait before trying again. "
                        f"(Limit: {cmd_max} per {cmd_window}s)"
                    )

                self._command_requests[key].append(now)

            # Record the request
            self._requests[user_id].append(now)
            return True, None


rate_limiter = RateLimiter()


def rate_limited(command_name: Optional[str] = None):
    """Decorator to apply rate limiting to handlers."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            user_id = update.effective_user.id
            user_role = get_user_role(user_id)

            allowed, error_msg = rate_limiter.check_rate_limit(
                user_id, user_role, command_name
            )

            if not allowed:
                await update.message.reply_text(f"⏳ {error_msg}")
                return

            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator


# -----------------------------------------------------------------------------
# Phase 13.12: System Mode & Degraded State Detection
# -----------------------------------------------------------------------------
class SystemMode(str, Enum):
    """System operational modes."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class SystemStateManager:
    """
    Manages system state and degraded mode detection.
    Auto-detects controller/CI availability and adjusts system mode.
    """

    def __init__(self):
        self._mode = SystemMode.NORMAL
        self._mode_lock = Lock()
        self._last_controller_check: Optional[datetime] = None
        self._controller_available = True
        self._ci_available = True
        self._degraded_reason: Optional[str] = None
        self._owner_override_active = False
        self._owner_override_justification: Optional[str] = None
        self._owner_override_user: Optional[int] = None

    @property
    def mode(self) -> SystemMode:
        """Get current system mode."""
        with self._mode_lock:
            return self._mode

    @property
    def degraded_reason(self) -> Optional[str]:
        """Get reason for degraded state."""
        with self._mode_lock:
            return self._degraded_reason

    def update_controller_status(self, available: bool, error: Optional[str] = None) -> None:
        """Update controller availability status."""
        with self._mode_lock:
            self._controller_available = available
            self._last_controller_check = datetime.utcnow()
            self._recalculate_mode(error)

    def update_ci_status(self, available: bool) -> None:
        """Update CI availability status."""
        with self._mode_lock:
            self._ci_available = available
            self._recalculate_mode()

    def _recalculate_mode(self, error: Optional[str] = None) -> None:
        """Recalculate system mode based on component availability."""
        old_mode = self._mode

        if not self._controller_available:
            self._mode = SystemMode.CRITICAL
            self._degraded_reason = f"Controller unavailable: {error or 'Connection failed'}"
        elif not self._ci_available:
            self._mode = SystemMode.DEGRADED
            self._degraded_reason = "CI system unavailable"
        else:
            self._mode = SystemMode.NORMAL
            self._degraded_reason = None
            # Clear owner override when system returns to normal
            if self._owner_override_active:
                logger.info("System restored to NORMAL, clearing owner override")
                self._owner_override_active = False
                self._owner_override_justification = None
                self._owner_override_user = None

        if old_mode != self._mode:
            logger.warning(
                f"SYSTEM MODE CHANGE: {old_mode.value} -> {self._mode.value} "
                f"(reason: {self._degraded_reason})"
            )

    def is_action_allowed(
        self,
        action: str,
        user_id: int,
        user_role: UserRole
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if an action is allowed in current system mode.
        Returns (allowed, error_message).
        """
        with self._mode_lock:
            # Read-only actions always allowed
            read_only_actions = [
                "health_check", "status", "projects_list", "dashboard",
                "ledger", "whoami", "help", "start"
            ]
            if action in read_only_actions:
                return True, None

            # NORMAL mode: all actions allowed
            if self._mode == SystemMode.NORMAL:
                return True, None

            # OWNER override check
            if self._owner_override_active and user_role == UserRole.OWNER:
                logger.warning(
                    f"OWNER OVERRIDE: user={user_id} action={action} "
                    f"justification={self._owner_override_justification}"
                )
                return True, None

            # DEGRADED mode: block approvals and deployments
            if self._mode == SystemMode.DEGRADED:
                blocked_actions = [
                    "approve_production", "deploy", "submit_feedback"
                ]
                if action in blocked_actions:
                    return False, (
                        f"⚠️ System is in DEGRADED mode ({self._degraded_reason}). "
                        f"Action '{action}' is temporarily disabled. "
                        f"Read-only commands are still available."
                    )
                return True, None

            # CRITICAL mode: block all non-read actions
            if self._mode == SystemMode.CRITICAL:
                return False, (
                    f"🚨 System is in CRITICAL mode ({self._degraded_reason}). "
                    f"Only read-only commands are available. "
                    f"Please wait for system recovery."
                )

            return True, None

    def set_owner_override(
        self,
        user_id: int,
        justification: str
    ) -> Tuple[bool, str]:
        """
        Allow OWNER to override degraded mode restrictions.
        Requires explicit justification.
        """
        with self._mode_lock:
            if self._mode == SystemMode.NORMAL:
                return False, "System is operating normally. No override needed."

            if len(justification.strip()) < 20:
                return False, "Override justification must be at least 20 characters."

            self._owner_override_active = True
            self._owner_override_justification = justification
            self._owner_override_user = user_id

            logger.warning(
                f"OWNER OVERRIDE ACTIVATED: user={user_id} "
                f"mode={self._mode.value} "
                f"justification={justification}"
            )

            safety.log_action(user_id, "owner_override_activated", {
                "mode": self._mode.value,
                "reason": self._degraded_reason,
                "justification": justification
            })

            return True, (
                f"Override activated. You can now perform restricted actions. "
                f"This has been logged for audit purposes."
            )

    def get_status(self) -> Dict[str, Any]:
        """Get current system state for /health display."""
        with self._mode_lock:
            return {
                "mode": self._mode.value,
                "controller_available": self._controller_available,
                "ci_available": self._ci_available,
                "degraded_reason": self._degraded_reason,
                "owner_override_active": self._owner_override_active,
                "last_check": self._last_controller_check.isoformat() if self._last_controller_check else None
            }


system_state = SystemStateManager()


def check_system_mode(action: str):
    """Decorator to check system mode before executing action."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
            user_id = update.effective_user.id
            user_role = get_user_role(user_id)

            allowed, error_msg = system_state.is_action_allowed(
                action, user_id, user_role
            )

            if not allowed:
                await update.message.reply_text(error_msg)
                return

            return await func(update, context, *args, **kwargs)
        return wrapper
    return decorator


# -----------------------------------------------------------------------------
# HTTP Client for Controller Communication (Phase 13.11: Timeouts & Retries)
# -----------------------------------------------------------------------------
class ControllerClient:
    """
    HTTP client for communicating with the controller API.
    Phase 13.11: Includes timeout management and retry logic with exponential backoff.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._last_health_check: Optional[datetime] = None
        self._health_cache: Optional[Dict] = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        action_name: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP request to controller with retry logic (Phase 13.11).

        Args:
            method: HTTP method (GET/POST)
            endpoint: API endpoint
            data: Request body for POST
            params: Query parameters for GET
            action_name: Name of action for retry decision
            timeout: Custom timeout (defaults to API_TIMEOUT_DEFAULT)
        """
        url = f"{self.base_url}{endpoint}"
        request_timeout = timeout or API_TIMEOUT_DEFAULT

        # Determine if retries are allowed for this action
        allow_retries = action_name not in NO_RETRY_ACTIONS if action_name else True
        max_attempts = API_MAX_RETRIES if allow_retries else 1

        last_error = None

        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(timeout=request_timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, params=params)
                    elif method.upper() == "POST":
                        response = await client.post(url, json=data)
                    else:
                        raise ValueError(f"Unsupported method: {method}")

                    response.raise_for_status()
                    result = response.json()

                    # Update system state on successful controller contact
                    system_state.update_controller_status(True)

                    return result

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on attempt {attempt + 1}/{max_attempts} to {url}: {e}"
                )
                # Update system state
                system_state.update_controller_status(False, "Timeout")

            except httpx.ConnectError as e:
                last_error = e
                logger.warning(
                    f"Connection error on attempt {attempt + 1}/{max_attempts} to {url}: {e}"
                )
                system_state.update_controller_status(False, "Connection refused")

            except httpx.HTTPStatusError as e:
                # Don't retry on HTTP errors (4xx, 5xx) - they're not transient
                logger.error(f"HTTP error from controller: {e.response.status_code}")
                system_state.update_controller_status(True)  # Controller is reachable
                try:
                    return e.response.json()
                except Exception:
                    return {"error": str(e), "status": "error"}

            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error on attempt {attempt + 1}/{max_attempts}: {e}")
                system_state.update_controller_status(False, str(e))

            # Exponential backoff before retry (if more attempts remain)
            if attempt < max_attempts - 1 and allow_retries:
                backoff = API_RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.info(f"Retrying in {backoff}s...")
                await asyncio.sleep(backoff)

        # All retries exhausted
        error_msg = f"Controller unreachable after {max_attempts} attempts"
        logger.error(f"{error_msg}: {last_error}")

        return {
            "error": error_msg,
            "status": "error",
            "details": str(last_error) if last_error else "Unknown error"
        }

    async def health_check(self) -> Dict[str, Any]:
        """Check controller health with caching."""
        now = datetime.utcnow()
        if self._health_cache and self._last_health_check:
            if (now - self._last_health_check).seconds < 30:
                return self._health_cache

        result = await self._request("GET", "/health")
        if "error" not in result:
            self._health_cache = result
            self._last_health_check = now
        return result

    async def create_project(
        self,
        description: str,
        user_id: str,
        requirements: List[str] = None,
        repo_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new project from natural language (no retries - destructive)."""
        return await self._request(
            "POST", "/v2/project/create",
            data={
                "description": description,
                "requirements": requirements or [],
                "repo_url": repo_url,
                "reference_urls": [],
                "user_id": user_id
            },
            action_name="create_project"
        )

    async def get_project(self, project_name: str) -> Dict[str, Any]:
        """Get project details."""
        return await self._request("GET", f"/v2/project/{project_name}")

    async def get_aspect(self, project_name: str, aspect: str) -> Dict[str, Any]:
        """Get aspect state."""
        return await self._request("GET", f"/v2/project/{project_name}/aspect/{aspect}")

    async def submit_feedback(
        self,
        project_name: str,
        aspect: str,
        feedback_type: str,
        user_id: str,
        explanation: Optional[str] = None,
        affected_features: List[str] = None
    ) -> Dict[str, Any]:
        """Submit testing feedback (no retries - stateful action)."""
        return await self._request(
            "POST", f"/v2/project/{project_name}/aspect/{aspect}/feedback",
            data={
                "project_name": project_name,
                "aspect": aspect,
                "feedback_type": feedback_type,
                "explanation": explanation,
                "affected_features": affected_features or [],
                "user_id": user_id
            },
            action_name="submit_feedback"
        )

    async def approve_production(
        self,
        project_name: str,
        aspect: str,
        user_id: str,
        justification: str,
        risk_acknowledged: bool,
        rollback_plan: str
    ) -> Dict[str, Any]:
        """Approve production deployment (no retries - critical action)."""
        return await self._request(
            "POST", f"/v2/project/{project_name}/aspect/{aspect}/approve-production",
            data={
                "project_name": project_name,
                "aspect": aspect,
                "justification": justification,
                "risk_acknowledged": risk_acknowledged,
                "rollback_plan": rollback_plan,
                "user_id": user_id
            },
            action_name="approve_production"
        )

    async def get_dashboard(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get dashboard data."""
        if project_name:
            return await self._request("GET", f"/v2/dashboard/{project_name}")
        return await self._request("GET", "/v2/dashboard")

    async def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get enhanced dashboard summary (Phase 16B)."""
        return await self._request("GET", "/dashboard")

    async def get_dashboard_jobs(self) -> Dict[str, Any]:
        """Get Claude jobs activity (Phase 16B)."""
        return await self._request("GET", "/dashboard/jobs")

    async def get_dashboard_audit(self, limit: int = 10) -> Dict[str, Any]:
        """Get recent audit events (Phase 16B)."""
        return await self._request("GET", "/dashboard/audit", params={"limit": limit})

    # Phase 17A: Runtime Intelligence Methods
    async def get_runtime_signals(
        self,
        project_id: Optional[str] = None,
        since_hours: int = 24,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get runtime signals (Phase 17A)."""
        params = {
            "since_hours": since_hours,
            "limit": limit,
        }
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/runtime/signals", params=params)

    async def get_runtime_summary(self, since_hours: int = 24) -> Dict[str, Any]:
        """Get runtime signal summary (Phase 17A)."""
        return await self._request("GET", "/runtime/summary", params={"since_hours": since_hours})

    async def get_runtime_status(self) -> Dict[str, Any]:
        """Get runtime intelligence status (Phase 17A)."""
        return await self._request("GET", "/runtime/status")

    async def poll_runtime_signals(self) -> Dict[str, Any]:
        """Trigger a manual signal poll (Phase 17A)."""
        return await self._request("POST", "/runtime/poll")

    # Phase 17B: Incident Classification Methods (READ-ONLY)
    async def get_incidents(
        self,
        project_id: Optional[str] = None,
        incident_type: Optional[str] = None,
        severity: Optional[str] = None,
        since_hours: int = 24,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Get incidents with optional filtering (Phase 17B)."""
        params = {
            "since_hours": since_hours,
            "limit": limit,
        }
        if project_id:
            params["project_id"] = project_id
        if incident_type:
            params["incident_type"] = incident_type
        if severity:
            params["severity"] = severity
        return await self._request("GET", "/incidents", params=params)

    async def get_incidents_recent(
        self,
        hours: int = 24,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Get recent incidents (Phase 17B)."""
        params = {
            "hours": hours,
            "limit": limit,
        }
        return await self._request("GET", "/incidents/recent", params=params)

    async def get_incidents_summary(self, since_hours: int = 24) -> Dict[str, Any]:
        """Get incident summary (Phase 17B)."""
        return await self._request("GET", "/incidents/summary", params={"since_hours": since_hours})

    async def get_incident_by_id(self, incident_id: str) -> Dict[str, Any]:
        """Get a specific incident by ID (Phase 17B)."""
        return await self._request("GET", f"/incidents/{incident_id}")

    # Phase 18C: Execution Dispatcher Methods (CONTROLLED EXECUTION)
    async def get_execution(self, execution_id: str) -> Dict[str, Any]:
        """Get execution result by ID (Phase 18C)."""
        return await self._request("GET", f"/execution/{execution_id}")

    async def get_execution_recent(
        self,
        limit: int = 20,
        status: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent executions (Phase 18C)."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/execution/recent", params=params)

    async def get_execution_summary(
        self,
        project_id: Optional[str] = None,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get execution summary (Phase 18C)."""
        params = {"since_hours": since_hours}
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/execution/summary", params=params)

    # Phase 18D: Post-Execution Verification Methods (READ-ONLY)
    async def get_execution_verification(self, execution_id: str) -> Dict[str, Any]:
        """Get verification result for an execution (Phase 18D)."""
        return await self._request("GET", f"/execution/{execution_id}/verification")

    async def get_execution_violations(self, execution_id: str) -> Dict[str, Any]:
        """Get violations for an execution (Phase 18D)."""
        return await self._request("GET", f"/execution/{execution_id}/violations")

    async def get_verification_recent(
        self,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent verification results (Phase 18D)."""
        params = {"limit": limit}
        if status:
            params["status"] = status
        return await self._request("GET", "/execution/verification/recent", params=params)

    async def get_verification_summary(
        self,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get verification summary (Phase 18D)."""
        params = {"since_hours": since_hours}
        return await self._request("GET", "/execution/verification/summary", params=params)

    # Phase 19: Learning, Memory & System Intelligence Methods (READ-ONLY)
    async def get_learning_patterns(
        self,
        limit: int = 20,
        pattern_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get observed patterns (Phase 19)."""
        params = {"limit": limit}
        if pattern_type:
            params["pattern_type"] = pattern_type
        return await self._request("GET", "/learning/patterns", params=params)

    async def get_learning_trends(
        self,
        limit: int = 20,
        metric_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get observed trends (Phase 19)."""
        params = {"limit": limit}
        if metric_name:
            params["metric_name"] = metric_name
        return await self._request("GET", "/learning/trends", params=params)

    async def get_learning_history(
        self,
        limit: int = 20,
        entry_type: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get memory history (Phase 19)."""
        params = {"limit": limit}
        if entry_type:
            params["entry_type"] = entry_type
        if project_id:
            params["project_id"] = project_id
        return await self._request("GET", "/learning/history", params=params)

    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get latest learning summary (Phase 19)."""
        return await self._request("GET", "/learning/summary")

    async def get_learning_statistics(
        self,
        since_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get learning statistics (Phase 19)."""
        params = {"since_hours": since_hours}
        return await self._request("GET", "/learning/statistics", params=params)

    async def get_notifications(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get pending notifications."""
        params = {"project_name": project_name} if project_name else None
        return await self._request("GET", "/v2/notifications", params=params)

    async def get_projects_list(self) -> Dict[str, Any]:
        """Get list of all projects."""
        return await self._request("GET", "/v2/dashboard")

    async def get_ledger(self, project_name: str) -> Dict[str, Any]:
        """Get project ledger/audit trail."""
        return await self._request("GET", f"/v2/ledger/{project_name}")

    # Phase 14: Claude CLI Job Methods
    async def get_claude_status(self) -> Dict[str, Any]:
        """Get Claude CLI availability and job queue status."""
        return await self._request("GET", "/claude/status")

    async def get_claude_queue(self) -> Dict[str, Any]:
        """Get Claude job queue status."""
        return await self._request("GET", "/claude/queue")

    async def create_claude_job(
        self,
        project_name: str,
        task_description: str,
        task_type: str = "feature_development",
        copy_from_job: Optional[str] = None,
        copy_artifacts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create a new Claude CLI job with optional artifact copying."""
        data = {
            "project_name": project_name,
            "task_description": task_description,
            "task_type": task_type
        }
        if copy_from_job:
            data["copy_from_job"] = copy_from_job
        if copy_artifacts:
            data["copy_artifacts"] = copy_artifacts
        return await self._request(
            "POST", "/claude/job",
            data=data,
            action_name="create_claude_job"
        )

    async def get_claude_job(self, job_id: str) -> Dict[str, Any]:
        """Get Claude job status by ID."""
        return await self._request("GET", f"/claude/job/{job_id}")

    async def list_claude_jobs(
        self,
        state: Optional[str] = None,
        project: Optional[str] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """List Claude jobs with optional filtering."""
        params = {"limit": limit}
        if state:
            params["state"] = state
        if project:
            params["project"] = project
        return await self._request("GET", "/claude/jobs", params=params)

    async def cancel_claude_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a Claude job."""
        return await self._request(
            "POST", f"/claude/job/{job_id}/cancel",
            action_name="cancel_claude_job"
        )


controller = ControllerClient(CONTROLLER_BASE_URL)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def get_system_status() -> Dict[str, Any]:
    """Get system status including memory usage."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        memory_mb = usage.ru_maxrss / 1024  # Convert to MB (Linux)
        if sys.platform == "darwin":
            memory_mb = usage.ru_maxrss / (1024 * 1024)  # macOS uses bytes
    except Exception:
        memory_mb = 0

    uptime = datetime.utcnow() - BOT_START_TIME

    return {
        "version": BOT_VERSION,
        "uptime_seconds": int(uptime.total_seconds()),
        "uptime_human": str(uptime).split(".")[0],
        "memory_mb": round(memory_mb, 2),
        "start_time": BOT_START_TIME.isoformat()
    }


def get_systemd_status(service: str) -> Dict[str, Any]:
    """Get systemd service status."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True,
            timeout=5
        )
        is_active = result.stdout.strip() == "active"

        result2 = subprocess.run(
            ["systemctl", "show", service, "--property=ActiveEnterTimestamp"],
            capture_output=True,
            text=True,
            timeout=5
        )
        timestamp = result2.stdout.strip().split("=")[1] if "=" in result2.stdout else "unknown"

        return {
            "service": service,
            "active": is_active,
            "status": "running" if is_active else "stopped",
            "since": timestamp
        }
    except Exception as e:
        return {
            "service": service,
            "active": False,
            "status": "unknown",
            "error": str(e)
        }


def format_project_status(project: Dict[str, Any]) -> str:
    """Format project status for display."""
    lines = [
        f"*Project:* {project.get('project_name', 'Unknown')}",
        f"*Status:* {project.get('overall_status', 'Unknown')}",
        "",
        "*Aspects:*"
    ]

    aspects = project.get("aspects", {})
    aspect_states = project.get("aspect_states", {})

    for aspect_type in ["core", "backend", "frontend"]:
        config = aspects.get(aspect_type, {})
        state = aspect_states.get(aspect_type, {})

        if config.get("enabled", False):
            phase = state.get("current_phase", "not_started")
            emoji = get_phase_emoji(phase)
            lines.append(f"  {emoji} *{aspect_type.title()}:* {phase.replace('_', ' ').title()}")

    return "\n".join(lines)


def get_phase_emoji(phase: str) -> str:
    """Get emoji for phase status."""
    emoji_map = {
        "not_started": "⬜",
        "planning": "📝",
        "development": "💻",
        "unit_testing": "🧪",
        "integration": "🔗",
        "code_review": "👀",
        "ci_running": "⏳",
        "ci_passed": "✅",
        "ci_failed": "❌",
        "ready_for_testing": "🎯",
        "deployed_testing": "🚀",
        "awaiting_feedback": "💬",
        "bug_fixing": "🐛",
        "improvements": "✨",
        "ready_for_production": "📦",
        "deployed_production": "🌐",
        "completed": "🏁"
    }
    return emoji_map.get(phase, "❓")


def format_dashboard(dashboard: Dict[str, Any]) -> str:
    """Format dashboard data for display."""
    projects = dashboard.get("projects", [])
    if not projects:
        return "No projects found."

    lines = [
        f"*Dashboard Overview*",
        f"Total Projects: {dashboard.get('total_projects', 0)}",
        f"System Health: {dashboard.get('system_health', 'unknown')}",
        ""
    ]

    for proj in projects[:5]:
        lines.append(f"*{proj.get('project_name', 'Unknown')}*")
        lines.append(f"  Status: {proj.get('overall_status', 'unknown')}")

        aspects = proj.get("aspects", {})
        for aspect_type, aspect_data in aspects.items():
            phase = aspect_data.get("current_phase", "unknown")
            emoji = get_phase_emoji(phase)
            lines.append(f"  {emoji} {aspect_type}: {phase.replace('_', ' ')}")

        lines.append("")

    return "\n".join(lines)


def format_dashboard_enhanced(summary: Dict[str, Any]) -> str:
    """
    Format enhanced dashboard summary for Telegram (Phase 16B).

    Returns: Summary counts, Active projects, Active jobs, System health snapshot.
    """
    lines = []

    # Phase 19 fix: Handle nested data structure from /dashboard endpoint
    # The API returns {"phase": "16B", "data": {...}} format
    data = summary.get("data", summary)  # Use nested data if present, else use summary directly

    # System health status with emoji
    health = data.get("system_health", "unknown")
    health_emoji = {
        "healthy": "🟢",
        "degraded": "🟡",
        "unhealthy": "🔴",
        "unknown": "⚪"
    }.get(health, "⚪")

    lines.append(f"📊 *Platform Dashboard*")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"")
    lines.append(f"{health_emoji} *System Health:* {health.upper()}")
    lines.append(f"")

    # Summary counts - handle both flat and nested structures
    projects_data = data.get("projects", {})
    jobs_data = data.get("jobs", {})
    lifecycles_data = data.get("lifecycles", {})

    total_projects = projects_data.get("total", data.get("total_projects", 0)) if isinstance(projects_data, dict) else 0
    active_lifecycles = lifecycles_data.get("total", data.get("total_lifecycles", 0)) if isinstance(lifecycles_data, dict) else 0
    pending_jobs = jobs_data.get("queued", data.get("pending_jobs", 0)) if isinstance(jobs_data, dict) else 0
    active_workers = jobs_data.get("active", data.get("active_workers", 0)) if isinstance(jobs_data, dict) else 0

    lines.append(f"📈 *Summary Counts*")
    lines.append(f"  Projects: {total_projects}")
    lines.append(f"  Active Lifecycles: {active_lifecycles}")
    lines.append(f"  Pending Jobs: {pending_jobs}")
    lines.append(f"  Active Workers: {active_workers}/{data.get('max_workers', 3)}")
    lines.append(f"")

    # Active projects (top 5) - use data dict
    active_projects = data.get("active_projects", [])
    if active_projects:
        lines.append(f"🚀 *Active Projects* ({len(active_projects)})")
        for proj in active_projects[:5]:
            name = proj.get("project_name", "Unknown")
            state = proj.get("lifecycle_state", "unknown")
            state_emoji = get_lifecycle_state_emoji(state)
            lines.append(f"  {state_emoji} {name}: {state.replace('_', ' ')}")
        if len(active_projects) > 5:
            lines.append(f"  ... and {len(active_projects) - 5} more")
        lines.append(f"")

    # Claude jobs activity - use data dict
    claude_activity = data.get("claude_activity", {})
    if claude_activity:
        running = claude_activity.get("running_jobs", 0)
        queued = claude_activity.get("queued_jobs", 0)
        completed_24h = claude_activity.get("completed_24h", 0)

        lines.append(f"🤖 *Claude Activity*")
        lines.append(f"  Running: {running}")
        lines.append(f"  Queued: {queued}")
        lines.append(f"  Completed (24h): {completed_24h}")
        lines.append(f"")

    # Security alerts (if any) - use data dict
    security = data.get("security", {})
    gate_denials = security.get("gate_denials_today", security.get("recent_gate_denials", 0))
    if gate_denials > 0:
        lines.append(f"🔒 *Security Alerts*")
        lines.append(f"  ⚠️ Gate Denials (24h): {gate_denials}")
        lines.append(f"")

    # Services status
    services = data.get("services", {})
    if services:
        lines.append(f"🔧 *Services*")
        for svc, status in services.items():
            status_emoji = "✅" if status else "❌"
            lines.append(f"  {status_emoji} {svc.replace('_', ' ').title()}")
        lines.append(f"")

    # Timestamp - use data dict
    timestamp = data.get("timestamp", summary.get("timestamp", ""))
    if timestamp:
        lines.append(f"⏱️ _Updated: {timestamp[:19]}_")

    return "\n".join(lines)


def get_lifecycle_state_emoji(state: str) -> str:
    """Get emoji for lifecycle state."""
    emoji_map = {
        "created": "⬜",
        "planning": "📝",
        "development": "💻",
        "testing": "🧪",
        "awaiting_feedback": "💬",
        "ready_for_production": "📦",
        "production_approved": "✅",
        "deployed": "🚀",
        "archived": "📁",
        "rejected": "❌"
    }
    return emoji_map.get(state.lower(), "❓")


def get_severity_emoji(severity: str) -> str:
    """Get emoji for signal severity (Phase 17A)."""
    emoji_map = {
        "info": "ℹ️",
        "warning": "⚠️",
        "degraded": "🟡",
        "critical": "🔴",
        "unknown": "❓",
    }
    return emoji_map.get(severity.lower(), "❓")


def get_signal_type_emoji(signal_type: str) -> str:
    """Get emoji for signal type (Phase 17A)."""
    emoji_map = {
        "system_resource": "💻",
        "worker_queue": "📋",
        "job_failure": "❌",
        "test_regression": "🧪",
        "deployment_failure": "🚀",
        "drift_warning": "📊",
        "human_override": "👤",
        "config_anomaly": "⚙️",
    }
    return emoji_map.get(signal_type.lower(), "📡")


def format_signals_summary(summary: Dict[str, Any]) -> str:
    """
    Format runtime signals summary for Telegram (Phase 17A).

    Returns: Summary of collected signals with severity breakdown and confidence indicators.
    Rules:
    - Summarized output only (never raw dumps)
    - Always include confidence indicator
    - READ-ONLY display
    """
    lines = []

    lines.append("📡 *Runtime Intelligence Summary*")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    # Collection status
    poll_running = summary.get("poll_running", False)
    status_emoji = "🟢" if poll_running else "⚪"
    lines.append(f"{status_emoji} *Collection:* {'Active' if poll_running else 'Inactive'}")
    lines.append("")

    # Severity breakdown
    by_severity = summary.get("by_severity", {})
    lines.append("*📊 Severity Breakdown*")
    for sev in ["critical", "degraded", "warning", "info", "unknown"]:
        count = by_severity.get(sev, 0)
        emoji = get_severity_emoji(sev)
        if count > 0 or sev in ["critical", "unknown"]:
            lines.append(f"  {emoji} {sev.upper()}: {count}")
    lines.append("")

    # Signal types
    by_type = summary.get("by_type", {})
    if by_type:
        lines.append("*📈 Signal Types*")
        for sig_type, count in sorted(by_type.items(), key=lambda x: -x[1])[:5]:
            emoji = get_signal_type_emoji(sig_type)
            lines.append(f"  {emoji} {sig_type}: {count}")
        lines.append("")

    # Recent signals (summarized)
    recent = summary.get("recent_signals", [])
    if recent:
        lines.append("*🕐 Recent Signals*")
        for sig in recent[:5]:
            sev_emoji = get_severity_emoji(sig.get("severity", "unknown"))
            type_emoji = get_signal_type_emoji(sig.get("signal_type", ""))
            desc = sig.get("description", "No description")[:50]
            confidence = sig.get("confidence", 0.0)
            conf_indicator = "●" if confidence >= 0.8 else "◐" if confidence >= 0.5 else "○"
            lines.append(f"  {sev_emoji}{type_emoji} {desc}... [{conf_indicator}]")
        lines.append("")

    # Totals
    total = summary.get("total_signals", 0)
    period = summary.get("period_hours", 24)
    lines.append(f"*Total Signals ({period}h):* {total}")

    # Timestamp
    timestamp = summary.get("timestamp", "")
    if timestamp:
        lines.append(f"\n_Last updated: {timestamp[:19]}_")

    # Confidence legend
    lines.append("")
    lines.append("_Confidence: ● high | ◐ medium | ○ low_")

    return "\n".join(lines)


def format_signals_list(signals_data: Dict[str, Any], limit: int = 10) -> str:
    """
    Format list of runtime signals for Telegram (Phase 17A).

    Returns: Formatted list with summarized output and confidence indicators.
    """
    lines = []
    signals = signals_data.get("signals", [])
    total = signals_data.get("total", len(signals))

    lines.append("📡 *Recent Runtime Signals*")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    if not signals:
        lines.append("_No signals found in the specified period._")
        return "\n".join(lines)

    for sig in signals[:limit]:
        sev_emoji = get_severity_emoji(sig.get("severity", "unknown"))
        type_emoji = get_signal_type_emoji(sig.get("signal_type", ""))

        # Timestamp (shortened)
        ts = sig.get("timestamp", "")[:16].replace("T", " ")

        # Description (truncated)
        desc = sig.get("description", "No description")
        if len(desc) > 60:
            desc = desc[:57] + "..."

        # Confidence indicator
        confidence = sig.get("confidence", 0.0)
        conf_indicator = "●" if confidence >= 0.8 else "◐" if confidence >= 0.5 else "○"

        # Project context
        project = sig.get("project_id", "")
        project_str = f" [{project}]" if project else ""

        lines.append(f"{sev_emoji}{type_emoji} `{ts}`{project_str}")
        lines.append(f"   {desc} [{conf_indicator}]")
        lines.append("")

    if total > limit:
        lines.append(f"_...and {total - limit} more signals_")

    # Confidence legend
    lines.append("")
    lines.append("_Confidence: ● high | ◐ medium | ○ low_")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Phase 17B: Incident Classification Formatting (READ-ONLY)
# -----------------------------------------------------------------------------

def get_incident_severity_emoji(severity: str) -> str:
    """Get emoji for incident severity (Phase 17B)."""
    emoji_map = {
        "info": "ℹ️",
        "low": "🟢",
        "medium": "🟡",
        "high": "🟠",
        "critical": "🔴",
        "unknown": "❓",
    }
    return emoji_map.get(severity.lower(), "❓")


def get_incident_type_emoji(incident_type: str) -> str:
    """Get emoji for incident type (Phase 17B)."""
    emoji_map = {
        "performance": "⚡",
        "reliability": "🔧",
        "security": "🔒",
        "governance": "📋",
        "resource": "💾",
        "configuration": "⚙️",
        "unknown": "❓",
    }
    return emoji_map.get(incident_type.lower(), "📋")


def get_incident_scope_emoji(scope: str) -> str:
    """Get emoji for incident scope (Phase 17B)."""
    emoji_map = {
        "system": "🌐",
        "project": "📁",
        "project_aspect": "📄",
        "job": "⚙️",
        "unknown": "❓",
    }
    return emoji_map.get(scope.lower(), "❓")


def format_incidents_summary(summary: Dict[str, Any]) -> str:
    """
    Format incident summary for Telegram (Phase 17B).

    Returns: Summary of classified incidents with severity/type breakdown.
    Rules:
    - Summarized output only (never raw dumps)
    - Always include confidence indicator
    - READ-ONLY display
    - UNKNOWN incidents clearly labeled
    """
    lines = []

    lines.append("🚨 *Incident Classification Summary*")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    # Total incidents
    total = summary.get("total_incidents", 0)
    open_count = summary.get("open_count", 0)
    unknown_count = summary.get("unknown_count", 0)

    lines.append(f"*Total Incidents:* {total}")
    lines.append(f"*Open:* {open_count} | *Unknown:* {unknown_count}")
    lines.append("")

    # Severity breakdown
    by_severity = summary.get("by_severity", {})
    lines.append("*📊 By Severity*")
    for sev in ["critical", "high", "medium", "low", "info", "unknown"]:
        count = by_severity.get(sev, 0)
        emoji = get_incident_severity_emoji(sev)
        if count > 0 or sev in ["critical", "unknown"]:
            lines.append(f"  {emoji} {sev.upper()}: {count}")
    lines.append("")

    # Type breakdown
    by_type = summary.get("by_type", {})
    if by_type:
        lines.append("*📈 By Type*")
        for inc_type, count in sorted(by_type.items(), key=lambda x: -x[1])[:6]:
            emoji = get_incident_type_emoji(inc_type)
            lines.append(f"  {emoji} {inc_type}: {count}")
        lines.append("")

    # Recent incidents
    recent = summary.get("recent_incidents", [])
    if recent:
        lines.append("*🕐 Recent Incidents*")
        for inc in recent[:5]:
            sev_emoji = get_incident_severity_emoji(inc.get("severity", "unknown"))
            type_emoji = get_incident_type_emoji(inc.get("incident_type", "unknown"))
            title = inc.get("title", "No title")[:40]
            state = inc.get("state", "unknown")
            state_indicator = "🟢" if state == "open" else "⚪"
            lines.append(f"  {sev_emoji}{type_emoji} {title}")
            lines.append(f"     {state_indicator} {state}")
        lines.append("")

    # Time window
    start = summary.get("time_window_start", "")
    end = summary.get("time_window_end", "")
    if start and end:
        start_short = start[:16].replace("T", " ")
        end_short = end[:16].replace("T", " ")
        lines.append(f"_Period: {start_short} to {end_short}_")

    return "\n".join(lines)


def format_incidents_list(incidents_data: Dict[str, Any], limit: int = 10) -> str:
    """
    Format list of incidents for Telegram (Phase 17B).

    Returns: Formatted list with summarized output and confidence indicators.
    """
    lines = []
    incidents = incidents_data.get("incidents", [])
    total = incidents_data.get("total", len(incidents))

    lines.append("🚨 *Recent Incidents*")
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append("")

    if not incidents:
        lines.append("_No incidents found in the specified period._")
        return "\n".join(lines)

    for inc in incidents[:limit]:
        sev_emoji = get_incident_severity_emoji(inc.get("severity", "unknown"))
        type_emoji = get_incident_type_emoji(inc.get("incident_type", "unknown"))
        scope_emoji = get_incident_scope_emoji(inc.get("scope", "unknown"))

        # Timestamp (shortened)
        ts = inc.get("created_at", "")[:16].replace("T", " ")

        # Title (truncated)
        title = inc.get("title", "No title")
        if len(title) > 50:
            title = title[:47] + "..."

        # Confidence indicator
        confidence = inc.get("confidence", 0.0)
        conf_indicator = "●" if confidence >= 0.8 else "◐" if confidence >= 0.5 else "○"

        # State indicator
        state = inc.get("state", "unknown")
        state_indicator = "🟢" if state == "open" else "⚪"

        # Project context
        project = inc.get("project_id", "")
        project_str = f" [{project}]" if project else ""

        lines.append(f"{sev_emoji}{type_emoji}{scope_emoji} `{ts}`{project_str}")
        lines.append(f"   {title}")
        lines.append(f"   {state_indicator} {state} | Signals: {inc.get('signal_count', 0)} [{conf_indicator}]")
        lines.append("")

    if total > limit:
        lines.append(f"_...and {total - limit} more incidents_")

    # Confidence legend
    lines.append("")
    lines.append("_Confidence: ● high | ◐ medium | ○ low_")

    return "\n".join(lines)


def get_feedback_keyboard(project_name: str, aspect: str) -> InlineKeyboardMarkup:
    """Create inline keyboard for feedback options."""
    keyboard = [
        [
            InlineKeyboardButton(
                "✅ Approve",
                callback_data=f"{CallbackAction.FEEDBACK_APPROVE.value}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "🐞 Bug",
                callback_data=f"{CallbackAction.FEEDBACK_BUG.value}:{project_name}:{aspect}"
            ),
            InlineKeyboardButton(
                "✨ Improvements",
                callback_data=f"{CallbackAction.FEEDBACK_IMPROVEMENTS.value}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "❌ Reject",
                callback_data=f"{CallbackAction.FEEDBACK_REJECT.value}:{project_name}:{aspect}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_rejection_reason_keyboard(project_name: str, aspect: str) -> InlineKeyboardMarkup:
    """Create inline keyboard for rejection reasons."""
    keyboard = [
        [
            InlineKeyboardButton(
                "🐞 Bug Found",
                callback_data=f"{CallbackAction.REJECT_BUG.value}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "✨ Needs Improvement",
                callback_data=f"{CallbackAction.REJECT_IMPROVEMENT.value}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "📋 Invalid Requirement",
                callback_data=f"{CallbackAction.REJECT_INVALID.value}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "📝 Other (explain)",
                callback_data=f"{CallbackAction.REJECT_OTHER.value}:{project_name}:{aspect}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_conflict_resolution_keyboard(
    existing_project: str,
    new_description: str,
    conflict_id: str
) -> InlineKeyboardMarkup:
    """
    Create inline keyboard for conflict resolution (Phase 16E).

    Shows user choices when a similar project already exists.
    """
    keyboard = [
        [
            InlineKeyboardButton(
                "1️⃣ Improve existing project",
                callback_data=f"{CallbackAction.CONFLICT_IMPROVE.value}:{existing_project}:{conflict_id}"
            )
        ],
        [
            InlineKeyboardButton(
                "2️⃣ Add new module",
                callback_data=f"{CallbackAction.CONFLICT_ADD_MODULE.value}:{existing_project}:{conflict_id}"
            )
        ],
        [
            InlineKeyboardButton(
                "3️⃣ Create new version",
                callback_data=f"{CallbackAction.CONFLICT_NEW_VERSION.value}:{existing_project}:{conflict_id}"
            )
        ],
        [
            InlineKeyboardButton(
                "4️⃣ Cancel",
                callback_data=f"{CallbackAction.CONFLICT_CANCEL.value}:{existing_project}:{conflict_id}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# -----------------------------------------------------------------------------
# Notification Formatters (Phase 13.6)
# -----------------------------------------------------------------------------
def format_ci_notification(notif: Dict[str, Any]) -> str:
    """Format CI completion notification."""
    details = notif.get("details", {})
    result = details.get("result", "unknown")
    emoji = "✅" if result == "passed" else "❌" if result == "failed" else "⏳"

    lines = [
        f"{emoji} *CI {result.upper()}*",
        "",
        f"*Project:* {notif.get('project_name', 'Unknown')}",
        f"*Aspect:* {notif.get('aspect', 'Unknown')}",
        f"*Phase:* {details.get('phase_completed', 'Unknown')}",
        "",
    ]

    if details.get("tests_passed"):
        lines.append(f"✅ Tests Passed: {details.get('tests_passed', 0)}")
    if details.get("tests_failed"):
        lines.append(f"❌ Tests Failed: {details.get('tests_failed', 0)}")
    if details.get("coverage"):
        lines.append(f"📊 Coverage: {details.get('coverage')}%")

    if notif.get("next_action"):
        lines.append("")
        lines.append(f"👉 *Next:* {notif.get('next_action')}")

    return "\n".join(lines)


def format_testing_deployment_notification(notif: Dict[str, Any]) -> str:
    """Format testing deployment notification with full details."""
    details = notif.get("details", {})

    lines = [
        "🚀 *DEPLOYED TO TESTING*",
        "",
        f"*Project:* {notif.get('project_name', 'Unknown')}",
        f"*Aspect:* {notif.get('aspect', 'Unknown')}",
        "",
    ]

    if notif.get("environment_url"):
        lines.append(f"🔗 *Testing URL:* {notif.get('environment_url')}")
        lines.append("")

    features = notif.get("features_completed", [])
    if features:
        lines.append("✅ *Features Deployed:*")
        for feat in features[:10]:
            lines.append(f"  • {feat}")
        lines.append("")

    if notif.get("test_coverage_summary"):
        lines.append(f"🧪 *Test Summary:* {notif.get('test_coverage_summary')}")
        lines.append("")

    limitations = notif.get("known_limitations", [])
    if limitations:
        lines.append("⚠️ *Known Limitations:*")
        for lim in limitations[:5]:
            lines.append(f"  • {lim}")
        lines.append("")

    if details.get("testing_focus"):
        lines.append("👉 *What to Test:*")
        for focus in details.get("testing_focus", []):
            lines.append(f"  • {focus}")
        lines.append("")

    lines.append("Please test and provide feedback using /feedback command")

    return "\n".join(lines)


def format_production_deployment_notification(notif: Dict[str, Any]) -> str:
    """Format production deployment notification."""
    details = notif.get("details", {})

    lines = [
        "🌐 *DEPLOYED TO PRODUCTION*",
        "",
        f"*Project:* {notif.get('project_name', 'Unknown')}",
        f"*Aspect:* {notif.get('aspect', 'Unknown')}",
        "",
    ]

    if notif.get("environment_url"):
        lines.append(f"🔗 *Production URL:* {notif.get('environment_url')}")
        lines.append("")

    features = notif.get("features_completed", [])
    if features:
        lines.append("✅ *Features Deployed:*")
        for feat in features[:10]:
            lines.append(f"  • {feat}")
        lines.append("")

    if notif.get("test_coverage_summary"):
        lines.append(f"🧪 *Test Summary:* {notif.get('test_coverage_summary')}")
        lines.append("")

    # Rollback instructions
    lines.append("🔄 *Rollback Instructions:*")
    if details.get("rollback_command"):
        lines.append(f"  Command: `{details.get('rollback_command')}`")
    else:
        lines.append("  Contact admin for rollback if issues detected")
    lines.append("")

    lines.append("Monitor for any issues and report via /feedback")

    return "\n".join(lines)


def format_production_approval_notification(notif: Dict[str, Any]) -> str:
    """Format production approval request notification."""
    details = notif.get("details", {})

    lines = [
        "⚠️ *PRODUCTION APPROVAL REQUIRED*",
        "",
        f"*Project:* {notif.get('project_name', 'Unknown')}",
        f"*Aspect:* {notif.get('aspect', 'Unknown')}",
        "",
        f"*Requested by:* {details.get('requested_by', 'Unknown')}",
        f"*Requested at:* {details.get('requested_at', 'Unknown')}",
        "",
    ]

    if details.get("testing_results"):
        lines.append("*Testing Results:*")
        lines.append(f"  {details.get('testing_results')}")
        lines.append("")

    if details.get("changes_summary"):
        lines.append("*Changes Summary:*")
        lines.append(f"  {details.get('changes_summary')}")
        lines.append("")

    lines.append("Use /prod_approve to approve this deployment")
    lines.append("_(Requires different user from requester)_")

    return "\n".join(lines)


def format_notification(notif: Dict[str, Any]) -> Tuple[str, Optional[InlineKeyboardMarkup]]:
    """Format any notification with appropriate template."""
    notif_type = notif.get("notification_type", "")
    keyboard = None

    if notif_type == "ci_result":
        text = format_ci_notification(notif)
    elif notif_type == "testing_ready":
        text = format_testing_deployment_notification(notif)
        project = notif.get("project_name", "")
        aspect = notif.get("aspect", "")
        if project and aspect:
            keyboard = get_feedback_keyboard(project, aspect)
    elif notif_type == "deployment_complete":
        if notif.get("details", {}).get("environment") == "production":
            text = format_production_deployment_notification(notif)
        else:
            text = format_testing_deployment_notification(notif)
    elif notif_type == "approval_required":
        text = format_production_approval_notification(notif)
    else:
        lines = [
            f"🔔 *{notif.get('title', 'Notification')}*",
            "",
            notif.get("summary", ""),
        ]
        if notif.get("environment_url"):
            lines.append(f"\n🔗 {notif.get('environment_url')}")
        if notif.get("next_action"):
            lines.append(f"\n👉 {notif.get('next_action')}")
        text = "\n".join(lines)

    return text, keyboard


# -----------------------------------------------------------------------------
# Phase 13.3: Health & Status Commands (READ-ONLY, NO RBAC)
# -----------------------------------------------------------------------------
async def whoami_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /whoami - Show current user identity and role (Phase 13.4)

    ALWAYS responds - no RBAC restriction.
    """
    try:
        user = update.effective_user
        user_id = user.id
        user_role = get_user_role(user_id)
        username = user.username or "Not set"
        full_name = user.full_name or "Not set"

        safety.log_action(user_id, "whoami", {})

        lines = [
            "*Your Identity*",
            "",
            f"*User ID:* `{user_id}`",
            f"*Username:* @{username}" if user.username else f"*Username:* {username}",
            f"*Name:* {full_name}",
            f"*Role:* {user_role.value}",
            "",
        ]

        # Role capabilities
        if user_role == UserRole.OWNER:
            lines.append("✅ Full access (Owner)")
            lines.append("  • All commands available")
            lines.append("  • Production approvals")
            lines.append("  • Break-glass operations")
        elif user_role == UserRole.ADMIN:
            lines.append("✅ Admin access")
            lines.append("  • Project management")
            lines.append("  • Production approvals")
            lines.append("  • Testing feedback")
        elif user_role == UserRole.TESTER:
            lines.append("✅ Tester access")
            lines.append("  • Submit feedback")
            lines.append("  • Approve testing deployments")
            lines.append("  • Create projects")
        else:
            lines.append("👁️ Viewer access (read-only)")
            lines.append("  • View projects and status")
            lines.append("  • View notifications")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in whoami_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /health - Check system health (Phase 13.3, updated 13.12)

    ALWAYS responds - no RBAC restriction.
    Shows:
    - System mode (NORMAL/DEGRADED/CRITICAL) - Phase 13.12
    - Controller reachability
    - systemd status (controller + bot)
    - Last deployment timestamp
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "health_check", {})

        lines = ["System Health Check", ""]

        # Phase 13.12: Show system mode prominently
        sys_status = system_state.get_status()
        mode = sys_status["mode"]
        mode_emoji = {"normal": "🟢", "degraded": "🟡", "critical": "🔴"}.get(mode, "⚪")
        lines.append(f"{mode_emoji} System Mode: {mode.upper()}")
        if sys_status["degraded_reason"]:
            lines.append(f"   Reason: {sys_status['degraded_reason']}")
        if sys_status["owner_override_active"]:
            lines.append("   ⚠️ Owner override is active")
        lines.append("")

        # Check controller health
        controller_health = await controller.health_check()
        if "error" in controller_health:
            lines.append("❌ Controller: Unreachable")
            lines.append(f"   Error: {controller_health.get('error', '')}")
        else:
            lines.append("✅ Controller: Healthy")
            controller_phase = controller_health.get('phase', 'unknown')
            controller_version = controller_health.get('version', 'unknown')
            lines.append(f"   Phase: {controller_phase}")
            lines.append(f"   Version: {controller_version}")

            # Safeguard: Check version/phase consistency
            # If bot is 13.x but controller reports phase < 13, log ERROR
            try:
                bot_major = int(BOT_VERSION.split('.')[1])  # "0.13.12" -> 13
                # Extract phase number from "Phase 13.9" or "Phase 12"
                phase_str = controller_phase.replace("Phase ", "").split(".")[0]
                controller_major = int(phase_str)

                if bot_major >= 13 and controller_major < 13:
                    logger.error(
                        f"VERSION MISMATCH: Bot v{BOT_VERSION} (Phase 13+) but "
                        f"controller reports {controller_phase}. Controller needs update!"
                    )
                    lines.append("")
                    lines.append("⚠️ WARNING: Version Mismatch")
                    lines.append(f"   Bot expects Phase 13+, controller reports {controller_phase}")
                    lines.append("   Controller metadata may need updating!")
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse version for consistency check: {e}")

        lines.append("")

        # Check systemd services
        services = ["ai-testing-controller", "ai-telegram-bot"]
        for svc in services:
            status = get_systemd_status(svc)
            emoji = "✅" if status.get("active") else "❌"
            lines.append(f"{emoji} {svc}: {status.get('status')}")
            if status.get("since") and status.get("since") != "unknown":
                lines.append(f"   Since: {status.get('since')}")

        lines.append("")

        # Bot info
        lines.append(f"🤖 Bot Version: {BOT_VERSION}")
        lines.append("")

        # Last deployment info (from dashboard)
        dashboard = await controller.get_dashboard()
        if "error" not in dashboard:
            projects = dashboard.get("projects", [])
            if projects:
                latest_deploy = None
                for proj in projects:
                    for aspect_data in proj.get("aspects", {}).values():
                        deploy_time = aspect_data.get("last_deploy_at")
                        if deploy_time:
                            if not latest_deploy or deploy_time > latest_deploy:
                                latest_deploy = deploy_time

                if latest_deploy:
                    lines.append(f"📦 Last Deployment: {latest_deploy}")
                else:
                    lines.append("📦 Last Deployment: None recorded")

        # Phase 14/15.5: Claude CLI Status (updated for session-based auth)
        lines.append("")
        claude_status = await controller.get_claude_status()
        if "error" in claude_status:
            lines.append("❓ Claude CLI: Status unknown")
        else:
            cli_info = claude_status.get("cli", {})
            scheduler_info = claude_status.get("scheduler", {})

            if claude_status.get("available"):
                lines.append("✅ Claude CLI: Available")
                lines.append(f"   Version: {cli_info.get('version', 'unknown')}")
                # Phase 15.5: Show auth type
                auth_type = cli_info.get("auth_type", "unknown")
                if auth_type == "cli_session":
                    lines.append("   Auth: Session (CLI login)")
                elif auth_type == "api_key":
                    lines.append("   Auth: API Key")
                else:
                    lines.append(f"   Auth: {auth_type}")
                # Show scheduler info if available
                if scheduler_info:
                    active = scheduler_info.get('active_workers', 0)
                    queued = scheduler_info.get('queued_jobs', 0)
                    lines.append(f"   Jobs: {active} running, {queued} queued")
            else:
                lines.append("⚠️ Claude CLI: Not available")
                # Phase 15.5: More specific error messages
                if not cli_info.get("installed"):
                    lines.append("   CLI not installed")
                elif not cli_info.get("authenticated"):
                    lines.append("   Not authenticated (run 'claude auth login')")
                elif cli_info.get("error"):
                    lines.append(f"   Error: {cli_info.get('error', '')[:50]}")

        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        logger.error(f"Error in health_command: {e}")
        await update.message.reply_text(f"❌ Error checking health: {str(e)}")


async def bot_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /status - Bot status (Phase 13.3)

    ALWAYS responds - no RBAC restriction.
    Shows:
    - Version
    - Environment
    - Uptime
    - Memory usage
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "bot_status", {})

        sys_status = get_system_status()
        user_role = get_user_role(user_id)

        environment = os.getenv("ENVIRONMENT", "testing")

        lines = [
            "*Bot Status*",
            "",
            f"*Version:* {sys_status['version']}",
            f"*Environment:* {environment}",
            f"*Uptime:* {sys_status['uptime_human']}",
            f"*Memory:* ~{sys_status['memory_mb']} MB",
            f"*Started:* {sys_status['start_time']}",
            "",
            f"*Your Role:* {user_role.value}",
        ]

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in bot_status_command: {e}")
        await update.message.reply_text(f"❌ Error getting status: {str(e)}")


async def projects_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /projects - List all projects (Phase 13.3)

    ALWAYS responds - no RBAC restriction (read-only).
    Shows:
    - Project list
    - Current phase per project
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "list_projects", {})

        dashboard = await controller.get_dashboard()

        if "error" in dashboard:
            await update.message.reply_text(f"❌ Error: {dashboard.get('error')}")
            return

        projects = dashboard.get("projects", [])

        if not projects:
            await update.message.reply_text("No projects found. Use /new_project to create one.")
            return

        lines = [f"*Projects ({len(projects)})*", ""]

        for proj in projects:
            name = proj.get("project_name", "Unknown")
            status = proj.get("overall_status", "unknown")
            lines.append(f"📁 *{name}*")
            lines.append(f"   Status: {status}")

            aspects = proj.get("aspects", {})
            for aspect_name, aspect_data in aspects.items():
                phase = aspect_data.get("current_phase", "unknown")
                emoji = get_phase_emoji(phase)
                lines.append(f"   {emoji} {aspect_name}: {phase.replace('_', ' ')}")

            lines.append("")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in projects_command: {e}")
        await update.message.reply_text(f"❌ Error listing projects: {str(e)}")


async def deployments_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /deployments - Show deployment history (Phase 13.3)

    ALWAYS responds - no RBAC restriction (read-only).
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "list_deployments", {})

        dashboard = await controller.get_dashboard()

        if "error" in dashboard:
            await update.message.reply_text(f"❌ Error: {dashboard.get('error')}")
            return

        projects = dashboard.get("projects", [])

        if not projects:
            await update.message.reply_text("No projects found.")
            return

        lines = ["*Deployment History*", ""]

        has_deployments = False
        for proj in projects:
            name = proj.get("project_name", "Unknown")

            for aspect_name, aspect_data in proj.get("aspects", {}).items():
                testing_deployed = aspect_data.get("deploy_status") == "deployed_testing"
                prod_deployed = aspect_data.get("deploy_status") == "deployed_production"
                last_deploy = aspect_data.get("last_deploy_at")

                if testing_deployed or prod_deployed or last_deploy:
                    has_deployments = True
                    env_emoji = "🌐" if prod_deployed else "🚀" if testing_deployed else "📦"
                    lines.append(f"{env_emoji} *{name}* / {aspect_name}")

                    if last_deploy:
                        lines.append(f"   Last: {last_deploy}")
                    if testing_deployed:
                        lines.append("   Testing: ✅ Deployed")
                    if prod_deployed:
                        lines.append("   Production: ✅ Deployed")
                    lines.append("")

        if not has_deployments:
            lines.append("No deployments recorded yet.")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in deployments_command: {e}")
        await update.message.reply_text(f"❌ Error listing deployments: {str(e)}")


# -----------------------------------------------------------------------------
# Core Command Handlers
# -----------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - Register user. ALWAYS responds."""
    try:
        user = update.effective_user
        user_role = get_user_role(user.id)
        logger.info(f"User {user.id} ({user.username}) started bot, role: {user_role}")

        await update.message.reply_text(
            f"Welcome to the AI Development Platform!\n\n"
            f"I help you create and manage software projects using natural language.\n\n"
            f"*Your Role:* {user_role.value}\n\n"
            f"*Identity:*\n"
            f"/whoami - Check your identity and permissions\n\n"
            f"*Health & Status (Read-Only):*\n"
            f"/health - System health check\n"
            f"/status - Bot status\n"
            f"/projects - List all projects\n"
            f"/deployments - Deployment history\n\n"
            f"*Project Management:*\n"
            f"/new\\_project - Start a new project\n"
            f"/dashboard - System overview\n\n"
            f"*Testing & Approval:*\n"
            f"/approve - Approve deployment\n"
            f"/reject - Reject with reason\n"
            f"/feedback - Submit feedback\n"
            f"/prod\\_approve - Production approval\n\n"
            f"/notifications - View pending actions\n"
            f"/help - Show detailed help\n\n"
            f"Or just describe your project in plain English!",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in start_command: {e}")
        await update.message.reply_text(f"Welcome! Error loading full message: {str(e)}")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command. ALWAYS responds."""
    try:
        user_role = get_user_role(update.effective_user.id)

        base_help = """
*AI Development Platform - Help*

*Identity:*
/whoami - Check your user ID and role

*Health & Status (All Users):*
/health - Check system health
/status - Bot version and uptime
/projects - List all projects
/deployments - Deployment history

*Project Management:*
/new\\_project - Create a new project
/dashboard - System overview

*Testing & Approval:*
/approve <project> <aspect> - Approve testing
/reject <project> <aspect> - Reject with reason
/feedback <project> <aspect> - Submit feedback
/prod\\_approve <project> <aspect> - Production approval

*Lifecycle Management (Phase 15.1):*
/lifecycle\\_status [id] - View lifecycle state
/lifecycle\\_approve <id> - Approve transition
/lifecycle\\_reject <id> <reason> - Reject lifecycle
/lifecycle\\_feedback <id> <text> - Submit feedback
/lifecycle\\_prod\\_approve <id> - Production approval

*Runtime Intelligence (Phase 17A):*
/signals [project] - View signals summary
/signals\\_recent [hours] [limit] - Recent signals
/runtime\\_status - Collection status

*Incident Classification (Phase 17B):*
/incidents [project] - View incidents summary
/incidents\\_recent [hours] [limit] - Recent incidents
/incidents\\_summary - Detailed incident statistics

*Notifications:*
/notifications - View pending actions
"""

        role_info = f"\n*Your Role:* {user_role.value}\n"

        if user_role == UserRole.OWNER:
            role_info += "_(Full access including break-glass operations)_"
        elif user_role == UserRole.ADMIN:
            role_info += "_(Can approve deployments and manage projects)_"
        elif user_role == UserRole.TESTER:
            role_info += "_(Can test and provide feedback)_"
        else:
            role_info += "_(Read-only access)_"

        await update.message.reply_text(base_help + role_info, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in help_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def new_project_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new_project command."""
    user_id = update.effective_user.id
    safety.log_action(user_id, "new_project_start", {})

    if context.args:
        description = " ".join(context.args)
        await create_project_from_description(update, description, str(user_id))
    else:
        user_state.awaiting_input[user_id] = "project_description"
        await update.message.reply_text(
            "Please describe your project in plain English.\n\n"
            "Example: \"Build a SaaS CRM with admin panel and customer website\""
        )


async def project_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /project_status command - detailed project view. ALWAYS responds."""
    try:
        if not context.args:
            result = await controller.get_dashboard()
            if "error" in result:
                await update.message.reply_text(f"Error: {result.get('error')}")
                return

            await update.message.reply_text(
                format_dashboard(result),
                parse_mode="Markdown"
            )
        else:
            project_name = context.args[0]
            result = await controller.get_project(project_name)

            if "error" in result:
                await update.message.reply_text(f"Error: {result.get('error')}")
                return

            await update.message.reply_text(
                format_project_status(result),
                parse_mode="Markdown"
            )
    except Exception as e:
        logger.error(f"Error in project_status_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle /dashboard command (Phase 16B Enhanced).

    Returns: Summary counts, Active projects, Active jobs, System health snapshot.
    ALWAYS responds.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_dashboard", {})

        # Try enhanced dashboard first (Phase 16B)
        try:
            result = await controller.get_dashboard_summary()
            if "error" not in result:
                await update.message.reply_text(
                    format_dashboard_enhanced(result),
                    parse_mode="Markdown"
                )
                return
        except Exception as e:
            logger.warning(f"Enhanced dashboard unavailable, falling back: {e}")

        # Fallback to legacy dashboard
        result = await controller.get_dashboard()

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        await update.message.reply_text(
            format_dashboard(result),
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in dashboard_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 13.5: Human Approval Workflow Commands
# -----------------------------------------------------------------------------
@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /approve <project> <aspect> - Approve testing (Phase 13.5)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /approve <project_name> <aspect>\n"
            "Example: /approve my-crm core"
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]
    user_id = str(update.effective_user.id)

    safety.log_action(int(user_id), "approve_testing", {
        "project": project_name,
        "aspect": aspect
    })

    result = await controller.submit_feedback(
        project_name=project_name,
        aspect=aspect,
        feedback_type=FeedbackType.APPROVE.value,
        user_id=user_id
    )

    if "error" in result:
        await update.message.reply_text(f"❌ Error: {result.get('error')}")
        return

    await update.message.reply_text(
        f"✅ *Approved!*\n\n"
        f"Project: {project_name}\n"
        f"Aspect: {aspect}\n"
        f"Action: {result.get('action_taken', 'Unknown')}\n\n"
        f"Next steps:\n" + "\n".join(f"• {s}" for s in result.get('next_steps', [])),
        parse_mode="Markdown"
    )


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def reject_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /reject <project> <aspect> - Reject with structured reason (Phase 13.5)

    Rejection MUST include structured reason:
    - Bug
    - Improvement
    - Invalid Requirement
    - Other
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /reject <project_name> <aspect>\n"
            "Example: /reject my-crm core"
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]
    user_id = update.effective_user.id

    safety.log_action(user_id, "reject_start", {
        "project": project_name,
        "aspect": aspect
    })

    # Show rejection reason options
    await update.message.reply_text(
        f"*Rejection for {project_name} - {aspect}*\n\n"
        f"Please select the rejection reason:",
        parse_mode="Markdown",
        reply_markup=get_rejection_reason_keyboard(project_name, aspect)
    )


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /feedback <project> <aspect> - Submit feedback (Phase 13.5)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /feedback <project_name> <aspect>\n"
            "Example: /feedback my-crm core"
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]
    user_id = update.effective_user.id

    safety.log_action(user_id, "feedback_start", {
        "project": project_name,
        "aspect": aspect
    })

    await update.message.reply_text(
        f"*Feedback for {project_name} - {aspect}*\n\n"
        f"Select your feedback type:",
        parse_mode="Markdown",
        reply_markup=get_feedback_keyboard(project_name, aspect)
    )


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def prod_approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /prod_approve <project> <aspect> - Production approval (Phase 13.5)

    Enforces:
    - Dual approval (same user cannot approve twice)
    - Requires justification
    - Requires rollback plan
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /prod_approve <project_name> <aspect>\n"
            "Example: /prod_approve my-crm core\n\n"
            "Note: You will be asked for justification and rollback plan.\n"
            "Dual approval is required (different users must approve)."
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]
    user_id = update.effective_user.id
    deployment_id = f"{project_name}:{aspect}"

    # Check dual approval
    can_approve, error_msg = safety.check_dual_approval(deployment_id, user_id)
    if not can_approve:
        await update.message.reply_text(f"⛔ {error_msg}")
        return

    safety.log_action(user_id, "prod_approve_start", {
        "project": project_name,
        "aspect": aspect
    })

    user_state.pending_feedback[user_id] = {
        "type": "prod_approval",
        "project_name": project_name,
        "aspect": aspect,
        "deployment_id": deployment_id,
        "step": "justification"
    }
    user_state.awaiting_input[user_id] = "prod_justification"

    await update.message.reply_text(
        f"*Production Approval Request*\n\n"
        f"Project: {project_name}\n"
        f"Aspect: {aspect}\n\n"
        f"⚠️ This will deploy to PRODUCTION.\n\n"
        f"Please provide your justification for this production deployment:",
        parse_mode="Markdown"
    )


async def notifications_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /notifications command. ALWAYS responds."""
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_notifications", {})

        project_name = context.args[0] if context.args else None
        result = await controller.get_notifications(project_name)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        notifications = result.get("notifications", [])
        if not notifications:
            await update.message.reply_text("✅ No pending notifications.")
            return

        first_notif = notifications[0]
        text, keyboard = format_notification(first_notif)
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)

        if len(notifications) > 1:
            lines = [f"\n*+{len(notifications) - 1} more notifications:*\n"]
            for notif in notifications[1:10]:
                emoji = "🔔"
                if notif.get("notification_type") == "testing_ready":
                    emoji = "🎯"
                elif notif.get("notification_type") == "approval_required":
                    emoji = "⚠️"
                elif notif.get("notification_type") == "ci_result":
                    emoji = "🔧"

                lines.append(f"{emoji} {notif.get('title', 'Notification')}")

            await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Error in notifications_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 22: Rescue System Commands
# -----------------------------------------------------------------------------

@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def rescue_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /rescue <project> - View rescue status or reset rescue state.

    Phase 22: Rescue & Recovery System command.

    Usage:
      /rescue <project>         - View rescue status for project
      /rescue <project> reset   - Reset rescue state and re-validate
      /rescue <project> validate - Manually trigger validation
    """
    try:
        from controller.rescue_engine import get_rescue_engine

        if not context.args:
            await update.message.reply_text(
                "*🔧 Rescue System*\n\n"
                "Usage:\n"
                "• `/rescue <project>` - View rescue status\n"
                "• `/rescue <project> reset` - Reset rescue state\n"
                "• `/rescue <project> validate` - Manually validate deployment\n",
                parse_mode="Markdown"
            )
            return

        project_name = context.args[0]
        action = context.args[1] if len(context.args) > 1 else "status"

        rescue_engine = get_rescue_engine()

        if action == "status":
            # Show rescue status
            state = rescue_engine.get_rescue_state(project_name)
            if state is None:
                await update.message.reply_text(
                    f"*🔧 Rescue Status: {escape_markdown(project_name)}*\n\n"
                    f"No rescue attempts recorded for this project.\n"
                    f"Use `/rescue {project_name} validate` to check deployment.",
                    parse_mode="Markdown"
                )
            else:
                attempts = len(state.attempts)
                status_emoji = "✅" if state.resolved else "🔄"
                status_text = "Resolved" if state.resolved else "In Progress"

                message = (
                    f"*🔧 Rescue Status: {escape_markdown(project_name)}*\n\n"
                    f"*Status:* {status_emoji} {status_text}\n"
                    f"*Attempts:* {attempts}/3\n"
                    f"*Deployment Job:* `{state.deployment_job_id[:12]}...`\n\n"
                    f"*Attempt History:*\n"
                )

                for attempt in state.attempts:
                    emoji = "✅" if attempt.success else "❌"
                    message += (
                        f"{emoji} Attempt {attempt.attempt_number}: "
                        f"{attempt.failure_type} - {attempt.outcome or 'Pending'}\n"
                    )

                if not state.resolved and attempts >= 3:
                    message += "\n⚠️ *Max attempts reached.* Use `/rescue {project_name} reset` to retry."

                await update.message.reply_text(message, parse_mode="Markdown")

        elif action == "reset":
            # Reset rescue state
            rescue_engine.clear_rescue_state(project_name)
            await update.message.reply_text(
                f"*🔧 Rescue Reset: {escape_markdown(project_name)}*\n\n"
                f"Rescue state cleared. You can now:\n"
                f"• Use `/rescue {project_name} validate` to check deployment\n"
                f"• Wait for automatic validation on next deployment",
                parse_mode="Markdown"
            )

        elif action == "validate":
            # Manually trigger validation
            urls = await get_project_deployment_urls(project_name)
            if not urls:
                await update.message.reply_text(
                    f"*❌ No URLs found for {escape_markdown(project_name)}*\n\n"
                    f"Cannot validate - no deployment URLs configured.",
                    parse_mode="Markdown"
                )
                return

            await update.message.reply_text(
                f"*🔍 Validating {escape_markdown(project_name)}...*\n\n"
                f"Checking endpoints:\n"
                f"• API: {urls.get('api', 'N/A')}\n"
                f"• Frontend: {urls.get('frontend', 'N/A')}\n"
                f"• Admin: {urls.get('admin', 'N/A')}",
                parse_mode="Markdown"
            )

            # Trigger validation
            mock_job = {"job_id": f"manual-{project_name}-{datetime.utcnow().timestamp()}"}
            asyncio.create_task(
                trigger_deployment_validation(
                    context.application, project_name, mock_job, urls
                )
            )

        else:
            await update.message.reply_text(
                f"Unknown action: {action}\n"
                f"Valid actions: status, reset, validate"
            )

    except ImportError:
        await update.message.reply_text(
            "❌ Rescue system not available.\n"
            "Please check that controller modules are deployed."
        )
    except Exception as e:
        logger.error(f"Error in rescue_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def rescue_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /rescue_history <project> - View detailed rescue history.

    Phase 22: Rescue & Recovery System history view.
    """
    try:
        from controller.rescue_engine import get_rescue_engine

        if not context.args:
            await update.message.reply_text(
                "Usage: `/rescue_history <project>`",
                parse_mode="Markdown"
            )
            return

        project_name = context.args[0]
        rescue_engine = get_rescue_engine()
        history = rescue_engine.get_rescue_history(project_name)

        if not history:
            await update.message.reply_text(
                f"*📜 Rescue History: {escape_markdown(project_name)}*\n\n"
                f"No rescue attempts recorded.",
                parse_mode="Markdown"
            )
            return

        message = f"*📜 Rescue History: {escape_markdown(project_name)}*\n\n"

        for attempt in history:
            emoji = "✅" if attempt.success else "❌"
            message += (
                f"{emoji} *Attempt {attempt.attempt_number}*\n"
                f"   Job: `{attempt.job_id[:12]}...`\n"
                f"   Failure: {attempt.failure_type}\n"
                f"   Created: {attempt.created_at[:16]}\n"
            )
            if attempt.completed_at:
                message += f"   Completed: {attempt.completed_at[:16]}\n"
            if attempt.outcome:
                message += f"   Outcome: {attempt.outcome}\n"
            message += "\n"

        await update.message.reply_text(message, parse_mode="Markdown")

    except ImportError:
        await update.message.reply_text(
            "❌ Rescue system not available."
        )
    except Exception as e:
        logger.error(f"Error in rescue_history_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Message Handler (Natural Language Input)
# -----------------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages - Natural language project creation."""
    try:
        user_id = update.effective_user.id
        text = update.message.text.strip()

        awaiting = user_state.awaiting_input.get(user_id)

        if awaiting == "project_description":
            del user_state.awaiting_input[user_id]
            await create_project_from_description(update, text, str(user_id))
            return

        elif awaiting == "feedback_explanation":
            feedback_ctx = user_state.pending_feedback.get(user_id)
            if feedback_ctx:
                is_valid, error_msg = safety.validate_feedback_has_explanation(
                    feedback_ctx["feedback_type"],
                    text
                )
                if not is_valid:
                    await update.message.reply_text(
                        f"Warning: {error_msg}\n\nPlease provide more details:"
                    )
                    return

                del user_state.awaiting_input[user_id]
                del user_state.pending_feedback[user_id]

                safety.log_action(user_id, "submit_feedback", {
                    "project": feedback_ctx["project_name"],
                    "aspect": feedback_ctx["aspect"],
                    "type": feedback_ctx["feedback_type"]
                })

                result = await controller.submit_feedback(
                    project_name=feedback_ctx["project_name"],
                    aspect=feedback_ctx["aspect"],
                    feedback_type=feedback_ctx["feedback_type"],
                    user_id=str(user_id),
                    explanation=text
                )

                if "error" in result:
                    await update.message.reply_text(f"Error: {result.get('error')}")
                    return

                emoji_map = {"bug": "Bug", "improvements": "Improvement", "reject": "Rejected"}
                feedback_label = emoji_map.get(feedback_ctx["feedback_type"], "Feedback")
                await update.message.reply_text(
                    f"*{feedback_label} Submitted*\n\n"
                    f"Project: {feedback_ctx['project_name']}\n"
                    f"Aspect: {feedback_ctx['aspect']}\n"
                    f"Type: {feedback_ctx['feedback_type']}\n"
                    f"Action: {result.get('action_taken', 'Processed')}\n\n"
                    f"Next steps:\n" + "\n".join(f"- {s}" for s in result.get('next_steps', [])),
                    parse_mode="Markdown"
                )
            return

        elif awaiting == "rejection_explanation":
            feedback_ctx = user_state.pending_feedback.get(user_id)
            if feedback_ctx:
                if len(text.strip()) < 10:
                    await update.message.reply_text(
                        "Rejection reason must be at least 10 characters. Please explain:"
                    )
                    return

                del user_state.awaiting_input[user_id]
                del user_state.pending_feedback[user_id]

                safety.log_action(user_id, "submit_rejection", {
                    "project": feedback_ctx["project_name"],
                    "aspect": feedback_ctx["aspect"],
                    "reason": feedback_ctx.get("rejection_reason", "other")
                })

                rejection_reason = feedback_ctx.get("rejection_reason", "other")
                result = await controller.submit_feedback(
                    project_name=feedback_ctx["project_name"],
                    aspect=feedback_ctx["aspect"],
                    feedback_type=FeedbackType.REJECT.value,
                    user_id=str(user_id),
                    explanation=f"[{rejection_reason}] {text}"
                )

                if "error" in result:
                    await update.message.reply_text(f"Error: {result.get('error')}")
                    return

                await update.message.reply_text(
                    f"*Rejected*\n\n"
                    f"Project: {feedback_ctx['project_name']}\n"
                    f"Aspect: {feedback_ctx['aspect']}\n"
                    f"Reason: {rejection_reason}\n"
                    f"Action: {result.get('action_taken', 'Processed')}\n\n"
                    f"Next steps:\n" + "\n".join(f"- {s}" for s in result.get('next_steps', [])),
                    parse_mode="Markdown"
                )
            return

        elif awaiting == "prod_justification":
            feedback_ctx = user_state.pending_feedback.get(user_id)
            if feedback_ctx:
                if len(text.strip()) < 20:
                    await update.message.reply_text(
                        "Justification must be at least 20 characters. Please explain why:"
                    )
                    return

                feedback_ctx["justification"] = text
                feedback_ctx["step"] = "rollback_plan"
                user_state.awaiting_input[user_id] = "prod_rollback_plan"

                await update.message.reply_text(
                    "Please provide your rollback plan in case of issues:"
                )
            return

        elif awaiting == "prod_rollback_plan":
            feedback_ctx = user_state.pending_feedback.get(user_id)
            if feedback_ctx:
                if len(text.strip()) < 10:
                    await update.message.reply_text(
                        "Rollback plan must be at least 10 characters. Please describe:"
                    )
                    return

                del user_state.awaiting_input[user_id]

                deployment_id = feedback_ctx.get("deployment_id")

                safety.log_action(user_id, "prod_approval_attempt", {
                    "project": feedback_ctx["project_name"],
                    "aspect": feedback_ctx["aspect"]
                })

                result = await controller.approve_production(
                    project_name=feedback_ctx["project_name"],
                    aspect=feedback_ctx["aspect"],
                    user_id=str(user_id),
                    justification=feedback_ctx["justification"],
                    risk_acknowledged=True,
                    rollback_plan=text
                )

                if "error" in result:
                    safety.log_action(user_id, "prod_approval_failed", {
                        "project": feedback_ctx["project_name"],
                        "error": result.get("error")
                    })
                    del user_state.pending_feedback[user_id]
                    await update.message.reply_text(f"Error: {result.get('error')}")
                    return

                if deployment_id:
                    if deployment_id not in user_state.approval_history:
                        user_state.approval_history[deployment_id] = []
                    user_state.approval_history[deployment_id].append({
                        "user_id": user_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "type": "production_approval"
                    })

                del user_state.pending_feedback[user_id]

                if result.get("approved"):
                    safety.log_action(user_id, "prod_approval_success", {
                        "project": feedback_ctx["project_name"],
                        "aspect": feedback_ctx["aspect"],
                        "approval_id": result.get("approval_id")
                    })
                    await update.message.reply_text(
                        f"*Production Deployment Approved*\n\n"
                        f"Project: {feedback_ctx['project_name']}\n"
                        f"Aspect: {feedback_ctx['aspect']}\n"
                        f"Status: {result.get('deployment_status', 'Processing')}\n"
                        f"URL: {result.get('production_url', 'Pending')}",
                        parse_mode="Markdown"
                    )
                else:
                    await update.message.reply_text(
                        f"*Production Approval Failed*\n\n"
                        f"{result.get('message', 'Unknown error')}",
                        parse_mode="Markdown"
                    )
            return

        # Default: Treat as project description
        user_role = get_user_role(user_id)
        if user_role in [UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER]:
            if len(text) > 20 and any(word in text.lower() for word in [
                "build", "create", "make", "develop", "want", "need", "app", "website",
                "system", "platform", "saas", "crm", "api", "service"
            ]):
                await create_project_from_description(update, text, str(user_id))
                return

        await update.message.reply_text(
            "I did not understand that. You can:\n\n"
            "- Describe a project to create\n"
            "- Use /help to see available commands\n"
            "- Use /projects to view projects\n"
            "- Use /health to check system status\n"
            "- Use /whoami to check your identity"
        )
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text(f"Error processing message: {str(e)}")


async def create_project_from_description(
    update: Update,
    description: str,
    user_id: str
) -> None:
    """
    Create a project from natural language description (Phase 16C Enhanced).

    Uses the unified ProjectService with:
    - CHD validation
    - Project registry
    - Lifecycle creation
    - Progress feedback
    """
    progress_message = None
    last_step = ""

    async def progress_callback(step: str, status: str, details: dict = None):
        """Update progress message in Telegram."""
        nonlocal progress_message, last_step

        # Progress emojis
        step_emojis = {
            "input_received": "📥",
            "parsing": "🧠",
            "validating": "📋",
            "creating_project": "📁",
            "creating_lifecycles": "🔄",
            "scheduling_planning": "🚀",
            "completed": "✅",
        }

        step_labels = {
            "input_received": "Input received",
            "parsing": "Parsing requirements",
            "validating": "Validating execution plan",
            "creating_project": "Creating project",
            "creating_lifecycles": "Initializing aspects",
            "scheduling_planning": "Scheduling planning",
            "completed": "Project created successfully",
        }

        emoji = step_emojis.get(step, "⏳")
        label = step_labels.get(step, step)

        if status == "failed":
            emoji = "❌"
            label = f"{label} - FAILED"

        # Build progress text
        lines = ["*Project Creation Progress*", ""]
        for s, lbl in step_labels.items():
            if s == step:
                lines.append(f"{emoji} {lbl}")
                last_step = s
                if status == "completed" and s != "completed":
                    lines[-1] = f"✅ {lbl}"
            elif list(step_labels.keys()).index(s) < list(step_labels.keys()).index(step):
                lines.append(f"✅ {lbl}")
            else:
                lines.append(f"⬜ {lbl}")

        progress_text = "\n".join(lines)

        try:
            if progress_message:
                await progress_message.edit_text(progress_text, parse_mode="Markdown")
            else:
                progress_message = await update.message.reply_text(progress_text, parse_mode="Markdown")
        except Exception as e:
            logger.debug(f"Could not update progress: {e}")

    try:
        logger.info(f"Creating project from description: {description[:50]}...")

        # Send initial progress
        await progress_callback("input_received", "in_progress", {"source": "text"})

        # Phase 16E: Check for conflicts FIRST before project creation
        try:
            from controller.project_registry import get_registry
            from controller.project_decision_engine import (
                evaluate_project_creation,
                DecisionType,
            )

            await progress_callback("validating", "in_progress", None)

            registry = get_registry()
            existing_identities = registry.get_all_identities()

            # Evaluate for conflicts
            decision = evaluate_project_creation(
                description=description,
                requirements=description,
                tech_stack=None,
                aspects=None,
                existing_identities=existing_identities,
            )

            # If conflict detected, show user choices
            if decision.requires_user_confirmation:
                conflict_id = f"conf_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                existing_project = decision.existing_project_name or "unknown"

                # Store conflict state for callback
                user_id_int = int(user_id) if user_id.isdigit() else hash(user_id) % (10**9)
                user_state.pending_conflicts[user_id_int] = {
                    "conflict_id": conflict_id,
                    "description": description,
                    "existing_project": existing_project,
                    "decision": decision.to_dict(),
                    "created_at": datetime.utcnow().isoformat(),
                }

                # Format conflict message
                conflict_text = (
                    f"⚠️ *Conflict Detected*\n\n"
                    f"A similar project already exists:\n"
                    f"*Existing:* `{existing_project}`\n\n"
                    f"*Similarity:* {decision.confidence * 100:.0f}%\n"
                    f"*Reason:* {decision.explanation}\n\n"
                )

                if decision.differences:
                    conflict_text += "*Key differences:*\n"
                    for diff in decision.differences[:3]:
                        conflict_text += f"  • {diff}\n"
                    conflict_text += "\n"

                conflict_text += "*Choose an action:*"

                await update.message.reply_text(
                    conflict_text,
                    parse_mode="Markdown",
                    reply_markup=get_conflict_resolution_keyboard(
                        existing_project=existing_project,
                        new_description=description[:50],
                        conflict_id=conflict_id
                    )
                )
                return

            await progress_callback("validating", "completed", None)

        except ImportError as e:
            logger.warning(f"Decision engine not available: {e}")

        # Phase 19 fix: Use controller API for project creation
        # This ensures the controller's registry singleton is the single source of truth
        # The bot should NOT directly import project_service to avoid registry split-brain
        result = await controller.create_project(
            description=description,
            user_id=user_id
        )

        if not result.get("success", False) and "error" in result:
            error_msg = extract_api_error(result)
            validation = result.get("validation_result", {})

            # Escape all dynamic content to prevent Markdown parsing errors
            safe_error_msg = escape_markdown(error_msg)
            error_lines = [f"*Error Creating Project*", "", f"❌ {safe_error_msg}"]

            if validation.get("suggestions"):
                error_lines.append("")
                error_lines.append("*Suggestions:*")
                for suggestion in validation.get("suggestions", []):
                    error_lines.append(f"  • {escape_markdown(suggestion)}")

            await update.message.reply_text("\n".join(error_lines), parse_mode="Markdown")
            return

        # Success - format response
        project_name = result.get("project_name", "Unknown")
        project_id = result.get("project_id", "Unknown")
        aspects = result.get("aspects", result.get("aspects_initialized", []))
        next_steps = result.get("next_steps", [])

        # Escape all dynamic content to prevent Markdown parsing errors
        safe_project_name = escape_markdown(project_name)
        safe_project_id = escape_markdown(project_id[:8])
        aspects_list = "\n".join(f"  • {escape_markdown(a)}" for a in aspects) if aspects else "  • None"
        next_steps_list = "\n".join(f"  • {escape_markdown(s)}" for s in next_steps) if next_steps else ""

        success_text = (
            f"🎉 *Project Created Successfully!*\n\n"
            f"*Name:* `{safe_project_name}`\n"
            f"*ID:* `{safe_project_id}...`\n\n"
            f"*Aspects:*\n{aspects_list}\n"
        )

        if next_steps_list:
            success_text += f"\n*Next Steps:*\n{next_steps_list}\n"

        success_text += f"\nUse /dashboard to view all projects."

        await update.message.reply_text(success_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in create_project_from_description: {e}", exc_info=True)
        # Don't use Markdown in error message to avoid parsing issues with dynamic content
        await update.message.reply_text(
            f"❌ Error creating project\n\n"
            f"An unexpected error occurred: {str(e)}\n\n"
            f"Please try again or contact support."
        )


async def create_project_from_file(
    update: Update,
    filename: str,
    file_content: bytes,
    user_id: str
) -> None:
    """
    Create a project from uploaded file (Phase 16C).

    Supports .md and .txt files containing project requirements.
    """
    progress_message = None

    async def progress_callback(step: str, status: str, details: dict = None):
        """Update progress message."""
        nonlocal progress_message

        step_emojis = {
            "input_received": "📥",
            "parsing": "🧠",
            "validating": "📋",
            "creating_project": "📁",
            "creating_lifecycles": "🔄",
            "scheduling_planning": "🤖",
            "completed": "✅",
        }

        step_labels = {
            "input_received": f"File received: {filename}",
            "parsing": "Parsing requirements",
            "validating": "Validating execution plan",
            "creating_project": "Creating project",
            "creating_lifecycles": "Initializing aspects",
            "scheduling_planning": "Scheduling Claude planning job",
            "completed": "Project created successfully",
        }

        emoji = step_emojis.get(step, "⏳")
        label = step_labels.get(step, step)

        if status == "failed":
            emoji = "❌"

        lines = ["*Project Creation Progress*", ""]
        for s, lbl in step_labels.items():
            if s == step:
                lines.append(f"{emoji} {lbl}")
            elif list(step_labels.keys()).index(s) < list(step_labels.keys()).index(step):
                lines.append(f"✅ {lbl}")
            else:
                lines.append(f"⬜ {lbl}")

        try:
            if progress_message:
                await progress_message.edit_text("\n".join(lines), parse_mode="Markdown")
            else:
                progress_message = await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
        except Exception:
            pass

    try:
        logger.info(f"Creating project from file: {filename}")

        # Phase 19 fix: Show progress steps during file upload
        await progress_callback("input_received", "in_progress", {"source": "file"})

        # Phase 19 fix: Use controller API for project creation
        # This ensures the controller's registry singleton is the single source of truth

        # Validate file extension
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
        if ext not in [".md", ".txt", ".text"]:
            await progress_callback("input_received", "failed", {"error": "invalid_type"})
            await update.message.reply_text(
                f"❌ Invalid file type. Allowed: .md, .txt, .text"
            )
            return

        # Decode file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            try:
                content = file_content.decode("latin-1")
            except Exception:
                await progress_callback("input_received", "failed", {"error": "decode_error"})
                await update.message.reply_text("❌ Could not decode file content")
                return

        # Check minimum content
        if len(content.strip()) < 20:
            await progress_callback("input_received", "failed", {"error": "too_short"})
            await update.message.reply_text("❌ File content too short. Minimum: 20 characters")
            return

        await progress_callback("input_received", "completed", {"size": len(content)})
        await progress_callback("parsing", "in_progress", None)

        # Short delay for visual feedback
        await asyncio.sleep(0.3)
        await progress_callback("validating", "in_progress", None)
        await asyncio.sleep(0.3)
        await progress_callback("creating_project", "in_progress", None)

        # Use controller API with file content as requirements
        # The first 500 chars as description, full content as requirements
        # The CHD validator will extract project_name from the requirements
        logger.info(f"Calling controller.create_project with description length={len(content[:500])}, user_id={user_id}")
        result = await controller.create_project(
            description=content[:500],
            user_id=user_id,
            requirements=[content]  # Pass full content so CHD can extract project_name
        )
        logger.info(f"Controller response: success={result.get('success')}, has_error={'error' in result}, result_keys={list(result.keys())}")

        if not result.get("success", False):
            await progress_callback("creating_project", "failed", None)
            # Log full result for debugging conflict detection
            metadata = result.get("metadata", {})
            logger.info(f"Project creation failed - metadata: {metadata}, message: {result.get('message')}")
            error_msg = extract_api_error(result)
            logger.info(f"Extracted error message: {error_msg}")
            safe_error = escape_markdown(error_msg)
            await update.message.reply_text(
                f"❌ *Error Creating Project*\n\n{safe_error}",
                parse_mode="Markdown"
            )
            return

        # Show remaining progress steps
        await progress_callback("creating_lifecycles", "in_progress", None)
        await asyncio.sleep(0.2)
        await progress_callback("scheduling_planning", "in_progress", None)
        await asyncio.sleep(0.2)
        await progress_callback("completed", "completed", None)

        # Success
        project_name = result.get("project_name", "Unknown")
        # Phase 19 fix: API returns aspects_initialized, not aspects
        aspects = result.get("aspects_initialized", result.get("aspects", []))
        next_steps = result.get("next_steps", [])

        # Build response message
        aspects_str = ", ".join(aspects) if aspects else "None detected"
        next_steps_str = "\n".join(f"  • {escape_markdown(s)}" for s in next_steps[:3]) if next_steps else ""

        # Escape all dynamic content to prevent Markdown parsing errors
        safe_project_name = escape_markdown(project_name)
        safe_aspects_str = escape_markdown(aspects_str)

        response_text = (
            f"🎉 *Project Created from File!*\n\n"
            f"*Name:* `{safe_project_name}`\n"
            f"*Aspects:* {safe_aspects_str}\n"
        )

        if next_steps_str:
            response_text += f"\n*Next Steps:*\n{next_steps_str}\n"

        response_text += f"\nUse /dashboard to view all projects."

        await update.message.reply_text(response_text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error creating project from file: {e}", exc_info=True)
        # Don't use Markdown in error message to avoid parsing issues
        await update.message.reply_text(
            f"❌ Error processing file: {str(e)}"
        )


# -----------------------------------------------------------------------------
# Callback Query Handler (Inline Buttons)
# -----------------------------------------------------------------------------
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks."""
    try:
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        user_role = get_user_role(user_id)
        data = query.data
        parts = data.split(":")

        if len(parts) < 3:
            await query.edit_message_text("Invalid callback data.")
            return

        action = parts[0]
        project_name = parts[1]
        aspect = parts[2]

        # Check role for approval actions (compare string values)
        if action in [CallbackAction.FEEDBACK_APPROVE.value, CallbackAction.PROD_APPROVE.value]:
            if user_role not in [UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER]:
                await query.edit_message_text("Access denied. Insufficient permissions.")
                return

        if action == CallbackAction.FEEDBACK_APPROVE.value:
            safety.log_action(user_id, "approve_via_button", {
                "project": project_name,
                "aspect": aspect
            })

            result = await controller.submit_feedback(
                project_name=project_name,
                aspect=aspect,
                feedback_type=FeedbackType.APPROVE.value,
                user_id=str(user_id)
            )

            if "error" in result:
                await query.edit_message_text(f"Error: {result.get('error')}")
                return

            await query.edit_message_text(
                f"*Approved!*\n\n"
                f"Project: {project_name}\n"
                f"Aspect: {aspect}\n"
                f"Action: {result.get('action_taken', 'Unknown')}\n\n"
                f"Next steps:\n" + "\n".join(f"- {s}" for s in result.get('next_steps', [])),
                parse_mode="Markdown"
            )

        elif action in [
            CallbackAction.FEEDBACK_BUG.value,
            CallbackAction.FEEDBACK_IMPROVEMENTS.value,
            CallbackAction.FEEDBACK_REJECT.value
        ]:
            feedback_type_map = {
                CallbackAction.FEEDBACK_BUG.value: FeedbackType.BUG.value,
                CallbackAction.FEEDBACK_IMPROVEMENTS.value: FeedbackType.IMPROVEMENTS.value,
                CallbackAction.FEEDBACK_REJECT.value: FeedbackType.REJECT.value,
            }
            feedback_type = feedback_type_map[action]

            if action == CallbackAction.FEEDBACK_REJECT.value:
                # Show rejection reason options
                await query.edit_message_text(
                    f"*Rejection for {project_name} - {aspect}*\n\n"
                    f"Please select the rejection reason:",
                    parse_mode="Markdown",
                    reply_markup=get_rejection_reason_keyboard(project_name, aspect)
                )
                return

            user_state.pending_feedback[user_id] = {
                "project_name": project_name,
                "aspect": aspect,
                "feedback_type": feedback_type
            }
            user_state.awaiting_input[user_id] = "feedback_explanation"

            type_label = feedback_type.replace("_", " ").title()
            await query.edit_message_text(
                f"*{type_label} Feedback*\n\n"
                f"Project: {project_name}\n"
                f"Aspect: {aspect}\n\n"
                f"Please describe the issue or suggestion in detail.\n"
                f"(This explanation is mandatory)",
                parse_mode="Markdown"
            )

        elif action in [
            CallbackAction.REJECT_BUG.value,
            CallbackAction.REJECT_IMPROVEMENT.value,
            CallbackAction.REJECT_INVALID.value,
            CallbackAction.REJECT_OTHER.value
        ]:
            reason_map = {
                CallbackAction.REJECT_BUG.value: RejectionReason.BUG.value,
                CallbackAction.REJECT_IMPROVEMENT.value: RejectionReason.IMPROVEMENT.value,
                CallbackAction.REJECT_INVALID.value: RejectionReason.INVALID_REQUIREMENT.value,
                CallbackAction.REJECT_OTHER.value: RejectionReason.OTHER.value,
            }
            rejection_reason = reason_map[action]

            user_state.pending_feedback[user_id] = {
                "project_name": project_name,
                "aspect": aspect,
                "feedback_type": FeedbackType.REJECT.value,
                "rejection_reason": rejection_reason
            }
            user_state.awaiting_input[user_id] = "rejection_explanation"

            await query.edit_message_text(
                f"*Rejection: {rejection_reason.replace('_', ' ').title()}*\n\n"
                f"Project: {project_name}\n"
                f"Aspect: {aspect}\n\n"
                f"Please explain the rejection reason in detail:",
                parse_mode="Markdown"
            )

        # Phase 16E: Conflict Resolution Callbacks
        elif action in [
            CallbackAction.CONFLICT_IMPROVE.value,
            CallbackAction.CONFLICT_ADD_MODULE.value,
            CallbackAction.CONFLICT_NEW_VERSION.value,
            CallbackAction.CONFLICT_CANCEL.value
        ]:
            # project_name is actually existing_project, aspect is conflict_id
            existing_project = project_name
            conflict_id = aspect

            # Get stored conflict state
            conflict_state = user_state.pending_conflicts.get(user_id, {})
            description = conflict_state.get("description", "")

            safety.log_action(user_id, f"conflict_resolution_{action}", {
                "existing_project": existing_project,
                "conflict_id": conflict_id,
                "choice": action
            })

            if action == CallbackAction.CONFLICT_CANCEL.value:
                # Clear conflict state
                user_state.pending_conflicts.pop(user_id, None)
                await query.edit_message_text(
                    "✅ *Project creation cancelled.*\n\n"
                    "No changes were made.",
                    parse_mode="Markdown"
                )

            elif action == CallbackAction.CONFLICT_IMPROVE.value:
                # Route to change mode for existing project
                user_state.pending_conflicts.pop(user_id, None)
                await query.edit_message_text(
                    f"🔄 *Switching to CHANGE_MODE*\n\n"
                    f"Your request will improve the existing project:\n"
                    f"`{existing_project}`\n\n"
                    f"Use /change_request {existing_project} to add modifications.",
                    parse_mode="Markdown"
                )

            elif action == CallbackAction.CONFLICT_ADD_MODULE.value:
                # Add new module to existing project
                user_state.pending_conflicts.pop(user_id, None)
                await query.edit_message_text(
                    f"📦 *Adding Module*\n\n"
                    f"A new module will be added to:\n"
                    f"`{existing_project}`\n\n"
                    f"The system will analyze your requirements and add appropriate modules.\n"
                    f"Use /project_status {existing_project} to check progress.",
                    parse_mode="Markdown"
                )

                # Trigger module addition via registry
                try:
                    from controller.project_registry import get_registry
                    registry = get_registry()
                    registry.add_change_record(
                        project_name=existing_project,
                        change_type="module_addition",
                        description=description,
                        changed_by=str(user_id),
                        details={"from_conflict": conflict_id}
                    )
                except Exception as e:
                    logger.error(f"Error recording module addition: {e}")

            elif action == CallbackAction.CONFLICT_NEW_VERSION.value:
                # Create new version of the project
                user_state.pending_conflicts.pop(user_id, None)

                try:
                    from controller.project_registry import get_registry
                    registry = get_registry()

                    existing = registry.get_project(existing_project)
                    if existing:
                        success, message, new_project = registry.create_project_version(
                            parent_project=existing,
                            description=description,
                            created_by=str(user_id),
                            requirements_raw=description,
                        )

                        if success:
                            await query.edit_message_text(
                                f"🆕 *New Version Created*\n\n"
                                f"*Original:* `{existing_project}`\n"
                                f"*New Version:* `{new_project.name}`\n"
                                f"*Version:* {new_project.version}\n\n"
                                f"The new version will be developed independently.\n"
                                f"Use /project_status {new_project.name} to check progress.",
                                parse_mode="Markdown"
                            )
                        else:
                            await query.edit_message_text(
                                f"❌ *Error Creating Version*\n\n{message}",
                                parse_mode="Markdown"
                            )
                    else:
                        await query.edit_message_text(
                            f"❌ *Error:* Project `{existing_project}` not found.",
                            parse_mode="Markdown"
                        )
                except Exception as e:
                    logger.error(f"Error creating new version: {e}")
                    await query.edit_message_text(
                        f"❌ *Error:* {str(e)}",
                        parse_mode="Markdown"
                    )

        else:
            await query.edit_message_text(f"Unknown action: {action}")
    except Exception as e:
        logger.error(f"Error in handle_callback: {e}")
        if update.callback_query:
            await update.callback_query.edit_message_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 15.1: Lifecycle Commands
# -----------------------------------------------------------------------------

@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def lifecycle_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /lifecycle_status [lifecycle_id] - Show lifecycle state and guidance (Phase 15.1)
    """
    try:
        if not context.args:
            # List recent lifecycles
            result = await controller.get("/lifecycle?limit=10")
            if "error" in result:
                await update.message.reply_text(f"❌ Error: {result.get('error')}")
                return

            lifecycles = result.get("lifecycles", [])
            if not lifecycles:
                await update.message.reply_text("No lifecycles found.")
                return

            text = "*Recent Lifecycles:*\n\n"
            for lc in lifecycles[:10]:
                text += f"• `{lc['lifecycle_id'][:8]}...`\n"
                text += f"  Project: {lc['project_name']}\n"
                text += f"  Aspect: {lc['aspect']}\n"
                text += f"  State: *{lc['state']}*\n"
                text += f"  Mode: {lc['mode']}\n\n"

            text += "Use /lifecycle\\_status <id> for details"
            await update.message.reply_text(text, parse_mode="Markdown")
        else:
            lifecycle_id = context.args[0]
            result = await controller.get(f"/lifecycle/{lifecycle_id}")
            if "error" in result:
                await update.message.reply_text(f"❌ Error: {result.get('error')}")
                return

            lc = result.get("lifecycle", {})
            guidance = result.get("guidance", {})

            text = f"*Lifecycle Details*\n\n"
            text += f"*ID:* `{lc.get('lifecycle_id', 'N/A')}`\n"
            text += f"*Project:* {lc.get('project_name', 'N/A')}\n"
            text += f"*Aspect:* {lc.get('aspect', 'N/A')}\n"
            text += f"*Mode:* {lc.get('mode', 'N/A')}\n"
            text += f"*State:* *{lc.get('state', 'N/A')}*\n"
            text += f"*Transitions:* {lc.get('transition_count', 0)}\n\n"

            if guidance:
                text += "*Next Steps:*\n"
                text += f"• {guidance.get('next_step', 'None')}\n"
                if guidance.get('waiting_for'):
                    text += f"• Waiting for: {guidance.get('waiting_for')}\n"
                if guidance.get('available_actions'):
                    text += f"• Actions: {', '.join(guidance.get('available_actions', []))}\n"

            await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in lifecycle_status_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def lifecycle_approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /lifecycle_approve <lifecycle_id> [reason] - Approve lifecycle transition (Phase 15.1)
    """
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /lifecycle\\_approve <lifecycle\\_id> [reason]\n"
            "Example: /lifecycle\\_approve abc123 Looks good"
        )
        return

    lifecycle_id = context.args[0]
    reason = " ".join(context.args[1:]) if len(context.args) > 1 else "Approved via Telegram"
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    try:
        result = await controller.post(f"/lifecycle/{lifecycle_id}/transition", {
            "trigger": "human_approval",
            "triggered_by": user_id,
            "role": user_role.value,
            "reason": reason,
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        new_state = result.get("new_state", "unknown")
        await update.message.reply_text(
            f"✅ Lifecycle approved!\n\n"
            f"*New State:* {new_state}\n"
            f"*Reason:* {reason}",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), "lifecycle_approve", {
            "lifecycle_id": lifecycle_id,
            "new_state": new_state
        })

    except Exception as e:
        logger.error(f"Error in lifecycle_approve_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def lifecycle_reject_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /lifecycle_reject <lifecycle_id> <reason> - Reject lifecycle (Phase 15.1)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /lifecycle\\_reject <lifecycle\\_id> <reason>\n"
            "Example: /lifecycle\\_reject abc123 Needs more testing"
        )
        return

    lifecycle_id = context.args[0]
    reason = " ".join(context.args[1:])
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    if len(reason) < 10:
        await update.message.reply_text("Rejection reason must be at least 10 characters.")
        return

    try:
        result = await controller.post(f"/lifecycle/{lifecycle_id}/transition", {
            "trigger": "human_rejection",
            "triggered_by": user_id,
            "role": user_role.value,
            "reason": reason,
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        new_state = result.get("new_state", "unknown")
        await update.message.reply_text(
            f"🚫 Lifecycle rejected!\n\n"
            f"*New State:* {new_state}\n"
            f"*Reason:* {reason}",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), "lifecycle_reject", {
            "lifecycle_id": lifecycle_id,
            "new_state": new_state,
            "reason": reason
        })

    except Exception as e:
        logger.error(f"Error in lifecycle_reject_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def lifecycle_feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /lifecycle_feedback <lifecycle_id> <feedback> - Submit feedback (Phase 15.1)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /lifecycle\\_feedback <lifecycle\\_id> <feedback>\n"
            "Example: /lifecycle\\_feedback abc123 Button color needs adjustment"
        )
        return

    lifecycle_id = context.args[0]
    feedback_text = " ".join(context.args[1:])
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    if len(feedback_text) < 10:
        await update.message.reply_text("Feedback must be at least 10 characters.")
        return

    try:
        result = await controller.post(f"/lifecycle/{lifecycle_id}/transition", {
            "trigger": "telegram_feedback",
            "triggered_by": user_id,
            "role": user_role.value,
            "reason": feedback_text,
            "metadata": {"feedback_type": "improvement"}
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        new_state = result.get("new_state", "unknown")
        await update.message.reply_text(
            f"📝 Feedback submitted!\n\n"
            f"*New State:* {new_state}\n"
            f"The system will work on addressing your feedback.",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), "lifecycle_feedback", {
            "lifecycle_id": lifecycle_id,
            "new_state": new_state,
            "feedback": feedback_text
        })

    except Exception as e:
        logger.error(f"Error in lifecycle_feedback_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def lifecycle_prod_approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /lifecycle_prod_approve <lifecycle_id> - Final production approval (Phase 15.1)
    """
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /lifecycle\\_prod\\_approve <lifecycle\\_id>\n"
            "Example: /lifecycle\\_prod\\_approve abc123\n\n"
            "⚠️ This is a PRODUCTION approval - use carefully!"
        )
        return

    lifecycle_id = context.args[0]
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    try:
        # First check if lifecycle is ready for production
        check_result = await controller.get(f"/lifecycle/{lifecycle_id}")
        if "error" in check_result:
            await update.message.reply_text(f"❌ Error: {check_result.get('error')}")
            return

        lc = check_result.get("lifecycle", {})
        if lc.get("state") != "ready_for_production":
            await update.message.reply_text(
                f"❌ Lifecycle is not ready for production.\n"
                f"Current state: {lc.get('state')}"
            )
            return

        result = await controller.post(f"/lifecycle/{lifecycle_id}/transition", {
            "trigger": "human_approval",
            "triggered_by": user_id,
            "role": user_role.value,
            "reason": "Production approval via Telegram",
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        new_state = result.get("new_state", "unknown")
        await update.message.reply_text(
            f"🚀 Production deployment approved!\n\n"
            f"*New State:* {new_state}\n"
            f"*Approved by:* {user_id}",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), "lifecycle_prod_approve", {
            "lifecycle_id": lifecycle_id,
            "new_state": new_state
        })

    except Exception as e:
        logger.error(f"Error in lifecycle_prod_approve_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 15.2: Continuous Change Cycle Commands
# -----------------------------------------------------------------------------

async def _request_change(update: Update, context: ContextTypes.DEFAULT_TYPE, change_type: str, change_name: str) -> None:
    """
    Helper function to request a continuous change on a deployed lifecycle.
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            f"Usage: /{change_name} <lifecycle\\_id> <description>\n"
            f"Example: /{change_name} abc123 Add user login form\n\n"
            f"Note: Lifecycle must be in DEPLOYED state."
        )
        return

    lifecycle_id = context.args[0]
    description = " ".join(context.args[1:])
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    if len(description) < 10:
        await update.message.reply_text("Description must be at least 10 characters.")
        return

    try:
        result = await controller.post(f"/lifecycle/{lifecycle_id}/change", {
            "change_type": change_type,
            "change_summary": description,
            "requested_by": user_id,
            "role": user_role.value,
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        cycle_number = result.get("cycle_number", "N/A")
        change_type_emoji = {
            "feature": "✨",
            "bug": "🐛",
            "improvement": "📈",
            "refactor": "🔧",
            "security": "🔒",
        }
        emoji = change_type_emoji.get(change_type, "📝")

        await update.message.reply_text(
            f"{emoji} *Change Request Submitted!*\n\n"
            f"*Type:* {change_type}\n"
            f"*Cycle:* {cycle_number}\n"
            f"*Description:* {description}\n\n"
            f"The system will process your request.",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), f"change_{change_type}", {
            "lifecycle_id": lifecycle_id,
            "cycle_number": cycle_number,
            "description": description
        })

    except Exception as e:
        logger.error(f"Error in {change_name} command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def new_feature_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /new_feature <lifecycle_id> <description> - Request a new feature (Phase 15.2)
    """
    await _request_change(update, context, "feature", "new_feature")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER)
async def report_bug_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /report_bug <lifecycle_id> <description> - Report a bug (Phase 15.2)
    """
    await _request_change(update, context, "bug", "report_bug")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def improve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /improve <lifecycle_id> <description> - Request an improvement (Phase 15.2)
    """
    await _request_change(update, context, "improvement", "improve")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def refactor_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /refactor <lifecycle_id> <description> - Request a refactoring (Phase 15.2)
    """
    await _request_change(update, context, "refactor", "refactor")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def security_fix_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /security_fix <lifecycle_id> <description> - Request a security fix (Phase 15.2)
    """
    await _request_change(update, context, "security", "security_fix")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER)
async def cycle_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /cycle_history <lifecycle_id> - Show cycle history for a lifecycle (Phase 15.2)
    """
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /cycle\\_history <lifecycle\\_id>\n"
            "Example: /cycle\\_history abc123"
        )
        return

    lifecycle_id = context.args[0]

    try:
        result = await controller.get(f"/lifecycle/{lifecycle_id}/cycles")
        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        cycles = result.get("cycles", [])
        if not cycles:
            await update.message.reply_text("No cycle history found.")
            return

        text = f"*Cycle History*\n\n"
        for cycle in cycles:
            is_current = cycle.get("is_current", False)
            status = "🔄 Current" if is_current else "✅ Complete"
            text += f"*Cycle {cycle.get('cycle_number')}* {status}\n"
            if cycle.get("change_summary"):
                text += f"  Summary: {cycle.get('change_summary')[:50]}...\n"
            if cycle.get("state"):
                text += f"  State: {cycle.get('state')}\n"
            text += "\n"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in cycle_history_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER)
async def change_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /change_summary <lifecycle_id> - Show deployment summary (Phase 15.2)
    """
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /change\\_summary <lifecycle\\_id>\n"
            "Example: /change\\_summary abc123"
        )
        return

    lifecycle_id = context.args[0]

    try:
        result = await controller.get(f"/lifecycle/{lifecycle_id}/summary")
        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        summary = result

        text = f"*Deployment Summary*\n\n"
        text += f"*Project:* {summary.get('project_name', 'N/A')}\n"
        text += f"*Aspect:* {summary.get('aspect', 'N/A')}\n"
        text += f"*Mode:* {summary.get('mode', 'N/A')}\n"
        text += f"*State:* {summary.get('current_state', 'N/A')}\n"
        text += f"*Cycle:* {summary.get('current_cycle', 1)}\n"
        text += f"*Transitions:* {summary.get('total_transitions', 0)}\n"
        text += f"*Claude Jobs:* {summary.get('total_claude_jobs', 0)}\n\n"

        changes = summary.get("changes", [])
        if changes:
            text += "*Changes:*\n"
            for change in changes[-5:]:  # Show last 5 changes
                in_progress = "🔄" if change.get("in_progress") else "✅"
                text += f"{in_progress} Cycle {change.get('cycle')}: {change.get('summary', 'N/A')[:40]}...\n"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in change_summary_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 15.3: Project Ingestion Commands
# -----------------------------------------------------------------------------

@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def ingest_git_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /ingest_git <project_name> <git_url> - Ingest a project from Git (Phase 15.3)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /ingest\\_git <project\\_name> <git\\_url>\n"
            "Example: /ingest\\_git my-project https://github.com/user/repo.git\n\n"
            "This will clone and analyze the repository, then prepare it for registration."
        )
        return

    project_name = context.args[0]
    git_url = context.args[1]
    user_id = str(update.effective_user.id)

    # Basic validation
    if not git_url.startswith(("https://", "git@")):
        await update.message.reply_text(
            "Invalid Git URL. Must start with https:// or git@"
        )
        return

    try:
        await update.message.reply_text(
            f"🔍 *Starting ingestion analysis...*\n\n"
            f"*Project:* {project_name}\n"
            f"*Source:* {git_url}\n\n"
            f"This may take a few minutes...",
            parse_mode="Markdown"
        )

        # Create ingestion request
        result = await controller.post("/ingestion", {
            "project_name": project_name,
            "source_type": "git_repository",
            "source_location": git_url,
            "requested_by": user_id,
            "description": f"Ingested via Telegram by user {user_id}",
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        ingestion = result.get("ingestion", {})
        ingestion_id = ingestion.get("ingestion_id")

        # Start analysis
        analyze_result = await controller.post(f"/ingestion/{ingestion_id}/analyze")

        if "error" in analyze_result:
            await update.message.reply_text(f"❌ Analysis failed: {analyze_result.get('error')}")
            return

        report = analyze_result.get("report", {})
        aspects = report.get("aspects", {})
        risk = report.get("risk_assessment", {})
        structure = report.get("structure", {})

        # Format response
        text = f"✅ *Analysis Complete!*\n\n"
        text += f"*Ingestion ID:* `{ingestion_id[:8]}...`\n"
        text += f"*Project:* {project_name}\n\n"

        text += f"*📊 Structure:*\n"
        text += f"  Files: {structure.get('total_files', 'N/A')}\n"
        text += f"  Size: {structure.get('total_size_bytes', 0) / 1024 / 1024:.1f} MB\n"
        text += f"  Tests: {'✅' if structure.get('has_tests') else '❌'}\n"
        text += f"  CI/CD: {'✅' if structure.get('has_ci') else '❌'}\n\n"

        text += f"*🎯 Detected Aspects:*\n"
        for aspect in aspects.get("detected_aspects", [])[:5]:
            text += f"  • {aspect}\n"

        text += f"\n*🔒 Risk Level:* {risk.get('risk_level', 'unknown').upper()}\n"
        if risk.get("total_issues", 0) > 0:
            text += f"  Issues found: {risk.get('total_issues')}\n"

        ready = report.get("ready_for_registration", False)
        if ready:
            text += f"\n✅ *Ready for registration!*\n"
            text += f"Use `/approve_ingestion {ingestion_id[:8]}` to approve."
        else:
            text += f"\n⚠️ *Blocking issues found:*\n"
            for issue in report.get("blocking_issues", []):
                text += f"  • {issue}\n"

        await update.message.reply_text(text, parse_mode="Markdown")

        safety.log_action(int(user_id), "ingest_git", {
            "project_name": project_name,
            "ingestion_id": ingestion_id,
            "git_url": git_url,
            "ready_for_registration": ready,
        })

    except Exception as e:
        logger.error(f"Error in ingest_git_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER)
async def ingest_local_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /ingest_local <project_name> <path> - Ingest a project from local path (Phase 15.3)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /ingest\\_local <project\\_name> <path>\n"
            "Example: /ingest\\_local my-project /home/user/my-project\n\n"
            "This will analyze the local directory and prepare it for registration."
        )
        return

    project_name = context.args[0]
    local_path = " ".join(context.args[1:])  # Path might have spaces
    user_id = str(update.effective_user.id)

    try:
        await update.message.reply_text(
            f"🔍 *Starting ingestion analysis...*\n\n"
            f"*Project:* {project_name}\n"
            f"*Source:* {local_path}\n\n"
            f"This may take a moment...",
            parse_mode="Markdown"
        )

        # Create ingestion request
        result = await controller.post("/ingestion", {
            "project_name": project_name,
            "source_type": "local_path",
            "source_location": local_path,
            "requested_by": user_id,
            "description": f"Ingested via Telegram by user {user_id}",
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        ingestion = result.get("ingestion", {})
        ingestion_id = ingestion.get("ingestion_id")

        # Start analysis
        analyze_result = await controller.post(f"/ingestion/{ingestion_id}/analyze")

        if "error" in analyze_result:
            await update.message.reply_text(f"❌ Analysis failed: {analyze_result.get('error')}")
            return

        report = analyze_result.get("report", {})
        aspects = report.get("aspects", {})
        risk = report.get("risk_assessment", {})
        structure = report.get("structure", {})

        # Format response
        text = f"✅ *Analysis Complete!*\n\n"
        text += f"*Ingestion ID:* `{ingestion_id[:8]}...`\n"
        text += f"*Project:* {project_name}\n\n"

        text += f"*📊 Structure:*\n"
        text += f"  Files: {structure.get('total_files', 'N/A')}\n"
        text += f"  Size: {structure.get('total_size_bytes', 0) / 1024 / 1024:.1f} MB\n\n"

        text += f"*🎯 Detected Aspects:*\n"
        for aspect in aspects.get("detected_aspects", [])[:5]:
            text += f"  • {aspect}\n"

        text += f"\n*🔒 Risk Level:* {risk.get('risk_level', 'unknown').upper()}\n"

        ready = report.get("ready_for_registration", False)
        if ready:
            text += f"\n✅ *Ready for registration!*\n"
            text += f"Use `/approve_ingestion {ingestion_id[:8]}` to approve."
        else:
            text += f"\n⚠️ *Review required before registration.*"

        await update.message.reply_text(text, parse_mode="Markdown")

        safety.log_action(int(user_id), "ingest_local", {
            "project_name": project_name,
            "ingestion_id": ingestion_id,
            "local_path": local_path,
            "ready_for_registration": ready,
        })

    except Exception as e:
        logger.error(f"Error in ingest_local_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def approve_ingestion_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /approve_ingestion <ingestion_id> - Approve an ingestion (Phase 15.3)
    """
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /approve\\_ingestion <ingestion\\_id>\n"
            "Example: /approve\\_ingestion abc12345"
        )
        return

    ingestion_id = context.args[0]
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    try:
        result = await controller.post(f"/ingestion/{ingestion_id}/approve", {
            "approved_by": user_id,
            "role": user_role.value,
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        await update.message.reply_text(
            f"✅ *Ingestion Approved!*\n\n"
            f"*Ingestion ID:* `{ingestion_id[:8]}...`\n\n"
            f"Now use `/register_ingestion {ingestion_id[:8]}` to complete registration.",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), "approve_ingestion", {
            "ingestion_id": ingestion_id,
        })

    except Exception as e:
        logger.error(f"Error in approve_ingestion_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def reject_ingestion_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /reject_ingestion <ingestion_id> <reason> - Reject an ingestion (Phase 15.3)
    """
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /reject\\_ingestion <ingestion\\_id> <reason>\n"
            "Example: /reject\\_ingestion abc12345 Contains sensitive data"
        )
        return

    ingestion_id = context.args[0]
    reason = " ".join(context.args[1:])
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    try:
        result = await controller.post(f"/ingestion/{ingestion_id}/reject", {
            "rejected_by": user_id,
            "reason": reason,
            "role": user_role.value,
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        await update.message.reply_text(
            f"❌ *Ingestion Rejected*\n\n"
            f"*Ingestion ID:* `{ingestion_id[:8]}...`\n"
            f"*Reason:* {reason}",
            parse_mode="Markdown"
        )

        safety.log_action(int(user_id), "reject_ingestion", {
            "ingestion_id": ingestion_id,
            "reason": reason,
        })

    except Exception as e:
        logger.error(f"Error in reject_ingestion_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN)
async def register_ingestion_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /register_ingestion <ingestion_id> - Register an approved ingestion (Phase 15.3)
    """
    if len(context.args) < 1:
        await update.message.reply_text(
            "Usage: /register\\_ingestion <ingestion\\_id>\n"
            "Example: /register\\_ingestion abc12345\n\n"
            "Note: Ingestion must be approved first."
        )
        return

    ingestion_id = context.args[0]
    user_id = str(update.effective_user.id)
    user_role = get_user_role(update.effective_user.id)

    try:
        await update.message.reply_text(
            f"🔄 *Registering project...*\n\n"
            f"This will create lifecycle instances and generate governance documents.",
            parse_mode="Markdown"
        )

        result = await controller.post(f"/ingestion/{ingestion_id}/register", {
            "registered_by": user_id,
            "role": user_role.value,
        })

        if "error" in result:
            await update.message.reply_text(f"❌ Error: {result.get('error')}")
            return

        lifecycle_ids = result.get("lifecycle_ids", [])

        text = f"✅ *Project Registered Successfully!*\n\n"
        text += f"*Ingestion ID:* `{ingestion_id[:8]}...`\n"
        text += f"*Lifecycles Created:* {len(lifecycle_ids)}\n\n"

        if lifecycle_ids:
            text += "*Lifecycle IDs:*\n"
            for lid in lifecycle_ids[:5]:
                text += f"  • `{lid[:8]}...`\n"

        text += f"\n🎉 The project is now registered and in DEPLOYED state.\n"
        text += f"You can use change commands like /new\\_feature, /report\\_bug, etc."

        await update.message.reply_text(text, parse_mode="Markdown")

        safety.log_action(int(user_id), "register_ingestion", {
            "ingestion_id": ingestion_id,
            "lifecycle_ids": lifecycle_ids,
        })

    except Exception as e:
        logger.error(f"Error in register_ingestion_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.DEVELOPER, UserRole.TESTER)
async def ingestion_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /ingestion_status [ingestion_id] - Check ingestion status (Phase 15.3)
    """
    try:
        if context.args:
            # Get specific ingestion
            ingestion_id = context.args[0]
            result = await controller.get(f"/ingestion/{ingestion_id}")

            if "error" in result:
                await update.message.reply_text(f"❌ Error: {result.get('error')}")
                return

            text = f"*Ingestion Status*\n\n"
            text += f"*ID:* `{result.get('ingestion_id', 'N/A')[:8]}...`\n"
            text += f"*Project:* {result.get('project_name', 'N/A')}\n"
            text += f"*Status:* {result.get('status', 'N/A').upper()}\n"
            text += f"*Source:* {result.get('source_type', 'N/A')}\n"
            text += f"*Requested:* {result.get('requested_at', 'N/A')[:10]}\n"

            if result.get("report"):
                report = result.get("report", {})
                text += f"\n*Analysis:*\n"
                text += f"  Risk: {report.get('risk_assessment', {}).get('risk_level', 'N/A')}\n"
                text += f"  Files: {report.get('structure', {}).get('total_files', 'N/A')}\n"
                text += f"  Ready: {'✅' if report.get('ready_for_registration') else '❌'}\n"

            if result.get("lifecycle_ids"):
                text += f"\n*Lifecycles:* {len(result.get('lifecycle_ids', []))}\n"

        else:
            # List recent ingestions
            result = await controller.get("/ingestion?limit=10")

            if "error" in result:
                await update.message.reply_text(f"❌ Error: {result.get('error')}")
                return

            requests = result.get("requests", [])
            if not requests:
                await update.message.reply_text("No ingestion requests found.")
                return

            text = f"*Recent Ingestions*\n\n"
            for req in requests[:10]:
                status_emoji = {
                    "pending": "⏳",
                    "analyzing": "🔍",
                    "awaiting_approval": "⏸",
                    "approved": "✅",
                    "rejected": "❌",
                    "registered": "🎉",
                    "failed": "💥",
                }.get(req.get("status", ""), "❓")

                text += f"{status_emoji} `{req.get('ingestion_id', 'N/A')[:8]}` "
                text += f"*{req.get('project_name', 'N/A')}*\n"
                text += f"   Status: {req.get('status', 'N/A')}\n"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in ingestion_status_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 17A: Runtime Intelligence Commands (READ-ONLY)
# -----------------------------------------------------------------------------
async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /signals [project] - View runtime signals summary (Phase 17A)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Signal summary with severity breakdown and confidence indicators.

    Rules:
    - Summarized output only
    - Never raw dumps
    - Always include confidence indicator
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_signals", {})

        # Optional project filter
        project_id = context.args[0] if context.args else None

        # Get summary from controller
        summary = await controller.get_runtime_summary(since_hours=24)

        if "error" in summary:
            await update.message.reply_text(f"❌ Error: {summary.get('error')}")
            return

        # If project specified, also get project-specific signals
        if project_id:
            signals_data = await controller.get_runtime_signals(
                project_id=project_id,
                since_hours=24,
                limit=10
            )

            if "error" not in signals_data:
                # Modify summary to show project-filtered view
                summary["project_filter"] = project_id
                summary["filtered_signals"] = signals_data.get("signals", [])

        text = format_signals_summary(summary)

        # Add project filter note if applicable
        if project_id:
            text = text.replace(
                "*Runtime Intelligence Summary*",
                f"*Runtime Intelligence Summary*\n_Project: {project_id}_"
            )

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in signals_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def signals_recent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /signals_recent [hours] [limit] - View recent runtime signals (Phase 17A)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Recent signals list with summarized output.

    Usage:
    /signals_recent - Last 24 hours, 10 signals
    /signals_recent 6 - Last 6 hours, 10 signals
    /signals_recent 12 20 - Last 12 hours, 20 signals

    Rules:
    - Summarized output only
    - Never raw dumps
    - Always include confidence indicator
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_signals_recent", {})

        # Parse arguments
        since_hours = 24
        limit = 10

        if context.args:
            try:
                since_hours = int(context.args[0])
                since_hours = min(max(since_hours, 1), 168)  # 1 hour to 1 week
            except ValueError:
                pass

            if len(context.args) > 1:
                try:
                    limit = int(context.args[1])
                    limit = min(max(limit, 1), 50)  # 1 to 50
                except ValueError:
                    pass

        # Get signals from controller
        signals_data = await controller.get_runtime_signals(
            since_hours=since_hours,
            limit=limit
        )

        if "error" in signals_data:
            await update.message.reply_text(f"❌ Error: {signals_data.get('error')}")
            return

        text = format_signals_list(signals_data, limit=limit)

        # Add period note
        text = text.replace(
            "*Recent Runtime Signals*",
            f"*Recent Runtime Signals* (last {since_hours}h)"
        )

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in signals_recent_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def runtime_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /runtime_status - View runtime intelligence collection status (Phase 17A)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Collection status, polling state, storage stats.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_runtime_status", {})

        # Get status from controller
        status = await controller.get_runtime_status()

        if "error" in status:
            await update.message.reply_text(f"❌ Error: {status.get('error')}")
            return

        lines = []
        lines.append("📡 *Runtime Intelligence Status*")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")

        # Collection status
        poll_running = status.get("poll_running", False)
        poll_emoji = "🟢" if poll_running else "⚪"
        lines.append(f"{poll_emoji} *Polling:* {'Active' if poll_running else 'Inactive'}")
        lines.append(f"*Poll Interval:* {status.get('poll_interval_seconds', 'N/A')}s")
        lines.append("")

        # Storage stats
        storage = status.get("storage", {})
        lines.append("*📁 Storage*")
        lines.append(f"  File: {storage.get('file_path', 'N/A')}")
        lines.append(f"  Size: {storage.get('file_size_bytes', 0) / 1024:.1f} KB")
        lines.append(f"  Signals: {storage.get('total_signals', 0)}")
        lines.append("")

        # In-memory buffer
        lines.append("*🧠 Buffer*")
        lines.append(f"  Signals: {status.get('buffer_size', 0)}")
        lines.append(f"  Max: {status.get('buffer_max_size', 'N/A')}")
        lines.append("")

        # Last collection
        last_poll = status.get("last_poll_timestamp", "")
        if last_poll:
            lines.append(f"*Last Poll:* {last_poll[:19]}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in runtime_status_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 17B: Incident Classification Commands (READ-ONLY)
# -----------------------------------------------------------------------------
async def incidents_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /incidents [project] - View incidents summary (Phase 17B)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Incident summary with severity/type breakdown and confidence indicators.

    Rules:
    - Summarized output only
    - Never raw dumps
    - Always include confidence indicator
    - UNKNOWN incidents clearly labeled
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_incidents", {})

        # Optional project filter
        project_id = context.args[0] if context.args else None

        # Get summary from controller
        summary = await controller.get_incidents_summary(since_hours=24)

        if "error" in summary:
            await update.message.reply_text(f"❌ Error: {summary.get('error')}")
            return

        text = format_incidents_summary(summary)

        # Add project filter note if applicable
        if project_id:
            text = text.replace(
                "*Incident Classification Summary*",
                f"*Incident Classification Summary*\n_Project: {project_id}_"
            )

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in incidents_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def incidents_recent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /incidents_recent [hours] [limit] - View recent incidents (Phase 17B)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Recent incidents list with summarized output.

    Usage:
    /incidents_recent - Last 24 hours, 10 incidents
    /incidents_recent 6 - Last 6 hours, 10 incidents
    /incidents_recent 12 20 - Last 12 hours, 20 incidents

    Rules:
    - Summarized output only
    - Never raw dumps
    - Always include confidence indicator
    - UNKNOWN incidents clearly labeled
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_incidents_recent", {})

        # Parse arguments
        since_hours = 24
        limit = 10

        if context.args:
            try:
                since_hours = int(context.args[0])
                since_hours = min(max(since_hours, 1), 168)  # 1 hour to 1 week
            except ValueError:
                pass

            if len(context.args) > 1:
                try:
                    limit = int(context.args[1])
                    limit = min(max(limit, 1), 50)  # 1 to 50
                except ValueError:
                    pass

        # Get incidents from controller
        incidents_data = await controller.get_incidents_recent(
            hours=since_hours,
            limit=limit
        )

        if "error" in incidents_data:
            await update.message.reply_text(f"❌ Error: {incidents_data.get('error')}")
            return

        text = format_incidents_list(incidents_data, limit=limit)

        # Add period note
        text = text.replace(
            "*Recent Incidents*",
            f"*Recent Incidents* (last {since_hours}h)"
        )

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in incidents_recent_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def incidents_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /incidents_summary - View detailed incident classification statistics (Phase 17B)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Detailed breakdown by severity, type, scope, and state.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_incidents_summary", {})

        # Get summary from controller
        summary = await controller.get_incidents_summary(since_hours=24)

        if "error" in summary:
            await update.message.reply_text(f"❌ Error: {summary.get('error')}")
            return

        lines = []
        lines.append("🚨 *Incident Classification Statistics*")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        lines.append("")

        # Totals
        total = summary.get("total_incidents", 0)
        open_count = summary.get("open_count", 0)
        unknown_count = summary.get("unknown_count", 0)

        lines.append("*📊 Overview*")
        lines.append(f"  Total: {total}")
        lines.append(f"  Open: {open_count}")
        lines.append(f"  Unknown: {unknown_count}")
        lines.append("")

        # Severity breakdown
        by_severity = summary.get("by_severity", {})
        lines.append("*🔴 By Severity*")
        for sev in ["critical", "high", "medium", "low", "info", "unknown"]:
            count = by_severity.get(sev, 0)
            emoji = get_incident_severity_emoji(sev)
            lines.append(f"  {emoji} {sev}: {count}")
        lines.append("")

        # Type breakdown
        by_type = summary.get("by_type", {})
        if by_type:
            lines.append("*📋 By Type*")
            for inc_type in ["performance", "reliability", "security", "governance", "resource", "configuration", "unknown"]:
                count = by_type.get(inc_type, 0)
                emoji = get_incident_type_emoji(inc_type)
                if count > 0 or inc_type == "unknown":
                    lines.append(f"  {emoji} {inc_type}: {count}")
            lines.append("")

        # Scope breakdown
        by_scope = summary.get("by_scope", {})
        if by_scope:
            lines.append("*🌐 By Scope*")
            for scope in ["system", "project", "project_aspect", "job", "unknown"]:
                count = by_scope.get(scope, 0)
                emoji = get_incident_scope_emoji(scope)
                if count > 0 or scope == "unknown":
                    lines.append(f"  {emoji} {scope}: {count}")
            lines.append("")

        # State breakdown
        by_state = summary.get("by_state", {})
        if by_state:
            lines.append("*🔄 By State*")
            open_c = by_state.get("open", 0)
            closed_c = by_state.get("closed", 0)
            unknown_s = by_state.get("unknown", 0)
            lines.append(f"  🟢 open: {open_c}")
            lines.append(f"  ⚪ closed: {closed_c}")
            lines.append(f"  ❓ unknown: {unknown_s}")
            lines.append("")

        # Time window
        start = summary.get("time_window_start", "")
        end = summary.get("time_window_end", "")
        if start and end:
            start_short = start[:16].replace("T", " ")
            end_short = end[:16].replace("T", " ")
            lines.append(f"_Period: {start_short} to {end_short}_")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error in incidents_summary_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 18C: Execution Dispatcher Commands (CONTROLLED EXECUTION)
# -----------------------------------------------------------------------------

def get_execution_status_emoji(status: str) -> str:
    """Get emoji for execution status."""
    status_emojis = {
        "execution_blocked": "🚫",
        "execution_pending": "⏳",
        "execution_success": "✅",
        "execution_failed": "❌",
    }
    return status_emojis.get(status, "❓")


async def executions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /executions - View recent executions (Phase 18C)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Recent execution results with status and action type.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_executions", {})

        # Parse arguments
        args = context.args or []
        limit = 10
        status_filter = None

        for arg in args:
            if arg.isdigit():
                limit = min(int(arg), 50)
            elif arg in ["blocked", "pending", "success", "failed"]:
                status_filter = f"execution_{arg}"

        # Get executions from controller
        result = await controller.get_execution_recent(
            limit=limit,
            status=status_filter,
        )

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        executions = result.get("executions", [])

        if not executions:
            await update.message.reply_text(
                "No recent executions found.\n\n"
                "Executions appear here when actions flow through Phase 18C dispatcher."
            )
            return

        lines = []
        lines.append("Execution Results (Phase 18C)")
        lines.append("=" * 35)
        lines.append("")

        for exec_data in executions[:limit]:
            status = exec_data.get("status", "unknown")
            status_emoji = get_execution_status_emoji(status)
            exec_id = exec_data.get("execution_id", "unknown")[:20]
            intent_id = exec_data.get("intent_id", "unknown")[:15]
            timestamp = exec_data.get("timestamp", "")[:16].replace("T", " ")

            lines.append(f"{status_emoji} {status.replace('execution_', '')}")
            lines.append(f"   ID: {exec_id}")
            lines.append(f"   Intent: {intent_id}")
            lines.append(f"   Time: {timestamp}")

            if exec_data.get("block_reason"):
                lines.append(f"   Blocked: {exec_data['block_reason']}")
            if exec_data.get("failure_reason"):
                lines.append(f"   Failed: {exec_data['failure_reason']}")

            lines.append("")

        lines.append(f"Total: {len(executions)} executions")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in executions_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def execution_status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /execution_status [execution_id] - View specific execution status (Phase 18C)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Detailed execution result for a specific execution ID.
    """
    try:
        user_id = update.effective_user.id
        args = context.args or []

        if not args:
            await update.message.reply_text(
                "Usage: /execution_status <execution_id>\n\n"
                "Example: /execution_status exec-2024-01-21..."
            )
            return

        execution_id = args[0]
        safety.log_action(user_id, "view_execution_status", {"execution_id": execution_id})

        # Get execution from controller
        result = await controller.get_execution(execution_id)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        exec_result = result.get("result")
        if not exec_result:
            await update.message.reply_text(f"Execution not found: {execution_id}")
            return

        status = exec_result.get("status", "unknown")
        status_emoji = get_execution_status_emoji(status)

        lines = []
        lines.append(f"{status_emoji} Execution Detail")
        lines.append("=" * 35)
        lines.append("")
        lines.append(f"ID: {exec_result.get('execution_id', 'unknown')}")
        lines.append(f"Intent: {exec_result.get('intent_id', 'unknown')}")
        lines.append(f"Status: {status}")
        lines.append(f"Timestamp: {exec_result.get('timestamp', 'unknown')[:19].replace('T', ' ')}")
        lines.append(f"Version: {exec_result.get('dispatcher_version', 'unknown')}")
        lines.append("")

        if exec_result.get("block_reason"):
            lines.append(f"Block Reason: {exec_result['block_reason']}")
        if exec_result.get("failure_reason"):
            lines.append(f"Failure Reason: {exec_result['failure_reason']}")
        if exec_result.get("gate_decision_allowed") is not None:
            gate = "Allowed" if exec_result["gate_decision_allowed"] else "Denied"
            lines.append(f"Gate Decision: {gate}")
        if exec_result.get("rollback_performed"):
            lines.append("Rollback: Yes")
        if exec_result.get("execution_output"):
            output = exec_result["execution_output"][:200]
            lines.append(f"Output: {output}...")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in execution_status_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def execution_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /execution_summary - View execution statistics (Phase 18C)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Summary counts by status, action type, and block reason.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_execution_summary", {})

        # Parse arguments for hours
        args = context.args or []
        since_hours = 24
        for arg in args:
            if arg.isdigit():
                since_hours = int(arg)

        # Get summary from controller
        result = await controller.get_execution_summary(since_hours=since_hours)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        lines = []
        lines.append("Execution Summary (Phase 18C)")
        lines.append("=" * 35)
        lines.append("")

        total = result.get("total_executions", 0)
        blocked = result.get("blocked_count", 0)
        pending = result.get("pending_count", 0)
        success = result.get("success_count", 0)
        failed = result.get("failed_count", 0)

        lines.append("Overview")
        lines.append(f"  Total: {total}")
        lines.append(f"  {get_execution_status_emoji('execution_blocked')} Blocked: {blocked}")
        lines.append(f"  {get_execution_status_emoji('execution_pending')} Pending: {pending}")
        lines.append(f"  {get_execution_status_emoji('execution_success')} Success: {success}")
        lines.append(f"  {get_execution_status_emoji('execution_failed')} Failed: {failed}")
        lines.append("")

        # Success rate
        if total > 0:
            completed = success + failed
            if completed > 0:
                rate = (success / completed) * 100
                lines.append(f"Success Rate: {rate:.1f}%")
                lines.append("")

        # By action type
        by_action = result.get("by_action_type", {})
        if by_action:
            lines.append("By Action Type")
            for action, count in sorted(by_action.items()):
                lines.append(f"  {action}: {count}")
            lines.append("")

        # By block reason
        by_block = result.get("by_block_reason", {})
        if by_block:
            lines.append("Block Reasons")
            for reason, count in sorted(by_block.items()):
                lines.append(f"  {reason}: {count}")
            lines.append("")

        lines.append(f"Period: last {since_hours} hours")
        lines.append(f"Generated: {result.get('generated_at', '')[:19].replace('T', ' ')}")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in execution_summary_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 18D: Post-Execution Verification Commands (OBSERVATION ONLY)
# -----------------------------------------------------------------------------
def get_verification_status_emoji(status: str) -> str:
    """Get emoji for verification status."""
    status_emojis = {
        "passed": "✅",
        "failed": "❌",
        "unknown": "❓",
    }
    return status_emojis.get(status, "❔")


def get_violation_severity_emoji(severity: str) -> str:
    """Get emoji for violation severity."""
    severity_emojis = {
        "info": "ℹ️",
        "low": "🔵",
        "medium": "🟡",
        "high": "🔴",
    }
    return severity_emojis.get(severity, "❔")


async def execution_verify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /execution_verify [execution_id] - View verification result (Phase 18D)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Verification result for a specific execution.
    OBSERVATION ONLY: Never suggests fixes, never escalates.
    """
    try:
        user_id = update.effective_user.id
        args = context.args or []

        if not args:
            await update.message.reply_text(
                "Usage: /execution_verify <execution_id>\n\n"
                "Example: /execution_verify exec-2024-01-21...\n\n"
                "Shows verification result for an execution."
            )
            return

        execution_id = args[0]
        safety.log_action(user_id, "view_execution_verification", {"execution_id": execution_id})

        # Get verification from controller
        result = await controller.get_execution_verification(execution_id)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        verification = result.get("verification")
        if not verification:
            await update.message.reply_text(
                f"No verification found for execution: {execution_id}\n\n"
                "Verification runs after execution completes."
            )
            return

        status = verification.get("verification_status", "unknown")
        status_emoji = get_verification_status_emoji(status)

        lines = []
        lines.append(f"{status_emoji} Execution Verification (Phase 18D)")
        lines.append("=" * 40)
        lines.append("")
        lines.append(f"Verification ID: {verification.get('verification_id', 'unknown')[:25]}")
        lines.append(f"Execution ID: {verification.get('execution_id', 'unknown')[:25]}")
        lines.append(f"Status: {status.upper()}")
        lines.append(f"Checked: {verification.get('checked_at', 'unknown')[:19].replace('T', ' ')}")
        lines.append("")

        # Violation summary
        violation_count = verification.get("violation_count", 0)
        high_count = verification.get("high_severity_count", 0)

        if violation_count > 0:
            lines.append(f"Violations: {violation_count}")
            if high_count > 0:
                lines.append(f"High Severity: {high_count}")
            lines.append("")

            # List violations
            violations = verification.get("violations", [])
            if violations:
                lines.append("Violation Details:")
                for i, v in enumerate(violations[:5], 1):
                    v_type = v.get("violation_type", "unknown")
                    severity = v.get("severity", "unknown")
                    sev_emoji = get_violation_severity_emoji(severity)
                    lines.append(f"  {i}. {sev_emoji} {v_type} ({severity})")
                    if v.get("field"):
                        lines.append(f"      Field: {v['field']}")
                if len(violations) > 5:
                    lines.append(f"  ... and {len(violations) - 5} more")
                lines.append("")

        if verification.get("unknown_reason"):
            lines.append(f"Unknown Reason: {verification['unknown_reason']}")
            lines.append("")

        lines.append(f"Engine Version: {verification.get('engine_version', 'unknown')}")
        lines.append("")
        lines.append("(Observation only - no fixes suggested)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in execution_verify_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def execution_violations_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /execution_violations [execution_id] - View violations for execution (Phase 18D)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: All invariant violations detected for an execution.
    OBSERVATION ONLY: Never suggests fixes, never escalates.
    """
    try:
        user_id = update.effective_user.id
        args = context.args or []

        if not args:
            await update.message.reply_text(
                "Usage: /execution_violations <execution_id>\n\n"
                "Example: /execution_violations exec-2024-01-21...\n\n"
                "Shows all violations detected for an execution."
            )
            return

        execution_id = args[0]
        safety.log_action(user_id, "view_execution_violations", {"execution_id": execution_id})

        # Get violations from controller
        result = await controller.get_execution_violations(execution_id)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        violations = result.get("violations", [])

        if not violations:
            await update.message.reply_text(
                f"No violations found for execution: {execution_id}\n\n"
                "Either verification hasn't run or execution passed all checks."
            )
            return

        lines = []
        lines.append("Execution Violations (Phase 18D)")
        lines.append("=" * 40)
        lines.append("")
        lines.append(f"Execution: {execution_id[:30]}")
        lines.append(f"Total Violations: {len(violations)}")
        lines.append("")

        # Group by severity
        by_severity = {}
        for v in violations:
            sev = v.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1

        lines.append("By Severity:")
        for sev in ["high", "medium", "low", "info"]:
            if sev in by_severity:
                emoji = get_violation_severity_emoji(sev)
                lines.append(f"  {emoji} {sev.upper()}: {by_severity[sev]}")
        lines.append("")

        # List violations
        lines.append("Violations:")
        for i, v in enumerate(violations[:10], 1):
            v_type = v.get("violation_type", "unknown")
            severity = v.get("severity", "unknown")
            sev_emoji = get_violation_severity_emoji(severity)
            lines.append(f"")
            lines.append(f"{i}. {sev_emoji} {v_type}")
            lines.append(f"   Severity: {severity}")
            if v.get("field"):
                lines.append(f"   Field: {v['field']}")
            if v.get("expected"):
                lines.append(f"   Expected: {str(v['expected'])[:50]}")
            if v.get("actual"):
                lines.append(f"   Actual: {str(v['actual'])[:50]}")
            if v.get("message"):
                lines.append(f"   Message: {v['message'][:80]}")

        if len(violations) > 10:
            lines.append(f"")
            lines.append(f"... and {len(violations) - 10} more violations")

        lines.append("")
        lines.append("(Observation only - no fixes suggested)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in execution_violations_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def verification_recent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /verification_recent [limit] [status] - View recent verifications (Phase 18D)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Recent verification results.
    OBSERVATION ONLY: Never suggests fixes, never escalates.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_verification_recent", {})

        # Parse arguments
        args = context.args or []
        limit = 10
        status_filter = None

        for arg in args:
            if arg.isdigit():
                limit = min(int(arg), 50)
            elif arg in ["passed", "failed", "unknown"]:
                status_filter = arg

        # Get recent verifications from controller
        result = await controller.get_verification_recent(
            limit=limit,
            status=status_filter,
        )

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        verifications = result.get("verifications", [])

        if not verifications:
            await update.message.reply_text(
                "No recent verifications found.\n\n"
                "Verifications appear after execution completes."
            )
            return

        lines = []
        lines.append("Recent Verifications (Phase 18D)")
        lines.append("=" * 40)
        lines.append("")

        for v in verifications[:limit]:
            status = v.get("verification_status", "unknown")
            status_emoji = get_verification_status_emoji(status)
            v_id = v.get("verification_id", "unknown")[:20]
            exec_id = v.get("execution_id", "unknown")[:15]
            violation_count = v.get("violation_count", 0)
            checked = v.get("checked_at", "")[:16].replace("T", " ")

            lines.append(f"{status_emoji} {status.upper()}")
            lines.append(f"   ID: {v_id}")
            lines.append(f"   Exec: {exec_id}")
            if violation_count > 0:
                lines.append(f"   Violations: {violation_count}")
            lines.append(f"   Time: {checked}")
            lines.append("")

        lines.append(f"Showing {len(verifications)} verifications")
        lines.append("")
        lines.append("(Observation only - no fixes suggested)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in verification_recent_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def verification_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /verification_summary [hours] - View verification statistics (Phase 18D)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Summary counts by status, violation type, and severity.
    OBSERVATION ONLY: Never suggests fixes, never escalates.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_verification_summary", {})

        # Parse arguments for hours
        args = context.args or []
        since_hours = 24
        for arg in args:
            if arg.isdigit():
                since_hours = int(arg)

        # Get summary from controller
        result = await controller.get_verification_summary(since_hours=since_hours)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        lines = []
        lines.append("Verification Summary (Phase 18D)")
        lines.append("=" * 40)
        lines.append("")

        total = result.get("total_verifications", 0)
        passed = result.get("passed_count", 0)
        failed = result.get("failed_count", 0)
        unknown = result.get("unknown_count", 0)
        pass_rate = result.get("pass_rate", 0.0)

        lines.append("Overview")
        lines.append(f"  Total: {total}")
        lines.append(f"  {get_verification_status_emoji('passed')} Passed: {passed}")
        lines.append(f"  {get_verification_status_emoji('failed')} Failed: {failed}")
        lines.append(f"  {get_verification_status_emoji('unknown')} Unknown: {unknown}")
        lines.append(f"  Pass Rate: {pass_rate * 100:.1f}%")
        lines.append("")

        # Violation totals
        total_violations = result.get("total_violations", 0)
        high_severity = result.get("high_severity_violations", 0)

        if total_violations > 0:
            lines.append("Violations")
            lines.append(f"  Total: {total_violations}")
            if high_severity > 0:
                lines.append(f"  High Severity: {high_severity}")
            lines.append("")

        # By violation type
        by_type = result.get("by_violation_type", {})
        if by_type:
            lines.append("By Violation Type:")
            for v_type, count in sorted(by_type.items()):
                lines.append(f"  {v_type}: {count}")
            lines.append("")

        # By severity
        by_severity = result.get("by_severity", {})
        if by_severity:
            lines.append("By Severity:")
            for sev in ["high", "medium", "low", "info"]:
                if sev in by_severity:
                    emoji = get_violation_severity_emoji(sev)
                    lines.append(f"  {emoji} {sev}: {by_severity[sev]}")
            lines.append("")

        lines.append(f"Period: last {since_hours} hours")
        lines.append(f"Generated: {result.get('generated_at', '')[:19].replace('T', ' ')}")
        lines.append("")
        lines.append("(Observation only - no fixes suggested)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in verification_summary_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 19: Learning, Memory & System Intelligence Commands (INSIGHT ONLY)
# -----------------------------------------------------------------------------
def get_trend_direction_emoji(direction: str) -> str:
    """Get emoji for trend direction."""
    direction_emojis = {
        "increasing": "📈",
        "decreasing": "📉",
        "stable": "➡️",
        "unknown": "❓",
    }
    return direction_emojis.get(direction, "❔")


def get_confidence_emoji(confidence: str) -> str:
    """Get emoji for confidence level."""
    confidence_emojis = {
        "high": "🟢",
        "medium": "🟡",
        "low": "🟠",
        "insufficient": "⚪",
    }
    return confidence_emojis.get(confidence, "❔")


async def learning_summary_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /learning_summary - View latest learning summary (Phase 19)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Aggregate statistics and key metrics.
    INSIGHT ONLY: Never suggests actions, never automates.
    NO BEHAVIORAL COUPLING: Does not influence any other phase.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_learning_summary", {})

        # Get summary from controller
        result = await controller.get_learning_summary()

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        summary = result.get("summary")
        if not summary:
            await update.message.reply_text(
                "No learning summary available yet.\n\n"
                "Summaries are generated after system activity is analyzed."
            )
            return

        lines = []
        lines.append("Learning Summary (Phase 19)")
        lines.append("=" * 40)
        lines.append("")
        lines.append("Activity Overview")
        lines.append(f"  Executions: {summary.get('total_executions', 0)}")
        lines.append(f"  Verifications: {summary.get('total_verifications', 0)}")
        lines.append(f"  Approvals: {summary.get('total_approvals', 0)}")
        lines.append(f"  Incidents: {summary.get('total_incidents', 0)}")
        lines.append("")

        # Rates
        exec_rate = summary.get('execution_success_rate', 0) * 100
        ver_rate = summary.get('verification_pass_rate', 0) * 100
        appr_rate = summary.get('approval_grant_rate', 0) * 100

        lines.append("Success Rates")
        lines.append(f"  Execution: {exec_rate:.1f}%")
        lines.append(f"  Verification: {ver_rate:.1f}%")
        lines.append(f"  Approval: {appr_rate:.1f}%")
        lines.append("")

        # Patterns and trends
        lines.append("Observations")
        lines.append(f"  Patterns detected: {summary.get('pattern_count', 0)}")
        lines.append(f"  Trends observed: {summary.get('trend_count', 0)}")
        lines.append("")

        # Period info
        period_start = summary.get('period_start', '')[:10]
        period_end = summary.get('period_end', '')[:10]
        lines.append(f"Period: {period_start} to {period_end}")
        lines.append(f"Generated: {summary.get('generated_at', '')[:19].replace('T', ' ')}")
        lines.append("")
        lines.append("(Insight only - no actions triggered)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in learning_summary_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def learning_patterns_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /learning_patterns [limit] [type] - View observed patterns (Phase 19)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Detected patterns in system behavior.
    INSIGHT ONLY: Never suggests actions, never automates.
    NO BEHAVIORAL COUPLING: Does not influence any other phase.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_learning_patterns", {})

        # Parse arguments
        args = context.args or []
        limit = 10
        pattern_type = None

        for arg in args:
            if arg.isdigit():
                limit = min(int(arg), 50)
            else:
                pattern_type = arg

        # Get patterns from controller
        result = await controller.get_learning_patterns(
            limit=limit,
            pattern_type=pattern_type,
        )

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        patterns = result.get("patterns", [])

        if not patterns:
            await update.message.reply_text(
                "No patterns observed yet.\n\n"
                "Patterns emerge after sufficient system activity."
            )
            return

        lines = []
        lines.append("Observed Patterns (Phase 19)")
        lines.append("=" * 40)
        lines.append("")

        for i, p in enumerate(patterns[:limit], 1):
            p_type = p.get("pattern_type", "unknown")
            confidence = p.get("confidence", "unknown")
            conf_emoji = get_confidence_emoji(confidence)
            frequency = p.get("frequency", 0)
            description = p.get("description", "No description")[:60]

            lines.append(f"{i}. {conf_emoji} {p_type}")
            lines.append(f"   Frequency: {frequency}")
            lines.append(f"   Confidence: {confidence}")
            lines.append(f"   {description}")
            lines.append("")

        lines.append(f"Showing {len(patterns)} patterns")
        lines.append("")
        lines.append("(Insight only - no actions triggered)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in learning_patterns_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def learning_trends_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /learning_trends [limit] [metric] - View observed trends (Phase 19)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Observed trends in system metrics.
    INSIGHT ONLY: Never suggests actions, never predicts.
    NO BEHAVIORAL COUPLING: Does not influence any other phase.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_learning_trends", {})

        # Parse arguments
        args = context.args or []
        limit = 10
        metric_name = None

        for arg in args:
            if arg.isdigit():
                limit = min(int(arg), 50)
            else:
                metric_name = arg

        # Get trends from controller
        result = await controller.get_learning_trends(
            limit=limit,
            metric_name=metric_name,
        )

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        trends = result.get("trends", [])

        if not trends:
            await update.message.reply_text(
                "No trends observed yet.\n\n"
                "Trends emerge after sufficient time-series data."
            )
            return

        lines = []
        lines.append("Observed Trends (Phase 19)")
        lines.append("=" * 40)
        lines.append("")

        for i, t in enumerate(trends[:limit], 1):
            metric = t.get("metric_name", "unknown")
            direction = t.get("direction", "unknown")
            dir_emoji = get_trend_direction_emoji(direction)
            change_rate = t.get("change_rate", 0)
            confidence = t.get("confidence", "unknown")
            conf_emoji = get_confidence_emoji(confidence)
            start_val = t.get("start_value", 0)
            end_val = t.get("end_value", 0)

            lines.append(f"{i}. {dir_emoji} {metric}")
            lines.append(f"   Direction: {direction}")
            lines.append(f"   Change: {change_rate:+.1f}%")
            lines.append(f"   {conf_emoji} Confidence: {confidence}")
            lines.append(f"   Values: {start_val:.2f} -> {end_val:.2f}")
            lines.append("")

        lines.append(f"Showing {len(trends)} trends")
        lines.append("")
        lines.append("(Insight only - no predictions made)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in learning_trends_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


async def learning_statistics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /learning_stats [hours] - View learning statistics (Phase 19)

    ALWAYS responds - no RBAC restriction (READ-ONLY).
    Shows: Statistics about patterns, trends, and memory.
    INSIGHT ONLY: Never suggests actions, never automates.
    NO BEHAVIORAL COUPLING: Does not influence any other phase.
    """
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_learning_statistics", {})

        # Parse arguments for hours
        args = context.args or []
        since_hours = 24
        for arg in args:
            if arg.isdigit():
                since_hours = int(arg)

        # Get statistics from controller
        result = await controller.get_learning_statistics(since_hours=since_hours)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        stats = result.get("statistics", {})

        lines = []
        lines.append("Learning Statistics (Phase 19)")
        lines.append("=" * 40)
        lines.append("")

        lines.append("Counts")
        lines.append(f"  Patterns: {stats.get('pattern_count', 0)}")
        lines.append(f"  Trends: {stats.get('trend_count', 0)}")
        lines.append(f"  Memory entries: {stats.get('memory_count', 0)}")
        lines.append(f"  Aggregates: {stats.get('aggregate_count', 0)}")
        lines.append(f"  Summaries: {stats.get('summary_count', 0)}")
        lines.append("")

        # By pattern type
        by_pattern = stats.get("by_pattern_type", {})
        if by_pattern:
            lines.append("By Pattern Type:")
            for p_type, count in sorted(by_pattern.items()):
                lines.append(f"  {p_type}: {count}")
            lines.append("")

        # By trend direction
        by_direction = stats.get("by_trend_direction", {})
        if by_direction:
            lines.append("By Trend Direction:")
            for direction, count in sorted(by_direction.items()):
                emoji = get_trend_direction_emoji(direction)
                lines.append(f"  {emoji} {direction}: {count}")
            lines.append("")

        lines.append(f"Period: last {since_hours} hours")
        lines.append(f"Generated: {stats.get('generated_at', '')[:19].replace('T', ' ')}")
        lines.append("")
        lines.append("(Insight only - no actions triggered)")

        await update.message.reply_text("\n".join(lines))

    except Exception as e:
        logger.error(f"Error in learning_statistics_command: {e}")
        await update.message.reply_text(f"Error: {str(e)}")


# -----------------------------------------------------------------------------
# Phase 16C: Document Handler (File Upload Support)
# -----------------------------------------------------------------------------
@role_required(UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER)
async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle document uploads for project creation (Phase 16C).

    Supports:
    - .md files (Markdown)
    - .txt files (Plain text)

    The file content will be used as project requirements.
    """
    try:
        document = update.message.document
        if not document:
            return

        filename = document.file_name or "unknown"
        user_id = str(update.effective_user.id)

        # Check file extension
        allowed_extensions = [".md", ".txt", ".text"]
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

        if ext not in allowed_extensions:
            await update.message.reply_text(
                f"❌ *Unsupported file type*\n\n"
                f"Received: `{ext}`\n"
                f"Allowed: {', '.join(allowed_extensions)}\n\n"
                f"Please upload a `.md` or `.txt` file with your project requirements.",
                parse_mode="Markdown"
            )
            return

        # Check file size (max 100KB)
        if document.file_size > 100 * 1024:
            await update.message.reply_text(
                f"❌ *File too large*\n\n"
                f"Size: {document.file_size // 1024}KB\n"
                f"Maximum: 100KB\n\n"
                f"Please use a smaller file or describe your project directly.",
                parse_mode="Markdown"
            )
            return

        # Download file
        await update.message.reply_text(
            f"📥 *File received*: `{filename}`\n\n"
            f"Downloading and analyzing your project requirements...",
            parse_mode="Markdown"
        )

        file = await document.get_file()
        file_content = await file.download_as_bytearray()

        safety.log_action(int(user_id), "upload_project_file", {
            "filename": filename,
            "size": len(file_content),
        })

        # Create project from file
        await create_project_from_file(
            update=update,
            filename=filename,
            file_content=bytes(file_content),
            user_id=user_id,
        )

    except Exception as e:
        logger.error(f"Error handling document: {e}", exc_info=True)
        await update.message.reply_text(
            f"❌ *Error processing file*\n\n"
            f"Error: {str(e)}\n\n"
            f"Please try again or use /new_project to describe your project.",
            parse_mode="Markdown"
        )


# -----------------------------------------------------------------------------
# Error Handler
# -----------------------------------------------------------------------------
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors - ALWAYS tries to respond."""
    logger.error(f"Update {update} caused error {context.error}")

    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(
                "An error occurred. Please try again later.\n"
                f"Error: {str(context.error)[:100]}"
            )
        elif update and update.callback_query:
            await update.callback_query.answer(f"Error: {str(context.error)[:50]}")
    except Exception as e:
        logger.error(f"Failed to send error response: {e}")


# -----------------------------------------------------------------------------
# Phase 20: Job Monitor & Notification System
# Phase 21: CI Monitor Integration
# -----------------------------------------------------------------------------
# Track jobs we've already notified about to avoid duplicates
# Format: job_id -> {"state": last_state, "notified_at": timestamp, "phase_triggered": bool}
_notified_jobs: Dict[str, Dict] = {}
_job_monitor_running = False
_ci_monitor_running = False
JOB_MONITOR_INTERVAL = 10  # seconds between checks
CI_MONITOR_INTERVAL = 60  # seconds between CI checks
NOTIFICATION_CHAT_IDS: List[int] = []  # Will be populated from ROLE_OWNERS

# Phase 21: CI Monitor state
_monitored_repos: Dict[str, str] = {}  # project_name -> repo (owner/repo)
_ci_processed_runs: set = set()  # Processed workflow run IDs
_ci_fix_attempts: Dict[str, int] = {}  # repo -> fix attempt count
CI_MAX_AUTO_FIX_ATTEMPTS = 3

# Track completed jobs to prevent re-triggering phases
_completed_job_ids: set = set()  # Jobs that have completed (persistent for session)

# Project deployment URLs cache
_project_urls_cache: Dict[str, Dict[str, str]] = {}


async def get_project_deployment_urls(project_name: str) -> Dict[str, str]:
    """
    Get deployment URLs for a project from its contract/config.
    Returns dict with keys like 'api', 'frontend', 'admin', 'docs'.
    """
    global _project_urls_cache

    # Check cache first
    if project_name in _project_urls_cache:
        return _project_urls_cache[project_name]

    urls = {}
    import re

    try:
        # Try to get project details from controller
        result = await controller.get_project(project_name)
        if result and "project" in result:
            project_data = result["project"]

            # Check for deployment_targets in original requirements or config
            if "original_requirements" in project_data:
                req_text = str(project_data.get("original_requirements", ""))
                urls = _extract_urls_from_text(req_text)

            # Also check aspects for testing_url/production_url
            aspects = project_data.get("aspects", {})
            for aspect_name, aspect_data in aspects.items():
                if isinstance(aspect_data, dict):
                    testing_url = aspect_data.get("testing_url")
                    if testing_url:
                        urls[aspect_name] = testing_url

    except Exception as e:
        logger.debug(f"Controller project lookup failed for {project_name}: {e}")

    # Fallback: Try reading project contract directly from filesystem
    if not urls:
        try:
            import yaml
            contract_path = Path(f"/home/aitesting.mybd.in/public_html/projects/{project_name}/INTERNAL_PROJECT_CONTRACT.yaml")
            if contract_path.exists():
                with open(contract_path) as f:
                    contract = yaml.safe_load(f)

                # Check original_requirements in contract
                orig_reqs = contract.get("original_requirements", [])
                if orig_reqs:
                    req_text = str(orig_reqs[0]) if isinstance(orig_reqs, list) else str(orig_reqs)
                    urls = _extract_urls_from_text(req_text)
        except Exception as e:
            logger.debug(f"Contract file lookup failed for {project_name}: {e}")

    # Cache the result
    if urls:
        _project_urls_cache[project_name] = urls

    return urls


def _extract_urls_from_text(text: str) -> Dict[str, str]:
    """
    Extract deployment URLs from requirements text.

    Supports multiple YAML formats and flexible pattern matching.
    Phase 22: Fixed to extract all 3 deployment targets (api, frontend, admin).
    """
    import re
    urls = {}

    # API patterns - multiple formats supported
    api_patterns = [
        r'api:\s*\n\s*domain:\s*([^\s\n]+)',  # api:\n  domain: xxx
        r'api:\s*([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',  # api: domain.com
        r'API[:\s]+(?:https?://)?([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',  # API: domain.com
    ]

    # Frontend patterns - multiple formats supported
    frontend_patterns = [
        r'frontend_web:\s*\n\s*domain:\s*([^\s\n]+)',  # frontend_web:\n  domain: xxx
        r'frontend:\s*\n\s*domain:\s*([^\s\n]+)',  # frontend:\n  domain: xxx
        r'frontend:\s*([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',  # frontend: domain.com
        r'Frontend[:\s]+(?:https?://)?([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',  # Frontend: domain.com
    ]

    # Admin panel patterns - multiple formats supported (FIXED: was too restrictive)
    admin_patterns = [
        r'admin_panel:\s*\n\s*domain:\s*([^\s\n]+)',  # admin_panel:\n  domain: xxx
        r'admin:\s*\n\s*domain:\s*([^\s\n]+)',  # admin:\n  domain: xxx
        r'backend[^:]*:\s*\n\s*domain:\s*([^\s\n]+)',  # backend (admin):\n  domain: xxx
        r'Admin\s*Panel[:\s]+(?:https?://)?([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',  # Admin Panel: domain
        r'admin[:\s]+(?:https?://)?([a-zA-Z0-9][a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',  # admin: domain.com
        r'testhealth\.[a-zA-Z0-9\-\.]+',  # Direct match for testhealth.* domains
    ]

    # Extract API domain
    for pattern in api_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            domain = match.group(1).strip() if match.lastindex else match.group(0).strip()
            # Remove any trailing punctuation
            domain = re.sub(r'[,;:\s]+$', '', domain)
            if domain and '.' in domain:
                urls['api'] = f"https://{domain}"
                urls['docs'] = f"https://{domain}/docs"
                break

    # Extract Frontend domain
    for pattern in frontend_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            domain = match.group(1).strip() if match.lastindex else match.group(0).strip()
            domain = re.sub(r'[,;:\s]+$', '', domain)
            if domain and '.' in domain:
                urls['frontend'] = f"https://{domain}"
                break

    # Extract Admin panel domain
    for pattern in admin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            domain = match.group(1).strip() if match.lastindex else match.group(0).strip()
            domain = re.sub(r'[,;:\s]+$', '', domain)
            if domain and '.' in domain:
                urls['admin'] = f"https://{domain}"
                break

    # Fallback: Try to extract any domains containing specific keywords
    if 'admin' not in urls:
        # Look for domains with "admin", "test", "backend" in them
        admin_domain_match = re.search(
            r'(?:https?://)?([a-zA-Z0-9\-]*(?:admin|test|backend)[a-zA-Z0-9\-]*\.[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,})',
            text, re.IGNORECASE
        )
        if admin_domain_match:
            domain = admin_domain_match.group(1).strip()
            # Don't use if it's already the API or frontend domain
            if domain not in [urls.get('api', '').replace('https://', ''),
                              urls.get('frontend', '').replace('https://', '')]:
                urls['admin'] = f"https://{domain}"

    logger.debug(f"Extracted URLs from text: {urls}")
    return urls


async def send_notification(application, message: str, parse_mode: str = "Markdown"):
    """Send notification to all owners with fallback for markdown errors."""
    for chat_id in NOTIFICATION_CHAT_IDS:
        try:
            await application.bot.send_message(
                chat_id=chat_id,
                text=message,
                parse_mode=parse_mode
            )
            logger.info(f"Notification sent to {chat_id}")
        except Exception as e:
            error_str = str(e)
            # If markdown parsing fails, retry without parse mode
            if "parse entities" in error_str.lower() or "parse_mode" in error_str.lower():
                try:
                    # Strip markdown and send as plain text
                    plain_message = message.replace("*", "").replace("`", "").replace("_", "")
                    await application.bot.send_message(
                        chat_id=chat_id,
                        text=plain_message
                    )
                    logger.info(f"Notification sent to {chat_id} (plain text fallback)")
                except Exception as e2:
                    logger.error(f"Failed to send notification to {chat_id} even in plain text: {e2}")
            else:
                logger.error(f"Failed to send notification to {chat_id}: {e}")


async def job_monitor_task(application):
    """Background task to monitor job status changes and send notifications."""
    global _notified_jobs, _job_monitor_running, _completed_job_ids
    _job_monitor_running = True
    logger.info("Job monitor started")

    while _job_monitor_running:
        try:
            # Get all jobs
            result = await controller.list_claude_jobs(limit=50)
            jobs = result.get("jobs", [])

            for job in jobs:
                job_id = job.get("job_id")
                state = job.get("state")
                project = job.get("project_name", "unknown")
                task_type = job.get("task_type", "unknown")

                if not job_id:
                    continue

                # Skip if this job is already in our completed set (from startup or previous processing)
                if job_id in _completed_job_ids:
                    continue

                # Get tracking info for this job
                job_tracking = _notified_jobs.get(job_id, {})
                last_notified_state = job_tracking.get("state")
                phase_triggered = job_tracking.get("phase_triggered", False)

                # Skip if we already notified about this exact state
                if last_notified_state == state:
                    continue

                # Notify on completion or failure
                if state == "completed":
                    # Mark as completed immediately to prevent duplicates
                    _completed_job_ids.add(job_id)
                    _notified_jobs[job_id] = {"state": state, "phase_triggered": phase_triggered}

                    result_summary = job.get("result_summary", "")[:500]
                    message = (
                        f"✅ *Job Completed*\n\n"
                        f"*Project:* {escape_markdown(project)}\n"
                        f"*Type:* {task_type}\n"
                        f"*Job ID:* `{job_id[:12]}...`\n\n"
                    )
                    if result_summary:
                        message += f"*Summary:*\n{escape_markdown(result_summary[:300])}...\n\n"

                    # Automatic phase progression - only if not already triggered
                    if not phase_triggered:
                        _notified_jobs[job_id]["phase_triggered"] = True
                        if task_type == "planning":
                            message += "🚀 *Triggering development phase...*"
                            asyncio.create_task(trigger_development_phase(application, project, job))
                        elif task_type == "development":
                            message += "🚀 *Triggering deployment phase...*"
                            asyncio.create_task(trigger_deployment_phase(application, project, job))
                        elif task_type == "deployment":
                            # Phase 22: Trigger validation instead of immediate success
                            urls = await get_project_deployment_urls(project)
                            message += "\n\n🔍 *Validating deployment...*"

                            # Trigger async validation (will send its own notifications)
                            asyncio.create_task(
                                trigger_deployment_validation(application, project, job, urls)
                            )
                        elif task_type == "rescue":
                            # Phase 22: Handle rescue job completion
                            message += "\n\n🔧 *Rescue job completed, re-validating...*"
                            asyncio.create_task(
                                handle_rescue_job_completion(application, project, job)
                            )
                    else:
                        # Phase already triggered, just show completion
                        if task_type == "deployment":
                            # Phase 22: Still validate even if phase was triggered before
                            urls = await get_project_deployment_urls(project)
                            message += "\n\n🔍 *Validating deployment...*"
                            asyncio.create_task(
                                trigger_deployment_validation(application, project, job, urls)
                            )
                        elif task_type == "rescue":
                            # Phase 22: Re-validate after rescue
                            message += "\n\n🔧 *Rescue completed, validating...*"
                            asyncio.create_task(
                                handle_rescue_job_completion(application, project, job)
                            )

                    await send_notification(application, message)

                elif state == "failed":
                    _notified_jobs[job_id] = {"state": state, "phase_triggered": False}
                    error = job.get("error_message", "Unknown error")[:300]
                    exit_code = job.get("exit_code", "N/A")
                    message = (
                        f"❌ *Job Failed*\n\n"
                        f"*Project:* {escape_markdown(project)}\n"
                        f"*Type:* {task_type}\n"
                        f"*Job ID:* `{job_id[:12]}...`\n"
                        f"*Exit Code:* {exit_code}\n\n"
                        f"*Error:*\n```\n{error}\n```\n\n"
                        f"Use /dashboard to view details."
                    )
                    await send_notification(application, message)

                elif state == "running" and last_notified_state is None:
                    # First time seeing this job - notify it started
                    _notified_jobs[job_id] = {"state": "started", "phase_triggered": False}
                    message = (
                        f"🔄 *Job Started*\n\n"
                        f"*Project:* {escape_markdown(project)}\n"
                        f"*Type:* {task_type}\n"
                        f"*Job ID:* `{job_id[:12]}...`\n\n"
                        f"View live logs: /dashboard → Live"
                    )
                    await send_notification(application, message)

            # Clean up old notified jobs (keep last 200)
            if len(_notified_jobs) > 200:
                # Remove oldest entries
                keys_to_remove = list(_notified_jobs.keys())[:-200]
                for key in keys_to_remove:
                    del _notified_jobs[key]

            # Keep completed job IDs set manageable (last 500)
            if len(_completed_job_ids) > 500:
                # Convert to list, keep last 500
                completed_list = list(_completed_job_ids)
                _completed_job_ids = set(completed_list[-500:])

        except Exception as e:
            logger.error(f"Job monitor error: {e}")

        await asyncio.sleep(JOB_MONITOR_INTERVAL)


async def trigger_development_phase(application, project_name: str, planning_job: Dict):
    """Trigger development phase after planning completes."""
    try:
        logger.info(f"Triggering development phase for {project_name}")

        # Get planning job ID to copy artifacts
        planning_job_id = planning_job.get("job_id")

        # Create development task
        task_description = f"""DEVELOPMENT PHASE - Implement Features from Planning

PROJECT: {project_name}

INSTRUCTIONS:
1. Read PLANNING_OUTPUT.yaml (copied from planning phase) in your workspace
2. Implement Phase 1 features as defined in the plan
3. Create all necessary files and code for the FastAPI backend
4. Write unit tests for implemented features
5. Update CURRENT_STATE.md with progress
6. Follow all governance documents (AI_POLICY.md, ARCHITECTURE.md)

IMPORTANT:
- The PLANNING_OUTPUT.yaml file is in your workspace - read it first
- Start with the first implementation phase from the plan
- Create working, tested code
- Do NOT deploy - just implement and test locally
- Report any blockers in logs/BLOCKERS.md

After completing this phase, the next job will handle deployment.
"""

        # Copy PLANNING_OUTPUT.yaml from planning job to development job
        result = await controller.create_claude_job(
            project_name=project_name,
            task_description=task_description,
            task_type="development",
            copy_from_job=planning_job_id,
            copy_artifacts=["PLANNING_OUTPUT.yaml", "CURRENT_STATE.md"]
        )

        if result.get("job_id"):
            message = (
                f"🚀 *Development Phase Started*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Job ID:* `{result['job_id'][:12]}...`\n\n"
                f"Claude is now implementing features from the planning output."
            )
            await send_notification(application, message)
        else:
            error = extract_api_error(result)
            message = (
                f"⚠️ *Failed to Start Development Phase*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Error:* {escape_markdown(error)}"
            )
            await send_notification(application, message)

    except Exception as e:
        logger.error(f"Failed to trigger development phase: {e}")
        message = f"⚠️ *Error triggering development phase for {project_name}:* {str(e)}"
        await send_notification(application, message)


async def trigger_deployment_phase(application, project_name: str, dev_job: Dict):
    """Trigger deployment to testing after development completes."""
    try:
        logger.info(f"Triggering deployment phase for {project_name}")

        # Get development job ID to copy artifacts
        dev_job_id = dev_job.get("job_id")

        # Phase 22: Get ALL deployment URLs to pass to Claude
        urls = await get_project_deployment_urls(project_name)
        api_url = urls.get('api', 'Not configured - discover from PLANNING_OUTPUT.yaml')
        frontend_url = urls.get('frontend', 'Not configured - discover from PLANNING_OUTPUT.yaml')
        admin_url = urls.get('admin', 'Not configured - discover from PLANNING_OUTPUT.yaml')
        docs_url = urls.get('docs', '')

        # Build deployment targets section
        deployment_targets = f"""DEPLOYMENT TARGETS:
1. API Backend:
   - URL: {api_url}
   - Deploy FastAPI/backend code
   - Verify /health and /docs endpoints work

2. Frontend Web:
   - URL: {frontend_url}
   - Deploy React/frontend build
   - Verify main page loads

3. Admin Panel (if applicable):
   - URL: {admin_url}
   - Deploy admin interface
   - Verify admin login page loads"""

        task_description = f"""DEPLOYMENT PHASE - Deploy to Testing Environment

PROJECT: {project_name}

INSTRUCTIONS:
1. Read PLANNING_OUTPUT.yaml for deployment targets and configuration
2. Read DEPLOYMENT.md for deployment procedures
3. The project code has been copied to your workspace in the "{project_name}/" directory
4. Deploy ALL components to the TESTING environment
5. Verify ALL deployments are successful
6. Update CURRENT_STATE.md with deployment status

PROJECT CODE LOCATION:
- The complete project code from development is in: ./{project_name}/
- Backend: ./{project_name}/backend/
- Frontend: ./{project_name}/frontend/
- Admin Panel: ./{project_name}/admin/ (if exists)

{deployment_targets}

CRITICAL - DEPLOY ALL TARGETS:
- You MUST deploy all configured components (API, Frontend, Admin if specified)
- Do NOT skip any deployment target
- If a target URL says "Not configured", check PLANNING_OUTPUT.yaml for the actual domain

IMPORTANT:
- Deploy to TESTING only, NOT production
- Run smoke tests after deployment for EACH deployed component
- Report ALL deployment URLs in logs/DEPLOYMENT_LOG.md
- Note any issues in logs/BLOCKERS.md

After deployment, human validation will be required before production.
"""

        # Copy artifacts from development job - including the project directory
        result = await controller.create_claude_job(
            project_name=project_name,
            task_description=task_description,
            task_type="deployment",
            copy_from_job=dev_job_id,
            copy_artifacts=["PLANNING_OUTPUT.yaml", "CURRENT_STATE.md", project_name]
        )

        if result.get("job_id"):
            message = (
                f"🚀 *Deployment Phase Started*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Job ID:* `{result['job_id'][:12]}...`\n\n"
                f"Claude is deploying to testing environment."
            )
            await send_notification(application, message)
        else:
            error = extract_api_error(result)
            message = (
                f"⚠️ *Failed to Start Deployment Phase*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Error:* {escape_markdown(error)}"
            )
            await send_notification(application, message)

    except Exception as e:
        logger.error(f"Failed to trigger deployment phase: {e}")


def stop_job_monitor():
    """Stop the job monitor."""
    global _job_monitor_running
    _job_monitor_running = False
    logger.info("Job monitor stopped")


# -----------------------------------------------------------------------------
# Phase 22: Deployment Validation & Rescue System
# -----------------------------------------------------------------------------

async def trigger_deployment_validation(
    application,
    project_name: str,
    deployment_job: Dict,
    urls: Dict[str, str]
) -> bool:
    """
    Validate deployment endpoints and trigger rescue if needed.

    Phase 22: Post-deployment validation with automatic rescue.

    Args:
        application: Telegram application for notifications
        project_name: Name of the project
        deployment_job: The completed deployment job dict
        urls: Dict of deployment URLs (api, frontend, admin)

    Returns:
        True if validation passed, False if rescue triggered or failed
    """
    try:
        # Import validation components
        from controller.deployment_validator import (
            DeploymentValidator,
            DeploymentFailureClassifier,
        )
        from controller.rescue_engine import get_rescue_engine
        from controller.server_detector import ServerDetector

        logger.info(f"Starting deployment validation for {project_name}")

        # Build list of endpoints to validate
        endpoints_to_check = []

        if urls.get('api'):
            api_base = urls['api'].rstrip('/')
            if not api_base.startswith('http'):
                api_base = f"https://{api_base}"
            endpoints_to_check.extend([
                api_base,
                f"{api_base}/health",
                f"{api_base}/docs",
            ])

        if urls.get('frontend'):
            frontend_base = urls['frontend'].rstrip('/')
            if not frontend_base.startswith('http'):
                frontend_base = f"https://{frontend_base}"
            endpoints_to_check.append(frontend_base)

        if urls.get('admin'):
            admin_base = urls['admin'].rstrip('/')
            if not admin_base.startswith('http'):
                admin_base = f"https://{admin_base}"
            endpoints_to_check.append(admin_base)

        if not endpoints_to_check:
            logger.warning(f"No endpoints to validate for {project_name}")
            return True  # Nothing to validate

        # Run validation
        validator = DeploymentValidator()
        validation_result = await validator.validate_deployment(
            project_name=project_name,
            deployment_job_id=deployment_job.get("job_id", "unknown"),
            endpoints=endpoints_to_check
        )

        if validation_result.all_healthy:
            # All endpoints are healthy - send success notification
            logger.info(f"Deployment validation passed for {project_name}")

            # Emit success signal to runtime intelligence
            try:
                from controller.runtime_intelligence import SignalCollector, SignalPersister
                signal_collector = SignalCollector()
                signal = signal_collector.emit_deployment_success_signal(
                    project_name=project_name,
                    validated_endpoints=endpoints_to_check,
                    was_rescue=False
                )
                persister = SignalPersister()
                persister.persist([signal])
            except Exception as sig_err:
                logger.debug(f"Signal emission failed (non-blocking): {sig_err}")

            message = (
                f"✅ *Deployment Validated Successfully!*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n\n"
                f"📍 *Working URLs:*\n"
            )
            if urls.get('frontend'):
                message += f"• Frontend: {urls['frontend']}\n"
            if urls.get('api'):
                message += f"• API: {urls['api']}\n"
            if urls.get('admin'):
                message += f"• Admin: {urls['admin']}\n"

            message += (
                f"\nPlease test the application and provide feedback:\n"
                f"• `/approve {project_name}` - Approve for production\n"
                f"• `/feedback {project_name} <issues>` - Report issues"
            )
            await send_notification(application, message)
            return True

        # Validation failed - classify and potentially rescue
        classifier = DeploymentFailureClassifier()
        failure = classifier.classify_failure(validation_result)

        if failure is None:
            # Couldn't classify - partial success perhaps
            logger.warning(f"Could not classify failure for {project_name}")
            return True

        # Check if we can attempt rescue
        rescue_engine = get_rescue_engine()
        can_rescue, reason = rescue_engine.can_attempt_rescue(
            project_name,
            deployment_job.get("job_id")
        )

        if not can_rescue:
            # Emit critical failure signal (max attempts reached)
            try:
                from controller.runtime_intelligence import SignalCollector, SignalPersister
                signal_collector = SignalCollector()
                signal = signal_collector.emit_deployment_failure_signal(
                    project_name=project_name,
                    failure_type=failure.failure_type.value,
                    failed_endpoints=failure.failed_urls,
                    rescue_attempt=3,  # Max attempts
                    max_attempts=3
                )
                persister = SignalPersister()
                persister.persist([signal])
            except Exception as sig_err:
                logger.debug(f"Signal emission failed (non-blocking): {sig_err}")

            # Max attempts reached - notify for manual intervention
            message = (
                f"🚨 *MANUAL INTERVENTION REQUIRED*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Reason:* {escape_markdown(reason)}\n\n"
                f"*Failure Type:* {failure.failure_type.value}\n\n"
                f"*Failed Endpoints:*\n"
            )
            for f_url in failure.failed_urls[:5]:
                message += f"• {f_url.get('url', 'unknown')}: {f_url.get('error', 'unknown')}\n"

            message += (
                f"\n⚠️ Automatic rescue attempts exhausted.\n"
                f"Please investigate manually and use:\n"
                f"• `/feedback {project_name} <issues>` to report\n"
                f"• `/rescue {project_name}` to reset and retry"
            )
            await send_notification(application, message)
            return False

        # Detect server environment for rescue instructions
        server_config = None
        try:
            detector = ServerDetector()
            # Try to detect from any available source
            server_config = detector.detect_from_paths()
        except Exception as e:
            logger.debug(f"Server detection failed: {e}")

        # Create rescue job
        success, msg, rescue_job_id = await rescue_engine.create_rescue_job(
            project_name=project_name,
            failure=failure,
            source_deployment_job_id=deployment_job.get("job_id", "unknown"),
            controller_client=controller,
            server_config=server_config
        )

        if success and rescue_job_id:
            # Extract attempt number from reason string (e.g., "Attempt 2 of 3")
            attempt_num = 1
            try:
                import re
                match = re.search(r'Attempt (\d+)', reason)
                if match:
                    attempt_num = int(match.group(1))
            except Exception:
                pass

            # Emit failure signal to runtime intelligence
            try:
                from controller.runtime_intelligence import SignalCollector, SignalPersister
                signal_collector = SignalCollector()
                signal = signal_collector.emit_deployment_failure_signal(
                    project_name=project_name,
                    failure_type=failure.failure_type.value,
                    failed_endpoints=failure.failed_urls,
                    rescue_attempt=attempt_num,
                    max_attempts=3
                )
                persister = SignalPersister()
                persister.persist([signal])
            except Exception as sig_err:
                logger.debug(f"Signal emission failed (non-blocking): {sig_err}")

            # Notify about rescue job
            attempt_info = reason  # Contains attempt X of Y
            message = (
                f"❌ *Deployment Validation Failed*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Failure Type:* {failure.failure_type.value}\n\n"
                f"*Failed Endpoints:*\n"
            )
            for f_url in failure.failed_urls[:5]:
                message += f"• {f_url.get('url', 'unknown')}: {f_url.get('error', 'unknown')}\n"

            message += (
                f"\n🔧 *Rescue Job Created*\n"
                f"*Job ID:* `{rescue_job_id[:12]}...`\n"
                f"*Status:* {escape_markdown(attempt_info)}\n\n"
                f"Claude is attempting to fix the deployment issues."
            )
            await send_notification(application, message)
            return False  # Rescue in progress
        else:
            # Failed to create rescue job
            message = (
                f"⚠️ *Rescue Job Creation Failed*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Error:* {escape_markdown(msg)}\n\n"
                f"Please investigate manually."
            )
            await send_notification(application, message)
            return False

    except ImportError as e:
        logger.error(f"Rescue system not available: {e}")
        return True  # Continue without validation
    except Exception as e:
        logger.error(f"Deployment validation failed: {e}")
        return True  # Don't block on validation errors


async def handle_rescue_job_completion(
    application,
    project_name: str,
    rescue_job: Dict
) -> None:
    """
    Handle completion of a rescue job - re-validate deployment.

    Phase 22: Re-run validation after rescue completes.
    """
    try:
        from controller.rescue_engine import get_rescue_engine

        rescue_engine = get_rescue_engine()
        job_id = rescue_job.get("job_id", "")

        # Get deployment URLs
        urls = await get_project_deployment_urls(project_name)

        if not urls:
            rescue_engine.mark_rescue_failure(
                project_name, job_id, "No URLs found to validate"
            )
            return

        # Create a mock deployment job for validation
        mock_deployment_job = {
            "job_id": job_id,
            "project_name": project_name,
        }

        # Re-run validation
        validation_passed = await trigger_deployment_validation(
            application, project_name, mock_deployment_job, urls
        )

        if validation_passed:
            # Mark rescue as successful
            rescue_engine.mark_rescue_success(project_name, job_id)

            # Get attempt number from rescue state
            rescue_state = rescue_engine.get_rescue_state(project_name)
            attempt_num = len(rescue_state.attempts) if rescue_state else 1

            # Emit success signal to runtime intelligence
            try:
                from controller.runtime_intelligence import SignalCollector, SignalPersister
                signal_collector = SignalCollector()
                validated_endpoints = []
                if urls.get('api'):
                    validated_endpoints.append(urls['api'])
                if urls.get('frontend'):
                    validated_endpoints.append(urls['frontend'])
                if urls.get('admin'):
                    validated_endpoints.append(urls['admin'])

                signal = signal_collector.emit_deployment_success_signal(
                    project_name=project_name,
                    validated_endpoints=validated_endpoints,
                    was_rescue=True,
                    rescue_attempt=attempt_num
                )
                persister = SignalPersister()
                persister.persist([signal])
            except Exception as sig_err:
                logger.debug(f"Signal emission failed (non-blocking): {sig_err}")

            # Send success notification
            message = (
                f"✅ *Rescue Successful!*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Fixed after:* {attempt_num} attempt(s)\n\n"
                f"Deployment is now working:\n"
            )
            if urls.get('frontend'):
                message += f"• Frontend: {urls['frontend']}\n"
            if urls.get('api'):
                message += f"• API: {urls['api']}\n"
            if urls.get('admin'):
                message += f"• Admin: {urls['admin']}\n"

            message += (
                f"\nPlease test and provide feedback:\n"
                f"• `/approve {project_name}` - Approve for production\n"
                f"• `/feedback {project_name} <issues>` - Report issues"
            )
            await send_notification(application, message)
        else:
            # Validation still failing - rescue_engine handles next attempt or max-attempts notification
            rescue_engine.mark_rescue_failure(
                project_name, job_id, "Validation still failing after rescue"
            )

    except ImportError:
        logger.warning("Rescue system not available")
    except Exception as e:
        logger.error(f"Failed to handle rescue completion: {e}")


# -----------------------------------------------------------------------------
# Phase 21: CI Monitor for Auto-Detection and Self-Healing
# -----------------------------------------------------------------------------
def register_project_repo(project_name: str, github_repo: str):
    """Register a project's GitHub repo for CI monitoring."""
    global _monitored_repos
    _monitored_repos[project_name] = github_repo
    logger.info(f"Registered CI monitoring for {project_name}: {github_repo}")


async def ci_monitor_task(application):
    """Background task to monitor GitHub CI status and trigger auto-fixes."""
    global _ci_monitor_running, _ci_processed_runs, _ci_fix_attempts
    _ci_monitor_running = True

    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        logger.warning("GITHUB_TOKEN not set, CI monitor disabled")
        return

    logger.info(f"CI Monitor started, checking every {CI_MONITOR_INTERVAL}s")

    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    while _ci_monitor_running:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for project_name, repo in list(_monitored_repos.items()):
                    try:
                        # Get recent workflow runs
                        response = await client.get(
                            f"https://api.github.com/repos/{repo}/actions/runs",
                            headers=headers,
                            params={"per_page": 5}
                        )

                        if response.status_code != 200:
                            continue

                        runs = response.json().get("workflow_runs", [])

                        for run in runs:
                            run_id = run["id"]

                            # Skip already processed
                            if run_id in _ci_processed_runs:
                                continue

                            # Only process completed runs
                            if run["status"] != "completed":
                                continue

                            _ci_processed_runs.add(run_id)

                            conclusion = run["conclusion"]
                            workflow_name = run["name"]

                            if conclusion == "failure":
                                # CI Failed - notify and attempt auto-fix
                                attempts = _ci_fix_attempts.get(repo, 0)

                                if attempts < CI_MAX_AUTO_FIX_ATTEMPTS:
                                    message = (
                                        f"🔴 *CI Failed*\n\n"
                                        f"*Project:* {escape_markdown(project_name)}\n"
                                        f"*Repo:* {repo}\n"
                                        f"*Workflow:* {workflow_name}\n"
                                        f"*Attempt:* {attempts + 1}/{CI_MAX_AUTO_FIX_ATTEMPTS}\n\n"
                                        f"🔧 *Creating auto-fix job...*"
                                    )
                                    await send_notification(application, message)

                                    # Create bug_fix job
                                    fix_task = f"""CI FIX - Auto-remediation for {workflow_name}

PROJECT: {project_name}
REPO: {repo}
RUN: {run_id}

INSTRUCTIONS:
1. Clone the repository: git clone https://github.com/{repo}.git
2. Analyze the CI failure (check GitHub Actions logs)
3. Fix the identified issues
4. Run black and flake8 to fix Python formatting/linting
5. For React: ensure useCallback is used properly
6. Commit and push the fix
7. The CI will automatically re-run

Common fixes:
- Remove unused imports (flake8 F401)
- Use npm install --legacy-peer-deps for npm issues
- Pin bcrypt>=4.0.0,<4.2.0 for passlib compatibility
- Add useCallback for React hooks
"""
                                    result = await controller.create_claude_job(
                                        project_name=project_name,
                                        task_description=fix_task,
                                        task_type="bug_fix"
                                    )

                                    _ci_fix_attempts[repo] = attempts + 1

                                    if result.get("job_id"):
                                        logger.info(f"Created auto-fix job {result['job_id']} for {repo}")
                                else:
                                    # Max attempts reached
                                    message = (
                                        f"⚠️ *CI Auto-Fix Limit Reached*\n\n"
                                        f"*Project:* {escape_markdown(project_name)}\n"
                                        f"*Repo:* {repo}\n\n"
                                        f"Max auto-fix attempts ({CI_MAX_AUTO_FIX_ATTEMPTS}) reached.\n"
                                        f"Manual intervention required."
                                    )
                                    await send_notification(application, message)

                            elif conclusion == "success":
                                # CI Passed
                                if repo in _ci_fix_attempts and _ci_fix_attempts[repo] > 0:
                                    # Success after previous failure - reset and notify
                                    _ci_fix_attempts[repo] = 0
                                    message = (
                                        f"✅ *CI Passed After Fix*\n\n"
                                        f"*Project:* {escape_markdown(project_name)}\n"
                                        f"*Repo:* {repo}\n"
                                        f"*Workflow:* {workflow_name}\n\n"
                                        f"Auto-fix successful! Triggering deployment..."
                                    )
                                    await send_notification(application, message)

                                    # Auto-trigger deployment workflow
                                    await trigger_github_deployment(
                                        application, client, headers,
                                        project_name, repo
                                    )

                    except Exception as e:
                        logger.error(f"CI monitor error for {repo}: {e}")

            # Keep last 1000 processed runs
            if len(_ci_processed_runs) > 1000:
                _ci_processed_runs = set(list(_ci_processed_runs)[-500:])

        except Exception as e:
            logger.error(f"CI monitor error: {e}")

        await asyncio.sleep(CI_MONITOR_INTERVAL)


async def trigger_github_deployment(
    application,
    client: httpx.AsyncClient,
    headers: Dict,
    project_name: str,
    repo: str
):
    """Trigger GitHub deployment workflow after CI passes."""
    try:
        # Trigger deploy-testing workflow
        response = await client.post(
            f"https://api.github.com/repos/{repo}/actions/workflows/deploy-testing.yml/dispatches",
            headers=headers,
            json={
                "ref": "main",
                "inputs": {"confirm_deployment": "DEPLOY"}
            }
        )

        if response.status_code in (204, 200):
            message = (
                f"🚀 *Deployment Triggered*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Repo:* {repo}\n\n"
                f"Deployment to testing environment started."
            )
        else:
            message = (
                f"⚠️ *Deployment Trigger Failed*\n\n"
                f"*Project:* {escape_markdown(project_name)}\n"
                f"*Status:* {response.status_code}\n\n"
                f"Please trigger deployment manually."
            )

        await send_notification(application, message)

    except Exception as e:
        logger.error(f"Failed to trigger deployment for {repo}: {e}")


def stop_ci_monitor():
    """Stop the CI monitor."""
    global _ci_monitor_running
    _ci_monitor_running = False
    logger.info("CI monitor stopped")


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Start the bot."""
    global NOTIFICATION_CHAT_IDS

    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        sys.exit(1)

    logger.info("Starting Telegram bot...")
    logger.info(f"Controller URL: {CONTROLLER_BASE_URL}")
    logger.info(f"Bot Version: {BOT_VERSION}")

    # Log role configuration and set up notification recipients
    for role, users in ROLE_CONFIG.items():
        if users:
            logger.info(f"Role {role.value}: {len(users)} users configured")
            if role == UserRole.OWNER:
                NOTIFICATION_CHAT_IDS = users.copy()
                logger.info(f"Notification recipients: {NOTIFICATION_CHAT_IDS}")

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Identity command (Phase 13.4)
    application.add_handler(CommandHandler("whoami", whoami_command))

    # Health & Status commands (Phase 13.3)
    application.add_handler(CommandHandler("health", health_command))
    application.add_handler(CommandHandler("status", bot_status_command))
    application.add_handler(CommandHandler("projects", projects_command))
    application.add_handler(CommandHandler("deployments", deployments_command))

    # Core commands
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("new_project", new_project_command))
    application.add_handler(CommandHandler("project_status", project_status_command))
    application.add_handler(CommandHandler("dashboard", dashboard_command))

    # Approval workflow commands (Phase 13.5)
    application.add_handler(CommandHandler("approve", approve_command))
    application.add_handler(CommandHandler("reject", reject_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(CommandHandler("prod_approve", prod_approve_command))
    application.add_handler(CommandHandler("notifications", notifications_command))

    # Phase 15.1: Lifecycle commands
    application.add_handler(CommandHandler("lifecycle_status", lifecycle_status_command))
    application.add_handler(CommandHandler("lifecycle_approve", lifecycle_approve_command))
    application.add_handler(CommandHandler("lifecycle_reject", lifecycle_reject_command))
    application.add_handler(CommandHandler("lifecycle_feedback", lifecycle_feedback_command))
    application.add_handler(CommandHandler("lifecycle_prod_approve", lifecycle_prod_approve_command))

    # Phase 15.2: Continuous Change Cycle commands
    application.add_handler(CommandHandler("new_feature", new_feature_command))
    application.add_handler(CommandHandler("report_bug", report_bug_command))
    application.add_handler(CommandHandler("improve", improve_command))
    application.add_handler(CommandHandler("refactor", refactor_command))
    application.add_handler(CommandHandler("security_fix", security_fix_command))
    application.add_handler(CommandHandler("cycle_history", cycle_history_command))
    application.add_handler(CommandHandler("change_summary", change_summary_command))

    # Phase 15.3: Project Ingestion commands
    application.add_handler(CommandHandler("ingest_git", ingest_git_command))
    application.add_handler(CommandHandler("ingest_local", ingest_local_command))
    application.add_handler(CommandHandler("approve_ingestion", approve_ingestion_command))
    application.add_handler(CommandHandler("reject_ingestion", reject_ingestion_command))
    application.add_handler(CommandHandler("register_ingestion", register_ingestion_command))
    application.add_handler(CommandHandler("ingestion_status", ingestion_status_command))

    # Phase 17A: Runtime Intelligence commands (READ-ONLY)
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("signals_recent", signals_recent_command))
    application.add_handler(CommandHandler("runtime_status", runtime_status_command))

    # Phase 17B: Incident Classification commands (READ-ONLY)
    application.add_handler(CommandHandler("incidents", incidents_command))
    application.add_handler(CommandHandler("incidents_recent", incidents_recent_command))
    application.add_handler(CommandHandler("incidents_summary", incidents_summary_command))

    # Phase 18C: Execution Dispatcher commands (READ-ONLY status views)
    application.add_handler(CommandHandler("executions", executions_command))
    application.add_handler(CommandHandler("execution_status", execution_status_command))
    application.add_handler(CommandHandler("execution_summary", execution_summary_command))

    # Phase 18D: Post-Execution Verification commands (OBSERVATION ONLY)
    application.add_handler(CommandHandler("execution_verify", execution_verify_command))
    application.add_handler(CommandHandler("execution_violations", execution_violations_command))
    application.add_handler(CommandHandler("verification_recent", verification_recent_command))
    application.add_handler(CommandHandler("verification_summary", verification_summary_command))

    # Phase 19: Learning, Memory & System Intelligence commands (INSIGHT ONLY)
    application.add_handler(CommandHandler("learning_summary", learning_summary_command))
    application.add_handler(CommandHandler("learning_patterns", learning_patterns_command))
    application.add_handler(CommandHandler("learning_trends", learning_trends_command))
    application.add_handler(CommandHandler("learning_stats", learning_statistics_command))

    # Phase 22: Rescue & Recovery System commands
    application.add_handler(CommandHandler("rescue", rescue_command))
    application.add_handler(CommandHandler("rescue_history", rescue_history_command))

    # Callback query handler for inline buttons
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Message handler for natural language input
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Phase 16C: Document handler for file-based project creation
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # Error handler
    application.add_error_handler(error_handler)

    # Phase 20-21: Start background monitoring tasks
    async def post_init(app):
        """Start background tasks after bot is initialized."""
        global _completed_job_ids, _notified_jobs

        # Pre-populate all existing jobs to prevent re-notification on restart
        logger.info("Loading existing jobs to skip re-notification...")
        try:
            result = await controller.list_claude_jobs(limit=100)
            jobs = result.get("jobs", [])
            running_count = 0
            completed_count = 0
            for job in jobs:
                job_id = job.get("job_id")
                state = job.get("state")
                if not job_id:
                    continue
                # Mark ALL existing jobs as already processed
                _completed_job_ids.add(job_id)
                _notified_jobs[job_id] = {"state": state, "phase_triggered": True}
                if state in ("completed", "failed"):
                    completed_count += 1
                elif state == "running":
                    running_count += 1
            logger.info(f"Loaded {completed_count} completed/failed, {running_count} running jobs to skip")
        except Exception as e:
            logger.warning(f"Failed to load existing jobs: {e}")

        logger.info("Starting job monitor background task...")
        asyncio.create_task(job_monitor_task(app))

        # Phase 21: Start CI monitor if GitHub token available
        if os.getenv("GITHUB_TOKEN"):
            logger.info("Starting CI monitor background task...")
            asyncio.create_task(ci_monitor_task(app))
        else:
            logger.warning("GITHUB_TOKEN not set, CI auto-fix monitoring disabled")

    application.post_init = post_init

    # Start polling
    logger.info("Bot started. Polling for updates...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
