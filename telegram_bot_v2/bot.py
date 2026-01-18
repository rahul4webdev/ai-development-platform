"""
Telegram Bot - Phase 13.3-13.12

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

Safety:
- Bot cannot trigger prod deploy directly
- Bot cannot skip CI
- Bot cannot self-approve
- All actions logged
- Dual approval rules enforced via controller
- Rate limiting prevents abuse
- Degraded mode protects system integrity
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
BOT_VERSION = "0.13.12"
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

# -----------------------------------------------------------------------------
# Phase 13.4: Role System
# -----------------------------------------------------------------------------
class UserRole(str, Enum):
    """User roles for access control."""
    OWNER = "owner"
    ADMIN = "admin"
    TESTER = "tester"
    VIEWER = "viewer"


# Role configuration - loaded from environment or config
# Format: ROLE_OWNERS=123456,789012 ROLE_ADMINS=111111,222222
def load_role_config() -> Dict[UserRole, List[int]]:
    """Load role mappings from environment."""
    config = {
        UserRole.OWNER: [],
        UserRole.ADMIN: [],
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
    for role in [UserRole.OWNER, UserRole.ADMIN, UserRole.TESTER]:
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

        lines = ["*System Health Check*", ""]

        # Phase 13.12: Show system mode prominently
        sys_status = system_state.get_status()
        mode = sys_status["mode"]
        mode_emoji = {"normal": "🟢", "degraded": "🟡", "critical": "🔴"}.get(mode, "⚪")
        lines.append(f"{mode_emoji} *System Mode:* {mode.upper()}")
        if sys_status["degraded_reason"]:
            lines.append(f"   Reason: {sys_status['degraded_reason']}")
        if sys_status["owner_override_active"]:
            lines.append("   ⚠️ Owner override is active")
        lines.append("")

        # Check controller health
        controller_health = await controller.health_check()
        if "error" in controller_health:
            lines.append("❌ *Controller:* Unreachable")
            lines.append(f"   Error: {controller_health.get('error')}")
        else:
            lines.append("✅ *Controller:* Healthy")
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
                    lines.append("⚠️ *WARNING: Version Mismatch*")
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
            lines.append(f"{emoji} *{svc}:* {status.get('status')}")
            if status.get("since") and status.get("since") != "unknown":
                lines.append(f"   Since: {status.get('since')}")

        lines.append("")

        # Bot info
        lines.append(f"🤖 *Bot Version:* {BOT_VERSION}")
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
                    lines.append(f"📦 *Last Deployment:* {latest_deploy}")
                else:
                    lines.append("📦 *Last Deployment:* None recorded")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")
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
    """Handle /dashboard command. ALWAYS responds."""
    try:
        user_id = update.effective_user.id
        safety.log_action(user_id, "view_dashboard", {})

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
    """Create a project from natural language description."""
    try:
        logger.info(f"Creating project from description: {description[:50]}...")

        await update.message.reply_text(
            "Creating your project...\n\n"
            "Claude is analyzing your requirements and generating the project structure."
        )

        result = await controller.create_project(
            description=description,
            user_id=user_id
        )

        if "error" in result:
            await update.message.reply_text(
                f"*Error creating project:*\n{result.get('error')}",
                parse_mode="Markdown"
            )
            return

        aspects_list = "\n".join(f"- {a}" for a in result.get("aspects_initialized", []))
        next_steps = "\n".join(f"- {s}" for s in result.get("next_steps", []))

        await update.message.reply_text(
            f"*Project Created!*\n\n"
            f"*Name:* {result.get('project_name', 'Unknown')}\n"
            f"*Contract ID:* {result.get('contract_id', 'Unknown')}\n\n"
            f"*Aspects Initialized:*\n{aspects_list}\n\n"
            f"*Next Steps:*\n{next_steps}\n\n"
            f"Use /projects to view all projects.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in create_project_from_description: {e}")
        await update.message.reply_text(f"Error creating project: {str(e)}")


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
        else:
            await query.edit_message_text(f"Unknown action: {action}")
    except Exception as e:
        logger.error(f"Error in handle_callback: {e}")
        if update.callback_query:
            await update.callback_query.edit_message_text(f"Error: {str(e)}")


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
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    """Start the bot."""
    if not TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        sys.exit(1)

    logger.info("Starting Telegram bot...")
    logger.info(f"Controller URL: {CONTROLLER_BASE_URL}")
    logger.info(f"Bot Version: {BOT_VERSION}")

    # Log role configuration
    for role, users in ROLE_CONFIG.items():
        if users:
            logger.info(f"Role {role.value}: {len(users)} users configured")

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

    # Callback query handler for inline buttons
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Message handler for natural language input
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Error handler
    application.add_error_handler(error_handler)

    # Start polling
    logger.info("Bot started. Polling for updates...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
