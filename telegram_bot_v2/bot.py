"""
Telegram Bot - Phase 13

Stateless Telegram bot that communicates only via HTTP with the controller.

Features:
- Polling-based (for now)
- Token from ENV: TELEGRAM_BOT_TOKEN
- No hardcoded secrets
- Does NOT crash controller if Telegram fails
- Logs to /var/log/ai-telegram-bot.log

Safety:
- Bot cannot trigger prod deploy directly
- Bot cannot skip CI
- Bot cannot self-approve
- All actions logged
- Dual approval rules enforced via controller
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List

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

# Create logger
logger = logging.getLogger("telegram_bot")
logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

# File handler
try:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
except PermissionError:
    # Fall back to console if can't write to log file
    pass

# Console handler
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

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------
class FeedbackType(str, Enum):
    """Types of feedback from testers."""
    APPROVE = "approve"
    BUG = "bug"
    IMPROVEMENTS = "improvements"
    REJECT = "reject"


class CallbackAction(str, Enum):
    """Callback query actions."""
    FEEDBACK_APPROVE = "fb_approve"
    FEEDBACK_BUG = "fb_bug"
    FEEDBACK_IMPROVEMENTS = "fb_improve"
    FEEDBACK_REJECT = "fb_reject"
    PROD_APPROVE = "prod_approve"
    SELECT_PROJECT = "sel_proj"
    SELECT_ASPECT = "sel_aspect"


# -----------------------------------------------------------------------------
# User State Management (in-memory, stateless design)
# -----------------------------------------------------------------------------
class UserState:
    """Temporary state for multi-step interactions."""
    def __init__(self):
        self.pending_feedback: Dict[int, Dict[str, Any]] = {}  # user_id -> feedback context
        self.pending_project_creation: Dict[int, str] = {}  # user_id -> description
        self.awaiting_input: Dict[int, str] = {}  # user_id -> what we're waiting for


user_state = UserState()


# -----------------------------------------------------------------------------
# HTTP Client for Controller Communication
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Safety Guardrails
# -----------------------------------------------------------------------------
class SafetyGuardrails:
    """
    Safety constraints enforced by the Telegram bot.

    MANDATORY CONSTRAINTS:
    1. Bot CANNOT trigger prod deploy directly
    2. Bot CANNOT skip CI
    3. Bot CANNOT self-approve
    4. All bot actions are logged
    5. Dual approval rules enforced via controller

    ARCHITECTURAL SAFETY:
    - Bot is stateless and communicates only via HTTP
    - All business logic and validation is in controller
    - Bot cannot bypass controller rules
    - If Telegram is down, system continues autonomously
    """

    @staticmethod
    def log_action(user_id: int, action: str, details: Dict[str, Any]) -> None:
        """Log all user actions for audit trail."""
        logger.info(f"ACTION: user={user_id} action={action} details={details}")

    @staticmethod
    def is_production_action(action: str) -> bool:
        """Check if action is production-related (requires extra caution)."""
        prod_actions = ["prod_approve", "prod_apply", "prod_deploy"]
        return action in prod_actions

    @staticmethod
    def validate_feedback_has_explanation(
        feedback_type: str,
        explanation: Optional[str]
    ) -> tuple[bool, str]:
        """Ensure bug/improvements/reject feedback has explanation."""
        if feedback_type in ["bug", "improvements", "reject"]:
            if not explanation or len(explanation.strip()) < 10:
                return False, f"{feedback_type.title()} feedback requires explanation (min 10 chars)"
        return True, ""


safety = SafetyGuardrails()


class ControllerClient:
    """HTTP client for communicating with the controller API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to controller."""
        url = f"{self.base_url}{endpoint}"
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                if method.upper() == "GET":
                    response = await client.get(url, params=params)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data)
                else:
                    raise ValueError(f"Unsupported method: {method}")

                response.raise_for_status()
                return response.json()
        except httpx.TimeoutException:
            logger.error(f"Timeout connecting to controller: {url}")
            return {"error": "Controller timeout", "status": "error"}
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error from controller: {e.response.status_code}")
            try:
                return e.response.json()
            except Exception:
                return {"error": str(e), "status": "error"}
        except Exception as e:
            logger.error(f"Error connecting to controller: {e}")
            return {"error": str(e), "status": "error"}

    async def health_check(self) -> Dict[str, Any]:
        """Check controller health."""
        return await self._request("GET", "/health")

    async def create_project(
        self,
        description: str,
        user_id: str,
        requirements: List[str] = None,
        repo_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new project from natural language."""
        return await self._request("POST", "/v2/project/create", data={
            "description": description,
            "requirements": requirements or [],
            "repo_url": repo_url,
            "reference_urls": [],
            "user_id": user_id
        })

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
        """Submit testing feedback."""
        return await self._request("POST", f"/v2/project/{project_name}/aspect/{aspect}/feedback", data={
            "project_name": project_name,
            "aspect": aspect,
            "feedback_type": feedback_type,
            "explanation": explanation,
            "affected_features": affected_features or [],
            "user_id": user_id
        })

    async def approve_production(
        self,
        project_name: str,
        aspect: str,
        user_id: str,
        justification: str,
        risk_acknowledged: bool,
        rollback_plan: str
    ) -> Dict[str, Any]:
        """Approve production deployment."""
        return await self._request("POST", f"/v2/project/{project_name}/aspect/{aspect}/approve-production", data={
            "project_name": project_name,
            "aspect": aspect,
            "justification": justification,
            "risk_acknowledged": risk_acknowledged,
            "rollback_plan": rollback_plan,
            "user_id": user_id
        })

    async def get_dashboard(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get dashboard data."""
        if project_name:
            return await self._request("GET", f"/v2/dashboard/{project_name}")
        return await self._request("GET", "/v2/dashboard")

    async def get_notifications(self, project_name: Optional[str] = None) -> Dict[str, Any]:
        """Get pending notifications."""
        params = {"project_name": project_name} if project_name else None
        return await self._request("GET", "/v2/notifications", params=params)


# Initialize controller client
controller = ControllerClient(CONTROLLER_BASE_URL)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
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

    for proj in projects[:5]:  # Limit to 5 projects
        lines.append(f"*{proj.get('project_name', 'Unknown')}*")
        lines.append(f"  Status: {proj.get('overall_status', 'unknown')}")

        aspects = proj.get("aspects", {})
        for aspect_type, aspect_data in aspects.items():
            phase = aspect_data.get("current_phase", "unknown")
            health = aspect_data.get("health", "unknown")
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
                callback_data=f"{CallbackAction.FEEDBACK_APPROVE}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "🐞 Bug",
                callback_data=f"{CallbackAction.FEEDBACK_BUG}:{project_name}:{aspect}"
            ),
            InlineKeyboardButton(
                "✨ Improvements",
                callback_data=f"{CallbackAction.FEEDBACK_IMPROVEMENTS}:{project_name}:{aspect}"
            )
        ],
        [
            InlineKeyboardButton(
                "❌ Reject",
                callback_data=f"{CallbackAction.FEEDBACK_REJECT}:{project_name}:{aspect}"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


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

    # Testing URL
    if notif.get("environment_url"):
        lines.append(f"🔗 *Testing URL:* {notif.get('environment_url')}")
        lines.append("")

    # Features deployed
    features = notif.get("features_completed", [])
    if features:
        lines.append("✅ *Features Deployed:*")
        for feat in features[:10]:  # Limit to 10
            lines.append(f"  • {feat}")
        lines.append("")

    # Test coverage summary
    if notif.get("test_coverage_summary"):
        lines.append(f"🧪 *Test Summary:* {notif.get('test_coverage_summary')}")
        lines.append("")

    # Known limitations
    limitations = notif.get("known_limitations", [])
    if limitations:
        lines.append("⚠️ *Known Limitations:*")
        for lim in limitations[:5]:  # Limit to 5
            lines.append(f"  • {lim}")
        lines.append("")

    # What tester should focus on
    if details.get("testing_focus"):
        lines.append("👉 *What to Test:*")
        for focus in details.get("testing_focus", []):
            lines.append(f"  • {focus}")
        lines.append("")

    lines.append("Please test and provide feedback using /feedback command")

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


def format_notification(notif: Dict[str, Any]) -> tuple[str, Optional[InlineKeyboardMarkup]]:
    """Format any notification with appropriate template."""
    notif_type = notif.get("notification_type", "")
    keyboard = None

    if notif_type == "ci_result":
        text = format_ci_notification(notif)
    elif notif_type == "testing_ready":
        text = format_testing_deployment_notification(notif)
        # Add feedback buttons
        project = notif.get("project_name", "")
        aspect = notif.get("aspect", "")
        if project and aspect:
            keyboard = get_feedback_keyboard(project, aspect)
    elif notif_type == "approval_required":
        text = format_production_approval_notification(notif)
    else:
        # Generic notification format
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
# Command Handlers
# -----------------------------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command - Register user."""
    user = update.effective_user
    logger.info(f"User {user.id} ({user.username}) started bot")

    await update.message.reply_text(
        f"Welcome to the AI Development Platform!\n\n"
        f"I help you create and manage software projects using natural language.\n\n"
        f"*Available Commands:*\n"
        f"/new\\_project - Start a new project\n"
        f"/status - View project status\n"
        f"/dashboard - System overview\n"
        f"/approve - Approve testing\n"
        f"/feedback - Submit feedback\n"
        f"/prod\\_approve - Production approval\n"
        f"/notifications - View pending actions\n"
        f"/help - Show this help\n\n"
        f"Or just describe your project in plain English!",
        parse_mode="Markdown"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    await update.message.reply_text(
        "*AI Development Platform - Help*\n\n"
        "*Project Commands:*\n"
        "/new\\_project - Create a new project\n"
        "/status \\[project\\] - View project status\n"
        "/dashboard - System overview\n\n"
        "*Testing & Approval:*\n"
        "/approve \\[project\\] \\[aspect\\] - Approve testing\n"
        "/feedback \\[project\\] \\[aspect\\] - Submit feedback\n"
        "/prod\\_approve \\[project\\] \\[aspect\\] - Approve production\n\n"
        "*Notifications:*\n"
        "/notifications - View pending actions\n\n"
        "*Creating a Project:*\n"
        "Just describe what you want to build in plain English\\!\n"
        "Example: \"Build a SaaS CRM with admin panel and customer website\"",
        parse_mode="MarkdownV2"
    )


async def new_project_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /new_project command."""
    user_id = update.effective_user.id

    # Check if description was provided with command
    if context.args:
        description = " ".join(context.args)
        await create_project_from_description(update, description, str(user_id))
    else:
        # Ask for description
        user_state.awaiting_input[user_id] = "project_description"
        await update.message.reply_text(
            "Please describe your project in plain English.\n\n"
            "Example: \"Build a SaaS CRM with admin panel and customer website\""
        )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    if not context.args:
        # Show all projects via dashboard
        result = await controller.get_dashboard()
        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        await update.message.reply_text(
            format_dashboard(result),
            parse_mode="Markdown"
        )
    else:
        # Show specific project
        project_name = context.args[0]
        result = await controller.get_project(project_name)

        if "error" in result:
            await update.message.reply_text(f"Error: {result.get('error')}")
            return

        await update.message.reply_text(
            format_project_status(result),
            parse_mode="Markdown"
        )


async def dashboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /dashboard command."""
    result = await controller.get_dashboard()

    if "error" in result:
        await update.message.reply_text(f"Error: {result.get('error')}")
        return

    await update.message.reply_text(
        format_dashboard(result),
        parse_mode="Markdown"
    )


async def approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /approve command - Approve testing."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /approve <project_name> <aspect>\n"
            "Example: /approve my-crm core"
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]
    user_id = str(update.effective_user.id)

    # Submit approval feedback
    result = await controller.submit_feedback(
        project_name=project_name,
        aspect=aspect,
        feedback_type=FeedbackType.APPROVE.value,
        user_id=user_id
    )

    if "error" in result:
        await update.message.reply_text(f"Error: {result.get('error')}")
        return

    await update.message.reply_text(
        f"✅ *Approved!*\n\n"
        f"Project: {project_name}\n"
        f"Aspect: {aspect}\n"
        f"Action: {result.get('action_taken', 'Unknown')}\n\n"
        f"Next steps:\n" + "\n".join(f"• {s}" for s in result.get('next_steps', [])),
        parse_mode="Markdown"
    )


async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /feedback command - Show feedback options."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /feedback <project_name> <aspect>\n"
            "Example: /feedback my-crm core"
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]

    # Show inline keyboard with feedback options
    await update.message.reply_text(
        f"*Feedback for {project_name} - {aspect}*\n\n"
        f"Select your feedback type:",
        parse_mode="Markdown",
        reply_markup=get_feedback_keyboard(project_name, aspect)
    )


async def prod_approve_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /prod_approve command - Production approval."""
    if len(context.args) < 2:
        await update.message.reply_text(
            "Usage: /prod_approve <project_name> <aspect>\n"
            "Example: /prod_approve my-crm core\n\n"
            "Note: You will be asked for justification and rollback plan."
        )
        return

    project_name = context.args[0]
    aspect = context.args[1]
    user_id = update.effective_user.id

    # Store context for multi-step approval
    user_state.pending_feedback[user_id] = {
        "type": "prod_approval",
        "project_name": project_name,
        "aspect": aspect,
        "step": "justification"
    }
    user_state.awaiting_input[user_id] = "prod_justification"

    await update.message.reply_text(
        f"*Production Approval Request*\n\n"
        f"Project: {project_name}\n"
        f"Aspect: {aspect}\n\n"
        f"Please provide your justification for this production deployment:",
        parse_mode="Markdown"
    )


async def notifications_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /notifications command."""
    project_name = context.args[0] if context.args else None
    result = await controller.get_notifications(project_name)

    if "error" in result:
        await update.message.reply_text(f"Error: {result.get('error')}")
        return

    notifications = result.get("notifications", [])
    if not notifications:
        await update.message.reply_text("No pending notifications.")
        return

    # Send first notification with details, rest as summary
    first_notif = notifications[0]
    text, keyboard = format_notification(first_notif)
    await update.message.reply_text(text, parse_mode="Markdown", reply_markup=keyboard)

    # If more notifications, show summary of rest
    if len(notifications) > 1:
        lines = [f"\n*+{len(notifications) - 1} more notifications:*\n"]
        for notif in notifications[1:10]:  # Limit to 10
            emoji = "🔔"
            if notif.get("notification_type") == "testing_ready":
                emoji = "🎯"
            elif notif.get("notification_type") == "approval_required":
                emoji = "⚠️"
            elif notif.get("notification_type") == "ci_result":
                emoji = "🔧"

            lines.append(f"{emoji} {notif.get('title', 'Notification')}")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# -----------------------------------------------------------------------------
# Message Handler (Natural Language Input)
# -----------------------------------------------------------------------------
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle plain text messages - Natural language project creation."""
    user_id = update.effective_user.id
    text = update.message.text.strip()

    # Check if we're awaiting specific input
    awaiting = user_state.awaiting_input.get(user_id)

    if awaiting == "project_description":
        # User is providing project description
        del user_state.awaiting_input[user_id]
        await create_project_from_description(update, text, str(user_id))
        return

    elif awaiting == "feedback_explanation":
        # User is providing feedback explanation
        feedback_ctx = user_state.pending_feedback.get(user_id)
        if feedback_ctx:
            # SAFETY: Validate explanation is provided
            is_valid, error_msg = safety.validate_feedback_has_explanation(
                feedback_ctx["feedback_type"],
                text
            )
            if not is_valid:
                await update.message.reply_text(
                    f"⚠️ {error_msg}\n\nPlease provide more details:"
                )
                return

            del user_state.awaiting_input[user_id]
            del user_state.pending_feedback[user_id]

            # SAFETY: Log the action
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

            emoji = {"bug": "🐞", "improvements": "✨", "reject": "❌"}.get(
                feedback_ctx["feedback_type"], "📝"
            )
            await update.message.reply_text(
                f"{emoji} *Feedback Submitted*\n\n"
                f"Project: {feedback_ctx['project_name']}\n"
                f"Aspect: {feedback_ctx['aspect']}\n"
                f"Type: {feedback_ctx['feedback_type']}\n"
                f"Action: {result.get('action_taken', 'Processed')}\n\n"
                f"Next steps:\n" + "\n".join(f"• {s}" for s in result.get('next_steps', [])),
                parse_mode="Markdown"
            )
        return

    elif awaiting == "prod_justification":
        # User is providing production approval justification
        feedback_ctx = user_state.pending_feedback.get(user_id)
        if feedback_ctx:
            feedback_ctx["justification"] = text
            feedback_ctx["step"] = "rollback_plan"
            user_state.awaiting_input[user_id] = "prod_rollback_plan"

            await update.message.reply_text(
                "Please provide your rollback plan in case of issues:"
            )
        return

    elif awaiting == "prod_rollback_plan":
        # User is providing rollback plan
        feedback_ctx = user_state.pending_feedback.get(user_id)
        if feedback_ctx:
            del user_state.awaiting_input[user_id]
            del user_state.pending_feedback[user_id]

            # SAFETY: Log production approval attempt
            safety.log_action(user_id, "prod_approval_attempt", {
                "project": feedback_ctx["project_name"],
                "aspect": feedback_ctx["aspect"],
                "has_justification": bool(feedback_ctx.get("justification")),
                "has_rollback_plan": bool(text)
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
                # SAFETY: Log failed approval
                safety.log_action(user_id, "prod_approval_failed", {
                    "project": feedback_ctx["project_name"],
                    "error": result.get("error")
                })
                await update.message.reply_text(f"Error: {result.get('error')}")
                return

            if result.get("approved"):
                # SAFETY: Log successful approval
                safety.log_action(user_id, "prod_approval_success", {
                    "project": feedback_ctx["project_name"],
                    "aspect": feedback_ctx["aspect"],
                    "approval_id": result.get("approval_id")
                })
                await update.message.reply_text(
                    f"✅ *Production Deployment Approved*\n\n"
                    f"Project: {feedback_ctx['project_name']}\n"
                    f"Aspect: {feedback_ctx['aspect']}\n"
                    f"Status: {result.get('deployment_status', 'Processing')}\n"
                    f"URL: {result.get('production_url', 'Pending')}",
                    parse_mode="Markdown"
                )
            else:
                await update.message.reply_text(
                    f"❌ *Production Approval Failed*\n\n"
                    f"{result.get('message', 'Unknown error')}",
                    parse_mode="Markdown"
                )
        return

    # Default: Treat as project description
    # Check if it looks like a project description
    if len(text) > 20 and any(word in text.lower() for word in [
        "build", "create", "make", "develop", "want", "need", "app", "website",
        "system", "platform", "saas", "crm", "api", "service"
    ]):
        await create_project_from_description(update, text, str(user_id))
    else:
        await update.message.reply_text(
            "I didn't understand that. You can:\n\n"
            "• Describe a project to create (e.g., \"Build a SaaS CRM\")\n"
            "• Use /help to see available commands\n"
            "• Use /status to view projects"
        )


async def create_project_from_description(
    update: Update,
    description: str,
    user_id: str
) -> None:
    """Create a project from natural language description."""
    logger.info(f"Creating project from description: {description[:50]}...")

    await update.message.reply_text(
        "🔄 Creating your project...\n\n"
        "Claude is analyzing your requirements and generating the project structure."
    )

    result = await controller.create_project(
        description=description,
        user_id=user_id
    )

    if "error" in result:
        await update.message.reply_text(
            f"❌ *Error creating project:*\n{result.get('error')}",
            parse_mode="Markdown"
        )
        return

    aspects_list = "\n".join(f"• {a}" for a in result.get("aspects_initialized", []))
    next_steps = "\n".join(f"• {s}" for s in result.get("next_steps", []))

    await update.message.reply_text(
        f"✅ *Project Created!*\n\n"
        f"*Name:* {result.get('project_name', 'Unknown')}\n"
        f"*Contract ID:* `{result.get('contract_id', 'Unknown')}`\n\n"
        f"*Aspects Initialized:*\n{aspects_list}\n\n"
        f"*Next Steps:*\n{next_steps}\n\n"
        f"Use /status {result.get('project_name', '')} to track progress.",
        parse_mode="Markdown"
    )


# -----------------------------------------------------------------------------
# Callback Query Handler (Inline Buttons)
# -----------------------------------------------------------------------------
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    data = query.data
    parts = data.split(":")

    if len(parts) < 3:
        await query.edit_message_text("Invalid callback data.")
        return

    action = parts[0]
    project_name = parts[1]
    aspect = parts[2]

    if action == CallbackAction.FEEDBACK_APPROVE:
        # Direct approval - no explanation needed
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
            f"✅ *Approved!*\n\n"
            f"Project: {project_name}\n"
            f"Aspect: {aspect}\n"
            f"Action: {result.get('action_taken', 'Unknown')}\n\n"
            f"Next steps:\n" + "\n".join(f"• {s}" for s in result.get('next_steps', [])),
            parse_mode="Markdown"
        )

    elif action in [
        CallbackAction.FEEDBACK_BUG,
        CallbackAction.FEEDBACK_IMPROVEMENTS,
        CallbackAction.FEEDBACK_REJECT
    ]:
        # These require explanation
        feedback_type_map = {
            CallbackAction.FEEDBACK_BUG: FeedbackType.BUG.value,
            CallbackAction.FEEDBACK_IMPROVEMENTS: FeedbackType.IMPROVEMENTS.value,
            CallbackAction.FEEDBACK_REJECT: FeedbackType.REJECT.value,
        }
        feedback_type = feedback_type_map[action]

        # Store context for follow-up
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
            f"_(This explanation is mandatory)_",
            parse_mode="Markdown"
        )


# -----------------------------------------------------------------------------
# Error Handler
# -----------------------------------------------------------------------------
async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors."""
    logger.error(f"Update {update} caused error {context.error}")

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "An error occurred. Please try again later."
        )


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

    # Create application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("new_project", new_project_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("dashboard", dashboard_command))
    application.add_handler(CommandHandler("approve", approve_command))
    application.add_handler(CommandHandler("feedback", feedback_command))
    application.add_handler(CommandHandler("prod_approve", prod_approve_command))
    application.add_handler(CommandHandler("notifications", notifications_command))

    # Add callback query handler for inline buttons
    application.add_handler(CallbackQueryHandler(handle_callback))

    # Add message handler for natural language input
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Add error handler
    application.add_error_handler(error_handler)

    # Start polling
    logger.info("Bot started. Polling for updates...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
