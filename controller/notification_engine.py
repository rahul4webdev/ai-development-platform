"""
Notification Engine - Centralized Notification Management
Phase 21: Push Notifications for Job Events

This module provides a centralized notification system that:
1. Manages notification templates
2. Routes notifications to appropriate channels (Telegram, etc.)
3. Tracks notification delivery
4. Supports push notifications via webhooks

IMPORTANT:
- All notifications are logged for audit
- Rate limiting is applied per recipient
- No sensitive data in notifications
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Awaitable

import httpx

logger = logging.getLogger("notification_engine")

# Configuration
NOTIFICATION_LOG_DIR = Path(os.getenv("NOTIFICATION_LOG_DIR", "/tmp/notifications"))
NOTIFICATION_LOG_DIR.mkdir(parents=True, exist_ok=True)
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 10  # max notifications per window


class NotificationType(str, Enum):
    """
    Types of notifications.

    Each type has specific formatting and priority.
    """
    # Job lifecycle
    JOB_STARTED = "job_started"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"

    # CI/CD
    CI_STARTED = "ci_started"
    CI_PASSED = "ci_passed"
    CI_FAILED = "ci_failed"
    CI_AUTO_FIX = "ci_auto_fix"

    # Deployment
    DEPLOY_TESTING_STARTED = "deploy_testing_started"
    DEPLOY_TESTING_COMPLETE = "deploy_testing_complete"
    DEPLOY_TESTING_FAILED = "deploy_testing_failed"
    DEPLOY_PROD_REQUESTED = "deploy_prod_requested"
    DEPLOY_PROD_COMPLETE = "deploy_prod_complete"

    # Approval
    APPROVAL_REQUIRED = "approval_required"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"

    # System
    SYSTEM_ALERT = "system_alert"
    SYSTEM_DEGRADED = "system_degraded"


class NotificationPriority(str, Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """Represents a notification to be sent."""
    notification_type: NotificationType
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    project_name: Optional[str] = None
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    delivered_at: Optional[datetime] = None
    delivery_channel: Optional[str] = None
    delivery_error: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "type": self.notification_type.value,
            "title": self.title,
            "message": self.message,
            "priority": self.priority.value,
            "project_name": self.project_name,
            "job_id": self.job_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None,
            "delivery_channel": self.delivery_channel,
            "delivery_error": self.delivery_error
        }


class NotificationTemplates:
    """Pre-defined notification templates."""

    @staticmethod
    def job_started(project_name: str, job_id: str, task_type: str) -> Notification:
        return Notification(
            notification_type=NotificationType.JOB_STARTED,
            title="Job Started",
            message=(
                f"🔄 *Job Started*\n\n"
                f"*Project:* {project_name}\n"
                f"*Type:* {task_type}\n"
                f"*Job ID:* `{job_id[:12]}...`\n\n"
                f"View progress in dashboard."
            ),
            priority=NotificationPriority.NORMAL,
            project_name=project_name,
            job_id=job_id,
            metadata={"task_type": task_type}
        )

    @staticmethod
    def job_completed(
        project_name: str,
        job_id: str,
        task_type: str,
        duration_seconds: int,
        next_step: Optional[str] = None
    ) -> Notification:
        duration_str = f"{duration_seconds // 60}m {duration_seconds % 60}s"
        next_step_str = f"\n\n*Next:* {next_step}" if next_step else ""

        return Notification(
            notification_type=NotificationType.JOB_COMPLETED,
            title="Job Completed",
            message=(
                f"✅ *Job Completed*\n\n"
                f"*Project:* {project_name}\n"
                f"*Type:* {task_type}\n"
                f"*Duration:* {duration_str}\n"
                f"*Job ID:* `{job_id[:12]}...`"
                f"{next_step_str}"
            ),
            priority=NotificationPriority.NORMAL,
            project_name=project_name,
            job_id=job_id,
            metadata={"task_type": task_type, "duration": duration_seconds}
        )

    @staticmethod
    def job_failed(
        project_name: str,
        job_id: str,
        task_type: str,
        error: str,
        exit_code: Optional[int] = None
    ) -> Notification:
        return Notification(
            notification_type=NotificationType.JOB_FAILED,
            title="Job Failed",
            message=(
                f"❌ *Job Failed*\n\n"
                f"*Project:* {project_name}\n"
                f"*Type:* {task_type}\n"
                f"*Exit Code:* {exit_code or 'N/A'}\n"
                f"*Job ID:* `{job_id[:12]}...`\n\n"
                f"*Error:*\n```\n{error[:300]}\n```"
            ),
            priority=NotificationPriority.HIGH,
            project_name=project_name,
            job_id=job_id,
            metadata={"task_type": task_type, "error": error, "exit_code": exit_code}
        )

    @staticmethod
    def ci_passed(project_name: str, repo: str, workflow: str) -> Notification:
        return Notification(
            notification_type=NotificationType.CI_PASSED,
            title="CI Passed",
            message=(
                f"✅ *CI Passed*\n\n"
                f"*Project:* {project_name}\n"
                f"*Repo:* {repo}\n"
                f"*Workflow:* {workflow}\n\n"
                f"Ready for deployment to testing."
            ),
            priority=NotificationPriority.NORMAL,
            project_name=project_name,
            metadata={"repo": repo, "workflow": workflow}
        )

    @staticmethod
    def ci_failed(
        project_name: str,
        repo: str,
        workflow: str,
        failure_type: str,
        auto_fix: bool = False
    ) -> Notification:
        auto_fix_str = "\n\n🔧 *Auto-fix initiated*" if auto_fix else ""
        return Notification(
            notification_type=NotificationType.CI_FAILED,
            title="CI Failed",
            message=(
                f"🔴 *CI Failed*\n\n"
                f"*Project:* {project_name}\n"
                f"*Repo:* {repo}\n"
                f"*Workflow:* {workflow}\n"
                f"*Failure:* {failure_type}"
                f"{auto_fix_str}"
            ),
            priority=NotificationPriority.HIGH,
            project_name=project_name,
            metadata={"repo": repo, "workflow": workflow, "failure_type": failure_type}
        )

    @staticmethod
    def deploy_testing_complete(
        project_name: str,
        api_url: str,
        web_url: str
    ) -> Notification:
        return Notification(
            notification_type=NotificationType.DEPLOY_TESTING_COMPLETE,
            title="Deployed to Testing",
            message=(
                f"🚀 *Deployed to Testing*\n\n"
                f"*Project:* {project_name}\n\n"
                f"*URLs:*\n"
                f"• API: {api_url}\n"
                f"• Web: {web_url}\n\n"
                f"Please test and provide feedback:\n"
                f"• `/approve {project_name}` - Approve for production\n"
                f"• `/feedback {project_name} <issues>` - Report issues"
            ),
            priority=NotificationPriority.HIGH,
            project_name=project_name,
            metadata={"api_url": api_url, "web_url": web_url}
        )

    @staticmethod
    def approval_required(
        project_name: str,
        action: str,
        requester: str
    ) -> Notification:
        return Notification(
            notification_type=NotificationType.APPROVAL_REQUIRED,
            title="Approval Required",
            message=(
                f"⏳ *Approval Required*\n\n"
                f"*Project:* {project_name}\n"
                f"*Action:* {action}\n"
                f"*Requested by:* {requester}\n\n"
                f"A different user must approve this action."
            ),
            priority=NotificationPriority.URGENT,
            project_name=project_name,
            metadata={"action": action, "requester": requester}
        )


class NotificationEngine:
    """
    Central notification engine for the platform.

    Features:
    - Multiple delivery channels (Telegram, webhook)
    - Rate limiting per recipient
    - Delivery tracking and logging
    - Template-based notifications
    """

    def __init__(self):
        self._channels: Dict[str, Callable[[Notification], Awaitable[bool]]] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}
        self._pending: List[Notification] = []
        self._delivered: List[Notification] = []

    def register_channel(
        self,
        name: str,
        handler: Callable[[Notification], Awaitable[bool]]
    ):
        """Register a notification delivery channel."""
        self._channels[name] = handler
        logger.info(f"Registered notification channel: {name}")

    def _check_rate_limit(self, recipient: str) -> bool:
        """Check if recipient is within rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)

        # Clean old entries
        if recipient in self._rate_limits:
            self._rate_limits[recipient] = [
                t for t in self._rate_limits[recipient]
                if t > window_start
            ]
        else:
            self._rate_limits[recipient] = []

        # Check limit
        if len(self._rate_limits[recipient]) >= RATE_LIMIT_MAX:
            return False

        self._rate_limits[recipient].append(now)
        return True

    def _log_notification(self, notification: Notification):
        """Log notification for audit."""
        log_file = NOTIFICATION_LOG_DIR / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(notification.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to log notification: {e}")

    async def send(
        self,
        notification: Notification,
        channel: Optional[str] = None,
        recipient: str = "default"
    ) -> bool:
        """
        Send a notification through specified channel.

        Args:
            notification: The notification to send
            channel: Channel name (None = all channels)
            recipient: Recipient identifier for rate limiting

        Returns:
            True if delivered successfully
        """
        # Check rate limit
        if not self._check_rate_limit(recipient):
            logger.warning(f"Rate limit exceeded for {recipient}")
            notification.delivery_error = "Rate limit exceeded"
            self._log_notification(notification)
            return False

        # Determine channels
        channels = [channel] if channel else list(self._channels.keys())

        # Try each channel
        delivered = False
        for ch_name in channels:
            if ch_name not in self._channels:
                continue

            try:
                handler = self._channels[ch_name]
                success = await handler(notification)
                if success:
                    notification.delivered_at = datetime.utcnow()
                    notification.delivery_channel = ch_name
                    delivered = True
                    break
            except Exception as e:
                logger.error(f"Channel {ch_name} delivery failed: {e}")
                notification.delivery_error = str(e)

        self._log_notification(notification)
        return delivered

    async def send_batch(
        self,
        notifications: List[Notification],
        channel: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Send multiple notifications.

        Returns:
            Dict with 'delivered' and 'failed' counts
        """
        delivered = 0
        failed = 0

        for notification in notifications:
            success = await self.send(notification, channel)
            if success:
                delivered += 1
            else:
                failed += 1

        return {"delivered": delivered, "failed": failed}

    def get_recent_notifications(
        self,
        project_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Get recent notifications from log."""
        notifications = []
        log_file = NOTIFICATION_LOG_DIR / f"{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"

        if log_file.exists():
            try:
                with open(log_file, "r") as f:
                    for line in f:
                        n = json.loads(line)
                        if project_name is None or n.get("project_name") == project_name:
                            notifications.append(n)
            except Exception as e:
                logger.error(f"Failed to read notifications: {e}")

        return notifications[-limit:]


# Telegram channel implementation
async def telegram_channel(notification: Notification) -> bool:
    """Send notification via Telegram."""
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_ids = os.getenv("NOTIFICATION_CHAT_IDS", "").split(",")

    if not bot_token or not chat_ids:
        logger.warning("Telegram not configured")
        return False

    async with httpx.AsyncClient(timeout=10.0) as client:
        for chat_id in chat_ids:
            if not chat_id.strip():
                continue
            try:
                response = await client.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={
                        "chat_id": chat_id.strip(),
                        "text": notification.message,
                        "parse_mode": "Markdown",
                        "disable_web_page_preview": True
                    }
                )
                if response.status_code == 200:
                    return True
            except Exception as e:
                logger.error(f"Telegram send error: {e}")

    return False


# Global instance
_notification_engine: Optional[NotificationEngine] = None


def get_notification_engine() -> NotificationEngine:
    """Get or create the notification engine instance."""
    global _notification_engine

    if _notification_engine is None:
        _notification_engine = NotificationEngine()
        _notification_engine.register_channel("telegram", telegram_channel)

    return _notification_engine
