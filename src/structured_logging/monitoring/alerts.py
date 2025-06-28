"""
Alert management for monitoring dashboard

This module provides alert rules and notifications for log streams.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ..logger import get_logger


class AlertLevel(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertCondition:
    """Alert trigger condition"""

    # Condition types
    field_name: Optional[str] = None
    operator: str = "equals"  # equals, contains, gt, lt, gte, lte, regex
    value: Any = None

    # Aggregation conditions
    count_threshold: Optional[int] = None
    time_window_seconds: Optional[int] = None

    # Complex conditions
    expression: Optional[str] = None  # e.g., "error_rate > 0.1"

    def evaluate(self, log: Dict[str, Any]) -> bool:
        """Evaluate condition against a log entry"""
        if self.expression:
            # Evaluate expression (simplified)
            try:
                # In production, use safe expression evaluation
                return self._evaluate_expression(log)
            except Exception:
                return False

        if self.field_name is None:
            return False

        field_value = log.get(self.field_name)

        if self.operator == "equals":
            return field_value == self.value
        elif self.operator == "contains":
            return str(self.value) in str(field_value)
        elif self.operator == "gt":
            return float(field_value) > float(self.value)
        elif self.operator == "lt":
            return float(field_value) < float(self.value)
        elif self.operator == "gte":
            return float(field_value) >= float(self.value)
        elif self.operator == "lte":
            return float(field_value) <= float(self.value)
        elif self.operator == "regex":
            return bool(re.match(str(self.value), str(field_value)))

        return False

    def _evaluate_expression(self, log: Dict[str, Any]) -> bool:
        """Evaluate complex expression (simplified)"""
        # In production, use a proper expression parser
        # This is a simplified version for demo
        expr = self.expression

        # Replace field references with values
        for key, value in log.items():
            expr = expr.replace(f"{{{key}}}", str(value))

        # Very basic evaluation (NOT SAFE for production)
        try:
            return eval(expr)
        except Exception:
            return False


@dataclass
class AlertRule:
    """Alert rule definition"""

    id: str
    name: str
    description: str = ""
    level: AlertLevel = AlertLevel.WARNING

    # Conditions
    conditions: List[AlertCondition] = field(default_factory=list)
    condition_operator: str = "and"  # and, or

    # Rate limiting
    cooldown_seconds: int = 300  # 5 minutes default
    max_alerts_per_hour: int = 10

    # Actions
    enabled: bool = True
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # State
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    recent_triggers: List[datetime] = field(default_factory=list)

    def should_trigger(self, log: Dict[str, Any]) -> bool:
        """Check if rule should trigger for log"""
        if not self.enabled:
            return False

        # Check cooldown
        if self.last_triggered:
            cooldown_elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if cooldown_elapsed < self.cooldown_seconds:
                return False

        # Check rate limit
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_count = sum(1 for t in self.recent_triggers if t > hour_ago)
        if recent_count >= self.max_alerts_per_hour:
            return False

        # Evaluate conditions
        if self.condition_operator == "and":
            return all(cond.evaluate(log) for cond in self.conditions)
        else:  # or
            return any(cond.evaluate(log) for cond in self.conditions)

    def trigger(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger the alert"""
        self.last_triggered = datetime.now()
        self.trigger_count += 1
        self.recent_triggers.append(datetime.now())

        # Cleanup old triggers
        hour_ago = datetime.now() - timedelta(hours=1)
        self.recent_triggers = [t for t in self.recent_triggers if t > hour_ago]

        return {
            "id": self.id,
            "name": self.name,
            "level": self.level.value,
            "description": self.description,
            "triggered_at": datetime.now().isoformat(),
            "trigger_count": self.trigger_count,
            "log": log,
            "actions": self.actions,
        }


class AlertManager:
    """
    Manages alert rules and notifications

    Features:
    - Rule-based alerting
    - Rate limiting and cooldowns
    - Multiple notification channels
    - Alert aggregation
    - Historical tracking
    """

    def __init__(self):
        self.logger = get_logger("monitoring.alerts")
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
        self._notification_handlers: List[Callable] = []

        # Default rules
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default alert rules"""
        # High error rate
        self.add_rule(
            name="High Error Rate",
            conditions=[
                AlertCondition(field_name="level", operator="equals", value="error")
            ],
            level=AlertLevel.ERROR,
            description="Triggered when error logs exceed threshold",
            cooldown_seconds=300,
        )

        # Slow response time
        self.add_rule(
            name="Slow Response Time",
            conditions=[
                AlertCondition(
                    field_name="response_time", operator="gt", value=1000  # 1 second
                )
            ],
            level=AlertLevel.WARNING,
            description="Response time exceeds 1 second",
        )

        # Critical error patterns
        self.add_rule(
            name="Critical Error Pattern",
            conditions=[
                AlertCondition(
                    field_name="message",
                    operator="regex",
                    value=r"(fatal|critical|emergency|panic)",
                )
            ],
            level=AlertLevel.CRITICAL,
            description="Critical error pattern detected",
        )

    def add_rule(
        self,
        name: str,
        conditions: List[AlertCondition],
        level: AlertLevel = AlertLevel.WARNING,
        description: str = "",
        actions: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Add a new alert rule"""
        rule_id = f"rule_{len(self.rules) + 1}"

        rule = AlertRule(
            id=rule_id,
            name=name,
            description=description,
            level=level,
            conditions=conditions,
            actions=actions or [],
            **kwargs,
        )

        self.rules[rule_id] = rule
        self.logger.info(f"Added alert rule: {name} (ID: {rule_id})")

        return rule_id

    def remove_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            self.logger.info(f"Removed alert rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> None:
        """Enable an alert rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str) -> None:
        """Disable an alert rule"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    async def check(self, log: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check log against all rules"""
        triggered_alerts = []

        for rule in self.rules.values():
            if rule.should_trigger(log):
                alert = rule.trigger(log)
                triggered_alerts.append(alert)

                # Add to active alerts
                self.active_alerts.append(alert)

                # Add to history
                self.alert_history.append(alert)

                # Execute actions
                await self._execute_actions(alert)

                # Notify handlers
                await self._notify_handlers(alert)

        # Cleanup old alerts
        self._cleanup_alerts()

        return triggered_alerts

    async def _execute_actions(self, alert: Dict[str, Any]) -> None:
        """Execute alert actions"""
        for action in alert.get("actions", []):
            action_type = action.get("type")

            try:
                if action_type == "webhook":
                    await self._send_webhook(action, alert)
                elif action_type == "email":
                    await self._send_email(action, alert)
                elif action_type == "log":
                    self.logger.warning(
                        f"Alert triggered: {alert['name']}", extra={"alert": alert}
                    )

            except Exception as e:
                self.logger.error(f"Failed to execute action {action_type}: {e}")

    async def _send_webhook(
        self, action: Dict[str, Any], alert: Dict[str, Any]
    ) -> None:
        """Send webhook notification"""
        import aiohttp

        url = action.get("url")
        if not url:
            return

        payload = {"alert": alert, "timestamp": datetime.now().isoformat()}

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    self.logger.error(f"Webhook failed: {response.status}")

    async def _send_email(self, action: Dict[str, Any], alert: Dict[str, Any]) -> None:
        """Send email notification (placeholder)"""
        self.logger.info(
            f"Would send email to {action.get('to')} about alert: {alert['name']}"
        )

    async def _notify_handlers(self, alert: Dict[str, Any]) -> None:
        """Notify registered handlers"""
        for handler in self._notification_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler error: {e}")

    def register_handler(self, handler: Callable) -> None:
        """Register a notification handler"""
        self._notification_handlers.append(handler)

    def _cleanup_alerts(self) -> None:
        """Cleanup old alerts"""
        # Keep only last hour of active alerts
        hour_ago = datetime.now() - timedelta(hours=1)

        self.active_alerts = [
            alert
            for alert in self.active_alerts
            if datetime.fromisoformat(alert["triggered_at"]) > hour_ago
        ]

        # Keep only last 24 hours of history
        day_ago = datetime.now() - timedelta(days=1)

        self.alert_history = [
            alert
            for alert in self.alert_history
            if datetime.fromisoformat(alert["triggered_at"]) > day_ago
        ]

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        return self.active_alerts

    def get_all_alerts(self) -> List[Dict[str, Any]]:
        """Get all alerts (active and history)"""
        return self.alert_history

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        stats = {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for r in self.rules.values() if r.enabled),
            "active_alerts": len(self.active_alerts),
            "total_triggered": sum(r.trigger_count for r in self.rules.values()),
            "alerts_by_level": {},
            "top_triggered_rules": [],
        }

        # Count by level
        for level in AlertLevel:
            count = sum(
                1 for alert in self.active_alerts if alert["level"] == level.value
            )
            stats["alerts_by_level"][level.value] = count

        # Top triggered rules
        sorted_rules = sorted(
            self.rules.values(), key=lambda r: r.trigger_count, reverse=True
        )

        stats["top_triggered_rules"] = [
            {"name": rule.name, "count": rule.trigger_count}
            for rule in sorted_rules[:5]
        ]

        return stats


def create_alert_rule(
    name: str,
    field: str,
    operator: str,
    value: Any,
    level: AlertLevel = AlertLevel.WARNING,
    **kwargs,
) -> AlertRule:
    """
    Create a simple alert rule

    Args:
        name: Rule name
        field: Log field to check
        operator: Comparison operator
        value: Value to compare against
        level: Alert severity level
        **kwargs: Additional rule parameters

    Returns:
        AlertRule instance
    """
    condition = AlertCondition(field_name=field, operator=operator, value=value)

    return AlertRule(
        id=f"rule_{datetime.now().timestamp()}",
        name=name,
        level=level,
        conditions=[condition],
        **kwargs,
    )
