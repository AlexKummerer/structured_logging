"""
Azure Monitor integration for structured logging
"""

import base64
import gzip
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    requests = None
    HAS_REQUESTS = False

try:
    from azure.core.exceptions import AzureError
    from azure.identity import DefaultAzureCredential
    from azure.monitor.ingestion import LogsIngestionClient

    HAS_AZURE_MONITOR = True
except ImportError:
    LogsIngestionClient = None
    DefaultAzureCredential = None
    AzureError = Exception
    HAS_AZURE_MONITOR = False

from .base import CloudHandlerConfig, CloudLogHandler


@dataclass
class AzureMonitorConfig(CloudHandlerConfig):
    """Configuration for Azure Monitor handler"""

    # Azure Monitor settings
    workspace_id: Optional[str] = None  # Log Analytics workspace ID
    workspace_key: Optional[str] = None  # Primary or secondary key (for direct API)

    # Data Collection Endpoint (DCE) - for new ingestion API
    dce_endpoint: Optional[str] = None  # https://my-dce.region.ingest.monitor.azure.com
    dcr_immutable_id: Optional[str] = None  # Data Collection Rule immutable ID
    stream_name: str = "Custom-StructuredLogs"  # DCR stream name

    # Application Insights settings (alternative)
    instrumentation_key: Optional[str] = None  # App Insights key
    ingestion_endpoint: Optional[str] = None  # Override default endpoint

    # Log type and configuration
    log_type: str = "StructuredLogs"  # Table name in Log Analytics
    time_field: str = "TimeGenerated"  # Timestamp field name

    # Authentication
    use_managed_identity: bool = True  # Use managed identity if available
    tenant_id: Optional[str] = None  # For service principal auth
    client_id: Optional[str] = None  # For service principal auth
    client_secret: Optional[str] = None  # For service principal auth

    # Performance
    use_compression: bool = True  # Compress payloads
    api_version: str = "2016-04-01"  # Log Analytics API version

    # Additional fields to include
    include_cloud_role: bool = True  # Add cloud role information
    cloud_role_name: Optional[str] = None  # Override auto-detected role
    cloud_role_instance: Optional[str] = None  # Override auto-detected instance

    def __post_init__(self):
        """Validate configuration"""
        # Ensure we have at least one authentication method
        has_workspace = bool(self.workspace_id and self.workspace_key)
        has_dce = bool(self.dce_endpoint and self.dcr_immutable_id)
        has_app_insights = bool(self.instrumentation_key)

        if not (has_workspace or has_dce or has_app_insights):
            raise ValueError(
                "Azure Monitor requires either: "
                "1) workspace_id and workspace_key, "
                "2) dce_endpoint and dcr_immutable_id, or "
                "3) instrumentation_key"
            )


class AzureMonitorHandler(CloudLogHandler):
    """
    Azure Monitor handler for structured logging

    Supports multiple ingestion methods:
    1. Direct Log Analytics API (workspace ID + key)
    2. Data Collection Endpoint (DCE) with managed identity
    3. Application Insights (instrumentation key)

    Features:
    - Automatic field mapping to Azure Monitor schema
    - Compression and batching
    - Managed identity support
    - Custom properties and metrics
    - Automatic retry with exponential backoff
    """

    def __init__(self, config: AzureMonitorConfig):
        if not HAS_REQUESTS and (config.workspace_id or config.instrumentation_key):
            raise ImportError(
                "requests is required for Azure Monitor integration. "
                "Install with: pip install structured-logging[azure]"
            )

        if config.dce_endpoint and not HAS_AZURE_MONITOR:
            raise ImportError(
                "azure-monitor-ingestion is required for DCE integration. "
                "Install with: pip install structured-logging[azure]"
            )

        super().__init__(config)
        self.config: AzureMonitorConfig = config

        # Initialize appropriate client
        self._init_client()

        # Track cloud role information
        self._init_cloud_role()

    def _init_client(self) -> None:
        """Initialize Azure Monitor client based on configuration"""
        self.client = None
        self.ingestion_type = None

        if self.config.dce_endpoint and self.config.dcr_immutable_id:
            # Use new ingestion API with DCE
            self.ingestion_type = "dce"
            if HAS_AZURE_MONITOR:
                # Create credential
                if self.config.client_id and self.config.client_secret:
                    from azure.identity import ClientSecretCredential

                    credential = ClientSecretCredential(
                        tenant_id=self.config.tenant_id,
                        client_id=self.config.client_id,
                        client_secret=self.config.client_secret,
                    )
                else:
                    credential = DefaultAzureCredential()

                # Create ingestion client
                self.client = LogsIngestionClient(
                    endpoint=self.config.dce_endpoint,
                    credential=credential,
                    logging_enable=False,  # Avoid recursion
                )

        elif self.config.workspace_id and self.config.workspace_key:
            # Use direct Log Analytics API
            self.ingestion_type = "workspace"
            self.workspace_url = (
                f"https://{self.config.workspace_id}.ods.opinsights.azure.com"
                f"/api/logs?api-version={self.config.api_version}"
            )

        elif self.config.instrumentation_key:
            # Use Application Insights
            self.ingestion_type = "appinsights"
            self.appinsights_url = (
                self.config.ingestion_endpoint
                or "https://dc.services.visualstudio.com/v2/track"
            )

    def _init_cloud_role(self) -> None:
        """Initialize cloud role information"""
        if self.config.cloud_role_name:
            self.cloud_role_name = self.config.cloud_role_name
        else:
            # Auto-detect from environment
            self.cloud_role_name = (
                os.environ.get("WEBSITE_SITE_NAME")  # App Service
                or os.environ.get("FUNCTIONS_WORKER_RUNTIME")  # Functions
                or os.environ.get("CONTAINER_APP_NAME")  # Container Apps
                or "StructuredLoggingApp"
            )

        if self.config.cloud_role_instance:
            self.cloud_role_instance = self.config.cloud_role_instance
        else:
            import socket

            self.cloud_role_instance = (
                os.environ.get("WEBSITE_INSTANCE_ID")  # App Service
                or socket.gethostname()
            )

    def _prepare_log_entry(self, record: Any) -> Dict[str, Any]:
        """Prepare log entry for Azure Monitor format"""
        # Extract basic fields
        message = record.getMessage() if hasattr(record, "getMessage") else str(record)
        timestamp = datetime.now(timezone.utc).isoformat()

        # Map Python log levels to Azure severity
        severity_map = {
            "DEBUG": 0,  # Verbose
            "INFO": 1,  # Information
            "WARNING": 2,  # Warning
            "ERROR": 3,  # Error
            "CRITICAL": 4,  # Critical
        }

        severity = 1  # Default to Information
        if hasattr(record, "levelname"):
            severity = severity_map.get(record.levelname, 1)

        # Build base entry
        entry = {
            self.config.time_field: timestamp,
            "Message": message,
            "SeverityLevel": severity,
            "Logger": getattr(record, "name", "unknown"),
            "Module": getattr(record, "module", None),
            "Function": getattr(record, "funcName", None),
            "Line": getattr(record, "lineno", None),
        }

        # Add cloud role if configured
        if self.config.include_cloud_role:
            entry["CloudRoleName"] = self.cloud_role_name
            entry["CloudRoleInstance"] = self.cloud_role_instance

        # Add custom properties
        custom_props = {}

        # Add context fields (prefixed with ctx_)
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key.startswith("ctx_"):
                    # Remove ctx_ prefix and add to custom properties
                    prop_name = key[4:]
                    custom_props[prop_name] = self._serialize_value(value)

        # Add any extra fields
        if hasattr(record, "extra"):
            for key, value in record.extra.items():
                if not key.startswith("_"):  # Skip private fields
                    custom_props[key] = self._serialize_value(value)

        if custom_props:
            entry["Properties"] = custom_props

        return entry

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for Azure Monitor"""
        if value is None:
            return None
        elif isinstance(value, (str, int, float, bool)):
            return value
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        else:
            return str(value)

    def _upload_logs(self, batch: List[Dict[str, Any]]) -> None:
        """Upload logs to Azure Monitor"""
        if not batch:
            return

        entries = []
        for entry in batch:
            if "record" in entry:
                log_entry = self._prepare_log_entry(entry["record"])
                entries.append(log_entry)

        if not entries:
            return

        # Route to appropriate upload method
        if self.ingestion_type == "dce":
            self._upload_via_dce(entries)
        elif self.ingestion_type == "workspace":
            self._upload_via_workspace(entries)
        elif self.ingestion_type == "appinsights":
            self._upload_via_appinsights(entries)

    def _upload_via_dce(self, entries: List[Dict[str, Any]]) -> None:
        """Upload logs via Data Collection Endpoint"""
        if not self.client:
            return

        try:
            # Upload to DCE
            self.client.upload(
                rule_id=self.config.dcr_immutable_id,
                stream_name=self.config.stream_name,
                logs=entries,
            )
        except AzureError as e:
            # Handle Azure-specific errors
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to upload logs to Azure Monitor DCE: {e}")

            # Retry on transient errors
            if hasattr(e, "status_code") and e.status_code in [429, 500, 503]:
                time.sleep(self.config.retry_delay)
                self._upload_via_dce(entries)

    def _upload_via_workspace(self, entries: List[Dict[str, Any]]) -> None:
        """Upload logs via direct Log Analytics API"""
        if not HAS_REQUESTS:
            return

        # Prepare request body
        body = json.dumps(entries)

        # Compress if configured
        if self.config.use_compression:
            body = gzip.compress(body.encode("utf-8"))
            content_type = "application/octet-stream"
        else:
            body = body.encode("utf-8")
            content_type = "application/json"

        # Create signature for authentication
        method = "POST"
        content_length = len(body)
        date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")

        # Build signature
        string_to_hash = (
            f"{method}\n{content_length}\n{content_type}\nx-ms-date:{date}\n/api/logs"
        )
        bytes_to_hash = string_to_hash.encode("utf-8")
        decoded_key = base64.b64decode(self.config.workspace_key)
        signature = base64.b64encode(
            hmac.new(decoded_key, bytes_to_hash, digestmod=hashlib.sha256).digest()
        ).decode("utf-8")

        # Prepare headers
        headers = {
            "Content-Type": content_type,
            "Authorization": f"SharedKey {self.config.workspace_id}:{signature}",
            "Log-Type": self.config.log_type,
            "x-ms-date": date,
            "time-generated-field": self.config.time_field,
        }

        if self.config.use_compression:
            headers["Content-Encoding"] = "gzip"

        # Send request
        try:
            response = requests.post(
                self.workspace_url, data=body, headers=headers, timeout=30
            )

            if response.status_code >= 400:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    f"Azure Monitor API error: {response.status_code} - {response.text}"
                )

                # Retry on server errors
                if response.status_code in [429, 500, 503]:
                    time.sleep(self.config.retry_delay)
                    self._upload_via_workspace(entries)

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to upload logs to Azure Monitor: {e}")

    def _upload_via_appinsights(self, entries: List[Dict[str, Any]]) -> None:
        """Upload logs via Application Insights"""
        if not HAS_REQUESTS:
            return

        # Convert to Application Insights telemetry format
        telemetry_items = []

        for entry in entries:
            # Create trace telemetry
            telemetry = {
                "name": "Microsoft.ApplicationInsights.Trace",
                "time": entry.get(self.config.time_field),
                "iKey": self.config.instrumentation_key,
                "tags": {
                    "ai.cloud.role": entry.get("CloudRoleName", self.cloud_role_name),
                    "ai.cloud.roleInstance": entry.get(
                        "CloudRoleInstance", self.cloud_role_instance
                    ),
                },
                "data": {
                    "baseType": "MessageData",
                    "baseData": {
                        "message": entry.get("Message", ""),
                        "severityLevel": entry.get("SeverityLevel", 1),
                        "properties": entry.get("Properties", {}),
                    },
                },
            }
            telemetry_items.append(telemetry)

        # Send to Application Insights
        try:
            response = requests.post(
                self.appinsights_url,
                json=telemetry_items,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )

            if response.status_code >= 400:
                import logging

                logger = logging.getLogger(__name__)
                logger.error(
                    f"Application Insights error: {response.status_code} - "
                    f"{response.text}"
                )

        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Failed to upload logs to Application Insights: {e}")


@dataclass
class ApplicationInsightsConfig(AzureMonitorConfig):
    """Configuration specifically for Application Insights"""

    def __post_init__(self):
        """Set defaults for Application Insights"""
        if not self.instrumentation_key:
            raise ValueError("instrumentation_key is required for Application Insights")

        # Set appropriate defaults
        self.log_type = "AppTraces"
        self.time_field = "time"


class ApplicationInsightsHandler(AzureMonitorHandler):
    """Specialized handler for Application Insights"""

    pass

