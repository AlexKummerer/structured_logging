"""
Google Cloud Logging integration for structured logging
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

try:
    from google.cloud import logging as gcp_logging
    from google.cloud.logging_v2.handlers import CloudLoggingHandler as GCPHandler
    from google.api_core import retry
    from google.api_core.exceptions import GoogleAPIError
    HAS_GOOGLE_CLOUD = True
except ImportError:
    gcp_logging = None
    GCPHandler = None
    retry = None
    GoogleAPIError = Exception
    HAS_GOOGLE_CLOUD = False

from .base import CloudHandlerConfig, CloudLogHandler


@dataclass
class GoogleCloudConfig(CloudHandlerConfig):
    """Configuration for Google Cloud Logging handler"""
    
    # Google Cloud specific settings
    project_id: Optional[str] = None  # Auto-detected if not provided
    log_name: str = "structured-logging"
    resource_type: str = "global"
    resource_labels: Optional[Dict[str, str]] = None
    
    # Authentication
    credentials_path: Optional[str] = None  # Path to service account JSON
    
    # Performance tuning
    use_background_thread: bool = True  # Use background thread for uploads
    grace_period: float = 5.0  # Seconds to wait for pending logs on shutdown
    
    # Structured logging
    use_structured_logging: bool = True  # Send as structured JSON
    include_trace_id: bool = True  # Include trace ID if available
    
    def __post_init__(self):
        """Initialize resource labels if not provided"""
        if self.resource_labels is None:
            self.resource_labels = {}


class GoogleCloudHandler(CloudLogHandler):
    """
    Google Cloud Logging handler for structured logging
    
    Features:
    - Automatic project detection
    - Structured JSON logging
    - Resource type and labels support
    - Trace ID correlation
    - Background thread for async uploads
    - Automatic retry with exponential backoff
    """
    
    def __init__(self, config: GoogleCloudConfig):
        if not HAS_GOOGLE_CLOUD:
            raise ImportError(
                "google-cloud-logging is required for Google Cloud integration. "
                "Install with: pip install structured-logging[gcp]"
            )
        
        super().__init__(config)
        self.config: GoogleCloudConfig = config
        
        # Initialize Google Cloud client
        self._init_client()
        
        # Create logger
        self._create_logger()
        
        # Track background transport if used
        self._transport = None
    
    def _init_client(self) -> None:
        """Initialize Google Cloud Logging client"""
        # Build client kwargs
        kwargs = {}
        
        if self.config.project_id:
            kwargs['project'] = self.config.project_id
        
        # Handle credentials
        if self.config.credentials_path:
            # Use service account credentials
            from google.oauth2 import service_account
            credentials = service_account.Credentials.from_service_account_file(
                self.config.credentials_path
            )
            kwargs['credentials'] = credentials
        
        # Create client
        self.client = gcp_logging.Client(**kwargs)
    
    def _create_logger(self) -> None:
        """Create Google Cloud logger instance"""
        self.logger = self.client.logger(self.config.log_name)
        
        # Create resource descriptor
        self.resource = gcp_logging.Resource(
            type=self.config.resource_type,
            labels=self.config.resource_labels
        )
    
    def _prepare_log_entry(self, record: Any) -> Dict[str, Any]:
        """Prepare log entry for Google Cloud format"""
        # Extract message and level
        message = record.getMessage() if hasattr(record, 'getMessage') else str(record)
        
        # Map Python log levels to Google Cloud severity
        severity_map = {
            'DEBUG': 'DEBUG',
            'INFO': 'INFO',
            'WARNING': 'WARNING',
            'ERROR': 'ERROR',
            'CRITICAL': 'CRITICAL'
        }
        
        severity = 'DEFAULT'
        if hasattr(record, 'levelname'):
            severity = severity_map.get(record.levelname, 'DEFAULT')
        
        # Build structured payload if enabled
        if self.config.use_structured_logging:
            payload = {
                'message': message,
                'logger': getattr(record, 'name', 'unknown'),
                'module': getattr(record, 'module', None),
                'funcName': getattr(record, 'funcName', None),
                'lineno': getattr(record, 'lineno', None),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            }
            
            # Add context fields (prefixed with ctx_)
            if hasattr(record, '__dict__'):
                for key, value in record.__dict__.items():
                    if key.startswith('ctx_'):
                        # Remove ctx_ prefix for cleaner logs
                        clean_key = key[4:]
                        payload[clean_key] = value
            
            # Add any extra fields
            if hasattr(record, 'extra'):
                payload.update(record.extra)
            
            return {
                'json_payload': payload,
                'severity': severity,
                'resource': self.resource
            }
        else:
            # Simple text logging
            return {
                'text_payload': message,
                'severity': severity,
                'resource': self.resource
            }
    
    def _upload_logs(self, batch: List[Dict[str, Any]]) -> None:
        """Upload logs to Google Cloud Logging"""
        if not batch:
            return
        
        entries = []
        for entry in batch:
            if 'record' in entry:
                log_entry = self._prepare_log_entry(entry['record'])
                
                # Add trace ID if available
                if self.config.include_trace_id and hasattr(entry['record'], 'trace_id'):
                    log_entry['trace'] = entry['record'].trace_id
                
                entries.append(log_entry)
        
        if not entries:
            return
        
        # Write entries with retry
        try:
            if len(entries) == 1:
                # Single entry
                entry = entries[0]
                if 'json_payload' in entry:
                    self.logger.log_struct(
                        entry['json_payload'],
                        severity=entry['severity'],
                        resource=entry['resource']
                    )
                else:
                    self.logger.log_text(
                        entry['text_payload'],
                        severity=entry['severity'],
                        resource=entry['resource']
                    )
            else:
                # Batch write
                batch_entries = []
                for entry in entries:
                    if 'json_payload' in entry:
                        log_entry = self.logger.entry(
                            payload=entry['json_payload'],
                            severity=entry['severity'],
                            resource=entry['resource']
                        )
                    else:
                        log_entry = self.logger.entry(
                            payload=entry['text_payload'],
                            severity=entry['severity'],
                            resource=entry['resource']
                        )
                    batch_entries.append(log_entry)
                
                # Write batch
                self.logger.log_batch(batch_entries)
        
        except GoogleAPIError as e:
            # Handle specific Google Cloud errors
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to upload logs to Google Cloud: {e}")
            
            # Retry if transient error
            if hasattr(e, 'code') and e.code in [500, 503, 504]:
                # Server errors - retry with backoff
                time.sleep(self.config.retry_delay)
                self._upload_logs(batch)
    
    def emit(self, record) -> None:
        """Emit a log record to Google Cloud"""
        if self.config.use_background_thread:
            # Use Google Cloud's background transport
            if self._transport is None:
                from google.cloud.logging_v2.handlers.transports import BackgroundThreadTransport
                self._transport = BackgroundThreadTransport(
                    self.client,
                    self.config.log_name,
                    grace_period=self.config.grace_period
                )
            
            # Send via background thread
            self._transport.send(
                record,
                message=record.getMessage() if hasattr(record, 'getMessage') else str(record),
                resource=self.resource
            )
        else:
            # Use base class queue-based approach
            super().emit(record)
    
    def flush(self) -> None:
        """Flush any pending logs"""
        if self._transport:
            self._transport.flush()
        else:
            super().flush()
    
    def close(self) -> None:
        """Close the handler and flush remaining logs"""
        if self._transport:
            self._transport.close()
        super().close()


class StackdriverHandler(GoogleCloudHandler):
    """Alias for Google Cloud Logging (formerly Stackdriver)"""
    pass


@dataclass 
class StackdriverConfig(GoogleCloudConfig):
    """Alias for Google Cloud Logging config (formerly Stackdriver)"""
    pass