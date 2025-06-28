"""
AWS CloudWatch Logs integration for structured logging
"""

import json
import time
import gzip
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
    HAS_BOTO3 = True
except ImportError:
    boto3 = None
    ClientError = Exception
    BotoCoreError = Exception
    HAS_BOTO3 = False

from .base import CloudHandlerConfig, CloudLogHandler


@dataclass
class CloudWatchConfig(CloudHandlerConfig):
    """Configuration for AWS CloudWatch Logs handler"""
    
    # CloudWatch specific settings
    log_group: str = "/aws/application/structured-logging"
    log_stream: Optional[str] = None  # Auto-generated if not provided
    create_log_group: bool = True
    create_log_stream: bool = True
    
    # AWS credentials (optional - uses boto3 credential chain if not provided)
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_session_token: Optional[str] = None
    
    # Performance tuning
    max_batch_size: int = 1048576  # 1MB - CloudWatch limit
    max_batch_count: int = 10000   # CloudWatch limit
    
    def __post_init__(self):
        """Generate log stream name if not provided"""
        if not self.log_stream:
            # Use hostname and timestamp for unique stream name
            import socket
            hostname = socket.gethostname()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.log_stream = f"{hostname}_{timestamp}"


class CloudWatchHandler(CloudLogHandler):
    """
    AWS CloudWatch Logs handler for structured logging
    
    Features:
    - Automatic log group/stream creation
    - Batch uploading with size limits
    - Compression support
    - Sequence token management
    - Retry logic with exponential backoff
    """
    
    def __init__(self, config: CloudWatchConfig):
        if not HAS_BOTO3:
            raise ImportError(
                "boto3 is required for CloudWatch integration. "
                "Install with: pip install structured-logging[aws]"
            )
        
        super().__init__(config)
        self.config: CloudWatchConfig = config
        
        # Initialize AWS client
        self._init_client()
        
        # Ensure log group and stream exist
        self._ensure_log_infrastructure()
        
        # Sequence token for CloudWatch
        self._sequence_token = None
    
    def _init_client(self) -> None:
        """Initialize boto3 CloudWatch Logs client"""
        # Build credentials dict if provided
        credentials = {}
        if self.config.aws_access_key_id:
            credentials['aws_access_key_id'] = self.config.aws_access_key_id
        if self.config.aws_secret_access_key:
            credentials['aws_secret_access_key'] = self.config.aws_secret_access_key
        if self.config.aws_session_token:
            credentials['aws_session_token'] = self.config.aws_session_token
        if self.config.region:
            credentials['region_name'] = self.config.region
        
        self.client = boto3.client('logs', **credentials)
    
    def _ensure_log_infrastructure(self) -> None:
        """Ensure log group and stream exist"""
        # Create log group if needed
        if self.config.create_log_group:
            try:
                self.client.create_log_group(logGroupName=self.config.log_group)
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    raise
        
        # Create log stream if needed
        if self.config.create_log_stream:
            try:
                self.client.create_log_stream(
                    logGroupName=self.config.log_group,
                    logStreamName=self.config.log_stream
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    raise
    
    def _prepare_log_events(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare log events for CloudWatch format"""
        events = []
        total_size = 0
        
        for entry in batch:
            # CloudWatch expects millisecond timestamps
            timestamp = int(entry['timestamp'] * 1000)
            
            # Format message
            if isinstance(entry['message'], str):
                message = entry['message']
            else:
                # Already formatted as dict/JSON
                message = json.dumps(entry['message'])
            
            # Add CloudWatch metadata
            if hasattr(entry.get('record'), '__dict__'):
                record = entry['record']
                # Add useful fields from LogRecord
                log_data = {
                    'message': message,
                    'level': record.levelname,
                    'logger': record.name,
                    'module': record.module,
                    'funcName': record.funcName,
                    'lineno': record.lineno,
                }
                
                # Add any context fields (prefixed with ctx_)
                for key, value in record.__dict__.items():
                    if key.startswith('ctx_'):
                        log_data[key[4:]] = value
                
                message = json.dumps(log_data)
            
            # Check size limits
            event_size = len(message) + 26  # CloudWatch overhead
            if total_size + event_size > self.config.max_batch_size:
                break
            
            events.append({
                'timestamp': timestamp,
                'message': message
            })
            
            total_size += event_size
            
            # Check count limit
            if len(events) >= self.config.max_batch_count:
                break
        
        # Sort by timestamp (CloudWatch requirement)
        events.sort(key=lambda x: x['timestamp'])
        
        return events
    
    def _upload_logs(self, batch: List[Dict[str, Any]]) -> None:
        """Upload logs to CloudWatch"""
        if not batch:
            return
        
        events = self._prepare_log_events(batch)
        if not events:
            return
        
        # Build request parameters
        params = {
            'logGroupName': self.config.log_group,
            'logStreamName': self.config.log_stream,
            'logEvents': events
        }
        
        # Add sequence token if we have one
        if self._sequence_token:
            params['sequenceToken'] = self._sequence_token
        
        try:
            response = self.client.put_log_events(**params)
            
            # Update sequence token for next batch
            self._sequence_token = response.get('nextSequenceToken')
            
            # Check for rejected events
            rejected = response.get('rejectedLogEventsInfo', {})
            if rejected:
                # Log warning about rejected events
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(
                    f"CloudWatch rejected {len(rejected)} log events: {rejected}"
                )
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            
            # Handle specific errors
            if error_code == 'InvalidSequenceTokenException':
                # Extract correct sequence token and retry
                import re
                match = re.search(r'sequenceToken is: (\S+)', str(e))
                if match:
                    self._sequence_token = match.group(1)
                    # Retry with correct token
                    params['sequenceToken'] = self._sequence_token
                    response = self.client.put_log_events(**params)
                    self._sequence_token = response.get('nextSequenceToken')
                else:
                    raise
            
            elif error_code == 'DataAlreadyAcceptedException':
                # Update sequence token from error message
                import re
                match = re.search(r'sequenceToken is: (\S+)', str(e))
                if match:
                    self._sequence_token = match.group(1)
            else:
                raise
    
    def _compress_batch(self, batch: List[Dict[str, Any]]) -> bytes:
        """Compress batch for more efficient upload"""
        if not self.config.compress_logs:
            return json.dumps(batch).encode('utf-8')
        
        return gzip.compress(json.dumps(batch).encode('utf-8'))
    
    def flush(self) -> None:
        """Flush any pending logs to CloudWatch"""
        super().flush()
        
        # Force a final upload of any remaining logs
        remaining = []
        while not self.queue.empty():
            try:
                remaining.append(self.queue.get_nowait())
            except:
                break
        
        if remaining:
            self._upload_batch(remaining)