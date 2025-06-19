"""
Network logging handlers for remote logging capabilities
"""

import json
import logging
import socket
import ssl
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, List, Optional, Union
from urllib.error import HTTPError, URLError


@dataclass
class NetworkHandlerConfig:
    """Base configuration for network handlers"""
    
    # Connection settings
    host: str = "localhost"
    port: int = 514
    timeout: float = 5.0
    
    # Reliability settings
    max_retries: int = 3
    retry_delay: float = 1.0
    fallback_to_local: bool = True
    local_fallback_file: Optional[str] = "network_fallback.log"
    
    # Buffer settings
    buffer_size: int = 1024
    batch_size: int = 1
    flush_interval: float = 1.0
    
    # Security settings
    use_ssl: bool = False
    verify_ssl: bool = True
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None


@dataclass
class SyslogConfig(NetworkHandlerConfig):
    """Configuration for Syslog handler"""
    
    port: int = 514  # Standard syslog port
    facility: int = 16  # Local use facility (local0)
    rfc_format: str = "3164"  # RFC 3164 or 5424
    hostname: Optional[str] = None
    app_name: str = "python-app"
    process_id: Optional[int] = None
    
    # Message formatting
    include_timestamp: bool = True
    include_hostname: bool = True
    include_process_id: bool = True


@dataclass  
class HTTPConfig(NetworkHandlerConfig):
    """Configuration for HTTP handler"""
    
    url: str = "http://localhost:8080/logs"
    method: str = "POST"
    
    # Authentication
    auth_type: str = "none"  # none, basic, bearer, api_key
    username: Optional[str] = None
    password: Optional[str] = None
    token: Optional[str] = None
    api_key: Optional[str] = None
    api_key_header: str = "X-API-Key"
    
    # Request settings
    headers: Dict[str, str] = field(default_factory=dict)
    batch_size: int = 10
    max_batch_time: float = 5.0
    
    # Content settings
    content_type: str = "application/json"
    compress_payload: bool = True
    
    # HTTP specific
    user_agent: str = "StructuredLogging/0.6.0"


@dataclass
class SocketConfig(NetworkHandlerConfig):
    """Configuration for Socket handler (TCP/UDP)"""
    
    protocol: str = "tcp"  # tcp or udp
    port: int = 5140
    
    # TCP specific
    keep_alive: bool = True
    tcp_nodelay: bool = True
    connection_pool_size: int = 5
    
    # UDP specific  
    udp_buffer_size: int = 65507  # Max UDP payload
    
    # Message formatting
    message_delimiter: str = "\n"
    encoding: str = "utf-8"


class BaseNetworkHandler(logging.Handler):
    """Base class for all network handlers"""
    
    def __init__(self, config: NetworkHandlerConfig):
        super().__init__()
        self.config = config
        self.buffer = Queue(maxsize=1000)
        self.batch = []
        self.last_flush = time.time()
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="network-handler")
        self.running = True
        self.lock = threading.Lock()
        
        # Start background flush thread
        self.flush_thread = threading.Thread(
            target=self._flush_worker, 
            daemon=True,
            name="network-flush"
        )
        self.flush_thread.start()
        
        # Local fallback handler
        self.fallback_handler = None
        if config.fallback_to_local and config.local_fallback_file:
            self.fallback_handler = logging.FileHandler(config.local_fallback_file)
            self.fallback_handler.setFormatter(self.formatter)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Add log record to buffer for network transmission"""
        if not self.running:
            return
            
        try:
            # Format the record
            message = self.format(record)
            
            # Add to buffer (non-blocking)
            try:
                self.buffer.put_nowait({
                    'message': message,
                    'record': record,
                    'timestamp': time.time()
                })
            except:
                # Buffer full - use fallback or drop
                if self.fallback_handler:
                    self.fallback_handler.emit(record)
                # Otherwise drop the message
                
        except Exception:
            self.handleError(record)
    
    def _flush_worker(self) -> None:
        """Background worker that flushes messages"""
        while self.running:
            try:
                current_time = time.time()
                should_flush = False
                
                # Collect messages from buffer
                messages = []
                try:
                    while len(messages) < self.config.batch_size:
                        item = self.buffer.get(timeout=0.1)
                        messages.append(item)
                except Empty:
                    pass
                
                # Check if we should flush
                if messages:
                    if (len(messages) >= self.config.batch_size or 
                        current_time - self.last_flush >= self.config.flush_interval):
                        should_flush = True
                
                if should_flush and messages:
                    self.executor.submit(self._send_batch, messages)
                    self.last_flush = current_time
                elif messages:
                    # Put messages back in buffer
                    for msg in messages:
                        try:
                            self.buffer.put_nowait(msg)
                        except:
                            pass  # Buffer full, drop message
                            
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception:
                pass  # Continue running even if flush fails
    
    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send a batch of messages - implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _send_batch")
    
    def _send_with_retry(self, send_func, *args, **kwargs) -> bool:
        """Execute send function with retry logic"""
        for attempt in range(self.config.max_retries + 1):
            try:
                send_func(*args, **kwargs)
                return True
            except Exception as e:
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    # All retries failed - use fallback if available
                    if self.fallback_handler and 'record' in kwargs:
                        self.fallback_handler.emit(kwargs['record'])
                    return False
        return False
    
    def flush(self) -> None:
        """Flush any buffered messages"""
        # Flush remaining messages in buffer
        messages = []
        try:
            while True:
                item = self.buffer.get_nowait()
                messages.append(item)
        except Empty:
            pass
        
        if messages:
            self._send_batch(messages)
    
    def close(self) -> None:
        """Close the handler and clean up resources"""
        self.running = False
        
        # Flush remaining messages
        self.flush()
        
        # Wait for flush thread to finish
        if self.flush_thread.is_alive():
            self.flush_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Close fallback handler
        if self.fallback_handler:
            self.fallback_handler.close()
        
        super().close()


class SyslogHandler(BaseNetworkHandler):
    """RFC 3164/5424 compliant Syslog handler"""
    
    # Syslog facilities (RFC 3164)
    FACILITIES = {
        'kernel': 0, 'user': 1, 'mail': 2, 'daemon': 3,
        'auth': 4, 'syslog': 5, 'lpr': 6, 'news': 7,
        'uucp': 8, 'cron': 9, 'authpriv': 10, 'ftp': 11,
        'local0': 16, 'local1': 17, 'local2': 18, 'local3': 19,
        'local4': 20, 'local5': 21, 'local6': 22, 'local7': 23
    }
    
    # Syslog severities (RFC 3164)
    SEVERITIES = {
        logging.CRITICAL: 2,  # Critical
        logging.ERROR: 3,     # Error  
        logging.WARNING: 4,   # Warning
        logging.INFO: 6,      # Informational
        logging.DEBUG: 7      # Debug
    }
    
    def __init__(self, config: SyslogConfig):
        super().__init__(config)
        self.syslog_config = config
        self.socket = None
        self.hostname = getattr(config, 'hostname', None) or socket.gethostname()
        
    def _get_priority(self, record: logging.LogRecord) -> int:
        """Calculate syslog priority (facility * 8 + severity)"""
        facility = self.syslog_config.facility
        severity = self.SEVERITIES.get(record.levelno, 7)
        return facility * 8 + severity
    
    def _format_rfc3164(self, record: logging.LogRecord, message: str) -> str:
        """Format message according to RFC 3164"""
        priority = self._get_priority(record)
        
        # Timestamp in RFC 3164 format
        timestamp = datetime.fromtimestamp(record.created).strftime("%b %d %H:%M:%S")
        
        # Build message
        parts = [f"<{priority}>"]
        
        if self.syslog_config.include_timestamp:
            parts.append(timestamp)
            
        if self.syslog_config.include_hostname:
            parts.append(self.hostname)
            
        # Tag (app_name with optional PID)
        tag = self.syslog_config.app_name
        if self.syslog_config.include_process_id and self.syslog_config.process_id:
            tag += f"[{self.syslog_config.process_id}]"
        parts.append(f"{tag}:")
        
        parts.append(message)
        
        return " ".join(parts)
    
    def _format_rfc5424(self, record: logging.LogRecord, message: str) -> str:
        """Format message according to RFC 5424"""
        priority = self._get_priority(record)
        
        # Timestamp in ISO 8601 format
        timestamp = datetime.fromtimestamp(record.created).isoformat() + "Z"
        
        # Structured data (can be extended)
        structured_data = "-"  # NILVALUE for now
        
        # Build message parts
        version = "1"
        hostname = self.hostname if self.syslog_config.include_hostname else "-"
        app_name = self.syslog_config.app_name or "-"
        proc_id = str(self.syslog_config.process_id) if self.syslog_config.process_id else "-"
        msg_id = "-"  # Could be record.name or similar
        
        header = f"<{priority}>{version} {timestamp} {hostname} {app_name} {proc_id} {msg_id} {structured_data}"
        
        # BOM + message
        return f"{header} \ufeff{message}"
    
    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send batch of syslog messages"""
        if not messages:
            return
            
        def send_messages():
            sock = None
            try:
                # Create socket
                if self.syslog_config.use_ssl:
                    context = ssl.create_default_context()
                    if not self.syslog_config.verify_ssl:
                        context.check_hostname = False
                        context.verify_mode = ssl.CERT_NONE
                    
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock = context.wrap_socket(sock, server_hostname=self.config.host)
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                
                sock.settimeout(self.config.timeout)
                
                if sock.type == socket.SOCK_STREAM:  # TCP
                    sock.connect((self.config.host, self.config.port))
                
                # Send each message
                for msg_info in messages:
                    record = msg_info['record']
                    message = msg_info['message']
                    
                    # Format according to RFC
                    if self.syslog_config.rfc_format == "5424":
                        syslog_message = self._format_rfc5424(record, message)
                    else:
                        syslog_message = self._format_rfc3164(record, message)
                    
                    # Encode and send
                    data = syslog_message.encode('utf-8')
                    
                    if sock.type == socket.SOCK_STREAM:  # TCP
                        sock.sendall(data + b'\n')
                    else:  # UDP
                        sock.sendto(data, (self.config.host, self.config.port))
                        
            finally:
                if sock:
                    sock.close()
        
        self._send_with_retry(send_messages)


class HTTPHandler(BaseNetworkHandler):
    """HTTP handler for sending logs to REST APIs"""
    
    def __init__(self, config: HTTPConfig):
        super().__init__(config)
        self.http_config = config
        self._setup_auth()
    
    def _setup_auth(self) -> None:
        """Setup authentication headers"""
        self.auth_headers = {}
        
        if self.http_config.auth_type == "bearer" and self.http_config.token:
            self.auth_headers["Authorization"] = f"Bearer {self.http_config.token}"
        elif self.http_config.auth_type == "api_key" and self.http_config.api_key:
            self.auth_headers[self.http_config.api_key_header] = self.http_config.api_key
        elif self.http_config.auth_type == "basic" and self.http_config.username:
            import base64
            credentials = f"{self.http_config.username}:{self.http_config.password or ''}"
            encoded = base64.b64encode(credentials.encode()).decode()
            self.auth_headers["Authorization"] = f"Basic {encoded}"
    
    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send batch of messages via HTTP"""
        if not messages:
            return
            
        def send_http_request():
            # Prepare payload
            if self.http_config.batch_size == 1 and len(messages) == 1:
                # Single message
                payload = {
                    "message": messages[0]["message"],
                    "timestamp": messages[0]["timestamp"],
                    "level": messages[0]["record"].levelname,
                    "logger": messages[0]["record"].name
                }
            else:
                # Batch of messages
                payload = {
                    "logs": [
                        {
                            "message": msg["message"],
                            "timestamp": msg["timestamp"], 
                            "level": msg["record"].levelname,
                            "logger": msg["record"].name
                        }
                        for msg in messages
                    ],
                    "batch_size": len(messages)
                }
            
            # Convert to JSON
            data = json.dumps(payload).encode('utf-8')
            
            # Prepare headers
            headers = {
                "Content-Type": self.http_config.content_type,
                "User-Agent": self.http_config.user_agent,
                "Content-Length": str(len(data))
            }
            headers.update(self.http_config.headers)
            headers.update(self.auth_headers)
            
            # Create request
            request = urllib.request.Request(
                self.http_config.url,
                data=data,
                headers=headers,
                method=self.http_config.method
            )
            
            # Send request
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                if response.status >= 400:
                    raise HTTPError(
                        self.http_config.url, 
                        response.status, 
                        f"HTTP {response.status}", 
                        response.headers, 
                        None
                    )
        
        self._send_with_retry(send_http_request)


class SocketHandler(BaseNetworkHandler):
    """Raw socket handler for TCP/UDP logging"""
    
    def __init__(self, config: SocketConfig):
        super().__init__(config)
        self.socket_config = config
        self.connections = Queue(maxsize=config.connection_pool_size)
        self._init_connection_pool()
    
    def _init_connection_pool(self) -> None:
        """Initialize connection pool for TCP"""
        if self.socket_config.protocol.lower() == "tcp":
            for _ in range(self.socket_config.connection_pool_size):
                try:
                    sock = self._create_socket()
                    self.connections.put_nowait(sock)
                except:
                    pass  # Pool will be smaller
    
    def _create_socket(self) -> socket.socket:
        """Create and configure a socket"""
        if self.socket_config.protocol.lower() == "tcp":
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            if self.socket_config.tcp_nodelay:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.socket_config.keep_alive:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                
            sock.settimeout(self.config.timeout)
            sock.connect((self.config.host, self.config.port))
            return sock
        else:  # UDP
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(self.config.timeout)
            return sock
    
    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send batch of messages via raw socket"""
        if not messages:
            return
            
        def send_socket_messages():
            sock = None
            is_tcp = self.socket_config.protocol.lower() == "tcp"
            
            try:
                # Get socket from pool (TCP) or create new (UDP)
                if is_tcp:
                    try:
                        sock = self.connections.get_nowait()
                    except Empty:
                        sock = self._create_socket()
                else:
                    sock = self._create_socket()
                
                # Send messages
                for msg_info in messages:
                    message = msg_info["message"]
                    data = message.encode(self.socket_config.encoding)
                    
                    if self.socket_config.message_delimiter:
                        data += self.socket_config.message_delimiter.encode(
                            self.socket_config.encoding
                        )
                    
                    if is_tcp:
                        sock.sendall(data)
                    else:
                        # Check UDP size limit
                        if len(data) > self.socket_config.udp_buffer_size:
                            data = data[:self.socket_config.udp_buffer_size]
                        sock.sendto(data, (self.config.host, self.config.port))
                
                # Return TCP socket to pool
                if is_tcp:
                    try:
                        self.connections.put_nowait(sock)
                        sock = None  # Don't close it
                    except:
                        pass  # Pool full, will close socket
                        
            finally:
                if sock and (not is_tcp or sock):
                    sock.close()
        
        self._send_with_retry(send_socket_messages)
    
    def close(self) -> None:
        """Close handler and connection pool"""
        # Close all pooled connections
        while True:
            try:
                sock = self.connections.get_nowait()
                if sock and hasattr(sock, 'close'):
                    sock.close()
            except (Empty, Exception):
                break
        
        super().close()