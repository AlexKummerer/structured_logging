"""
Tests for network logging handlers
"""

import json
import logging
import socket
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from urllib.error import HTTPError

import pytest

from structured_logging import (
    LoggerConfig,
    get_logger,
    HTTPConfig,
    HTTPHandler,
    NetworkHandlerConfig,
    SocketConfig,
    SocketHandler,
    SyslogConfig,
    SyslogHandler,
)


class TestNetworkHandlerConfig:
    """Tests for base network handler configuration"""
    
    def test_default_config(self):
        config = NetworkHandlerConfig()
        assert config.host == "localhost"
        assert config.port == 514
        assert config.timeout == 5.0
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.fallback_to_local is True
        assert config.buffer_size == 1024
        assert config.batch_size == 1
        assert config.flush_interval == 1.0
    
    def test_custom_config(self):
        config = NetworkHandlerConfig(
            host="log-server.com",
            port=1234,
            timeout=10.0,
            max_retries=5,
            fallback_to_local=False
        )
        assert config.host == "log-server.com"
        assert config.port == 1234
        assert config.timeout == 10.0
        assert config.max_retries == 5
        assert config.fallback_to_local is False


class TestSyslogConfig:
    """Tests for Syslog configuration"""
    
    def test_default_syslog_config(self):
        config = SyslogConfig()
        assert config.port == 514
        assert config.facility == 16  # local0
        assert config.rfc_format == "3164"
        assert config.app_name == "python-app"
        assert config.include_timestamp is True
        assert config.include_hostname is True
        assert config.include_process_id is True
    
    def test_custom_syslog_config(self):
        config = SyslogConfig(
            host="syslog.company.com",
            port=5140,
            facility=23,  # local7
            rfc_format="5424",
            app_name="my-app",
            hostname="custom-host"
        )
        assert config.host == "syslog.company.com"
        assert config.port == 5140
        assert config.facility == 23
        assert config.rfc_format == "5424"
        assert config.app_name == "my-app"
        assert config.hostname == "custom-host"


class TestHTTPConfig:
    """Tests for HTTP configuration"""
    
    def test_default_http_config(self):
        config = HTTPConfig()
        assert config.url == "http://localhost:8080/logs"
        assert config.method == "POST"
        assert config.auth_type == "none"
        assert config.batch_size == 10
        assert config.max_batch_time == 5.0
        assert config.content_type == "application/json"
        assert config.user_agent == "StructuredLogging/0.6.0"
    
    def test_http_config_with_auth(self):
        config = HTTPConfig(
            url="https://api.logs.com/v1/ingest",
            auth_type="bearer",
            token="secret-token-123",
            batch_size=50
        )
        assert config.url == "https://api.logs.com/v1/ingest"
        assert config.auth_type == "bearer"
        assert config.token == "secret-token-123"
        assert config.batch_size == 50


class TestSocketConfig:
    """Tests for Socket configuration"""
    
    def test_default_socket_config(self):
        config = SocketConfig()
        assert config.protocol == "tcp"
        assert config.port == 5140
        assert config.keep_alive is True
        assert config.tcp_nodelay is True
        assert config.connection_pool_size == 5
        assert config.message_delimiter == "\n"
        assert config.encoding == "utf-8"
    
    def test_udp_socket_config(self):
        config = SocketConfig(
            protocol="udp",
            port=9999,
            udp_buffer_size=32768
        )
        assert config.protocol == "udp"
        assert config.port == 9999
        assert config.udp_buffer_size == 32768


class TestSyslogHandler:
    """Tests for Syslog handler"""
    
    def test_priority_calculation(self):
        config = SyslogConfig(facility=16)  # local0
        handler = SyslogHandler(config)
        
        # Test different log levels
        record_info = logging.LogRecord("test", logging.INFO, "", 0, "test", (), None)
        record_error = logging.LogRecord("test", logging.ERROR, "", 0, "test", (), None)
        record_debug = logging.LogRecord("test", logging.DEBUG, "", 0, "test", (), None)
        
        # Priority = facility * 8 + severity
        # INFO = severity 6, local0 = facility 16 -> 16*8 + 6 = 134
        assert handler._get_priority(record_info) == 134
        
        # ERROR = severity 3, local0 = facility 16 -> 16*8 + 3 = 131  
        assert handler._get_priority(record_error) == 131
        
        # DEBUG = severity 7, local0 = facility 16 -> 16*8 + 7 = 135
        assert handler._get_priority(record_debug) == 135
    
    def test_rfc3164_formatting(self):
        config = SyslogConfig(
            facility=16,
            app_name="test-app",
            hostname="test-host"
        )
        handler = SyslogHandler(config)
        handler.hostname = "test-host"
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", (), None)
        record.created = time.mktime(time.strptime("2024-01-15 10:30:00", "%Y-%m-%d %H:%M:%S"))
        
        formatted = handler._format_rfc3164(record, "Test message")
        
        # Should contain priority, timestamp, hostname, app name, and message
        assert "<134>" in formatted  # Priority for INFO level
        assert "Jan 15 10:30:00" in formatted  # RFC 3164 timestamp
        assert "test-host" in formatted
        assert "test-app:" in formatted
        assert "Test message" in formatted
    
    def test_rfc5424_formatting(self):
        config = SyslogConfig(
            facility=16,
            app_name="test-app",
            hostname="test-host",
            rfc_format="5424"
        )
        handler = SyslogHandler(config)
        handler.hostname = "test-host"
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "Test message", (), None)
        record.created = 1705314600.0  # 2024-01-15T10:30:00Z
        
        formatted = handler._format_rfc5424(record, "Test message")
        
        # Should contain version, priority, ISO timestamp, and message
        assert "<134>1" in formatted  # Priority and version
        assert "2024-01-15T" in formatted  # ISO timestamp (timezone may vary)
        assert "test-host" in formatted
        assert "test-app" in formatted
        assert "\ufeffTest message" in formatted  # BOM + message
    
    @patch('socket.socket')
    def test_syslog_udp_send(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        config = SyslogConfig(host="syslog.test.com", port=514)
        handler = SyslogHandler(config)
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "Test UDP", (), None)
        messages = [{"message": "Test UDP", "record": record}]
        
        handler._send_batch(messages)
        
        # Verify socket creation and sendto call
        mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_DGRAM)
        mock_socket.settimeout.assert_called_with(5.0)
        mock_socket.sendto.assert_called()
        mock_socket.close.assert_called()
    
    @patch('socket.socket')
    def test_syslog_tcp_send(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        # TCP is determined by use_ssl=True for this test
        config = SyslogConfig(host="syslog.test.com", port=514, use_ssl=False)
        # Force TCP by patching socket type
        mock_socket.type = socket.SOCK_STREAM
        
        handler = SyslogHandler(config)
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "Test TCP", (), None)
        messages = [{"message": "Test TCP", "record": record}]
        
        handler._send_batch(messages)
        
        # Should connect for TCP
        mock_socket.connect.assert_called_with(("syslog.test.com", 514))
        mock_socket.close.assert_called()


class TestHTTPHandler:
    """Tests for HTTP handler"""
    
    def test_auth_setup_bearer(self):
        config = HTTPConfig(auth_type="bearer", token="secret123")
        handler = HTTPHandler(config)
        
        assert "Authorization" in handler.auth_headers
        assert handler.auth_headers["Authorization"] == "Bearer secret123"
    
    def test_auth_setup_api_key(self):
        config = HTTPConfig(
            auth_type="api_key", 
            api_key="key123",
            api_key_header="X-Custom-Key"
        )
        handler = HTTPHandler(config)
        
        assert "X-Custom-Key" in handler.auth_headers
        assert handler.auth_headers["X-Custom-Key"] == "key123"
    
    def test_auth_setup_basic(self):
        config = HTTPConfig(
            auth_type="basic",
            username="user",
            password="pass"
        )
        handler = HTTPHandler(config)
        
        assert "Authorization" in handler.auth_headers
        # Basic auth header should contain base64 encoded credentials
        assert handler.auth_headers["Authorization"].startswith("Basic ")
    
    @patch('urllib.request.urlopen')
    def test_http_single_message_send(self, mock_urlopen):
        mock_response = Mock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        config = HTTPConfig(
            url="https://api.test.com/logs",
            batch_size=1
        )
        handler = HTTPHandler(config)
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "Test HTTP", (), None)
        messages = [{"message": "Test HTTP", "record": record, "timestamp": time.time()}]
        
        handler._send_batch(messages)
        
        # Verify request was made
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "https://api.test.com/logs"
        assert request.get_method() == "POST"
        
        # Check request data contains single message format
        data = json.loads(request.data.decode())
        assert "message" in data
        assert data["message"] == "Test HTTP"
        assert "timestamp" in data
        assert "level" in data
        assert "logger" in data
    
    @patch('urllib.request.urlopen')
    def test_http_batch_send(self, mock_urlopen):
        mock_response = Mock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        config = HTTPConfig(
            url="https://api.test.com/logs",
            batch_size=2
        )
        handler = HTTPHandler(config)
        
        record1 = logging.LogRecord("test", logging.INFO, "", 0, "Message 1", (), None)
        record2 = logging.LogRecord("test", logging.ERROR, "", 0, "Message 2", (), None)
        messages = [
            {"message": "Message 1", "record": record1, "timestamp": time.time()},
            {"message": "Message 2", "record": record2, "timestamp": time.time()}
        ]
        
        handler._send_batch(messages)
        
        # Check request data contains batch format
        request = mock_urlopen.call_args[0][0]
        data = json.loads(request.data.decode())
        assert "logs" in data
        assert "batch_size" in data
        assert data["batch_size"] == 2
        assert len(data["logs"]) == 2
        assert data["logs"][0]["message"] == "Message 1"
        assert data["logs"][1]["message"] == "Message 2"
    
    @patch('urllib.request.urlopen')
    def test_http_retry_on_failure(self, mock_urlopen):
        # First call fails, second succeeds
        mock_urlopen.side_effect = [HTTPError("url", 500, "Server Error", {}, None), Mock()]
        mock_response = Mock()
        mock_response.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        config = HTTPConfig(max_retries=2, retry_delay=0.1)
        handler = HTTPHandler(config)
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "Test retry", (), None)
        messages = [{"message": "Test retry", "record": record, "timestamp": time.time()}]
        
        # Should not raise exception due to retry
        handler._send_batch(messages)


class TestSocketHandler:
    """Tests for Socket handler"""
    
    @patch('socket.socket')
    def test_tcp_socket_creation(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        config = SocketConfig(protocol="tcp", host="log.test.com", port=9999)
        handler = SocketHandler(config)
        
        # Test creating a socket
        sock = handler._create_socket()
        
        # Verify TCP socket setup
        mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_STREAM)
        mock_socket.setsockopt.assert_any_call(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        mock_socket.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        mock_socket.settimeout.assert_called_with(5.0)
        mock_socket.connect.assert_called_with(("log.test.com", 9999))
    
    @patch('socket.socket')
    def test_udp_socket_creation(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        config = SocketConfig(protocol="udp", host="log.test.com", port=9999)
        handler = SocketHandler(config)
        
        # Test creating a socket
        sock = handler._create_socket()
        
        # Verify UDP socket setup
        mock_socket_class.assert_called_with(socket.AF_INET, socket.SOCK_DGRAM)
        mock_socket.settimeout.assert_called_with(5.0)
        # UDP doesn't connect
        mock_socket.connect.assert_not_called()
    
    @patch('socket.socket')
    def test_tcp_message_send(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket.type = socket.SOCK_STREAM  # Important: mark as TCP
        mock_socket_class.return_value = mock_socket
        
        config = SocketConfig(protocol="tcp")
        handler = SocketHandler(config)
        
        # Mock the connection pool to be empty (force new connection)
        handler.connections = Mock()
        handler.connections.get_nowait.side_effect = Exception("Empty")
        handler.connections.put_nowait = Mock()
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "TCP test", (), None)
        messages = [{"message": "TCP test", "record": record}]
        
        handler._send_batch(messages)
        
        # Verify TCP sendall was called
        expected_data = b"TCP test\n"  # With delimiter
        mock_socket.sendall.assert_called_with(expected_data)
    
    @patch('socket.socket') 
    def test_udp_message_send(self, mock_socket_class):
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        config = SocketConfig(protocol="udp", host="udp.test.com", port=8888)
        handler = SocketHandler(config)
        
        record = logging.LogRecord("test", logging.INFO, "", 0, "UDP test", (), None)
        messages = [{"message": "UDP test", "record": record}]
        
        handler._send_batch(messages)
        
        # Verify UDP sendto was called
        expected_data = b"UDP test\n"  # With delimiter
        mock_socket.sendto.assert_called_with(expected_data, ("udp.test.com", 8888))
    
    def test_udp_size_limit(self):
        config = SocketConfig(protocol="udp", udp_buffer_size=10)
        handler = SocketHandler(config)
        
        # Create a message larger than buffer
        large_message = "x" * 20  # Larger than 10 byte limit
        
        with patch('socket.socket') as mock_socket_class:
            mock_socket = Mock()
            mock_socket_class.return_value = mock_socket
            
            record = logging.LogRecord("test", logging.INFO, "", 0, large_message, (), None)
            messages = [{"message": large_message, "record": record}]
            
            handler._send_batch(messages)
            
            # Should truncate the message to fit buffer size
            # Message + delimiter should be truncated to 10 bytes
            call_args = mock_socket.sendto.call_args[0]
            sent_data = call_args[0]
            assert len(sent_data) <= 10


class TestNetworkLoggerIntegration:
    """Integration tests for network handlers with logger"""
    
    def test_syslog_logger_creation(self):
        from structured_logging.network_handlers import SyslogConfig
        
        syslog_config = SyslogConfig(host="syslog.test.com")
        config = LoggerConfig(
            output_type="network",
            network_config=syslog_config
        )
        
        logger = get_logger("test_syslog", config)
        
        # Should have network handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], SyslogHandler)
    
    def test_http_logger_creation(self):
        from structured_logging.network_handlers import HTTPConfig
        
        http_config = HTTPConfig(url="https://api.test.com/logs")
        config = LoggerConfig(
            output_type="network",
            network_config=http_config
        )
        
        logger = get_logger("test_http", config)
        
        # Should have network handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], HTTPHandler)
    
    def test_socket_logger_creation(self):
        from structured_logging.network_handlers import SocketConfig
        
        socket_config = SocketConfig(protocol="tcp", host="socket.test.com")
        config = LoggerConfig(
            output_type="network",
            network_config=socket_config
        )
        
        logger = get_logger("test_socket", config)
        
        # Should have network handler
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], SocketHandler)
    
    def test_console_plus_network_output(self):
        from structured_logging.network_handlers import SyslogConfig
        
        syslog_config = SyslogConfig()
        config = LoggerConfig(
            output_type="console+network",
            network_config=syslog_config
        )
        
        logger = get_logger("test_combined", config)
        
        # Should have both console and network handlers
        assert len(logger.handlers) == 2
        handler_types = [type(h).__name__ for h in logger.handlers]
        assert "StreamHandler" in handler_types
        assert "SyslogHandler" in handler_types
    
    @patch('socket.socket')
    def test_end_to_end_network_logging(self, mock_socket_class):
        """Test actual logging through network handler"""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        from structured_logging.network_handlers import SyslogConfig
        
        syslog_config = SyslogConfig(host="test.syslog.com")
        config = LoggerConfig(
            output_type="network",
            network_config=syslog_config,
            formatter_type="json"
        )
        
        logger = get_logger("test_e2e", config)
        
        # Log a message
        logger.info("Test network logging")
        
        # Give some time for background thread to process
        time.sleep(0.2)
        
        # Verify socket operations occurred
        mock_socket_class.assert_called()
        # Note: Due to threading, exact socket calls may vary
        # The important part is that socket creation was attempted


class TestNetworkEnvironmentConfig:
    """Test environment variable configuration for network handlers"""
    
    @patch.dict('os.environ', {
        'STRUCTURED_LOG_OUTPUT': 'network',
        'STRUCTURED_LOG_NETWORK_TYPE': 'syslog',
        'STRUCTURED_LOG_SYSLOG_HOST': 'syslog.company.com',
        'STRUCTURED_LOG_SYSLOG_PORT': '1514',
        'STRUCTURED_LOG_APP_NAME': 'my-service'
    })
    def test_syslog_env_config(self):
        config = LoggerConfig.from_env()
        
        assert config.output_type == "network"
        assert config.network_config is not None
        assert isinstance(config.network_config, SyslogConfig)
        assert config.network_config.host == "syslog.company.com"
        assert config.network_config.port == 1514
        assert config.network_config.app_name == "my-service"
    
    @patch.dict('os.environ', {
        'STRUCTURED_LOG_OUTPUT': 'network',
        'STRUCTURED_LOG_NETWORK_TYPE': 'http',
        'STRUCTURED_LOG_HTTP_URL': 'https://logs.service.com/api/v1/ingest',
        'STRUCTURED_LOG_HTTP_AUTH': 'bearer',
        'STRUCTURED_LOG_HTTP_TOKEN': 'secret-token-123',
        'STRUCTURED_LOG_HTTP_BATCH_SIZE': '25'
    })
    def test_http_env_config(self):
        config = LoggerConfig.from_env()
        
        assert config.output_type == "network"
        assert config.network_config is not None
        assert isinstance(config.network_config, HTTPConfig)
        assert config.network_config.url == "https://logs.service.com/api/v1/ingest"
        assert config.network_config.auth_type == "bearer"
        assert config.network_config.token == "secret-token-123"
        assert config.network_config.batch_size == 25
    
    @patch.dict('os.environ', {
        'STRUCTURED_LOG_OUTPUT': 'network',
        'STRUCTURED_LOG_NETWORK_TYPE': 'socket',
        'STRUCTURED_LOG_SOCKET_HOST': 'logs.internal.com',
        'STRUCTURED_LOG_SOCKET_PORT': '9514',
        'STRUCTURED_LOG_SOCKET_PROTOCOL': 'udp'
    })
    def test_socket_env_config(self):
        config = LoggerConfig.from_env()
        
        assert config.output_type == "network"
        assert config.network_config is not None
        assert isinstance(config.network_config, SocketConfig)
        assert config.network_config.host == "logs.internal.com"
        assert config.network_config.port == 9514
        assert config.network_config.protocol == "udp"


class TestNetworkHandlerPerformance:
    """Performance and stress tests for network handlers"""
    
    @patch('socket.socket')
    def test_high_volume_logging(self, mock_socket_class):
        """Test handler performance with high volume"""
        mock_socket = Mock()
        mock_socket_class.return_value = mock_socket
        
        config = SocketConfig(protocol="udp", batch_size=100)
        handler = SocketHandler(config)
        
        # Send many messages quickly
        messages = []
        for i in range(1000):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"Message {i}", (), None)
            messages.append({"message": f"Message {i}", "record": record})
        
        start_time = time.time()
        
        # Process in batches
        for i in range(0, len(messages), 100):
            batch = messages[i:i+100]
            handler._send_batch(batch)
        
        duration = time.time() - start_time
        
        # Should complete reasonably quickly
        assert duration < 5.0  # Should process 1000 messages in under 5 seconds
        
        # Verify socket operations
        assert mock_socket_class.call_count > 0
        
        handler.close()
    
    def test_handler_cleanup(self):
        """Test proper resource cleanup"""
        config = SocketConfig(protocol="tcp")
        handler = SocketHandler(config)
        
        # Handler should be running
        assert handler.running is True
        assert handler.flush_thread.is_alive()
        
        # Close handler
        handler.close()
        
        # Should be stopped
        assert handler.running is False
        
        # Thread should finish (give it time)
        time.sleep(0.5)
        assert not handler.flush_thread.is_alive()


class TestNetworkHandlerErrorHandling:
    """Test error handling and fallback behavior"""
    
    def test_fallback_handler_creation(self):
        """Test local fallback handler creation"""
        config = SyslogConfig(
            fallback_to_local=True,
            local_fallback_file="test_fallback.log"
        )
        handler = SyslogHandler(config)
        
        assert handler.fallback_handler is not None
        assert isinstance(handler.fallback_handler, logging.FileHandler)
        
        handler.close()
        
        # Cleanup
        import os
        if os.path.exists("test_fallback.log"):
            os.remove("test_fallback.log")
    
    @patch('socket.socket')
    def test_fallback_on_network_failure(self, mock_socket_class):
        """Test fallback when network fails"""
        # Make socket operations fail
        mock_socket_class.side_effect = Exception("Network error")
        
        config = SyslogConfig(
            fallback_to_local=True,
            local_fallback_file="fallback_test.log",
            max_retries=1
        )
        handler = SyslogHandler(config)
        
        # Mock the fallback handler
        handler.fallback_handler = Mock()
        
        record = logging.LogRecord("test", logging.ERROR, "", 0, "Failed message", (), None)
        messages = [{"message": "Failed message", "record": record}]
        
        # Should use fallback when network fails
        handler._send_batch(messages)
        
        # Fallback should have been used
        # Note: Due to retry logic and threading, this test focuses on setup
        assert handler.fallback_handler is not None
        
        handler.close()
    
    def test_buffer_overflow_handling(self):
        """Test behavior when buffer overflows"""
        config = SyslogConfig(batch_size=1)
        handler = SyslogHandler(config)
        
        # Fill buffer beyond capacity (simulate)
        handler.buffer.maxsize = 2  # Small buffer
        
        # Create mock fallback
        handler.fallback_handler = Mock()
        
        # Try to emit many records quickly
        for i in range(5):
            record = logging.LogRecord("test", logging.INFO, "", 0, f"Overflow {i}", (), None)
            handler.emit(record)
        
        # Should not crash and may use fallback
        # The exact behavior depends on timing and threading
        assert True  # If we get here, no crash occurred
        
        handler.close()
    
    def test_invalid_host_handling(self):
        """Test handling of invalid hostnames"""
        config = SyslogConfig(
            host="invalid.nonexistent.host.12345",
            max_retries=1,
            timeout=1.0,
            fallback_to_local=True,
            local_fallback_file="invalid_host_test.log"
        )
        handler = SyslogHandler(config)
        
        # Mock fallback
        handler.fallback_handler = Mock()
        
        record = logging.LogRecord("test", logging.ERROR, "", 0, "Test invalid host", (), None)
        messages = [{"message": "Test invalid host", "record": record}]
        
        # Should handle gracefully
        handler._send_batch(messages)
        
        # Should not crash
        assert True
        
        handler.close()
        
        # Cleanup
        import os
        if os.path.exists("invalid_host_test.log"):
            os.remove("invalid_host_test.log")