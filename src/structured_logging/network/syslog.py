"""
Syslog handler for RFC 3164/5424 compliant remote logging
"""

import logging
import socket
import ssl
from datetime import datetime
from typing import Any, Dict, List

from .base import BaseNetworkHandler
from .config import SyslogConfig


class SyslogHandler(BaseNetworkHandler):
    """RFC 3164/5424 compliant Syslog handler"""

    # Syslog facilities (RFC 3164)
    FACILITIES = {
        "kernel": 0,
        "user": 1,
        "mail": 2,
        "daemon": 3,
        "auth": 4,
        "syslog": 5,
        "lpr": 6,
        "news": 7,
        "uucp": 8,
        "cron": 9,
        "authpriv": 10,
        "ftp": 11,
        "local0": 16,
        "local1": 17,
        "local2": 18,
        "local3": 19,
        "local4": 20,
        "local5": 21,
        "local6": 22,
        "local7": 23,
    }

    # Syslog severities (RFC 3164)
    SEVERITIES = {
        logging.CRITICAL: 2,  # Critical
        logging.ERROR: 3,  # Error
        logging.WARNING: 4,  # Warning
        logging.INFO: 6,  # Informational
        logging.DEBUG: 7,  # Debug
    }

    def __init__(self, config: SyslogConfig):
        super().__init__(config)
        self.syslog_config = config
        self.socket = None
        self.hostname = getattr(config, "hostname", None) or socket.gethostname()

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
        proc_id = (
            str(self.syslog_config.process_id) if self.syslog_config.process_id else "-"
        )
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
                    record = msg_info["record"]
                    message = msg_info["message"]

                    # Format according to RFC
                    if self.syslog_config.rfc_format == "5424":
                        syslog_message = self._format_rfc5424(record, message)
                    else:
                        syslog_message = self._format_rfc3164(record, message)

                    # Encode and send
                    data = syslog_message.encode("utf-8")

                    if sock.type == socket.SOCK_STREAM:  # TCP
                        sock.sendall(data + b"\n")
                    else:  # UDP
                        sock.sendto(data, (self.config.host, self.config.port))

            finally:
                if sock:
                    sock.close()

        self._send_with_retry(send_messages)