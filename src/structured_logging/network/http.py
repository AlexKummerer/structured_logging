"""
HTTP handler for sending logs to REST APIs
"""

import base64
import json
import urllib.request
from typing import Any, Dict, List
from urllib.error import HTTPError

from .base import BaseNetworkHandler
from .config import HTTPConfig


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
            self.auth_headers[self.http_config.api_key_header] = (
                self.http_config.api_key
            )
        elif self.http_config.auth_type == "basic" and self.http_config.username:
            credentials = (
                f"{self.http_config.username}:{self.http_config.password or ''}"
            )
            encoded = base64.b64encode(credentials.encode()).decode()
            self.auth_headers["Authorization"] = f"Basic {encoded}"

    def _prepare_single_message_payload(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare payload for single message"""
        return {
            "message": message["message"],
            "timestamp": message["timestamp"],
            "level": message["record"].levelname,
            "logger": message["record"].name,
        }

    def _prepare_batch_payload(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare payload for batch of messages"""
        return {
            "logs": [
                {
                    "message": msg["message"],
                    "timestamp": msg["timestamp"],
                    "level": msg["record"].levelname,
                    "logger": msg["record"].name,
                }
                for msg in messages
            ],
            "batch_size": len(messages),
        }

    def _prepare_request_headers(self, data_length: int) -> Dict[str, str]:
        """Prepare HTTP request headers"""
        headers = {
            "Content-Type": self.http_config.content_type,
            "User-Agent": self.http_config.user_agent,
            "Content-Length": str(data_length),
        }
        headers.update(self.http_config.headers)
        headers.update(self.auth_headers)
        return headers

    def _execute_http_request(self, request: urllib.request.Request) -> None:
        """Execute HTTP request and handle response"""
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            if response.status >= 400:
                raise HTTPError(
                    self.http_config.url,
                    response.status,
                    f"HTTP {response.status}",
                    response.headers,
                    None,
                )

    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send batch of messages via HTTP"""
        if not messages:
            return

        def send_http_request():
            # Prepare payload
            if self.http_config.batch_size == 1 and len(messages) == 1:
                payload = self._prepare_single_message_payload(messages[0])
            else:
                payload = self._prepare_batch_payload(messages)

            # Convert to JSON
            data = json.dumps(payload).encode("utf-8")
            
            # Prepare request
            headers = self._prepare_request_headers(len(data))
            request = urllib.request.Request(
                self.http_config.url,
                data=data,
                headers=headers,
                method=self.http_config.method,
            )
            
            # Execute request
            self._execute_http_request(request)

        self._send_with_retry(send_http_request)