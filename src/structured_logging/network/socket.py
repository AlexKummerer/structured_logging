"""
Raw socket handler for TCP/UDP logging
"""

import socket
from queue import Empty, Queue
from typing import Any, Dict, List

from .base import BaseNetworkHandler
from .config import SocketConfig


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

    def _get_or_create_socket(self, is_tcp: bool) -> socket.socket:
        """Get socket from pool or create new one"""
        if is_tcp:
            try:
                return self.connections.get_nowait()
            except Empty:
                return self._create_socket()
        else:
            return self._create_socket()

    def _encode_message(self, message: str) -> bytes:
        """Encode message with optional delimiter"""
        data = message.encode(self.socket_config.encoding)
        if self.socket_config.message_delimiter:
            data += self.socket_config.message_delimiter.encode(
                self.socket_config.encoding
            )
        return data

    def _send_message_data(self, sock: socket.socket, data: bytes, is_tcp: bool) -> None:
        """Send message data through socket"""
        if is_tcp:
            sock.sendall(data)
        else:
            # Check UDP size limit
            if len(data) > self.socket_config.udp_buffer_size:
                data = data[: self.socket_config.udp_buffer_size]
            sock.sendto(data, (self.config.host, self.config.port))

    def _return_socket_to_pool(self, sock: socket.socket) -> bool:
        """Try to return TCP socket to pool, return True if successful"""
        try:
            self.connections.put_nowait(sock)
            return True
        except:
            return False  # Pool full, will close socket

    def _send_batch(self, messages: List[Dict[str, Any]]) -> None:
        """Send batch of messages via raw socket"""
        if not messages:
            return

        def send_socket_messages():
            sock = None
            is_tcp = self.socket_config.protocol.lower() == "tcp"

            try:
                sock = self._get_or_create_socket(is_tcp)
                
                for msg_info in messages:
                    data = self._encode_message(msg_info["message"])
                    self._send_message_data(sock, data, is_tcp)

                if is_tcp and self._return_socket_to_pool(sock):
                    sock = None  # Don't close it
            finally:
                if sock:
                    sock.close()

        self._send_with_retry(send_socket_messages)

    def close(self) -> None:
        """Close handler and connection pool"""
        # Close all pooled connections
        while True:
            try:
                sock = self.connections.get_nowait()
                if sock and hasattr(sock, "close"):
                    sock.close()
            except (Empty, Exception):
                break

        super().close()