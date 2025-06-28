"""
Real-time monitoring dashboard server

This module provides a web-based dashboard for monitoring log streams
with WebSocket support for real-time updates.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import aiohttp
from aiohttp import web

from ..logger import get_logger
from ..streaming import LogStreamProcessor
from .alerts import AlertManager
from .metrics import MetricsAggregator


@dataclass
class DashboardConfig:
    """Configuration for monitoring dashboard"""

    # Server settings
    host: str = "localhost"
    port: int = 8080
    static_path: Optional[str] = None

    # WebSocket settings
    ws_path: str = "/ws"
    ws_ping_interval: float = 30.0
    ws_max_connections: int = 100

    # Data settings
    buffer_size: int = 1000  # Keep last N logs
    metrics_window: int = 300  # 5 minutes of metrics
    update_interval: float = 1.0  # Update frequency

    # Features
    enable_alerts: bool = True
    enable_metrics: bool = True
    enable_search: bool = True

    # Security
    require_auth: bool = False
    auth_token: Optional[str] = None
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


class DashboardServer:
    """
    Real-time monitoring dashboard server

    Provides:
    - WebSocket API for real-time log streaming
    - REST API for metrics and configuration
    - Static file serving for dashboard UI
    - Alert management
    - Metrics aggregation
    """

    def __init__(self, config: Optional[DashboardConfig] = None):
        self.config = config or DashboardConfig()
        self.logger = get_logger("monitoring.dashboard")

        # Components
        self.app = web.Application()
        self.metrics = MetricsAggregator()
        self.alerts = AlertManager()

        # State
        self._running = False
        self._clients: Set[web.WebSocketResponse] = set()
        self._log_buffer: List[Dict[str, Any]] = []
        self._stream_processor: Optional[LogStreamProcessor] = None

        # Setup routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup HTTP routes"""
        # WebSocket endpoint
        self.app.router.add_get(self.config.ws_path, self._ws_handler)

        # REST API
        self.app.router.add_get("/api/metrics", self._get_metrics)
        self.app.router.add_get("/api/logs", self._get_logs)
        self.app.router.add_get("/api/alerts", self._get_alerts)
        self.app.router.add_post("/api/alerts", self._create_alert)
        self.app.router.add_delete("/api/alerts/{id}", self._delete_alert)
        self.app.router.add_get("/api/status", self._get_status)
        self.app.router.add_post("/api/search", self._search_logs)

        # Static files
        if self.config.static_path:
            self.app.router.add_static("/", self.config.static_path)
        else:
            # Serve default dashboard
            self.app.router.add_get("/", self._serve_dashboard)

        # CORS middleware
        self._setup_cors()

    def _setup_cors(self) -> None:
        """Setup CORS headers"""

        @web.middleware
        async def cors_middleware(request, handler):
            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = ", ".join(
                self.config.cors_origins
            )
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            return response

        self.app.middlewares.append(cors_middleware)

    async def _ws_handler(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket connection handler"""
        # Check auth if required
        if self.config.require_auth:
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            if token != self.config.auth_token:
                return web.Response(status=401, text="Unauthorized")

        # Check connection limit
        if len(self._clients) >= self.config.ws_max_connections:
            return web.Response(status=503, text="Too many connections")

        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self._clients.add(ws)
        self.logger.info(f"WebSocket client connected. Total: {len(self._clients)}")

        try:
            # Send initial data
            await ws.send_json(
                {
                    "type": "init",
                    "data": {
                        "logs": self._log_buffer[-100:],  # Last 100 logs
                        "metrics": await self.metrics.get_current_snapshot(),
                        "alerts": self.alerts.get_active_alerts(),
                    },
                }
            )

            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_ws_message(ws, data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    self.logger.error(f"WebSocket error: {ws.exception()}")

        finally:
            self._clients.remove(ws)
            self.logger.info(
                f"WebSocket client disconnected. Total: {len(self._clients)}"
            )

        return ws

    async def _handle_ws_message(
        self, ws: web.WebSocketResponse, message: Dict[str, Any]
    ) -> None:
        """Handle incoming WebSocket messages"""
        msg_type = message.get("type")

        if msg_type == "ping":
            await ws.send_json({"type": "pong"})
        elif msg_type == "subscribe":
            # Handle subscription to specific log streams
            filters = message.get("filters", {})
            await ws.send_json({"type": "subscribed", "filters": filters})
        elif msg_type == "query":
            # Handle log queries
            query = message.get("query", {})
            results = self._query_logs(query)
            await ws.send_json({"type": "query_result", "data": results})

    async def _get_metrics(self, request: web.Request) -> web.Response:
        """Get current metrics"""
        time_range = request.query.get("range", "5m")
        metrics = await self.metrics.get_metrics(time_range)
        return web.json_response(metrics)

    async def _get_logs(self, request: web.Request) -> web.Response:
        """Get recent logs"""
        limit = int(request.query.get("limit", 100))
        offset = int(request.query.get("offset", 0))

        logs = self._log_buffer[offset : offset + limit]
        return web.json_response(
            {
                "logs": logs,
                "total": len(self._log_buffer),
                "offset": offset,
                "limit": limit,
            }
        )

    async def _get_alerts(self, request: web.Request) -> web.Response:
        """Get alerts"""
        active_only = request.query.get("active", "true").lower() == "true"

        if active_only:
            alerts = self.alerts.get_active_alerts()
        else:
            alerts = self.alerts.get_all_alerts()

        return web.json_response(alerts)

    async def _create_alert(self, request: web.Request) -> web.Response:
        """Create new alert rule"""
        data = await request.json()

        try:
            alert_id = self.alerts.add_rule(
                name=data["name"],
                condition=data["condition"],
                actions=data.get("actions", []),
            )

            return web.json_response({"id": alert_id, "status": "created"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def _delete_alert(self, request: web.Request) -> web.Response:
        """Delete alert rule"""
        alert_id = request.match_info["id"]

        if self.alerts.remove_rule(alert_id):
            return web.json_response({"status": "deleted"})
        else:
            return web.json_response({"error": "Alert not found"}, status=404)

    async def _get_status(self, request: web.Request) -> web.Response:
        """Get dashboard status"""
        return web.json_response(
            {
                "status": "running" if self._running else "stopped",
                "clients": len(self._clients),
                "logs_buffered": len(self._log_buffer),
                "alerts_active": len(self.alerts.get_active_alerts()),
                "uptime": self._get_uptime(),
            }
        )

    async def _search_logs(self, request: web.Request) -> web.Response:
        """Search logs"""
        if not self.config.enable_search:
            return web.json_response({"error": "Search disabled"}, status=403)

        data = await request.json()
        query = data.get("query", "")
        filters = data.get("filters", {})

        results = self._search_log_buffer(query, filters)
        return web.json_response(
            {"results": results, "query": query, "filters": filters}
        )

    async def _serve_dashboard(self, request: web.Request) -> web.Response:
        """Serve default dashboard HTML"""
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type="text/html")

    def _generate_dashboard_html(self) -> str:
        """Generate default dashboard HTML"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Structured Logging Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .logs {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            height: 500px;
            overflow-y: auto;
        }
        .log-entry {
            padding: 10px;
            border-bottom: 1px solid #eee;
            font-family: monospace;
            font-size: 12px;
        }
        .log-entry.error {
            background: #fee;
        }
        .log-entry.warning {
            background: #ffd;
        }
        .status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            border-radius: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }
        .status.disconnected {
            background: #f44336;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Structured Logging Dashboard</h1>
        </div>

        <div class="metrics" id="metrics">
            <div class="metric-card">
                <h3>Total Logs</h3>
                <div class="value" id="total-logs">0</div>
            </div>
            <div class="metric-card">
                <h3>Error Rate</h3>
                <div class="value" id="error-rate">0%</div>
            </div>
            <div class="metric-card">
                <h3>Avg Response Time</h3>
                <div class="value" id="avg-response">0ms</div>
            </div>
            <div class="metric-card">
                <h3>Active Alerts</h3>
                <div class="value" id="active-alerts">0</div>
            </div>
        </div>

        <div class="logs" id="logs"></div>
    </div>

    <div class="status" id="status">Connected</div>

    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        const logsContainer = document.getElementById('logs');
        const statusElement = document.getElementById('status');

        ws.onopen = () => {
            statusElement.textContent = 'Connected';
            statusElement.classList.remove('disconnected');
        };

        ws.onclose = () => {
            statusElement.textContent = 'Disconnected';
            statusElement.classList.add('disconnected');
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);

            if (message.type === 'init') {
                // Initial data
                updateMetrics(message.data.metrics);
                displayLogs(message.data.logs);
                updateAlerts(message.data.alerts);
            } else if (message.type === 'log') {
                // New log entry
                addLog(message.data);
            } else if (message.type === 'metrics') {
                // Metrics update
                updateMetrics(message.data);
            } else if (message.type === 'alert') {
                // Alert notification
                showAlert(message.data);
            }
        };

        function updateMetrics(metrics) {
            if (metrics) {
                document.getElementById('total-logs').textContent =
                    metrics.total_logs || 0;
                document.getElementById('error-rate').textContent =
                    (metrics.error_rate || 0).toFixed(1) + '%';
                document.getElementById('avg-response').textContent =
                    (metrics.avg_response_time || 0).toFixed(0) + 'ms';
                document.getElementById('active-alerts').textContent =
                    metrics.active_alerts || 0;
            }
        }

        function displayLogs(logs) {
            logsContainer.innerHTML = '';
            logs.forEach(log => addLog(log));
        }

        function addLog(log) {
            const entry = document.createElement('div');
            entry.className = 'log-entry';

            if (log.level === 'error') {
                entry.classList.add('error');
            } else if (log.level === 'warning') {
                entry.classList.add('warning');
            }

            entry.textContent = JSON.stringify(log);
            logsContainer.appendChild(entry);

            // Keep only last 100 logs
            if (logsContainer.children.length > 100) {
                logsContainer.removeChild(logsContainer.firstChild);
            }

            // Auto-scroll to bottom
            logsContainer.scrollTop = logsContainer.scrollHeight;
        }

        function updateAlerts(alerts) {
            document.getElementById('active-alerts').textContent = alerts.length;
        }

        function showAlert(alert) {
            console.log('Alert:', alert);
            // Could show notification here
        }

        // Send ping every 30s to keep connection alive
        setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({type: 'ping'}));
            }
        }, 30000);
    </script>
</body>
</html>
        """

    def _query_logs(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Query logs from buffer"""
        results = self._log_buffer

        # Apply filters
        if "level" in query:
            results = [log for log in results if log.get("level") == query["level"]]

        if "start_time" in query:
            start = datetime.fromisoformat(query["start_time"])
            results = [
                log
                for log in results
                if datetime.fromisoformat(log.get("timestamp", "")) >= start
            ]

        if "end_time" in query:
            end = datetime.fromisoformat(query["end_time"])
            results = [
                log
                for log in results
                if datetime.fromisoformat(log.get("timestamp", "")) <= end
            ]

        if "search" in query:
            search_term = query["search"].lower()
            results = [log for log in results if search_term in json.dumps(log).lower()]

        return results

    def _search_log_buffer(
        self, query: str, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search logs in buffer"""
        results = self._log_buffer

        # Text search
        if query:
            query_lower = query.lower()
            results = [log for log in results if query_lower in json.dumps(log).lower()]

        # Apply filters
        for key, value in filters.items():
            results = [log for log in results if log.get(key) == value]

        return results

    def _get_uptime(self) -> str:
        """Get server uptime"""
        if hasattr(self, "_start_time"):
            uptime = datetime.now() - self._start_time
            return str(uptime).split(".")[0]
        return "0:00:00"

    async def process_log(self, log: Dict[str, Any]) -> None:
        """Process incoming log entry"""
        # Add to buffer
        self._log_buffer.append(log)

        # Maintain buffer size
        if len(self._log_buffer) > self.config.buffer_size:
            self._log_buffer.pop(0)

        # Update metrics
        if self.config.enable_metrics:
            await self.metrics.update(log)

        # Check alerts
        if self.config.enable_alerts:
            alerts = await self.alerts.check(log)
            for alert in alerts:
                await self._broadcast({"type": "alert", "data": alert})

        # Broadcast to clients
        await self._broadcast({"type": "log", "data": log})

    async def _broadcast(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients"""
        if not self._clients:
            return

        # Send to all clients
        disconnected = []
        for ws in self._clients:
            try:
                await ws.send_json(message)
            except ConnectionResetError:
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            self._clients.discard(ws)

    async def _update_metrics_loop(self) -> None:
        """Periodically broadcast metrics updates"""
        while self._running:
            await asyncio.sleep(self.config.update_interval)

            if self._clients and self.config.enable_metrics:
                metrics = await self.metrics.get_current_snapshot()
                await self._broadcast({"type": "metrics", "data": metrics})

    async def start(self) -> None:
        """Start the dashboard server"""
        self._running = True
        self._start_time = datetime.now()

        # Start background tasks
        asyncio.create_task(self._update_metrics_loop())

        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()

        self.logger.info(
            f"Dashboard started at http://{self.config.host}:{self.config.port}"
        )

    async def stop(self) -> None:
        """Stop the dashboard server"""
        self._running = False

        # Close all WebSocket connections
        for ws in list(self._clients):
            await ws.close()

        self.logger.info("Dashboard stopped")

    def attach_to_stream(self, stream: LogStreamProcessor) -> None:
        """Attach dashboard to a log stream processor"""

        # Create a custom sink that feeds the dashboard
        async def dashboard_sink(items: List[Dict[str, Any]]) -> None:
            for item in items:
                await self.process_log(item)

        stream.sink(dashboard_sink)
        self._stream_processor = stream


def create_dashboard(
    config: Optional[DashboardConfig] = None,
    stream: Optional[LogStreamProcessor] = None,
) -> DashboardServer:
    """
    Create a monitoring dashboard

    Args:
        config: Dashboard configuration
        stream: Optional stream processor to attach to

    Returns:
        DashboardServer instance
    """
    dashboard = DashboardServer(config)

    if stream:
        dashboard.attach_to_stream(stream)

    return dashboard
