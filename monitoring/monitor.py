"""
Monitoring module for StarCraft 2 AI agents
Provides Prometheus metrics and FastAPI endpoints for monitoring
"""

import time
import os
import psutil
from prometheus_client import Counter, Histogram, Gauge, Summary, start_http_server
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging
from typing import Optional
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get agent ID from environment
AGENT_ID = os.getenv("AGENT_ID", "1")

# Prometheus metrics
decision_time = Histogram(
    "sc2_agent_decision_time_seconds",
    "Time taken for AI to make decisions",
    ["agent_id"],
)

actions_total = Counter(
    "sc2_agent_actions_total",
    "Total number of actions taken by the agent",
    ["agent_id", "action_type"],
)

wins_total = Counter("sc2_agent_wins_total", "Total number of games won", ["agent_id"])

losses_total = Counter(
    "sc2_agent_losses_total", "Total number of games lost", ["agent_id"]
)

errors_total = Counter(
    "sc2_agent_errors_total",
    "Total number of errors encountered",
    ["agent_id", "error_type"],
)

cpu_usage = Gauge(
    "sc2_agent_cpu_usage_seconds_total", "CPU usage by the agent process", ["agent_id"]
)

memory_usage = Gauge(
    "sc2_agent_memory_usage_bytes", "Memory usage by the agent process", ["agent_id"]
)

connected = Gauge(
    "sc2_agent_connected", "Whether the agent is connected to the game", ["agent_id"]
)

game_duration = Gauge(
    "sc2_game_duration_seconds", "Current game duration", ["agent_id"]
)

unit_count = Gauge(
    "sc2_agent_unit_count",
    "Number of units controlled by the agent",
    ["agent_id", "unit_type"],
)

minerals_collected = Counter(
    "sc2_agent_minerals_collected_total", "Total minerals collected", ["agent_id"]
)

vespene_collected = Counter(
    "sc2_agent_vespene_collected_total", "Total vespene collected", ["agent_id"]
)


class Monitoring:
    """Monitoring class for AI agents"""

    def __init__(self, agent_id: str = AGENT_ID):
        self.agent_id = agent_id
        self.game_start_time: Optional[float] = None
        self.process = psutil.Process()
        self.last_cpu_time = self.process.cpu_times()
        self.last_update = time.time()

        # Initialize FastAPI app
        self.app = FastAPI(
            title=f"StarCraft 2 AI Agent {agent_id}",
            description="Monitoring and metrics API for SC2 AI agent",
            version="1.0.0",
        )

        # Setup endpoints
        self._setup_routes()

        # Setup Prometheus instrumentation
        instrumentator = Instrumentator()
        instrumentator.instrument(self.app).expose(self.app, endpoint="/metrics")

        logger.info(f"Monitoring initialized for agent {agent_id}")

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        async def root():
            return {
                "agent_id": self.agent_id,
                "status": "running",
                "uptime": time.time() - self.last_update,
            }

        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy", "agent_id": self.agent_id}

        @self.app.get("/metrics/summary")
        async def metrics_summary():
            """Get a summary of current metrics"""
            return {
                "agent_id": self.agent_id,
                "cpu_usage": cpu_usage.labels(agent_id=self.agent_id)._value.get(),
                "memory_usage": memory_usage.labels(
                    agent_id=self.agent_id
                )._value.get(),
                "connected": connected.labels(agent_id=self.agent_id)._value.get(),
                "wins": wins_total.labels(agent_id=self.agent_id)._value.get(),
                "losses": losses_total.labels(agent_id=self.agent_id)._value.get(),
                "actions": actions_total._metrics,
            }

        @self.app.get("/stats")
        async def get_stats():
            """Get detailed agent statistics"""
            try:
                mem_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                return {
                    "agent_id": self.agent_id,
                    "memory": {"rss": mem_info.rss, "vms": mem_info.vms},
                    "cpu_percent": cpu_percent,
                    "threads": self.process.num_threads(),
                    "connections": len(self.process.connections()),
                    "uptime": time.time() - self.last_update,
                }
            except Exception as e:
                logger.error(f"Error getting stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/game/start")
        async def game_start():
            """Called when a new game starts"""
            self.game_start_time = time.time()
            connected.labels(agent_id=self.agent_id).set(1)
            logger.info(f"Game started for agent {self.agent_id}")
            return {"status": "started", "timestamp": self.game_start_time}

        @self.app.post("/game/end")
        async def game_end(won: bool = False):
            """Called when a game ends"""
            if self.game_start_time:
                duration = time.time() - self.game_start_time
                game_duration.labels(agent_id=self.agent_id).set(duration)
                self.game_start_time = None

            if won:
                wins_total.labels(agent_id=self.agent_id).inc()
            else:
                losses_total.labels(agent_id=self.agent_id).inc()

            connected.labels(agent_id=self.agent_id).set(0)
            logger.info(f"Game ended for agent {self.agent_id}, won: {won}")
            return {"status": "ended", "won": won}

    def update_system_metrics(self):
        """Update CPU and memory metrics"""
        try:
            # Update memory usage
            memory_info = self.process.memory_info()
            memory_usage.labels(agent_id=self.agent_id).set(memory_info.rss)

            # Update CPU usage
            current_cpu_time = self.process.cpu_times()
            cpu_diff = current_cpu_time.user - self.last_cpu_time.user
            time_diff = time.time() - self.last_update

            if time_diff > 0:
                cpu_usage.labels(agent_id=self.agent_id).set(cpu_diff / time_diff)

            self.last_cpu_time = current_cpu_time
            self.last_update = time.time()
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            errors_total.labels(agent_id=self.agent_id, error_type="monitoring").inc()

    def record_action(self, action_type: str):
        """Record an action taken by the agent"""
        actions_total.labels(agent_id=self.agent_id, action_type=action_type).inc()

    def record_decision_time(self, duration: float):
        """Record the time taken to make a decision"""
        decision_time.labels(agent_id=self.agent_id).observe(duration)

    def record_error(self, error_type: str):
        """Record an error"""
        errors_total.labels(agent_id=self.agent_id, error_type=error_type).inc()

    def record_unit_count(self, unit_type: str, count: int):
        """Record the count of a specific unit type"""
        unit_count.labels(agent_id=self.agent_id, unit_type=unit_type).set(count)

    def record_minerals(self, amount: float):
        """Record minerals collected"""
        minerals_collected.labels(agent_id=self.agent_id).inc(amount)

    def record_vespene(self, amount: float):
        """Record vespene collected"""
        vespene_collected.labels(agent_id=self.agent_id).inc(amount)

    def get_app(self):
        """Get the FastAPI application"""
        return self.app

    def start_metrics_server(self, port: int = 9090):
        """Start the Prometheus metrics server"""
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            errors_total.labels(agent_id=self.agent_id, error_type="server").inc()
            raise

    def get_game_server_info(self):
        """Get game server connection information"""
        server = os.getenv("GAME_SERVER", "host.docker.internal")
        port = int(os.getenv("GAME_PORT", "8000"))
        return {"server": server, "port": port, "url": f"http://{server}:{port}"}

    def check_game_connection(self) -> bool:
        """Check if the game server is reachable"""
        server_info = self.get_game_server_info()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((server_info["server"], server_info["port"]))
            sock.close()

            is_connected = result == 0
            connected.labels(agent_id=self.agent_id).set(1 if is_connected else 0)
            return is_connected
        except Exception as e:
            logger.error(f"Error checking game connection: {e}")
            connected.labels(agent_id=self.agent_id).set(0)
            return False


# Singleton instance
_monitoring_instance = None


def get_monitoring() -> Monitoring:
    """Get the singleton monitoring instance"""
    global _monitoring_instance
    if _monitoring_instance is None:
        _monitoring_instance = Monitoring()
    return _monitoring_instance
