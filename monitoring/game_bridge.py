"""
Game Bridge - Runs on the host to bridge StarCraft 2 with AI agents in Docker

This script serves as an intermediary between the SC2 game running on the host
and the AI agents running in Docker containers. It exposes APIs for the agents
to interact with the game and provides game state information.
"""

import os
import sys
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple
from queue import Queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [GameBridge] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GameBridge:
    """Bridge between StarCraft 2 game and AI agents"""

    def __init__(self, port: int = 8000):
        self.port = port
        self.running = False
        self.game_env = None
        self.agents: Dict[str, "AIAgentConnection"] = {}
        self.game_state: Dict = {}
        self.action_queue = Queue()
        self.observation_queue = Queue()

        logger.info(f"GameBridge initialized on port {port}")

    def start(self):
        """Start the game bridge"""
        logger.info("Starting GameBridge...")
        self.running = True

        # Initialize StarCraft 2 environment
        try:
            from pysc2.env import sc2_env
            from pysc2.lib import features

            logger.info("Initializing StarCraft 2 environment...")
            self.game_env = sc2_env.SC2Env(
                map_name="Simple64",
                players=[
                    sc2_env.Agent(sc2_env.Race.protoss),
                    sc2_env.Agent(sc2_env.Race.protoss),
                ],
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    feature_screen=64, feature_minimap=64
                ),
                step_mul=16,
                game_steps_per_episode=0,
                realtime=False,
                visualize=True,
            )
            logger.info("StarCraft 2 environment initialized successfully")

            # Start game loop in separate thread
            self._start_game_loop()

        except ImportError as e:
            logger.error(f"PySC2 not installed: {e}")
            logger.error("Please install PySC2: pip install pysc2")
            self.running = False
        except Exception as e:
            logger.error(f"Failed to initialize game environment: {e}")
            self.running = False

    def _start_game_loop(self):
        """Start the game loop in a separate thread"""

        def game_loop():
            logger.info("Starting game loop...")
            timesteps = self.game_env.reset()
            episode = 0

            while self.running:
                episode += 1
                logger.info(f"Starting episode {episode}")

                # Send initial observations to agents
                self._broadcast_observations(timesteps)

                while self.running and not timesteps[0].last():
                    # Get actions from agents
                    actions = self._collect_actions(timesteps)

                    # Execute actions
                    timesteps = self.game_env.step(actions)

                    # Broadcast new observations
                    self._broadcast_observations(timesteps)

                    # Small delay to prevent CPU overload
                    time.sleep(0.001)

                logger.info(f"Episode {episode} ended")
                time.sleep(1)

                # Reset for next episode
                timesteps = self.game_env.reset()

        thread = threading.Thread(target=game_loop, daemon=True)
        thread.start()
        logger.info("Game loop started in background thread")

    def _broadcast_observations(self, timesteps):
        """Broadcast observations to all connected agents"""
        for agent_id, agent in self.agents.items():
            try:
                agent_index = int(agent_id) - 1
                if agent_index < len(timesteps):
                    agent.send_observation(timesteps[agent_index])
            except Exception as e:
                logger.error(f"Error broadcasting to agent {agent_id}: {e}")

    def _collect_actions(self, timesteps) -> List:
        """Collect actions from all connected agents"""
        actions = []

        for agent_id, agent in self.agents.items():
            try:
                action = agent.get_action(timeout=0.1)
                actions.append(action)
            except Exception as e:
                logger.warning(f"Error getting action from agent {agent_id}: {e}")
                from pysc2.lib import actions

                actions.append(actions.FUNCTIONS.no_op())

        # Fill missing actions with no_op
        while len(actions) < 2:
            from pysc2.lib import actions

            actions.append(actions.FUNCTIONS.no_op())

        return actions

    def register_agent(self, agent_id: str, agent_connection: "AIAgentConnection"):
        """Register an AI agent"""
        self.agents[agent_id] = agent_connection
        logger.info(f"Agent {agent_id} registered. Total agents: {len(self.agents)}")

    def unregister_agent(self, agent_id: str):
        """Unregister an AI agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered")

    def get_game_state(self) -> Dict:
        """Get current game state"""
        return self.game_state.copy()

    def stop(self):
        """Stop the game bridge"""
        logger.info("Stopping GameBridge...")
        self.running = False

        if self.game_env:
            self.game_env.close()

        logger.info("GameBridge stopped")


class AIAgentConnection:
    """Represents a connection to an AI agent"""

    def __init__(self, agent_id: str, bridge: GameBridge):
        self.agent_id = agent_id
        self.bridge = bridge
        self.observation_queue = Queue(maxsize=10)
        self.action_queue = Queue(maxsize=10)
        self.connected = False

        logger.info(f"AI Agent connection created for agent {agent_id}")

    def connect(self, agent_host: str, agent_port: int):
        """Connect to the AI agent"""
        logger.info(f"Connecting to agent {self.agent_id} at {agent_host}:{agent_port}")
        try:
            # For now, we'll use a simple local connection
            # In production, this would use HTTP/gRPC
            self.connected = True
            logger.info(f"Connected to agent {self.agent_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to agent {self.agent_id}: {e}")
            return False

    def send_observation(self, observation):
        """Send observation to the agent"""
        try:
            self.observation_queue.put(observation, block=False)
        except Exception as e:
            logger.warning(f"Failed to send observation to agent {self.agent_id}: {e}")

    def get_action(self, timeout: float = 1.0):
        """Get action from the agent"""
        try:
            return self.action_queue.get(timeout=timeout)
        except Exception as e:
            logger.warning(f"Failed to get action from agent {self.agent_id}: {e}")
            from pysc2.lib import actions

            return actions.FUNCTIONS.no_op()

    def disconnect(self):
        """Disconnect from the agent"""
        self.connected = False
        logger.info(f"Disconnected from agent {self.agent_id}")


def create_simple_http_server(bridge: GameBridge):
    """Create a simple HTTP server for agent communication"""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    import urllib.parse

    class BridgeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = json.dumps(
                    {
                        "status": "healthy",
                        "running": bridge.running,
                        "connected_agents": len(bridge.agents),
                    }
                )
                self.wfile.write(response.encode())

            elif parsed.path == "/game/state":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = json.dumps(bridge.get_game_state())
                self.wfile.write(response.encode())

            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            if parsed.path == "/agent/register":
                agent_id = data.get("agent_id")
                agent_host = data.get("host", "localhost")
                agent_port = data.get("port", 8080)

                connection = AIAgentConnection(agent_id, bridge)
                if connection.connect(agent_host, agent_port):
                    bridge.register_agent(agent_id, connection)
                    response = {"status": "registered", "agent_id": agent_id}
                    status = 200
                else:
                    response = {"status": "error", "message": "Failed to connect"}
                    status = 500

                self.send_response(status)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            elif parsed.path == "/agent/unregister":
                agent_id = data.get("agent_id")
                bridge.unregister_agent(agent_id)
                response = {"status": "unregistered", "agent_id": agent_id}
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            logger.info(f"[HTTP] {format % args}")

    try:
        server = HTTPServer(("0.0.0.0", bridge.port), BridgeHandler)
        logger.info(f"HTTP server started on port {bridge.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("HTTP server stopped")
    except Exception as e:
        logger.error(f"HTTP server error: {e}")


def main():
    """Main entry point - Run on the host"""
    import argparse

    parser = argparse.ArgumentParser(description="StarCraft 2 Game Bridge")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for the bridge server"
    )
    args = parser.parse_args()

    logger.info("Starting StarCraft 2 Game Bridge on host...")
    logger.info("This script should run on the host machine")
    logger.info("AI agents in Docker containers will connect to this bridge")

    # Create and start bridge
    bridge = GameBridge(port=args.port)
    bridge.start()

    # Start HTTP server in separate thread
    http_thread = threading.Thread(
        target=create_simple_http_server, args=(bridge,), daemon=True
    )
    http_thread.start()

    logger.info(f"Game Bridge running on http://localhost:{args.port}")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        bridge.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
