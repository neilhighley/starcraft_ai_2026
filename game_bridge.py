"""
Game Bridge - Runs on the host to bridge StarCraft 2 with AI agents in Docker

This script serves as an intermediary between the SC2 game running on the host
and the AI agents running in Docker containers. It exposes APIs for the agents
to interact with the game and sends game state via HTTP.
"""

import os
import sys
import time
import logging
import threading
import json
import numpy as np
from typing import Dict, List, Optional, Any
from queue import Queue
from dataclasses import dataclass, asdict
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# HTTP client for communicating with Docker agents
import urllib.request
import urllib.error

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [GameBridge] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Game State Serialization
# ============================================================================

@dataclass
class SerializedGameState:
    """Serialized game state to send to agents"""
    step: int = 0
    game_loop: int = 0
    minerals: int = 0
    vespene: int = 0
    food_used: int = 0
    food_cap: int = 0
    food_army: int = 0
    food_workers: int = 0
    idle_worker_count: int = 0
    army_count: int = 0
    warp_gate_count: int = 0
    larva_count: int = 0
    units: List[Dict] = None
    enemy_units: List[Dict] = None
    map_name: str = ""
    player_id: int = 1
    episode_ended: bool = False
    reward: float = 0.0
    
    def __post_init__(self):
        if self.units is None:
            self.units = []
        if self.enemy_units is None:
            self.enemy_units = []
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def serialize_timestep(timestep, player_id: int = 1, map_name: str = "Simple64") -> SerializedGameState:
    """
    Convert a PySC2 timestep to a serialized game state.
    This extracts the relevant information for the AI agent.
    """
    try:
        obs = timestep.observation
        
        # Extract player info
        player = obs.get("player", [0] * 11)
        
        # Create serialized state
        state = SerializedGameState(
            step=timestep.step_type,
            game_loop=obs.get("game_loop", [0])[0] if hasattr(obs.get("game_loop", 0), '__len__') else obs.get("game_loop", 0),
            minerals=int(player[1]) if len(player) > 1 else 0,
            vespene=int(player[2]) if len(player) > 2 else 0,
            food_used=int(player[3]) if len(player) > 3 else 0,
            food_cap=int(player[4]) if len(player) > 4 else 0,
            food_army=int(player[5]) if len(player) > 5 else 0,
            food_workers=int(player[6]) if len(player) > 6 else 0,
            idle_worker_count=int(player[7]) if len(player) > 7 else 0,
            army_count=int(player[8]) if len(player) > 8 else 0,
            warp_gate_count=int(player[9]) if len(player) > 9 else 0,
            larva_count=int(player[10]) if len(player) > 10 else 0,
            map_name=map_name,
            player_id=player_id,
            episode_ended=timestep.last(),
            reward=float(timestep.reward) if timestep.reward else 0.0,
        )
        
        # Extract unit information (simplified)
        if "feature_units" in obs:
            units = obs["feature_units"]
            for unit in units:
                unit_dict = {
                    "unit_type": int(unit[0]) if len(unit) > 0 else 0,
                    "alliance": int(unit[1]) if len(unit) > 1 else 0,
                    "x": float(unit[12]) if len(unit) > 12 else 0,
                    "y": float(unit[13]) if len(unit) > 13 else 0,
                    "health": int(unit[2]) if len(unit) > 2 else 0,
                }
                if unit_dict["alliance"] == 1:  # Self
                    state.units.append(unit_dict)
                elif unit_dict["alliance"] == 4:  # Enemy
                    state.enemy_units.append(unit_dict)
        
        return state
        
    except Exception as e:
        logger.error(f"Error serializing timestep: {e}")
        return SerializedGameState()


# ============================================================================
# Agent Connection (HTTP-based)
# ============================================================================

class HTTPAgentConnection:
    """HTTP-based connection to an AI agent in Docker"""
    
    def __init__(self, agent_id: str, host: str, port: int):
        self.agent_id = agent_id
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.connected = False
        self.last_action = None
        
        logger.info(f"Agent connection created: {agent_id} -> {self.base_url}")
    
    def check_connection(self) -> bool:
        """Check if the agent is reachable"""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/health",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=2) as response:
                data = json.loads(response.read().decode())
                self.connected = data.get("status") == "healthy"
                return self.connected
        except Exception as e:
            logger.debug(f"Agent {self.agent_id} not reachable: {e}")
            self.connected = False
            return False
    
    def send_observation(self, game_state: SerializedGameState) -> Optional[Dict]:
        """Send game state to the agent and get action back"""
        try:
            data = json.dumps(game_state.to_dict()).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/observation",
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode())
                self.last_action = result.get("action", {"action_type": "no_op"})
                return self.last_action
                
        except urllib.error.URLError as e:
            logger.warning(f"Failed to send observation to agent {self.agent_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error sending observation to agent {self.agent_id}: {e}")
            return None
    
    def get_action(self) -> Dict:
        """Get the last action from the agent"""
        if self.last_action:
            return self.last_action
        return {"action_type": "no_op"}
    
    def reset(self):
        """Reset the agent for a new game"""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/reset",
                data=b"{}",
                method="POST",
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.loads(response.read().decode())
        except Exception as e:
            logger.warning(f"Failed to reset agent {self.agent_id}: {e}")
            return None


# ============================================================================
# Game Bridge
# ============================================================================

class GameBridge:
    """Bridge between StarCraft 2 game and AI agents"""

    def __init__(self, port: int = 8000, map_name: str = "Simple64"):
        self.port = port
        self.map_name = map_name
        self.running = False
        self.game_env = None
        self.agents: Dict[str, HTTPAgentConnection] = {}
        self.game_state: SerializedGameState = SerializedGameState()
        self.current_step = 0
        self.http_server_started = False
        
        # Default agent endpoints (Docker containers)
        self.default_agents = {
            "1": ("localhost", 8080),
            "2": ("localhost", 8081),
        }
        
        logger.info(f"GameBridge initialized on port {port}")

    def discover_agents(self):
        """Try to connect to default agent endpoints"""
        for agent_id, (host, port) in self.default_agents.items():
            if agent_id not in self.agents:
                connection = HTTPAgentConnection(agent_id, host, port)
                if connection.check_connection():
                    self.agents[agent_id] = connection
                    logger.info(f"Discovered agent {agent_id} at {host}:{port}")
                else:
                    logger.debug(f"Agent {agent_id} not available at {host}:{port}")

    def start(self, skip_game: bool = False):
        """Start the game bridge"""
        logger.info("Starting GameBridge...")
        self.running = True
        
        if skip_game:
            logger.info("Skipping game initialization (HTTP server only mode)")
            return
        
        # Initialize StarCraft 2 environment
        try:
            from pysc2.env import sc2_env
            from pysc2.lib import features

            logger.info("Initializing StarCraft 2 environment...")
            self.game_env = sc2_env.SC2Env(
                map_name=self.map_name,
                players=[
                    sc2_env.Agent(sc2_env.Race.protoss),
                    sc2_env.Agent(sc2_env.Race.zerg),
                ],
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    feature_screen=64, 
                    feature_minimap=64,
                    use_feature_units=True,
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
            logger.info("Running in HTTP server only mode")
        except Exception as e:
            logger.error(f"Failed to initialize game environment: {e}")
            logger.info("Running in HTTP server only mode")

    def _start_game_loop(self):
        """Start the game loop in a separate thread"""

        def game_loop():
            from pysc2.lib import actions as sc2_actions
            
            logger.info("Starting game loop...")
            timesteps = self.game_env.reset()
            episode = 0

            while self.running:
                episode += 1
                logger.info(f"Starting episode {episode}")
                
                # Discover agents at start of each episode
                self.discover_agents()
                
                # Reset agents
                for agent_id, agent in self.agents.items():
                    agent.reset()

                # Send initial observations to agents
                self._broadcast_observations(timesteps)

                while self.running and not timesteps[0].last():
                    self.current_step += 1
                    
                    # Get actions from agents via HTTP
                    actions = self._collect_actions_http(timesteps)

                    # Execute actions in game
                    timesteps = self.game_env.step(actions)

                    # Broadcast new observations to agents
                    self._broadcast_observations(timesteps)

                    # Small delay to prevent CPU overload
                    time.sleep(0.01)

                logger.info(f"Episode {episode} ended")
                
                # Notify agents of episode end
                self._broadcast_episode_end(timesteps)
                
                time.sleep(2)

                # Reset for next episode
                timesteps = self.game_env.reset()

        thread = threading.Thread(target=game_loop, daemon=True)
        thread.start()
        logger.info("Game loop started in background thread")

    def _broadcast_observations(self, timesteps):
        """Broadcast observations to all connected agents via HTTP"""
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            try:
                if i < len(timesteps):
                    # Serialize the timestep
                    state = serialize_timestep(
                        timesteps[i], 
                        player_id=int(agent_id),
                        map_name=self.map_name
                    )
                    state.step = self.current_step
                    
                    # Store current game state
                    self.game_state = state
                    
                    # Send to agent via HTTP
                    agent.send_observation(state)
                    
            except Exception as e:
                logger.error(f"Error broadcasting to agent {agent_id}: {e}")

    def _broadcast_episode_end(self, timesteps):
        """Notify agents that the episode has ended"""
        for i, (agent_id, agent) in enumerate(self.agents.items()):
            try:
                if i < len(timesteps):
                    state = serialize_timestep(timesteps[i], player_id=int(agent_id))
                    state.episode_ended = True
                    agent.send_observation(state)
            except Exception as e:
                logger.error(f"Error sending episode end to agent {agent_id}: {e}")

    def _collect_actions_http(self, timesteps) -> List:
        """Collect actions from all agents via HTTP and convert to PySC2 actions"""
        from pysc2.lib import actions as sc2_actions
        
        actions = []
        
        for agent_id, agent in self.agents.items():
            try:
                action_dict = agent.get_action()
                action = self._convert_action(action_dict)
                actions.append(action)
            except Exception as e:
                logger.warning(f"Error getting action from agent {agent_id}: {e}")
                actions.append(sc2_actions.FUNCTIONS.no_op())

        # Fill missing actions with no_op
        while len(actions) < 2:
            actions.append(sc2_actions.FUNCTIONS.no_op())

        return actions

    def _convert_action(self, action_dict: Dict):
        """Convert an action dictionary from agent to PySC2 action"""
        from pysc2.lib import actions as sc2_actions
        
        action_type = action_dict.get("action_type", "no_op")
        
        # Map action types to PySC2 functions
        if action_type == "no_op":
            return sc2_actions.FUNCTIONS.no_op()
        
        elif action_type == "train_worker":
            # Train Probe for Protoss
            if sc2_actions.FUNCTIONS.Train_Probe_quick.id in sc2_actions.FUNCTIONS:
                return sc2_actions.FUNCTIONS.Train_Probe_quick("now")
            return sc2_actions.FUNCTIONS.no_op()
        
        elif action_type == "train_unit":
            ability_id = action_dict.get("ability_id", 916)  # Default to Zealot
            if ability_id == 916:  # Zealot
                return sc2_actions.FUNCTIONS.Train_Zealot_quick("now")
            elif ability_id == 917:  # Stalker
                return sc2_actions.FUNCTIONS.Train_Stalker_quick("now")
            return sc2_actions.FUNCTIONS.no_op()
        
        elif action_type == "build_supply":
            # Build Pylon
            target = action_dict.get("target_point", [30, 30])
            return sc2_actions.FUNCTIONS.Build_Pylon_screen("now", target)
        
        elif action_type == "attack_move":
            target = action_dict.get("target_point", [32, 32])
            return sc2_actions.FUNCTIONS.Attack_minimap("now", target)
        
        elif action_type == "harvest_gather":
            return sc2_actions.FUNCTIONS.Harvest_Gather_screen("now", [20, 20])
        
        # Default to no_op
        return sc2_actions.FUNCTIONS.no_op()

    def register_agent(self, agent_id: str, host: str, port: int):
        """Register an AI agent"""
        connection = HTTPAgentConnection(agent_id, host, port)
        if connection.check_connection():
            self.agents[agent_id] = connection
            logger.info(f"Agent {agent_id} registered at {host}:{port}. Total: {len(self.agents)}")
            return True
        else:
            logger.warning(f"Could not connect to agent {agent_id} at {host}:{port}")
            return False

    def unregister_agent(self, agent_id: str):
        """Unregister an AI agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Agent {agent_id} unregistered")

    def get_game_state(self) -> Dict:
        """Get current game state"""
        return self.game_state.to_dict()

    def stop(self):
        """Stop the game bridge"""
        logger.info("Stopping GameBridge...")
        self.running = False

        if self.game_env:
            self.game_env.close()

        logger.info("GameBridge stopped")


# ============================================================================
# HTTP Server
# ============================================================================

def create_http_server(bridge: GameBridge):
    """Create HTTP server for bridge API"""
    
    class BridgeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)

            if parsed.path == "/health":
                self._send_json(200, {
                    "status": "healthy",
                    "running": bridge.running,
                    "connected_agents": len(bridge.agents),
                    "current_step": bridge.current_step,
                })

            elif parsed.path == "/":
                self._send_json(200, {
                    "service": "StarCraft 2 Game Bridge",
                    "version": "2.0.0",
                    "status": "running" if bridge.running else "stopped",
                    "agents": list(bridge.agents.keys()),
                    "game_state": bridge.get_game_state(),
                })

            elif parsed.path == "/game/state":
                self._send_json(200, bridge.get_game_state())

            elif parsed.path == "/agents":
                agents_info = {}
                for agent_id, agent in bridge.agents.items():
                    agents_info[agent_id] = {
                        "host": agent.host,
                        "port": agent.port,
                        "connected": agent.connected,
                    }
                self._send_json(200, {"agents": agents_info})

            else:
                self._send_json(404, {"error": "Not found"})

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                post_data = self.rfile.read(content_length) if content_length > 0 else b"{}"
                data = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON: {e}"})
                return

            if parsed.path == "/agent/register":
                agent_id = data.get("agent_id")
                host = data.get("host", "localhost")
                port = data.get("port", 8080)
                
                if not agent_id:
                    self._send_json(400, {"error": "agent_id required"})
                    return
                
                if bridge.register_agent(agent_id, host, port):
                    self._send_json(200, {"status": "registered", "agent_id": agent_id})
                else:
                    self._send_json(500, {"status": "error", "message": "Failed to connect"})

            elif parsed.path == "/agent/unregister":
                agent_id = data.get("agent_id")
                bridge.unregister_agent(agent_id)
                self._send_json(200, {"status": "unregistered", "agent_id": agent_id})

            elif parsed.path == "/discover":
                bridge.discover_agents()
                self._send_json(200, {
                    "status": "discovery_complete",
                    "agents": list(bridge.agents.keys())
                })

            else:
                self._send_json(404, {"error": "Not found"})

        def _send_json(self, status: int, data: dict):
            self.send_response(status)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())

        def log_message(self, format, *args):
            logger.debug(f"[HTTP] {format % args}")

    return BridgeHandler


def run_http_server(bridge: GameBridge):
    """Run the HTTP server"""
    try:
        handler = create_http_server(bridge)
        server = HTTPServer(("0.0.0.0", bridge.port), handler)
        bridge.http_server_started = True
        logger.info(f"HTTP server started on port {bridge.port}")
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("HTTP server stopped")
    except Exception as e:
        logger.error(f"HTTP server error: {e}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point - Run on the host"""
    import argparse

    parser = argparse.ArgumentParser(description="StarCraft 2 Game Bridge")
    parser.add_argument("--port", type=int, default=8000, help="Port for the bridge server")
    parser.add_argument("--map", type=str, default="Simple64", help="Map name")
    parser.add_argument("--no-game", action="store_true", help="Run HTTP server only (no SC2)")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("StarCraft 2 Game Bridge - Host Component")
    logger.info("=" * 60)
    logger.info("This script runs on the HOST machine with:")
    logger.info("  - PySC2 installed")
    logger.info("  - StarCraft 2 game installed")
    logger.info("")
    logger.info("AI agents run in Docker containers and connect via HTTP")
    logger.info("=" * 60)

    # Create bridge
    bridge = GameBridge(port=args.port, map_name=args.map)
    
    # Start HTTP server FIRST (in a thread)
    http_thread = threading.Thread(target=run_http_server, args=(bridge,), daemon=True)
    http_thread.start()
    
    # Wait for HTTP server to start
    time.sleep(1)
    
    if bridge.http_server_started:
        logger.info(f"✅ HTTP server running at http://localhost:{args.port}")
    
    # Then start game (or skip if --no-game)
    bridge.start(skip_game=args.no_game)
    
    # Try to discover agents
    logger.info("Looking for AI agents...")
    time.sleep(2)
    bridge.discover_agents()
    
    if bridge.agents:
        logger.info(f"✅ Found {len(bridge.agents)} agent(s)")
    else:
        logger.info("⏳ No agents found yet - they will be discovered when available")
    
    logger.info("")
    logger.info(f"🎮 Game Bridge running at http://localhost:{args.port}")
    logger.info("📊 Endpoints:")
    logger.info(f"   GET  /health     - Health check")
    logger.info(f"   GET  /game/state - Current game state")
    logger.info(f"   GET  /agents     - List connected agents")
    logger.info(f"   POST /discover   - Discover agents")
    logger.info("")
    logger.info("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(5)
            # Periodically try to discover agents
            bridge.discover_agents()
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
        bridge.stop()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
