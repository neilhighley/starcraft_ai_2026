"""
StarCraft 2 AI Agent with monitoring
Connects to game running on host and provides metrics via Prometheus
"""

import os
import sys
import time
import asyncio
import argparse
import signal
from typing import Dict, List, Optional
import logging

# Add parent directory to path
sys.path.append("/app")

# PySC2 imports
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.lib.point import Point

# Local imports
try:
    from monitoring.monitor import Monitoring, get_monitoring
    import uvicorn
except ImportError:
    Monitoring = None
    get_monitoring = None
    uvicorn = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MonitoredAI(base_agent.BaseAgent):
    """AI agent with monitoring capabilities"""

    def __init__(self, agent_id: str = "1"):
        super().__init__()
        self.agent_id = agent_id
        self.monitoring = get_monitoring() if get_monitoring else None
        self.action_count = 0
        self.game_step = 0
        logger.info(f"AI Agent {agent_id} initialized")

    def reset(self):
        """Reset the agent for a new game"""
        super().reset()
        self.action_count = 0
        self.game_step = 0

        if self.monitoring:
            asyncio.create_task(self._notify_game_start())

        logger.info(f"Agent {self.agent_id} reset for new game")

    async def _notify_game_start(self):
        """Notify monitoring that a game has started"""
        if self.monitoring:
            try:
                import httpx

                async with httpx.AsyncClient() as client:
                    await client.post("http://localhost:8080/game/start")
            except Exception as e:
                logger.error(f"Failed to notify game start: {e}")

    def step(self, obs):
        """Execute one game step"""
        start_time = time.time()
        self.game_step += 1

        try:
            # Update system metrics
            if self.monitoring:
                self.monitoring.update_system_metrics()

            # Check if this is the first step
            if obs.first():
                return self._handle_first_step(obs)

            # Get observation
            player_common = obs.observation["player_common"]

            # Record resources
            if self.monitoring:
                self.monitoring.record_minerals(player_common.minerals)
                self.monitoring.record_vespene(player_common.vespene)

            # Simple strategy: Build workers, army, and attack
            action = self._decide_action(obs)

            # Record action
            if self.monitoring:
                self.monitoring.record_action(
                    action.action_name if hasattr(action, "action_name") else "unknown"
                )

            self.action_count += 1

            # Record decision time
            decision_duration = time.time() - start_time
            if self.monitoring:
                self.monitoring.record_decision_time(decision_duration)

            return action

        except Exception as e:
            logger.error(f"Error in step: {e}")
            if self.monitoring:
                self.monitoring.record_error("step_error")
            return actions.FUNCTIONS.no_op()

    def _handle_first_step(self, obs):
        """Handle the first observation"""
        # Center camera on base
        player_y, player_x = (
            obs.observation.feature_minimap.player_relative
            == features.PlayerRelative.SELF
        ).nonzero()

        if len(player_x) > 0:
            x_mean = int(player_x.mean())
            y_mean = int(player_y.mean())
            return actions.FUNCTIONS.move_camera([x_mean, y_mean])

        return actions.FUNCTIONS.no_op()

    def _decide_action(self, obs):
        """Decide what action to take"""
        player_common = obs.observation["player_common"]
        minerals = player_common.minerals
        vespene = player_common.vespene

        # Get idle workers
        idle_workers = self._get_idle_workers(obs)

        # Get army units
        army = self._get_army(obs)

        # Prioritize actions
        if idle_workers > 0 and minerals >= 50:
            return actions.FUNCTIONS.Harvest_Gather_worker(
                "select_idle_worker", Point(0, 0), Point(0, 0)
            )

        # Build workers if we have minerals and idle nexus
        if minerals >= 50 and self._can_build_worker(obs):
            return actions.FUNCTIONS.Train_Probe_quick("now")

        # Build army if we have resources
        if minerals >= 100 and vespene >= 50:
            unit_type = self._get_random_unit_type()
            return actions.FUNCTIONS.Train_unit_quick("now", unit_type)

        # Attack if we have army
        if len(army) >= 5:
            return actions.FUNCTIONS.Attack_minimap("now", (32, 32))

        return actions.FUNCTIONS.no_op()

    def _get_idle_workers(self, obs) -> int:
        """Get count of idle workers"""
        idle_workers = 0
        for unit in obs.observation.feature_units:
            if unit.unit_type == units.Protoss.Probe and unit.orders.length == 0:
                idle_workers += 1
        return idle_workers

    def _get_army(self, obs) -> List:
        """Get army units"""
        army = []
        for unit in obs.observation.feature_units:
            if unit.unit_type in [units.Protoss.Zealot, units.Protoss.Stalker]:
                army.append(unit)
        return army

    def _can_build_worker(self, obs) -> bool:
        """Check if we can build a worker"""
        for unit in obs.observation.feature_units:
            if unit.unit_type == units.Protoss.Nexus:
                return unit.order_length == 0
        return False

    def _get_random_unit_type(self):
        """Get a random unit type to train"""
        import random

        unit_types = [units.Protoss.Zealot, units.Protoss.Stalker, units.Protoss.Sentry]
        return random.choice(unit_types)


class AIOrchestrator:
    """Orchestrates the AI agent and monitoring server"""

    def __init__(self, agent_id: str = "1"):
        self.agent_id = agent_id
        self.agent = MonitoredAI(agent_id)
        self.monitoring = get_monitoring() if get_monitoring else None
        self.running = False

        # Game environment will connect to host
        self.env = None

        logger.info(f"AI Orchestrator {agent_id} initialized")

    async def start(self):
        """Start the AI agent and monitoring server"""
        logger.info("Starting AI Orchestrator...")
        self.running = True

        # Start monitoring server if available
        if self.monitoring and uvicorn:
            logger.info("Starting monitoring server...")
            config = uvicorn.Config(
                self.monitoring.get_app(), host="0.0.0.0", port=8080, log_level="info"
            )
            server = uvicorn.Server(config)
            asyncio.create_task(server.serve())

            # Start Prometheus metrics server
            self.monitoring.start_metrics_server(port=9090)

        # Connect to game on host
        await self._connect_to_game()

    async def _connect_to_game(self):
        """Connect to game server on host"""
        if not self.monitoring:
            logger.warning("Monitoring not available, skipping game connection check")
            return

        server_info = self.monitoring.get_game_server_info()
        logger.info(f"Attempting to connect to game at {server_info['url']}")

        # Check connection
        connected = self.monitoring.check_game_connection()
        if connected:
            logger.info("Successfully connected to game server")
        else:
            logger.warning(f"Could not connect to game server at {server_info['url']}")
            logger.info("The game should be running on the host")

    def setup_environment(self, map_name: str = "Simple64"):
        """Setup the SC2 environment"""
        try:
            self.env = sc2_env.SC2Env(
                map_name=map_name,
                players=[sc2_env.Agent(sc2_env.Race.protoss)],
                agent_interface_format=sc2_env.parse_agent_interface_format(
                    feature_screen=64, feature_minimap=64
                ),
                step_mul=16,
                game_steps_per_episode=0,
                realtime=False,
                visualize=False,
            )
            self.agent.setup(self.env.observation_spec()[0], self.env.action_spec()[0])
            logger.info(f"Environment setup for map {map_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to setup environment: {e}")
            if self.monitoring:
                self.monitoring.record_error("environment_setup")
            return False

    def run_episode(self):
        """Run one episode/game"""
        if not self.env:
            logger.error("Environment not setup")
            return

        logger.info("Starting episode...")
        timesteps = self.env.reset()
        self.agent.reset()

        episode_rewards = 0
        while True:
            # Get action from agent
            action = self.agent.step(timesteps[0])

            # Step environment
            if timesteps[0].last():
                break

            timesteps = self.env.step([action])
            episode_rewards += timesteps[0].reward

        logger.info(f"Episode complete, total reward: {episode_rewards}")
        return episode_rewards

    def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping AI Orchestrator...")
        self.running = False

        if self.env:
            self.env.close()

        logger.info("AI Orchestrator stopped")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StarCraft 2 AI Agent with Monitoring")
    parser.add_argument(
        "--agent-id", type=str, default=os.getenv("AGENT_ID", "1"), help="Agent ID"
    )
    parser.add_argument("--map", type=str, default="Simple64", help="Map name")
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of episodes to run"
    )
    args = parser.parse_args()

    logger.info(f"Starting SC2 AI Agent {args.agent_id}")

    orchestrator = AIOrchestrator(args.agent_id)
    await orchestrator.start()

    # Setup environment
    if orchestrator.setup_environment(args.map):
        # Run episodes
        for i in range(args.episodes):
            logger.info(f"Running episode {i + 1}/{args.episodes}")
            reward = orchestrator.run_episode()
            logger.info(f"Episode {i + 1} complete, reward: {reward}")

            # Small delay between episodes
            await asyncio.sleep(1)

    orchestrator.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
