"""
StarCraft 2 AI Runner - Runs 2 AI agents against each other
This script runs on the host and creates a single SC2 game with 2 AI players.
"""

import os
import sys
import time
import logging
from typing import List

# IMPORTANT: Must import absl and parse flags before importing pysc2
from absl import app, flags

FLAGS = flags.FLAGS

# Define our own flags
flags.DEFINE_string("map", "Simple64", "Map name")
flags.DEFINE_integer("episodes", 1, "Number of episodes")
flags.DEFINE_integer("steps", 0, "Max steps per episode (0=unlimited)")
flags.DEFINE_integer("step_mul", 8, "Game steps per agent step")
flags.DEFINE_bool("realtime", False, "Run in realtime")
flags.DEFINE_bool("visualize", True, "Enable visualization")
flags.DEFINE_string("sc2_path", r"T:\act\StarCraft II", "Path to SC2")
flags.DEFINE_list("window_size", "1280,720", "SC2 window size as width,height")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SC2Runner] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PySC2 imports (after absl flags are defined)
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import point
from pysc2.lib import actions, features


class SimpleAgent(base_agent.BaseAgent):
    """A simple scripted agent"""
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.action_count = 0
        
    def step(self, obs):
        """Execute one game step"""
        super().step(obs)
        self.action_count += 1
        
        # Simple logic: just do no_op for now
        # Can be extended with actual strategy
        if self.action_count % 100 == 0:
            logger.info(f"Agent {self.agent_id}: Step {self.action_count}")
        
        return actions.FUNCTIONS.no_op()


def run_game(
    map_name: str = "Simple64",
    max_episodes: int = 1,
    max_steps: int = 0,
    step_mul: int = 8,
    realtime: bool = False,
    visualize: bool = True,
):
    """Run a game with 2 AI agents"""
    
    logger.info(f"Starting SC2 game on map: {map_name}")
    logger.info(f"SC2PATH: {os.environ.get('SC2PATH', 'Not set')}")
    
    # Create agents
    agent1 = SimpleAgent(agent_id=1)
    agent2 = SimpleAgent(agent_id=2)
    agents = [agent1, agent2]
    
    try:
        # Create environment with 2 agents
        # Parse window size
        window_size = point.Point(int(FLAGS.window_size[0]), int(FLAGS.window_size[1]))
        logger.info(f"Window size: {window_size.x}x{window_size.y}")
        
        with sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.protoss, name="Agent1"),
                sc2_env.Agent(sc2_env.Race.zerg, name="Agent2"),
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True,
            ),
            step_mul=step_mul,
            game_steps_per_episode=max_steps,
            realtime=realtime,
            visualize=visualize,
            window_size=window_size,
        ) as env:
            
            logger.info("SC2 environment created successfully!")
            
            for episode in range(max_episodes):
                logger.info(f"Starting episode {episode + 1}/{max_episodes}")
                
                # Reset environment
                timesteps = env.reset()
                
                # Reset agents
                for agent in agents:
                    agent.reset()
                
                episode_reward = [0, 0]
                step_count = 0
                
                # Game loop
                while not timesteps[0].last():
                    step_count += 1
                    
                    # Get actions from both agents
                    agent_actions = []
                    for i, agent in enumerate(agents):
                        action = agent.step(timesteps[i])
                        agent_actions.append(action)
                    
                    # Execute actions
                    timesteps = env.step(agent_actions)
                    
                    # Track rewards
                    for i in range(2):
                        episode_reward[i] += timesteps[i].reward
                
                logger.info(f"Episode {episode + 1} complete after {step_count} steps")
                logger.info(f"  Agent 1 reward: {episode_reward[0]}")
                logger.info(f"  Agent 2 reward: {episode_reward[1]}")
                
                # Determine winner
                if episode_reward[0] > episode_reward[1]:
                    logger.info("  Winner: Agent 1 (Protoss)")
                elif episode_reward[1] > episode_reward[0]:
                    logger.info("  Winner: Agent 2 (Zerg)")
                else:
                    logger.info("  Result: Draw")
                    
    except Exception as e:
        logger.error(f"Error running game: {e}")
        raise


def main(argv):
    """Main entry point"""
    del argv  # Unused
    
    # Set SC2PATH environment variable
    os.environ["SC2PATH"] = FLAGS.sc2_path
    logger.info(f"SC2PATH set to: {FLAGS.sc2_path}")
    
    # Run the game
    run_game(
        map_name=FLAGS.map,
        max_episodes=FLAGS.episodes,
        max_steps=FLAGS.steps,
        step_mul=FLAGS.step_mul,
        realtime=FLAGS.realtime,
        visualize=FLAGS.visualize,
    )


if __name__ == "__main__":
    app.run(main)
