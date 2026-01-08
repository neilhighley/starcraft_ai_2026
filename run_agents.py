"""
StarCraft 2 AI Runner - Runs 2 AI agents against each other
This script runs on the host and creates a single SC2 game with 2 AI players.

=============================================================================
AVAILABLE MAPS (use --map=<name>)
=============================================================================
Melee Maps (simple ML training maps):
  1. Simple64    - 64x64, balanced with expansions and ramps (DEFAULT)
  2. Simple96    - 96x96, larger version
  3. Simple128   - 128x128, largest simple map
  4. Flat32      - 32x32, no terrain features
  5. Flat48      - 48x48, flat terrain
  6. Flat64      - 64x64, flat terrain
  7. Flat96      - 96x96, flat terrain
  8. Flat128     - 128x128, flat terrain
  9. Empty128    - 128x128, completely empty

Ladder Maps (competitive maps, more complex):
  10. AbyssalReefLE
  11. BelShirVestigeLE
  12. CactusValleyLE
  13. HonorgroundsLE
  14. NewkirkPrecinctTE
  15. PaladinoTerminalLE
  16. ProximaStationLE

=============================================================================
CHANGING AGENT MODELS
=============================================================================
To use different AI agents, modify this file:

1. SIMPLE EDIT - Change the agent class instantiation (line ~75):
   Replace:
       agent1 = SimpleAgent(agent_id=1)
       agent2 = SimpleAgent(agent_id=2)
   With your custom agents:
       agent1 = YourCustomAgent(agent_id=1)
       agent2 = YourCustomAgent(agent_id=2)

2. CREATE CUSTOM AGENT - Subclass BaseAgent:
   class MyAgent(base_agent.BaseAgent):
       def step(self, obs):
           super().step(obs)
           # Your logic here - analyze obs, return action
           return actions.FUNCTIONS.no_op()

3. LOAD TRAINED MODEL - Example with PyTorch:
   class NeuralAgent(base_agent.BaseAgent):
       def __init__(self, model_path):
           super().__init__()
           self.model = torch.load(model_path)
       
       def step(self, obs):
           state = self.preprocess(obs)
           action = self.model(state)
           return action

4. USE DIFFERENT AGENTS FOR EACH PLAYER:
   agent1 = AggressiveAgent()   # Attacks early
   agent2 = DefensiveAgent()    # Builds up economy

5. CHANGE RACES - Modify players list in SC2Env (line ~90):
   Available: sc2_env.Race.terran, .protoss, .zerg, .random
=============================================================================
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

# Model selection flags
flags.DEFINE_string("model1", "simple", "Model for Agent 1 (see --list_models)")
flags.DEFINE_string("model2", "simple", "Model for Agent 2 (see --list_models)")
flags.DEFINE_bool("list_models", False, "List available models and exit")
flags.DEFINE_bool("download_models", False, "Download all pretrained models")
flags.DEFINE_bool("interactive", False, "Interactive model selection")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SC2Runner] - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# PySC2 imports (after absl flags are defined)
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions, features

# Add parent dir to path for model_registry import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_agent.model_registry import ModelRegistry, MODEL_ALIASES


# =============================================================================
# AVAILABLE MODELS (use --model1=<name> --model2=<name>)
# =============================================================================
# Built-in:
#   simple     - Basic no-op agent (default)
#   random     - Random action selection
#   heuristic  - Rule-based AI with basic strategies
#
# Pretrained (auto-download):
#   zerg_rush      - Aggressive Zerg rush strategy
#   terran_macro   - Macro-focused Terran strategy  
#   protoss_gateway - Gateway-based Protoss strategy
#
# RL Algorithms:
#   ppo  - Proximal Policy Optimization
#   dqn  - Deep Q-Network
#   a2c  - Advantage Actor-Critic
#
# Custom: Provide path to .pt file
# =============================================================================


class SimpleAgent(base_agent.BaseAgent):
    """A simple scripted agent"""
    
    def __init__(self, agent_id: int = 1, name: str = "Simple"):
        super().__init__()
        self.agent_id = agent_id
        self.name = name
        self.action_count = 0
        
    def step(self, obs):
        """Execute one game step"""
        super().step(obs)
        self.action_count += 1
        
        # Simple logic: just do no_op for now
        # Can be extended with actual strategy
        if self.action_count % 100 == 0:
            logger.info(f"Agent {self.agent_id} ({self.name}): Step {self.action_count}")
        
        return actions.FUNCTIONS.no_op()


class RandomAgent(base_agent.BaseAgent):
    """Agent that selects random valid actions"""
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.name = "Random"
        
    def step(self, obs):
        """Execute random valid action"""
        super().step(obs)
        import random
        
        # Get available actions
        available_actions = obs.observation.available_actions
        
        if len(available_actions) > 0:
            # Select random action
            action_id = random.choice(available_actions)
            action_func = actions.FUNCTIONS[action_id]
            
            # Build action with random arguments
            args = []
            for arg in action_func.args:
                if arg.name == 'screen' or arg.name == 'minimap':
                    # Random point on screen/minimap
                    args.append([random.randint(0, 83), random.randint(0, 83)])
                elif arg.name == 'screen2':
                    args.append([random.randint(0, 83), random.randint(0, 83)])
                elif arg.name == 'queued':
                    args.append([random.randint(0, 1)])
                elif arg.name == 'select_point_act':
                    args.append([random.randint(0, 3)])
                elif arg.name == 'select_add':
                    args.append([random.randint(0, 1)])
                elif arg.name == 'control_group_act':
                    args.append([random.randint(0, 4)])
                elif arg.name == 'control_group_id':
                    args.append([random.randint(0, 9)])
                elif arg.name == 'select_unit_act':
                    args.append([random.randint(0, 3)])
                elif arg.name == 'select_unit_id':
                    args.append([0])  # First unit
                elif arg.name == 'select_worker':
                    args.append([random.randint(0, 3)])
                elif arg.name == 'build_queue_id':
                    args.append([0])
                elif arg.name == 'unload_id':
                    args.append([0])
                else:
                    args.append([0])
            
            return action_func(*args)
        
        return actions.FUNCTIONS.no_op()


class HeuristicAgent(base_agent.BaseAgent):
    """Rule-based agent with basic strategies"""
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.name = "Heuristic"
        self.base_built = False
        
    def step(self, obs):
        """Execute heuristic-based action"""
        super().step(obs)
        
        # Simple heuristic: train workers early, then attack
        available = obs.observation.available_actions
        
        # Priority: train workers if possible
        if actions.FUNCTIONS.Train_Probe_quick.id in available:
            return actions.FUNCTIONS.Train_Probe_quick("now")
        if actions.FUNCTIONS.Train_SCV_quick.id in available:
            return actions.FUNCTIONS.Train_SCV_quick("now")
        if actions.FUNCTIONS.Train_Drone_quick.id in available:
            return actions.FUNCTIONS.Train_Drone_quick("now")
            
        # Attack if possible
        if actions.FUNCTIONS.Attack_minimap.id in available:
            # Attack enemy base (approximate location)
            return actions.FUNCTIONS.Attack_minimap("now", [50, 50])
        
        return actions.FUNCTIONS.no_op()


class PretrainedAgent(base_agent.BaseAgent):
    """Agent that uses a pretrained neural network model"""
    
    def __init__(self, agent_id: int = 1, model_path: str = None, model_data: dict = None):
        super().__init__()
        self.agent_id = agent_id
        self.model_path = model_path
        self.model_data = model_data
        self.name = f"Pretrained({os.path.basename(model_path) if model_path else 'loaded'})"
        
        # Load model
        if model_path and not model_data:
            import torch
            self.model_data = torch.load(model_path, map_location='cpu')
        
        logger.info(f"Pretrained agent {agent_id} loaded")
        
    def step(self, obs):
        """Execute action from pretrained model"""
        super().step(obs)
        
        # TODO: Implement model inference
        # This is a placeholder - actual implementation depends on model architecture
        # For now, fall back to no_op
        return actions.FUNCTIONS.no_op()


def create_agent(model_name: str, agent_id: int, registry: ModelRegistry) -> base_agent.BaseAgent:
    """Factory function to create agent based on model name"""
    model_name = model_name.lower()
    
    if model_name == "simple":
        return SimpleAgent(agent_id=agent_id, name="Simple")
    
    elif model_name == "random":
        return RandomAgent(agent_id=agent_id)
    
    elif model_name == "heuristic":
        return HeuristicAgent(agent_id=agent_id)
    
    elif model_name in MODEL_ALIASES:
        config = MODEL_ALIASES[model_name]
        
        if config["type"] in ["pretrained", "rl"]:
            # Try to load the model
            try:
                model_data = registry.load_model(model_name, agent_id)
                model_path = registry.models_dir / config["path"].split("/")[-1]
                return PretrainedAgent(agent_id=agent_id, model_path=str(model_path), model_data=model_data)
            except Exception as e:
                logger.warning(f"Failed to load model {model_name}: {e}")
                logger.info("Falling back to SimpleAgent")
                return SimpleAgent(agent_id=agent_id, name=f"Simple(fallback from {model_name})")
        else:
            # Heuristic or random type
            if config["type"] == "heuristic":
                return HeuristicAgent(agent_id=agent_id)
            else:
                return RandomAgent(agent_id=agent_id)
    
    elif os.path.exists(model_name):
        # Custom model path
        return PretrainedAgent(agent_id=agent_id, model_path=model_name)
    
    else:
        logger.warning(f"Unknown model: {model_name}, using SimpleAgent")
        return SimpleAgent(agent_id=agent_id, name="Simple")


def run_game(
    map_name: str = "Simple64",
    max_episodes: int = 12,
    max_steps: int = 0,
    step_mul: int = 8,
    realtime: bool = False,
    visualize: bool = True,
    model1: str = "simple",
    model2: str = "simple",
):
    """Run a game with 2 AI agents"""
    
    logger.info(f"Starting SC2 game on map: {map_name}")
    logger.info(f"SC2PATH: {os.environ.get('SC2PATH', 'Not set')}")
    logger.info(f"Agent 1 model: {model1}")
    logger.info(f"Agent 2 model: {model2}")
    
    # Initialize model registry
    registry = ModelRegistry(models_dir="ai_agent/models")
    
    # Create agents based on model selection
    agent1 = create_agent(model1, agent_id=1, registry=registry)
    agent2 = create_agent(model2, agent_id=2, registry=registry)
    agents = [agent1, agent2]
    
    logger.info(f"Agent 1: {agent1.name}")
    logger.info(f"Agent 2: {agent2.name}")
    
    try:
        # Create environment with 2 agents
        with sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.protoss, name="SKYNET"),
                sc2_env.Agent(sc2_env.Race.zerg, name="BUGS"),
            ],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True,
            ),
            step_mul=step_mul,
            game_steps_per_episode=max_steps,
            realtime=realtime,
            visualize=visualize,
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


def list_available_models():
    """Print list of available models"""
    registry = ModelRegistry(models_dir="ai_agent/models")
    models = registry.get_available_models()
    
    print("\n" + "=" * 70)
    print("🤖 AVAILABLE MODELS")
    print("=" * 70)
    
    # Built-in agents
    print("\n📦 BUILT-IN AGENTS (always available):")
    print(f"  {'simple':20s} - Basic no-op agent (default)")
    print(f"  {'random':20s} - Random action selection")
    print(f"  {'heuristic':20s} - Rule-based AI with basic strategies")
    
    # Pretrained models
    print("\n🧠 PRETRAINED MODELS (auto-download):")
    for model in models:
        if model.get("type") == "pretrained":
            status = "✅" if model.get("available") else "⬇️"
            print(f"  {status} {model['name']:18s} - {model.get('description', 'No description')}")
    
    # RL models
    print("\n🎮 RL ALGORITHM MODELS:")
    for model in models:
        if model.get("type") == "rl":
            status = "✅" if model.get("available") else "⬇️"
            print(f"  {status} {model['name']:18s} - {model.get('description', 'No description')}")
    
    # Custom models
    custom_models = [m for m in models if m.get("type") == "custom"]
    if custom_models:
        print("\n📁 CUSTOM MODELS (in ai_agent/models/):")
        for model in custom_models:
            print(f"  ✅ {model['name']:18s} - {model.get('description', '')}")
    
    print("\n" + "=" * 70)
    print("USAGE: python run_agents.py --model1=<name> --model2=<name>")
    print("       python run_agents.py --model1=ppo --model2=random")
    print("       python run_agents.py --model1=/path/to/custom.pt --model2=simple")
    print("=" * 70 + "\n")


def download_all_models():
    """Download all pretrained models"""
    registry = ModelRegistry(models_dir="ai_agent/models")
    
    print("\n" + "=" * 70)
    print("📥 DOWNLOADING ALL PRETRAINED MODELS")
    print("=" * 70 + "\n")
    
    for name, config in MODEL_ALIASES.items():
        if config.get("type") in ["pretrained", "rl"] and "download_url" in config:
            print(f"\nDownloading: {name}")
            registry.download_model(name)
    
    print("\n✅ Download complete!\n")


def interactive_model_selection(registry: ModelRegistry):
    """Interactive model selection"""
    print("\n🎮 INTERACTIVE MODEL SELECTION")
    model1 = registry.select_model_interactive(agent_id=1)
    model2 = registry.select_model_interactive(agent_id=2)
    return model1, model2


def main(argv):
    """Main entry point"""
    del argv  # Unused
    
    # Handle special flags first
    if FLAGS.list_models:
        list_available_models()
        return
    
    if FLAGS.download_models:
        download_all_models()
        return
    
    # Set SC2PATH environment variable
    os.environ["SC2PATH"] = FLAGS.sc2_path
    logger.info(f"SC2PATH set to: {FLAGS.sc2_path}")
    
    # Get model selections
    model1 = FLAGS.model1
    model2 = FLAGS.model2
    
    # Interactive selection if requested
    if FLAGS.interactive:
        registry = ModelRegistry(models_dir="ai_agent/models")
        model1, model2 = interactive_model_selection(registry)
    
    # Run the game
    run_game(
        map_name=FLAGS.map,
        max_episodes=FLAGS.episodes,
        max_steps=FLAGS.steps,
        step_mul=FLAGS.step_mul,
        realtime=FLAGS.realtime,
        visualize=FLAGS.visualize,
        model1=model1,
        model2=model2,
    )


if __name__ == "__main__":
    app.run(main)
