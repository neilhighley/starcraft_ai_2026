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
import random
import logging
from typing import List

# IMPORTANT: Must import absl and parse flags before importing pysc2
from absl import app, flags

FLAGS = flags.FLAGS

# Define our own flags
flags.DEFINE_string("map", "Simple64", "Map name")
flags.DEFINE_integer("episodes", 2, "Number of episodes")
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

# Unit type IDs (from pysc2.lib.units)
from pysc2.lib import units

# Common unit type IDs
UNIT_TYPES = {
    # Protoss
    'nexus': 59,
    'probe': 84,
    'pylon': 60,
    'gateway': 62,
    'zealot': 73,
    'cyberneticscore': 72,
    'stalker': 74,
    'assimilator': 61,
    # Zerg
    'hatchery': 86,
    'drone': 104,
    'overlord': 106,
    'spawningpool': 89,
    'zergling': 105,
    'queen': 126,
    'extractor': 88,
    'larva': 151,
    # Terran
    'commandcenter': 18,
    'scv': 45,
    'supplydepot': 19,
    'barracks': 21,
    'marine': 48,
    'refinery': 20,
}


def get_units_by_type(obs, unit_type_id, alliance=1):
    """Get units of a specific type. alliance: 1=self, 4=enemy"""
    units_list = []
    if hasattr(obs.observation, 'feature_units'):
        for unit in obs.observation.feature_units:
            if unit.unit_type == unit_type_id and unit.alliance == alliance:
                units_list.append(unit)
    return units_list


def get_my_units(obs):
    """Get all my units"""
    units_list = []
    if hasattr(obs.observation, 'feature_units'):
        for unit in obs.observation.feature_units:
            if unit.alliance == 1:  # Self
                units_list.append(unit)
    return units_list


class SimpleAgent(base_agent.BaseAgent):
    """A simple scripted agent that does basic macro"""
    
    def __init__(self, agent_id: int = 1, name: str = "Simple"):
        super().__init__()
        self.agent_id = agent_id
        self.name = name
        self.action_count = 0
        
    def step(self, obs):
        """Execute one game step"""
        super().step(obs)
        self.action_count += 1
        
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
        
        available_actions = obs.observation.available_actions
        
        if len(available_actions) > 0:
            action_id = random.choice(available_actions)
            action_func = actions.FUNCTIONS[action_id]
            
            args = []
            for arg in action_func.args:
                if arg.name in ['screen', 'minimap', 'screen2']:
                    args.append([random.randint(0, 83), random.randint(0, 83)])
                elif arg.name == 'queued':
                    args.append([0])
                else:
                    args.append([0])
            
            return action_func(*args)
        
        return actions.FUNCTIONS.no_op()


class ScriptedProtossAgent(base_agent.BaseAgent):
    """Scripted Protoss agent that builds units and attacks"""
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.name = "ScriptedProtoss"
        self.step_count = 0
        self.attack_sent = False
        
    def step(self, obs):
        super().step(obs)
        self.step_count += 1
        
        available = obs.observation.available_actions
        minerals = obs.observation.player.minerals
        food_used = obs.observation.player.food_used
        food_cap = obs.observation.player.food_cap
        
        # Log every 50 steps
        if self.step_count % 50 == 0:
            logger.info(f"Protoss Agent: Step {self.step_count}, Minerals: {minerals}, Supply: {food_used}/{food_cap}")
        
        # Priority 1: Build Pylon if supply blocked
        if food_cap - food_used <= 2 and minerals >= 100:
            if actions.FUNCTIONS.Build_Pylon_screen.id in available:
                # Build pylon at random valid location
                return actions.FUNCTIONS.Build_Pylon_screen("now", [30 + (self.step_count % 20), 30])
        
        # Priority 2: Build Gateway
        if minerals >= 150:
            if actions.FUNCTIONS.Build_Gateway_screen.id in available:
                return actions.FUNCTIONS.Build_Gateway_screen("now", [40, 35])
        
        # Priority 3: Train Zealots
        if minerals >= 100:
            if actions.FUNCTIONS.Train_Zealot_quick.id in available:
                return actions.FUNCTIONS.Train_Zealot_quick("now")
        
        # Priority 4: Train Probes (up to 20)
        if minerals >= 50 and food_used < 20:
            if actions.FUNCTIONS.Train_Probe_quick.id in available:
                return actions.FUNCTIONS.Train_Probe_quick("now")
        
        # Priority 5: Select idle workers to mine
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
        
        # Priority 6: Send idle workers to harvest
        if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
            return actions.FUNCTIONS.Harvest_Gather_screen("now", [40, 40])
        
        # Priority 7: Attack when we have army
        if food_used >= 30 and not self.attack_sent:
            if actions.FUNCTIONS.Attack_minimap.id in available:
                self.attack_sent = True
                logger.info("Protoss Agent: Sending attack!")
                return actions.FUNCTIONS.Attack_minimap("now", [50, 50])
        
        # Priority 8: Select army
        if actions.FUNCTIONS.select_army.id in available:
            return actions.FUNCTIONS.select_army("select")
        
        # Default: Select Nexus to train more units
        if actions.FUNCTIONS.select_point.id in available:
            return actions.FUNCTIONS.select_point("select", [30, 30])
        
        return actions.FUNCTIONS.no_op()


class ScriptedZergAgent(base_agent.BaseAgent):
    """Scripted Zerg agent that builds units and attacks"""
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.name = "ScriptedZerg"
        self.step_count = 0
        self.pool_built = False
        self.attack_sent = False
        
    def step(self, obs):
        super().step(obs)
        self.step_count += 1
        
        available = obs.observation.available_actions
        minerals = obs.observation.player.minerals
        food_used = obs.observation.player.food_used
        food_cap = obs.observation.player.food_cap
        larva_count = obs.observation.player.larva_count if hasattr(obs.observation.player, 'larva_count') else 3
        
        # Log every 50 steps
        if self.step_count % 50 == 0:
            logger.info(f"Zerg Agent: Step {self.step_count}, Minerals: {minerals}, Supply: {food_used}/{food_cap}")
        
        # Priority 1: Build Overlord if supply blocked
        if food_cap - food_used <= 2 and minerals >= 100:
            if actions.FUNCTIONS.Train_Overlord_quick.id in available:
                return actions.FUNCTIONS.Train_Overlord_quick("now")
        
        # Priority 2: Build Spawning Pool
        if minerals >= 200 and not self.pool_built:
            if actions.FUNCTIONS.Build_SpawningPool_screen.id in available:
                self.pool_built = True
                return actions.FUNCTIONS.Build_SpawningPool_screen("now", [35, 35])
        
        # Priority 3: Train Zerglings
        if minerals >= 50:
            if actions.FUNCTIONS.Train_Zergling_quick.id in available:
                return actions.FUNCTIONS.Train_Zergling_quick("now")
        
        # Priority 4: Train Drones (up to 16)
        if minerals >= 50 and food_used < 16:
            if actions.FUNCTIONS.Train_Drone_quick.id in available:
                return actions.FUNCTIONS.Train_Drone_quick("now")
        
        # Priority 5: Select Larva
        if actions.FUNCTIONS.select_larva.id in available:
            return actions.FUNCTIONS.select_larva("select")
        
        # Priority 6: Select idle workers to mine
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
        
        # Priority 7: Send idle workers to harvest
        if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
            return actions.FUNCTIONS.Harvest_Gather_screen("now", [40, 40])
        
        # Priority 8: Attack when we have army
        if food_used >= 30 and not self.attack_sent:
            if actions.FUNCTIONS.Attack_minimap.id in available:
                self.attack_sent = True
                logger.info("Zerg Agent: Sending attack!")
                return actions.FUNCTIONS.Attack_minimap("now", [50, 50])
        
        # Priority 9: Select army
        if actions.FUNCTIONS.select_army.id in available:
            return actions.FUNCTIONS.select_army("select")
        
        # Default: Select Hatchery
        if actions.FUNCTIONS.select_point.id in available:
            return actions.FUNCTIONS.select_point("select", [35, 35])
        
        return actions.FUNCTIONS.no_op()


class HeuristicAgent(base_agent.BaseAgent):
    """Generic heuristic agent that adapts to any race"""
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.name = "Heuristic"
        self.step_count = 0
        self.attack_sent = False
        
    def step(self, obs):
        super().step(obs)
        self.step_count += 1
        
        available = obs.observation.available_actions
        minerals = obs.observation.player.minerals
        food_used = obs.observation.player.food_used
        food_cap = obs.observation.player.food_cap
        
        # Supply building
        if food_cap - food_used <= 2 and minerals >= 100:
            # Try all race supply buildings
            if actions.FUNCTIONS.Build_Pylon_screen.id in available:
                return actions.FUNCTIONS.Build_Pylon_screen("now", [30 + (self.step_count % 10), 30])
            if actions.FUNCTIONS.Build_SupplyDepot_screen.id in available:
                return actions.FUNCTIONS.Build_SupplyDepot_screen("now", [30 + (self.step_count % 10), 30])
            if actions.FUNCTIONS.Train_Overlord_quick.id in available:
                return actions.FUNCTIONS.Train_Overlord_quick("now")
        
        # Build production
        if minerals >= 150:
            if actions.FUNCTIONS.Build_Gateway_screen.id in available:
                return actions.FUNCTIONS.Build_Gateway_screen("now", [40, 35])
            if actions.FUNCTIONS.Build_Barracks_screen.id in available:
                return actions.FUNCTIONS.Build_Barracks_screen("now", [40, 35])
            if actions.FUNCTIONS.Build_SpawningPool_screen.id in available:
                return actions.FUNCTIONS.Build_SpawningPool_screen("now", [40, 35])
        
        # Train army
        if minerals >= 50:
            if actions.FUNCTIONS.Train_Zealot_quick.id in available:
                return actions.FUNCTIONS.Train_Zealot_quick("now")
            if actions.FUNCTIONS.Train_Marine_quick.id in available:
                return actions.FUNCTIONS.Train_Marine_quick("now")
            if actions.FUNCTIONS.Train_Zergling_quick.id in available:
                return actions.FUNCTIONS.Train_Zergling_quick("now")
        
        # Train workers
        if minerals >= 50 and food_used < 20:
            if actions.FUNCTIONS.Train_Probe_quick.id in available:
                return actions.FUNCTIONS.Train_Probe_quick("now")
            if actions.FUNCTIONS.Train_SCV_quick.id in available:
                return actions.FUNCTIONS.Train_SCV_quick("now")
            if actions.FUNCTIONS.Train_Drone_quick.id in available:
                return actions.FUNCTIONS.Train_Drone_quick("now")
        
        # Select idle workers
        if actions.FUNCTIONS.select_idle_worker.id in available:
            return actions.FUNCTIONS.select_idle_worker("select")
        
        # Harvest
        if actions.FUNCTIONS.Harvest_Gather_screen.id in available:
            return actions.FUNCTIONS.Harvest_Gather_screen("now", [40, 40])
        
        # Attack
        if food_used >= 25 and not self.attack_sent:
            if actions.FUNCTIONS.Attack_minimap.id in available:
                self.attack_sent = True
                return actions.FUNCTIONS.Attack_minimap("now", [50, 50])
        
        # Select army
        if actions.FUNCTIONS.select_army.id in available:
            return actions.FUNCTIONS.select_army("select")
        
        # Select larva for Zerg
        if actions.FUNCTIONS.select_larva.id in available:
            return actions.FUNCTIONS.select_larva("select")
        
        # Select base
        if actions.FUNCTIONS.select_point.id in available:
            return actions.FUNCTIONS.select_point("select", [30, 30])
        
        return actions.FUNCTIONS.no_op()


class ZergBotAgent(base_agent.BaseAgent):
    """
    Advanced Zerg agent based on CharlieRuiz's ZergBot.
    Source: https://github.com/CharlieRuiz/IA/blob/main/ZergBot/ZergBot.py
    
    Features:
    - Builds spawning pool, extractors, lair, spire, hydralisk den, lurker den
    - Trains drones, zerglings, mutalisks, corruptors, hydralisks, lurkers
    - Expands to second base
    - Coordinates attacks with mixed army
    """
    
    def __init__(self, agent_id: int = 1):
        super().__init__()
        self.agent_id = agent_id
        self.name = "ZergBot"
        self.attack_coordinates = None
        self.safe_coordinates = None
        self.expand_coordinates = None
        self.ban = 0  # State flag for expansion logic
        self.hatch = True
        self.harvest_gas_mode = False
        
    def unit_type_is_selected(self, obs, unit_type):
        """Check if a specific unit type is selected"""
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True
        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False
    
    def get_units_by_type(self, obs, unit_type):
        """Get all units of a specific type"""
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]
    
    def can_do(self, obs, action):
        """Check if an action is available"""
        return action in obs.observation.available_actions
    
    def select_drone(self, obs):
        """Select a random drone"""
        drones = self.get_units_by_type(obs, units.Zerg.Drone)
        if len(drones) > 0:
            drone = random.choice(drones)
            if drone.x >= 0 and drone.y >= 0:
                return actions.FUNCTIONS.select_point("select_all_type", (drone.x, drone.y))
        return None
    
    def select_larva(self, obs):
        """Select larva for training"""
        larvae = self.get_units_by_type(obs, units.Zerg.Larva)
        if len(larvae) > 0:
            larva = random.choice(larvae)
            if larva.x >= 0 and larva.y >= 0:
                return actions.FUNCTIONS.select_point("select_all_type", (larva.x, larva.y))
        return None
    
    def build_spawning_pool(self, obs):
        """Build a spawning pool if none exists"""
        pools = self.get_units_by_type(obs, units.Zerg.SpawningPool)
        if len(pools) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_SpawningPool_screen.id):
                    x = random.randint(10, 50)
                    y = random.randint(10, 50)
                    return actions.FUNCTIONS.Build_SpawningPool_screen("now", (x, y))
            return self.select_drone(obs)
        return None
    
    def build_extractor(self, obs):
        """Build extractors on geysers"""
        extractors = self.get_units_by_type(obs, units.Zerg.Extractor)
        if len(extractors) < 2:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_Extractor_screen.id):
                    geysers = self.get_units_by_type(obs, units.Neutral.VespeneGeyser)
                    if len(geysers) > 0:
                        geyser = random.choice(geysers)
                        return actions.FUNCTIONS.Build_Extractor_screen("now", (geyser.x, geyser.y))
            return self.select_drone(obs)
        return None
    
    def build_lair(self, obs):
        """Morph hatchery into lair"""
        lairs = self.get_units_by_type(obs, units.Zerg.Lair)
        if len(lairs) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Hatchery):
                if self.can_do(obs, actions.FUNCTIONS.Morph_Lair_quick.id):
                    return actions.FUNCTIONS.Morph_Lair_quick("now")
            hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
            if len(hatcheries) > 0:
                hatch = random.choice(hatcheries)
                if hatch.x >= 0 and hatch.y >= 0:
                    return actions.FUNCTIONS.select_point("select_all_type", (hatch.x, hatch.y))
        return None
    
    def build_spire(self, obs):
        """Build a spire for air units"""
        spires = self.get_units_by_type(obs, units.Zerg.Spire)
        if len(spires) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_Spire_screen.id):
                    x = random.randint(10, 50)
                    y = random.randint(10, 50)
                    return actions.FUNCTIONS.Build_Spire_screen("now", (x, y))
            return self.select_drone(obs)
        return None
    
    def build_hydralisk_den(self, obs):
        """Build hydralisk den"""
        dens = self.get_units_by_type(obs, units.Zerg.HydraliskDen)
        if len(dens) == 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Build_HydraliskDen_screen.id):
                    x = random.randint(10, 50)
                    y = random.randint(10, 50)
                    return actions.FUNCTIONS.Build_HydraliskDen_screen("now", (x, y))
            return self.select_drone(obs)
        return None
    
    def train_unit(self, obs, unit_type):
        """Train a specific unit type from larva"""
        if self.unit_type_is_selected(obs, units.Zerg.Larva):
            free_supply = obs.observation.player.food_cap - obs.observation.player.food_used
            
            # Build overlords if supply blocked
            if free_supply < 4:
                if self.can_do(obs, actions.FUNCTIONS.Train_Overlord_quick.id):
                    return actions.FUNCTIONS.Train_Overlord_quick("now")
            
            if unit_type == "zergling":
                if self.can_do(obs, actions.FUNCTIONS.Train_Zergling_quick.id):
                    return actions.FUNCTIONS.Train_Zergling_quick("now")
            elif unit_type == "drone":
                if self.can_do(obs, actions.FUNCTIONS.Train_Drone_quick.id):
                    return actions.FUNCTIONS.Train_Drone_quick("now")
            elif unit_type == "mutalisk":
                if self.can_do(obs, actions.FUNCTIONS.Train_Mutalisk_quick.id):
                    return actions.FUNCTIONS.Train_Mutalisk_quick("now")
            elif unit_type == "hydralisk":
                if self.can_do(obs, actions.FUNCTIONS.Train_Hydralisk_quick.id):
                    return actions.FUNCTIONS.Train_Hydralisk_quick("now")
            elif unit_type == "corruptor":
                if self.can_do(obs, actions.FUNCTIONS.Train_Corruptor_quick.id):
                    return actions.FUNCTIONS.Train_Corruptor_quick("now")
        
        return self.select_larva(obs)
    
    def attack(self, obs):
        """Attack with army when ready"""
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        hydralisks = self.get_units_by_type(obs, units.Zerg.Hydralisk)
        mutalisks = self.get_units_by_type(obs, units.Zerg.Mutalisk)
        
        army_size = len(zerglings) + len(hydralisks) + len(mutalisks)
        
        if army_size >= 10:
            # Check if army is selected
            if (self.unit_type_is_selected(obs, units.Zerg.Zergling) or
                self.unit_type_is_selected(obs, units.Zerg.Hydralisk) or
                self.unit_type_is_selected(obs, units.Zerg.Mutalisk)):
                if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                    return actions.FUNCTIONS.Attack_minimap("now", self.attack_coordinates)
            
            # Select army
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")
        
        return None
    
    def harvest_minerals(self, obs):
        """Send drones to harvest minerals"""
        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
        lairs = self.get_units_by_type(obs, units.Zerg.Lair)
        
        if len(hatcheries) > 0 or len(lairs) > 0:
            if self.unit_type_is_selected(obs, units.Zerg.Drone):
                if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                    minerals = self.get_units_by_type(obs, units.Neutral.MineralField)
                    if len(minerals) > 0:
                        mineral = random.choice(minerals)
                        return actions.FUNCTIONS.Harvest_Gather_screen("now", (mineral.x, mineral.y))
            
            drones = self.get_units_by_type(obs, units.Zerg.Drone)
            if len(drones) > 0:
                drone = random.choice(drones)
                if drone.x >= 0 and drone.y >= 0:
                    return actions.FUNCTIONS.select_point("select", (drone.x, drone.y))
        
        return None
    
    def step(self, obs):
        super().step(obs)
        
        # Initialize attack coordinates based on spawn position
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative ==
                                  features.PlayerRelative.SELF).nonzero()
            
            if len(player_x) > 0 and len(player_y) > 0:
                xmean = player_x.mean()
                ymean = player_y.mean()
                
                if xmean <= 31 and ymean <= 31:
                    self.attack_coordinates = [49, 49]
                    self.safe_coordinates = [15, 15]
                    self.expand_coordinates = [49, 22]
                else:
                    self.attack_coordinates = [15, 15]
                    self.safe_coordinates = [49, 49]
                    self.expand_coordinates = [19, 49]
            else:
                # Default coordinates if detection fails
                self.attack_coordinates = [49, 49]
                self.safe_coordinates = [15, 15]
        
        # Priority 1: Attack if we have enough army
        attack_action = self.attack(obs)
        if attack_action:
            return attack_action
        
        hatcheries = self.get_units_by_type(obs, units.Zerg.Hatchery)
        lairs = self.get_units_by_type(obs, units.Zerg.Lair)
        
        # Priority 2: Build spawning pool
        if len(hatcheries) >= 1 or len(lairs) >= 1:
            pool_action = self.build_spawning_pool(obs)
            if pool_action:
                return pool_action
        
        # Priority 3: Train drones (up to 16)
        drones = self.get_units_by_type(obs, units.Zerg.Drone)
        if len(drones) < 16:
            train_action = self.train_unit(obs, "drone")
            if train_action:
                return train_action
        
        # Priority 4: Train zerglings
        zerglings = self.get_units_by_type(obs, units.Zerg.Zergling)
        if len(zerglings) < 12:
            train_action = self.train_unit(obs, "zergling")
            if train_action:
                return train_action
        
        # Priority 5: Build extractor
        extractor_action = self.build_extractor(obs)
        if extractor_action:
            return extractor_action
        
        # Priority 6: Build lair (enables advanced units)
        lair_action = self.build_lair(obs)
        if lair_action:
            return lair_action
        
        # Priority 7: Build hydralisk den
        lairs = self.get_units_by_type(obs, units.Zerg.Lair)
        if len(lairs) > 0:
            den_action = self.build_hydralisk_den(obs)
            if den_action:
                return den_action
            
            # Train hydralisks
            hydralisks = self.get_units_by_type(obs, units.Zerg.Hydralisk)
            if len(hydralisks) < 4:
                train_action = self.train_unit(obs, "hydralisk")
                if train_action:
                    return train_action
        
        # Priority 8: Build spire
        if len(lairs) > 0:
            spire_action = self.build_spire(obs)
            if spire_action:
                return spire_action
            
            # Train mutalisks
            spires = self.get_units_by_type(obs, units.Zerg.Spire)
            if len(spires) > 0:
                mutalisks = self.get_units_by_type(obs, units.Zerg.Mutalisk)
                if len(mutalisks) < 4:
                    train_action = self.train_unit(obs, "mutalisk")
                    if train_action:
                        return train_action
        
        # Priority 9: Harvest minerals
        harvest_action = self.harvest_minerals(obs)
        if harvest_action:
            return harvest_action
        
        # Priority 10: More zerglings
        train_action = self.train_unit(obs, "zergling")
        if train_action:
            return train_action
        
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
    
    elif model_name == "scripted_protoss" or model_name == "protoss":
        return ScriptedProtossAgent(agent_id=agent_id)
    
    elif model_name == "scripted_zerg" or model_name == "zerg":
        return ScriptedZergAgent(agent_id=agent_id)
    
    elif model_name == "zergbot":
        return ZergBotAgent(agent_id=agent_id)
    
    elif model_name in MODEL_ALIASES:
        config = MODEL_ALIASES[model_name]
        
        if config["type"] in ["pretrained", "rl"]:
            # Check if model file exists
            model_path = registry.models_dir / config["path"].split("/")[-1]
            
            if model_path.exists():
                # Load the trained model
                try:
                    model_data = registry.load_model(model_name, agent_id)
                    return PretrainedAgent(agent_id=agent_id, model_path=str(model_path), model_data=model_data)
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
            
            # Model doesn't exist - use fallback
            fallback = config.get("fallback", "heuristic")
            logger.info(f"Model '{model_name}' not trained yet. Using '{fallback}' agent instead.")
            logger.info(f"To use this model, train and save to: {model_path}")
            
            # Recursively create the fallback agent
            return create_agent(fallback, agent_id, registry)
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
        logger.warning(f"Unknown model: {model_name}, using HeuristicAgent")
        return HeuristicAgent(agent_id=agent_id)


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
        #select zerg vs zerg for both agents



        # change the agent race here if needed
        
        
        
        
        with sc2_env.SC2Env(
            map_name=map_name,
            players=[
                sc2_env.Agent(sc2_env.Race.zerg, name="PESTS"),
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
                    logger.info("  Winner: Agent 1 (Zerg)")
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
    
    # Scripted agents (recommended)
    print("\n⭐ SCRIPTED AGENTS (recommended - actually play the game!):")
    print(f"  {'protoss':20s} - Scripted Protoss: builds pylons, gateways, zealots")
    print(f"  {'zerg':20s} - Scripted Zerg: builds pools, overlords, zerglings")
    print(f"  {'zergbot':20s} - Advanced ZergBot: lair, hydras, mutas, coordinated attacks")
    print(f"  {'heuristic':20s} - Generic AI that adapts to any race")
    
    # Basic agents
    print("\n📦 BASIC AGENTS:")
    print(f"  {'simple':20s} - Does nothing (no_op) - for testing")
    print(f"  {'random':20s} - Random action selection")
    
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
    print("       python run_agents.py --model1=protoss --model2=zerg")
    print("       python run_agents.py --model1=zergbot --model2=protoss")
    print("       python run_agents.py --model1=heuristic --model2=heuristic")
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
