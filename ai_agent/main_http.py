"""
StarCraft 2 AI Agent - Docker Container Version
Connects to game_bridge.py on host via HTTP to receive game state and send actions.
NO PySC2 dependencies - pure decision-making logic only.
"""

import os
import sys
import time
import asyncio
import argparse
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

# Web framework
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import httpx

# Add parent directory to path for monitoring imports
sys.path.insert(0, "/app")

# Local imports - these don't depend on pysc2
try:
    from monitoring.monitor import Monitoring, get_monitoring
except ImportError:
    Monitoring = None
    get_monitoring = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [Agent] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Get agent ID from environment
AGENT_ID = os.getenv("AGENT_ID", "1")
GAME_SERVER = os.getenv("GAME_SERVER", "host.docker.internal")
GAME_PORT = int(os.getenv("GAME_PORT", "8000"))


# ============================================================================
# Data Models (mirroring what game_bridge.py will send)
# ============================================================================

class GameState(BaseModel):
    """Game state received from the host bridge"""
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
    
    # Simplified unit lists
    units: List[Dict[str, Any]] = field(default_factory=list)
    enemy_units: List[Dict[str, Any]] = field(default_factory=list)
    
    # Map info
    map_name: str = ""
    player_id: int = 1
    
    # Episode info
    episode_ended: bool = False
    reward: float = 0.0
    
    class Config:
        # Allow extra fields for forward compatibility
        extra = "allow"


class ActionRequest(BaseModel):
    """Action request to send back to the host"""
    action_type: str = "no_op"
    target_unit_tag: Optional[int] = None
    target_point: Optional[List[float]] = None
    ability_id: Optional[int] = None
    queue_command: bool = False


class ActionResponse(BaseModel):
    """Response from the agent with the chosen action"""
    agent_id: str
    action: ActionRequest
    decision_time_ms: float
    step: int


# ============================================================================
# AI Decision Logic (No PySC2 dependencies)
# ============================================================================

class AIStrategy(Enum):
    """High-level strategy states"""
    OPENING = "opening"
    ECONOMY = "economy"
    ARMY_BUILD = "army_build"
    ATTACK = "attack"
    DEFEND = "defend"


class AIDecisionEngine:
    """
    Pure AI decision-making engine.
    Receives serialized game state, returns action requests.
    No PySC2 dependencies.
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.strategy = AIStrategy.OPENING
        self.action_count = 0
        self.game_step = 0
        self.last_minerals = 0
        self.last_vespene = 0
        
        # Strategy thresholds
        self.economy_workers_target = 22
        self.army_supply_threshold = 30
        self.attack_army_threshold = 50
        
        logger.info(f"AI Decision Engine initialized for agent {agent_id}")
    
    def reset(self):
        """Reset for a new game"""
        self.strategy = AIStrategy.OPENING
        self.action_count = 0
        self.game_step = 0
        self.last_minerals = 0
        self.last_vespene = 0
        logger.info(f"Agent {self.agent_id} reset for new game")
    
    def decide(self, game_state: GameState) -> ActionRequest:
        """
        Make a decision based on the current game state.
        Returns an ActionRequest to send back to the host.
        """
        self.game_step = game_state.step
        self.action_count += 1
        
        # Update strategy based on game state
        self._update_strategy(game_state)
        
        # Make decision based on current strategy
        if self.strategy == AIStrategy.OPENING:
            return self._opening_action(game_state)
        elif self.strategy == AIStrategy.ECONOMY:
            return self._economy_action(game_state)
        elif self.strategy == AIStrategy.ARMY_BUILD:
            return self._army_build_action(game_state)
        elif self.strategy == AIStrategy.ATTACK:
            return self._attack_action(game_state)
        elif self.strategy == AIStrategy.DEFEND:
            return self._defend_action(game_state)
        
        return ActionRequest(action_type="no_op")
    
    def _update_strategy(self, state: GameState):
        """Update strategy based on game state"""
        # Opening phase - first 2 minutes
        if state.game_loop < 2688:  # ~2 minutes at normal speed
            self.strategy = AIStrategy.OPENING
            return
        
        # Check if we need to defend
        if len(state.enemy_units) > 0:
            # Simple check: if enemies are close, defend
            self.strategy = AIStrategy.DEFEND
            return
        
        # Economy phase - need more workers
        if state.food_workers < self.economy_workers_target:
            self.strategy = AIStrategy.ECONOMY
            return
        
        # Army build phase
        if state.food_army < self.army_supply_threshold:
            self.strategy = AIStrategy.ARMY_BUILD
            return
        
        # Attack phase
        if state.food_army >= self.attack_army_threshold:
            self.strategy = AIStrategy.ATTACK
            return
        
        # Default to army building
        self.strategy = AIStrategy.ARMY_BUILD
    
    def _opening_action(self, state: GameState) -> ActionRequest:
        """Actions during opening phase"""
        # Priority: build workers
        if state.minerals >= 50 and state.food_used < state.food_cap:
            return ActionRequest(
                action_type="train_worker",
                ability_id=1006  # Train Probe (Protoss)
            )
        
        # Send idle workers to mine
        if state.idle_worker_count > 0:
            return ActionRequest(action_type="harvest_gather")
        
        return ActionRequest(action_type="no_op")
    
    def _economy_action(self, state: GameState) -> ActionRequest:
        """Actions focused on economy"""
        # Build workers
        if state.minerals >= 50 and state.food_workers < self.economy_workers_target:
            return ActionRequest(
                action_type="train_worker",
                ability_id=1006
            )
        
        # Build supply if needed
        if state.food_cap - state.food_used <= 2 and state.minerals >= 100:
            return ActionRequest(
                action_type="build_supply",
                ability_id=881  # Build Pylon
            )
        
        # Gather resources
        if state.idle_worker_count > 0:
            return ActionRequest(action_type="harvest_gather")
        
        return ActionRequest(action_type="no_op")
    
    def _army_build_action(self, state: GameState) -> ActionRequest:
        """Actions focused on building army"""
        # Build supply if needed
        if state.food_cap - state.food_used <= 4 and state.minerals >= 100:
            return ActionRequest(
                action_type="build_supply",
                ability_id=881
            )
        
        # Build gateway units
        if state.minerals >= 100:
            # Randomly choose between zealot and stalker
            if random.random() > 0.5:
                return ActionRequest(
                    action_type="train_unit",
                    ability_id=916  # Train Zealot
                )
            else:
                return ActionRequest(
                    action_type="train_unit", 
                    ability_id=917  # Train Stalker
                )
        
        return ActionRequest(action_type="no_op")
    
    def _attack_action(self, state: GameState) -> ActionRequest:
        """Actions when attacking"""
        # Attack enemy base location (simplified)
        return ActionRequest(
            action_type="attack_move",
            target_point=[32.0, 32.0]  # Attack towards map center/enemy
        )
    
    def _defend_action(self, state: GameState) -> ActionRequest:
        """Actions when defending"""
        # Pull army back to base
        return ActionRequest(
            action_type="attack_move",
            target_point=[20.0, 20.0]  # Defend near base
        )


# ============================================================================
# FastAPI Application
# ============================================================================

def create_app(agent_id: str) -> FastAPI:
    """Create the FastAPI application for the AI agent"""
    
    app = FastAPI(
        title=f"StarCraft 2 AI Agent {agent_id}",
        description="AI Agent that receives game state and returns actions via HTTP",
        version="2.0.0",
    )
    
    # Initialize components
    decision_engine = AIDecisionEngine(agent_id)
    monitoring = get_monitoring() if get_monitoring else None
    
    # State
    last_game_state: Dict = {}
    pending_action: Optional[ActionResponse] = None
    
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "agent_id": agent_id,
            "status": "running",
            "type": "sc2_ai_agent",
            "version": "2.0.0",
            "strategy": decision_engine.strategy.value,
            "action_count": decision_engine.action_count,
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "agent_id": agent_id,
            "uptime_actions": decision_engine.action_count,
        }
    
    @app.post("/observation")
    async def receive_observation(game_state: GameState):
        """
        Receive game state observation from the host bridge.
        Process it and prepare an action.
        """
        nonlocal last_game_state, pending_action
        
        start_time = time.time()
        
        try:
            # Store the game state
            last_game_state = game_state.model_dump()
            
            # Check for episode end
            if game_state.episode_ended:
                decision_engine.reset()
                if monitoring:
                    monitoring.record_action("episode_end")
                return {"status": "episode_ended", "agent_id": agent_id}
            
            # Update monitoring metrics
            if monitoring:
                monitoring.update_system_metrics()
                monitoring.record_minerals(game_state.minerals - decision_engine.last_minerals)
                monitoring.record_vespene(game_state.vespene - decision_engine.last_vespene)
            
            decision_engine.last_minerals = game_state.minerals
            decision_engine.last_vespene = game_state.vespene
            
            # Make a decision
            action = decision_engine.decide(game_state)
            
            # Calculate decision time
            decision_time = (time.time() - start_time) * 1000  # ms
            
            if monitoring:
                monitoring.record_decision_time(decision_time / 1000)  # Convert to seconds
                monitoring.record_action(action.action_type)
            
            # Prepare response
            pending_action = ActionResponse(
                agent_id=agent_id,
                action=action,
                decision_time_ms=decision_time,
                step=game_state.step,
            )
            
            logger.debug(f"Step {game_state.step}: {action.action_type} (strategy: {decision_engine.strategy.value})")
            
            return {
                "status": "ok",
                "agent_id": agent_id,
                "action": action.model_dump(),
                "decision_time_ms": decision_time,
            }
            
        except Exception as e:
            logger.error(f"Error processing observation: {e}")
            if monitoring:
                monitoring.record_error("observation_error")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/action")
    async def get_action():
        """
        Get the pending action for the current game step.
        Called by the host bridge to retrieve the agent's decision.
        """
        nonlocal pending_action
        
        if pending_action is None:
            return ActionResponse(
                agent_id=agent_id,
                action=ActionRequest(action_type="no_op"),
                decision_time_ms=0,
                step=0,
            ).model_dump()
        
        response = pending_action.model_dump()
        return response
    
    @app.post("/reset")
    async def reset_agent():
        """Reset the agent for a new game"""
        nonlocal last_game_state, pending_action
        
        decision_engine.reset()
        last_game_state = {}
        pending_action = None
        
        logger.info(f"Agent {agent_id} reset")
        return {"status": "reset", "agent_id": agent_id}
    
    @app.get("/state")
    async def get_state():
        """Get current agent state"""
        return {
            "agent_id": agent_id,
            "strategy": decision_engine.strategy.value,
            "action_count": decision_engine.action_count,
            "game_step": decision_engine.game_step,
            "last_game_state": last_game_state,
        }
    
    @app.get("/stats")
    async def get_stats():
        """Get agent statistics"""
        return {
            "agent_id": agent_id,
            "total_actions": decision_engine.action_count,
            "current_strategy": decision_engine.strategy.value,
            "game_step": decision_engine.game_step,
        }
    
    return app


# ============================================================================
# Main Entry Point
# ============================================================================

async def register_with_bridge(agent_id: str, agent_port: int):
    """Register this agent with the game bridge on the host"""
    bridge_url = f"http://{GAME_SERVER}:{GAME_PORT}"
    
    logger.info(f"Attempting to register with game bridge at {bridge_url}")
    
    for attempt in range(10):
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{bridge_url}/agent/register",
                    json={
                        "agent_id": agent_id,
                        "host": "host.docker.internal",  # From host's perspective
                        "port": agent_port,
                    },
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully registered with game bridge")
                    return True
                else:
                    logger.warning(f"Registration failed: {response.text}")
                    
        except httpx.ConnectError:
            logger.warning(f"Cannot connect to bridge (attempt {attempt + 1}/10)")
        except Exception as e:
            logger.warning(f"Registration error: {e}")
        
        await asyncio.sleep(5)
    
    logger.warning("Could not register with game bridge - will wait for bridge to connect to us")
    return False


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="StarCraft 2 AI Agent (Docker)")
    parser.add_argument(
        "--agent-id", 
        type=str, 
        default=AGENT_ID,
        help="Agent ID"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the agent's HTTP server"
    )
    parser.add_argument(
        "--metrics-port",
        type=int,
        default=9090,
        help="Port for Prometheus metrics"
    )
    args = parser.parse_args()
    
    agent_id = args.agent_id
    logger.info(f"Starting SC2 AI Agent {agent_id} (Docker version - no PySC2)")
    logger.info(f"Game bridge: http://{GAME_SERVER}:{GAME_PORT}")
    
    # Get monitoring instance
    monitoring = get_monitoring() if get_monitoring else None
    
    # Start Prometheus metrics server
    if monitoring:
        try:
            monitoring.start_metrics_server(port=args.metrics_port)
            logger.info(f"Prometheus metrics on port {args.metrics_port}")
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")
    
    # Create the FastAPI app
    app = create_app(agent_id)
    
    # Try to register with the game bridge (non-blocking)
    asyncio.create_task(register_with_bridge(agent_id, args.port))
    
    # Start the HTTP server
    logger.info(f"Starting HTTP server on port {args.port}")
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)
    
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
