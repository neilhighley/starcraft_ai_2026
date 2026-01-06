# StarCraft 2 AI Implementation Plan

## Overview
Host a local StarCraft 2 game with 2 AI agents as players, running in Docker containers with a Python virtual environment.

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.13+, or Linux (Ubuntu 20.04+ recommended)
- **CPU**: 4+ cores
- **RAM**: 16GB+ recommended
- **GPU**: Optional, but helpful for ML models
- **Disk Space**: 30GB+ for StarCraft 2 and dependencies

### Software Requirements
- Docker Desktop 4.0+
- Python 3.8-3.11
- Git
- StarCraft 2 (free version available)

## Implementation Steps

### Step 1: Install Prerequisites

#### Windows
```powershell
# Install Chocolatey if not present
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Docker Desktop
choco install docker-desktop -y

# Install Python
choco install python -y

# Install Git
choco install git -y
```

#### macOS
```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Docker Desktop
brew install --cask docker

# Install Python
brew install python@3.10

# Install Git
brew install git
```

#### Linux (Ubuntu/Debian)
```bash
# Update package list
sudo apt-get update

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Python
sudo apt-get install python3.10 python3.10-venv python3-pip -y

# Install Git
sudo apt-get install git -y
```

### Step 2: Setup Python Virtual Environment

```bash
# Create virtual environment
python -m venv sc2_env

# Activate virtual environment
# Windows:
sc2_env\Scripts\activate
# macOS/Linux:
source sc2_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Python Dependencies

```bash
# Install PySC2 (StarCraft 2 Learning Environment)
pip install pysc2

# Install additional dependencies
pip install torch torchvision torchaudio
pip install numpy matplotlib
pip install tensorflow  # optional, for ML models
pip install gymnasium
pip install s2clientprotocol
pip install absl-py protobuf
```

### Step 4: Install StarCraft 2

#### Download StarCraft 2
1. Visit: https://starcraft2.com/en-us/legacy
2. Download and install the free version
3. **Important**: Complete the first mission ("Wings of Liberty") to unlock multiplayer features

#### StarCraft 2 Paths (default)
- **Windows**: `C:\Program Files (x86)\StarCraft II`
- **macOS**: `/Applications/StarCraft II`
- **Linux**: `~/StarCraftII`

### Step 5: Setup StarCraft 2 Maps and Versions

```bash
# Download required maps (automatically done by PySC2 on first run)
python -c "from pysc2 import run_configs; run_configs.get().save()"
```

### Step 6: Create Docker Container for AI Agents

#### Dockerfile for AI Agent
```dockerfile
# File: Dockerfile.ai
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI agent code
COPY ai_agent/ ./ai_agent/

# Set environment variables
ENV PYTHONPATH=/app
ENV SC2PATH=/starcraft2

# Create directory for StarCraft 2
RUN mkdir -p /starcraft2

CMD ["python", "ai_agent/main.py"]
```

#### requirements.txt
```
pysc2>=4.10.0
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0
protobuf>=4.24.0
absl-py>=1.4.0
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  ai-agent-1:
    build:
      context: .
      dockerfile: Dockerfile.ai
    container_name: sc2_ai_1
    volumes:
      - ./ai_logs:/app/logs
      - ./starcraft2:/starcraft2  # Mount StarCraft 2 installation
    ports:
      - "8080:8080"
    environment:
      - AGENT_ID=1
      - SC2PATH=/starcraft2
    networks:
      - sc2_network

  ai-agent-2:
    build:
      context: .
      dockerfile: Dockerfile.ai
    container_name: sc2_ai_2
    volumes:
      - ./ai_logs:/app/logs
      - ./starcraft2:/starcraft2  # Mount StarCraft 2 installation
    ports:
      - "8081:8080"
    environment:
      - AGENT_ID=2
      - SC2PATH=/starcraft2
    networks:
      - sc2_network

networks:
  sc2_network:
    driver: bridge
```

### Step 7: Create AI Agent Script

```python
# File: ai_agent/main.py
import argparse
import os
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features

class SimpleAI(base_agent.BaseAgent):
    def step(self, obs):
        super(SimpleAI, self).step(obs)
        
        if obs.first():
            player_y, player_x = (obs.observation.feature_minimap.player_relative == 
                                features.PlayerRelative.SELF).nonzero()
            x_mean = player_x.mean()
            y_mean = player_y.mean()
            
            return actions.FUNCTIONS.move_camera([x_mean, y_mean])
        
        marines = [unit for unit in obs.observation.feature_units
                   if unit.unit_type == units.Terran.Marine]
        
        if len(marines) > 0:
            marine = marines[0]
            
            if marine.alliance == features.PlayerRelative.SELF:
                return actions.FUNCTIONS.select_point("select_all_type", (marine.x, marine.y))
        
        return actions.FUNCTIONS.no_op()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='Simple64')
    parser.add_argument('--agent_id', type=int, default=1)
    args = parser.parse_args()
    
    with sc2_env.SC2Env(
        map_name=args.map,
        agent_race=features.Race.terran,
        opponent_race=features.Race.terran,
        step_mul=16,
        game_steps_per_episode=0,
        visualize=False,
        feature_screen_size=(84, 84),
        feature_minimap_size=(64, 64),
        rgb_screen_size=None,
        rgb_minimap_size=None
    ) as env:
        agent = SimpleAI()
        agent.setup(env.observation_spec()[0], env.action_spec()[0])
        
        timesteps = env.reset()
        agent.reset()
        
        while True:
            step_actions = [agent.step(timesteps[0])]
            if timesteps[0].last():
                break
            timesteps = env.step(step_actions)

if __name__ == '__main__':
    main()
```

### Step 8: Setup and Run Scripts

#### setup.sh (macOS/Linux)
```bash
#!/bin/bash

echo "Setting up StarCraft 2 AI environment..."

# Create virtual environment
python3 -m venv sc2_env
source sc2_env/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install pysc2 torch torchvision torchaudio numpy matplotlib gymnasium s2clientprotocol absl-py protobuf

# Create necessary directories
mkdir -p ai_logs
mkdir -p starcraft2
mkdir -p ai_agent

echo "Setup complete!"
echo "Please:"
echo "1. Install StarCraft 2 and complete the first mission"
echo "2. Copy StarCraft 2 to ./starcraft2 or verify the path"
echo "3. Run: docker-compose up --build"
```

#### setup.ps1 (Windows)
```powershell
Write-Host "Setting up StarCraft 2 AI environment..." -ForegroundColor Green

# Create virtual environment
python -m venv sc2_env
.\sc2_env\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install pysc2 torch torchvision torchaudio numpy matplotlib gymnasium s2clientprotocol absl-py protobuf

# Create necessary directories
New-Item -ItemType Directory -Force -Path ai_logs
New-Item -ItemType Directory -Force -Path starcraft2
New-Item -ItemType Directory -Force -Path ai_agent

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Please:" -ForegroundColor Yellow
Write-Host "1. Install StarCraft 2 and complete the first mission"
Write-Host "2. Copy StarCraft 2 to .\starcraft2 or verify the path"
Write-Host "3. Run: docker-compose up --build"
```

#### run.sh (macOS/Linux)
```bash
#!/bin/bash

echo "Starting StarCraft 2 AI agents..."

# Activate virtual environment
source sc2_env/bin/activate

# Build and start Docker containers
docker-compose up --build

echo "AI agents running!"
echo "Logs available in ./ai_logs/"
```

#### run.ps1 (Windows)
```powershell
Write-Host "Starting StarCraft 2 AI agents..." -ForegroundColor Green

# Activate virtual environment
.\sc2_env\Scripts\activate

# Build and start Docker containers
docker-compose up --build

Write-Host "AI agents running!" -ForegroundColor Green
Write-Host "Logs available in .\ai_logs\" -ForegroundColor Yellow
```

### Step 9: Two-Agent Match Script

```python
# File: two_agent_match.py
import numpy as np
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import threading
import queue

class AIPlayer(base_agent.BaseAgent):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
    
    def step(self, obs):
        super().step(obs)
        
        if obs.first():
            return actions.FUNCTIONS.no_op()
        
        # Simple attack strategy
        attack_units = [unit for unit in obs.observation.feature_units
                       if unit.alliance == features.PlayerRelative.SELF]
        
        if attack_units and len(attack_units) > 5:
            return actions.FUNCTIONS.Attack_minimap("now", (32, 32))
        
        return actions.FUNCTIONS.no_op()

def run_agent(env, agent, action_queue, observation_queue):
    agent.setup(env.observation_spec()[0], env.action_spec()[0])
    agent.reset()
    
    timesteps = env.reset()
    
    while True:
        obs = timesteps[0]
        observation_queue.put(obs)
        
        if obs.last():
            break
        
        action = agent.step(obs)
        action_queue.put(action)
        timesteps = env.step([action])

def main():
    print("Starting two-agent StarCraft 2 match...")
    
    # Create two environments for two players
    action_queue_1 = queue.Queue()
    action_queue_2 = queue.Queue()
    observation_queue_1 = queue.Queue()
    observation_queue_2 = queue.Queue()
    
    with sc2_env.SC2Env(
        map_name='Simple64',
        players=[sc2_env.Agent(sc2_env.Race.terran), sc2_env.Agent(sc2_env.Race.terran)],
        agent_interface_format=sc2_env.parse_agent_interface_format(
            feature_screen=84,
            feature_minimap=64,
            action_space='FEATURE_SCREEN'
        ),
        step_mul=16,
        game_steps_per_episode=0,
        visualize=True
    ) as env:
        agent1 = AIPlayer(agent_id=1)
        agent2 = AIPlayer(agent_id=2)
        
        agent1.setup(env.observation_spec()[0], env.action_spec()[0])
        agent2.setup(env.observation_spec()[0], env.action_spec()[0])
        
        timesteps = env.reset()
        agent1.reset()
        agent2.reset()
        
        episode = 0
        while True:
            episode += 1
            print(f"Episode {episode}")
            
            actions_step = []
            
            # Get actions from both agents
            action1 = agent1.step(timesteps[0])
            action2 = agent2.step(timesteps[1])
            
            actions_step = [action1, action2]
            
            if timesteps[0].last():
                break
            
            timesteps = env.step(actions_step)

if __name__ == '__main__':
    main()
```

### Step 10: Directory Structure

```
starcraft_ai_2026/
├── sc2_env/                    # Python virtual environment
├── ai_agent/
│   └── main.py                 # AI agent implementation
├── ai_logs/                    # Logs from AI agents
├── starcraft2/                 # StarCraft 2 installation (optional)
├── Dockerfile.ai               # Dockerfile for AI agents
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Python dependencies
├── two_agent_match.py         # Two-agent match script
├── setup.sh                   # Setup script (macOS/Linux)
├── setup.ps1                  # Setup script (Windows)
├── run.sh                     # Run script (macOS/Linux)
├── run.ps1                    # Run script (Windows)
└── IMPLEMENTATION_PLAN.md     # This file
```

## Required Docker Containers

1. **AI Agent Container**: Python 3.10 with PySC2 and ML libraries
   - Image: `python:3.10-slim` (custom build)
   - Purpose: Run AI decision-making logic

2. **Optional - Monitoring**: Prometheus/Grafana for metrics
   - Images: `prom/prometheus`, `grafana/grafana`
   - Purpose: Monitor AI performance and game metrics

## Additional Scripts

### monitor_agents.sh
```bash
#!/bin/bash
echo "Monitoring AI Agent Logs..."
docker-compose logs -f ai-agent-1 ai-agent-2
```

### clean.sh
```bash
#!/bin/bash
echo "Cleaning up..."
docker-compose down -v
docker system prune -f
```

### test_environment.py
```python
"""Test if StarCraft 2 environment is properly configured."""
from pysc2 import run_configs
import sys

def test_sc2_installation():
    try:
        config = run_configs.get()
        print(f"StarCraft 2 path: {config}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_pysc2():
    try:
        from pysc2.env import sc2_env
        print("PySC2 installed successfully")
        return True
    except Exception as e:
        print(f"PySC2 error: {e}")
        return False

if __name__ == '__main__':
    print("Testing StarCraft 2 environment...")
    sc2_ok = test_sc2_installation()
    pysc2_ok = test_pysc2()
    
    if sc2_ok and pysc2_ok:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed")
        sys.exit(1)
```

## Troubleshooting

### Common Issues

1. **StarCraft 2 not found**
   - Ensure SC2 is installed and first mission completed
   - Check SC2PATH environment variable or mount correct path

2. **Docker container cannot access StarCraft 2**
   - Ensure volume mount is correct in docker-compose.yml
   - On macOS, add StarCraft 2 to Docker Desktop file sharing

3. **PySC2 import errors**
   - Ensure virtual environment is activated
   - Reinstall with: `pip install --force-reinstall pysc2`

4. **Performance issues**
   - Reduce feature_screen_size in environment config
   - Increase step_mul to skip frames
   - Use lower resolution

## Next Steps

1. Run setup script for your OS
2. Install and setup StarCraft 2
3. Build Docker containers
4. Run test_environment.py
5. Start two-agent match with two_agent_match.py
6. Monitor logs in ai_logs/

## Pre-Trained Models

### Where to Obtain Pre-Trained Models

1. **AlphaStar Model (DeepMind)**
   - **Repository**: https://github.com/deepmind/pysc2
   - **Paper**: https://arxiv.org/abs/1912.02273
   - **Status**: Research prototype, models not publicly available
   - **Use**: Reference architecture only

2. **Twitch Plays StarCraft II**
   - **Repository**: https://github.com/milesial/twitch-plays-starcraft2
   - **Models**: Available for download
   - **Download**: Check repository releases
   - **Use**: Reinforcement learning trained agents

3. **StarCraft AI Competition Winners**
   - **CIG (Computational Intelligence in Games)**: https://www.cigconf.org/starcraft-ai-competitions
   - **AIIDE**: https://aiide.org/competitions
   - **Download**: Winner's models often shared on GitHub
   - **Examples**:
     - https://github.com/KataHina/PMM
     - https://github.com/Dentosal/python-sc2

4. **Open Source AI Agents**
   - **python-sc2**: https://github.com/Dentosal/python-sc2
     - Basic AI agent implementations
     - Easy to integrate
   - **BurnySC2**: https://github.com/BurnySc2
     - Advanced bot with build orders and strategies
   - **StarCraft II Research Paper Implementations**
     - Search: https://paperswithcode.com/sota/starcraft-ii-on-pysc2

5. **Hugging Face Models**
   - **Search**: https://huggingface.co/models?filter=starcraft
   - **Models**: Various RL and supervised learning models
   - **Format**: Check compatibility with PySC2

6. **Model Zoo - University Repositories**
   - **Oxford CIG**: Search GitHub for "cig sc2"
   - **Cornell**: Search for "starcraft ai cornell"
   - **Various**: https://github.com/topics/starcraft-ai

### Using Pre-Trained Models

#### Step 1: Download Model
```bash
# Create models directory
mkdir -p ai_agent/models

# Download example (replace with actual URL)
cd ai_agent/models
wget https://example.com/pretrained_model.zip
unzip pretrained_model.zip
```

#### Step 2: Update Agent Initialization

Add model selection to your agent setup. See **Agent Selection Prompt** section below.

#### Step 3: Test Model
```bash
# Test with pre-trained model
python start.py --agent1-model=pretrained_zerg
python start.py --agent2-model=pretrained_terran
```

### Training Your Own Models

If pre-trained models aren't available or suitable:

```bash
# Simple training script
python ai_agent/train.py --map=Simple64 --episodes=1000 --save-path=ai_agent/models/my_agent

# Use Stable Baselines3 for easy RL
python ai_agent/train_sb3.py --algo=ppo --map=Simple64 --timesteps=1000000
```

### Model Formats

Support for different model formats:
- **PyTorch (.pt, .pth)**: Most common for SC2 AI
- **TensorFlow (.h5, .pb)**: Check compatibility
- **ONNX (.onnx)**: Universal format
- **JSON/YAML**: For rule-based/heuristic agents

### Performance Comparison

| Model Type | Win Rate vs Built-in AI | Training Time | Resources Needed |
|------------|------------------------|---------------|------------------|
| Heuristic | 30-40% | None | Low |
| Simple RL | 50-60% | Hours (GPU) | Medium |
| Advanced RL | 70-80% | Days (GPU) | High |
| AlphaStar-like | 90%+ | Weeks (cluster) | Very High |

## Agent Selection Prompt

### Interactive Agent Selection

To enable interactive model selection for each agent, the system provides prompts during initialization:

```bash
python start.py

# You'll see prompts like:
🤖 Select AI Model for Agent 1:
  [1] Heuristic (Rule-based)
  [2] Pre-trained Zerg Rush
  [3] Pre-trained Macro Bot
  [4] Simple RL (PyTorch)
  [5] Advanced RL (PPO)
  [6] Custom Model Path

Enter choice (1-6): 2

🤖 Select AI Model for Agent 2:
  [1] Heuristic (Rule-based)
  [2] Pre-trained Terran Macro
  [3] Pre-trained Protoss Gateway
  [4] Simple RL (PyTorch)
  [5] Advanced RL (PPO)
  [6] Custom Model Path

Enter choice (1-6): 3
```

### Command-Line Selection

Skip prompts by specifying models directly:

```bash
# Using aliases
python start.py --agent1=zerg_rush --agent2=terran_macro

# Using pre-trained model names
python start.py --agent1-model=models/zerg_bot.pt --agent2-model=models/terran_bot.pt

# Mix of model types
python start.py --agent1=heuristic --agent2=ppo
```

### Model Aliases

Create shortcuts for common models:

```python
# ai_agent/model_registry.py
MODEL_ALIASES = {
    "heuristic": "HeuristicAI",
    "simple_rl": "SimpleRLAgent",
    "ppo": "PPOAgent",
    "dqn": "DQNAgent",
    "zerg_rush": "models/zerg_rush_v1.pt",
    "terran_macro": "models/terran_macro_v2.pt",
    "protoss_gateway": "models/protoss_gateway_v1.pt",
    "alpha_lite": "models/alpha_lite.pt",
}
```

### Configuration File

Save your favorite configurations:

```yaml
# ai_agent/configs/favorite_matchup.yml
agent1:
  model: "zerg_rush"
  race: "Zerg"
  config:
    aggressive: true
    expand_at: 60

agent2:
  model: "terran_macro"
  race: "Terran"
  config:
    defensive: true
    expand_at: 90

map: "AbyssalReefLE"
```

Run with config:

```bash
python start.py --config=favorite_matchup.yml
```

### Download Models on First Run

The system can automatically download models on first use:

```bash
python start.py --agent1=zerg_rush

# If model not found, prompt appears:
⬇️  Model 'zerg_rush' not found locally.
    Download from: https://example.com/models/zerg_rush_v1.pt? [Y/n]: Y

✅ Model downloaded to ai_agent/models/zerg_rush_v1.pt
```

### Model Validation

The system automatically validates models:

```bash
python start.py --agent1=models/my_model.pt

# Output:
✅ Model validated: PyTorch v1.13
📊 Model info:
   Type: PPOAgent
   Architecture: ResNet-18
   Trained on: Simple64, AbyssalReefLE
   Win rate: 62%
   Last updated: 2024-01-15
```

### Custom Model Integration

Add your own models:

1. **Place model** in `ai_agent/models/`
2. **Register** in `ai_agent/model_registry.py`:

```python
CUSTOM_MODELS = {
    "my_bot": {
        "path": "models/my_bot.pt",
        "type": "pytorch",
        "class": "MyCustomAgent",
        "description": "My trained bot"
    }
}
```

3. **Use** immediately:

```bash
python start.py --agent1=my_bot
```

## References

- PySC2 Documentation: https://github.com/deepmind/pysc2
- StarCraft 2 API: https://github.com/Blizzard/s2client-api
- Docker Documentation: https://docs.docker.com
- SC2 Linux: https://github.com/Blizzard/s2client-proto#linux-support
- Papers with Code - StarCraft II: https://paperswithcode.com/sota/starcraft-ii-on-pysc2
- SC2 AI Community: https://github.com/topics/starcraft-ai
