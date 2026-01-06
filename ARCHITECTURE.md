# StarCraft 2 AI - Option 1 Architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     HOST MACHINE                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─────────────┐          ┌──────────────────────────┐ │
│  │ StarCraft 2 │◄─────────┤   game_bridge.py         │ │
│  │   (Game)    │  PySC2   │   - PySC2 on host        │ │
│  └─────────────┘          │   - Game loop            │ │
│                          │   - Agent coordination    │ │
│                          │   - HTTP Server (8000)    │ │
│                          └─────────────┬──────────────┘ │
│                                        │                │
│                                 HTTP/gRPC               │
│                                        │                │
└────────────────────────────────────────┼────────────────┘
                                         │
                        ┌────────────────┴────────────────┐
                        │             Network            │
                        └────────────────┬────────────────┘
                                         │
         ┌───────────────────────────────┼────────────────────────┐
         │                               │                        │
         ▼                               ▼                        ▼
┌──────────────────┐          ┌──────────────────┐   ┌──────────────────┐
│  Docker Agent 1  │          │  Docker Agent 2  │   │  Monitoring      │
│  - AI logic      │          │  - AI logic      │   │  - Prometheus    │
│  - FastAPI (8080)│          │  - FastAPI (8081)│   │  - Grafana       │
│  - Metrics (9091)│          │  - Metrics (9092)│   │                  │
└──────────────────┘          └──────────────────┘   └──────────────────┘
```

## Component Responsibilities

### Host Machine
1. **StarCraft 2 Game**: Runs the actual game
2. **game_bridge.py**: 
   - Initializes PySC2 environment
   - Runs game loop
   - Coordinates with AI agents
   - Exposes HTTP API for agent communication
3. **Python Environment**: PySC2 installed locally

### Docker Containers
1. **AI Agents**: 
   - Pure decision-making logic
   - No PySC2 or SC2 game access
   - FastAPI endpoints for receiving state and returning actions
   - Prometheus metrics for monitoring
2. **Monitoring**:
   - Prometheus collects metrics from agents
   - Grafana visualizes metrics
   - Redis for caching/state (optional)

## Installation

### Host Machine Requirements

```bash
# 1. Install StarCraft 2 (complete first mission)
# Download from: https://starcraft2.com/en-us/legacy

# 2. Create Python virtual environment
python -m venv host_env
source host_env/bin/activate  # Windows: host_env\Scripts\activate

# 3. Install PySC2 (on host only)
pip install pysc2 torch numpy matplotlib

# 4. Install additional host dependencies
pip install fastapi uvicorn httpx
pip install websockets aiohttp

# 5. Verify PySC2 installation
python -c "from pysc2 import run_configs; print(run_configs.get().save())"
```

### Docker Setup (No PySC2)

The Docker containers only need AI logic, not PySC2. Update `requirements.txt`:

```
# NO PySC2 or s2clientprotocol - these are on the host only
torch>=2.0.0
numpy>=1.24.0
scipy>=1.11.0

# Machine Learning (for decision models)
stable-baselines3>=2.2.0
gymnasium>=0.29.0

# Web API (for communication with host)
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
httpx>=0.26.0
websockets>=12.0
aiohttp>=3.9.0

# Metrics and monitoring
prometheus-client>=0.19.0
prometheus-fastapi-instrumentator>=6.1.0

# Data processing
pandas>=2.0.0
matplotlib>=3.7.0

# Utilities
python-json-logger>=2.0.7
pyyaml>=6.0.1
requests>=2.31.0
tqdm>=4.66.0
```

## Running the System

### Step 1: Start Docker Stack (Monitoring + Agents)

```bash
# Build and start all containers
docker-compose up -d

# Check status
docker-compose ps
```

This starts:
- AI Agent 1 (port 8080)
- AI Agent 2 (port 8081)
- Prometheus (port 9090)
- Grafana (port 3000)
- Redis (port 6379)

### Step 2: Start Game Bridge on Host

```bash
# Activate host environment
source host_env/bin/activate

# Run game bridge
python game_bridge.py --port 8000
```

The game bridge will:
1. Initialize StarCraft 2 game
2. Connect to AI agents running in Docker
3. Run the game loop, sending state to agents and receiving actions

### Step 3: Monitor

- **Grafana**: http://localhost:3000 (admin/sc2admin)
- **Prometheus**: http://localhost:9090
- **Agent 1 API**: http://localhost:8080
- **Agent 2 API**: http://localhost:8081

## Communication Flow

1. **Game Loop** (on host):
   - game_bridge.py gets game observation from PySC2
   - Serializes observation to JSON
   - Sends via HTTP POST to agent containers

2. **AI Agent** (in Docker):
   - Receives game state via FastAPI
   - Runs decision logic (ML model, heuristic, etc.)
   - Returns action as JSON

3. **Game Loop** (on host):
   - Receives action from agent
   - Sends action to PySC2
   - Gets next observation
   - Repeat

## Advantages

- ✅ Clean separation: Game/PySC2 on host, AI logic in containers
- ✅ Simpler Docker setup (no SC2/PySC2 in containers)
- ✅ Better performance (no display forwarding)
- ✅ Easier debugging (game visible on host)
- ✅ Scalable (can add more agents without affecting game)
- ✅ Monitoring works seamlessly

## Files Structure

```
starcraft_ai_2026/
├── host_env/                      # Host Python environment (not in git)
├── game_bridge.py                 # Main entry point on host
├── run_game.sh                    # Script to start everything
├── docker-compose.yml             # Docker stack (agents + monitoring)
├── requirements.txt              # Docker container dependencies
├── host_requirements.txt          # Host dependencies (PySC2)
├── ai_agent/
│   ├── main.py                    # AI decision logic (Docker)
│   ├── models/                    # ML models
│   └── config.py                  # Agent configuration
├── monitoring/
│   ├── monitor.py                 # Monitoring utilities
│   ├── grafana/
│   └── prometheus/
└── docs/
    └── ARCHITECTURE.md            # This file
```
