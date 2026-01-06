# StarCraft 2 AI - Quick Start Guide

## Overview

This system provides a StarCraft 2 AI environment with:
- **Game Bridge** running on host (PySC2 + StarCraft 2)
- **AI Agents** running in Docker containers
- **Monitoring** via Prometheus + Grafana
- **Model Selection** with pre-trained or custom models

## Quick Start (5 Minutes)

### 1. Setup
```bash
# Run setup (creates venv, installs dependencies)
python setup.py
```

### 2. Select Models
```bash
# Interactive model selection
python start.py --select-only

# Or specify directly
python start.py --agent1=zerg_rush --agent2=terran_macro
```

### 3. Start Everything
```bash
# Start all components
python start.py

# Access monitoring
# Grafana: http://localhost:3000 (admin/sc2admin)
# Prometheus: http://localhost:9090
```

## Model Selection

### Available Models

| Alias | Type | Description | Download |
|-------|------|-------------|----------|
| `heuristic` | Rule-based | Basic strategy | Built-in |
| `random` | Random | Random actions | Built-in |
| `zerg_rush` | Pre-trained | Aggressive Zerg | Auto-download |
| `terran_macro` | Pre-trained | Macro Terran | Auto-download |
| `protoss_gateway` | Pre-trained | Gateway Protoss | Auto-download |
| `ppo` | RL | PPO algorithm | Train yourself |
| `dqn` | RL | DQN algorithm | Train yourself |

### Interactive Selection

```bash
# Run with prompts
python start.py

# Select models for 2 agents
python ai_agent/model_registry.py --select 2
```

### Command-Line Selection

```bash
# Use aliases
python start.py --agent1=heuristic --agent2=ppo

# Use custom paths
python start.py --agent1=models/my_bot.pt --agent2=models/other_bot.pt

# Mix types
python start.py --agent1=zerg_rush --agent2=/path/to/custom.pt
```

### Model Management

```bash
# List available models
python ai_agent/model_registry.py --list

# Show model info
python ai_agent/model_registry.py --info zerg_rush

# Validate a model
python ai_agent/model_registry.py --validate models/my_model.pt

# Download pre-trained model
python ai_agent/model_registry.py --download zerg_rush
```

## Architecture

```
┌─────────────────────────────────────────┐
│          HOST MACHINE                   │
├─────────────────────────────────────────┤
│  StarCraft 2 (Game)                    │
│       ↓                                │
│  game_bridge.py (PySC2)                │
│  - Game loop                           │
│  - Agent coordination                   │
│  - HTTP Server (8000)                   │
└─────────────┬─────────────────────────┘
              │
        HTTP/gRPC
              │
┌─────────────┴─────────────────────────┐
│         DOCKER NETWORK                 │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌──────┐│
│  │ Agent 1 │  │ Agent 2 │  │ Mon. ││
│  │ (8080)  │  │ (8081)  │  │      ││
│  └─────────┘  └─────────┘  └──────┘│
└─────────────────────────────────────────┘
```

## Getting Pre-Trained Models

### Official Sources

1. **DeepMind PySC2**
   - https://github.com/deepmind/pysc2
   - Research reference only

2. **Competition Winners**
   - CIG: https://www.cigconf.org/starcraft-ai-competitions
   - AIIDE: https://aiide.org/competitions

3. **Community Bots**
   - https://github.com/topics/starcraft-ai
   - https://github.com/Dentosal/python-sc2
   - https://github.com/BurnySc2

4. **Hugging Face**
   - https://huggingface.co/models?filter=starcraft

### Download Process

```bash
# Option 1: Auto-download via model registry
python ai_agent/model_registry.py --download zerg_rush

# Option 2: Manual download
cd ai_agent/models
wget https://example.com/model.pt

# Option 3: Download via web browser
# Visit model source → Download → Place in ai_agent/models/
```

## Training Your Own Models

### Basic Training

```bash
# Train a simple model
python ai_agent/train.py --map=Simple64 --episodes=1000

# Train with specific agent
python ai_agent/train.py --agent=HeuristicAI --map=AbyssalReafLE
```

### RL Training (Stable Baselines3)

```bash
# PPO training
python ai_agent/train_sb3.py --algo=ppo --map=Simple64 --timesteps=1000000

# DQN training
python ai_agent/train_sb3.py --algo=dqn --map=AbyssalReefLE --timesteps=500000

# A2C training
python ai_agent/train_sb3.py --algo=a2c --map=Simple64 --timesteps=1000000
```

## Common Commands

### System Management

```bash
# Setup environment
python setup.py

# Check prerequisites
python start.py --check

# Start all components
python start.py

# Stop all components
python start.py --stop

# Check status
python start.py --status

# Verify installation
python start.py --verify-only
```

### Docker Management

```bash
# Build containers
docker-compose build

# Start containers
docker-compose up -d

# Stop containers
docker-compose down

# View logs
docker-compose logs -f

# Restart containers
docker-compose restart
```

### Model Management

```bash
# List models
python ai_agent/model_registry.py --list

# Download model
python ai_agent/model_registry.py --download <model_name>

# Validate model
python ai_agent/model_registry.py --validate <model_path>

# Select models interactively
python ai_agent/model_registry.py --select 2
```

## Troubleshooting

### Prerequisites Check
```bash
python start.py --check
```

### StarCraft 2 Issues
- Ensure SC2 is installed
- Complete first mission
- Check SC2 path in PySC2

### Docker Issues
```bash
# Restart Docker Desktop
# Check Docker is running
docker ps
```

### Model Loading Issues
```bash
# Validate model
python ai_agent/model_registry.py --validate models/model.pt

# Check model format
file models/model.pt
```

## URLs After Starting

| Service | URL | Credentials |
|---------|-----|-------------|
| Grafana | http://localhost:3000 | admin/sc2admin |
| Prometheus | http://localhost:9090 | - |
| Game Bridge | http://localhost:8000 | - |
| Agent 1 | http://localhost:8080 | - |
| Agent 2 | http://localhost:8081 | - |
| Redis Commander | http://localhost:8082 | - |

## File Structure

```
starcraft_ai_2026/
├── host_env/                  # Host Python venv (PySC2)
├── ai_agent/
│   ├── main.py               # AI agent logic
│   ├── model_registry.py     # Model selection system
│   └── models/
│       ├── README.md         # Model documentation
│       └── *.pt             # Pre-trained models
├── monitoring/
│   ├── game_bridge.py        # Game bridge (runs on host)
│   ├── monitor.py          # Monitoring utilities
│   ├── grafana/            # Grafana dashboards
│   └── prometheus/         # Prometheus config
├── start.py               # Startup orchestrator
├── setup.py               # Setup script
├── docker-compose.yml     # Docker configuration
├── requirements.txt       # Docker dependencies
├── host_requirements.txt  # Host dependencies
└── docs/
    ├── QUICKSTART.md      # This file
    ├── ARCHITECTURE.md    # Architecture details
    └── PRETRAINED.md     # Pre-trained models guide
```

## Next Steps

1. ✅ Run `python setup.py`
2. ✅ Install StarCraft 2 (if not installed)
3. ✅ Run `python start.py` for model selection
4. ✅ Monitor at http://localhost:3000
5. 📊 Analyze metrics in Grafana
6. 🧠 Train your own models
7. 🏆 Compete in AI competitions

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Check troubleshooting section
- **Community**: https://github.com/topics/starcraft-ai

## License

See project repository for license information.
