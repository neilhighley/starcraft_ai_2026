# StarCraft 2 AI Models

This directory contains pre-trained AI models for StarCraft 2 agents.

## Model Types

### 1. Heuristic Agents
Rule-based agents that use predetermined strategies.

**Available:**
- `heuristic` - Basic rule-based AI
- `random` - Random action selection

### 2. Pre-Trained Models
Models trained on large datasets of SC2 games.

**Available:**
- `zerg_rush` - Aggressive early-game Zerg strategy
- `terran_macro` - Macro-focused Terran strategy  
- `protoss_gateway` - Gateway-based Protoss strategy

### 3. Reinforcement Learning Models
Models trained using RL algorithms.

**Available:**
- `ppo` - Proximal Policy Optimization
- `dqn` - Deep Q-Network
- `a2c` - Advantage Actor-Critic

### 4. Custom Models
Your own trained models can be placed here.

## Usage

### Interactive Selection
```bash
# Run with interactive prompts
python start.py

# Or use model registry directly
python ai_agent/model_registry.py --select 2
```

### Command-Line Selection
```bash
# Use model aliases
python start.py --agent1=zerg_rush --agent2=terran_macro

# Use model paths
python start.py --agent1=models/my_model.pt --agent2=models/my_other_model.pt

# Mix types
python start.py --agent1=heuristic --agent2=ppo
```

### Model Registry Commands
```bash
# List available models
python ai_agent/model_registry.py --list

# Show model info
python ai_agent/model_registry.py --info zerg_rush

# Validate a model
python ai_agent/model_registry.py --validate models/my_model.pt

# Download a pre-trained model
python ai_agent/model_registry.py --download zerg_rush
```

## Obtaining Pre-Trained Models

### Official Sources

1. **DeepMind PySC2 Models**
   - Repository: https://github.com/deepmind/pysc2
   - Note: Research prototypes only, models not publicly distributed

2. **CIG/SC2AI Competition Winners**
   - CIG: https://www.cigconf.org/starcraft-ai-competitions
   - AIIDE: https://aiide.org/competitions
   - Winners often release their bots

3. **Community Repositories**
   - python-sc2: https://github.com/Dentosal/python-sc2
   - BurnySC2: https://github.com/BurnySc2
   - Search GitHub for "starcraft 2 ai bot"

4. **Hugging Face**
   - Search: https://huggingface.co/models?filter=starcraft
   - Check format compatibility

### Downloading Models

Using model registry:
```bash
python ai_agent/model_registry.py --download zerg_rush
```

Manual download:
```bash
# Create directory
mkdir -p models

# Download model (replace with actual URL)
wget https://example.com/model.zip -O models/model.zip
unzip models/model.zip -d models/
```

## Training Your Own Models

### Simple Training
```bash
# Train a basic model
python ai_agent/train.py --map=Simple64 --episodes=1000

# Save with custom name
python ai_agent/train.py --map=AbyssalReefLE --save-model=models/my_bot_v1
```

### Using Stable Baselines3
```bash
# Train with PPO
python ai_agent/train_sb3.py --algo=ppo --map=Simple64 --timesteps=1000000

# Train with DQN
python ai_agent/train_sb3.py --algo=dqn --map=AbyssalReefLE --timesteps=500000
```

### Custom Training
See `ai_agent/train.py` for training templates.

## Model Formats

Supported formats:
- **PyTorch** (.pt, .pth) - Preferred
- **TensorFlow** (.h5, .hdf5)
- **ONNX** (.onnx) - Universal format
- **JSON/YAML** - For configuration/heuristic agents

## Model Validation

Check if a model is compatible:
```bash
python ai_agent/model_registry.py --validate models/my_model.pt

# Output:
# ✅ Valid PyTorch Model
# Type: PPOAgent
# Size: 45.2 MB
# Trained on: Simple64, AbyssalReefLE
```

## Adding Custom Models

1. **Place model file** in this directory:
   ```bash
   cp /path/to/my_model.pt models/
   ```

2. **Register in model_registry.py** (optional):
   ```python
   CUSTOM_MODELS = {
       "my_bot": {
           "path": "models/my_model.pt",
           "type": "pytorch",
           "class": "MyCustomAgent",
           "description": "My trained bot"
       }
   }
   ```

3. **Use immediately**:
   ```bash
   python start.py --agent1=my_bot
   ```

## Model Performance

| Model Type | Win Rate | Training Time | Resources |
|------------|----------|---------------|-----------|
| Heuristic  | 30-40%   | None          | Low       |
| Pre-trained | 50-70%  | N/A           | Medium    |
| PPO        | 60-75%   | Hours (GPU)   | High      |
| DQN        | 55-70%   | Hours (GPU)   | High      |
| Alpha*     | 90%+     | Weeks         | Very High |

## Troubleshooting

### Model Not Found
```bash
# Check if model exists
ls -la models/

# Download if missing
python ai_agent/model_registry.py --download <model_name>
```

### Model Load Error
```bash
# Validate model
python ai_agent/model_registry.py --validate models/model.pt

# Check dependencies
pip install torch torchvision
```

### Out of Memory
```bash
# Use smaller model or reduce batch size
python start.py --agent1=heuristic  # Uses minimal resources
```

## Model Registry API

```python
from ai_agent.model_registry import ModelRegistry

# Create registry
registry = ModelRegistry()

# List models
models = registry.get_available_models()

# Load model
model = registry.load_model("zerg_rush")

# Interactive selection
model_name = registry.select_model_interactive(agent_id=1)
```

## References

- **Papers with Code**: https://paperswithcode.com/sota/starcraft-ii-on-pysc2
- **PySC2**: https://github.com/deepmind/pysc2
- **Stable Baselines3**: https://github.com/DLR-RM/stable-baselines3
- **SC2AI Community**: https://sc2ai.net/
