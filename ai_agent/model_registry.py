"""
Model Registry for StarCraft 2 AI Agents
Handles selection, loading, and validation of pre-trained models
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import Dict, Optional, Any, List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Available model aliases
MODEL_ALIASES = {
    # Built-in heuristic agents
    "heuristic": {
        "type": "heuristic",
        "class": "HeuristicAI",
        "description": "Rule-based AI with basic strategies",
    },
    "random": {
        "type": "random",
        "class": "RandomAI",
        "description": "Random action selection",
    },
    # Pre-trained models (download from sources)
    "zerg_rush": {
        "type": "pretrained",
        "path": "models/zerg_rush_v1.pt",
        "download_url": "https://github.com/example/sc2-models/releases/download/v1/zerg_rush_v1.pt",
        "race": "Zerg",
        "description": "Aggressive Zerg rush strategy",
    },
    "terran_macro": {
        "type": "pretrained",
        "path": "models/terran_macro_v2.pt",
        "download_url": "https://github.com/example/sc2-models/releases/download/v1/terran_macro_v2.pt",
        "race": "Terran",
        "description": "Macro-focused Terran strategy",
    },
    "protoss_gateway": {
        "type": "pretrained",
        "path": "models/protoss_gateway_v1.pt",
        "download_url": "https://github.com/example/sc2-models/releases/download/v1/protoss_gateway_v1.pt",
        "race": "Protoss",
        "description": "Gateway-based Protoss strategy",
    },
    # RL models
    "ppo": {
        "type": "rl",
        "algorithm": "PPO",
        "path": "models/ppo_default.pt",
        "description": "Proximal Policy Optimization agent",
    },
    "dqn": {
        "type": "rl",
        "algorithm": "DQN",
        "path": "models/dqn_default.pt",
        "description": "Deep Q-Network agent",
    },
    "a2c": {
        "type": "rl",
        "algorithm": "A2C",
        "path": "models/a2c_default.pt",
        "description": "Advantage Actor-Critic agent",
    },
}


class ModelRegistry:
    """Registry for AI models with selection and loading capabilities"""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.loaded_models: Dict[str, Any] = {}

    def get_available_models(self) -> List[Dict]:
        """Get list of available models"""
        models = []

        for name, config in MODEL_ALIASES.items():
            model_info = {"name": name, **config}

            # Check if model exists
            if "path" in config:
                model_path = self.models_dir / Path(config["path"]).name
                model_info["available"] = model_path.exists()
                model_info["local_path"] = (
                    str(model_path) if model_info["available"] else None
                )
            else:
                model_info["available"] = True
                model_info["local_path"] = None

            models.append(model_info)

        # Add any custom models in models directory
        for model_file in self.models_dir.glob("*.pt"):
            model_name = model_file.stem
            if model_name not in MODEL_ALIASES:
                models.append(
                    {
                        "name": model_name,
                        "type": "custom",
                        "path": f"models/{model_file.name}",
                        "available": True,
                        "local_path": str(model_file),
                        "description": f"Custom model: {model_file.name}",
                    }
                )

        return models

    def select_model_interactive(self, agent_id: int = 1) -> str:
        """Interactive model selection for an agent"""
        models = self.get_available_models()

        print(f"\n{'=' * 60}")
        print(f"🤖 Select AI Model for Agent {agent_id}:")
        print(f"{'=' * 60}")

        # Display available models
        for i, model in enumerate(models, 1):
            status = "✅" if model["available"] else "⬇️ "
            desc = model.get("description", "No description")
            mtype = model["type"].upper()

            print(f"  [{i}] {status} {model['name']:20s} [{mtype:12s}]")
            print(f"       {desc}")

        print(f"  [{len(models) + 1}] 📁 Custom model path")
        print()

        while True:
            try:
                choice = input(f"Enter choice (1-{len(models) + 1}): ").strip()
                choice = int(choice)

                if 1 <= choice <= len(models):
                    selected = models[choice - 1]
                    model_name = selected["name"]

                    # Download if not available
                    if not selected["available"]:
                        self.download_model(selected["name"])

                    return model_name

                elif choice == len(models) + 1:
                    # Custom model path
                    return self.select_custom_model()

                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(models) + 1}")

            except ValueError:
                print("❌ Please enter a valid number")
            except KeyboardInterrupt:
                print("\n❌ Model selection cancelled")
                sys.exit(1)

    def select_custom_model(self) -> str:
        """Select a custom model file"""
        print("\n📁 Custom Model Selection")
        print("Enter the path to your model file (e.g., /path/to/model.pt)")

        while True:
            model_path = input("Model path: ").strip()

            if not model_path:
                print("❌ Path cannot be empty")
                continue

            if not os.path.exists(model_path):
                print(f"❌ File not found: {model_path}")
                continue

            # Validate model
            try:
                self.validate_model(model_path)
                return model_path
            except Exception as e:
                print(f"❌ Invalid model: {e}")
                continue

    def load_model(self, model_name: str, agent_id: int = 1) -> Any:
        """Load a model by name or path"""
        logger.info(f"Loading model '{model_name}' for agent {agent_id}")

        # Check if it's a file path
        if os.path.exists(model_name):
            return self._load_from_file(model_name)

        # Check if it's an alias
        if model_name in MODEL_ALIASES:
            config = MODEL_ALIASES[model_name]

            if config["type"] == "heuristic":
                return self._load_heuristic(config["class"])
            elif config["type"] == "random":
                return self._load_random()
            elif config["type"] in ["pretrained", "rl", "custom"]:
                model_path = self.models_dir / Path(config["path"]).name

                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    if "download_url" in config:
                        logger.info(f"Attempting to download...")
                        self.download_model(model_name)

                return self._load_from_file(str(model_path), config.get("algorithm"))

        # Try to load as direct file path
        model_path = self.models_dir / model_name
        if model_path.exists():
            return self._load_from_file(str(model_path))

        raise ValueError(f"Model not found: {model_name}")

    def _load_from_file(self, path: str, algorithm: Optional[str] = None) -> Any:
        """Load model from file"""
        logger.info(f"Loading model from {path}")

        # Check cache
        cache_key = f"{path}:{algorithm}"
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model")
            return self.loaded_models[cache_key]

        # Load based on file extension
        if path.endswith((".pt", ".pth")):
            try:
                model_data = torch.load(path, map_location="cpu")

                # Check if it's a full model or just state dict
                if isinstance(model_data, dict) and "model_state_dict" in model_data:
                    # This is a checkpoint
                    model_class = model_data.get("model_class", "GenericModel")
                    logger.info(f"Checkpoint detected: {model_class}")
                    return model_data

                # Assume it's a state dict
                logger.info("PyTorch state dict loaded")
                self.loaded_models[cache_key] = model_data
                return model_data

            except Exception as e:
                logger.error(f"Failed to load PyTorch model: {e}")
                raise

        elif path.endswith((".h5", ".hdf5")):
            try:
                import tensorflow as tf

                model = tf.keras.models.load_model(path)
                logger.info("TensorFlow model loaded")
                return model
            except Exception as e:
                logger.error(f"Failed to load TensorFlow model: {e}")
                raise

        elif path.endswith((".json", ".yaml", ".yml")):
            # Configuration file
            import yaml

            with open(path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {path}")
            return config

        else:
            raise ValueError(f"Unsupported model format: {path}")

    def _load_heuristic(self, class_name: str) -> Any:
        """Load heuristic agent"""
        logger.info(f"Loading heuristic agent: {class_name}")
        return {"type": "heuristic", "class": class_name}

    def _load_random(self) -> Any:
        """Load random agent"""
        logger.info("Loading random agent")
        return {"type": "random"}

    def download_model(self, model_name: str) -> bool:
        """Download a pre-trained model"""
        if model_name not in MODEL_ALIASES:
            print(f"❌ Model '{model_name}' not available for download")
            return False

        config = MODEL_ALIASES[model_name]

        if "download_url" not in config:
            print(f"❌ No download URL for model '{model_name}'")
            return False

        # Prompt for download
        print(f"\n⬇️  Model '{model_name}' not found locally.")
        print(f"    Description: {config['description']}")
        print(f"    URL: {config['download_url']}")

        choice = input("    Download now? [Y/n]: ").strip().lower()

        if choice and choice != "y":
            print("❌ Download cancelled")
            return False

        # Download
        try:
            import urllib.request
            import urllib.error

            model_path = self.models_dir / Path(config["path"]).name

            print(f"📥 Downloading to {model_path}...")
            urllib.request.urlretrieve(config["download_url"], model_path)

            print(f"✅ Model downloaded successfully")
            return True

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False

    def validate_model(self, model_path: str) -> Dict:
        """Validate a model file and return metadata"""
        logger.info(f"Validating model: {model_path}")

        metadata = {"valid": False, "path": model_path, "type": None, "size": None}

        # Check file size
        try:
            file_size = os.path.getsize(model_path)
            metadata["size"] = file_size
            logger.info(f"File size: {file_size / (1024 * 1024):.2f} MB")
        except Exception as e:
            logger.error(f"Cannot access file: {e}")
            return metadata

        # Try to load and validate
        try:
            if model_path.endswith((".pt", ".pth")):
                model_data = torch.load(model_path, map_location="cpu")

                metadata["valid"] = True
                metadata["type"] = "pytorch"

                if isinstance(model_data, dict):
                    metadata["metadata"] = {
                        k: str(v)
                        for k, v in model_data.items()
                        if k not in ["model_state_dict", "optimizer_state_dict"]
                    }
                else:
                    metadata["model_class"] = str(type(model_data))

            elif model_path.endswith((".h5", ".hdf5")):
                import tensorflow as tf

                model = tf.keras.models.load_model(model_path, compile=False)

                metadata["valid"] = True
                metadata["type"] = "tensorflow"
                metadata["model_summary"] = str(model.summary())

            logger.info(f"✅ Model valid: {metadata['type']}")

        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            metadata["error"] = str(e)

        return metadata

    def print_model_info(self, model_name: str):
        """Print information about a model"""
        if model_name in MODEL_ALIASES:
            config = MODEL_ALIASES[model_name]
            print(f"\n📊 Model: {model_name}")
            print(f"{'=' * 50}")
            print(f"Type: {config['type']}")
            print(f"Description: {config['description']}")

            if "race" in config:
                print(f"Race: {config['race']}")

            if "algorithm" in config:
                print(f"Algorithm: {config['algorithm']}")

            # Check availability
            if "path" in config:
                model_path = self.models_dir / Path(config["path"]).name
                if model_path.exists():
                    print(f"✅ Available: {model_path}")
                    self.validate_model(str(model_path))
                else:
                    print(f"❌ Not available locally")
                    if "download_url" in config:
                        print(f"⬇️  Download from: {config['download_url']}")
        else:
            print(f"❌ Unknown model: {model_name}")


def select_agent_models(agent_count: int = 2) -> List[str]:
    """Select models for all agents"""
    registry = ModelRegistry()
    selections = []

    for i in range(1, agent_count + 1):
        model_name = registry.select_model_interactive(i)
        selections.append(model_name)

        print(f"\n✅ Agent {i}: {model_name}")
        registry.print_model_info(model_name)

    return selections


# CLI interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StarCraft 2 AI Model Registry")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--info", type=str, help="Show model info")
    parser.add_argument("--validate", type=str, help="Validate a model file")
    parser.add_argument("--download", type=str, help="Download a pre-trained model")
    parser.add_argument(
        "--select", type=int, default=2, help="Select models for N agents"
    )

    args = parser.parse_args()

    registry = ModelRegistry()

    if args.list:
        print("\n📋 Available Models:\n")
        for model in registry.get_available_models():
            status = "✅" if model["available"] else "⬇️ "
            print(f"{status} {model['name']:20s} - {model['description']}")

    elif args.info:
        registry.print_model_info(args.info)

    elif args.validate:
        metadata = registry.validate_model(args.validate)
        print(f"\n📊 Validation Result: {args.validate}")
        print(f"Valid: {'✅' if metadata['valid'] else '❌'}")
        if "error" in metadata:
            print(f"Error: {metadata['error']}")

    elif args.download:
        success = registry.download_model(args.download)
        if success:
            print(f"✅ Model downloaded successfully")
        else:
            print(f"❌ Download failed")
            sys.exit(1)

    elif args.select:
        selections = select_agent_models(args.select)
        print(f"\n{'=' * 60}")
        print("✅ Selection Complete!")
        print(f"{'=' * 60}")
        for i, model in enumerate(selections, 1):
            print(f"  Agent {i}: {model}")
        print(f"\nTo use these selections, run:")
        print(
            f"  python start.py {' '.join([f'--agent{i}={sel}' for i, sel in enumerate(selections, 1)])}"
        )

    else:
        print("Use --help to see available commands")
