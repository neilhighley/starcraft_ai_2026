"""
Startup Orchestrator - Start all StarCraft 2 AI components
Run this on the host machine

Architecture:
- Docker: Monitoring only (Prometheus, Grafana, Redis)
- Host: AI Agents + Game Bridge (need access to SC2)
"""

import os
import sys
import subprocess
import time
import argparse
import signal
import logging
from typing import List, Tuple, Dict
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# StarCraft 2 installation path
SC2_PATH = r"T:\act\StarCraft II"


class StartupOrchestrator:
    """Manages startup of all SC2 AI components"""

    def __init__(self, sc2_path: str = SC2_PATH, model1: str = "simple", model2: str = "simple",
                 map_name: str = "Simple64", episodes: int = 1):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.running = False
        self.sc2_path = sc2_path
        self.model1 = model1
        self.model2 = model2
        self.map_name = map_name
        self.episodes = episodes
        self.python_exe = (
            "host_env/Scripts/python.exe" if os.name == "nt" else "host_env/bin/python"
        )

    def check_port(self, port: int) -> bool:
        """Check if a port is in use"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", port))
            sock.close()
            return False  # Port is free
        except:
            return True  # Port is in use

    def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """Check if all prerequisites are met"""
        issues = []

        # Check if StarCraft 2 is installed
        if not os.path.exists(self.sc2_path):
            issues.append(f"StarCraft 2 not found at {self.sc2_path}")
        else:
            sc2_exe = os.path.join(self.sc2_path, "Versions")
            if not os.path.exists(sc2_exe):
                issues.append(f"StarCraft 2 Versions folder not found in {self.sc2_path}")

        # Check if Docker is running
        try:
            result = subprocess.run(["docker", "ps"], capture_output=True, timeout=5)
            if result.returncode != 0:
                issues.append("Docker is not running or not installed")
        except:
            issues.append("Docker is not installed or not accessible")

        # Check if docker-compose is available
        try:
            result = subprocess.run(
                ["docker-compose", "--version"], capture_output=True, timeout=5
            )
            if result.returncode != 0:
                issues.append("docker-compose is not installed")
        except:
            issues.append("docker-compose is not installed")

        # Check if Python virtual environment exists
        if not os.path.exists("host_env"):
            issues.append(
                "Python virtual environment not found. Run: python -m venv host_env"
            )

        # Check if PySC2 is installed
        try:
            result = subprocess.run(
                [self.python_exe, "-c", "import pysc2"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                issues.append("PySC2 is not installed. Run: pip install pysc2")
        except:
            issues.append("Cannot check PySC2 installation")

        # Check required files
        required_files = ["game_bridge.py", "docker-compose.yml", "ai_agent/main.py"]
        for f in required_files:
            if not os.path.exists(f):
                issues.append(f"{f} not found")

        # Warn if ports are already in use
        ports = [8000, 3000, 9090, 8080, 8081]
        occupied = [p for p in ports if self.check_port(p)]
        if occupied:
            issues.append(f"Ports already in use: {occupied}")

        return len(issues) == 0, issues

    def start_docker(self) -> bool:
        """Start Docker containers (monitoring only)"""
        print("🐳 Starting Docker containers (monitoring stack)...")
        try:
            # Stop existing containers
            subprocess.run(["docker-compose", "down"], capture_output=True)
            time.sleep(2)

            # Start only monitoring containers (not AI agents)
            result = subprocess.run(
                ["docker-compose", "up", "-d", "prometheus", "grafana", "redis", "redis-commander"],
                capture_output=True,
                timeout=60
            )

            if result.returncode != 0:
                print(f"❌ Failed to start Docker: {result.stderr.decode()}")
                return False

            # Wait for containers to be ready
            print("⏳ Waiting for containers to start...")
            time.sleep(5)

            # Check container status
            result = subprocess.run(["docker-compose", "ps"], capture_output=True)
            print(result.stdout.decode())

            print("✅ Docker monitoring stack started")
            return True

        except subprocess.TimeoutExpired:
            print("❌ Docker startup timed out")
            return False
        except Exception as e:
            print(f"❌ Docker error: {e}")
            return False

    def start_agent(self, agent_id: int, port: int) -> bool:
        """Start an AI agent on the host - DEPRECATED, use start_game instead"""
        # Both agents now run in a single process via run_agents.py
        logger.warning("start_agent is deprecated - agents run together in start_game")
        return True

    def start_game(self) -> bool:
        """Start the SC2 game with both AI agents"""
        print("🎮 Starting StarCraft 2 with AI agents...")
        print(f"   Map: {self.map_name}")
        print(f"   Agent 1 model: {self.model1}")
        print(f"   Agent 2 model: {self.model2}")

        try:
            # Set SC2PATH environment variable
            env = os.environ.copy()
            env["SC2PATH"] = self.sc2_path

            # Use absl flags format (--flag=value)
            process = subprocess.Popen(
                [
                    self.python_exe, 
                    "run_agents.py",
                    f"--map={self.map_name}",
                    f"--episodes={self.episodes}",
                    f"--sc2_path={self.sc2_path}",
                    f"--model1={self.model1}",
                    f"--model2={self.model2}",
                    "--visualize=True",
                ],
                stdout=None,  # Let output go to console
                stderr=None,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            self.processes["game"] = process

            # Wait a bit and check if it's still running
            time.sleep(5)
            if process.poll() is not None:
                print(f"❌ Game exited immediately with code: {process.returncode}")
                return False

            print(f"✅ SC2 game started (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"❌ Failed to start game: {e}")
            return False

    def start_game_bridge(self) -> bool:
        """Start game bridge on host"""
        print("🎮 Starting game bridge...")

        try:
            # Set SC2PATH environment variable
            env = os.environ.copy()
            env["SC2PATH"] = self.sc2_path

            process = subprocess.Popen(
                [self.python_exe, "game_bridge.py", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

            self.processes["game_bridge"] = process

            # Wait a bit and check if it's still running
            time.sleep(3)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Game bridge exited immediately")
                print(f"STDERR: {stderr[:500] if stderr else 'None'}")
                return False

            print(f"✅ Game bridge started (PID: {process.pid})")
            return True

        except Exception as e:
            print(f"❌ Failed to start game bridge: {e}")
            return False

    def check_status(self):
        """Check status of all components"""
        print("\n" + "=" * 50)
        print("📊 SYSTEM STATUS")
        print("=" * 50)

        # Docker containers
        try:
            result = subprocess.run(["docker-compose", "ps"], capture_output=True)
            print("\n🐳 Docker Containers:")
            print(result.stdout.decode())
        except:
            print("❌ Cannot check Docker status")

        # Ports
        ports = {
            8000: "Game Bridge",
            3000: "Grafana",
            9090: "Prometheus",
            8080: "AI Agent 1",
            8081: "AI Agent 2",
        }
        print("\n🔌 Port Status:")
        for port, name in ports.items():
            status = "✅ Running" if self.check_port(port) else "❌ Stopped"
            print(f"  {port} ({name}): {status}")

        # URLs
        print("\n🌐 Access URLs:")
        print("  Grafana:     http://localhost:3000 (admin/sc2admin)")
        print("  Prometheus:  http://localhost:9090")
        print("  Game Bridge: http://localhost:8000")
        print("  Agent 1:     http://localhost:8080")
        print("  Agent 2:     http://localhost:8081")

    def stop_all(self):
        """Stop all components"""
        print("\n🛑 Stopping all components...")

        # Stop all host processes
        for name, process in self.processes.items():
            try:
                print(f"  Stopping {name}...")
                if os.name == "nt":
                    # Windows: send CTRL_BREAK_EVENT
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    process.terminate()
                process.wait(timeout=5)
                print(f"  ✅ {name} stopped")
            except Exception as e:
                print(f"  ⚠️ Force killing {name}: {e}")
                process.kill()

        self.processes.clear()

        # Stop Docker
        print("🐳 Stopping Docker containers...")
        subprocess.run(["docker-compose", "down"], capture_output=True)
        print("✅ Docker containers stopped")

        self.running = False

    def start_all(self):
        """Start all components"""
        print("🚀 Starting StarCraft 2 AI System")
        print(f"   SC2 Path: {self.sc2_path}\n")

        # Check prerequisites
        print("🔍 Checking prerequisites...")
        ready, issues = self.check_prerequisites()

        if not ready:
            print("❌ Prerequisites not met:")
            for issue in issues:
                print(f"   • {issue}")
            print("\nPlease fix the issues above before continuing.")
            return False

        print("✅ Prerequisites OK\n")

        # Start Docker (monitoring only)
        if not self.start_docker():
            return False

        # Start SC2 game with both agents
        if not self.start_game():
            self.stop_all()
            return False

        self.running = True

        # Show status
        self.check_status()

        print("\n✅ All components started successfully!")
        print("\n" + "=" * 50)
        print("🎮 StarCraft 2 game is running with 2 AI agents")
        print("📊 Grafana: http://localhost:3000 (admin/sc2admin)")
        print("=" * 50)
        print("\nPress Ctrl+C to stop all components\n")

        return True

    def run(self, start: bool = True):
        """Main run loop"""
        if start:
            if not self.start_all():
                return

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\nReceived interrupt signal...")
            self.stop_all()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            self.stop_all()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="StarCraft 2 AI Startup Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start.py                              # Start with default simple agents
  python start.py --model1=random --model2=heuristic  # Use specific agents
  python start.py --list-models                # Show available models
  python start.py --download-models            # Download all pretrained models
  python start.py --interactive                # Interactive model selection
  python start.py --map=Flat64 --episodes=5    # Custom map and episodes
  python start.py --status                     # Check status only
  python start.py --stop                       # Stop all components

Available Models:
  Built-in:    simple, random, heuristic
  Pretrained:  zerg_rush, terran_macro, protoss_gateway
  RL:          ppo, dqn, a2c
  Custom:      /path/to/your/model.pt
        """,
    )

    parser.add_argument(
        "--start",
        action="store_true",
        default=True,
        help="Start all components (default)",
    )
    parser.add_argument("--stop", action="store_true", help="Stop all components")
    parser.add_argument("--status", action="store_true", help="Check status only")
    parser.add_argument("--check", action="store_true", help="Check prerequisites only")
    parser.add_argument(
        "--sc2-path",
        type=str,
        default=SC2_PATH,
        help=f"Path to StarCraft 2 installation (default: {SC2_PATH})",
    )
    
    # Model selection arguments
    parser.add_argument(
        "--model1",
        type=str,
        default="protoss_gateway",
        help="Model for Agent 1 (default: protoss_gateway)",
    )
    parser.add_argument(
        "--model2",
        type=str,
        default="zerg_rush",
        help="Model for Agent 2 (default: zerg_rush)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download all pretrained models",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive model selection",
    )
    
    # Game configuration
    parser.add_argument(
        "--map",
        type=str,
        default="Simple64",
        help="Map name (default: Simple64)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )

    args = parser.parse_args()
    
    # Handle special model commands (delegate to run_agents.py)
    if args.list_models:
        subprocess.run([
            "host_env/Scripts/python.exe" if os.name == "nt" else "host_env/bin/python",
            "run_agents.py",
            "--list_models"
        ])
        return 0
    
    if args.download_models:
        subprocess.run([
            "host_env/Scripts/python.exe" if os.name == "nt" else "host_env/bin/python",
            "run_agents.py",
            "--download_models"
        ])
        return 0

    orchestrator = StartupOrchestrator(
        sc2_path=args.sc2_path,
        model1=args.model1,
        model2=args.model2,
        map_name=args.map,
        episodes=args.episodes,
    )

    if args.check:
        print("🔍 Checking prerequisites...\n")
        ready, issues = orchestrator.check_prerequisites()

        if ready:
            print("✅ All prerequisites met!")
            return 0
        else:
            print("❌ Prerequisites not met:")
            for issue in issues:
                print(f"   • {issue}")
            return 1

    elif args.stop:
        orchestrator.stop_all()
        return 0

    elif args.status:
        orchestrator.check_status()
        return 0

    else:
        # Start all components
        orchestrator.run(start=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
