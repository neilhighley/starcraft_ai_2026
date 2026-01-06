"""
Startup Orchestrator - Start all StarCraft 2 AI components
Run this on the host machine
"""

import os
import sys
import subprocess
import time
import argparse
import signal
from typing import List, Tuple
import socket


class StartupOrchestrator:
    """Manages startup of all SC2 AI components"""

    def __init__(self):
        self.processes = {}
        self.running = False

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
                [
                    "host_env/Scripts/python.exe"
                    if os.name == "nt"
                    else "host_env/bin/python",
                    "-c",
                    "import pysc2",
                ],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                issues.append("PySC2 is not installed. Run: pip install pysc2")
        except:
            issues.append("Cannot check PySC2 installation")

        # Check if game_bridge.py exists
        if not os.path.exists("game_bridge.py"):
            issues.append("game_bridge.py not found")

        # Check if docker-compose.yml exists
        if not os.path.exists("docker-compose.yml"):
            issues.append("docker-compose.yml not found")

        # Warn if ports are already in use
        ports = [8000, 3000, 9090, 8080, 8081]
        occupied = [p for p in ports if self.check_port(p)]
        if occupied:
            issues.append(f"Ports already in use: {occupied}")

        return len(issues) == 0, issues

    def start_docker(self) -> bool:
        """Start Docker containers"""
        print("🐳 Starting Docker containers...")
        try:
            # Stop existing containers
            subprocess.run(["docker-compose", "down"], capture_output=True)
            time.sleep(2)

            # Start containers
            result = subprocess.run(
                ["docker-compose", "up", "-d"], capture_output=True, timeout=60
            )

            if result.returncode != 0:
                print(f"❌ Failed to start Docker: {result.stderr.decode()}")
                return False

            # Wait for containers to be ready
            print("⏳ Waiting for containers to start...")
            time.sleep(10)

            # Check container status
            result = subprocess.run(["docker-compose", "ps"], capture_output=True)
            print(result.stdout.decode())

            print("✅ Docker containers started")
            return True

        except subprocess.TimeoutExpired:
            print("❌ Docker startup timed out")
            return False
        except Exception as e:
            print(f"❌ Docker error: {e}")
            return False

    def start_game_bridge(self) -> bool:
        """Start game bridge on host"""
        print("🎮 Starting game bridge on host...")

        python_exe = (
            "host_env/Scripts/python.exe" if os.name == "nt" else "host_env/bin/python"
        )

        try:
            process = subprocess.Popen(
                [python_exe, "game_bridge.py", "--port", "8000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            self.processes["game_bridge"] = process

            # Wait a bit and check if it's still running
            time.sleep(3)
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                print(f"❌ Game bridge exited immediately")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False

            print("✅ Game bridge started")
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

        # Stop game bridge
        if "game_bridge" in self.processes:
            self.processes["game_bridge"].terminate()
            self.processes["game_bridge"].wait(timeout=5)
            print("✅ Game bridge stopped")

        # Stop Docker
        print("🐳 Stopping Docker containers...")
        subprocess.run(["docker-compose", "down"], capture_output=True)
        print("✅ Docker containers stopped")

        self.running = False

    def start_all(self):
        """Start all components"""
        print("🚀 Starting StarCraft 2 AI System\n")

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

        # Start Docker
        if not self.start_docker():
            return False

        # Start game bridge
        if not self.start_game_bridge():
            return False

        self.running = True

        # Show status
        self.check_status()

        print("\n✅ All components started successfully!")
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
  python start.py              # Start all components
  python start.py --status     # Check status only
  python start.py --stop       # Stop all components
  python start.py --check      # Check prerequisites only
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

    args = parser.parse_args()

    orchestrator = StartupOrchestrator()

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
