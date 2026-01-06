"""
Setup Script - Initialize the StarCraft 2 AI environment
Creates virtual environments and installs dependencies
"""

import os
import sys
import subprocess
import argparse
import platform


def run_command(cmd, description, check=True):
    """Run a command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=check, check=check)
        if check and result.returncode != 0:
            print(f"❌ Failed: {description}")
            print(result.stderr.decode() if result.stderr else "")
            return False
        print(f"✅ {description} complete")
        return True
    except Exception as e:
        print(f"❌ Error: {description}")
        print(f"   {e}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")

    major = sys.version_info.major
    minor = sys.version_info.minor

    if major != 3 or minor < 8:
        print(f"❌ Python 3.8+ required, found {major}.{minor}")
        return False

    print(f"✅ Python {major}.{minor} detected")
    return True


def create_host_env():
    """Create host Python virtual environment"""
    print("\n🏠 Setting up host environment...")

    venv_dir = "host_env"

    if os.path.exists(venv_dir):
        print(f"✅ Virtual environment already exists at {venv_dir}")
        return True

    print(f"📁 Creating virtual environment at {venv_dir}...")

    if not run_command(
        [sys.executable, "-m", "venv", venv_dir], "Creating virtual environment"
    ):
        return False

    print(f"✅ Host environment created")
    return True


def get_host_python():
    """Get the Python executable in the host venv"""
    if platform.system() == "Windows":
        return os.path.join("host_env", "Scripts", "python.exe")
    else:
        return os.path.join("host_env", "bin", "python")


def get_host_pip():
    """Get the pip executable in the host venv"""
    if platform.system() == "Windows":
        return os.path.join("host_env", "Scripts", "pip.exe")
    else:
        return os.path.join("host_env", "bin", "pip")


def install_host_requirements():
    """Install host requirements (PySC2, game bridge dependencies)"""
    print("\n📦 Installing host requirements...")

    python_exe = get_host_python()
    pip_exe = get_host_pip()

    if not os.path.exists(python_exe):
        print(f"❌ Python not found at {python_exe}")
        return False

    # Upgrade pip
    if not run_command(
        [python_exe, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip"
    ):
        return False

    # Install from host_requirements.txt if it exists
    if os.path.exists("host_requirements.txt"):
        print("📄 Installing from host_requirements.txt...")
        if not run_command(
            [pip_exe, "install", "-r", "host_requirements.txt"],
            "Installing host dependencies",
        ):
            return False
    else:
        # Install basic requirements manually
        packages = [
            "pysc2>=4.10.0",
            "s2clientprotocol>=4.10.1",
            "fastapi>=0.109.0",
            "uvicorn[standard]>=0.27.0",
            "httpx>=0.26.0",
            "websockets>=12.0",
            "aiohttp>=3.9.0",
            "requests>=2.31.0",
            "pyyaml>=6.0.1",
        ]

        for package in packages:
            if not run_command([pip_exe, "install", package], f"Installing {package}"):
                return False

    print("✅ Host requirements installed")
    return True


def verify_pysc2():
    """Verify PySC2 installation"""
    print("\n🔍 Verifying PySC2 installation...")

    python_exe = get_host_python()

    try:
        result = subprocess.run(
            [python_exe, "-c", "from pysc2 import run_configs; print('PySC2 OK')"],
            capture_output=True,
            timeout=10,
        )

        if result.returncode == 0:
            print("✅ PySC2 is properly installed")
            print(f"   {result.stdout.decode().strip()}")

            # Try to get SC2 path
            try:
                result = subprocess.run(
                    [
                        python_exe,
                        "-c",
                        "from pysc2 import run_configs; print(run_configs.get())",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    print(f"   SC2 Path: {result.stdout.decode().strip()}")
            except:
                pass

            return True
        else:
            print("❌ PySC2 verification failed")
            print(result.stderr.decode())
            return False

    except Exception as e:
        print(f"❌ PySC2 verification error: {e}")
        return False


def check_docker():
    """Check if Docker is available"""
    print("\n🐳 Checking Docker...")

    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Docker: {result.stdout.decode().strip()}")

            # Check if docker-compose is available
            result = subprocess.run(
                ["docker-compose", "--version"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                print(f"✅ Docker Compose: {result.stdout.decode().strip()}")
                return True
            else:
                print("⚠️  Docker Compose not found (optional)")
                return True
        else:
            print("❌ Docker not found")
            return False

    except Exception as e:
        print(f"❌ Docker check error: {e}")
        return False


def check_sc2_installation():
    """Check if StarCraft 2 is installed"""
    print("\n🎮 Checking StarCraft 2 installation...")

    sc2_paths = []

    if platform.system() == "Windows":
        sc2_paths = [
            os.path.join(os.environ.get("ProgramFiles(x86)", ""), "StarCraft II"),
            os.path.join(os.environ.get("ProgramFiles", ""), "StarCraft II"),
        ]
    elif platform.system() == "Darwin":  # macOS
        sc2_paths = ["/Applications/StarCraft II"]
    else:  # Linux
        sc2_paths = [os.path.expanduser("~/StarCraftII")]

    found_paths = [p for p in sc2_paths if os.path.exists(p)]

    if found_paths:
        print(f"✅ StarCraft 2 found at: {found_paths[0]}")
        print("   ⚠️  Make sure you've completed the first mission")
        return True
    else:
        print("⚠️  StarCraft 2 not found in default locations")
        print("   This is OK if SC2 is installed elsewhere")
        print("   Download from: https://starcraft2.com/en-us/legacy")
        return True  # Don't fail, just warn


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")

    dirs = [
        "ai_logs",
        "ai_logs/agent1",
        "ai_logs/agent2",
        "monitoring/prometheus",
        "monitoring/grafana",
        "monitoring/grafana/provisioning",
        "monitoring/grafana/provisioning/datasources",
        "monitoring/grafana/provisioning/dashboards",
        "monitoring/grafana/dashboards",
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

    print(f"✅ Created {len(dirs)} directories")


def setup_docker():
    """Setup Docker containers"""
    print("\n🐳 Setting up Docker containers...")

    if not os.path.exists("docker-compose.yml"):
        print("⚠️  docker-compose.yml not found")
        return True  # Don't fail

    # Build containers (without starting them)
    try:
        result = subprocess.run(
            ["docker-compose", "build"], capture_output=True, timeout=300
        )

        if result.returncode == 0:
            print("✅ Docker containers built successfully")
            return True
        else:
            print("❌ Docker build failed")
            print(result.stderr.decode())
            return False

    except Exception as e:
        print(f"❌ Docker setup error: {e}")
        return False


def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(
        description="Setup StarCraft 2 AI Environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--skip-docker", action="store_true", help="Skip Docker setup")
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing installation"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("🚀 StarCraft 2 AI Environment Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        return 1

    # Verify only mode
    if args.verify_only:
        print("\n🔍 Verifying existing installation...")
        verify_pysc2()
        check_docker()
        check_sc2_installation()
        print("\n✅ Verification complete")
        return 0

    # Create directories
    create_directories()

    # Create host virtual environment
    if not create_host_env():
        return 1

    # Install host requirements
    if not install_host_requirements():
        print("\n❌ Failed to install host requirements")
        return 1

    # Verify PySC2
    if not verify_pysc2():
        print("\n⚠️  PySC2 verification failed, but continuing...")

    # Check Docker
    docker_ok = check_docker()

    # Setup Docker containers
    if docker_ok and not args.skip_docker:
        if not setup_docker():
            print("\n⚠️  Docker setup failed, but you can start containers manually")

    # Check StarCraft 2
    check_sc2_installation()

    print("\n" + "=" * 60)
    print("✅ Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Install StarCraft 2 if not already installed")
    print("2. Complete the first mission to unlock multiplayer")
    print("3. Run: python start.py")
    print("\nOr verify: python start.py --check")
    print("\n🌐 Access URLs after starting:")
    print("  Grafana:     http://localhost:3000 (admin/sc2admin)")
    print("  Prometheus:  http://localhost:9090")
    print("  Game Bridge: http://localhost:8000")

    return 0


if __name__ == "__main__":
    sys.exit(main())
