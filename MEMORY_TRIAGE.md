# Triage Notes: StarCraft 2 AI System

**Date:** January 7, 2026  
**Status:** Architecture Updated - Agents run on HOST

---

## 🏗️ Current Architecture

**StarCraft 2 Location:** `T:\act\StarCraft II`

AI agents need direct access to the SC2 executable, so they **run on the host**, not in Docker.

```
HOST MACHINE (Windows)
┌──────────────────────────────────────────────────────────┐
│  StarCraft 2 Game (T:\act\StarCraft II)                  │
│       ↓                                                   │
│  game_bridge.py (PySC2) ←→ ai_agent/main.py (Agent 1)   │
│       ↓                     ai_agent/main.py (Agent 2)   │
│  HTTP API :8000                                          │
└──────────────────────────────────────────────────────────┘
                    │
                    ▼ metrics
┌──────────────────────────────────────────────────────────┐
│  DOCKER (Monitoring Only)                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ Prometheus  │  │  Grafana    │  │   Redis     │      │
│  │   :9090     │  │   :3000     │  │   :6379     │      │
│  └─────────────┘  └─────────────┘  └─────────────┘      │
└──────────────────────────────────────────────────────────┘
```

---

## �️ SC2 Maps Setup

**IMPORTANT:** PySC2 requires map files to be downloaded separately.

### Map Location
```
T:\act\StarCraft II\Maps\
```

### Download Maps
Download from Blizzard:
- Mini Games: https://github.com/Blizzard/s2client-proto#map-packs
- Ladder Maps: https://github.com/Blizzard/s2client-proto#map-packs

### Extract Maps (Password Protected!)
```powershell
# Password for all Blizzard map packs: iagreetotheeula

# Extract mini games (includes Simple64)
& "C:\Program Files\7-Zip\7z.exe" x "Melee.zip" -o"T:\act\StarCraft II\Maps" -y -p"iagreetotheeula"

# Extract ladder maps
& "C:\Program Files\7-Zip\7z.exe" x "Ladder.zip" -o"T:\act\StarCraft II\Maps" -y -p"iagreetotheeula"
```

### Verify Maps
```powershell
Get-ChildItem "T:\act\StarCraft II\Maps\Melee\*.SC2Map" | Select-Object Name, Length
```

---

## �🚀 How to Start

### Option 1: Single Command (Recommended)
```powershell
python start.py
```

This will:
1. Start Docker monitoring (Prometheus, Grafana, Redis)
2. Start AI Agent 1 on port 8080
3. Start AI Agent 2 on port 8081
4. Start Game Bridge (launches SC2)

### Option 2: Manual Start
```powershell
# Terminal 1: Docker
docker-compose up -d prometheus grafana redis redis-commander

# Terminal 2: Agent 1
.\host_env\Scripts\Activate.ps1
python ai_agent/main.py --agent-id 1

# Terminal 3: Agent 2
.\host_env\Scripts\Activate.ps1
python ai_agent/main.py --agent-id 2

# Terminal 4: Game Bridge
.\host_env\Scripts\Activate.ps1
python game_bridge.py --port 8000
```

### Other Commands
```powershell
python start.py --check    # Check prerequisites
python start.py --status   # Check status
python start.py --stop     # Stop everything
python start.py --sc2-path "D:\Games\StarCraft II"  # Custom SC2 path
```

---

## 📋 Docker Services (Monitoring Only)

| Service | Port | Description |
|---------|------|-------------|
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards (admin/sc2admin) |
| Redis | 6379 | Caching/state |
| Redis Commander | 8082 | Redis UI |

**AI agents are NOT in Docker** - they run on host with SC2 access.

---

## ⚠️ Previous Issue (Resolved)

The original setup tried to run AI agents in Docker containers, but:
- Agents need PySC2 which needs SC2 game access
- SC2 is installed at `T:\act\StarCraft II`
- Docker containers can't access this path

**Solution:** Run agents on host, keep only monitoring in Docker.

---

## 🔧 Required Fixes

### Fix 1: Refactor `ai_agent/main.py` to Remove PySC2 Dependencies

The Docker agent should:
1. **NOT** import pysc2 directly
2. Receive game state from the host via HTTP/API (from `game_bridge.py`)
3. Return actions via HTTP/API to the host
4. Only contain AI decision-making logic

**Recommended approach:**
- Create a pure AI decision class that receives serialized game state
- Use FastAPI to expose endpoints for receiving observations and returning actions
- Remove inheritance from `base_agent.BaseAgent`

### Fix 2: game_bridge.py HTTP Server Threading Issue

The `game_bridge.py` starts the HTTP server in a daemon thread, but the main thread blocks on `bridge.start()` which initializes the SC2 environment. If SC2 fails to start, the HTTP server thread never runs properly.

**Lines 324-331 in game_bridge.py:**
```python
bridge.start()  # This blocks until game loop is running

# Start HTTP server in separate thread
http_thread = threading.Thread(
    target=create_simple_http_server, args=(bridge,), daemon=True
)
http_thread.start()
```

**Issue:** If `bridge.start()` fails (e.g., SC2 not running), the HTTP server never starts.

### Fix 3: Port Check Logic is Inverted in start.py

**Lines 24-29 in start.py:**
```python
def check_port(self, port: int) -> bool:
    """Check if a port is in use"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.close()
        return False  # Port is free
    except:
        return True  # Port is in use
```

The logic is correct, but the **status display** at line 192 shows it wrong:
```python
status = "✅ Running" if self.check_port(port) else "❌ Stopped"
```

This shows "Running" if port is in use (correct), "Stopped" if free (also correct). However, this is confusing because a service being "stopped" vs port being "free" have different meanings.

---

## 📋 Architecture Reminder

```
HOST MACHINE                           DOCKER CONTAINERS
┌─────────────────────┐               ┌─────────────────────┐
│ StarCraft 2 Game    │               │ AI Agent 1 (8080)   │
│       ↓             │               │  - Pure AI logic    │
│ game_bridge.py      │◄─── HTTP ───►│  - No PySC2         │
│  - PySC2 installed  │               │  - FastAPI server   │
│  - HTTP Server:8000 │               └─────────────────────┘
│  - Game loop        │               ┌─────────────────────┐
└─────────────────────┘               │ AI Agent 2 (8081)   │
                                      │  - Pure AI logic    │
                                      │  - No PySC2         │
                                      └─────────────────────┘
```

---

## 🚀 Recommended Action Plan

### Step 1: Rewrite `ai_agent/main.py`
Create a new version that:
- Does NOT import pysc2
- Uses FastAPI to receive game state from host
- Returns actions via HTTP API
- Contains pure decision-making logic

### Step 2: Update `game_bridge.py`
- Start HTTP server BEFORE attempting SC2 connection
- Add proper retry logic for SC2 initialization
- Implement proper serialization of PySC2 observations for HTTP transport

### Step 3: Add Agent Communication Protocol
Define clear HTTP endpoints:
- `POST /observation` - Host sends game state to agent
- `GET /action` - Agent returns next action
- `GET /health` - Health check

### Step 4: Update `start.py`
- Add health checks for containers before marking as "started"
- Add retry logic for connection validation
- Improve error messages

---

## 🐳 Current Container Status

| Container | Status | Issue |
|-----------|--------|-------|
| sc2_prometheus | ✅ Running | OK |
| sc2_grafana | ✅ Running | OK |
| sc2_redis | ✅ Running | OK |
| sc2_redis_commander | ✅ Running | OK |
| sc2_ai_1 | ❌ Crash-looping | Missing pysc2 module |
| sc2_ai_2 | ❌ Crash-looping | Missing pysc2 module |

---

## 📝 Files to Modify

1. **ai_agent/main.py** - Complete rewrite (remove pysc2 imports)
2. **game_bridge.py** - Improve HTTP server startup order
3. **monitoring/monitor.py** - Already good (no pysc2 imports)
4. **docker-compose.yml** - Remove deprecated `version` attribute

---

## 🧪 Test Commands

```powershell
# Check container status
docker-compose ps

# View agent logs
docker logs sc2_ai_1 --tail 50
docker logs sc2_ai_2 --tail 50

# Test connectivity (when fixed)
curl http://localhost:8080/health
curl http://localhost:8081/health
curl http://localhost:8000/health

# Restart after fixes
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

---

## 📊 Summary

| Issue | Severity | Status |
|-------|----------|--------|
| AI agents import pysc2 | 🔴 Critical | Blocking |
| HTTP server startup order | 🟡 Medium | Secondary |
| Port status messaging | 🟢 Low | Cosmetic |
| docker-compose version attr | 🟢 Low | Warning only |
