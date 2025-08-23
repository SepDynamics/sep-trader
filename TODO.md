# SEP Home Dashboard Integration TODO

## ✅ What You Already Have Working
- Backend API on port 5000 with endpoints
- WebSocket service on port 8765 
- Docker containers running (redis, trading-backend, websocket-service, frontend)
- Basic frontend structure with Dashboard.js, useWebSocket hook, and api.js

## 🔧 Integration Steps

### 1. Frontend File Updates

#### A. Replace/Update HomeDashboard Component
```bash
# Copy the new HomeDashboard.jsx to your frontend
cp HomeDashboard.jsx frontend/src/components/HomeDashboard.jsx

# Update imports in App.js to use HomeDashboard
# In frontend/src/App.js, change:
import Dashboard from './components/Dashboard';
# To:
import HomeDashboard from './components/HomeDashboard';
```

#### B. Install Missing Dependencies
```bash
cd frontend
npm install lucide-react
```

#### C. Update App.js Router
```javascript
// In frontend/src/App.js, update the dashboard route:
case 'dashboard':
  return <HomeDashboard />;
```

### 2. Backend API Extensions Needed

#### A. Add Missing Endpoints to `scripts/trading_service.py`

```python
# Add these endpoints to your TradingAPIHandler class:

def get_live_metrics(self):
    """Return live QFH metrics"""
    return {
        'confidence': self.get_metric('confidence', 0),
        'coherence': self.get_metric('coherence', 0),
        'stability': self.get_metric('stability', 0),
        'flip_ratio': self.get_metric('flip_ratio', 0),
        'rupture_ratio': self.get_metric('rupture_ratio', 0),
        'entropy': self.get_metric('entropy', 0)
    }

def get_performance_history(self):
    """Return historical performance data"""
    # Read from your data files or database
    return {
        'data_points': [],  # Fill with actual historical data
        'summary': {}
    }

def get_performance_current(self):
    """Return current performance metrics"""
    return {
        'current_pnl': 0,  # Calculate from positions
        'daily_pnl': 0,    # Calculate from today's trades
        'total_return': 0,  # Calculate from all trades
        'daily_return': 0,  # Today's return percentage
        'sharpe_ratio': 0,  # Calculate Sharpe ratio
        'max_drawdown': 0,  # Calculate max drawdown
        'win_rate': 0,      # Calculate win rate
        'trades_count': 0   # Count total trades
    }

def execute_command(self, command):
    """Execute CLI command safely"""
    # Implement command execution with safety checks
    import subprocess
    allowed_commands = ['status', 'pairs', 'config']  # Whitelist
    
    if command.split()[0] in allowed_commands:
        result = subprocess.run(
            f'./bin/trader-cli {command}',
            shell=True,
            capture_output=True,
            text=True
        )
        return {
            'output': result.stdout,
            'error': result.stderr,
            'return_code': result.returncode
        }
    return {'error': 'Command not allowed'}

def enable_pair(self, pair):
    """Enable a trading pair"""
    if pair not in self.enabled_pairs:
        self.enabled_pairs.add(pair)
    return list(self.enabled_pairs)

def disable_pair(self, pair):
    """Disable a trading pair"""
    self.enabled_pairs.discard(pair)
    return list(self.enabled_pairs)
```

### 3. WebSocket Service Updates

#### A. Update `scripts/websocket_service.py` for Real Data

Replace the DataSimulator with real data connections:

```python
class RealDataProvider:
    """Connect to actual trading engine for real data"""
    
    def __init__(self, websocket_manager):
        self.ws_manager = websocket_manager
        self.redis_client = redis.Redis(host='redis', port=6380)
        
    async def start_streaming(self):
        """Start streaming real data from Redis/Files"""
        await asyncio.gather(
            self.stream_market_data(),
            self.stream_system_status(),
            self.stream_trading_signals(),
            self.stream_performance_updates()
        )
    
    async def stream_market_data(self):
        """Read market data from your data files"""
        while True:
            try:
                # Read from data/OANDA_*.csv or Redis
                market_data = self.read_market_data()
                await self.ws_manager.broadcast_to_channel('market', {
                    'type': 'market_update',
                    'data': market_data
                })
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Market data error: {e}")
                
    async def stream_trading_signals(self):
        """Read signals from output directory"""
        while True:
            try:
                # Read from output/*.json files
                signals = self.read_latest_signals()
                for signal in signals:
                    await self.ws_manager.broadcast_to_channel('signals', {
                        'type': 'trading_signal',
                        'data': signal
                    })
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Signal streaming error: {e}")
```

### 4. Docker Configuration Updates

#### A. Update `docker-compose.yml`

Ensure environment variables are set:

```yaml
frontend:
  environment:
    - REACT_APP_API_URL=http://trading-backend:5000
    - REACT_APP_WS_URL=ws://websocket-service:8765
```

#### B. Update Nginx Configuration

In `frontend/nginx.conf`, add WebSocket proxy support:

```nginx
location /ws {
    proxy_pass http://websocket-service:8765;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

### 5. Data Integration

#### A. Connect to C++ Engine Output

Create a file watcher to monitor the C++ engine output:

```python
# In scripts/file_watcher.py
import watchdog.observers
import watchdog.events

class SignalFileHandler(watchdog.events.FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith('.json'):
            # Parse and broadcast via WebSocket
            with open(event.src_path) as f:
                signal = json.load(f)
                # Send to WebSocket service
```

#### B. Read QFH Metrics from Shared Memory/Files

```python
# In scripts/metrics_reader.py
def read_qfh_metrics():
    """Read metrics from quantum_tracker output"""
    metrics_file = 'output/qfh_metrics.json'
    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            return json.load(f)
    return default_metrics
```

### 6. CLI Bridge Implementation

Create `scripts/cli_bridge.py`:

```python
#!/usr/bin/env python3
import subprocess
import json
import asyncio
from typing import Dict, Any

class CLIBridge:
    """Bridge between web API and CLI commands"""
    
    def __init__(self):
        self.cli_path = './bin/trader-cli'
        
    async def execute_command(self, command: str, args: list = None) -> Dict[str, Any]:
        """Execute CLI command asynchronously"""
        cmd = [self.cli_path, command]
        if args:
            cmd.extend(args)
            
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return {
            'success': process.returncode == 0,
            'output': stdout.decode(),
            'error': stderr.decode(),
            'command': ' '.join(cmd)
        }
```

### 7. Testing Steps

```bash
# 1. Verify all services are running
docker-compose ps

# 2. Test API endpoints
curl http://localhost:5000/api/status
curl http://localhost:5000/api/metrics/live
curl http://localhost:5000/api/performance/current

# 3. Test WebSocket connection
wscat -c ws://localhost:8765
> {"type": "subscribe", "channels": ["market", "signals"]}

# 4. Access the dashboard
open http://localhost:3000

# 5. Check logs for errors
docker-compose logs -f trading-backend
docker-compose logs -f websocket-service
docker-compose logs -f frontend
```

### 8. Production Deployment

```bash
# 1. Build production images
docker-compose -f docker-compose.production.yml build

# 2. Deploy to droplet
./scripts/deploy_to_droplet_complete.sh

# 3. Configure SSL (optional)
certbot --nginx -d your-domain.com

# 4. Monitor production
ssh sep-trader "docker-compose logs -f"
```

## 📋 Priority Order

- [x] **HIGH**: Update `trading_service.py` with missing endpoints
- [x] **HIGH**: Replace DataSimulator with real data in `websocket_service.py`
- [x] **MEDIUM**: Implement CLI bridge for command execution
- [x] **MEDIUM**: Connect to C++ engine output files
- [ ] **LOW**: Add SSL and production optimizations

## ➡️ Next Tasks

- [ ] Implement filesystem watcher to stream new engine signals automatically
- [ ] Expose trading signals through backend API for parity with file feeds
- [ ] Add integration tests for WebSocket streaming and signal handling

## 🚨 Important Notes

- The dashboard expects real data from your backend - no mock data
- Ensure your C++ engine is writing output files that can be read
- WebSocket channels must match between frontend and backend
- All API endpoints must return proper JSON responses
- CORS must be configured correctly for cross-origin requests

## 🔍 Debugging Tips

If the dashboard doesn't show data:
1. Check browser console for errors (F12)
2. Verify WebSocket connection in Network tab
3. Check API responses in Network tab
4. Ensure Docker containers can communicate
5. Check backend logs for errors

## 📊 Expected Data Flow

```
C++ Engine → Output Files → Python Services → API/WebSocket → React Dashboard
     ↓           ↓              ↓                  ↓              ↓
 QFH Metrics  Signals.json  trading_service   Port 5000/8765   Browser
```

## ✅ Success Criteria

- [ ] Dashboard displays real system status
- [ ] WebSocket shows live updates
- [ ] Trading pairs can be enabled/disabled
- [ ] Performance metrics update in real-time
- [ ] CLI commands execute from terminal tab
- [ ] QFH metrics display actual engine values
- [ ] Trading signals appear when generated
- [ ] Start/Stop trading works correctly