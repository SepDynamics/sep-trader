#!/usr/bin/env python3
"""
WebSocket Service for SEP Trading System
Provides real-time data streaming for the web interface
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
import websockets
from websockets.legacy.server import WebSocketServerProtocol
import argparse
from pathlib import Path
import aiohttp

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketUpdate:
    """Market data update message"""
    symbol: str
    price: float
    volume: float
    timestamp: str
    change_24h: float = 0.0
    
@dataclass
class SystemStatus:
    """System status update message"""
    status: str
    active_pairs: int
    total_trades: int
    uptime: str
    memory_usage: float
    timestamp: str
    
@dataclass
class TradingSignal:
    """Trading signal message"""
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float
    price: float
    timestamp: str
    reason: str = ""

@dataclass
class PerformanceUpdate:
    """Performance metrics update"""
    total_pnl: float
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    active_positions: int
    timestamp: str

class WebSocketManager:
    """Manages WebSocket connections and broadcasts"""
    
    def __init__(self):
        self.clients: Set[WebSocketServerProtocol] = set()
        self.subscriptions: Dict[WebSocketServerProtocol, Set[str]] = {}
        self.last_heartbeat = {}
        
    async def register_client(self, websocket: WebSocketServerProtocol, path: str):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        self.subscriptions[websocket] = set()
        self.last_heartbeat[websocket] = time.time()
        logger.info(f"Client connected from {websocket.remote_address}")
        
        try:
            # Send welcome message
            welcome_msg = {
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "available_channels": ["market", "system", "signals", "performance", "trades"]
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(websocket, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {websocket.remote_address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {websocket.remote_address}: {e}")
        finally:
            await self.unregister_client(websocket)
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a WebSocket client"""
        self.clients.discard(websocket)
        self.subscriptions.pop(websocket, None)
        self.last_heartbeat.pop(websocket, None)
    
    async def handle_client_message(self, websocket: WebSocketServerProtocol, message: str):
        """Handle incoming message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')
            
            if msg_type == 'subscribe':
                channels = data.get('channels', [])
                for channel in channels:
                    self.subscriptions[websocket].add(channel)
                
                response = {
                    "type": "subscription",
                    "status": "success",
                    "subscribed_channels": list(self.subscriptions[websocket]),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send(json.dumps(response))
                logger.info(f"Client subscribed to channels: {channels}")
                
            elif msg_type == 'unsubscribe':
                channels = data.get('channels', [])
                for channel in channels:
                    self.subscriptions[websocket].discard(channel)
                
                response = {
                    "type": "subscription",
                    "status": "success", 
                    "subscribed_channels": list(self.subscriptions[websocket]),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await websocket.send(json.dumps(response))
                
            elif msg_type == 'heartbeat':
                self.last_heartbeat[websocket] = time.time()
                await websocket.send(json.dumps({
                    "type": "heartbeat",
                    "timestamp": datetime.utcnow().isoformat()
                }))
                
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received from {websocket.remote_address}")
        except Exception as e:
            logger.error(f"Error processing message from {websocket.remote_address}: {e}")
    
    async def broadcast_to_channel(self, channel: str, message: Dict[str, Any]):
        """Broadcast message to all clients subscribed to a channel"""
        if not self.clients:
            return
            
        message['channel'] = channel
        message['timestamp'] = datetime.utcnow().isoformat()
        json_message = json.dumps(message)
        
        # Get clients subscribed to this channel
        target_clients = [
            client for client, subs in self.subscriptions.items() 
            if channel in subs and client in self.clients
        ]
        
        if target_clients:
            await asyncio.gather(
                *[self.send_safe(client, json_message) for client in target_clients],
                return_exceptions=True
            )
    
    async def send_safe(self, websocket: WebSocketServerProtocol, message: str):
        """Safely send message to client"""
        try:
            await websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.error(f"Error sending to {websocket.remote_address}: {e}")
            await self.unregister_client(websocket)

class BackendAPIConnector:
    """Connects to the trading service API and engine output files"""

    def __init__(
        self,
        websocket_manager: WebSocketManager,
        backend_url: str = 'http://localhost:8000',
        engine_output_dir: Path = project_root / 'output',
    ):
        self.ws_manager = websocket_manager
        self.backend_url = backend_url.rstrip('/')
        self.engine_output_dir = Path(engine_output_dir)
        self.running = False
        self.session = None
        self._processed_signals: Set[str] = set()
        
    async def start(self):
        """Start monitoring backend API"""
        self.running = True
        self.session = aiohttp.ClientSession()
        logger.info(f"Starting Backend API Connector to {self.backend_url}")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self.monitor_market_data()),
            asyncio.create_task(self.monitor_system_status()),
            asyncio.create_task(self.monitor_trading_signals()),
            asyncio.create_task(self.monitor_performance_updates()),
        ]
        
        # Wait for all tasks to complete (or be cancelled)
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Backend API Connector tasks cancelled")
        finally:
            if self.session:
                await self.session.close()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        logger.info("Stopping Backend API Connector...")

    def _read_engine_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Read a JSON file produced by the C++ engine"""
        try:
            path = self.engine_output_dir / filename
            if path.exists():
                with path.open('r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading {filename}: {e}")
        return None
    
    async def _fetch_api_data(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Fetch data from backend API endpoint"""
        try:
            url = f"{self.backend_url}/api/{endpoint}"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API endpoint {endpoint} returned {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {endpoint}: {e}")
            return None
    
    async def monitor_market_data(self):
        """Monitor market data from backend API or engine files"""
        
        while self.running:
            try:
                # Fetch live metrics from backend
                data = await self._fetch_api_data('metrics/live')
                if not data:
                    data = await asyncio.to_thread(
                        self._read_engine_file, 'market_data.json'
                    )

                if data and 'market_data' in data:
                    for market_info in data['market_data']:
                        market_update = MarketUpdate(
                            symbol=market_info.get('symbol', 'EUR/USD'),
                            price=float(market_info.get('price', 0.0)),
                            volume=float(market_info.get('volume', 0.0)),
                            change_24h=float(market_info.get('change_24h', 0.0)),
                            timestamp=datetime.utcnow().isoformat(),
                        )

                        await self.ws_manager.broadcast_to_channel(
                            'market',
                            {'type': 'market_update', 'data': asdict(market_update)},
                        )

                await asyncio.sleep(2)  # Update every 2 seconds
            except Exception as e:
                logger.error(f"Error monitoring market data: {e}")
                await asyncio.sleep(5)
    
    async def monitor_system_status(self):
        """Monitor system status from backend API"""
        while self.running:
            try:
                # Fetch status from backend or engine file
                data = await self._fetch_api_data('status')
                if not data:
                    data = await asyncio.to_thread(
                        self._read_engine_file, 'system_status.json'
                    )

                if data:
                    status_update = SystemStatus(
                        status=data.get('status', 'unknown'),
                        active_pairs=len(data.get('pairs', [])),
                        total_trades=int(data.get('total_trades', 0)),
                        uptime=data.get('uptime', '0s'),
                        memory_usage=float(data.get('memory_usage', 0.0)),
                        timestamp=datetime.utcnow().isoformat(),
                    )

                    await self.ws_manager.broadcast_to_channel(
                        'status',
                        {'type': 'system_status', 'data': asdict(status_update)},
                    )

                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error monitoring system status: {e}")
                await asyncio.sleep(10)
    
    async def monitor_trading_signals(self):
        """Monitor trading signals from engine output files"""

        while self.running:
            try:
                data = await asyncio.to_thread(
                    self._read_engine_file, 'trading_signals.json'
                )
                if isinstance(data, list):
                    signals = data
                elif isinstance(data, dict):
                    signals = data.get('signals', [])
                else:
                    signals = []

                for s in signals:
                    signal_id = s.get('id') or f"{s.get('symbol')}:{s.get('timestamp')}"
                    if signal_id in self._processed_signals:
                        continue
                    self._processed_signals.add(signal_id)
                    signal = TradingSignal(
                        symbol=s.get('symbol', ''),
                        signal_type=s.get('signal_type', 'hold'),
                        confidence=float(s.get('confidence', 0.0)),
                        price=float(s.get('price', 0.0)),
                        timestamp=s.get(
                            'timestamp', datetime.utcnow().isoformat()
                        ),
                        reason=s.get('reason', ''),
                    )
                    await self.ws_manager.broadcast_to_channel(
                        'signals',
                        {'type': 'trading_signal', 'data': asdict(signal)},
                    )

                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error monitoring trading signals: {e}")
                await asyncio.sleep(5)
    
    async def monitor_performance_updates(self):
        """Monitor performance metrics from backend API or engine files"""
        while self.running:
            try:
                # Fetch performance data from backend or engine file
                data = await self._fetch_api_data('performance/current')
                if not data:
                    data = await asyncio.to_thread(
                        self._read_engine_file, 'performance.json'
                    )

                if data:
                    perf_update = PerformanceUpdate(
                        total_pnl=float(data.get('total_pnl', 0.0)),
                        win_rate=float(data.get('win_rate', 0.0)),
                        sharpe_ratio=float(data.get('sharpe_ratio', 0.0)),
                        max_drawdown=float(data.get('max_drawdown', 0.0)),
                        active_positions=int(data.get('active_positions', 0)),
                        timestamp=datetime.utcnow().isoformat(),
                    )

                    await self.ws_manager.broadcast_to_channel(
                        'performance',
                        {'type': 'performance_update', 'data': asdict(perf_update)},
                    )

                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error monitoring performance: {e}")
                await asyncio.sleep(10)

class WebSocketService:
    """Main WebSocket service class"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765, backend_url: str = 'http://localhost:8000'):
        self.host = host
        self.port = port
        self.ws_manager = WebSocketManager()
        self.api_connector = BackendAPIConnector(self.ws_manager, backend_url)
        self.server = None
        
    async def start(self):
        """Start the WebSocket service"""
        logger.info(f"Starting WebSocket service on {self.host}:{self.port}")
        
        # Start the WebSocket server
        self.server = await websockets.serve(
            self.ws_manager.register_client,
            self.host,
            self.port,
            ping_interval=20,
            ping_timeout=10
        )
        
        # Start backend API connector
        asyncio.create_task(self.api_connector.start())
        
        logger.info(f"WebSocket service running on ws://{self.host}:{self.port}")
        
        # Keep running
        await self.server.wait_closed()
    
    def stop(self):
        """Stop the WebSocket service"""
        logger.info("Stopping WebSocket service")
        self.api_connector.stop()
        if self.server:
            self.server.close()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SEP WebSocket Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind to')
    parser.add_argument('--backend-url', default='http://localhost:8000', help='Backend API URL')
    parser.add_argument('--no-simulation', action='store_true', help='Disable data simulation')
    
    args = parser.parse_args()
    
    service = WebSocketService(args.host, args.port, args.backend_url)
    
    # Note: --no-simulation flag preserved for compatibility but not used with API connector
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        service.stop()

if __name__ == '__main__':
    asyncio.run(main())

def start_websocket_server(port: int = 8765, backend_url: str = 'http://localhost:8000'):
    """
    Start WebSocket server function for compatibility with trading service.
    This function starts the WebSocket service on the specified port.
    """
    import threading
    
    def run_server():
        service = WebSocketService('0.0.0.0', port, backend_url)
        asyncio.run(service.start())
    
    # Start the server in a separate thread so it doesn't block
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    return server_thread
