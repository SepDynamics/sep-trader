#!/usr/bin/env python3
"""
WebSocket Service for SEP Trading System
Provides real-time data streaming for the web interface
"""

import asyncio
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, asdict
import websockets
from websockets.legacy.server import WebSocketServerProtocol
import argparse
from pathlib import Path

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

class DataSimulator:
    """Simulates real-time trading data for demonstration"""
    
    def __init__(self, websocket_manager: WebSocketManager):
        self.ws_manager = websocket_manager
        self.symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD']
        self.base_prices = {
            'BTC-USD': 45000,
            'ETH-USD': 3200,
            'ADA-USD': 0.45,
            'DOT-USD': 12.50,
            'LINK-USD': 25.0
        }
        self.running = False
        
    async def start_simulation(self):
        """Start simulating real-time data"""
        self.running = True
        await asyncio.gather(
            self.simulate_market_data(),
            self.simulate_system_status(),
            self.simulate_trading_signals(),
            self.simulate_performance_updates()
        )
    
    def stop_simulation(self):
        """Stop the simulation"""
        self.running = False
    
    async def simulate_market_data(self):
        """Simulate market data updates"""
        while self.running:
            try:
                for symbol in self.symbols:
                    # Simulate price movement (random walk)
                    base_price = self.base_prices[symbol]
                    change_pct = random.uniform(-0.02, 0.02)  # -2% to +2%
                    new_price = base_price * (1 + change_pct)
                    self.base_prices[symbol] = new_price
                    
                    market_update = MarketUpdate(
                        symbol=symbol,
                        price=round(new_price, 4),
                        volume=random.uniform(1000, 10000),
                        timestamp=datetime.utcnow().isoformat(),
                        change_24h=change_pct * 100
                    )
                    
                    await self.ws_manager.broadcast_to_channel('market', {
                        'type': 'market_update',
                        'data': asdict(market_update)
                    })
                
                await asyncio.sleep(2)  # Update every 2 seconds
            except Exception as e:
                logger.error(f"Error in market simulation: {e}")
                await asyncio.sleep(1)
    
    async def simulate_system_status(self):
        """Simulate system status updates"""
        start_time = datetime.utcnow()
        
        while self.running:
            try:
                uptime = datetime.utcnow() - start_time
                status_update = SystemStatus(
                    status="active",
                    active_pairs=len(self.symbols),
                    total_trades=random.randint(100, 500),
                    uptime=str(uptime).split('.')[0],  # Remove microseconds
                    memory_usage=random.uniform(40, 80),
                    timestamp=datetime.utcnow().isoformat()
                )
                
                await self.ws_manager.broadcast_to_channel('system', {
                    'type': 'system_status',
                    'data': asdict(status_update)
                })
                
                await asyncio.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error in system status simulation: {e}")
                await asyncio.sleep(1)
    
    async def simulate_trading_signals(self):
        """Simulate trading signals"""
        while self.running:
            try:
                if random.random() < 0.3:  # 30% chance every interval
                    symbol = random.choice(self.symbols)
                    signal_types = ['buy', 'sell', 'hold']
                    
                    signal = TradingSignal(
                        symbol=symbol,
                        signal_type=random.choice(signal_types),
                        confidence=random.uniform(0.6, 0.95),
                        price=self.base_prices[symbol],
                        timestamp=datetime.utcnow().isoformat(),
                        reason="Pattern detected by quantum analyzer"
                    )
                    
                    await self.ws_manager.broadcast_to_channel('signals', {
                        'type': 'trading_signal',
                        'data': asdict(signal)
                    })
                
                await asyncio.sleep(15)  # Check every 15 seconds
            except Exception as e:
                logger.error(f"Error in signal simulation: {e}")
                await asyncio.sleep(1)
    
    async def simulate_performance_updates(self):
        """Simulate performance metrics updates"""
        while self.running:
            try:
                perf_update = PerformanceUpdate(
                    total_pnl=random.uniform(-1000, 5000),
                    win_rate=random.uniform(0.45, 0.75),
                    sharpe_ratio=random.uniform(0.8, 2.5),
                    max_drawdown=random.uniform(0.05, 0.25),
                    active_positions=random.randint(0, 8),
                    timestamp=datetime.utcnow().isoformat()
                )
                
                await self.ws_manager.broadcast_to_channel('performance', {
                    'type': 'performance_update', 
                    'data': asdict(perf_update)
                })
                
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Error in performance simulation: {e}")
                await asyncio.sleep(1)

class WebSocketService:
    """Main WebSocket service class"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        self.ws_manager = WebSocketManager()
        self.simulator = DataSimulator(self.ws_manager)
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
        
        # Start data simulation
        asyncio.create_task(self.simulator.start_simulation())
        
        logger.info(f"WebSocket service running on ws://{self.host}:{self.port}")
        
        # Keep running
        await self.server.wait_closed()
    
    def stop(self):
        """Stop the WebSocket service"""
        logger.info("Stopping WebSocket service")
        self.simulator.stop_simulation()
        if self.server:
            self.server.close()

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='SEP WebSocket Service')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8765, help='Port to bind to')
    parser.add_argument('--no-simulation', action='store_true', help='Disable data simulation')
    
    args = parser.parse_args()
    
    service = WebSocketService(args.host, args.port)
    
    if args.no_simulation:
        service.simulator.stop_simulation()
    
    try:
        await service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        service.stop()

if __name__ == '__main__':
    asyncio.run(main())
