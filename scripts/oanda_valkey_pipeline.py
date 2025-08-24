#!/usr/bin/env python3
"""
OANDA → Valkey Data Pipeline
============================

Fetches real-time and historical market data from OANDA API and populates
Valkey/Redis with timestamped keys following the time-intrinsic identity principle.

Key Design:
- Time-intrinsic keys: timestamp IS the identifier
- Backwards computation ready: each key stores complete state for derivation
- Multi-pair support: handles up to 16 currency pairs simultaneously
- Real-time streaming: WebSocket updates for live trading cockpit
"""

import os
import json
import time
import asyncio
import logging
import aioredis
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Market data point with quantum-ready metrics"""
    timestamp: str
    pair: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    
    # Quantum metrics for backwards computation
    entropy: float = 0.0
    stability: float = 0.0
    coherence: float = 0.0
    pin_state: str = "flux"  # flux, stabilizing, converged
    
    def to_valkey_key(self) -> str:
        """Generate time-intrinsic Valkey key"""
        # Key = timestamp (no external naming needed)
        return f"market:{self.pair}:{self.timestamp}"
    
    def to_valkey_value(self) -> Dict[str, Any]:
        """Convert to Valkey-storable dictionary"""
        return asdict(self)
    
    def calculate_quantum_metrics(self, previous_data: Optional['MarketData'] = None):
        """Calculate quantum metrics for backwards computation"""
        if previous_data is None:
            self.entropy = 0.5  # Initial state
            self.stability = 0.0
            self.coherence = 0.0
            return
            
        # Calculate price volatility (entropy)
        price_range = self.high - self.low
        avg_price = (self.high + self.low) / 2
        if avg_price > 0:
            self.entropy = min(1.0, price_range / avg_price)
        
        # Calculate stability (inverse of price change)
        price_change = abs(self.close - previous_data.close)
        if previous_data.close > 0:
            self.stability = max(0.0, 1.0 - (price_change / previous_data.close) * 10)
        
        # Calculate coherence (trend consistency)
        curr_trend = self.close - self.open
        prev_trend = previous_data.close - previous_data.open
        if curr_trend * prev_trend > 0:  # Same direction
            self.coherence = min(1.0, self.stability + 0.2)
        else:
            self.coherence = max(0.0, self.stability - 0.2)
        
        # Determine pin state based on metrics
        if self.entropy > 0.8:
            self.pin_state = "flux"
        elif self.entropy > 0.3:
            self.pin_state = "stabilizing"
        else:
            self.pin_state = "converged"

class OANDAConnector:
    """OANDA API connector with rate limiting and error handling"""
    
    def __init__(self, api_key: str, account_id: str, environment: str = "practice"):
        self.api_key = api_key
        self.account_id = account_id
        self.environment = environment
        
        if environment == "practice":
            self.base_url = "https://api-fxpractice.oanda.com"
        else:
            self.base_url = "https://api-fxtrade.oanda.com"
            
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_candles(self, instrument: str, granularity: str = "M1", 
                         count: int = 500, from_time: str = None) -> List[Dict]:
        """Fetch candlestick data from OANDA"""
        
        params = {
            "granularity": granularity,
            "count": min(count, 5000)  # OANDA limit
        }
        
        if from_time:
            params["from"] = from_time
            
        url = f"{self.base_url}/v3/instruments/{instrument}/candles"
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("candles", [])
                elif response.status == 429:  # Rate limited
                    logger.warning(f"Rate limited for {instrument}, retrying...")
                    await asyncio.sleep(self.rate_limit_delay * 5)
                    return await self.get_candles(instrument, granularity, count, from_time)
                else:
                    logger.error(f"OANDA API error {response.status} for {instrument}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching {instrument}: {str(e)}")
            return []
        
        finally:
            await asyncio.sleep(self.rate_limit_delay)

class ValkeyManager:
    """Valkey/Redis manager for timestamped market data"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self.redis = None
        self.websocket_clients = set()
        
    async def connect(self):
        """Connect to Valkey/Redis"""
        try:
            # Use environment variable URL with SSL support
            import os
            redis_url = os.environ.get('VALKEY_URL') or os.environ.get('REDIS_URL', f"redis://{self.host}:{self.port}/{self.db}")
            
            # Configure SSL if using rediss:// protocol
            connection_kwargs = {'decode_responses': True}
            if redis_url.startswith('rediss://'):
                connection_kwargs['ssl_cert_reqs'] = None
            
            self.redis = await aioredis.from_url(redis_url, **connection_kwargs)
            logger.info(f"Connected to Valkey/Redis at {redis_url.split('@')[-1] if '@' in redis_url else redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Valkey: {str(e)}")
            raise
            
    async def disconnect(self):
        """Disconnect from Valkey/Redis"""
        if self.redis:
            await self.redis.close()
            
    async def store_market_data(self, data: MarketData):
        """Store market data with time-intrinsic key"""
        key = data.to_valkey_key()
        value = json.dumps(data.to_valkey_value())
        
        # Store with 1 hour expiration for real-time data
        await self.redis.setex(key, 3600, value)
        
        # Add to time-sorted set for backwards computation queries
        timestamp_score = int(datetime.fromisoformat(data.timestamp.replace('Z', '+00:00')).timestamp())
        await self.redis.zadd(f"timeline:{data.pair}", {key: timestamp_score})
        
        logger.debug(f"Stored {key} with quantum metrics")
        
        # Broadcast to WebSocket clients for real-time updates
        await self.broadcast_update(data)
        
    async def get_market_data(self, pair: str, timestamp: str) -> Optional[MarketData]:
        """Retrieve market data by time-intrinsic key"""
        key = f"market:{pair}:{timestamp}"
        value = await self.redis.get(key)
        
        if value:
            data_dict = json.loads(value)
            return MarketData(**data_dict)
        return None
        
    async def get_historical_range(self, pair: str, start_time: str, end_time: str) -> List[MarketData]:
        """Get historical data range for backwards computation"""
        start_ts = int(datetime.fromisoformat(start_time.replace('Z', '+00:00')).timestamp())
        end_ts = int(datetime.fromisoformat(end_time.replace('Z', '+00:00')).timestamp())
        
        keys = await self.redis.zrangebyscore(f"timeline:{pair}", start_ts, end_ts)
        
        data_points = []
        for key in keys:
            value = await self.redis.get(key)
            if value:
                data_dict = json.loads(value)
                data_points.append(MarketData(**data_dict))
                
        return sorted(data_points, key=lambda x: x.timestamp)
        
    async def broadcast_update(self, data: MarketData):
        """Broadcast real-time updates to WebSocket clients"""
        if not self.websocket_clients:
            return
            
        update_message = {
            "type": "market_update",
            "data": data.to_valkey_value(),
            "backwards_computation_ready": True,
            "entropy_band": data.pin_state
        }
        
        message = json.dumps(update_message)
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
                
        # Remove disconnected clients
        self.websocket_clients -= disconnected

class OANDAValkeyPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self):
        self.oanda = None
        self.valkey = None
        self.enabled_pairs = []
        self.last_data_cache = {}  # For quantum metrics calculation
        self.running = False
        
    async def initialize(self):
        """Initialize pipeline components"""
        
        # Load OANDA credentials
        api_key = os.getenv("OANDA_API_KEY")
        account_id = os.getenv("OANDA_ACCOUNT_ID") 
        environment = os.getenv("OANDA_ENVIRONMENT", "practice")
        
        if not api_key or not account_id:
            raise ValueError("OANDA_API_KEY and OANDA_ACCOUNT_ID must be set")
            
        # Load enabled pairs from config
        try:
            with open("/sep/config/pair_registry.json", "r") as f:
                config = json.load(f)
                self.enabled_pairs = config.get("enabled_pairs", [])
        except FileNotFoundError:
            logger.warning("Using default currency pairs")
            self.enabled_pairs = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD"]
            
        logger.info(f"Enabled pairs: {self.enabled_pairs}")
        
        # Initialize connectors
        self.oanda = OANDAConnector(api_key, account_id, environment)
        self.valkey = ValkeyManager()
        await self.valkey.connect()
        
        logger.info("Pipeline initialized successfully")
        
    async def fetch_historical_data(self, hours: int = 24):
        """Fetch historical data for all enabled pairs"""
        logger.info(f"Fetching {hours} hours of historical data...")
        
        async with self.oanda:
            tasks = []
            for pair in self.enabled_pairs:
                task = self._fetch_pair_historical(pair, hours)
                tasks.append(task)
                
            await asyncio.gather(*tasks)
            
        logger.info("Historical data fetch completed")
        
    async def _fetch_pair_historical(self, pair: str, hours: int):
        """Fetch historical data for a single pair"""
        from_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat() + "Z"
        
        candles = await self.oanda.get_candles(pair, "M1", count=hours * 60, from_time=from_time)
        
        previous_data = None
        for candle in candles:
            if not candle.get("complete", False):
                continue
                
            mid_prices = candle["mid"]
            
            data = MarketData(
                timestamp=candle["time"],
                pair=pair,
                open=float(mid_prices["o"]),
                high=float(mid_prices["h"]),
                low=float(mid_prices["l"]),
                close=float(mid_prices["c"]),
                volume=candle.get("volume", 0)
            )
            
            # Calculate quantum metrics with previous data
            data.calculate_quantum_metrics(previous_data)
            
            # Store in Valkey
            await self.valkey.store_market_data(data)
            
            previous_data = data
            self.last_data_cache[pair] = data
            
        logger.info(f"Stored {len(candles)} candles for {pair}")
        
    async def start_real_time_streaming(self):
        """Start real-time data streaming"""
        logger.info("Starting real-time data streaming...")
        self.running = True
        
        async with self.oanda:
            while self.running:
                tasks = []
                for pair in self.enabled_pairs:
                    task = self._fetch_latest_candle(pair)
                    tasks.append(task)
                    
                await asyncio.gather(*tasks)
                await asyncio.sleep(60)  # Update every minute
                
    async def _fetch_latest_candle(self, pair: str):
        """Fetch latest candle for a pair"""
        candles = await self.oanda.get_candles(pair, "M1", count=1)
        
        if not candles or not candles[0].get("complete", False):
            return
            
        candle = candles[0]
        mid_prices = candle["mid"]
        
        data = MarketData(
            timestamp=candle["time"],
            pair=pair,
            open=float(mid_prices["o"]),
            high=float(mid_prices["h"]),
            low=float(mid_prices["l"]),
            close=float(mid_prices["c"]),
            volume=candle.get("volume", 0)
        )
        
        # Calculate quantum metrics
        previous_data = self.last_data_cache.get(pair)
        data.calculate_quantum_metrics(previous_data)
        
        # Store in Valkey
        await self.valkey.store_market_data(data)
        
        self.last_data_cache[pair] = data
        
    async def start_websocket_server(self, port: int = 8765):
        """Start WebSocket server for frontend connections"""
        
        async def websocket_handler(websocket, path):
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            self.valkey.websocket_clients.add(websocket)
            
            try:
                await websocket.wait_closed()
            finally:
                self.valkey.websocket_clients.discard(websocket)
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
                
        server = await websockets.serve(websocket_handler, "0.0.0.0", port)
        logger.info(f"WebSocket server started on port {port}")
        return server
        
    async def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping OANDA → Valkey pipeline...")
        self.running = False
        if self.valkey:
            await self.valkey.disconnect()

async def main():
    """Main execution function"""
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv("/sep/OANDA.env")
    
    pipeline = OANDAValkeyPipeline()
    
    try:
        # Initialize pipeline
        await pipeline.initialize()
        
        # Fetch historical data (24 hours)
        await pipeline.fetch_historical_data(24)
        
        # Start WebSocket server for frontend
        websocket_server = await pipeline.start_websocket_server(8765)
        
        # Start real-time streaming
        await pipeline.start_real_time_streaming()
        
    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        raise
    finally:
        await pipeline.stop()

if __name__ == "__main__":
    asyncio.run(main())