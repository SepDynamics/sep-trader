#!/usr/bin/env python3
"""
SEP Trading System - Database Connection Module
Handles connections to Valkey database for trading data storage
"""

import os
import json
import logging
import redis
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ValkeyConnection:
    """Professional Valkey database connection for SEP Trading System"""
    
    def __init__(self):
        """Initialize Valkey connection with configuration from environment"""
        # Load configuration from environment or config files
        # Check both VALKEY_* and REDIS_* variable names for compatibility
        self.host = os.environ.get('VALKEY_HOST') or os.environ.get('REDIS_HOST', 'localhost')
        self.port = int(os.environ.get('VALKEY_PORT') or os.environ.get('REDIS_PORT', '6379'))
        self.username = os.environ.get('VALKEY_USER') or os.environ.get('REDIS_USER', '')
        self.password = os.environ.get('VALKEY_PASSWORD') or os.environ.get('REDIS_PASSWORD', '')
        self.database = int(os.environ.get('VALKEY_DATABASE') or os.environ.get('REDIS_DATABASE', '0'))
        
        # Load from config file if available
        self._load_config_from_file()
        
        self.redis_client = None
        self.connected = False
        
        # Initialize connection
        self._initialize_connection()
    
    def _load_config_from_file(self):
        """Load configuration from config files"""
        config_paths = [
            '/app/config/.sep-config.env',
            'config/.sep-config.env',
            '.sep-config.env'
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"\'')
                                
                                if key in ('VALKEY_HOST', 'REDIS_HOST'):
                                    self.host = value
                                elif key in ('VALKEY_PORT', 'REDIS_PORT'):
                                    self.port = int(value)
                                elif key in ('VALKEY_USER', 'REDIS_USER'):
                                    self.username = value
                                elif key in ('VALKEY_PASSWORD', 'REDIS_PASSWORD'):
                                    self.password = value
                                elif key in ('VALKEY_DATABASE', 'REDIS_DATABASE'):
                                    self.database = int(value)
                    
                    logger.info(f"Loaded database config from {config_path}")
                    break
                except Exception as e:
                    logger.warning(f"Error loading config from {config_path}: {e}")
    
    def _initialize_connection(self):
        """Initialize Redis/Valkey connection"""
        try:
            connection_kwargs = {
                'host': self.host,
                'port': self.port,
                'db': self.database,
                'decode_responses': True,
                'socket_timeout': 10,
                'socket_connect_timeout': 10,
                'retry_on_timeout': True,
                'health_check_interval': 30,
                'ssl': True,  # Enable SSL for managed databases
                'ssl_check_hostname': False,  # Disable hostname verification for managed databases
                'ssl_cert_reqs': None  # Disable certificate verification
            }
            
            if self.username:
                connection_kwargs['username'] = self.username
            if self.password:
                connection_kwargs['password'] = self.password
            
            self.redis_client = redis.Redis(**connection_kwargs)
            
            # Test connection
            self.redis_client.ping()
            self.connected = True
            
            logger.info(f"✅ Valkey connection established - {self.host}:{self.port}/{self.database}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Valkey database: {e}")
            self.connected = False
            self.redis_client = None
    
    def is_connected(self) -> bool:
        """Check if database is connected"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            self.connected = False
            return False
    
    def reconnect(self) -> bool:
        """Attempt to reconnect to database"""
        logger.info("Attempting to reconnect to Valkey database...")
        self._initialize_connection()
        return self.connected
    
    @contextmanager
    def get_client(self):
        """Context manager for database operations"""
        if not self.is_connected():
            if not self.reconnect():
                raise ConnectionError("Unable to connect to Valkey database")
        
        try:
            yield self.redis_client
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            raise
    
    def store_trade_data(self, trade_data: Dict[str, Any]) -> bool:
        """Store trading data in database"""
        try:
            with self.get_client() as client:
                timestamp = datetime.now().isoformat()
                trade_key = f"trade:{timestamp}:{trade_data.get('pair', 'unknown')}"
                
                # Store trade data
                client.hset(trade_key, mapping=trade_data)
                
                # Add to trades list for querying
                client.lpush("trades:list", trade_key)
                
                # Keep only last 1000 trades
                client.ltrim("trades:list", 0, 999)
                
                # Store by pair for analysis
                pair = trade_data.get('pair', 'unknown')
                client.lpush(f"trades:pair:{pair}", trade_key)
                client.ltrim(f"trades:pair:{pair}", 0, 499)
                
                logger.info(f"Stored trade data: {trade_key}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing trade data: {e}")
            return False
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trades from database"""
        try:
            with self.get_client() as client:
                trade_keys = client.lrange("trades:list", 0, limit - 1)
                trades = []
                
                for key in trade_keys:
                    trade_data = client.hgetall(key)
                    if trade_data:
                        trades.append(trade_data)
                
                return trades
                
        except Exception as e:
            logger.error(f"Error retrieving recent trades: {e}")
            return []
    
    def get_trades_for_pair(self, pair: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get trades for specific currency pair"""
        try:
            with self.get_client() as client:
                trade_keys = client.lrange(f"trades:pair:{pair}", 0, limit - 1)
                trades = []
                
                for key in trade_keys:
                    trade_data = client.hgetall(key)
                    if trade_data:
                        trades.append(trade_data)
                
                return trades
                
        except Exception as e:
            logger.error(f"Error retrieving trades for {pair}: {e}")
            return []
    
    def store_market_data(self, pair: str, price_data: Dict[str, Any]) -> bool:
        """Store market price data"""
        try:
            with self.get_client() as client:
                timestamp = datetime.now().isoformat()
                market_key = f"market:{pair}:{timestamp}"
                
                # Store current price data
                client.hset(market_key, mapping=price_data)
                
                # Update latest price
                client.hset(f"market:latest:{pair}", mapping=price_data)
                
                # Add to time series for analysis
                client.lpush(f"market:series:{pair}", market_key)
                client.ltrim(f"market:series:{pair}", 0, 999)  # Keep last 1000 points
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing market data for {pair}: {e}")
            return False
    
    def get_latest_price(self, pair: str) -> Optional[Dict[str, Any]]:
        """Get latest price for currency pair"""
        try:
            with self.get_client() as client:
                price_data = client.hgetall(f"market:latest:{pair}")
                return price_data if price_data else None
                
        except Exception as e:
            logger.error(f"Error retrieving latest price for {pair}: {e}")
            return None
    
    def store_system_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Store system performance metrics"""
        try:
            with self.get_client() as client:
                timestamp = datetime.now().isoformat()
                
                # Store current metrics
                client.hset("system:metrics:current", mapping=metrics)
                
                # Add to metrics history
                metrics_key = f"system:metrics:{timestamp}"
                client.hset(metrics_key, mapping=metrics)
                
                # Add to time series
                client.lpush("system:metrics:history", metrics_key)
                client.ltrim("system:metrics:history", 0, 499)  # Keep last 500 points
                
                return True
                
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
            return False
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            with self.get_client() as client:
                metrics = client.hgetall("system:metrics:current")
                return metrics
                
        except Exception as e:
            logger.error(f"Error retrieving system metrics: {e}")
            return {}
    
    def store_candle_data(self, instrument: str, granularity: str, candles: List[Dict]) -> bool:
        """Store OANDA candle data in database"""
        try:
            with self.get_client() as client:
                stored_count = 0
                
                for candle in candles:
                    candle_time = candle.get('time', '')
                    if not candle_time:
                        continue
                    
                    # Create unique key for each candle
                    candle_key = f"candle:{instrument}:{granularity}:{candle_time}"
                    
                    # Prepare candle data for storage
                    candle_data = {
                        'instrument': instrument,
                        'granularity': granularity,
                        'time': candle_time,
                        'open': candle.get('mid', {}).get('o', '0'),
                        'high': candle.get('mid', {}).get('h', '0'),
                        'low': candle.get('mid', {}).get('l', '0'),
                        'close': candle.get('mid', {}).get('c', '0'),
                        'volume': str(candle.get('volume', 0)),
                        'complete': str(candle.get('complete', True)),
                        'stored_at': datetime.now().isoformat()
                    }
                    
                    # Store candle data
                    client.hset(candle_key, mapping=candle_data)
                    
                    # Add to instrument's candle list for querying
                    series_key = f"candles:series:{instrument}:{granularity}"
                    client.zadd(series_key, {candle_key: self._time_to_score(candle_time)})
                    
                    # Maintain reasonable data size (keep last 10000 candles per series)
                    client.zremrangebyrank(series_key, 0, -10001)
                    
                    stored_count += 1
                
                # Update latest candle for quick access
                if candles:
                    latest_candle = candles[-1]  # Assuming candles are in chronological order
                    latest_key = f"candle:latest:{instrument}:{granularity}"
                    latest_data = {
                        'instrument': instrument,
                        'granularity': granularity,
                        'time': latest_candle.get('time', ''),
                        'open': latest_candle.get('mid', {}).get('o', '0'),
                        'high': latest_candle.get('mid', {}).get('h', '0'),
                        'low': latest_candle.get('mid', {}).get('l', '0'),
                        'close': latest_candle.get('mid', {}).get('c', '0'),
                        'volume': str(latest_candle.get('volume', 0)),
                        'updated_at': datetime.now().isoformat()
                    }
                    client.hset(latest_key, mapping=latest_data)
                
                logger.info(f"Stored {stored_count} candles for {instrument} {granularity}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing candle data for {instrument}: {e}")
            return False
    
    def get_candle_data(self, instrument: str, granularity: str, limit: int = 100,
                       from_time: str = None, to_time: str = None) -> List[Dict[str, Any]]:
        """Retrieve stored candle data"""
        try:
            with self.get_client() as client:
                series_key = f"candles:series:{instrument}:{granularity}"
                
                # Determine score range for time filtering
                min_score = self._time_to_score(from_time) if from_time else 0
                max_score = self._time_to_score(to_time) if to_time else '+inf'
                
                # Get candle keys in time order
                candle_keys = client.zrevrangebyscore(series_key, max_score, min_score, start=0, num=limit)
                
                candles = []
                for key in candle_keys:
                    candle_data = client.hgetall(key)
                    if candle_data:
                        candles.append(candle_data)
                
                return candles[::-1]  # Return in chronological order
                
        except Exception as e:
            logger.error(f"Error retrieving candle data for {instrument}: {e}")
            return []
    
    def get_latest_candle(self, instrument: str, granularity: str) -> Optional[Dict[str, Any]]:
        """Get the most recent candle for an instrument"""
        try:
            with self.get_client() as client:
                latest_key = f"candle:latest:{instrument}:{granularity}"
                candle_data = client.hgetall(latest_key)
                return candle_data if candle_data else None
                
        except Exception as e:
            logger.error(f"Error retrieving latest candle for {instrument}: {e}")
            return None
    
    def get_available_instruments(self) -> List[str]:
        """Get list of instruments with stored candle data"""
        try:
            with self.get_client() as client:
                # Find all candle series keys
                pattern = "candles:series:*"
                keys = client.keys(pattern)
                
                instruments = set()
                for key in keys:
                    # Extract instrument from key pattern: candles:series:EUR_USD:M5
                    parts = key.split(':')
                    if len(parts) >= 4:
                        instrument = parts[2]
                        instruments.add(instrument)
                
                return sorted(list(instruments))
                
        except Exception as e:
            logger.error(f"Error retrieving available instruments: {e}")
            return []
    
    def _time_to_score(self, time_str: str) -> float:
        """Convert time string to numeric score for sorted sets"""
        try:
            if not time_str:
                return 0.0
            # Parse RFC3339 timestamp and convert to epoch seconds
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.timestamp()
        except Exception:
            return 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            with self.get_client() as client:
                # Test basic operations
                start_time = datetime.now()
                client.ping()
                ping_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Get database info
                info = client.info()
                
                return {
                    'status': 'healthy',
                    'connected': True,
                    'ping_time_ms': round(ping_time, 2),
                    'host': self.host,
                    'port': self.port,
                    'database': self.database,
                    'version': info.get('redis_version', 'unknown'),
                    'memory_used': info.get('used_memory_human', 'unknown'),
                    'connected_clients': info.get('connected_clients', 0)
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'connected': False,
                'error': str(e),
                'host': self.host,
                'port': self.port
            }

# Global database connection instance
_db_connection = None

def get_database_connection() -> ValkeyConnection:
    """Get global database connection instance"""
    global _db_connection
    if _db_connection is None:
        _db_connection = ValkeyConnection()
    return _db_connection

def test_connection() -> Dict[str, Any]:
    """Test database connection"""
    try:
        db = get_database_connection()
        return db.health_check()
    except Exception as e:
        return {
            'status': 'error',
            'connected': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Test the connection
    result = test_connection()
    print(json.dumps(result, indent=2))