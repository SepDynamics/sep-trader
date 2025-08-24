#!/usr/bin/env python3
"""
SEP Professional Trader-Bot - Lightweight Cloud Trading Service
Runs on Digital Ocean droplet for live trading execution
"""

import os
import sys
import time
import json
import logging
import contextvars
from uuid import uuid4
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
import signal

# Allow importing risk controls from scripts directory
sys.path.append(os.path.dirname(__file__))
from trading.risk import RiskManager, RiskLimits  # noqa: E402
from oanda_connector import OandaConnector  # noqa: E402
from websocket_service import start_websocket_server  # noqa: E402
from database_connection import get_database_connection  # noqa: E402
from cli_bridge import CLIBridge  # noqa: E402
import subprocess

# Correlation ID support
correlation_id_var = contextvars.ContextVar("correlation_id", default="-")


class CorrelationIdFilter(logging.Filter):
    """Ensure every log record contains a correlation_id field."""

    def filter(self, record):
        record.correlation_id = correlation_id_var.get()
        return True


# Setup logging with environment-appropriate paths and rotation
def setup_logging():
    # Determine log directory based on environment
    if os.path.exists('/app'):
        # Docker/containerized environment
        log_dir = '/app/logs'
    else:
        # Local development environment
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'trading_service.log')
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(correlation_id)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addFilter(CorrelationIdFilter())

    return logger

logger = setup_logging()

class TradingService:
    def __init__(self):
        self.running = False
        self.enabled_pairs = set()
        self.trading_active = False
        self.last_sync = None
        self.market_status = "unknown"
        self.current_config = {}
        self.build_hash = os.environ.get("BUILD_HASH", "unknown")
        self.config_version = os.environ.get("CONFIG_VERSION", "unknown")

        # Setup database connection
        try:
            self.database = get_database_connection()
            logger.info("Database connection initialized")
        except Exception as e:
            logger.warning(f"Database connection unavailable: {e}")
            self.database = None

        # Setup connectors and risk manager
        try:
            self.oanda = OandaConnector()
        except Exception as e:
            logger.warning(f"OANDA connector unavailable: {e}")
            self.oanda = None

        self.risk_manager = RiskManager(RiskLimits())

        # Setup SEP Engine CLI Bridge
        try:
            self.cli_bridge = CLIBridge()
            logger.info("SEP Engine CLI Bridge initialized")
        except Exception as e:
            logger.warning(f"SEP Engine CLI Bridge unavailable: {e}")
            self.cli_bridge = None

        # Load configuration
        self.load_config()

    def load_config(self):
        """Load configuration and enabled pairs"""
        try:
            config_path = "/app/config/pair_registry.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.enabled_pairs = set(config.get('enabled_pairs', []))
                    self.current_config = config
                    logger.info(f"Loaded {len(self.enabled_pairs)} enabled pairs")
            else:
                logger.warning("No pair registry found, using defaults")
                self.enabled_pairs = {"EUR_USD", "GBP_USD"}
                self.current_config = {'enabled_pairs': list(self.enabled_pairs)}
        except Exception as e:
            logger.error(f"Error loading config: {e}")

    def _save_pairs(self):
        """Persist enabled pairs to registry"""
        try:
            config_path = "/app/config/pair_registry.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            self.current_config['enabled_pairs'] = sorted(self.enabled_pairs)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving pair registry: {e}")

    # Pair management helpers
    def enable_pair(self, pair: str) -> list:
        """Enable a trading pair"""
        self.enabled_pairs.add(pair)
        self._save_pairs()
        logger.info(f"Enabled pair {pair}")
        return list(self.enabled_pairs)

    def disable_pair(self, pair: str) -> list:
        """Disable a trading pair"""
        self.enabled_pairs.discard(pair)
        self._save_pairs()
        logger.info(f"Disabled pair {pair}")
        return list(self.enabled_pairs)

    # Metrics and performance helpers
    def get_live_metrics(self) -> dict:
        """Return current live trading metrics"""
        metrics = {
            'market_status': self.market_status,
            'trading_active': self.trading_active,
            'enabled_pairs': list(self.enabled_pairs),
            'timestamp': datetime.now().isoformat(),
            'risk': self.risk_manager.get_risk_summary(),
        }
        return metrics

    def get_performance_history(self) -> dict:
        """Return historical trade performance"""
        log_file = "/app/logs/trades.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading performance history: {e}")
        return {'trades': []}

    def get_performance_current(self) -> dict:
        """Return current P&L and statistics"""
        try:
            trades_data = self.get_performance_history()
            trades = trades_data.get('trades', [])
            
            # Calculate current performance metrics
            total_trades = len(trades)
            winning_trades = len([t for t in trades if t.get('status') == 'executed'])
            current_pnl = 0.0  # Would calculate from actual trade results
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'current_pnl': current_pnl,
                'daily_pnl': current_pnl,  # Would filter by today's trades
                'risk_level': self.risk_manager.get_risk_summary().get('risk_level', 'low'),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error calculating current performance: {e}")
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'current_pnl': 0.0,
                'daily_pnl': 0.0,
                'risk_level': 'unknown',
                'timestamp': datetime.now().isoformat()
            }

    def get_trading_signals(self) -> dict:
        """Return latest trading signals from engine output"""
        signals_file = "/app/data/trading_signals.json"
        try:
            if os.path.exists(signals_file):
                with open(signals_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
                return {'signals': data}
        except Exception as e:
            logger.error(f"Error reading trading signals: {e}")
        return {'signals': []}

    def get_config(self, key: str = None) -> dict:
        """Get configuration values"""
        if key:
            return {key: self.current_config.get(key)}
        return self.current_config

    def set_config(self, key: str, value) -> dict:
        """Update configuration"""
        try:
            self.current_config[key] = value
            self._save_pairs()  # This saves the entire config
            logger.info(f"Updated config: {key} = {value}")
            return {'status': 'success', 'key': key, 'value': value}
        except Exception as e:
            logger.error(f"Error setting config {key}: {e}")
            return {'status': 'error', 'message': str(e)}

    def execute_command(self, command: str) -> dict:
        """Execute a CLI command safely"""

        allowed_commands = {"status", "pairs", "config"}
        cmd_parts = command.split()
        base_cmd = cmd_parts[0] if cmd_parts else ""

        if base_cmd not in allowed_commands:
            return {"error": "Command not allowed"}

        try:
            result = subprocess.run(
                ["./bin/trader_cli", *cmd_parts],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {"returncode": -1, "error": str(e)}

    def check_market_hours(self):
        """Check if forex market is currently open"""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()  # 0=Monday, 6=Sunday
        hour = now.hour

        # Forex market is closed from Friday 22:00 UTC to Sunday 22:00 UTC
        if weekday == 5 and hour >= 22:  # Friday after 22:00
            return False
        if weekday == 6:  # All day Saturday
            return False
        if weekday == 0 and hour < 22:  # Sunday before 22:00
            return False

        return True

    def start_trading(self):
        """Start the trading service"""
        self.running = True
        self.trading_active = self.check_market_hours()

        logger.info("🚀 SEP Trading Service Started")
        logger.info(f"Enabled pairs: {list(self.enabled_pairs)}")
        logger.info(f"Market open: {self.trading_active}")

        while self.running:
            try:
                # Update market status
                self.market_status = "open" if self.check_market_hours() else "closed"

                if self.trading_active and self.check_market_hours():
                    self.trading_loop()
                else:
                    logger.info("Market closed - waiting...")
                    time.sleep(300)  # Check every 5 minutes when closed

            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)

    def trading_loop(self):
        """Main trading execution loop"""
        logger.info("📊 Executing trading loop...")

        # Fetch fresh candle data periodically (every 5 minutes)
        current_time = time.time()
        if not hasattr(self, 'last_candle_fetch') or current_time - self.last_candle_fetch > 300:
            self.fetch_candles_for_enabled_pairs()
            self.last_candle_fetch = current_time

        # Check for new trading signals
        signals_file = "/app/data/trading_signals.json"
        if os.path.exists(signals_file):
            try:
                with open(signals_file, 'r') as f:
                    signals = json.load(f)

                for sig in signals.get('signals', []):
                    pair = sig.get('pair')
                    if pair in self.enabled_pairs:
                        self.execute_signal(sig)

            except Exception as e:
                logger.error(f"Error processing signals: {e}")

        # Sleep before next iteration
        time.sleep(60)  # Check every minute during trading hours

    def execute_signal(self, trade_signal):
        """Execute a trading signal"""
        pair = trade_signal.get('pair')
        direction = trade_signal.get('direction')  # BUY or SELL
        confidence = trade_signal.get('confidence', 0)
        logger.info(f"🎯 Signal: {pair} {direction} (confidence: {confidence:.3f})")

        units = int(trade_signal.get('units', 100))
        if direction == 'SELL':
            units = -units

        allowed, reason = self.risk_manager.can_open(units)
        if not allowed:
            logger.warning(f"Risk check failed ({reason}) for {pair}")
            return

        status = 'simulated'
        if self.oanda:
            try:
                self.oanda.place_market_order(pair, units)
                status = 'executed'
            except Exception as exc:
                logger.error(f"Order placement failed: {exc}")
                status = 'error'

        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'pair': pair,
            'direction': direction,
            'confidence': confidence,
            'units': units,
            'status': status
        }

        log_file = "/app/logs/trades.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    trades = json.load(f)
            else:
                trades = {'trades': []}

            trades['trades'].append(trade_log)

            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing trade log: {e}")

        self.risk_manager.record(0.0)
    
    def fetch_and_store_candles(self, instrument: str, granularity: str = 'M5', count: int = 100) -> bool:
        """Fetch candles from OANDA and store in Valkey database"""
        if not self.oanda:
            logger.warning("OANDA connector not available")
            return False
        
        if not self.database:
            logger.warning("Database not available")
            return False
        
        try:
            logger.info(f"Fetching {count} {granularity} candles for {instrument}")
            candles = self.oanda.get_latest_candles(instrument, granularity, count)
            
            if candles:
                success = self.database.store_candle_data(instrument, granularity, candles)
                if success:
                    logger.info(f"Successfully stored {len(candles)} candles for {instrument}")
                    return True
                else:
                    logger.error(f"Failed to store candles for {instrument}")
                    return False
            else:
                logger.warning(f"No candles received for {instrument}")
                return False
                
        except Exception as e:
            logger.error(f"Error fetching/storing candles for {instrument}: {e}")
            return False
    
    def fetch_candles_for_enabled_pairs(self, granularity: str = 'M5', count: int = 100):
        """Fetch and store candles for all enabled trading pairs"""
        logger.info("Fetching candles for enabled trading pairs...")
        
        for pair in self.enabled_pairs:
            try:
                self.fetch_and_store_candles(pair, granularity, count)
                time.sleep(1)  # Rate limiting - be respectful to OANDA API
            except Exception as e:
                logger.error(f"Error processing candles for {pair}: {e}")
    
    def get_stored_candles(self, instrument: str, granularity: str = 'M5', limit: int = 100) -> list[dict]:
        """Retrieve stored candle data from database"""
        if not self.database:
            return []
        
        try:
            candles = self.database.get_candle_data(instrument, granularity, limit)
            logger.info(f"Retrieved {len(candles)} stored candles for {instrument}")
            return candles
        except Exception as e:
            logger.error(f"Error retrieving stored candles for {instrument}: {e}")
            return []

    def stop_trading(self):
        """Stop the trading service"""
        self.running = False
        logger.info("🛑 Trading service stopped")

    def get_quantum_metrics(self, instrument: str) -> dict:
        """Get quantum processing metrics from SEP Engine"""
        if not self.cli_bridge:
            # Fallback to simulated metrics
            return self._simulate_quantum_metrics(instrument)
        
        try:
            # Execute analyze command through CLI bridge
            result = self.cli_bridge.execute_command('analyze', [instrument], timeout=30)
            
            if result.status.value == 'completed' and result.stdout:
                # Parse the CLI output for quantum metrics
                output_lines = result.stdout.strip().split('\n')
                metrics = {}
                
                for line in output_lines:
                    if 'coherence:' in line.lower():
                        metrics['coherence'] = float(line.split(':')[-1].strip())
                    elif 'stability:' in line.lower():
                        metrics['stability'] = float(line.split(':')[-1].strip())
                    elif 'entropy:' in line.lower():
                        metrics['entropy'] = float(line.split(':')[-1].strip())
                    elif 'energy:' in line.lower():
                        metrics['energy'] = float(line.split(':')[-1].strip())
                
                return {
                    'instrument': instrument,
                    'timestamp': int(time.time() * 1000),
                    'metrics': metrics,
                    'source': 'sep_engine'
                }
            else:
                logger.warning(f"SEP Engine analyze command failed: {result.error_message}")
                return self._simulate_quantum_metrics(instrument)
                
        except Exception as e:
            logger.error(f"Error getting quantum metrics: {e}")
            return self._simulate_quantum_metrics(instrument)
    
    def get_sep_trading_signals(self, instrument: str) -> dict:
        """Get trading signals from SEP Engine"""
        if not self.cli_bridge:
            return self._simulate_trading_signals(instrument)
        
        try:
            # Execute monitor command to get signals
            result = self.cli_bridge.execute_command('monitor', [instrument], timeout=30)
            
            if result.status.value == 'completed' and result.stdout:
                signals = []
                output_lines = result.stdout.strip().split('\n')
                
                for line in output_lines:
                    if 'SIGNAL:' in line.upper():
                        # Parse signal line format: "SIGNAL: BUY EUR_USD confidence:0.85 strength:0.92"
                        parts = line.split()
                        if len(parts) >= 3:
                            signal_type = parts[1]
                            confidence = 0.0
                            strength = 0.0
                            
                            for part in parts:
                                if part.startswith('confidence:'):
                                    confidence = float(part.split(':')[1])
                                elif part.startswith('strength:'):
                                    strength = float(part.split(':')[1])
                            
                            signals.append({
                                'type': signal_type,
                                'instrument': instrument,
                                'confidence': confidence,
                                'strength': strength,
                                'timestamp': int(time.time() * 1000)
                            })
                
                return {
                    'instrument': instrument,
                    'signals': signals,
                    'count': len(signals),
                    'source': 'sep_engine'
                }
            else:
                return self._simulate_trading_signals(instrument)
                
        except Exception as e:
            logger.error(f"Error getting SEP trading signals: {e}")
            return self._simulate_trading_signals(instrument)
    
    def get_quantum_analysis(self, instrument: str) -> dict:
        """Get complete quantum analysis from SEP Engine"""
        try:
            # Get both metrics and signals
            metrics = self.get_quantum_metrics(instrument)
            signals = self.get_sep_trading_signals(instrument)
            
            # Get market data for context
            key = f"market:price:{instrument}"
            market_data = []
            
            if self.database:
                try:
                    with self.database.get_client() as r:
                        to_ts = int(time.time() * 1000)
                        from_ts = to_ts - 3600 * 1000  # Last hour
                        raw_rows = r.zrangebyscore(key, from_ts, to_ts)
                        market_data = [json.loads(row) for row in raw_rows[-10:]]  # Last 10 points
                except Exception:
                    pass
            
            return {
                'instrument': instrument,
                'timestamp': int(time.time() * 1000),
                'quantum_metrics': metrics.get('metrics', {}),
                'trading_signals': signals.get('signals', []),
                'market_context': market_data,
                'analysis_summary': {
                    'signal_count': signals.get('count', 0),
                    'data_points': len(market_data),
                    'latest_price': market_data[-1].get('c', 0) if market_data else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting quantum analysis: {e}")
            return {
                'instrument': instrument,
                'timestamp': int(time.time() * 1000),
                'error': str(e)
            }
    
    def _simulate_quantum_metrics(self, instrument: str) -> dict:
        """Simulate quantum metrics when SEP Engine is unavailable"""
        import random
        import math
        
        # Generate realistic-looking quantum metrics
        base_time = time.time()
        volatility = random.uniform(0.1, 0.8)
        
        coherence = max(0.0, min(1.0, 0.7 - volatility + random.gauss(0, 0.1)))
        stability = max(0.0, min(1.0, 0.6 - volatility * 0.8 + random.gauss(0, 0.1))) 
        entropy = max(0.0, min(1.0, volatility * 1.2 + random.gauss(0.3, 0.1)))
        energy = random.uniform(0.5, 2.0) * (1 + volatility)
        
        return {
            'instrument': instrument,
            'timestamp': int(base_time * 1000),
            'metrics': {
                'coherence': round(coherence, 3),
                'stability': round(stability, 3), 
                'entropy': round(entropy, 3),
                'energy': round(energy, 3)
            },
            'source': 'simulation'
        }
    
    def _simulate_trading_signals(self, instrument: str) -> dict:
        """Simulate trading signals when SEP Engine is unavailable"""
        import random
        
        signals = []
        signal_count = random.randint(0, 3)
        
        for i in range(signal_count):
            signal_type = random.choice(['BUY', 'SELL', 'HOLD'])
            confidence = random.uniform(0.3, 0.95)
            strength = random.uniform(0.2, 0.9)
            
            signals.append({
                'type': signal_type,
                'instrument': instrument,
                'confidence': round(confidence, 3),
                'strength': round(strength, 3),
                'timestamp': int(time.time() * 1000) - random.randint(0, 300000)  # Up to 5 min ago
            })
        
        return {
            'instrument': instrument,
            'signals': signals,
            'count': len(signals),
            'source': 'simulation'
        }
        try:
            candles = self.database.get_candle_data(instrument, granularity, limit)
            logger.info(f"Retrieved {len(candles)} stored candles for {instrument}")
            return candles
        except Exception as e:
            logger.error(f"Error retrieving stored candles for {instrument}: {e}")
            return []

    def stop_trading(self):
        """Stop the trading service"""
        self.running = False
        logger.info("🛑 Trading service stopped")

class TradingAPIHandler(BaseHTTPRequestHandler):
    def __init__(self, trading_service, *args, **kwargs):
        self.trading_service = trading_service
        super().__init__(*args, **kwargs)

    def _set_cors_headers(self):
        """Set CORS headers for all responses"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With')
        self.send_header('Access-Control-Max-Age', '3600')

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        correlation_id = self.headers.get("X-Request-ID", str(uuid4()))
        token = correlation_id_var.set(correlation_id)
        try:
            self.send_response(200)
            self._set_cors_headers()
            self.end_headers()
        finally:
            correlation_id_var.reset(token)

    def do_HEAD(self):
        """Handle HEAD requests - same as GET but without response body"""
        correlation_id = self.headers.get("X-Request-ID", str(uuid4()))
        token = correlation_id_var.set(correlation_id)
        try:
            logger.info(f"Handling HEAD request for {self.path}")
            parsed_path = urlparse(self.path)
            path = parsed_path.path

            if path == '/health' or path == '/api/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                logger.info(f"Responded 200 to HEAD {self.path}")
            else:
                self.send_response(404)
                self._set_cors_headers()
                self.end_headers()
                logger.info(f"Responded 404 to HEAD {self.path}")
        finally:
            correlation_id_var.reset(token)

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        correlation_id = self.headers.get("X-Request-ID", str(uuid4()))
        token = correlation_id_var.set(correlation_id)
        logger.info(f"GET request for {path}")

        try:
            if path == '/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = {
                    'status': 'healthy',
                    'market_status': self.trading_service.market_status,
                    'trading_active': self.trading_service.trading_active,
                    'enabled_pairs': list(self.trading_service.enabled_pairs),
                    'build_hash': self.trading_service.build_hash,
                    'config_version': self.trading_service.config_version,
                    'timestamp': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = {
                    'status': 'healthy',
                    'service': 'SEP Professional Trader-Bot',
                    'market_status': self.trading_service.market_status,
                    'trading_active': self.trading_service.trading_active,
                    'build_hash': self.trading_service.build_hash,
                    'config_version': self.trading_service.config_version,
                    'timestamp': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/status':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = {
                    'service': 'SEP Professional Trader-Bot',
                    'version': '1.0.0',
                    'status': 'running' if self.trading_service.running else 'stopped',
                    'market': self.trading_service.market_status,
                    'pairs': list(self.trading_service.enabled_pairs),
                    'last_sync': self.trading_service.last_sync
                }
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/system-status/config':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                poll_interval = int(os.environ.get('SYSTEM_STATUS_POLL_INTERVAL_MS', '30000'))
                components = [
                    {'name': 'SEP Engine', 'key': 'engine_status'},
                    {'name': 'Memory Tiers', 'key': 'memory_status'},
                    {'name': 'Trading System', 'key': 'trading_status'},
                    {'name': 'WebSocket Service', 'key': 'websocket'},
                ]
                response = {
                    'poll_interval': poll_interval,
                    'components': components,
                }
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/pairs':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = {'pairs': list(self.trading_service.enabled_pairs)}
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/metrics/live':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = self.trading_service.get_live_metrics()
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/performance/history':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = self.trading_service.get_performance_history()
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/performance/current':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = self.trading_service.get_performance_current()
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/signals':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = self.trading_service.get_trading_signals()
                self.wfile.write(json.dumps(response).encode())

            elif path.startswith('/api/config'):
                query_params = urlparse(self.path).query
                key = None
                if query_params:
                    for param in query_params.split('&'):
                        if param.startswith('key='):
                            key = param.split('=', 1)[1]
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = self.trading_service.get_config(key)
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/market-data':
                query_params = dict(param.split('=') for param in urlparse(self.path).query.split('&') if '=' in param) if urlparse(self.path).query else {}
                instrument = query_params.get('instrument', 'EUR_USD')
                to_ts = int(query_params.get('to', time.time() * 1000))
                from_ts = int(query_params.get('from', to_ts - 48 * 3600 * 1000))
                
                key = f"market:price:{instrument}"
                
                try:
                    with self.trading_service.database.get_client() as r:
                        raw_rows = r.zrangebyscore(key, from_ts, to_ts)
                        rows = [json.loads(row) for row in raw_rows]
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    
                    response = {
                        "instrument": instrument,
                        "from": from_ts,
                        "to": to_ts,
                        "rows": rows
                    }
                    self.wfile.write(json.dumps(response).encode())
                except Exception as e:
                    logger.error(f"Error getting market data for {instrument}: {e}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())

            elif path == '/api/quantum-metrics':
                query_params = dict(param.split('=') for param in urlparse(self.path).query.split('&') if '=' in param) if urlparse(self.path).query else {}
                instrument = query_params.get('instrument', 'EUR_USD')
                
                try:
                    # Get quantum processing metrics from SEP Engine
                    metrics = self.trading_service.get_quantum_metrics(instrument)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps(metrics).encode())
                except Exception as e:
                    logger.error(f"Error getting quantum metrics for {instrument}: {e}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())

            elif path == '/api/quantum-signals':
                query_params = dict(param.split('=') for param in urlparse(self.path).query.split('&') if '=' in param) if urlparse(self.path).query else {}
                instrument = query_params.get('instrument', 'EUR_USD')
                
                try:
                    # Get trading signals from SEP Engine
                    signals = self.trading_service.get_sep_trading_signals(instrument)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps(signals).encode())
                except Exception as e:
                    logger.error(f"Error getting SEP trading signals for {instrument}: {e}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())

            elif path == '/api/quantum-analysis':
                query_params = dict(param.split('=') for param in urlparse(self.path).query.split('&') if '=' in param) if urlparse(self.path).query else {}
                instrument = query_params.get('instrument', 'EUR_USD')
                
                try:
                    # Get full quantum analysis from SEP Engine
                    analysis = self.trading_service.get_quantum_analysis(instrument)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps(analysis).encode())
                except Exception as e:
                    logger.error(f"Error getting quantum analysis for {instrument}: {e}")
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': str(e)}).encode())

            elif path.startswith('/api/candles/'):
                # Extract instrument from path: /api/candles/{instrument}
                instrument = path.split('/api/candles/')[-1]
                query_params = dict(param.split('=') for param in urlparse(self.path).query.split('&') if '=' in param) if urlparse(self.path).query else {}
                
                granularity = query_params.get('granularity', 'M5')
                limit = int(query_params.get('limit', '100'))
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                
                candles = self.trading_service.get_stored_candles(instrument, granularity, limit)
                response = {
                    'instrument': instrument,
                    'granularity': granularity,
                    'candles': candles,
                    'count': len(candles),
                    'timestamp': datetime.now().isoformat()
                }
                self.wfile.write(json.dumps(response).encode())

            else:
                self.send_response(404)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())

        except Exception as e:
            logger.error(f"Error handling GET {path}: {e}")
            self.send_response(500)
            self._set_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        finally:
            correlation_id_var.reset(token)

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        correlation_id = self.headers.get("X-Request-ID", str(uuid4()))
        token = correlation_id_var.set(correlation_id)
        logger.info(f"POST request for {path}")

        try:
            if path == '/api/data/reload':
                self.trading_service.load_config()
                self.trading_service.last_sync = datetime.now().isoformat()

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                response = {'status': 'reloaded', 'timestamp': datetime.now().isoformat()}
                self.wfile.write(json.dumps(response).encode())

            elif path.startswith('/api/pairs/') and path.endswith('/enable'):
                pair = path.split('/')[3]
                pairs = self.trading_service.enable_pair(pair)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'pairs': pairs}).encode())

            elif path.startswith('/api/pairs/') and path.endswith('/disable'):
                pair = path.split('/')[3]
                pairs = self.trading_service.disable_pair(pair)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'pairs': pairs}).encode())

            elif path == '/api/candles/fetch':
                # Trigger manual candle data fetching
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length) if length else b'{}'
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = {}
                
                instrument = data.get('instrument')
                granularity = data.get('granularity', 'M5')
                count = data.get('count', 100)
                
                if instrument:
                    success = self.trading_service.fetch_and_store_candles(instrument, granularity, count)
                    status_code = 200 if success else 500
                    response = {
                        'success': success,
                        'instrument': instrument,
                        'granularity': granularity,
                        'count': count,
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    # Fetch for all enabled pairs
                    self.trading_service.fetch_candles_for_enabled_pairs(granularity, count)
                    status_code = 200
                    response = {
                        'success': True,
                        'action': 'fetch_all_pairs',
                        'granularity': granularity,
                        'count': count,
                        'pairs': list(self.trading_service.enabled_pairs),
                        'timestamp': datetime.now().isoformat()
                    }
                
                self.send_response(status_code)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            elif path == '/api/commands/execute':
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length) if length else b'{}'
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = {}
                command = data.get('command', '')
                result = self.trading_service.execute_command(command)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())

            elif path == '/api/config/set':
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length) if length else b'{}'
                try:
                    data = json.loads(body)
                    key = data.get('key')
                    value = data.get('value')
                    if key:
                        result = self.trading_service.set_config(key, value)
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self._set_cors_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode())
                    else:
                        self.send_response(400)
                        self._set_cors_headers()
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({'error': 'Missing key parameter'}).encode())
                except json.JSONDecodeError:
                    self.send_response(400)
                    self._set_cors_headers()
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'error': 'Invalid JSON'}).encode())
            else:
                self.send_response(404)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Endpoint not found'}).encode())

        except Exception as e:
            logger.error(f"Error handling POST {path}: {e}")
            self.send_response(500)
            self._set_cors_headers()
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode())
        finally:
            correlation_id_var.reset(token)

    def log_message(self, format, *args):
        """Override to use our logger"""
        logger.info(f"HTTP: {format % args}")

def create_handler(trading_service):
    """Create HTTP handler with trading service"""
    def handler(*args, **kwargs):
        return TradingAPIHandler(trading_service, *args, **kwargs)
    return handler

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Received shutdown signal")
    trading_service.stop_trading()
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create trading service
    trading_service = TradingService()

    # Start HTTP server in background
    port = int(os.environ.get('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), create_handler(trading_service))
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    logger.info(f"🌐 API server started on port {port}")

    # Skip embedded WebSocket server - using dedicated WebSocket container instead
    logger.info(f"📡 Using dedicated WebSocket container instead of embedded server")

    try:
        # Start trading service (blocking)
        trading_service.start_trading()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        trading_service.stop_trading()
        server.shutdown()
