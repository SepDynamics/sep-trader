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
import subprocess

# Setup logging with environment-appropriate paths
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
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class TradingService:
    def __init__(self):
        self.running = False
        self.enabled_pairs = set()
        self.trading_active = False
        self.last_sync = None
        self.market_status = "unknown"
        self.current_config = {}

        # Setup connectors and risk manager
        try:
            self.oanda = OandaConnector()
        except Exception as e:
            logger.warning(f"OANDA connector unavailable: {e}")
            self.oanda = None

        self.risk_manager = RiskManager(RiskLimits())

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
        """Execute a CLI command and return its output"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
            }
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return {'returncode': -1, 'error': str(e)}

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
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_HEAD(self):
        """Handle HEAD requests - same as GET but without response body"""
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

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
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

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
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

    # Start WebSocket metrics server
    ws_port = int(os.environ.get('WS_PORT', 8765))
    backend_url = f"http://localhost:{port}"
    start_websocket_server(ws_port, backend_url)
    logger.info(f"📡 WebSocket server started on port {ws_port}")

    try:
        # Start trading service (blocking)
        trading_service.start_trading()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        trading_service.stop_trading()
        server.shutdown()
