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

# Allow importing risk controls
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from trading.risk import RiskManager, RiskLimits  # noqa: E402
from oanda_connector import OandaConnector  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/trading_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingService:
    def __init__(self):
        self.running = False
        self.enabled_pairs = set()
        self.trading_active = False
        self.last_sync = None
        self.market_status = "unknown"

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
                    logger.info(f"Loaded {len(self.enabled_pairs)} enabled pairs")
            else:
                logger.warning("No pair registry found, using defaults")
                self.enabled_pairs = {"EUR_USD", "GBP_USD"}
        except Exception as e:
            logger.error(f"Error loading config: {e}")

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

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'status': 'healthy',
                'market_status': self.trading_service.market_status,
                'trading_active': self.trading_service.trading_active,
                'enabled_pairs': list(self.trading_service.enabled_pairs),
                'timestamp': datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(response).encode())

        elif path == '/api/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
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

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == '/api/data/reload':
            self.trading_service.load_config()
            self.trading_service.last_sync = datetime.now().isoformat()

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'reloaded', 'timestamp': datetime.now().isoformat()}
            self.wfile.write(json.dumps(response).encode())

        else:
            self.send_response(404)
            self.end_headers()

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

    try:
        # Start trading service (blocking)
        trading_service.start_trading()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        trading_service.stop_trading()
        server.shutdown()
