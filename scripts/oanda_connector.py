"""
SEP Professional Trading System - OANDA Connector
Lightweight Python OANDA API connector for remote droplet execution
"""

import os
import json
import logging
import requests
from datetime import datetime
from typing import Optional, Dict, List, Tuple

# Optional dotenv import for development environments
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    def load_dotenv(*args, **kwargs):
        pass  # No-op fallback

logger = logging.getLogger(__name__)

class OandaConnector:
    """Professional OANDA API connector for live trading execution"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize OANDA connector with API credentials
        
        Args:
            config_path: Path to OANDA environment configuration file (auto-detected if None)
        """
        self.api_key = None
        self.account_id = None
        self.api_base = None
        self.session = requests.Session()
        self.connected = False
        
        # Auto-detect config path if not provided
        if config_path is None:
            if os.path.exists('/app'):
                # Docker/containerized environment
                config_path = "/app/config/OANDA.env"
            else:
                # Local development environment - try multiple locations
                local_paths = [
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'OANDA.env'),
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'OANDA.env'),
                    './config/OANDA.env',
                    './OANDA.env'
                ]
                config_path = None
                for path in local_paths:
                    if os.path.exists(path):
                        config_path = path
                        break
                
                if config_path is None:
                    config_path = "./config/OANDA.env"  # Default fallback
        
        # Load configuration
        self._load_config(config_path)
        
        if self.api_key and self.account_id:
            self._setup_session()
            self.connected = self._test_connection()
        else:
            logger.warning("OANDA credentials not found - connector not initialized")
    
    def _load_config(self, config_path: str):
        """Load OANDA API configuration from environment file"""
        try:
            if os.path.exists(config_path):
                if DOTENV_AVAILABLE:
                    load_dotenv(config_path)
                    logger.info("Loaded OANDA config using python-dotenv")
                else:
                    # Manual parsing fallback
                    with open(config_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                os.environ[key.strip()] = value.strip()
                    logger.info("Loaded OANDA config using manual parsing")
                
                self.api_key = os.getenv('OANDA_API_KEY')
                self.account_id = os.getenv('OANDA_ACCOUNT_ID')
                
                # Determine API environment
                environment = os.getenv('OANDA_ENVIRONMENT', 'practice').lower()
                if environment == 'live':
                    self.api_base = "https://api-fxtrade.oanda.com"
                    logger.warning("🔴 LIVE TRADING MODE ENABLED")
                else:
                    self.api_base = "https://api-fxpractice.oanda.com"
                    logger.info("📊 Practice trading mode enabled")
                
                logger.info(f"OANDA config loaded from {config_path}")
            else:
                logger.warning(f"OANDA config file not found: {config_path}")
                
        except Exception as e:
            logger.error(f"Error loading OANDA config: {e}")
    
    def _setup_session(self):
        """Setup HTTP session with OANDA API headers"""
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept-Datetime-Format': 'RFC3339'
        })
    
    def _test_connection(self) -> bool:
        """Test connection to OANDA API"""
        try:
            url = f"{self.api_base}/v3/accounts/{self.account_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                account_info = response.json()
                logger.info(f"✅ OANDA connection established - Account: {self.account_id}")
                return True
            else:
                logger.error(f"OANDA connection failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"OANDA connection test failed: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """Get current account information"""
        if not self.connected:
            logger.error("OANDA not connected")
            return None
        
        try:
            url = f"{self.api_base}/v3/accounts/{self.account_id}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()['account']
            else:
                logger.error(f"Failed to get account info: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    def get_current_price(self, instrument: str) -> Optional[Tuple[float, float]]:
        """
        Get current bid/ask prices for an instrument
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            
        Returns:
            Tuple of (bid, ask) prices or None
        """
        if not self.connected:
            logger.error("OANDA not connected")
            return None
        
        try:
            url = f"{self.api_base}/v3/accounts/{self.account_id}/pricing"
            params = {'instruments': instrument}
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                pricing = response.json()['prices'][0]
                bid = float(pricing['bids'][0]['price'])
                ask = float(pricing['asks'][0]['price'])
                return (bid, ask)
            else:
                logger.error(f"Failed to get price for {instrument}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting price for {instrument}: {e}")
            return None
    
    def place_market_order(self, instrument: str, units: int, 
                          stop_loss: Optional[float] = None,
                          take_profit: Optional[float] = None) -> Optional[Dict]:
        """
        Place a market order
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            units: Position size (positive for buy, negative for sell)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price
            
        Returns:
            Order response dictionary or None
        """
        if not self.connected:
            logger.error("OANDA not connected")
            return None
        
        try:
            # Build order request
            order_request = {
                'order': {
                    'type': 'MARKET',
                    'instrument': instrument,
                    'units': str(units),
                    'timeInForce': 'FOK'  # Fill or Kill
                }
            }
            
            # Add stop loss if specified
            if stop_loss:
                order_request['order']['stopLossOnFill'] = {
                    'price': str(stop_loss)
                }
            
            # Add take profit if specified
            if take_profit:
                order_request['order']['takeProfitOnFill'] = {
                    'price': str(take_profit)
                }
            
            url = f"{self.api_base}/v3/accounts/{self.account_id}/orders"
            response = self.session.post(url, 
                                       data=json.dumps(order_request),
                                       timeout=10)
            
            if response.status_code == 201:
                result = response.json()
                logger.info(f"✅ Market order executed: {instrument} {units} units")
                return result
            else:
                logger.error(f"Order failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    def get_open_positions(self) -> List[Dict]:
        """Get all open positions"""
        if not self.connected:
            logger.error("OANDA not connected")
            return []
        
        try:
            url = f"{self.api_base}/v3/accounts/{self.account_id}/openPositions"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json()['positions']
            else:
                logger.error(f"Failed to get positions: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def close_position(self, instrument: str, units: Optional[str] = None) -> Optional[Dict]:
        """
        Close a position (or part of it)
        
        Args:
            instrument: Currency pair to close
            units: Specific units to close (None for all)
            
        Returns:
            Close response dictionary or None
        """
        if not self.connected:
            logger.error("OANDA not connected")
            return None
        
        try:
            url = f"{self.api_base}/v3/accounts/{self.account_id}/positions/{instrument}/close"
            
            close_request = {}
            if units:
                # Determine if closing long or short
                if units.startswith('-'):
                    close_request['shortUnits'] = units[1:]  # Remove negative sign
                else:
                    close_request['longUnits'] = units
            else:
                close_request['longUnits'] = 'ALL'
                close_request['shortUnits'] = 'ALL'
            
            response = self.session.put(url, 
                                      data=json.dumps(close_request),
                                      timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Position closed: {instrument}")
                return result
            else:
                logger.error(f"Close failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def get_candles(self, instrument: str, granularity: str = 'M5',
                   count: int = 100, from_time: str = None, to_time: str = None) -> Optional[List[Dict]]:
        """
        Get historical candle data for an instrument
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            granularity: Candle granularity ('S5', 'S10', 'S15', 'S30', 'M1', 'M2', 'M4', 'M5', 'M10', 'M15', 'M30', 'H1', 'H2', 'H3', 'H4', 'H6', 'H8', 'H12', 'D', 'W', 'M')
            count: Number of candles to retrieve (max 5000)
            from_time: Start time in RFC3339 format
            to_time: End time in RFC3339 format
            
        Returns:
            List of candle dictionaries or None
        """
        if not self.connected:
            logger.error("OANDA not connected")
            return None
        
        try:
            url = f"{self.api_base}/v3/instruments/{instrument}/candles"
            params = {
                'granularity': granularity,
                'count': min(count, 5000)  # OANDA max limit
            }
            
            if from_time:
                params['from'] = from_time
            if to_time:
                params['to'] = to_time
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                candles = data.get('candles', [])
                logger.info(f"✅ Retrieved {len(candles)} {granularity} candles for {instrument}")
                return candles
            else:
                logger.error(f"Failed to get candles for {instrument}: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting candles for {instrument}: {e}")
            return None
    
    def get_latest_candles(self, instrument: str, granularity: str = 'M5', count: int = 100) -> Optional[List[Dict]]:
        """
        Get the most recent candle data for an instrument
        
        Args:
            instrument: Currency pair (e.g., 'EUR_USD')
            granularity: Candle granularity
            count: Number of recent candles to retrieve
            
        Returns:
            List of recent candles or None
        """
        return self.get_candles(instrument, granularity, count)
    
    def get_instruments(self) -> Optional[List[Dict]]:
        """
        Get available trading instruments
        
        Returns:
            List of instrument dictionaries or None
        """
        if not self.connected:
            logger.error("OANDA not connected")
            return None
        
        try:
            url = f"{self.api_base}/v3/accounts/{self.account_id}/instruments"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                instruments = response.json()['instruments']
                logger.info(f"✅ Retrieved {len(instruments)} trading instruments")
                return instruments
            else:
                logger.error(f"Failed to get instruments: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting instruments: {e}")
            return None
    
    def get_trading_summary(self) -> Dict:
        """Get trading activity summary"""
        account_info = self.get_account_info()
        open_positions = self.get_open_positions()
        
        return {
            'connected': self.connected,
            'account_id': self.account_id,
            'balance': account_info.get('balance', '0') if account_info else '0',
            'currency': account_info.get('currency', 'USD') if account_info else 'USD',
            'open_positions': len(open_positions),
            'margin_available': account_info.get('marginAvailable', '0') if account_info else '0',
            'api_base': self.api_base
        }