#!/usr/bin/env python3
"""
Live Stream Trader - Bootstraps with historical data, then uses live streaming
"""
import os
import requests
import json
import time
import subprocess
import threading
from datetime import datetime, timedelta
from collections import deque

class LiveStreamTrader:
    def __init__(self):
        self.api_key = os.getenv('OANDA_API_KEY')
        self.account_id = os.getenv('OANDA_ACCOUNT_ID')
        self.base_url = 'https://api-fxpractice.oanda.com'
        self.stream_url = 'https://stream-fxpractice.oanda.com'
        
        # Rolling window of candles (keep last 500 M1 candles = ~8 hours)
        self.candles = deque(maxlen=500)
        self.last_analysis_time = 0
        self.trading_active = True
        
        # Statistics tracking
        self.stats = {
            'ticks_received': 0,
            'candles_completed': 0,
            'analysis_runs': 0,
            'signals_generated': 0,
            'high_confidence_signals': 0,
            'trades_executed': 0,
            'trades_successful': 0,
            'trades_failed': 0,
            'last_price': 0,
            'last_tick_time': None,
            'stream_start_time': None,
            'last_signal': None,
            'last_confidence': 0,
            'last_coherence': 0
        }
        
    def bootstrap_historical_data(self):
        """Bootstrap with 24 hours of historical data"""
        print("🔄 Bootstrapping with historical data...")
        
        try:
            # Fetch last 24 hours of M1 data
            result = subprocess.run([
                './build/examples/oanda_historical_fetcher',
                'EUR_USD', 'M1', '24'
            ], capture_output=True, text=True, cwd='/sep')
            
            if result.returncode == 0:
                # Load the historical data
                with open('/tmp/live_oanda_data.json', 'r') as f:
                    data = json.load(f)
                    
                for candle in data['candles']:
                    self.candles.append({
                        'time': candle['time'],
                        'open': float(candle['mid']['o']),
                        'high': float(candle['mid']['h']),
                        'low': float(candle['mid']['l']),
                        'close': float(candle['mid']['c']),
                        'volume': candle.get('volume', 100)
                    })
                
                print(f"✅ Loaded {len(self.candles)} historical candles")
                return True
        except Exception as e:
            print(f"❌ Bootstrap failed: {e}")
            return False
    
    def get_account_balance(self):
        """Get current account balance"""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(f'{self.base_url}/v3/accounts/{self.account_id}', headers=headers)
            if response.status_code == 200:
                data = response.json()
                balance = float(data['account']['balance'])
                return balance
        except:
            pass
        return 0.0
    
    def get_open_trades(self):
        """Get current open trades"""
        try:
            headers = {'Authorization': f'Bearer {self.api_key}'}
            response = requests.get(f'{self.base_url}/v3/accounts/{self.account_id}/openTrades', headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get('trades', [])
        except:
            pass
        return []
    
    def print_status(self):
        """Print comprehensive status with all stats"""
        # Get current account info
        balance = self.get_account_balance()
        open_trades = self.get_open_trades()
        
        # Calculate uptime
        uptime = ""
        if self.stats['stream_start_time']:
            from datetime import timezone
            now_utc = datetime.now(timezone.utc)
            if self.stats['stream_start_time'].tzinfo is None:
                start_time_utc = self.stats['stream_start_time'].replace(tzinfo=timezone.utc)
            else:
                start_time_utc = self.stats['stream_start_time']
            uptime_seconds = (now_utc - start_time_utc).total_seconds()
            uptime = f"{int(uptime_seconds//3600):02d}:{int((uptime_seconds%3600)//60):02d}:{int(uptime_seconds%60):02d}"
        
        # Calculate total P/L from open trades
        total_pnl = sum(float(trade['unrealizedPL']) for trade in open_trades)
        
        # Calculate success rates
        success_rate = (self.stats['trades_successful'] / self.stats['trades_executed'] * 100) if self.stats['trades_executed'] > 0 else 0
        high_conf_rate = (self.stats['high_confidence_signals'] / self.stats['signals_generated'] * 100) if self.stats['signals_generated'] > 0 else 0
        
        print(f"\n{'='*100}")
        print(f"🔴 LIVE TRADING STATUS | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Uptime: {uptime}")
        print(f"{'='*100}")
        
        # Account Information
        print(f"💰 ACCOUNT: Balance: ${balance:.2f} | Total P/L: ${total_pnl:+.2f} | Open Trades: {len(open_trades)}")
        
        # Market Data Statistics  
        print(f"📊 MARKET DATA: Ticks: {self.stats['ticks_received']:,} | Candles: {self.stats['candles_completed']:,} | Current Price: {self.stats['last_price']:.5f}")
        if self.stats['last_tick_time']:
            # Convert to UTC for comparison
            from datetime import timezone
            now_utc = datetime.now(timezone.utc)
            if self.stats['last_tick_time'].tzinfo is None:
                # If last_tick_time is naive, assume it's UTC
                last_tick_utc = self.stats['last_tick_time'].replace(tzinfo=timezone.utc)
            else:
                last_tick_utc = self.stats['last_tick_time']
            
            last_tick_age = (now_utc - last_tick_utc).total_seconds()
            print(f"📡 STREAM STATUS: Last tick {last_tick_age:.1f}s ago | Stream active: {'✅' if last_tick_age < 10 else '❌'}")
        
        # Analysis Performance
        print(f"🔬 ANALYSIS: Runs: {self.stats['analysis_runs']} | Signals: {self.stats['signals_generated']} | High-Conf: {self.stats['high_confidence_signals']} ({high_conf_rate:.1f}%)")
        
        # Trading Performance
        print(f"⚡ TRADING: Executed: {self.stats['trades_executed']} | Success: {self.stats['trades_successful']} ({success_rate:.1f}%) | Failed: {self.stats['trades_failed']}")
        
        # Last Signal Details
        if self.stats['last_signal']:
            print(f"🎯 LAST SIGNAL: {self.stats['last_signal']} | Confidence: {self.stats['last_confidence']:.3f} | Coherence: {self.stats['last_coherence']:.3f}")
            threshold_status = "✅ TRADEABLE" if (self.stats['last_confidence'] >= 0.65 and self.stats['last_coherence'] >= 0.30) else "❌ BELOW THRESHOLD"
            print(f"📏 THRESHOLDS: {threshold_status} | Need: Conf≥0.65 & Coh≥0.30")
        
        # Open Positions Details
        if open_trades:
            print(f"📈 OPEN POSITIONS ({len(open_trades)}):")
            for trade in open_trades:
                direction = "BUY" if float(trade['currentUnits']) > 0 else "SELL"
                units = abs(float(trade['currentUnits']))
                pnl = float(trade['unrealizedPL'])
                open_time = trade.get('openTime', 'Unknown')[:19].replace('T', ' ')
                entry_price = float(trade.get('price', 0))
                current_price = float(trade.get('currentPrice', {}).get('mid', 0)) if trade.get('currentPrice') else 0
                
                # Calculate pips moved
                pips_moved = 0
                if entry_price and current_price:
                    if direction == "BUY":
                        pips_moved = (current_price - entry_price) * 10000
                    else:
                        pips_moved = (entry_price - current_price) * 10000
                
                pnl_color = "🟢" if pnl >= 0 else "🔴"
                print(f"   {pnl_color} Trade {trade['id']}: {direction} {units} units | Entry: {entry_price:.5f} | P/L: ${pnl:+.2f} ({pips_moved:+.1f} pips) | Open: {open_time}")
        else:
            print(f"📈 OPEN POSITIONS: None")
        
        print(f"{'='*100}")
        print("")  # Extra line for readability

    def start_live_stream(self):
        """Start live price streaming in background thread"""
        def stream_worker():
            headers = {'Authorization': f'Bearer {self.api_key}'}
            url = f"{self.stream_url}/v3/accounts/{self.account_id}/pricing/stream"
            params = {'instruments': 'EUR_USD'}
            
            print("🌊 Starting live price stream...")
            self.stats['stream_start_time'] = datetime.now()
            
            try:
                response = requests.get(url, headers=headers, params=params, stream=True)
                
                current_candle = None
                candle_start_time = None
                
                for line in response.iter_lines():
                    if not self.trading_active:
                        break
                        
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            
                            if data.get('type') == 'PRICE':
                                self.stats['ticks_received'] += 1
                                
                                price_data = data
                                timestamp = datetime.fromisoformat(price_data['time'].replace('Z', '+00:00'))
                                self.stats['last_tick_time'] = timestamp
                                
                                price = float(price_data['bids'][0]['price'])
                                self.stats['last_price'] = price
                                
                                # Debug output every 10 ticks
                                if self.stats['ticks_received'] % 10 == 0:
                                    print(f"[TICK {self.stats['ticks_received']:,}] {timestamp.strftime('%H:%M:%S.%f')[:-3]} | Price: {price:.5f}")
                                
                                # Round to minute boundary
                                minute_time = timestamp.replace(second=0, microsecond=0)
                                
                                if candle_start_time != minute_time:
                                    # New minute - close previous candle and start new one
                                    if current_candle:
                                        self.candles.append(current_candle)
                                        self.stats['candles_completed'] += 1
                                        print(f"🕯️  CANDLE COMPLETE #{self.stats['candles_completed']} | {candle_start_time.strftime('%H:%M')} | O:{current_candle['open']:.5f} H:{current_candle['high']:.5f} L:{current_candle['low']:.5f} C:{current_candle['close']:.5f} | Vol: {current_candle['volume']}")
                                        self.check_for_signals()
                                        self.print_status()
                                    
                                    # Start new candle
                                    candle_start_time = minute_time
                                    current_candle = {
                                        'time': minute_time.isoformat(),
                                        'open': price,
                                        'high': price,
                                        'low': price,
                                        'close': price,
                                        'volume': 1
                                    }
                                    print(f"🆕 NEW CANDLE STARTED | {candle_start_time.strftime('%H:%M')} | Open: {price:.5f}")
                                else:
                                    # Update current candle
                                    if current_candle:
                                        current_candle['high'] = max(current_candle['high'], price)
                                        current_candle['low'] = min(current_candle['low'], price)
                                        current_candle['close'] = price
                                        current_candle['volume'] += 1
                                        
                        except Exception as e:
                            print(f"Stream parsing error: {e}")
                            continue
                            
            except Exception as e:
                print(f"❌ Stream error: {e}")
                print("🔄 Restarting stream in 5 seconds...")
                time.sleep(5)
                if self.trading_active:
                    self.start_live_stream()
        
        # Start streaming in background thread
        stream_thread = threading.Thread(target=stream_worker, daemon=True)
        stream_thread.start()
    
    def save_current_data(self):
        """Save current candle data for analysis"""
        # Convert to OANDA format
        oanda_data = {
            'instrument': 'EUR_USD',
            'granularity': 'M1',
            'candles': []
        }
        
        for candle in list(self.candles):
            oanda_data['candles'].append({
                'time': candle['time'],
                'volume': candle['volume'],
                'mid': {
                    'o': f"{candle['open']:.5f}",
                    'h': f"{candle['high']:.5f}",
                    'l': f"{candle['low']:.5f}",
                    'c': f"{candle['close']:.5f}"
                }
            })
        
        with open('/tmp/live_stream_data.json', 'w') as f:
            json.dump(oanda_data, f)
    
    def run_analysis(self):
        """Run pme_testbed_phase2 analysis on current data"""
        try:
            self.save_current_data()
            
            result = subprocess.run([
                './build/examples/pme_testbed_phase2',
                '/tmp/live_stream_data.json'
            ], capture_output=True, text=True, cwd='/sep')
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines:
                    # Get last signal line
                    for line in reversed(lines):
                        if ',' in line and 'BUY' in line or 'SELL' in line:
                            return line
            return None
        except Exception as e:
            print(f"Analysis error: {e}")
            return None
    
    def check_for_signals(self):
        """Check for trading signals"""        
        if len(self.candles) < 100:  # Need enough data
            print(f"⏳ Need more data for analysis ({len(self.candles)}/100 candles)")
            return
            
        self.last_analysis_time = time.time()
        self.stats['analysis_runs'] += 1
        
        print(f"🔬 ANALYSIS RUN #{self.stats['analysis_runs']} | Analyzing {len(self.candles)} candles...")
        signal_line = self.run_analysis()
        
        if signal_line:
            try:
                parts = signal_line.split(',')
                if len(parts) >= 12:
                    direction = parts[10].strip()
                    confidence = float(parts[11].strip())
                    coherence = float(parts[8].strip()) if len(parts) > 8 else 0
                    
                    self.stats['signals_generated'] += 1
                    self.stats['last_signal'] = direction
                    self.stats['last_confidence'] = confidence
                    self.stats['last_coherence'] = coherence
                    
                    print(f"🎯 SIGNAL #{self.stats['signals_generated']}: {direction} | Conf: {confidence:.3f} | Coh: {coherence:.3f}")
                    
                    # Check thresholds
                    if confidence >= 0.65 and coherence >= 0.30:
                        self.stats['high_confidence_signals'] += 1
                        print(f"🚀 HIGH-CONFIDENCE SIGNAL #{self.stats['high_confidence_signals']}!")
                        print(f"   ✅ Confidence: {confidence:.3f} ≥ 0.65")
                        print(f"   ✅ Coherence: {coherence:.3f} ≥ 0.30")
                        self.execute_trade(direction)
                    else:
                        print("⏳ Signal below threshold:")
                        if confidence < 0.65:
                            print(f"   ❌ Confidence: {confidence:.3f} < 0.65")
                        if coherence < 0.30:
                            print(f"   ❌ Coherence: {coherence:.3f} < 0.30")
            except Exception as e:
                print(f"❌ Signal parsing error: {e}")
        else:
            print("📊 Analysis complete - no signal generated")
    
    def execute_trade(self, direction):
        """Execute the trade"""
        try:
            self.stats['trades_executed'] += 1
            print(f"💰 EXECUTING TRADE #{self.stats['trades_executed']}: {direction} 1000 units")
            
            result = subprocess.run([
                'python3', 'execute_real_trade.py',
                direction, '1000', '20', '40'
            ], cwd='/sep', capture_output=True, text=True)
            
            print(f"Trade output: {result.stdout}")
            if result.stderr:
                print(f"Trade errors: {result.stderr}")
            
            if result.returncode == 0:
                self.stats['trades_successful'] += 1
                print(f"✅ TRADE #{self.stats['trades_executed']} SUCCESSFUL!")
                print(f"   Success Rate: {self.stats['trades_successful']}/{self.stats['trades_executed']} ({100*self.stats['trades_successful']/self.stats['trades_executed']:.1f}%)")
                # Don't wait after trade - keep analyzing
            else:
                self.stats['trades_failed'] += 1
                print(f"❌ TRADE #{self.stats['trades_executed']} FAILED!")
                print(f"   Failure Rate: {self.stats['trades_failed']}/{self.stats['trades_executed']} ({100*self.stats['trades_failed']/self.stats['trades_executed']:.1f}%)")
                
        except Exception as e:
            self.stats['trades_failed'] += 1
            print(f"❌ Trade execution error: {e}")
    
    def run(self):
        """Main trading loop"""
        print("🚀 LIVE STREAM TRADER STARTING")
        print("==============================")
        
        # Bootstrap with historical data
        if not self.bootstrap_historical_data():
            print("❌ Failed to bootstrap")
            return
        
        # Start live streaming
        self.start_live_stream()
        
        print("🎯 Trading system active - watching for signals...")
        print(f"📊 Account: {self.account_id}")
        print("📈 Strategy: 56.22% accuracy pme_testbed_phase2")
        print("⚡ Thresholds: conf≥0.65, coh≥0.30")
        
        try:
            # Keep running and monitoring
            while self.trading_active:
                # Show brief status every 30 seconds when no new candles
                time.sleep(30)
                if self.stats['ticks_received'] > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Ticks: {self.stats['ticks_received']:,} | Price: {self.stats['last_price']:.5f} | Candles: {len(self.candles)} | Waiting for next candle...")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping trader...")
            self.trading_active = False

if __name__ == "__main__":
    trader = LiveStreamTrader()
    trader.run()
