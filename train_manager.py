#!/usr/bin/env python3
"""
Professional Training Manager for SEP Trader-Bot System
Unified interface for managing all currency pair training and status.

Usage:
    python train_manager.py status                    # Show all pair status
    python train_manager.py train EUR_USD            # Train specific pair
    python train_manager.py train-all --quick        # Train all pairs quickly
    python train_manager.py enable EUR_USD           # Enable pair for trading
    python train_manager.py disable EUR_USD          # Disable pair from trading
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Professional trader-bot configuration
PAIR_REGISTRY = "/sep/config/pair_registry.json"
TRAINING_CONFIG = "/sep/config/training_config.json"
STATE_STORE = "/sep/config/pair_states.json"

# Major currency pairs for professional trading
SUPPORTED_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CHF", "USD_CAD",
    "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_CHF",
    "GBP_CHF", "CHF_JPY", "EUR_AUD", "GBP_AUD", "USD_SGD", "EUR_SEK"
]

class PairState:
    """Professional pair state management"""
    UNTRAINED = "untrained"
    TRAINING = "training" 
    READY = "ready"
    TRADING = "trading"
    FAILED = "failed"
    DISABLED = "disabled"

class TrainingManager:
    """Professional training manager for SEP trader-bot system"""
    
    def __init__(self):
        self.ensure_directories()
        self.pair_states = self.load_pair_states()
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        os.makedirs("/sep/config", exist_ok=True)
        os.makedirs("/sep/cache", exist_ok=True)
        
    def load_pair_states(self) -> Dict:
        """Load persistent pair states"""
        if os.path.exists(STATE_STORE):
            with open(STATE_STORE, 'r') as f:
                return json.load(f)
        return {pair: {"state": PairState.UNTRAINED, "last_trained": None, "accuracy": None} 
                for pair in SUPPORTED_PAIRS}
    
    def save_pair_states(self):
        """Save persistent pair states"""
        with open(STATE_STORE, 'w') as f:
            json.dump(self.pair_states, f, indent=2)
    
    def get_pair_status(self, pair: str) -> Dict:
        """Get comprehensive status for a trading pair"""
        if pair not in self.pair_states:
            return {"state": PairState.UNTRAINED, "error": "Pair not supported"}
            
        state = self.pair_states[pair]
        status = {
            "pair": pair,
            "state": state["state"],
            "last_trained": state.get("last_trained"),
            "accuracy": state.get("accuracy"),
            "cache_valid": self.check_cache_valid(pair),
            "ready_to_trade": False
        }
        
        # Check if ready to trade
        if (state["state"] == PairState.READY and 
            status["cache_valid"] and 
            state.get("accuracy", 0) >= 60.0):
            status["ready_to_trade"] = True
            
        return status
    
    def check_cache_valid(self, pair: str) -> bool:
        """Check if pair has valid cache for last week"""
        cache_file = f"/sep/cache/{pair}_weekly.json"
        if not os.path.exists(cache_file):
            return False
            
        # Check if cache is from last 7 days
        cache_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(cache_file))
        return cache_age.days <= 7
    
    def show_status(self):
        """Show status of all trading pairs"""
        print("🤖 SEP Professional Trader-Bot System")
        print("="*50)
        print(f"{'Pair':<10} {'State':<10} {'Accuracy':<10} {'Cache':<8} {'Ready':<8}")
        print("-"*50)
        
        ready_count = 0
        training_count = 0
        failed_count = 0
        
        for pair in SUPPORTED_PAIRS:
            status = self.get_pair_status(pair)
            state = status["state"]
            accuracy = f"{status['accuracy']:.1f}%" if status["accuracy"] else "N/A"
            cache = "✅" if status["cache_valid"] else "❌"
            ready = "✅" if status["ready_to_trade"] else "❌"
            
            # Color coding for state
            state_display = state
            if state == PairState.READY:
                state_display = f"🟢 {state}"
                ready_count += 1
            elif state == PairState.TRAINING:
                state_display = f"🟡 {state}"
                training_count += 1
            elif state == PairState.FAILED:
                state_display = f"🔴 {state}"
                failed_count += 1
            elif state == PairState.DISABLED:
                state_display = f"⚫ {state}"
            else:
                state_display = f"⚪ {state}"
                
            print(f"{pair:<10} {state_display:<15} {accuracy:<10} {cache:<8} {ready:<8}")
        
        print("-"*50)
        print(f"Summary: {ready_count} ready, {training_count} training, {failed_count} failed")
        
        if ready_count > 0:
            print(f"\n✅ {ready_count} pairs ready for live trading")
        if failed_count > 0:
            print(f"❌ {failed_count} pairs need retraining")
    
    def train_pair(self, pair: str, quick: bool = False) -> bool:
        """Train a specific currency pair"""
        if pair not in SUPPORTED_PAIRS:
            print(f"❌ Error: {pair} not supported")
            return False
        
        print(f"🔧 Training {pair}...")
        
        # Update state to training
        self.pair_states[pair]["state"] = PairState.TRAINING
        self.save_pair_states()
        
        try:
            # Call the existing train_currency_pair.py script
            cmd = ["python", "train_currency_pair.py", pair]
            if quick:
                cmd.append("--quick")
                
            result = subprocess.run(cmd, capture_output=True, text=True, cwd="/sep")
            
            if result.returncode == 0:
                # Parse training results
                accuracy = self.parse_training_accuracy(result.stdout)
                
                if accuracy >= 60.0:
                    self.pair_states[pair].update({
                        "state": PairState.READY,
                        "last_trained": datetime.now().isoformat(),
                        "accuracy": accuracy
                    })
                    print(f"✅ {pair}: Training successful - {accuracy:.1f}% accuracy")
                    success = True
                else:
                    self.pair_states[pair]["state"] = PairState.FAILED
                    print(f"❌ {pair}: Training failed - {accuracy:.1f}% accuracy (below 60%)")
                    success = False
            else:
                self.pair_states[pair]["state"] = PairState.FAILED
                print(f"❌ {pair}: Training failed - {result.stderr}")
                success = False
                
        except Exception as e:
            self.pair_states[pair]["state"] = PairState.FAILED
            print(f"❌ {pair}: Training error - {str(e)}")
            success = False
        
        self.save_pair_states()
        return success
    
    def parse_training_accuracy(self, output: str) -> float:
        """Parse accuracy from training output"""
        import re
        match = re.search(r'High-Confidence Accuracy: ([\d.]+)%', output)
        if match:
            return float(match.group(1))
        return 0.0
    
    def train_all_pairs(self, quick: bool = False):
        """Train all pairs that need training"""
        untrained_pairs = [pair for pair in SUPPORTED_PAIRS 
                          if self.pair_states[pair]["state"] in [PairState.UNTRAINED, PairState.FAILED]]
        
        if not untrained_pairs:
            print("✅ All pairs are already trained")
            return
        
        print(f"🔧 Training {len(untrained_pairs)} pairs...")
        
        for i, pair in enumerate(untrained_pairs, 1):
            print(f"\n[{i}/{len(untrained_pairs)}] Training {pair}")
            self.train_pair(pair, quick)
    
    def enable_pair(self, pair: str) -> bool:
        """Enable pair for trading"""
        if pair not in SUPPORTED_PAIRS:
            print(f"❌ Error: {pair} not supported")
            return False
        
        status = self.get_pair_status(pair)
        if not status["ready_to_trade"]:
            print(f"❌ Error: {pair} not ready for trading (state: {status['state']}, accuracy: {status['accuracy']})")
            return False
        
        self.pair_states[pair]["state"] = PairState.TRADING
        self.save_pair_states()
        print(f"✅ {pair} enabled for trading")
        return True
    
    def disable_pair(self, pair: str) -> bool:
        """Disable pair from trading"""
        if pair not in SUPPORTED_PAIRS:
            print(f"❌ Error: {pair} not supported")
            return False
        
        self.pair_states[pair]["state"] = PairState.DISABLED
        self.save_pair_states()
        print(f"⚫ {pair} disabled from trading")
        return True

def main():
    parser = argparse.ArgumentParser(description='SEP Professional Training Manager')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show all pair status')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train specific pair')
    train_parser.add_argument('pair', help='Currency pair to train')
    train_parser.add_argument('--quick', action='store_true', help='Quick training mode')
    
    # Train all command
    train_all_parser = subparsers.add_parser('train-all', help='Train all pairs')
    train_all_parser.add_argument('--quick', action='store_true', help='Quick training mode')
    
    # Enable command
    enable_parser = subparsers.add_parser('enable', help='Enable pair for trading')
    enable_parser.add_argument('pair', help='Currency pair to enable')
    
    # Disable command
    disable_parser = subparsers.add_parser('disable', help='Disable pair from trading')
    disable_parser.add_argument('pair', help='Currency pair to disable')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    manager = TrainingManager()
    
    if args.command == 'status':
        manager.show_status()
    elif args.command == 'train':
        success = manager.train_pair(args.pair, args.quick)
        return 0 if success else 1
    elif args.command == 'train-all':
        manager.train_all_pairs(args.quick)
    elif args.command == 'enable':
        success = manager.enable_pair(args.pair)
        return 0 if success else 1
    elif args.command == 'disable':
        success = manager.disable_pair(args.pair)
        return 0 if success else 1
    
    return 0

if __name__ == "__main__":
    exit(main())
