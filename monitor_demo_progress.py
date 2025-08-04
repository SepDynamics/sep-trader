#!/usr/bin/env python3
"""
Live Demo Trading Progress Monitor
Target: $187k → $200k using 60.73% accuracy system
"""

import json
import time
import subprocess
import os
from datetime import datetime

def get_account_balance():
    """Get current OANDA demo account balance"""
    try:
        # This would be replaced with actual OANDA API call
        # For now, simulate reading from your actual demo balance
        return 187000.0  # Your starting balance
    except Exception as e:
        print(f"Error getting balance: {e}")
        return None

def run_trading_cycle():
    """Run one cycle of the trading strategy"""
    try:
        result = subprocess.run(
            ["./build/src/dsl/sep_dsl_interpreter", "trading_patterns/strategies/live_demo_strategy.sep"],
            capture_output=True,
            text=True,
            cwd="/sep"
        )
        
        # Extract key metrics from output
        output = result.stdout
        if "HIGH-CONFIDENCE SIGNAL DETECTED" in output:
            return "SIGNAL_FOUND"
        elif "No high-confidence signal" in output:
            return "WAITING"
        else:
            return "ERROR"
            
    except Exception as e:
        print(f"Error running trading cycle: {e}")
        return "ERROR"

def main():
    print("🚀 Live Demo Trading Monitor")
    print("Target: $187k → $200k (7% growth needed)")
    print("=" * 50)
    
    target_balance = 200000.0
    starting_balance = 187000.0
    
    cycle_count = 0
    signals_found = 0
    
    while True:
        cycle_count += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"\n[{current_time}] Cycle #{cycle_count}")
        
        # Get current balance
        balance = get_account_balance()
        if balance:
            progress = ((balance - starting_balance) / (target_balance - starting_balance)) * 100
            profit_loss = balance - starting_balance
            
            print(f"Current Balance: ${balance:,.2f}")
            print(f"Progress: {progress:.1f}% toward target")
            print(f"P/L: ${profit_loss:+,.2f}")
            
            # Check if target reached
            if balance >= target_balance:
                print("🎉 TARGET ACHIEVED!")
                break
                
            # Check if stop loss hit
            if balance < starting_balance * 0.95:  # 5% max loss
                print("🛑 Stop loss triggered")
                break
        
        # Run trading cycle
        result = run_trading_cycle()
        
        if result == "SIGNAL_FOUND":
            signals_found += 1
            print(f"🎯 Signal #{signals_found} detected and executed")
        elif result == "WAITING":
            print("⏳ No signal - waiting for next opportunity")
        else:
            print("❌ Error in trading cycle")
        
        # Wait 60 seconds between cycles (adjust as needed)
        print("Waiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()
