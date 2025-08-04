#!/bin/bash
# Test optimization for a single pair to debug

echo "🧪 Testing optimization for EUR_USD only..."

source OANDA.env

# Create test optimization script
cat > "/tmp/test_optimize_EUR_USD.py" << 'EOF'
#!/usr/bin/env python3
import subprocess
import re
import time
import numpy as np
import json
import os
from datetime import datetime

PME_TESTBED_PATH = "/sep/examples/pme_testbed_phase2.cpp"
OANDA_DATA_PATH = "/tmp/EUR_USD_optimization_data.json"

def fetch_last_5_days_data(pair):
    """Fetch 5 days of historical data for optimization"""
    print(f"Fetching 5 days of data for {pair}...")
    cmd = f"cd /sep && source ./OANDA.env && ./build/examples/oanda_historical_fetcher --instrument {pair} --granularity M1 --hours 120 --output {OANDA_DATA_PATH}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0 and os.path.exists(OANDA_DATA_PATH):
        print(f"Successfully fetched data for {pair}")
        return True
    else:
        print(f"Failed to fetch data for {pair}: {result.stderr}")
        return False

def modify_scoring_weights(stability_w: float, coherence_w: float, entropy_w: float):
    """Modifies the scoring weights in the C++ source file."""
    print(f"Modifying weights: S={stability_w}, C={coherence_w}, E={entropy_w}")
    
    with open(PME_TESTBED_PATH, 'r') as f:
        content = f.read()

    # Show original values
    orig_stability = re.search(r"double stability_w = ([0-9.]+);     // OPTIMIZED", content)
    orig_coherence = re.search(r"double coherence_w = ([0-9.]+);     // OPTIMIZED", content) 
    orig_entropy = re.search(r"double entropy_w = ([0-9.]+);       // OPTIMIZED", content)
    
    if orig_stability:
        print(f"  Original stability_w: {orig_stability.group(1)}")
    if orig_coherence:
        print(f"  Original coherence_w: {orig_coherence.group(1)}")
    if orig_entropy:
        print(f"  Original entropy_w: {orig_entropy.group(1)}")

    # Target the specific lines after "EXPERIMENT 001" around line 581-583
    content = re.sub(r"double stability_w = [0-9.]+;     // OPTIMIZED: Systematic weight tuning", 
                     f"double stability_w = {stability_w:.2f};     // OPTIMIZED: Systematic weight tuning", content)
    content = re.sub(r"double coherence_w = [0-9.]+;     // OPTIMIZED: Minimal influence discovered", 
                     f"double coherence_w = {coherence_w:.2f};     // OPTIMIZED: Minimal influence discovered", content)
    content = re.sub(r"double entropy_w = [0-9.]+;       // OPTIMIZED: Primary signal driver", 
                     f"double entropy_w = {entropy_w:.2f};       // OPTIMIZED: Primary signal driver", content)

    with open(PME_TESTBED_PATH, 'w') as f:
        f.write(content)
    
    # Verify changes
    with open(PME_TESTBED_PATH, 'r') as f:
        new_content = f.read()
    
    new_stability = re.search(r"double stability_w = ([0-9.]+);     // OPTIMIZED", new_content)
    new_coherence = re.search(r"double coherence_w = ([0-9.]+);     // OPTIMIZED", new_content)
    new_entropy = re.search(r"double entropy_w = ([0-9.]+);       // OPTIMIZED", new_content)
    
    if new_stability:
        print(f"  New stability_w: {new_stability.group(1)}")
    if new_coherence:
        print(f"  New coherence_w: {new_coherence.group(1)}")
    if new_entropy:
        print(f"  New entropy_w: {new_entropy.group(1)}")

def run_backtest():
    """Builds and runs the backtest, then parses the results."""
    print("  Building project...")
    build_proc = subprocess.run(['./build.sh'], capture_output=True, text=True, cwd='/sep')
    if build_proc.returncode != 0:
        print("  [ERROR] Build failed.")
        print(build_proc.stderr[-500:])
        return None

    print("  Running backtest...")
    test_proc = subprocess.run(
        ['./build/examples/pme_testbed_phase2', OANDA_DATA_PATH],
        capture_output=True, text=True, cwd='/sep'
    )
    output = test_proc.stdout + test_proc.stderr

    # Parse results
    try:
        overall_match = re.search(r'Overall Accuracy: ([\d.]+)%', output)
        high_conf_match = re.search(r'High Confidence Accuracy: ([\d.]+)%', output)
        rate_match = re.search(r'High Confidence Signals: \d+ \(([\\d.]+)%\)', output)
        
        overall = float(overall_match.group(1)) if overall_match else 0.0
        high_conf = float(high_conf_match.group(1)) if high_conf_match else 0.0
        high_conf_rate = float(rate_match.group(1)) if rate_match else 0.0
        
        return {"overall": overall, "high_conf": high_conf, "rate": high_conf_rate}
    except (AttributeError, IndexError, ValueError):
        print("  [ERROR] Could not parse output. Last 1000 chars:")
        print(output[-1000:])
        return None

if __name__ == "__main__":
    pair = "EUR_USD"
    
    # Save original content
    with open(PME_TESTBED_PATH, 'r') as f:
        original_content = f.read()
    
    try:
        # First fetch data
        if not fetch_last_5_days_data(pair):
            exit(1)
        
        print("Testing 3 different weight combinations...")
        
        # Test 1: Original weights
        print("\n=== TEST 1: Original Weights ===")
        metrics1 = run_backtest()
        if metrics1:
            print(f"Results: Overall={metrics1['overall']:.1f}% High-Conf={metrics1['high_conf']:.1f}% Rate={metrics1['rate']:.1f}%")
        
        # Test 2: Modified weights
        print("\n=== TEST 2: Modified Weights (S=0.5, C=0.1, E=0.4) ===")
        modify_scoring_weights(0.5, 0.1, 0.4)
        metrics2 = run_backtest()
        if metrics2:
            print(f"Results: Overall={metrics2['overall']:.1f}% High-Conf={metrics2['high_conf']:.1f}% Rate={metrics2['rate']:.1f}%")
        
        # Test 3: Another set of weights
        print("\n=== TEST 3: Different Weights (S=0.3, C=0.2, E=0.5) ===")
        modify_scoring_weights(0.3, 0.2, 0.5)
        metrics3 = run_backtest()
        if metrics3:
            print(f"Results: Overall={metrics3['overall']:.1f}% High-Conf={metrics3['high_conf']:.1f}% Rate={metrics3['rate']:.1f}%")
        
        print("\n=== COMPARISON ===")
        if metrics1: print(f"Test 1: {metrics1['high_conf']:.1f}%")
        if metrics2: print(f"Test 2: {metrics2['high_conf']:.1f}%")  
        if metrics3: print(f"Test 3: {metrics3['high_conf']:.1f}%")
        
        if metrics1 and metrics2 and metrics3:
            if metrics1['high_conf'] == metrics2['high_conf'] == metrics3['high_conf']:
                print("❌ All results identical - weight modifications not working!")
            else:
                print("✅ Weight modifications are working - results vary!")
    
    finally:
        # Restore original file
        with open(PME_TESTBED_PATH, 'w') as f:
            f.write(original_content)
        print("✅ Restored original source file.")
EOF

python3 "/tmp/test_optimize_EUR_USD.py"
