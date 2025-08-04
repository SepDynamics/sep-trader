#!/bin/bash
# Automated Parameter Optimization and Live Trading System
# Ensures 60% accuracy standard for all pairs before trading

echo "🚀 SEP AUTOMATED TRADING SYSTEM"
echo "================================="
echo "• Checks for optimized parameters from last 5 days"
echo "• Runs optimization trials if needed"
echo "• Only goes live with 60%+ accuracy"
echo "• Handles 16+ currency pairs simultaneously"
echo ""

# Source credentials
source OANDA.env

# Configuration
PAIRS_TO_TRADE=(
    "EUR_USD" "GBP_USD" "USD_JPY" "AUD_USD" "USD_CHF" "USD_CAD"
    "NZD_USD" "EUR_GBP" "EUR_JPY" "GBP_JPY" "EUR_AUD" "GBP_AUD"
    "AUD_JPY" "USD_SGD" "EUR_CHF" "GBP_CHF"
)

CONFIG_DIR="/sep/config/optimized_params"
OPTIMIZATION_LOG="/sep/output/optimization_log.txt"
MIN_ACCURACY=60.0
LOOKBACK_DAYS=5

# Create config directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

# Function to check if parameters exist and are recent
check_params_exist() {
    local pair=$1
    local config_file="$CONFIG_DIR/${pair}_optimized.json"
    
    if [[ ! -f "$config_file" ]]; then
        return 1  # File doesn't exist
    fi
    
    # Check if file is from last 5 days
    local file_age=$(( ($(date +%s) - $(stat -c %Y "$config_file")) / 86400 ))
    if [[ $file_age -gt $LOOKBACK_DAYS ]]; then
        echo "  ⚠️  $pair parameters are $file_age days old (> $LOOKBACK_DAYS days)"
        return 1  # Too old
    fi
    
    # Check if accuracy meets minimum threshold
    local accuracy=$(jq -r '.accuracy' "$config_file" 2>/dev/null)
    if [[ -z "$accuracy" || $(echo "$accuracy < $MIN_ACCURACY" | bc -l) -eq 1 ]]; then
        echo "  ❌ $pair accuracy ($accuracy%) below threshold ($MIN_ACCURACY%)"
        return 1  # Below threshold
    fi
    
    echo "  ✅ $pair: ${accuracy}% accuracy (updated $(stat -c %y "$config_file" | cut -d' ' -f1))"
    return 0  # Good to go
}

# Function to run optimization for a single pair
optimize_pair() {
    local pair=$1
    echo "🔧 Optimizing parameters for $pair..."
    
    # Create optimization script based on the proven tune_weights.py and tune_thresholds.py approach
    cat > "/tmp/optimize_${pair}.py" << EOF
#!/usr/bin/env python3
import subprocess
import re
import time
import numpy as np
import json
import os
from datetime import datetime

PME_TESTBED_PATH = "/sep/examples/pme_testbed_phase2.cpp"
OANDA_DATA_PATH = "/tmp/${pair}_optimization_data.json"

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
    with open(PME_TESTBED_PATH, 'r') as f:
        content = f.read()

    # Target the specific lines after "EXPERIMENT 001" around line 581-583
    # Use more specific regex patterns to target the right variables
    content = re.sub(r"double stability_w = [0-9.]+;     // OPTIMIZED: Systematic weight tuning", 
                     f"double stability_w = {stability_w:.2f};     // OPTIMIZED: Systematic weight tuning", content)
    content = re.sub(r"double coherence_w = [0-9.]+;     // OPTIMIZED: Minimal influence discovered", 
                     f"double coherence_w = {coherence_w:.2f};     // OPTIMIZED: Minimal influence discovered", content)
    content = re.sub(r"double entropy_w = [0-9.]+;       // OPTIMIZED: Primary signal driver", 
                     f"double entropy_w = {entropy_w:.2f};       // OPTIMIZED: Primary signal driver", content)

    # Verify changes were made
    if f"stability_w = {stability_w:.2f};" not in content:
        print(f"WARNING: Failed to update stability_w to {stability_w}")
    if f"coherence_w = {coherence_w:.2f};" not in content:
        print(f"WARNING: Failed to update coherence_w to {coherence_w}")
    if f"entropy_w = {entropy_w:.2f};" not in content:
        print(f"WARNING: Failed to update entropy_w to {entropy_w}")

    with open(PME_TESTBED_PATH, 'w') as f:
        f.write(content)

def modify_thresholds(confidence_t: float, coherence_t: float):
    """Modifies the filtering thresholds in the C++ source file."""
    with open(PME_TESTBED_PATH, 'r') as f:
        content = f.read()

    # Target the specific confidence and coherence thresholds around lines 778-779
    content = re.sub(r"double confidence_threshold = [0-9.]+; // OPTIMAL: High-confidence filter", 
                     f"double confidence_threshold = {confidence_t:.2f}; // OPTIMAL: High-confidence filter", content)
    content = re.sub(r"double coherence_threshold = [0-9.]+;  // OPTIMAL: Broad signal capture", 
                     f"double coherence_threshold = {coherence_t:.2f};  // OPTIMAL: Broad signal capture", content)

    # Verify changes were made
    if f"confidence_threshold = {confidence_t:.2f};" not in content:
        print(f"WARNING: Failed to update confidence_threshold to {confidence_t}")
    if f"coherence_threshold = {coherence_t:.2f};" not in content:
        print(f"WARNING: Failed to update coherence_threshold to {coherence_t}")

    with open(PME_TESTBED_PATH, 'w') as f:
        f.write(content)

def run_backtest():
    """Builds and runs the backtest, then parses the results."""
    # Build the system
    build_proc = subprocess.run(['./build.sh'], capture_output=True, text=True, cwd='/sep')
    if build_proc.returncode != 0:
        print("  [ERROR] Build failed. Skipping this configuration.")
        return None

    # Run the test with the pair-specific data
    test_proc = subprocess.run(
        ['./build/examples/pme_testbed_phase2', OANDA_DATA_PATH],
        capture_output=True, text=True, cwd='/sep'
    )
    output = test_proc.stdout + test_proc.stderr

    # Parse results
    try:
        overall_match = re.search(r'Overall Accuracy: ([\d.]+)%', output)
        high_conf_match = re.search(r'High Confidence Accuracy: ([\d.]+)%', output)
        rate_match = re.search(r'High Confidence Signals: \\d+ \\(([\\d.]+)%\\)', output)
        
        overall = float(overall_match.group(1)) if overall_match else 0.0
        high_conf = float(high_conf_match.group(1)) if high_conf_match else 0.0
        high_conf_rate = float(rate_match.group(1)) if rate_match else 0.0
        
        return {"overall": overall, "high_conf": high_conf, "rate": high_conf_rate}
    except (AttributeError, IndexError, ValueError):
        print("  [ERROR] Could not parse output. Output was:")
        print(output[-500:])  # Last 500 chars for debugging
        return None

def optimize_pair_parameters(pair):
    """Find optimal parameters for a currency pair using systematic grid search"""
    
    # First fetch the data
    if not fetch_last_5_days_data(pair):
        return None
    
    print(f"Starting systematic optimization for {pair}...")
    
    # Define parameter grid - start with smaller grid for faster debugging
    weight_steps = np.arange(0.1, 0.7, 0.2)  # 0.1, 0.3, 0.5 (faster testing)
    conf_thresholds = np.arange(0.50, 0.70, 0.10)  # 0.50, 0.60 (faster testing)
    coh_thresholds = np.arange(0.30, 0.60, 0.15)   # 0.30, 0.45 (faster testing)
    
    best_score = -1
    best_config = None
    results_log = []

    # Save original content
    with open(PME_TESTBED_PATH, 'r') as f:
        original_content = f.read()

    try:
        # Weight optimization phase
        print(f"Phase 1: Weight Optimization")
        weight_test_count = 0
        total_weight_tests = sum(1 for s_w in weight_steps for c_w in weight_steps 
                                if 1.0 - s_w - c_w >= 0.05)
        
        for s_w in weight_steps:
            for c_w in weight_steps:
                e_w = 1.0 - s_w - c_w
                if e_w >= 0.05:  # Ensure entropy has at least some weight
                    weight_test_count += 1
                    s_w, c_w, e_w = round(s_w, 2), round(c_w, 2), round(e_w, 2)
                    print(f"Testing weights {weight_test_count}/{total_weight_tests}: S:{s_w} C:{c_w} E:{e_w}")
                    
                    print(f"  Modifying weights in source file...")
                    modify_scoring_weights(s_w, c_w, e_w)
                    
                    # Verify the modifications took effect
                    with open(PME_TESTBED_PATH, 'r') as f:
                        check_content = f.read()
                    if f"stability_w = {s_w:.2f};" in check_content:
                        print(f"  ✅ Weight modifications applied successfully")
                    else:
                        print(f"  ❌ Weight modifications FAILED!")
                        continue
                    
                    # Test with default thresholds first
                    modify_thresholds(0.65, 0.30)
                    
                    metrics = run_backtest()
                    if metrics:
                        score = (metrics['high_conf'] * 0.7) + (metrics['overall'] * 0.2) + (metrics['rate'] * 0.1)
                        print(f"  Results: Overall:{metrics['overall']:.1f}% High-Conf:{metrics['high_conf']:.1f}% Rate:{metrics['rate']:.1f}% Score:{score:.2f}")
                        
                        if score > best_score:
                            best_score = score
                            best_config = {
                                "stability_weight": s_w,
                                "coherence_weight": c_w,
                                "entropy_weight": e_w,
                                "confidence_threshold": 0.65,
                                "coherence_threshold": 0.30,
                                "accuracy": metrics['high_conf'],
                                "overall_accuracy": metrics['overall'],
                                "signal_rate": metrics['rate'],
                                "score": score,
                                "pair": pair,
                                "optimized_date": datetime.now().isoformat()
                            }
                            print(f"    🎯 NEW BEST SCORE!")
        
        # If we found good weights, optimize thresholds
        if best_config and best_config['accuracy'] > 40.0:
            print(f"\\nPhase 2: Threshold Optimization with best weights S:{best_config['stability_weight']} C:{best_config['coherence_weight']} E:{best_config['entropy_weight']}")
            
            # Set the best weights
            modify_scoring_weights(best_config['stability_weight'], best_config['coherence_weight'], best_config['entropy_weight'])
            
            threshold_test_count = 0
            total_threshold_tests = len(conf_thresholds) * len(coh_thresholds)
            
            for conf_t in conf_thresholds:
                for coh_t in coh_thresholds:
                    threshold_test_count += 1
                    conf_t, coh_t = round(conf_t, 2), round(coh_t, 2)
                    
                    print(f"Testing thresholds {threshold_test_count}/{total_threshold_tests}: Conf:{conf_t} Coh:{coh_t}")
                    
                    modify_thresholds(conf_t, coh_t)
                    metrics = run_backtest()
                    
                    if metrics and metrics['rate'] > 0:
                        # Profitability score like tune_thresholds.py
                        profitability_score = (metrics['high_conf'] - 50) * metrics['rate']
                        print(f"  Results: High-Conf:{metrics['high_conf']:.1f}% Rate:{metrics['rate']:.1f}% Profit Score:{profitability_score:.2f}")
                        
                        if metrics['high_conf'] > best_config['accuracy']:
                            best_config.update({
                                "confidence_threshold": conf_t,
                                "coherence_threshold": coh_t,
                                "accuracy": metrics['high_conf'],
                                "overall_accuracy": metrics['overall'],
                                "signal_rate": metrics['rate'],
                                "profitability_score": profitability_score
                            })
                            print(f"    🏆 IMPROVED ACCURACY!")

    finally:
        # Restore original file
        with open(PME_TESTBED_PATH, 'w') as f:
            f.write(original_content)
        print("✅ Restored original source file.")

    return best_config

if __name__ == "__main__":
    pair = "$pair"
    result = optimize_pair_parameters(pair)
    
    if result and result["accuracy"] >= $MIN_ACCURACY:
        # Save optimized parameters
        config_file = "/sep/config/optimized_params/${pair}_optimized.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"✅ {pair}: {result['accuracy']:.2f}% - Parameters saved to {config_file}")
        exit(0)
    else:
        accuracy = result["accuracy"] if result else 0.0
        print(f"❌ {pair}: {accuracy:.2f}% - Below threshold")
        exit(1)
EOF

    # Run optimization
    python3 "/tmp/optimize_${pair}.py" 2>&1 | tee -a "$OPTIMIZATION_LOG"
    return $?
}

# Function to start live trading with optimized parameters
start_live_trading() {
    echo "🚀 Starting live trading with optimized parameters..."
    
    # Create master trading configuration
    cat > "/sep/config/live_trading_config.json" << EOF
{
    "trading_pairs": [
EOF

    local first=true
    for pair in "${PAIRS_TO_TRADE[@]}"; do
        local config_file="$CONFIG_DIR/${pair}_optimized.json"
        if [[ -f "$config_file" ]]; then
            if [[ "$first" == true ]]; then
                first=false
            else
                echo "        ," >> "/sep/config/live_trading_config.json"
            fi
            echo "        {" >> "/sep/config/live_trading_config.json"
            echo "            \"pair\": \"$pair\"," >> "/sep/config/live_trading_config.json"
            cat "$config_file" | jq -r 'to_entries | .[] | "            \"" + .key + "\": " + (.value | tostring)' | sed '$s/$//' >> "/sep/config/live_trading_config.json"
            echo "        }" >> "/sep/config/live_trading_config.json"
        fi
    done

    cat >> "/sep/config/live_trading_config.json" << EOF
    ],
    "system_config": {
        "max_concurrent_trades": 16,
        "risk_per_trade": 1.0,
        "max_daily_risk": 10.0,
        "min_accuracy_threshold": $MIN_ACCURACY
    }
}
EOF

    # Start the live trading system
    echo "🎯 Launching multi-pair live trading system..."
    ./live_trader.sh
}

# Main execution flow
echo "📊 Checking parameter status for all pairs..."

missing_pairs=()
optimization_needed=false

for pair in "${PAIRS_TO_TRADE[@]}"; do
    echo "Checking $pair..."
    if ! check_params_exist "$pair"; then
        missing_pairs+=("$pair")
        optimization_needed=true
    fi
done

if [[ "$optimization_needed" == true ]]; then
    echo ""
    echo "⚠️  Optimization needed for ${#missing_pairs[@]} pairs: ${missing_pairs[*]}"
    echo "🔧 Starting parameter optimization (this may take several hours)..."
    echo ""
    
    # Clear previous optimization log
    echo "Optimization started: $(date)" > "$OPTIMIZATION_LOG"
    
    # Run optimizations in parallel (but limit to 4 concurrent to avoid overload)
    max_concurrent=4
    running_jobs=0
    
    for pair in "${missing_pairs[@]}"; do
        # Wait if we have too many running jobs
        while [[ $running_jobs -ge $max_concurrent ]]; do
            wait -n  # Wait for any job to complete
            running_jobs=$((running_jobs - 1))
        done
        
        # Start optimization in background
        optimize_pair "$pair" &
        running_jobs=$((running_jobs + 1))
        
        echo "Started optimization for $pair (job $running_jobs/$max_concurrent)"
        sleep 5  # Stagger starts to avoid system overload
    done
    
    # Wait for all optimizations to complete
    echo "⏳ Waiting for all optimizations to complete..."
    wait
    
    echo ""
    echo "✅ Optimization phase complete!"
    echo "📋 Final parameter check..."
    
    # Re-check all pairs
    ready_pairs=()
    failed_pairs=()
    
    for pair in "${PAIRS_TO_TRADE[@]}"; do
        if check_params_exist "$pair"; then
            ready_pairs+=("$pair")
        else
            failed_pairs+=("$pair")
        fi
    done
    
    echo ""
    echo "📈 Ready for trading: ${#ready_pairs[@]} pairs"
    echo "❌ Failed optimization: ${#failed_pairs[@]} pairs"
    
    if [[ ${#failed_pairs[@]} -gt 0 ]]; then
        echo "⚠️  Failed pairs: ${failed_pairs[*]}"
        echo "💡 These pairs will be excluded from live trading"
    fi
    
    if [[ ${#ready_pairs[@]} -eq 0 ]]; then
        echo "❌ No pairs ready for trading! Check optimization log: $OPTIMIZATION_LOG"
        exit 1
    fi
    
else
    echo ""
    echo "✅ All ${#PAIRS_TO_TRADE[@]} pairs have current optimized parameters!"
    echo "🎯 Proceeding directly to live trading..."
fi

echo ""
echo "🚀 LAUNCHING LIVE TRADING SYSTEM"
echo "================================"
echo "• Trading pairs: ${#PAIRS_TO_TRADE[@]}"
echo "• Minimum accuracy: $MIN_ACCURACY%"
echo "• Parameter age: < $LOOKBACK_DAYS days"
echo "• Configuration: /sep/config/live_trading_config.json"
echo ""

start_live_trading
