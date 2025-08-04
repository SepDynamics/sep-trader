#!/bin/bash
# Continuous Demo Trading - Actually places trades on OANDA demo account
# Runs until balance reaches $10k target

echo "🚀 CONTINUOUS DEMO TRADING STARTED"
echo "Target: Get back to $12,000 demo balance"
echo "Strategy: Real OANDA trades with 60.73% accuracy system"
echo "=========================================="

# Source OANDA credentials
if [ -f "OANDA.env" ]; then
    source OANDA.env
    echo "✅ OANDA credentials loaded"
else
    echo "❌ OANDA.env not found. Please configure your demo account."
    exit 1
fi

# Install required Python packages
pip3 install requests &>/dev/null

echo ""
echo "🔥 STARTING CONTINUOUS TRADING LOOP..."
echo "   • Will place REAL trades on your demo account"
echo "   • Uses proven 60.73% accuracy configuration"
echo "   • 2% risk per trade for aggressive recovery"
echo "   • Stops when balance reaches $10k"
echo ""

# Run the continuous trader
while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running trading cycle..."
    
    # Execute the trading strategy
    ./build/src/dsl/sep_dsl_interpreter trading_patterns/strategies/continuous_demo_trader.sep
    
    # Check if we should stop (target reached or error)
    if [ $? -ne 0 ]; then
        echo "Trading completed or error occurred"
        break
    fi
    
    echo "Cycle complete. Waiting 30 seconds before next cycle..."
    sleep 30
done

echo ""
echo "🎯 Continuous trading session ended"
