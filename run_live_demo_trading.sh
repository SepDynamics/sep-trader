#!/bin/bash
# Live Demo Trading Strategy Runner
# Target: $187k → $200k using 60.73% accuracy system

echo "🚀 Starting Live Demo Trading Strategy"
echo "Target: $187k → $200k (7% growth needed)"
echo "Using proven 60.73% accuracy configuration"

# Source OANDA credentials
if [ -f "OANDA.env" ]; then
    source OANDA.env
    echo "✅ OANDA credentials loaded"
else
    echo "❌ OANDA.env not found. Please configure your demo account credentials."
    exit 1
fi

# Verify build
if [ ! -f "build/src/dsl/sep_dsl_interpreter" ]; then
    echo "❌ DSL interpreter not found. Running build..."
    ./build.sh
fi

echo ""
echo "📊 Running Live Demo Trading Strategy..."
echo "Risk Management: 1% per trade, max 3 concurrent positions"
echo "Configuration: S:0.4, C:0.1, E:0.5 | Conf:0.65, Coh:0.30"
echo ""

# Run the live trading strategy
./build/src/dsl/sep_dsl_interpreter trading_patterns/strategies/live_demo_strategy.sep

echo ""
echo "🎯 Demo trading session complete"
