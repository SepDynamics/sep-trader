#!/bin/bash
# Proper Market Analysis - Uses 48-hour data analysis before trading

echo "🚀 PROPER MARKET ANALYSIS SYSTEM"
echo "Analyzing 48 hours of market data with proven 60.73% accuracy thresholds"
echo "=================================================================="

# Source OANDA credentials
if [ -f "OANDA.env" ]; then
    source OANDA.env
    echo "✅ OANDA credentials loaded"
else
    echo "❌ OANDA.env not found. Please configure your demo account."
    exit 1
fi

echo ""
echo "🔬 Starting comprehensive market analysis..."
echo "   • Fetches 48 hours of multi-timeframe data"
echo "   • Uses STRICT thresholds: Conf:0.65, Coh:0.30"
echo "   • Applies proven weight configuration: S:0.4, C:0.1, E:0.5"
echo "   • Only trades on high-confidence signals"
echo ""

# Run the proper analysis
./build/src/dsl/sep_dsl_interpreter trading_patterns/strategies/proper_analysis_trader.sep

echo ""
echo "🎯 Market analysis complete"
