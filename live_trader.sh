#!/bin/bash
# Live Stream Trading System - Uses real-time OANDA streaming data

echo "🌊 LIVE STREAM TRADING SYSTEM"
echo "============================="
echo "• Bootstraps with 24h historical data"
echo "• Switches to real-time streaming"
echo "• Uses your proven pme_testbed_phase2 engine"
echo "• Places REAL trades on signals"
echo ""

# Source credentials
source OANDA.env

# Install dependencies
pip3 install requests &>/dev/null

# Run the live stream trader
python3 live_stream_trader.py
