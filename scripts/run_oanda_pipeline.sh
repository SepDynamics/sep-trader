#!/bin/bash
# OANDA → Valkey Data Pipeline Launcher
# ====================================

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting OANDA → Valkey Pipeline"
echo "=================================="

# Check if Python dependencies are installed
echo "📦 Checking Python dependencies..."
cd "$PROJECT_ROOT"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "📦 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q aioredis aiohttp websockets python-dotenv

# Check if OANDA.env exists
if [ ! -f "$PROJECT_ROOT/OANDA.env" ]; then
    echo "❌ OANDA.env file not found!"
    echo "   Please create OANDA.env with your API credentials:"
    echo "   OANDA_API_KEY=your_api_key"
    echo "   OANDA_ACCOUNT_ID=your_account_id"
    echo "   OANDA_ENVIRONMENT=practice"
    exit 1
fi

# Load OANDA environment
source "$PROJECT_ROOT/OANDA.env"

# Verify OANDA credentials
if [ -z "$OANDA_API_KEY" ] || [ -z "$OANDA_ACCOUNT_ID" ]; then
    echo "❌ OANDA credentials not set!"
    echo "   Please check your OANDA.env file"
    exit 1
fi

echo "✅ OANDA credentials loaded"

# Check if Valkey/Redis is running
echo "🔍 Checking Valkey/Redis connection..."
if ! command -v redis-cli &> /dev/null; then
    echo "⚠️ redis-cli not found, assuming Redis is available"
elif ! redis-cli ping &> /dev/null; then
    echo "❌ Cannot connect to Redis/Valkey!"
    echo "   Please start Redis/Valkey server first:"
    echo "   sudo systemctl start redis"
    echo "   or"
    echo "   redis-server"
    exit 1
else
    echo "✅ Redis/Valkey connection verified"
fi

# Check currency pair configuration
if [ ! -f "$PROJECT_ROOT/config/pair_registry.json" ]; then
    echo "⚠️ pair_registry.json not found, using defaults"
else
    ENABLED_PAIRS=$(jq -r '.enabled_pairs | length' "$PROJECT_ROOT/config/pair_registry.json" 2>/dev/null || echo "0")
    echo "✅ Found $ENABLED_PAIRS enabled currency pairs"
fi

# Run the pipeline
echo ""
echo "🌊 Launching OANDA → Valkey Data Pipeline..."
echo "   - Fetching historical data (24 hours)"
echo "   - Starting real-time streaming"
echo "   - WebSocket server on port 8765"
echo "   - Press Ctrl+C to stop"
echo ""

cd "$SCRIPT_DIR"
python3 oanda_valkey_pipeline.py

echo "Pipeline stopped."