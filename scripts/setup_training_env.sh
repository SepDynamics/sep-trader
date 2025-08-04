#!/bin/bash
# SEP Training Environment Setup
# Sets up the local training coordinator with remote sync capabilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Setting up SEP Training Environment"
echo "======================================"

# Check prerequisites
echo "🔧 Checking prerequisites..."

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA not found. Please install CUDA Toolkit 12.x"
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
echo "✅ CUDA $CUDA_VERSION detected"

# Check for Python dependencies
if ! python3 -c "import requests, pandas, numpy" &> /dev/null; then
    echo "📦 Installing Python dependencies..."
    pip3 install requests pandas numpy tqdm colorama
fi

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p "$SEP_ROOT/cache/weekly_data"
mkdir -p "$SEP_ROOT/cache/backups"
mkdir -p "$SEP_ROOT/logs/training"
mkdir -p "$SEP_ROOT/config"

# Set up training configuration
echo "⚙️  Setting up training configuration..."

# Create OANDA configuration if it doesn't exist
if [ ! -f "$SEP_ROOT/config/OANDA.env" ]; then
    echo "📝 Creating OANDA configuration template..."
    cat > "$SEP_ROOT/config/OANDA.env" << 'EOF'
# OANDA API Configuration for Training Data Fetching
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
OANDA_ENVIRONMENT=practice
OANDA_BASE_URL=https://api-fxpractice.oanda.com
EOF
    echo "⚠️  Please edit $SEP_ROOT/config/OANDA.env with your OANDA credentials"
fi

# Create remote auth token file
if [ ! -f "$SEP_ROOT/config/remote_auth.token" ]; then
    echo "🔐 Creating remote authentication token..."
    openssl rand -hex 32 > "$SEP_ROOT/config/remote_auth.token"
    chmod 600 "$SEP_ROOT/config/remote_auth.token"
fi

# Set up Tailscale aliases for easy remote management
echo "🌐 Setting up Tailscale aliases..."
cat > "$SEP_ROOT/scripts/training_aliases.sh" << 'EOF'
#!/bin/bash
# SEP Training Coordinator Aliases

# Local training operations
alias sep-train='cd /sep && ./build/src/training/sep_training_cli'
alias sep-train-status='sep-train status'
alias sep-train-all='sep-train train-all'
alias sep-train-quick='sep-train train-all --quick'

# Remote sync operations  
alias sep-sync-patterns='sep-train sync-patterns'
alias sep-sync-params='sep-train sync-parameters'
alias sep-remote-config='sep-train configure-remote 100.85.55.105'

# Data management
alias sep-fetch-data='sep-train fetch-weekly'
alias sep-validate-cache='sep-train validate-cache'

# Monitoring
alias sep-monitor='sep-train monitor'
alias sep-health='sep-train system-health'
alias sep-benchmark='sep-train benchmark'

# Live tuning
alias sep-tune-start='sep-train start-tuning'
alias sep-tune-stop='sep-train stop-tuning'
alias sep-tune-status='sep-train tuning-status'

# Remote trader status (via Tailscale)
alias sep-remote-status='curl -s http://100.85.55.105:8080/api/v1/status | jq'
alias sep-remote-pairs='curl -s http://100.85.55.105:8080/api/v1/pairs | jq'

echo "SEP Training Coordinator aliases loaded"
echo "Available commands:"
echo "  sep-train-status     - Show training status"
echo "  sep-train-all        - Train all pairs"
echo "  sep-sync-patterns    - Sync patterns to remote"
echo "  sep-remote-status    - Check remote trader status"
echo "  sep-monitor          - Real-time monitoring"
EOF

chmod +x "$SEP_ROOT/scripts/training_aliases.sh"

# Set up systemd service for continuous training (optional)
echo "🔧 Setting up optional systemd service..."
cat > "$SEP_ROOT/scripts/sep-training.service" << EOF
[Unit]
Description=SEP Training Coordinator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SEP_ROOT
Environment=CUDA_VISIBLE_DEVICES=0
ExecStart=$SEP_ROOT/build/src/training/sep_training_cli monitor --duration=86400
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

echo "📋 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit $SEP_ROOT/config/OANDA.env with your OANDA credentials"
echo "2. Build the system: cd $SEP_ROOT && ./build.sh"
echo "3. Load aliases: source $SEP_ROOT/scripts/training_aliases.sh"
echo "4. Test connection: sep-train test-connection"
echo "5. Fetch data: sep-fetch-data"
echo "6. Start training: sep-train-all --quick"
echo ""
echo "For continuous operation:"
echo "  sudo cp $SEP_ROOT/scripts/sep-training.service /etc/systemd/system/"
echo "  sudo systemctl enable sep-training"
echo "  sudo systemctl start sep-training"
echo ""
echo "🎯 Training coordinator ready for deployment!"
