#!/bin/bash
# SEP Droplet - Local Development Aliases

echo "🔧 Setting up SEP droplet aliases..."

# Add to your ~/.bashrc or ~/.zshrc
ALIAS_FILE="$HOME/.bashrc"
if [[ "$SHELL" == *"zsh"* ]]; then
    ALIAS_FILE="$HOME/.zshrc"
fi

# Create backup
cp "$ALIAS_FILE" "${ALIAS_FILE}.backup.$(date +%Y%m%d)"

# Add SEP aliases
cat >> "$ALIAS_FILE" << 'EOF'

# SEP Professional Trader-Bot Aliases
export SEP_DROPLET_IP="165.227.109.187"
export SEP_DROPLET_USER="root"

# Quick access
alias sep-ssh="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP"
alias sep-sync="cd /sep && ./scripts/sync_to_droplet.sh"
alias sep-deploy="cd /sep && ./scripts/deploy_to_droplet.sh"

# Status checks
alias sep-status="curl -s http://$SEP_DROPLET_IP/api/status | jq"
alias sep-health="curl -s http://$SEP_DROPLET_IP/health | jq"
alias sep-logs="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose logs -f --tail=50'"

# Database queries
alias sep-db="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'sudo -u postgres psql sep_trading'"
alias sep-pairs="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP \"sudo -u postgres psql sep_trading -c 'SELECT * FROM trading_pairs;'\""
alias sep-signals="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP \"sudo -u postgres psql sep_trading -c 'SELECT * FROM v_recent_signals LIMIT 10;'\""

# Service management
alias sep-restart="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose restart'"
alias sep-stop="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose stop'"
alias sep-start="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose up -d'"

# Quick development
alias sep-tail="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'tail -f /opt/sep-trader/logs/*.log'"
alias sep-ps="ssh $SEP_DROPLET_USER@$SEP_DROPLET_IP 'cd /opt/sep-trader/sep-trader && docker-compose ps'"

EOF

echo "✅ Aliases added to $ALIAS_FILE"
echo ""
echo "🔄 Reload your shell or run: source $ALIAS_FILE"
echo ""
echo "📋 Available commands:"
echo "   sep-ssh          # SSH to droplet"
echo "   sep-sync         # Sync data to droplet"
echo "   sep-status       # Check API status"
echo "   sep-health       # Check health status"
echo "   sep-db           # Connect to database"
echo "   sep-logs         # View service logs"
echo "   sep-restart      # Restart services"
echo ""
echo "🚀 Ready to deploy with: sep-deploy"
