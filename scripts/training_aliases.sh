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
