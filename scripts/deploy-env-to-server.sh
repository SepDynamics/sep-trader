#!/bin/bash

# SEP Environment File Deployment Script
# =====================================
# 
# This script securely transfers the production environment file to the DigitalOcean droplet
# and sets up proper permissions for the SEP trading system deployment.
#
# Usage: ./scripts/deploy-env-to-server.sh [droplet-ip] [username]
#

set -e

# Configuration
DROPLET_IP=${1:-"129.212.145.195"}
USERNAME=${2:-"root"}
LOCAL_ENV_FILE="config/.sep-config.env"
REMOTE_ENV_PATH="/sep/config/.sep-config.env"

echo "=== SEP Environment File Deployment ==="
echo "Droplet IP: $DROPLET_IP"
echo "Username: $USERNAME"
echo "Local file: $LOCAL_ENV_FILE"
echo "Remote path: $REMOTE_ENV_PATH"
echo

# Validate local environment file exists
if [ ! -f "$LOCAL_ENV_FILE" ]; then
    echo "ERROR: Local environment file not found: $LOCAL_ENV_FILE"
    echo "Please ensure the production environment file exists."
    exit 1
fi

echo "✓ Local environment file found"

# Test SSH connection
echo "Testing SSH connection to $USERNAME@$DROPLET_IP..."
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$USERNAME@$DROPLET_IP" exit 2>/dev/null; then
    echo "ERROR: Cannot connect to $USERNAME@$DROPLET_IP via SSH"
    echo "Please ensure:"
    echo "  1. SSH key is properly configured"
    echo "  2. Droplet IP address is correct"
    echo "  3. Username is correct"
    exit 1
fi

echo "✓ SSH connection successful"

# Create remote directory structure
echo "Creating remote directory structure..."
ssh "$USERNAME@$DROPLET_IP" "mkdir -p /sep/config"

# Transfer environment file
echo "Transferring environment file..."
scp "$LOCAL_ENV_FILE" "$USERNAME@$DROPLET_IP:$REMOTE_ENV_PATH"

# Set proper permissions
echo "Setting file permissions..."
ssh "$USERNAME@$DROPLET_IP" "chmod 600 $REMOTE_ENV_PATH && chown root:root $REMOTE_ENV_PATH"

# Verify deployment
echo "Verifying deployment..."
REMOTE_SIZE=$(ssh "$USERNAME@$DROPLET_IP" "stat -f%z $REMOTE_ENV_PATH 2>/dev/null || stat -c%s $REMOTE_ENV_PATH")
LOCAL_SIZE=$(stat -f%z "$LOCAL_ENV_FILE" 2>/dev/null || stat -c%s "$LOCAL_ENV_FILE")

if [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ]; then
    echo "✓ Environment file successfully deployed and verified"
else
    echo "WARNING: File sizes don't match (Local: $LOCAL_SIZE, Remote: $REMOTE_SIZE)"
fi

echo
echo "=== Deployment Complete ==="
echo
echo "Next steps:"
echo "1. SSH to your droplet: ssh $USERNAME@$DROPLET_IP"
echo "2. Navigate to SEP directory: cd /sep"
echo "3. Deploy the application: docker-compose up -d"
echo "4. Check logs: docker-compose logs -f"
echo
echo "Environment file deployed to: $REMOTE_ENV_PATH"
echo "File permissions: 600 (read/write for owner only)"