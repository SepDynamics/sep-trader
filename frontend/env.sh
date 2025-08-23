#!/bin/sh

# Environment variables injection script for React frontend
# This script runs at container startup to inject runtime environment variables

# Set default values if not provided
REACT_APP_API_URL=${REACT_APP_API_URL:-"http://165.227.109.187:5000"}
REACT_APP_WS_URL=${REACT_APP_WS_URL:-"ws://165.227.109.187:8765"}
REACT_APP_ENVIRONMENT=${REACT_APP_ENVIRONMENT:-"production"}

# Create env-config.js file with runtime environment variables
cat <<EOF > /usr/share/nginx/html/env-config.js
window._env_ = {
  REACT_APP_API_URL: '$REACT_APP_API_URL',
  REACT_APP_WS_URL: '$REACT_APP_WS_URL',
  REACT_APP_ENVIRONMENT: '$REACT_APP_ENVIRONMENT'
};
EOF

echo "Environment configuration injected:"
echo "API URL: $REACT_APP_API_URL"
echo "WebSocket URL: $REACT_APP_WS_URL"
echo "Environment: $REACT_APP_ENVIRONMENT"