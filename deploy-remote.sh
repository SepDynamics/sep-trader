#!/bin/bash

# SEP Trading System - Remote Deployment Script for mxbikes.xyz
# This script deploys the SEP system to your remote server

set -e

echo "🚀 Starting SEP Trading System deployment for mxbikes.xyz..."

# Check if we're running as root or with sudo
if [[ $EUID -eq 0 ]]; then
   echo "⚠️  This script should not be run as root. Please run as a regular user with docker permissions."
   exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Function to check if we need to install SSL certificates
setup_ssl() {
    echo "🔒 Setting up SSL certificates for mxbikes.xyz..."
    
    # Create SSL directory if it doesn't exist
    sudo mkdir -p /etc/ssl/certs/sep
    
    # Check if Let's Encrypt certificates exist
    if [ ! -f "/etc/letsencrypt/live/mxbikes.xyz/fullchain.pem" ]; then
        echo "📜 SSL certificates not found. You may want to set up Let's Encrypt later."
        echo "   Run: sudo certbot --nginx -d mxbikes.xyz"
        
        # Create self-signed certificates for now
        echo "🔧 Creating temporary self-signed certificates..."
        sudo openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout /etc/ssl/certs/sep/nginx.key \
            -out /etc/ssl/certs/sep/nginx.crt \
            -subj "/C=US/ST=State/L=City/O=SEP/OU=Trading/CN=mxbikes.xyz"
    fi
}

# Function to update nginx configuration for SSL
update_nginx_ssl() {
    echo "🌐 Updating nginx configuration for SSL..."
    
    # Create the nginx SSL config with proper permissions
    cat > /tmp/nginx-ssl.conf << 'EOF'
server {
    listen 80;
    server_name mxbikes.xyz www.mxbikes.xyz;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name mxbikes.xyz www.mxbikes.xyz;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/sep/nginx.crt;
    ssl_certificate_key /etc/ssl/certs/sep/nginx.key;
    ssl_session_cache shared:SSL:1m;
    ssl_session_timeout 10m;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    root /usr/share/nginx/html;
    index index.html index.htm;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "no-referrer-when-downgrade" always;
    add_header Content-Security-Policy "default-src 'self' https: data: blob: 'unsafe-inline'" always;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/x-javascript application/xml+rss application/javascript;
    
    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # API proxy to backend
    location /api/ {
        proxy_pass http://trading-backend:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }
    
    # WebSocket proxy
    location /ws/ {
        proxy_pass http://websocket-service:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # React Router support - serve index.html for all routes
    location / {
        try_files $uri $uri/ /index.html;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
EOF
    
    # Move the file to the frontend directory
    sudo mv /tmp/nginx-ssl.conf frontend/nginx-ssl.conf
    sudo chown $USER:$USER frontend/nginx-ssl.conf
}

# Function to start services
start_services() {
    echo "📦 Building and starting SEP services..."
    
    # Stop any existing services
    docker-compose down 2>/dev/null || true
    
    # Clean up old images if they exist
    echo "🧹 Cleaning up old Docker images..."
    docker system prune -f
    
    # Build and start services
    echo "🔨 Building services..."
    docker-compose build --no-cache
    
    echo "🚀 Starting services..."
    docker-compose up -d
    
    # Wait for services to be healthy
    echo "⏳ Waiting for services to start..."
    sleep 30
    
    # Check service status
    echo "📊 Service status:"
    docker-compose ps
}

# Function to show deployment status
show_status() {
    echo ""
    echo "✅ SEP Trading System deployment completed!"
    echo ""
    echo "📍 Access your system at: https://mxbikes.xyz"
    echo ""
    echo "🔧 Service URLs:"
    echo "   Frontend:  https://mxbikes.xyz"
    echo "   Backend:   https://mxbikes.xyz/api"
    echo "   WebSocket: wss://mxbikes.xyz/ws"
    echo ""
    echo "📋 Useful commands:"
    echo "   View logs:     docker-compose logs -f"
    echo "   Stop services: docker-compose down"
    echo "   Restart:       docker-compose restart"
    echo ""
    echo "🔒 SSL Certificate Setup:"
    echo "   For production SSL: sudo certbot --nginx -d mxbikes.xyz"
    echo ""
}

# Main deployment process
main() {
    echo "🏁 Starting deployment process..."
    
    # Check prerequisites
    if [ ! -f "docker-compose.yml" ]; then
        echo "❌ docker-compose.yml not found. Please run this script from the project root."
        exit 1
    fi
    
    # Setup SSL certificates
    setup_ssl
    
    # Update nginx configuration
    update_nginx_ssl
    
    # Start services
    start_services
    
    # Show final status
    show_status
}

# Run main function
main "$@"