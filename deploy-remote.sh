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

# Function to verify SSL configuration exists
check_ssl_config() {
    echo "🌐 Checking SSL nginx configuration..."
    
    if [ ! -f "frontend/nginx-ssl.conf" ]; then
        echo "❌ SSL nginx configuration not found at frontend/nginx-ssl.conf"
        exit 1
    fi
    
    echo "✅ SSL nginx configuration found"
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
    
    # Check SSL configuration
    check_ssl_config
    
    # Start services
    start_services
    
    # Show final status
    show_status
}

# Run main function
main "$@"