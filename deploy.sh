#!/bin/bash

# SEP Professional Trading System - Unified Deployment Script
# This script manages the complete deployment of the SEP trading system to DigitalOcean

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DROPLET_IP="129.212.145.195"
DROPLET_USER="root"
PROJECT_NAME="sep-trading"
LOCAL_COMPOSE_FILE="docker-compose.production.yml"
DOMAIN="mxbikes.xyz"

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to update GitHub repository
update_github() {
    log_info "Updating GitHub repository..."
    
    # Add all changes
    git add .
    
    # Commit changes with timestamp
    git commit -m "Automatic update commit $(date)"
    
    # Push to remote repository
    git push origin main
    
    log_success "GitHub repository updated successfully!"
}

# Function to update server with latest changes
update_server() {
    log_info "Updating server with latest changes..."
    
    # SSH into server and pull latest changes
    ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" << 'EOF'
        cd /sep
        git pull origin main
        echo "Server updated with latest changes"
       ./deploy.sh local

EOF
    
    log_success "Server updated successfully!"
}

check_dependencies() {
    local target=${1:-"local"}
    log_info "Checking system dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check ssh access to droplet only for droplet deployments
    if [ "$target" = "droplet" ] && [ -n "$DROPLET_IP" ] && ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" "echo 'SSH OK'" &> /dev/null; then
        log_error "Cannot connect to droplet $DROPLET_IP via SSH."
        log_error "Please ensure SSH keys are configured and droplet is accessible."
        exit 1
    fi
    
    log_success "All dependencies are available"
}

# Function to set up local environment
setup_local_environment() {
    log_info "Setting up local environment..."
    
    # Create necessary directories
    mkdir -p data/redis
    mkdir -p logs
    
    # Set proper permissions
    chmod 777 data/redis
    
    log_success "Local environment setup completed"
}

# Function to start local services
start_local_services() {
    log_info "Starting local services..."
    
    # Start services using docker-compose
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" up -d --build
    else
        docker compose -p "$PROJECT_NAME" up -d --build
    fi
    
    log_success "Local services started"
}

# Function to stop services
stop_services() {
    local target=${1:-"local"}
    
    if [ "$target" = "droplet" ] && [ -n "$DROPLET_IP" ]; then
        log_info "Stopping droplet services..."
        ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" \
            "cd /opt/sep-trader && docker-compose -f docker-compose.production.yml down"
    else
        log_info "Stopping local services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -p "$PROJECT_NAME" down
        else
            docker compose -p "$PROJECT_NAME" down
        fi
    fi
    
    log_success "Services stopped"
}



show_help() {
    echo "SEP Professional Trading System Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  local         - Start services locally"
    echo "  remote        - Update GitHub and server (git commit, push, then SSH to server and pull)"
    echo "  stop          - Stop local services"
    echo "  clean         - Stop services and remove containers"
    echo "  help          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local      # Start services locally"
    echo "  $0 remote     # Update GitHub and server"
}

# Main script logic
case "${1:-help}" in
    "local")
        check_dependencies "local"
        setup_local_environment
        start_local_services
        sleep 10
        log_success "SEP Trading System is running locally!"
        log_info "Frontend: http://localhost"
        log_info "Backend API: http://localhost:5000"
        log_info "WebSocket: ws://localhost:8765"
        ;;
    "remote")
        update_github
        update_server
        ;;
    "stop")
        stop_services "local"
        ;;
    "clean")
        if command -v docker-compose &> /dev/null; then
            docker-compose -p "$PROJECT_NAME" down -v --remove-orphans
        else
            docker compose -p "$PROJECT_NAME" down -v --remove-orphans
        fi
        log_success "System cleaned"
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac