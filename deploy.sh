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

setup_local_environment() {
    log_info "Setting up local development environment..."
    
    # Create required directories
    directories=("data" "logs" "config")
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    # Build frontend if needed
    if [ -d "frontend" ] && [ ! -d "frontend/build" ]; then
        log_info "Building frontend..."
        cd frontend
        npm install
        npm run build
        cd ..
    fi
    
    log_success "Local environment is ready"
}

deploy_to_droplet() {
    log_info "Deploying to DigitalOcean droplet ($DROPLET_IP)..."
    
    # Sync project files to droplet
    log_info "Syncing project files..."
    rsync -avz --exclude='.git' --exclude='node_modules' --exclude='build' --exclude='.cache' \
        --exclude='docs_archive' --exclude='pitch' --exclude='validation' \
        ./ "$DROPLET_USER@$DROPLET_IP:/opt/sep-trader/"
    
    # Setup and start services on droplet
    ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" << 'EOF'
        cd /opt/sep-trader
        
        # Stop any existing services
        if [ -f docker-compose.production.yml ]; then
            docker-compose -f docker-compose.production.yml down 2>/dev/null || true
        fi
        
        # Clean up old images
        docker system prune -f
        
        # Start services using production compose
        echo "Starting SEP services..."
        docker-compose -f docker-compose.production.yml up -d --build
        
        # Wait for services to be ready
        sleep 30
        
        echo "Service status:"
        docker-compose -f docker-compose.production.yml ps
EOF
    
    log_success "Deployment to droplet completed"
}

start_local_services() {
    log_info "Starting local services..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" -f docker-compose.yml up -d --build
    else
        docker compose -p "$PROJECT_NAME" -f docker-compose.yml up -d --build
    fi
    
    log_success "Local services started"
}

stop_services() {
    local target=${1:-"local"}
    
    if [ "$target" = "droplet" ] && [ -n "$DROPLET_IP" ]; then
        log_info "Stopping droplet services..."
        ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" << 'EOF'
            cd /opt/sep-trader
            docker-compose -f docker-compose.production.yml down
EOF
        log_success "Droplet services stopped"
    else
        log_info "Stopping local services..."
        if command -v docker-compose &> /dev/null; then
            docker-compose -p "$PROJECT_NAME" down
        else
            docker compose -p "$PROJECT_NAME" down
        fi
        log_success "Local services stopped"
    fi
}

health_check() {
    local target=${1:-"local"}
    local base_url
    
    if [ "$target" = "droplet" ] && [ -n "$DROPLET_IP" ]; then
        base_url="http://$DROPLET_IP"
    else
        base_url="http://localhost"
    fi
    
    log_info "Performing health checks on $target..."
    
    # Check Frontend
    if curl -f "$base_url/health" &> /dev/null; then
        log_success "Frontend: Healthy"
    else
        log_warning "Frontend: Not accessible at $base_url/health"
    fi
    
    # Check Backend API
    if curl -f "$base_url/api/health" &> /dev/null; then
        log_success "Backend API: Healthy"
    else
        log_warning "Backend API: Not accessible at $base_url/api/health"
    fi
    
    # Check WebSocket (if local)
    if [ "$target" = "local" ] && nc -z localhost 8765 &> /dev/null; then
        log_success "WebSocket Service: Healthy"
    elif [ "$target" = "droplet" ] && nc -z "$DROPLET_IP" 8765 &> /dev/null; then
        log_success "WebSocket Service: Healthy"
    else
        log_warning "WebSocket Service: Not accessible"
    fi
    
    # Check Redis (if local)
    if [ "$target" = "local" ] && nc -z localhost 6380 &> /dev/null; then
        log_success "Redis: Healthy"
    elif [ "$target" = "droplet" ] && nc -z "$DROPLET_IP" 6380 &> /dev/null; then
        log_success "Redis: Healthy"
    else
        log_warning "Redis: Not accessible"
    fi
}

show_logs() {
    local target=${1:-"local"}
    local service=${2:-""}
    
    if [ "$target" = "droplet" ] && [ -n "$DROPLET_IP" ]; then
        log_info "Showing droplet logs..."
        if [ -n "$service" ]; then
            ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" \
                "cd /opt/sep-trader && docker-compose -f docker-compose.production.yml logs -f $service"
        else
            ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" \
                "cd /opt/sep-trader && docker-compose -f docker-compose.production.yml logs -f"
        fi
    else
        log_info "Showing local logs..."
        if [ -n "$service" ]; then
            if command -v docker-compose &> /dev/null; then
                docker-compose -p "$PROJECT_NAME" logs -f "$service"
            else
                docker compose -p "$PROJECT_NAME" logs -f "$service"
            fi
        else
            if command -v docker-compose &> /dev/null; then
                docker-compose -p "$PROJECT_NAME" logs -f
            else
                docker compose -p "$PROJECT_NAME" logs -f
            fi
        fi
    fi
}

test_website() {
    local target=${1:-"local"}
    local base_url
    
    if [ "$target" = "droplet" ]; then
        base_url="https://$DOMAIN"
    else
        base_url="http://localhost"
    fi
    
    log_info "Testing website functionality at $base_url..."
    
    # Test main page
    if curl -s -o /dev/null -w "%{http_code}" "$base_url" | grep -q "200"; then
        log_success "Main page loads successfully"
    else
        log_error "Main page failed to load"
    fi
    
    # Test health endpoint
    if curl -s -o /dev/null -w "%{http_code}" "$base_url/health" | grep -q "200"; then
        log_success "Health endpoint responding"
    else
        log_warning "Health endpoint not responding"
    fi
    
    # Test API endpoint
    if curl -s -o /dev/null -w "%{http_code}" "$base_url/api/health" | grep -q "200"; then
        log_success "API endpoint responding"
    else
        log_warning "API endpoint not responding"
    fi
    
    log_info "Website test completed for $target"
}

show_status() {
    local target=${1:-"local"}
    
    if [ "$target" = "droplet" ] && [ -n "$DROPLET_IP" ]; then
        log_info "Droplet service status:"
        ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" \
            "cd /opt/sep-trader && docker-compose -f docker-compose.production.yml ps"
    else
        log_info "Local service status:"
        if command -v docker-compose &> /dev/null; then
            docker-compose -p "$PROJECT_NAME" ps
        else
            docker compose -p "$PROJECT_NAME" ps
        fi
    fi
}

show_help() {
    echo "SEP Professional Trading System Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND] [TARGET]"
    echo ""
    echo "Commands:"
    echo "  local         - Start services locally"
    echo "  deploy        - Deploy to DigitalOcean droplet"
    echo "  stop [target] - Stop services (local|droplet)"
    echo "  status [target] - Show service status (local|droplet)"
    echo "  logs [target] [service] - Show logs (local|droplet)"
    echo "  health [target] - Perform health checks (local|droplet)"
    echo "  test [target] - Test website functionality (local|droplet)"
    echo "  clean [target] - Stop services and remove containers (local|droplet)"
    echo "  help          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 local                    # Start services locally"
    echo "  $0 deploy                   # Deploy to droplet"
    echo "  $0 health droplet          # Check droplet health"
    echo "  $0 logs droplet frontend   # Show droplet frontend logs"
    echo "  $0 test droplet            # Test droplet website"
    echo ""
    echo "Targets:"
    echo "  local   - Local development environment"
    echo "  droplet - DigitalOcean droplet ($DROPLET_IP)"
}

# Main script logic
case "${1:-help}" in
    "local")
        check_dependencies "local"
        setup_local_environment
        start_local_services
        sleep 10
        health_check "local"
        log_success "SEP Trading System is running locally!"
        log_info "Frontend: http://localhost"
        log_info "Backend API: http://localhost:5000"
        log_info "WebSocket: ws://localhost:8765"
        ;;
    "deploy")
        check_dependencies "droplet"
        deploy_to_droplet
        sleep 15
        health_check "droplet"
        test_website "droplet"
        log_success "SEP Trading System deployed to $DOMAIN!"
        log_info "Website: https://$DOMAIN"
        log_info "API: https://$DOMAIN/api"
        ;;
    "stop")
        stop_services "${2:-local}"
        ;;
    "status")
        show_status "${2:-local}"
        ;;
    "logs")
        show_logs "${2:-local}" "$3"
        ;;
    "health")
        health_check "${2:-local}"
        ;;
    "test")
        test_website "${2:-local}"
        ;;
    "clean")
        target="${2:-local}"
        if [ "$target" = "droplet" ]; then
            ssh -o StrictHostKeyChecking=no "$DROPLET_USER@$DROPLET_IP" \
                "cd /opt/sep-trader && docker-compose -f docker-compose.production.yml down -v --remove-orphans"
        else
            if command -v docker-compose &> /dev/null; then
                docker-compose -p "$PROJECT_NAME" down -v --remove-orphans
            else
                docker compose -p "$PROJECT_NAME" down -v --remove-orphans
            fi
        fi
        log_success "System cleaned ($target)"
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