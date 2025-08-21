#!/bin/bash

# SEP Professional Trading System Deployment Script
# This script manages the deployment of the complete trading system

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="sep-trading"

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
    
    log_success "All dependencies are available"
}

create_directories() {
    log_info "Creating required directories..."
    
    directories=("data" "logs" "config")
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Required directories are ready"
}

build_services() {
    log_info "Building Docker services..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" build --no-cache
    else
        docker compose -p "$PROJECT_NAME" build --no-cache
    fi
    
    log_success "All services built successfully"
}

start_services() {
    log_info "Starting all services..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" up -d
    else
        docker compose -p "$PROJECT_NAME" up -d
    fi
    
    log_success "All services started successfully"
}

stop_services() {
    log_info "Stopping all services..."
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" down
    else
        docker compose -p "$PROJECT_NAME" down
    fi
    
    log_success "All services stopped"
}

show_status() {
    log_info "Service status:"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" ps
    else
        docker compose -p "$PROJECT_NAME" ps
    fi
}

show_logs() {
    local service=$1
    log_info "Showing logs for service: $service"
    
    if command -v docker-compose &> /dev/null; then
        docker-compose -p "$PROJECT_NAME" logs -f "$service"
    else
        docker compose -p "$PROJECT_NAME" logs -f "$service"
    fi
}

health_check() {
    log_info "Performing health checks..."
    
    # Check Redis
    if curl -f http://localhost:6379 &> /dev/null; then
        log_success "Redis: Healthy"
    else
        log_warning "Redis: Not accessible"
    fi
    
    # Check Backend API
    if curl -f http://localhost:5000/api/health &> /dev/null; then
        log_success "Backend API: Healthy"
    else
        log_warning "Backend API: Not accessible"
    fi
    
    # Check Frontend
    if curl -f http://localhost/health &> /dev/null; then
        log_success "Frontend: Healthy"
    else
        log_warning "Frontend: Not accessible"
    fi
    
    # Check WebSocket
    if nc -z localhost 8765 &> /dev/null; then
        log_success "WebSocket Service: Healthy"
    else
        log_warning "WebSocket Service: Not accessible"
    fi
}

show_help() {
    echo "SEP Professional Trading System Deployment Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start    - Build and start all services"
    echo "  stop     - Stop all services"
    echo "  restart  - Restart all services"
    echo "  status   - Show service status"
    echo "  logs     - Show logs for all services"
    echo "  logs SERVICE - Show logs for specific service"
    echo "  health   - Perform health checks"
    echo "  clean    - Stop services and remove containers"
    echo "  help     - Show this help message"
    echo ""
    echo "Available services: redis, trading-backend, websocket-service, frontend"
}

# Main script logic
case "${1:-start}" in
    "start")
        check_dependencies
        create_directories
        build_services
        start_services
        sleep 10
        health_check
        log_success "SEP Trading System is now running!"
        log_info "Frontend: http://localhost"
        log_info "Backend API: http://localhost:5000"
        log_info "WebSocket: ws://localhost:8765"
        ;;
    "stop")
        stop_services
        ;;
    "restart")
        stop_services
        sleep 5
        start_services
        sleep 10
        health_check
        ;;
    "status")
        show_status
        ;;
    "logs")
        if [ -n "$2" ]; then
            show_logs "$2"
        else
            if command -v docker-compose &> /dev/null; then
                docker-compose -p "$PROJECT_NAME" logs -f
            else
                docker compose -p "$PROJECT_NAME" logs -f
            fi
        fi
        ;;
    "health")
        health_check
        ;;
    "clean")
        stop_services
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