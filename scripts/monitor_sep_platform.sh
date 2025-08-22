#!/bin/bash

# SEP Trading Platform Monitoring Script
# Monitors all services and provides health status

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOGFILE="/var/log/sep_monitor.log"
ALERT_THRESHOLD=3  # Number of failed checks before alert

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOGFILE"
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url=$1
    local service_name=$2
    local timeout=${3:-10}
    
    if curl -s -f --max-time "$timeout" "$url" > /dev/null; then
        echo -e "${GREEN}✓${NC} $service_name is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name is unhealthy"
        return 1
    fi
}

# Function to check port connectivity
check_port() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-5}
    
    if timeout "$timeout" bash -c "echo >/dev/tcp/$host/$port" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $service_name port $port is accessible"
        return 0
    else
        echo -e "${RED}✗${NC} $service_name port $port is not accessible"
        return 1
    fi
}

# Function to check Docker service
check_docker_service() {
    local service_name=$1
    local status
    
    status=$(docker-compose ps -q "$service_name" | xargs docker inspect --format='{{.State.Status}}' 2>/dev/null || echo "not_found")
    
    case "$status" in
        "running")
            echo -e "${GREEN}✓${NC} Docker service $service_name is running"
            return 0
            ;;
        "not_found")
            echo -e "${RED}✗${NC} Docker service $service_name not found"
            return 1
            ;;
        *)
            echo -e "${YELLOW}⚠${NC} Docker service $service_name status: $status"
            return 1
            ;;
    esac
}

# Function to get service logs (last 10 lines)
get_service_logs() {
    local service_name=$1
    echo -e "${BLUE}📋 Last 10 log lines for $service_name:${NC}"
    docker-compose logs --tail=10 "$service_name" 2>/dev/null || echo "Could not retrieve logs"
    echo ""
}

# Function to check Redis
check_redis() {
    local redis_check
    redis_check=$(docker-compose exec -T redis redis-cli ping 2>/dev/null || echo "FAILED")
    
    if [ "$redis_check" = "PONG" ]; then
        echo -e "${GREEN}✓${NC} Redis is responding"
        return 0
    else
        echo -e "${RED}✗${NC} Redis is not responding"
        return 1
    fi
}

# Function to check disk space
check_disk_space() {
    local usage
    usage=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
    
    if [ "$usage" -lt 90 ]; then
        echo -e "${GREEN}✓${NC} Disk usage: ${usage}%"
        return 0
    elif [ "$usage" -lt 95 ]; then
        echo -e "${YELLOW}⚠${NC} Disk usage: ${usage}% (Warning)"
        return 0
    else
        echo -e "${RED}✗${NC} Disk usage: ${usage}% (Critical)"
        return 1
    fi
}

# Function to check memory usage
check_memory() {
    local mem_usage
    mem_usage=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
    
    if [ "$mem_usage" -lt 85 ]; then
        echo -e "${GREEN}✓${NC} Memory usage: ${mem_usage}%"
        return 0
    elif [ "$mem_usage" -lt 95 ]; then
        echo -e "${YELLOW}⚠${NC} Memory usage: ${mem_usage}% (Warning)"
        return 0
    else
        echo -e "${RED}✗${NC} Memory usage: ${mem_usage}% (Critical)"
        return 1
    fi
}

# Function to check OANDA data file
check_oanda_data() {
    local data_file="/sep/eur_usd_m1_48h.json"
    local file_age
    
    if [ -f "$data_file" ]; then
        file_age=$(find "$data_file" -mmin +120 2>/dev/null | wc -l)  # Check if older than 2 hours
        if [ "$file_age" -eq 0 ]; then
            echo -e "${GREEN}✓${NC} OANDA data file is recent"
            return 0
        else
            echo -e "${YELLOW}⚠${NC} OANDA data file is older than 2 hours"
            return 1
        fi
    else
        echo -e "${RED}✗${NC} OANDA data file not found"
        return 1
    fi
}

# Main monitoring function
run_health_checks() {
    local failed_checks=0
    
    echo -e "${BLUE}🚀 SEP Trading Platform Health Check - $(date)${NC}"
    echo "=================================================="
    
    # System resources
    echo -e "\n${BLUE}📊 System Resources:${NC}"
    check_disk_space || ((failed_checks++))
    check_memory || ((failed_checks++))
    
    # Docker services
    echo -e "\n${BLUE}🐳 Docker Services:${NC}"
    check_docker_service "redis" || ((failed_checks++))
    check_docker_service "trading-backend" || ((failed_checks++))
    check_docker_service "websocket-service" || ((failed_checks++))
    check_docker_service "frontend" || ((failed_checks++))
    
    # Service connectivity
    echo -e "\n${BLUE}🌐 Service Connectivity:${NC}"
    check_redis || ((failed_checks++))
    check_port "localhost" "5000" "Trading Backend API" || ((failed_checks++))
    check_port "localhost" "8765" "WebSocket Service" || ((failed_checks++))
    check_port "localhost" "3000" "Frontend" || ((failed_checks++))
    
    # HTTP endpoints
    echo -e "\n${BLUE}🔍 HTTP Health Endpoints:${NC}"
    check_http_endpoint "http://localhost:5000/api/health" "Trading Backend API" || ((failed_checks++))
    check_http_endpoint "http://localhost:3000/health" "Frontend" || ((failed_checks++))
    
    # Data integrity
    echo -e "\n${BLUE}📊 Data Integrity:${NC}"
    check_oanda_data || ((failed_checks++))
    
    # Summary
    echo -e "\n${BLUE}📋 Summary:${NC}"
    if [ "$failed_checks" -eq 0 ]; then
        echo -e "${GREEN}✅ All systems operational!${NC}"
        log_message "Health check passed - all systems operational"
    elif [ "$failed_checks" -lt "$ALERT_THRESHOLD" ]; then
        echo -e "${YELLOW}⚠️  $failed_checks issues detected (below alert threshold)${NC}"
        log_message "Health check warning - $failed_checks issues detected"
    else
        echo -e "${RED}🚨 $failed_checks critical issues detected!${NC}"
        log_message "Health check failed - $failed_checks critical issues detected"
        
        # Show recent logs for failed services
        echo -e "\n${BLUE}📋 Recent Service Logs:${NC}"
        get_service_logs "trading-backend"
        get_service_logs "websocket-service"
    fi
    
    echo "=================================================="
    
    return $failed_checks
}

# Function to show continuous monitoring
continuous_monitor() {
    local interval=${1:-300}  # Default 5 minutes
    
    echo "Starting continuous monitoring (every ${interval} seconds)..."
    echo "Press Ctrl+C to stop"
    
    while true; do
        clear
        run_health_checks
        echo ""
        echo "Next check in ${interval} seconds..."
        sleep "$interval"
    done
}

# Function to show service status
show_status() {
    echo -e "${BLUE}🐳 Docker Compose Services Status:${NC}"
    docker-compose ps
    
    echo -e "\n${BLUE}📊 Service Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
}

# Function to show logs
show_logs() {
    local service=${1:-"all"}
    local lines=${2:-50}
    
    if [ "$service" = "all" ]; then
        echo "Showing logs for all services:"
        docker-compose logs --tail="$lines"
    else
        echo "Showing logs for $service:"
        docker-compose logs --tail="$lines" "$service"
    fi
}

# Function to restart services
restart_services() {
    local service=${1:-"all"}
    
    if [ "$service" = "all" ]; then
        echo "Restarting all services..."
        docker-compose restart
    else
        echo "Restarting $service..."
        docker-compose restart "$service"
    fi
}

# Main script logic
case "${1:-check}" in
    "check"|"health")
        run_health_checks
        ;;
    "monitor")
        continuous_monitor "${2:-300}"
        ;;
    "status")
        show_status
        ;;
    "logs")
        show_logs "$2" "$3"
        ;;
    "restart")
        restart_services "$2"
        ;;
    "help"|"-h"|"--help")
        echo "SEP Trading Platform Monitor"
        echo ""
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  check          - Run health checks once (default)"
        echo "  monitor [sec]  - Continuous monitoring (default: 300s)"
        echo "  status         - Show service status and resource usage"
        echo "  logs [service] [lines] - Show logs (default: all services, 50 lines)"
        echo "  restart [service]      - Restart services (default: all)"
        echo "  help           - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 check                    # Run health check once"
        echo "  $0 monitor 60              # Monitor every 60 seconds"
        echo "  $0 logs trading-backend 20 # Show last 20 lines of backend logs"
        echo "  $0 restart frontend        # Restart just the frontend service"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac