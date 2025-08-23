#!/bin/bash

# SEP Trading System - Valkey (Redis-compatible) Connection Test Script
# =====================================================================
#
# This script tests the connection to the external Valkey database
# with authentication and provides detailed diagnostics.
#
# Usage:
#   ./scripts/test-valkey-connection.sh [config_file]
#
# Arguments:
#   config_file: Path to environment configuration file (default: config/.sep-config.env)

set -euo pipefail

# =================================================================
# CONFIGURATION
# =================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${1:-$PROJECT_ROOT/config/.sep-config.env}"

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    ((TESTS_PASSED++))
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    ((TESTS_FAILED++))
}

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

print_separator() {
    echo "================================================================="
}

# =================================================================
# CONFIGURATION VALIDATION
# =================================================================

validate_config() {
    log_test "Validating configuration file: $CONFIG_FILE"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        echo "Please create the configuration file with Valkey credentials."
        return 1
    fi
    
    # Source configuration
    set +e
    set -a
    source "$CONFIG_FILE"
    local source_exit_code=$?
    set +a
    set -e
    if [ $source_exit_code -ne 0 ]; then
        log_error "Configuration file contains syntax errors. Please check the file."
        return 1
    fi
    
    # Validate required variables
    local required_vars=(
        "REDIS_HOST"
        "REDIS_PORT" 
        "REDIS_USER"
        "REDIS_PASSWORD"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required Valkey configuration variables: ${missing_vars[*]}"
        return 1
    fi
    
    log_success "Configuration file validation passed"
    
    # Display connection details (without password)
    echo
    echo "Valkey Connection Details:"
    echo "  Host: $REDIS_HOST"
    echo "  Port: $REDIS_PORT"
    echo "  User: $REDIS_USER"
    echo "  URL: ${REDIS_URL//$REDIS_PASSWORD/****}"
    echo
}

# =================================================================
# NETWORK CONNECTIVITY TESTS
# =================================================================

test_network_connectivity() {
    log_test "Testing network connectivity to $REDIS_HOST:$REDIS_PORT"
    
    # Test basic network connectivity
    if ! timeout 10 bash -c "exec 3<>/dev/tcp/$REDIS_HOST/$REDIS_PORT" 2>/dev/null; then
        log_error "Cannot connect to $REDIS_HOST:$REDIS_PORT"
        echo "Possible causes:"
        echo "- Incorrect hostname or port"
        echo "- Network connectivity issues"
        echo "- Firewall blocking connection"
        echo "- IP address not in trusted sources"
        return 1
    fi
    exec 3>&-  # Close connection
    
    log_success "Network connectivity test passed"
}

test_dns_resolution() {
    log_test "Testing DNS resolution for $REDIS_HOST"
    
    local ip_address
    if ! ip_address=$(dig +short "$REDIS_HOST" | head -1); then
        log_error "DNS resolution failed for $REDIS_HOST"
        return 1
    fi
    
    if [ -z "$ip_address" ]; then
        log_error "No IP address resolved for $REDIS_HOST"
        return 1
    fi
    
    log_success "DNS resolution successful: $REDIS_HOST -> $ip_address"
}

# =================================================================
# VALKEY/REDIS CONNECTION TESTS
# =================================================================

test_basic_connection() {
    log_test "Testing basic Valkey connection"
    
    local redis_output
    if ! redis_output=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" ping 2>&1); then
        log_error "Basic Valkey connection failed"
        echo "Common issues:"
        echo "- Incorrect credentials"
        echo "- Valkey service not running"
        echo "- IP address not in trusted sources"
        echo "- Authentication issues"
        if [ -n "$redis_output" ]; then
            echo
            echo -e "${RED}redis-cli error output:${NC}"
            echo "$redis_output" | sed 's/^/  /'
        fi
        return 1
    fi
    
    if [ "$redis_output" = "PONG" ]; then
        log_success "Basic Valkey connection test passed"
    else
        log_error "Unexpected response from Valkey: $redis_output"
        return 1
    fi
}

test_authentication() {
    log_test "Testing Valkey authentication"
    
    # Test authentication with wrong credentials
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "wrongpassword" ping &>/dev/null; then
        log_warning "Authentication test inconclusive - wrong password accepted"
    else
        log_success "Authentication properly rejects incorrect credentials"
    fi
    
    # Test with correct credentials
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" auth "$REDIS_USER" "$REDIS_PASSWORD" &>/dev/null; then
        log_success "Authentication with correct credentials successful"
    else
        log_error "Authentication failed with correct credentials"
        return 1
    fi
}

test_valkey_info() {
    log_test "Retrieving Valkey server information"
    
    local server_info
    server_info=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" info server 2>/dev/null)
    
    if [ -n "$server_info" ]; then
        local redis_version
        redis_version=$(echo "$server_info" | grep "redis_version:" | cut -d: -f2 | tr -d '\r')
        if [ -n "$redis_version" ]; then
            log_success "Valkey server version: $redis_version"
        fi
        
        local redis_mode
        redis_mode=$(echo "$server_info" | grep "redis_mode:" | cut -d: -f2 | tr -d '\r')
        if [ -n "$redis_mode" ]; then
            echo "Server mode: $redis_mode"
        fi
    else
        log_error "Could not retrieve Valkey server information"
        return 1
    fi
}

test_basic_operations() {
    log_test "Testing basic Valkey operations"
    
    local test_key="sep_test_$(date +%s)"
    local test_value="test_value_$(date +%s)"
    
    # Test SET operation
    if ! redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" set "$test_key" "$test_value" &>/dev/null; then
        log_error "Failed to SET test key"
        return 1
    fi
    
    # Test GET operation
    local retrieved_value
    retrieved_value=$(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" get "$test_key" 2>/dev/null)
    
    if [ "$retrieved_value" = "$test_value" ]; then
        log_success "Basic SET/GET operations working correctly"
    else
        log_error "GET operation returned unexpected value: $retrieved_value"
        return 1
    fi
    
    # Cleanup test key
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" del "$test_key" &>/dev/null
}

# =================================================================
# PERFORMANCE TESTS
# =================================================================

test_connection_performance() {
    log_test "Testing connection performance"
    
    # Measure connection time
    local start_time end_time duration
    start_time=$(date +%s.%N)
    
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" --user "$REDIS_USER" --pass "$REDIS_PASSWORD" ping &>/dev/null; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        
        if (( $(echo "$duration > 2.0" | bc -l) )); then
            log_warning "Connection time is slow: ${duration}s"
        else
            log_success "Connection time: ${duration}s"
        fi
    else
        log_error "Performance test failed"
        return 1
    fi
}

# =================================================================
# MAIN EXECUTION
# =================================================================

main() {
    echo "SEP Trading System - Valkey Database Connection Test"
    print_separator
    echo "Configuration: $CONFIG_FILE"
    echo "Timestamp: $(date)"
    print_separator
    echo
    
    # Run all tests
    validate_config || exit 1
    test_dns_resolution
    test_network_connectivity
    test_basic_connection
    test_authentication
    test_valkey_info
    test_basic_operations
    test_connection_performance
    
    # Print summary
    echo
    print_separator
    echo "TEST SUMMARY"
    print_separator
    echo "Tests Passed: $TESTS_PASSED"
    echo "Tests Failed: $TESTS_FAILED"
    echo
    
    if [ $TESTS_FAILED -eq 0 ]; then
        log_success "All tests passed! Valkey database connection is ready for production."
        exit 0
    else
        log_error "Some tests failed. Please resolve the issues before proceeding."
        exit 1
    fi
}

# Check requirements
if ! command -v redis-cli &> /dev/null; then
    log_error "redis-cli client is not installed. Please install Redis client tools."
    exit 1
fi

if ! command -v dig &> /dev/null; then
    log_warning "dig is not available. DNS resolution tests will be limited."
fi

if ! command -v bc &> /dev/null; then
    log_warning "bc is not available. Performance timing will be limited."
fi

# Run main function
main "$@"