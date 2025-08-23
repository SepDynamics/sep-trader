#!/bin/bash

# SEP Trading System - DigitalOcean MySQL Connection Test Script
# =============================================================
#
# This script tests the connection to DigitalOcean managed MySQL
# with SSL/TLS verification and provides detailed diagnostics.
#
# Usage:
#   ./scripts/test-mysql-connection.sh [config_file]
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
        "DB_HOST"
        "DB_PORT" 
        "DB_USER"
        "DB_PASSWORD"
        "DB_NAME"
    )
    
    local missing_vars=()
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var:-}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required configuration variables: ${missing_vars[*]}"
        return 1
    fi
    
    log_success "Configuration file validation passed"
    
    # Display connection details (without password)
    echo
    echo "Connection Details:"
    echo "  Host: $DB_HOST"
    echo "  Port: $DB_PORT"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo "  SSL Mode: ${DB_SSL_MODE:-require}"
    echo "  SSL Cert: ${DB_SSL_ROOT_CERT:-not specified}"
    echo
}

# =================================================================
# SSL CERTIFICATE VALIDATION
# =================================================================

validate_ssl_certificate() {
    log_test "Validating SSL certificate configuration"
    
    if [ -z "${DB_SSL_ROOT_CERT:-}" ]; then
        log_warning "SSL root certificate path not specified"
        return 0
    fi
    
    if [ ! -f "$DB_SSL_ROOT_CERT" ]; then
        log_error "SSL certificate file not found: $DB_SSL_ROOT_CERT"
        return 1
    fi
    
    # Validate certificate format
    if ! openssl x509 -in "$DB_SSL_ROOT_CERT" -text -noout &>/dev/null; then
        log_error "SSL certificate file is invalid or corrupted: $DB_SSL_ROOT_CERT"
        return 1
    fi
    
    log_success "SSL certificate validation passed"
}

# =================================================================
# NETWORK CONNECTIVITY TESTS
# =================================================================

test_network_connectivity() {
    log_test "Testing network connectivity to $DB_HOST:$DB_PORT"
    
    # Test basic network connectivity
    if ! timeout 10 bash -c "exec 3<>/dev/tcp/$DB_HOST/$DB_PORT" 2>/dev/null; then
        log_error "Cannot connect to $DB_HOST:$DB_PORT"
        return 1
    fi
    exec 3>&-  # Close connection
    
    log_success "Network connectivity test passed"
}

test_dns_resolution() {
    log_test "Testing DNS resolution for $DB_HOST"
    
    local ip_address
    if ! ip_address=$(dig +short "$DB_HOST" | head -1); then
        log_error "DNS resolution failed for $DB_HOST"
        return 1
    fi
    
    if [ -z "$ip_address" ]; then
        log_error "No IP address resolved for $DB_HOST"
        return 1
    fi
    
    log_success "DNS resolution successful: $DB_HOST -> $ip_address"
}

# =================================================================
# DATABASE CONNECTION TESTS
# =================================================================

test_basic_connection() {
    log_test "Testing basic MySQL connection"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--ssl-mode=REQUIRED --ssl-ca=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--ssl-mode=REQUIRED"
    fi
    
    local mysql_output
    if ! mysql_output=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -D "$DB_NAME" $ssl_args -e "SELECT 1 as test;" 2>&1); then
        log_error "Basic database connection failed"
        echo "MySQL error output:"
        echo "$mysql_output" | sed 's/^/  /'
        return 1
    fi
    
    log_success "Basic database connection test passed"
}

test_database_version() {
    log_test "Retrieving database version and connection info"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--ssl-mode=REQUIRED --ssl-ca=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--ssl-mode=REQUIRED"
    fi
    
    local version_info
    version_info=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -D "$DB_NAME" $ssl_args -e "SELECT VERSION() as version;" -s -N 2>/dev/null)
    
    if [ -n "$version_info" ]; then
        log_success "Database version: MySQL $version_info"
    else
        log_error "Could not retrieve database version"
        return 1
    fi
    
    # Get connection info
    local connection_info
    connection_info=$(mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -D "$DB_NAME" $ssl_args -e "SHOW VARIABLES LIKE 'max_connections'; SHOW STATUS LIKE 'Threads_connected';" -s 2>/dev/null)
    
    if [ -n "$connection_info" ]; then
        echo "Connection Variables:"
        echo "$connection_info" | sed 's/^/  /'
    fi
}

test_connection_performance() {
    log_test "Testing connection performance"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--ssl-mode=REQUIRED --ssl-ca=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--ssl-mode=REQUIRED"
    fi
    
    # Measure connection time
    local start_time end_time duration
    start_time=$(date +%s.%N)
    
    if mysql -h "$DB_HOST" -P "$DB_PORT" -u "$DB_USER" -p"$DB_PASSWORD" -D "$DB_NAME" $ssl_args -e "SELECT SLEEP(0.1);" &>/dev/null; then
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc -l)
        
        if (( $(echo "$duration > 5.0" | bc -l) )); then
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
    echo "SEP Trading System - DigitalOcean MySQL Connection Test"
    print_separator
    echo "Configuration: $CONFIG_FILE"
    echo "Timestamp: $(date)"
    print_separator
    echo
    
    # Run all tests
    validate_config || exit 1
    validate_ssl_certificate
    test_dns_resolution
    test_network_connectivity
    test_basic_connection
    test_database_version
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
        log_success "All tests passed! MySQL database connection is ready for production."
        exit 0
    else
        log_error "Some tests failed. Please resolve the issues before proceeding."
        exit 1
    fi
}

# Check requirements
if ! command -v mysql &> /dev/null; then
    log_error "mysql client is not installed. Please install MySQL client tools."
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    log_warning "openssl is not available. SSL certificate validation will be limited."
fi

if ! command -v bc &> /dev/null; then
    log_warning "bc is not available. Performance timing may be limited."
fi

# Run main function
main "$@"