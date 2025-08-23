#!/bin/bash

# SEP Trading System - DigitalOcean PostgreSQL Connection Test Script
# ===================================================================
#
# This script tests the connection to DigitalOcean managed PostgreSQL
# with SSL/TLS verification and provides detailed diagnostics.
#
# Usage:
#   ./scripts/test-db-connection.sh [config_file]
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
        echo "Please create the configuration file or run: ./scripts/setup-digitalocean-config.sh"
        return 1
    fi
    
    # Source configuration
    set -a
    source "$CONFIG_FILE"
    set +a
    
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
        echo "Please download the DigitalOcean CA certificate:"
        echo "1. Go to DigitalOcean Control Panel > Databases"
        echo "2. Click your cluster > Overview > Download CA certificate"
        echo "3. Save as: $DB_SSL_ROOT_CERT"
        return 1
    fi
    
    # Validate certificate format
    if ! openssl x509 -in "$DB_SSL_ROOT_CERT" -text -noout &>/dev/null; then
        log_error "SSL certificate file is invalid or corrupted: $DB_SSL_ROOT_CERT"
        return 1
    fi
    
    # Check certificate expiration
    local expiry_date
    expiry_date=$(openssl x509 -in "$DB_SSL_ROOT_CERT" -noout -enddate | cut -d= -f2)
    local expiry_epoch
    expiry_epoch=$(date -d "$expiry_date" +%s)
    local current_epoch
    current_epoch=$(date +%s)
    local days_until_expiry=$(( (expiry_epoch - current_epoch) / 86400 ))
    
    if [ $days_until_expiry -lt 0 ]; then
        log_error "SSL certificate has expired on: $expiry_date"
        return 1
    elif [ $days_until_expiry -lt 30 ]; then
        log_warning "SSL certificate expires in $days_until_expiry days: $expiry_date"
    fi
    
    log_success "SSL certificate validation passed (expires: $expiry_date)"
}

# =================================================================
# NETWORK CONNECTIVITY TESTS
# =================================================================

test_network_connectivity() {
    log_test "Testing network connectivity to $DB_HOST:$DB_PORT"
    
    # Test basic network connectivity
    if ! timeout 10 bash -c "exec 3<>/dev/tcp/$DB_HOST/$DB_PORT" 2>/dev/null; then
        log_error "Cannot connect to $DB_HOST:$DB_PORT"
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
    log_test "Testing basic PostgreSQL connection"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require} --set=sslrootcert=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require}"
    fi
    
    if ! PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" $ssl_args -c "SELECT 1;" &>/dev/null; then
        log_error "Basic database connection failed"
        echo "Common issues:"
        echo "- Incorrect credentials"
        echo "- Database cluster not running"
        echo "- IP address not in trusted sources"
        echo "- SSL configuration issues"
        return 1
    fi
    
    log_success "Basic database connection test passed"
}

test_ssl_connection() {
    log_test "Testing SSL/TLS connection security"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require} --set=sslrootcert=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require}"
    fi
    
    # Test SSL connection with certificate verification
    local ssl_info
    if ssl_info=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" $ssl_args -c "SELECT ssl_version();" -t 2>/dev/null | xargs); then
        log_success "SSL connection established with protocol: $ssl_info"
    else
        log_error "SSL connection test failed"
        return 1
    fi
    
    # Test SSL certificate verification (if verify-full mode)
    if [ "${DB_SSL_MODE:-}" = "verify-full" ]; then
        log_test "Testing SSL certificate verification (verify-full mode)"
        if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" --set=sslmode=verify-full --set=sslrootcert="$DB_SSL_ROOT_CERT" -c "SELECT 1;" &>/dev/null; then
            log_success "SSL certificate verification passed"
        else
            log_error "SSL certificate verification failed"
            return 1
        fi
    fi
}

test_database_version() {
    log_test "Retrieving database version and connection info"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require} --set=sslrootcert=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require}"
    fi
    
    local version_info
    version_info=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" $ssl_args -c "SELECT version();" -t 2>/dev/null | head -1 | xargs)
    
    if [ -n "$version_info" ]; then
        log_success "Database version: $version_info"
    else
        log_error "Could not retrieve database version"
        return 1
    fi
    
    # Get connection count and limits
    local connection_info
    connection_info=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" $ssl_args -c "SELECT current_setting('max_connections') as max_conn, count(*) as current_conn FROM pg_stat_activity;" -t 2>/dev/null)
    
    if [ -n "$connection_info" ]; then
        echo "Connection Info: $connection_info"
    fi
}

# =================================================================
# CONNECTION POOL TESTS
# =================================================================

test_connection_pool() {
    if [ "${DB_POOL_ENABLED:-false}" != "true" ]; then
        log_info "Connection pooling not enabled, skipping pool tests"
        return 0
    fi
    
    log_test "Testing connection pool configuration"
    
    # Test if PgBouncer or connection pooling is available
    # This would need to be implemented based on your specific setup
    log_info "Connection pool testing would be implemented based on your pooling setup"
    
    # Example test for DigitalOcean connection pools
    if [ -n "${DATABASE_POOL_URL:-}" ]; then
        log_info "Connection pool URL configured: ${DATABASE_POOL_URL//$DB_PASSWORD/****}"
    fi
}

# =================================================================
# PERFORMANCE TESTS
# =================================================================

test_connection_performance() {
    log_test "Testing connection performance"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require} --set=sslrootcert=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require}"
    fi
    
    # Measure connection time
    local start_time end_time duration
    start_time=$(date +%s.%N)
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" $ssl_args -c "SELECT pg_sleep(0.1);" &>/dev/null; then
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
    echo "SEP Trading System - DigitalOcean PostgreSQL Connection Test"
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
    test_ssl_connection
    test_database_version
    test_connection_pool
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
        log_success "All tests passed! Database connection is ready for production."
        exit 0
    else
        log_error "Some tests failed. Please resolve the issues before proceeding."
        exit 1
    fi
}

# Check requirements
if ! command -v psql &> /dev/null; then
    log_error "psql client is not installed. Please install PostgreSQL client tools."
    exit 1
fi

if ! command -v openssl &> /dev/null; then
    log_warning "openssl is not available. SSL certificate validation will be limited."
fi

# Run main function
main "$@"