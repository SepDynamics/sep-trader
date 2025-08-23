#!/bin/bash

# SEP Trading System - DigitalOcean PostgreSQL Configuration Setup Script
# ========================================================================
#
# This script automates the setup of DigitalOcean PostgreSQL configuration
# files and SSL certificates for the SEP trading system.
#
# Usage:
#   ./scripts/setup-digitalocean-config.sh [environment]
#
# Arguments:
#   environment: development, staging, production (default: development)
#
# Requirements:
#   - curl (for downloading certificates)
#   - openssl (for certificate validation)
#   - psql (for connection testing)
#   - doctl (optional, for DigitalOcean CLI operations)

set -euo pipefail

# =================================================================
# CONFIGURATION VARIABLES
# =================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENVIRONMENT="${1:-development}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration paths
CONFIG_DIR="$PROJECT_ROOT/config"
PKI_DIR="$PROJECT_ROOT/pki"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# =================================================================
# UTILITY FUNCTIONS
# =================================================================

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

check_requirements() {
    log_info "Checking system requirements..."
    
    local missing_tools=()
    
    if ! command -v curl &> /dev/null; then
        missing_tools+=("curl")
    fi
    
    if ! command -v openssl &> /dev/null; then
        missing_tools+=("openssl")
    fi
    
    if ! command -v psql &> /dev/null; then
        missing_tools+=("psql")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        log_info "Please install missing tools and run again."
        exit 1
    fi
    
    log_success "All required tools are available"
}

create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$PKI_DIR"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/scripts"
    
    # Set proper permissions for PKI directory
    chmod 700 "$PKI_DIR"
    
    log_success "Directories created successfully"
}

# =================================================================
# CONFIGURATION FILE SETUP
# =================================================================

setup_backend_config() {
    log_info "Setting up backend configuration for $ENVIRONMENT environment..."
    
    local config_file
    local template_file
    
    if [ "$ENVIRONMENT" = "production" ]; then
        template_file="$CONFIG_DIR/.sep-config.production.env.template"
        config_file="$CONFIG_DIR/.sep-config.env"
    else
        template_file="$CONFIG_DIR/.sep-config.env.template"
        config_file="$CONFIG_DIR/.sep-config.env"
    fi
    
    if [ ! -f "$template_file" ]; then
        log_error "Template file not found: $template_file"
        return 1
    fi
    
    if [ -f "$config_file" ]; then
        log_warning "Configuration file already exists: $config_file"
        read -p "Do you want to overwrite it? [y/N]: " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping backend configuration setup"
            return 0
        fi
    fi
    
    cp "$template_file" "$config_file"
    chmod 600 "$config_file"
    
    log_success "Backend configuration template copied to: $config_file"
    log_warning "Please edit $config_file with your actual DigitalOcean credentials"
}

setup_frontend_config() {
    log_info "Setting up frontend configuration..."
    
    local template_file="$FRONTEND_DIR/.env.template"
    local config_file="$FRONTEND_DIR/.env"
    
    if [ ! -f "$template_file" ]; then
        log_error "Frontend template file not found: $template_file"
        return 1
    fi
    
    if [ -f "$config_file" ]; then
        log_warning "Frontend configuration file already exists: $config_file"
        read -p "Do you want to overwrite it? [y/N]: " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping frontend configuration setup"
            return 0
        fi
    fi
    
    cp "$template_file" "$config_file"
    chmod 644 "$config_file"
    
    log_success "Frontend configuration template copied to: $config_file"
    log_warning "Please edit $config_file with your actual API endpoints"
}

# =================================================================
# SSL CERTIFICATE MANAGEMENT
# =================================================================

download_digitalocean_certificate() {
    log_info "Setting up SSL certificate download instructions..."
    
    local cert_file="$PKI_DIR/digitalocean-ca-certificate.crt"
    
    if [ -f "$cert_file" ]; then
        log_info "DigitalOcean CA certificate already exists: $cert_file"
        
        # Validate existing certificate
        if openssl x509 -in "$cert_file" -text -noout &>/dev/null; then
            log_success "Existing certificate is valid"
            return 0
        else
            log_warning "Existing certificate appears to be invalid"
        fi
    fi
    
    cat << 'EOF'

=================================================================
DIGITALOCEAN SSL CERTIFICATE DOWNLOAD INSTRUCTIONS
=================================================================

To download your DigitalOcean managed PostgreSQL SSL certificate:

1. Log in to the DigitalOcean Control Panel
2. Navigate to Databases
3. Click on your PostgreSQL cluster name
4. Go to the "Overview" tab
5. In the "Connection Details" section, click "Download CA certificate"
6. Save the certificate as: pki/digitalocean-ca-certificate.crt

Alternatively, if you have doctl installed and configured:

  doctl databases connection <your-cluster-id> --format CA_CERT > pki/digitalocean-ca-certificate.crt

=================================================================

EOF
    
    read -p "Press Enter after downloading the certificate to continue..." -r
    
    if [ ! -f "$cert_file" ]; then
        log_error "Certificate file not found: $cert_file"
        log_error "Please download the certificate before continuing"
        return 1
    fi
    
    # Validate downloaded certificate
    if ! openssl x509 -in "$cert_file" -text -noout &>/dev/null; then
        log_error "Downloaded certificate is invalid or corrupted"
        return 1
    fi
    
    chmod 600 "$cert_file"
    log_success "DigitalOcean CA certificate is ready: $cert_file"
}

# =================================================================
# DATABASE CONNECTION TESTING
# =================================================================

test_database_connection() {
    log_info "Testing database connection..."
    
    local config_file="$CONFIG_DIR/.sep-config.env"
    
    if [ ! -f "$config_file" ]; then
        log_error "Configuration file not found: $config_file"
        log_error "Please run configuration setup first"
        return 1
    fi
    
    # Source configuration
    set -a
    source "$config_file"
    set +a
    
    # Check required variables
    if [ -z "${DB_HOST:-}" ] || [ -z "${DB_PORT:-}" ] || [ -z "${DB_USER:-}" ] || [ -z "${DB_PASSWORD:-}" ]; then
        log_error "Database connection variables not properly configured"
        log_error "Please edit $config_file with your actual credentials"
        return 1
    fi
    
    # Test connection
    log_info "Attempting to connect to: $DB_HOST:$DB_PORT"
    
    local ssl_args=""
    if [ -n "${DB_SSL_ROOT_CERT:-}" ] && [ -f "$DB_SSL_ROOT_CERT" ]; then
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require} --set=sslrootcert=$DB_SSL_ROOT_CERT"
    else
        ssl_args="--set=sslmode=${DB_SSL_MODE:-require}"
    fi
    
    if PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "${DB_NAME:-defaultdb}" $ssl_args -c "SELECT version();" &>/dev/null; then
        log_success "Database connection successful!"
        
        # Get database version
        local db_version
        db_version=$(PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "${DB_NAME:-defaultdb}" $ssl_args -t -c "SELECT version();" 2>/dev/null | head -1 | xargs)
        log_info "Database version: $db_version"
    else
        log_error "Database connection failed!"
        log_error "Please check your configuration and network connectivity"
        return 1
    fi
}

# =================================================================
# DOCKER SETUP
# =================================================================

setup_docker_config() {
    log_info "Setting up Docker configuration..."
    
    local docker_template="$CONFIG_DIR/docker-compose.digitalocean.yml.template"
    local docker_config="$PROJECT_ROOT/docker-compose.digitalocean.yml"
    
    if [ ! -f "$docker_template" ]; then
        log_error "Docker template file not found: $docker_template"
        return 1
    fi
    
    if [ -f "$docker_config" ]; then
        log_warning "Docker configuration already exists: $docker_config"
        read -p "Do you want to overwrite it? [y/N]: " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Skipping Docker configuration setup"
            return 0
        fi
    fi
    
    cp "$docker_template" "$docker_config"
    log_success "Docker configuration copied to: $docker_config"
}

# =================================================================
# MAIN EXECUTION
# =================================================================

main() {
    echo "SEP Trading System - DigitalOcean PostgreSQL Configuration Setup"
    echo "================================================================="
    echo "Environment: $ENVIRONMENT"
    echo
    
    check_requirements
    create_directories
    setup_backend_config
    setup_frontend_config
    download_digitalocean_certificate
    setup_docker_config
    
    echo
    log_success "Configuration setup completed!"
    echo
    echo "Next steps:"
    echo "1. Edit configuration files with your actual credentials"
    echo "2. Test database connection: ./scripts/test-db-connection.sh"
    echo "3. Start services: docker-compose -f docker-compose.digitalocean.yml up"
    echo
    
    # Offer to test connection
    read -p "Would you like to test the database connection now? [y/N]: " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_database_connection
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi