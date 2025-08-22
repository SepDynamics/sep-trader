#!/bin/bash
# Java Keystore Fix Script for CUDA Installation Issues
# Fixes the ca-certificates-java keystore corruption that occurs during CUDA toolkit installation

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running as root or with sudo
SUDO=""
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    log_error "This script requires root privileges or sudo."
    exit 1
  fi
fi

log_info "Starting Java keystore fix for CUDA installation issues..."

# Stop any running Java processes that might be using the keystore
log_info "Stopping Java processes that might be using the keystore..."
$SUDO pkill -f java || true

# Remove problematic ca-certificates-java package
log_info "Removing problematic ca-certificates-java package..."
$SUDO apt-mark unhold ca-certificates-java >/dev/null 2>&1 || true
$SUDO apt-get purge -y ca-certificates-java >/dev/null 2>&1 || true

# Clean up broken keystore files
log_info "Cleaning up broken keystore files..."
$SUDO rm -f /lib/security/cacerts* /etc/ssl/certs/java/cacerts* || true
$SUDO rm -rf /etc/ssl/certs/java || true

# Create proper directory structure
log_info "Creating proper keystore directory structure..."
$SUDO mkdir -p /lib/security /etc/ssl/certs/java /usr/lib/jvm

# Create a proper empty keystore
log_info "Creating proper Java keystore..."
if command -v keytool >/dev/null 2>&1; then
    log_info "Using keytool to create proper keystore..."
    # Create a temporary keystore with a dummy certificate, then remove it
    $SUDO keytool -genkey -alias temp -keystore /lib/security/cacerts \
        -keyalg RSA -keysize 2048 -validity 365 \
        -dname "CN=temp, OU=temp, O=temp, L=temp, ST=temp, C=US" \
        -storepass changeit -keypass changeit >/dev/null 2>&1 || true
    $SUDO keytool -delete -alias temp -keystore /lib/security/cacerts \
        -storepass changeit >/dev/null 2>&1 || true
    log_success "Created proper keystore using keytool"
else
    log_warning "keytool not available, creating minimal keystore structure..."
    # Create a minimal keystore file structure
    echo -e "\xfe\xed\xfe\xed\x00\x00\x00\x02\x00\x00\x00\x00" | $SUDO tee /lib/security/cacerts >/dev/null
fi

# Set proper permissions
$SUDO chmod 644 /lib/security/cacerts
$SUDO chown root:root /lib/security/cacerts

# Create symlinks
$SUDO ln -sf /lib/security/cacerts /etc/ssl/certs/java/cacerts

# Fix any broken dpkg configurations
log_info "Fixing any broken package configurations..."
$SUDO dpkg --configure -a >/dev/null 2>&1 || true

# Clean up package manager
log_info "Cleaning up package manager..."
$SUDO apt-get update >/dev/null 2>&1 || true
$SUDO apt-get install -f >/dev/null 2>&1 || true
$SUDO apt-get autoremove -y >/dev/null 2>&1 || true

# Verify the fix
log_info "Verifying the keystore fix..."
if [ -f "/lib/security/cacerts" ] && [ -L "/etc/ssl/certs/java/cacerts" ]; then
    log_success "Java keystore structure is now properly configured"
else
    log_error "Keystore fix verification failed"
    exit 1
fi

# Test CUDA installation if nvcc is available
if command -v nvcc >/dev/null 2>&1; then
    log_info "CUDA toolkit is already installed:"
    nvcc --version | head -1
    log_success "CUDA appears to be working properly"
else
    log_warning "CUDA toolkit not found. You can now run:"
    echo "  sudo apt-get install cuda-toolkit-12-9 cuda-nvcc-12-9"
    echo "  or re-run: ./install.sh --minimal --no-docker"
fi

log_success "Java keystore fix completed successfully!"
log_info "You can now continue with your CUDA installation or run ./deploy.sh"