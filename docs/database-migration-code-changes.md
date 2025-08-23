# SEP Database Migration - Code Changes Documentation

## Overview
This document outlines the specific code changes required to migrate from local PostgreSQL to DigitalOcean managed PostgreSQL with SSL/TLS support.

## Current Implementation Analysis

### C++ Database Connections
- **Primary File**: [`src/core/remote_data_manager.cpp`](src/core/remote_data_manager.cpp)
- **Configuration**: [`src/core/remote_data_manager.hpp`](src/core/remote_data_manager.hpp)
- **Library**: Uses libpqxx for PostgreSQL connections

### Python Database Connections
- **Primary File**: [`scripts/trading_service.py`](scripts/trading_service.py)
- **Setup Scripts**: Various deployment scripts using psycopg2
- **Library**: Uses psycopg2 for PostgreSQL connections

## Required Code Changes

### 1. Update C++ Configuration Structure

**File**: `src/core/remote_data_manager.hpp`

**Current DataSyncConfig** (lines 14-26):
```cpp
struct DataSyncConfig {
    std::string remote_host = "localhost";
    int remote_port = 5432;
    std::string db_name = "sep_trading";
    std::string redis_host = "localhost";
    int redis_port = 6380;
    std::string data_path = "/opt/sep-data";
    
    // Local cache settings
    std::string local_cache_path = ".cache/remote_data";
    int cache_ttl_hours = 24;
    bool enable_compression = true;
};
```

**Required Update**:
```cpp
struct DataSyncConfig {
    std::string remote_host = "localhost";
    int remote_port = 5432;
    std::string db_name = "sep_trading";
    std::string db_user = "sep_user";
    std::string db_password = "";
    
    // SSL/TLS Configuration for DigitalOcean managed PostgreSQL
    std::string ssl_mode = "require";           // require, verify-ca, verify-full
    std::string ssl_root_cert = "";             // Path to CA certificate
    std::string ssl_cert = "";                  // Client certificate (optional)
    std::string ssl_key = "";                   // Client private key (optional)
    bool ssl_compression = true;                // SSL compression
    std::string ssl_min_protocol_version = "TLSv1.2";  // Minimum TLS version
    
    std::string redis_host = "localhost";
    int redis_port = 6380;
    std::string data_path = "/opt/sep-data";
    
    // Local cache settings
    std::string local_cache_path = ".cache/remote_data";
    int cache_ttl_hours = 24;
    bool enable_compression = true;
    
    // Connection pool settings
    int max_connections = 20;
    int connection_timeout = 30;
    int command_timeout = 60;
};
```

### 2. Update C++ Connection String Builder

**File**: `src/core/remote_data_manager.cpp`

**Current Implementation** (lines 235-238):
```cpp
std::string build_connection_string() {
    return fmt::format("host={} port={} dbname={} user=sep_user password=sep_password",
        config_.remote_host, config_.remote_port, config_.db_name);
}
```

**Required Update**:
```cpp
std::string build_connection_string() {
    std::string conn_str = fmt::format(
        "host={} port={} dbname={} user={} password={} connect_timeout={}",
        config_.remote_host, config_.remote_port, config_.db_name,
        config_.db_user, config_.db_password, config_.connection_timeout
    );
    
    // Add SSL parameters for DigitalOcean managed PostgreSQL
    if (!config_.ssl_mode.empty()) {
        conn_str += fmt::format(" sslmode={}", config_.ssl_mode);
    }
    
    if (!config_.ssl_root_cert.empty()) {
        conn_str += fmt::format(" sslrootcert={}", config_.ssl_root_cert);
    }
    
    if (!config_.ssl_cert.empty()) {
        conn_str += fmt::format(" sslcert={}", config_.ssl_cert);
    }
    
    if (!config_.ssl_key.empty()) {
        conn_str += fmt::format(" sslkey={}", config_.ssl_key);
    }
    
    if (config_.ssl_compression) {
        conn_str += " sslcompression=1";
    }
    
    if (!config_.ssl_min_protocol_version.empty()) {
        conn_str += fmt::format(" ssl_min_protocol_version={}", config_.ssl_min_protocol_version);
    }
    
    return conn_str;
}
```

### 3. Add SSL Certificate Validation

**File**: `src/core/remote_data_manager.cpp` (new method)

```cpp
bool validate_ssl_certificate(const DataSyncConfig& config) {
    if (config.ssl_mode == "disable") {
        return true; // No SSL validation needed
    }
    
    if (!config.ssl_root_cert.empty()) {
        std::ifstream cert_file(config.ssl_root_cert);
        if (!cert_file.good()) {
            auto logger = ::sep::logging::Manager::getInstance().getLogger("database");
            if (logger) {
                logger->error("SSL root certificate not found: {}", config.ssl_root_cert);
            }
            return false;
        }
        
        // Basic PEM format validation
        std::string line;
        bool found_begin = false, found_end = false;
        while (std::getline(cert_file, line)) {
            if (line.find("-----BEGIN CERTIFICATE-----") != std::string::npos) {
                found_begin = true;
            }
            if (line.find("-----END CERTIFICATE-----") != std::string::npos) {
                found_end = true;
            }
        }
        
        if (!found_begin || !found_end) {
            auto logger = ::sep::logging::Manager::getInstance().getLogger("database");
            if (logger) {
                logger->error("Invalid SSL certificate format: {}", config.ssl_root_cert);
            }
            return false;
        }
    }
    
    return true;
}
```

### 4. Update Connection Test Method

**File**: `src/core/remote_data_manager.cpp` (modify existing test_connection)

**Current Implementation** (lines 195-210):
```cpp
bool test_connection() {
    try {
        // Test PostgreSQL
        pqxx::connection conn(build_connection_string());
        pqxx::work txn(conn);
        // ... existing code ...
    }
    // ... error handling ...
}
```

**Required Update**:
```cpp
bool test_connection() {
    auto logger = ::sep::logging::Manager::getInstance().getLogger("database");
    
    // Validate SSL configuration first
    if (!validate_ssl_certificate(config_)) {
        if (logger) logger->error("SSL certificate validation failed");
        return false;
    }
    
    try {
        // Test PostgreSQL with SSL
        std::string conn_str = build_connection_string();
        if (logger) {
            // Log connection string without password
            std::string safe_conn_str = conn_str;
            size_t pwd_pos = safe_conn_str.find("password=");
            if (pwd_pos != std::string::npos) {
                size_t space_pos = safe_conn_str.find(" ", pwd_pos);
                if (space_pos != std::string::npos) {
                    safe_conn_str.replace(pwd_pos, space_pos - pwd_pos, "password=****");
                } else {
                    safe_conn_str.replace(pwd_pos, std::string::npos, "password=****");
                }
            }
            logger->info("Testing database connection: {}", safe_conn_str);
        }
        
        pqxx::connection conn(conn_str);
        pqxx::work txn(conn);
        
        // Test SSL connection details
        auto ssl_result = txn.exec("SELECT ssl_version(), ssl_cipher()");
        if (!ssl_result.empty()) {
            auto row = ssl_result[0];
            if (logger) {
                logger->info("SSL connection established - Version: {}, Cipher: {}",
                    row[0].c_str(), row[1].c_str());
            }
        }
        
        auto version_result = txn.exec("SELECT version()");
        if (!version_result.empty() && logger) {
            logger->info("Connected to PostgreSQL: {}", version_result[0][0].c_str());
        }
        
        txn.commit();
        return true;
        
    } catch (const pqxx::sql_error& e) {
        if (logger) logger->error("Database SQL error: {}", e.what());
        return false;
    } catch (const pqxx::broken_connection& e) {
        if (logger) logger->error("Database connection failed: {}", e.what());
        return false;
    } catch (const std::exception& e) {
        if (logger) logger->error("Database connection error: {}", e.what());
        return false;
    }
}
```

### 5. Python Script Updates

**Files**: All Python scripts using psycopg2

**Required Function** (add to utility module):
```python
import os
import ssl
import psycopg2
from urllib.parse import quote_plus

def build_postgresql_connection_params():
    """Build PostgreSQL connection parameters with SSL support for DigitalOcean."""
    
    # Read from environment variables
    params = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'sep_trading'),
        'user': os.getenv('DB_USER', 'sep_user'),
        'password': os.getenv('DB_PASSWORD', ''),
        'connect_timeout': int(os.getenv('DB_CONNECT_TIMEOUT', 30)),
        'application_name': 'SEP-Trading-System'
    }
    
    # SSL Configuration for DigitalOcean managed PostgreSQL
    ssl_mode = os.getenv('DB_SSL_MODE', 'require')
    if ssl_mode != 'disable':
        params['sslmode'] = ssl_mode
        
        # Add SSL certificate if provided
        ssl_root_cert = os.getenv('DB_SSL_ROOT_CERT', '')
        if ssl_root_cert and os.path.exists(ssl_root_cert):
            params['sslrootcert'] = ssl_root_cert
            
        # Optional client certificates
        ssl_cert = os.getenv('DB_SSL_CERT', '')
        ssl_key = os.getenv('DB_SSL_KEY', '')
        if ssl_cert and ssl_key:
            params['sslcert'] = ssl_cert
            params['sslkey'] = ssl_key
    
    return params

def create_postgresql_connection():
    """Create a PostgreSQL connection with SSL support."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        params = build_postgresql_connection_params()
        
        # Log connection attempt (without password)
        safe_params = {k: v for k, v in params.items() if k != 'password'}
        safe_params['password'] = '****' if params.get('password') else ''
        logger.info(f"Connecting to PostgreSQL: {safe_params}")
        
        connection = psycopg2.connect(**params)
        
        # Test SSL connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT ssl_version(), ssl_cipher();")
            ssl_info = cursor.fetchone()
            if ssl_info:
                logger.info(f"SSL connection established - Version: {ssl_info[0]}, Cipher: {ssl_info[1]}")
        
        return connection
        
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        raise
```

### 6. Configuration Template Updates

**File**: `config/.sep-config.production.env.template`

**Add SSL Configuration Section**:
```bash
# DigitalOcean Managed PostgreSQL Configuration
DB_HOST=your-cluster-host.db.ondigitalocean.com
DB_PORT=25060
DB_NAME=defaultdb
DB_USER=doadmin
DB_PASSWORD=your-secure-password

# SSL/TLS Configuration (Required for DigitalOcean managed PostgreSQL)
DB_SSL_MODE=require
DB_SSL_ROOT_CERT=/sep/config/ca-certificate.crt
DB_SSL_CERT=
DB_SSL_KEY=
DB_CONNECT_TIMEOUT=30

# Connection Pool Settings
DB_MAX_CONNECTIONS=20
DB_CONNECTION_TIMEOUT=30
DB_COMMAND_TIMEOUT=60
```

## Implementation Priority

1. **High Priority** (Core Functionality):
   - Update `DataSyncConfig` structure
   - Modify `build_connection_string()` method
   - Add SSL certificate validation
   - Update Python connection utilities

2. **Medium Priority** (Reliability):
   - Enhanced connection testing with SSL verification
   - Connection pool configuration
   - Logging improvements

3. **Low Priority** (Optimization):
   - Connection retry logic
   - Performance monitoring
   - Advanced SSL configuration options

## Testing Requirements

1. **SSL Certificate Validation**:
   - Test with valid DigitalOcean CA certificate
   - Test with invalid/missing certificates
   - Test certificate expiration handling

2. **Connection Security**:
   - Verify TLS version enforcement
   - Test SSL compression settings
   - Validate cipher suites

3. **Error Handling**:
   - Test connection timeouts
   - Test network failures
   - Test authentication failures

4. **Performance**:
   - Connection establishment time
   - SSL handshake overhead
   - Connection pool efficiency

## Migration Checklist

- [ ] Update C++ configuration structures
- [ ] Modify connection string builders
- [ ] Add SSL certificate validation
- [ ] Update Python connection utilities
- [ ] Create new configuration templates
- [ ] Update deployment scripts
- [ ] Test SSL connections
- [ ] Validate error handling
- [ ] Performance testing
- [ ] Documentation updates

## Security Considerations

1. **Certificate Management**:
   - Secure storage of CA certificate
   - Certificate rotation procedures
   - Validation of certificate chains

2. **Connection Security**:
   - Enforce minimum TLS version (1.2+)
   - Disable weak cipher suites
   - Enable SSL compression only when needed

3. **Credential Protection**:
   - Environment variable security
   - Avoid logging passwords
   - Secure configuration file permissions

## Rollback Plan

1. **Configuration Rollback**:
   - Keep backup of working configuration
   - Environment variable restore
   - Certificate file restoration

2. **Code Rollback**:
   - Git branch for migration changes
   - Feature flag for SSL enable/disable
   - Fallback to local PostgreSQL if needed