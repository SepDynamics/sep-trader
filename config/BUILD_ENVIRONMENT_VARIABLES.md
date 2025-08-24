# SEP Engine Build Environment Variables Reference

This document provides a comprehensive reference for environment variables that control the SEP Engine build system and component behavior, eliminating hardcoded configurations for maximum deployment flexibility.

## Build System Configuration

### Compiler Configuration

- **`SEP_GCC_PATH`**
  - **Purpose**: Override default compiler installation path
  - **Default**: `/usr/bin`
  - **Usage**: Points to directory containing `gcc-14` and `g++-14` executables
  - **Example**: `SEP_GCC_PATH=/opt/gcc-14/bin`

- **`CUDA_HOME`**
  - **Purpose**: CUDA toolkit installation directory (existing)
  - **Required**: Yes (when CUDA support enabled)
  - **Example**: `CUDA_HOME=/usr/local/cuda-12.6`

### Directory Configuration

- **`SEP_WORKSPACE_PATH`**
  - **Purpose**: SEP workspace root directory (existing)
  - **Default**: `${CMAKE_CURRENT_SOURCE_DIR}`
  - **Usage**: Base directory for all SEP operations

- **`SEP_CACHE_DIR`**
  - **Purpose**: Cache directory for temporary build artifacts
  - **Default**: `${CMAKE_SOURCE_DIR}/cache/`
  - **Example**: `SEP_CACHE_DIR=/tmp/sep-cache`

- **`SEP_CONFIG_DIR`**
  - **Purpose**: Configuration files directory
  - **Default**: `${CMAKE_SOURCE_DIR}/config/`
  - **Example**: `SEP_CONFIG_DIR=/etc/sep/config`

- **`SEP_LOG_DIR`**
  - **Purpose**: Log files directory
  - **Default**: `${CMAKE_SOURCE_DIR}/logs/`
  - **Example**: `SEP_LOG_DIR=/var/log/sep`

### Runtime Library Configuration

- **`SEP_LIB_PATH`**
  - **Purpose**: Runtime library search path for executables
  - **Default**: `${CMAKE_BINARY_DIR}/lib`
  - **Usage**: Controls RPATH/RUNPATH for all SEP executables
  - **Example**: `SEP_LIB_PATH=/usr/local/lib/sep`

## Datastore Configuration

### Valkey/Redis Configuration

- **`VALKEY_HOST`**
  - **Purpose**: Valkey server hostname or IP address
  - **Default**: `localhost`
  - **Example**: `VALKEY_HOST=valkey.production.internal`

- **`VALKEY_PORT`**
  - **Purpose**: Valkey server port
  - **Default**: `6380`
  - **Example**: `VALKEY_PORT=6379`

- **`VALKEY_PASSWORD`**
  - **Purpose**: Valkey server authentication password
  - **Required**: Yes (in production)
  - **Example**: `VALKEY_PASSWORD=secure_random_password`

- **`VALKEY_DB_INDEX`**
  - **Purpose**: Valkey database index for SEP data isolation
  - **Default**: `0`
  - **Example**: `VALKEY_DB_INDEX=1`

### Database Configuration

- **`DB_HOST`**
  - **Purpose**: PostgreSQL database hostname
  - **Required**: Yes
  - **Example**: `DB_HOST=postgres.production.internal`

- **`DB_PORT`**
  - **Purpose**: PostgreSQL database port
  - **Default**: `5432`

- **`DB_NAME`**
  - **Purpose**: PostgreSQL database name
  - **Required**: Yes
  - **Example**: `DB_NAME=sep_trading_production`

- **`DB_USER`**
  - **Purpose**: PostgreSQL database username
  - **Required**: Yes
  - **Example**: `DB_USER=sep_trader`

- **`DB_PASSWORD`**
  - **Purpose**: PostgreSQL database password
  - **Required**: Yes
  - **Example**: `DB_PASSWORD=secure_db_password`

## Trading API Configuration

### OANDA Configuration

- **`OANDA_API_BASE_URL`**
  - **Purpose**: OANDA API endpoint (practice vs live)
  - **Required**: Yes
  - **Example**: `OANDA_API_BASE_URL=https://api-fxpractice.oanda.com`

- **`OANDA_ACCESS_TOKEN`**
  - **Purpose**: OANDA API authentication token
  - **Required**: Yes
  - **Example**: `OANDA_ACCESS_TOKEN=your_api_token_here`

- **`OANDA_ACCOUNT_ID`**
  - **Purpose**: OANDA trading account identifier
  - **Required**: Yes
  - **Example**: `OANDA_ACCOUNT_ID=101-123-456789-001`

## Usage Examples

### Development Environment
```bash
export SEP_WORKSPACE_PATH=/home/dev/sep
export SEP_CACHE_DIR=/tmp/sep-dev-cache
export SEP_LOG_DIR=/tmp/sep-dev-logs
export VALKEY_HOST=localhost
export VALKEY_PORT=6380
```

### Production Environment
```bash
export SEP_GCC_PATH=/opt/gcc-14/bin
export SEP_CONFIG_DIR=/etc/sep/config
export SEP_LOG_DIR=/var/log/sep
export SEP_LIB_PATH=/usr/local/lib/sep
export VALKEY_HOST=valkey.prod.internal
export VALKEY_PORT=6379
export VALKEY_PASSWORD=${VALKEY_PROD_PASSWORD}
```

### Docker Environment
```bash
export SEP_CACHE_DIR=/app/cache
export SEP_CONFIG_DIR=/app/config
export SEP_LOG_DIR=/app/logs
export SEP_LIB_PATH=/app/lib
```

## Migration Notes

This environment-driven configuration replaces the following hardcoded values:

- **Compiler paths**: `/usr/bin/gcc-14`, `/usr/bin/g++-14` → `${SEP_GCC_PATH}/{gcc-14,g++-14}`
- **Directory paths**: `CMAKE_SOURCE_DIR/{cache,config,logs}` → `SEP_{CACHE,CONFIG,LOG}_DIR`
- **Library paths**: `gnu_14.2_cxx20_64_release` → `${SEP_LIB_PATH}`
- **Datastore endpoints**: `localhost:6379` → `${VALKEY_HOST}:${VALKEY_PORT}`

## Validation

Before building or running SEP Engine components:

1. **Verify compiler availability**: `ls -la ${SEP_GCC_PATH:-/usr/bin}/{gcc-14,g++-14}`
2. **Check directory permissions**: Ensure write access to cache and log directories
3. **Test datastore connectivity**: `redis-cli -h ${VALKEY_HOST} -p ${VALKEY_PORT} ping`
4. **Validate CUDA environment**: `${CUDA_HOME}/bin/nvcc --version` (if using CUDA)