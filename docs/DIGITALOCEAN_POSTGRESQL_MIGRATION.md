# DigitalOcean PostgreSQL Migration Strategy

## Executive Summary

This document outlines the **strategic migration approach** for re-platforming the SEP trading system's **data persistence layer** from the current PostgreSQL implementation to a provisioned DigitalOcean managed PostgreSQL instance. The migration emphasizes **security hardening**, **operational readiness**, and **systemic integration** while maintaining **data integrity** and **system availability**.

## Current State Analysis

### Existing Infrastructure
- **Database Engine**: PostgreSQL (current version determined from codebase analysis)
- **SSL Certificate**: Custom certificate located at [`pki/postgresql-cert.pem`](pki/postgresql-cert.pem)
- **Connection Management**: Environment variable-driven configuration
- **Credential Storage**: `.env` files (excluded from version control per [`.gitignore`](.gitignore))

### Configuration Discovery
- **Primary Configuration Files**: `OANDA.env`, `frontend/.env`, `.sep-config.env`
- **Database Parameters**: No hardcoded database credentials in JSON configuration files
- **Environment Examples**: `.sep-config.env.example`, `frontend/.env.example` (referenced but not found in repository)

## Target Architecture: DigitalOcean Managed PostgreSQL

### Core Requirements
- **SSL/TLS Encryption**: Mandatory for all connections (`sslmode=require` minimum)
- **Default Port**: 25060 (DigitalOcean managed PostgreSQL standard)
- **Certificate Authority**: DigitalOcean-provided CA certificate (downloadable from cluster overview)
- **Connection Pooling**: PgBouncer-based connection pools for performance optimization

### Security Framework
- **Encryption at Rest**: LUKS encryption (managed by DigitalOcean)
- **Encryption in Transit**: SSL/TLS with certificate verification
- **Network Security**: Firewall rules and trusted sources configuration
- **TLS Protocol Support**: TLS 1.2+ (TLS 1.0/1.1 deprecated)

## Migration Strategy

### Phase 1: Infrastructure Provisioning
1. **Provision DigitalOcean PostgreSQL Cluster**
   - Select appropriate cluster size based on current workload analysis
   - Configure in target region for optimal latency
   - Enable automated backups and maintenance windows

2. **Download Security Certificates**
   ```bash
   # Download CA certificate from DigitalOcean control panel
   # Store in secure location within infrastructure
   ```

3. **Configure Network Security**
   - Set up firewall rules for trusted sources
   - Define IP whitelist for application servers
   - Configure VPC networking if applicable

### Phase 2: Connection String Migration

#### Current Connection Pattern (Inferred)
```bash
# Existing local PostgreSQL connection
postgresql://user:password@localhost:5432/database_name
```

#### Target DigitalOcean Connection Pattern
```bash
# DigitalOcean managed PostgreSQL with SSL
postgresql://doadmin:password@cluster-do-user-xxxx-0.db.ondigitalocean.com:25060/defaultdb?sslmode=require
```

#### Enhanced Security Connection (Recommended)
```bash
# With certificate verification
postgresql://doadmin:password@cluster-host:25060/defaultdb?sslmode=verify-full&sslrootcert=/path/to/ca-certificate.crt
```

### Phase 3: Configuration Updates

#### Environment Variable Schema
```bash
# Core Database Connection
DB_HOST=cluster-do-user-xxxx-0.db.ondigitalocean.com
DB_PORT=25060
DB_USER=doadmin
DB_PASSWORD=<secure_password>
DB_NAME=defaultdb
DB_SSL_MODE=require
DB_SSL_CERT=/path/to/ca-certificate.crt

# Complete Connection String
DATABASE_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}?sslmode=${DB_SSL_MODE}&sslrootcert=${DB_SSL_CERT}
```

#### Connection Pool Configuration (Recommended)
```bash
# Connection Pool Variables
DB_POOL_NAME=sep-trading-pool
DB_POOL_MODE=transaction
DB_POOL_SIZE=20
DB_POOL_URL=postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:25060/defaultdb?sslmode=require
```

## Required Code Changes

### 1. SSL Certificate Management
- **Update Certificate Path**: Replace [`pki/postgresql-cert.pem`](pki/postgresql-cert.pem) with DigitalOcean CA certificate
- **Certificate Validation**: Implement certificate verification in connection logic
- **Certificate Rotation**: Plan for certificate renewal and rotation procedures

### 2. Connection String Abstraction
- **Environment Variable Integration**: Ensure all database connections use environment variables
- **SSL Parameter Addition**: Add SSL mode and certificate parameters to connection strings
- **Port Migration**: Update hardcoded port references from 5432 to 25060

### 3. Connection Pool Implementation
```python
# Example connection pool configuration
import psycopg2.pool
from psycopg2.extras import RealDictCursor

# Create connection pool with SSL
connection_pool = psycopg2.pool.ThreadedConnectionPool(
    minconn=1,
    maxconn=20,
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT', 25060),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    sslmode='require',
    sslrootcert=os.getenv('DB_SSL_CERT'),
    cursor_factory=RealDictCursor
)
```

## Security Considerations

### 1. Credential Management
- **Secure Storage**: Use secure secret management for database credentials
- **Environment Segregation**: Separate credentials for development, staging, production
- **Access Control**: Implement least-privilege access principles

### 2. Network Security
- **Firewall Configuration**: Restrict database access to application servers only
- **VPC Integration**: Consider VPC networking for enhanced security
- **IP Whitelisting**: Maintain strict IP address access control

### 3. SSL/TLS Configuration
- **Certificate Verification**: Use `sslmode=verify-full` for production environments
- **Protocol Enforcement**: Ensure TLS 1.2+ for all connections
- **Certificate Monitoring**: Implement certificate expiration monitoring

## Connection Pool Strategy

### Benefits
- **Resource Optimization**: Efficient connection management
- **Performance Enhancement**: Reduced connection overhead
- **Scalability**: Support for high-concurrency workloads

### Configuration Recommendations
```json
{
  "connection_pools": [
    {
      "name": "sep-primary-pool",
      "mode": "transaction",
      "size": 25,
      "database": "defaultdb",
      "user": "doadmin",
      "priority": "high"
    },
    {
      "name": "sep-readonly-pool",
      "mode": "session",
      "size": 15,
      "database": "defaultdb",
      "user": "readonly_user",
      "priority": "medium"
    }
  ]
}
```

## Testing Strategy

### 1. Connection Verification
- **SSL Connection Test**: Verify SSL handshake and certificate validation
- **Performance Baseline**: Establish connection latency and throughput metrics
- **Failover Testing**: Test connection resilience and reconnection logic

### 2. Data Integrity Validation
- **Migration Verification**: Compare data checksums before and after migration
- **Transaction Testing**: Verify ACID properties in new environment
- **Concurrent Access**: Test multi-user access patterns

### 3. Security Validation
- **Certificate Verification**: Confirm SSL certificate chain validation
- **Access Control**: Test firewall rules and access restrictions
- **Credential Security**: Validate secure credential handling

## Migration Timeline

### Pre-Migration (Days 1-3)
- Provision DigitalOcean PostgreSQL cluster
- Configure security settings and firewall rules
- Download and integrate SSL certificates
- Update configuration templates

### Migration Execution (Day 4)
- Perform data migration using `pg_dump` and `pg_restore`
- Update application configuration
- Implement connection pool configuration
- Execute connection verification tests

### Post-Migration (Days 5-7)
- Monitor performance metrics
- Validate data integrity
- Conduct security audits
- Document operational procedures

## Risk Mitigation

### 1. Data Loss Prevention
- **Backup Strategy**: Full backup before migration initiation
- **Rollback Plan**: Detailed rollback procedures if migration fails
- **Data Validation**: Comprehensive data integrity checks

### 2. Downtime Minimization
- **Connection Pool Strategy**: Implement connection pooling for performance
- **Phased Migration**: Consider staged migration for large datasets
- **Health Monitoring**: Real-time monitoring during migration

### 3. Security Assurance
- **Certificate Validation**: Strict SSL certificate verification
- **Access Auditing**: Log and monitor all database access
- **Encryption Verification**: Confirm end-to-end encryption

## Success Criteria

1. **Functional**: All database operations function correctly with new connection parameters
2. **Performance**: Connection latency within acceptable thresholds
3. **Security**: SSL/TLS encryption verified and certificate validation successful
4. **Operational**: Monitoring and alerting systems properly configured
5. **Documentation**: Complete operational runbooks and troubleshooting guides

## Next Steps

1. **Configuration Template Creation**: Develop standardized `.env` templates for all environments
2. **Code Implementation**: Update database connection logic with SSL support
3. **Testing Framework**: Implement automated testing for database connectivity
4. **Operational Procedures**: Document monitoring, backup, and maintenance procedures

---

**Document Status**: Draft  
**Last Updated**: 2025-08-23  
**Migration Phase**: Planning  
**Security Classification**: Internal Use Only