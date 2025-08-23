# SEP Database Migration - Testing Strategy

## Overview
This document outlines the comprehensive testing strategy for migrating from local PostgreSQL to DigitalOcean managed PostgreSQL with SSL/TLS support.

## Testing Phases

### Phase 1: Pre-Migration Validation
**Objective**: Ensure current system is stable before migration

#### 1.1 Current System Health Check
```bash
# Test current local PostgreSQL connections
./scripts/test-db-connection.sh config/.sep-config.env.template

# Verify current data integrity
sudo -u postgres psql sep_trading -c "SELECT COUNT(*) FROM training_data;"
sudo -u postgres psql sep_trading -c "SELECT COUNT(*) FROM model_states;"

# Check current system performance
./build/src/trading/quantum_pair_trainer test-connection
```

#### 1.2 Data Backup Verification
```bash
# Create pre-migration backup
pg_dump sep_trading > backup/pre_migration_$(date +%Y%m%d_%H%M%S).sql

# Verify backup integrity
psql -f backup/pre_migration_*.sql test_restore_db

# Test backup restoration time
time pg_restore backup/pre_migration_*.sql
```

#### 1.3 Current Configuration Validation
- Verify all environment variables are properly set
- Test C++ application startup with current configuration
- Validate Python service connections
- Check Redis connectivity and functionality

### Phase 2: SSL/TLS Configuration Testing

#### 2.1 Certificate Validation Tests
```bash
# Test DigitalOcean CA certificate download
./scripts/setup-digitalocean-config.sh

# Validate certificate format and expiry
openssl x509 -in config/ca-certificate.crt -text -noout
openssl x509 -in config/ca-certificate.crt -noout -dates

# Test certificate chain validation
openssl verify -CAfile config/ca-certificate.crt config/ca-certificate.crt
```

**Expected Results**:
- Certificate downloads successfully
- Certificate is valid PEM format
- Certificate expiry is > 30 days in future
- Certificate chain validates correctly

#### 2.2 SSL Connection Tests
```bash
# Test SSL connectivity to DigitalOcean managed PostgreSQL
PGPASSWORD=$DB_PASSWORD psql \
  -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME \
  --set=sslmode=require \
  --set=sslrootcert=config/ca-certificate.crt \
  -c "SELECT ssl_version(), ssl_cipher();"
```

**Test Matrix**:
| SSL Mode | Certificate | Expected Result |
|----------|-------------|-----------------|
| disable | N/A | Connection (insecure) |
| require | None | Connection (unverified) |
| require | Valid CA | Secure connection |
| verify-ca | Valid CA | Verified connection |
| verify-full | Valid CA + hostname | Fully verified |

#### 2.3 Connection Security Validation
```python
# Python SSL connection test
import ssl
import psycopg2

def test_ssl_connection_security():
    """Test SSL connection security parameters."""
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST'),
        port=os.getenv('DB_PORT'),
        database=os.getenv('DB_NAME'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        sslmode='require',
        sslrootcert='config/ca-certificate.crt'
    )
    
    with conn.cursor() as cur:
        # Verify SSL version
        cur.execute("SELECT ssl_version();")
        ssl_version = cur.fetchone()[0]
        assert ssl_version.startswith('TLSv1.2') or ssl_version.startswith('TLSv1.3')
        
        # Verify SSL cipher
        cur.execute("SELECT ssl_cipher();")
        cipher = cur.fetchone()[0]
        assert cipher is not None and len(cipher) > 0
        
        # Test connection encryption
        cur.execute("SELECT pg_ssl_is_used();")
        is_ssl = cur.fetchone()[0]
        assert is_ssl is True
```

### Phase 3: Application Integration Testing

#### 3.1 C++ Application Tests
```cpp
// Test cases for C++ application
class DatabaseMigrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize with DigitalOcean configuration
        config_.remote_host = std::getenv("DB_HOST");
        config_.remote_port = std::stoi(std::getenv("DB_PORT"));
        config_.db_name = std::getenv("DB_NAME");
        config_.db_user = std::getenv("DB_USER");
        config_.db_password = std::getenv("DB_PASSWORD");
        config_.ssl_mode = "require";
        config_.ssl_root_cert = "config/ca-certificate.crt";
    }
    
    sep::trading::DataSyncConfig config_;
};

TEST_F(DatabaseMigrationTest, SSLConnectionEstablishment) {
    sep::trading::RemoteDataManager manager(config_);
    EXPECT_TRUE(manager.test_connection());
}

TEST_F(DatabaseMigrationTest, TrainingDataRetrieval) {
    sep::trading::RemoteDataManager manager(config_);
    auto start = std::chrono::system_clock::now() - std::chrono::hours(24);
    auto end = std::chrono::system_clock::now();
    
    auto future = manager.fetch_training_data("EUR_USD", start, end);
    auto data = future.get();
    
    EXPECT_GT(data.size(), 0);
}

TEST_F(DatabaseMigrationTest, ModelUploadDownload) {
    sep::trading::RemoteDataManager manager(config_);
    
    // Create test model
    sep::trading::ModelState test_model;
    test_model.model_id = "test_migration_model";
    test_model.pair = "EUR_USD";
    test_model.accuracy = 0.85;
    
    // Upload model
    auto upload_future = manager.upload_model(test_model);
    EXPECT_TRUE(upload_future.get());
    
    // Download model
    auto download_future = manager.download_latest_model("EUR_USD");
    auto downloaded_model = download_future.get();
    EXPECT_EQ(downloaded_model.model_id, test_model.model_id);
}
```

#### 3.2 Python Service Integration Tests
```python
import unittest
import psycopg2
from scripts.trading_service import create_postgresql_connection

class DatabaseMigrationPythonTest(unittest.TestCase):
    
    def test_ssl_connection_creation(self):
        """Test PostgreSQL connection with SSL."""
        conn = create_postgresql_connection()
        self.assertIsNotNone(conn)
        
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            result = cur.fetchone()
            self.assertEqual(result[0], 1)
    
    def test_trading_data_operations(self):
        """Test trading data CRUD operations."""
        conn = create_postgresql_connection()
        
        with conn.cursor() as cur:
            # Test INSERT
            cur.execute("""
                INSERT INTO training_data (pair, timestamp, features, target, metadata)
                VALUES (%s, %s, %s, %s, %s);
            """, ("TEST_PAIR", "2024-01-01", [1.0, 2.0], 1.5, "{}"))
            
            # Test SELECT
            cur.execute("SELECT * FROM training_data WHERE pair = %s;", ("TEST_PAIR",))
            result = cur.fetchone()
            self.assertIsNotNone(result)
            
            # Cleanup
            cur.execute("DELETE FROM training_data WHERE pair = %s;", ("TEST_PAIR",))
        
        conn.commit()
```

#### 3.3 End-to-End System Tests
```bash
#!/bin/bash
# E2E system test script

set -e

echo "🧪 Starting end-to-end migration tests..."

# 1. Start services with new configuration
export $(cat config/.sep-config.production.env | xargs)
python scripts/trading_service.py &
TRADING_SERVICE_PID=$!

# 2. Test API endpoints
sleep 5
curl -f http://localhost:8080/health || exit 1
curl -f http://localhost:8080/status || exit 1

# 3. Test WebSocket connections
python -c "
import websocket
import json
ws = websocket.WebSocket()
ws.connect('ws://localhost:8765')
ws.send(json.dumps({'type': 'ping'}))
response = ws.recv()
assert 'pong' in response
ws.close()
"

# 4. Test trading functionality
./build/src/trading/quantum_pair_trainer train EUR_USD --test-mode

# 5. Cleanup
kill $TRADING_SERVICE_PID

echo "✅ End-to-end tests completed successfully"
```

### Phase 4: Performance Testing

#### 4.1 Connection Performance Tests
```python
import time
import statistics
import psycopg2
from concurrent.futures import ThreadPoolExecutor

def measure_connection_time():
    """Measure SSL connection establishment time."""
    start_time = time.time()
    conn = create_postgresql_connection()
    conn.close()
    return time.time() - start_time

def test_connection_performance():
    """Test connection performance under load."""
    # Measure single connection time
    times = [measure_connection_time() for _ in range(10)]
    avg_time = statistics.mean(times)
    print(f"Average connection time: {avg_time:.3f}s")
    assert avg_time < 2.0, "Connection time too slow"
    
    # Test concurrent connections
    with ThreadPoolExecutor(max_workers=10) as executor:
        start_time = time.time()
        futures = [executor.submit(measure_connection_time) for _ in range(10)]
        results = [f.result() for f in futures]
        total_time = time.time() - start_time
        
    print(f"Concurrent connections time: {total_time:.3f}s")
    assert total_time < 5.0, "Concurrent connection time too slow"
```

#### 4.2 Query Performance Tests
```sql
-- Performance test queries
EXPLAIN ANALYZE SELECT * FROM training_data 
WHERE pair = 'EUR_USD' 
AND timestamp > NOW() - INTERVAL '1 day';

EXPLAIN ANALYZE SELECT COUNT(*) FROM model_states 
WHERE accuracy > 0.8;

-- Test index performance
\timing on
SELECT * FROM training_data WHERE pair = 'EUR_USD' LIMIT 1000;
```

**Performance Benchmarks**:
- Connection establishment: < 2 seconds
- Simple query response: < 100ms
- Complex aggregation: < 1 second
- Concurrent connections: Support 20+ simultaneous connections

### Phase 5: Error Handling and Recovery Testing

#### 5.1 Network Failure Tests
```python
def test_connection_timeout():
    """Test connection timeout handling."""
    # Test with invalid host
    with pytest.raises(psycopg2.OperationalError):
        psycopg2.connect(
            host="invalid-host.db.ondigitalocean.com",
            port=25060,
            database="defaultdb",
            user="doadmin",
            password="password",
            connect_timeout=5
        )

def test_connection_retry():
    """Test connection retry logic."""
    # Implement connection retry wrapper
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = create_postgresql_connection()
            return conn
        except psycopg2.OperationalError as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
```

#### 5.2 SSL Certificate Tests
```bash
# Test with expired certificate
echo "Testing with expired certificate..."
# Create expired test certificate and verify rejection

# Test with wrong hostname
echo "Testing hostname verification..."
# Test verify-full mode with IP instead of hostname

# Test without certificate
echo "Testing missing certificate..."
rm config/ca-certificate.crt
./scripts/test-db-connection.sh && echo "ERROR: Should have failed" || echo "OK: Failed as expected"
```

#### 5.3 Authentication Failure Tests
```python
def test_invalid_credentials():
    """Test handling of authentication failures."""
    with pytest.raises(psycopg2.OperationalError) as exc_info:
        psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user="invalid_user",
            password="invalid_password",
            sslmode='require'
        )
    assert "authentication failed" in str(exc_info.value)
```

### Phase 6: Data Consistency and Migration Validation

#### 6.1 Data Migration Tests
```bash
#!/bin/bash
# Data migration validation script

echo "🔍 Validating data migration..."

# Count records before migration
LOCAL_COUNT=$(psql sep_trading -t -c "SELECT COUNT(*) FROM training_data;")
echo "Local records: $LOCAL_COUNT"

# Count records after migration
REMOTE_COUNT=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME --set=sslmode=require -t -c "SELECT COUNT(*) FROM training_data;")
echo "Remote records: $REMOTE_COUNT"

# Validate counts match
if [ "$LOCAL_COUNT" -eq "$REMOTE_COUNT" ]; then
    echo "✅ Record counts match"
else
    echo "❌ Record count mismatch: Local=$LOCAL_COUNT, Remote=$REMOTE_COUNT"
    exit 1
fi

# Validate data checksums
LOCAL_CHECKSUM=$(psql sep_trading -t -c "SELECT MD5(string_agg(pair || timestamp::text, '')) FROM training_data ORDER BY pair, timestamp;")
REMOTE_CHECKSUM=$(PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME --set=sslmode=require -t -c "SELECT MD5(string_agg(pair || timestamp::text, '')) FROM training_data ORDER BY pair, timestamp;")

if [ "$LOCAL_CHECKSUM" = "$REMOTE_CHECKSUM" ]; then
    echo "✅ Data checksums match"
else
    echo "❌ Data checksum mismatch"
    echo "Local: $LOCAL_CHECKSUM"
    echo "Remote: $REMOTE_CHECKSUM"
    exit 1
fi
```

#### 6.2 Schema Validation Tests
```sql
-- Verify schema consistency
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;

-- Verify indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes 
WHERE schemaname = 'public'
ORDER BY tablename, indexname;

-- Verify constraints
SELECT 
    tc.constraint_name,
    tc.table_name,
    tc.constraint_type,
    rc.update_rule,
    rc.delete_rule
FROM information_schema.table_constraints tc
LEFT JOIN information_schema.referential_constraints rc
    ON tc.constraint_name = rc.constraint_name
WHERE tc.table_schema = 'public'
ORDER BY tc.table_name, tc.constraint_name;
```

### Phase 7: Production Readiness Testing

#### 7.1 Load Testing
```python
import concurrent.futures
import time
import psycopg2
from statistics import mean, stdev

def load_test_database():
    """Perform load testing on DigitalOcean PostgreSQL."""
    
    def worker_task(worker_id):
        """Individual worker task."""
        connection_times = []
        query_times = []
        
        for i in range(100):  # 100 operations per worker
            # Test connection establishment
            start_time = time.time()
            conn = create_postgresql_connection()
            connection_times.append(time.time() - start_time)
            
            # Test query execution
            with conn.cursor() as cur:
                start_time = time.time()
                cur.execute("SELECT COUNT(*) FROM training_data WHERE pair = %s;", ("EUR_USD",))
                cur.fetchone()
                query_times.append(time.time() - start_time)
            
            conn.close()
        
        return {
            'worker_id': worker_id,
            'connection_times': connection_times,
            'query_times': query_times
        }
    
    # Run load test with 10 concurrent workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_task, i) for i in range(10)]
        results = [future.result() for future in futures]
    
    # Analyze results
    all_connection_times = []
    all_query_times = []
    
    for result in results:
        all_connection_times.extend(result['connection_times'])
        all_query_times.extend(result['query_times'])
    
    print(f"Load Test Results:")
    print(f"  Total operations: {len(all_connection_times)}")
    print(f"  Average connection time: {mean(all_connection_times):.3f}s ± {stdev(all_connection_times):.3f}s")
    print(f"  Average query time: {mean(all_query_times):.3f}s ± {stdev(all_query_times):.3f}s")
    
    # Assertions for production readiness
    assert mean(all_connection_times) < 3.0, "Connection time too slow for production"
    assert mean(all_query_times) < 0.5, "Query time too slow for production"
    assert max(all_connection_times) < 10.0, "Maximum connection time too slow"
```

#### 7.2 Monitoring and Alerting Tests
```bash
# Test monitoring integration
./scripts/setup-monitoring.sh

# Verify metrics collection
curl -f http://localhost:9090/metrics | grep postgres_

# Test alert firing
# Simulate high connection count, slow queries, etc.
```

### Test Execution Plan

#### Phase Schedule
1. **Week 1**: Pre-migration validation and SSL configuration testing
2. **Week 2**: Application integration and performance testing
3. **Week 3**: Error handling, data consistency, and production readiness testing
4. **Week 4**: Final validation, documentation, and go-live preparation

#### Success Criteria
- [ ] All SSL connections establish successfully
- [ ] Performance meets or exceeds current benchmarks
- [ ] Data integrity verified (100% data consistency)
- [ ] Error handling works correctly for all failure modes
- [ ] Load testing shows system can handle production traffic
- [ ] All automated tests pass consistently
- [ ] Documentation is complete and accurate

#### Rollback Triggers
- Connection failure rate > 1%
- Query performance degradation > 50%
- Data consistency issues
- Security vulnerabilities detected
- SSL/TLS configuration issues

#### Test Environment Requirements
- Staging environment with DigitalOcean managed PostgreSQL
- Test data set (minimum 10,000 training records)
- Load generation tools
- Monitoring and logging systems
- Backup and recovery procedures tested

### Documentation and Sign-off
- [ ] Test results documented
- [ ] Performance benchmarks recorded
- [ ] Security audit completed
- [ ] Operations runbook updated
- [ ] Rollback procedures validated
- [ ] Stakeholder sign-off obtained

## Test Automation
All tests should be automated where possible and integrated into CI/CD pipeline:
- Unit tests: GoogleTest for C++, pytest for Python
- Integration tests: Docker Compose test environments
- Performance tests: Automated benchmarking
- Security tests: SSL/TLS validation scripts
- Data consistency: Automated checksum validation