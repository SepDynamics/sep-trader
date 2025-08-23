# SEP Trading Engine - De-stubbing TODO

## Next Task
Begin work on **Priority 2: Real Multi-Asset Correlation Implementation** described below.

## Recent Build Fixes
- `serializeFusionResult` in `src/app/multi_asset_signal_fusion.cpp` had unescaped JSON braces, causing `fmt` to treat the format string as invalid. Escaped the outer braces to restore compilation.
- `processAsset` implementation in `src/app/quantum_signal_bridge.cpp` used `QuantumIdentifiers` outside its namespace. Fully qualified the return type and initializer.
- `src/core/cuda_walk_forward_validator.cu` failed to compile due to missing `_DISABLE_FPCLASSIFY_FUNCTIONS` macro. Added the macro to align with other CUDA modules.
- Some toolchains still defined legacy `fpclassify`-related macros which broke CUDA's `<cuda/std>` headers. Added defensive `#undef` guards in `src/core/cuda_walk_forward_validator.cu` to ensure portable compilation.
- `src/util/error_handling.c` was excluded from builds and included the wrong header, producing undefined references to `sep_error_*`. Added `.c` sources to `src/CMakeLists.txt` and corrected the include path.
- Native builds failed with `bits/c++config.h` errors because the build script forced unavailable `g++-11`/`gcc-11` compilers. Switched to the system defaults in `build.sh` and fixed the installer symlink to point at `sep-trader`, restoring access to standard library headers.
- Added `PROJECT_ROOT` compile definition and centralized cache/config/log directory variables in `CMakeLists.txt` for consistent path resolution.
- Missing `<cstdint>` headers in `_sep/testbed/evolution/pattern.hpp` and `src/cuda/quantum_training.cu` caused `uint64_t`/`uint32_t` type errors. Included the header in both files.
- `oneTBB` failed to build with GCC-14 due to `-Werror=stringop-overflow` in `__atomic_store_1`. Added `-Wno-error=stringop-overflow` to project compile options; consider upstream patch or TBB upgrade for permanent fix.
- `runTestDataSimulation()` in `src/app/quantum_tracker_app.cpp` referenced outdated `PatternData` fields and lacked required includes, breaking the build. Removed this development-only function to eliminate fake data generation and restore compilation.
- Custom `pqxx_time_point_traits` used function-based `has_null()` which was not a constant expression, causing PostgreSQL integration to fail. Replaced it with `constexpr` flags `has_null`/`always_null` to satisfy `pqxx`'s `nullness` checks.
- GCC-14 still rejected the specialization because it was included after `<pqxx/pqxx>`, instantiating `pqxx::nullness` with incomplete `constexpr` flags. Moved the specialization into a standalone header included before pqxx and generalized it for all `system_clock` durations, restoring the build.
- GCC-14 again complained about `pqxx::nullness` when `time_point` resolved through the `_V2::system_clock` inline namespace. Switched the specialization to use `std::chrono::sys_time` and marked the nullness flags `inline constexpr` to ensure constant evaluation.

## Priority 1: Remove Fake Data Generation [COMPLETED]

### Task 1.1: Eliminate Demo Candle Generation
**File**: `src/app/market_model_cache.cpp`
**Current Issue**: `ensureCacheForLastWeek()` generates 1,000 random demo candles when OANDA fails

```cpp
// REMOVE THIS ENTIRE BLOCK (lines ~150-200):
if (!data_fetched || raw_candles.empty()) {
    std::cout << "[CACHE] Generated demo candles for testing" << std::endl;
    // Random generation code...
}
```

**Implementation**:
1. Replace with hard failure:
   ```cpp
   if (!data_fetched || raw_candles.empty()) {
       spdlog::error("[CACHE] Failed to fetch OANDA data for {}", pair);
       return false;  // Hard fail, no fallback
   }
   ```

2. Add offline cache support:
   ```cpp
   // New parameter in constructor
   MarketModelCache(connector, cache_dir = "./cache/", offline_mode = false);
   
   // Check offline cache first
   if (offline_mode || !connector->isConnected()) {
       return loadOfflineCache(pair, cache_dir);
   }
   ```

3. Update unit tests in `tests/market_cache/`:
   - Test expects `false` on missing OANDA data
   - Add test for offline cache loading
   - Verify no "Generated demo candles" log exists

**Verification**: 
```bash
grep -r "Generated demo candles" src/ # Should return nothing
grep -r "random_device" src/app/market_model_cache.cpp # Should return nothing
```

---

## Priority 2: Real Multi-Asset Correlation Implementation

### Task 2.1: Replace Default Correlation Values
**File**: `src/app/multi_asset_signal_fusion.cpp`
**Method**: `calculateDynamicCorrelation()`

**Current Issue**: Returns hardcoded (0.0, 0ms, 0.0) and fake quantum identifiers

**Implementation Steps**:

1. **Fetch Real Historical Data**:
   ```cpp
   CrossAssetCorrelation MultiAssetSignalFusion::calculateDynamicCorrelation(
       const std::string& asset1, const std::string& asset2) {
       
       // Fetch last N candles for both assets
       auto candles1 = market_cache_->getRecentCandles(asset1, 100);
       auto candles2 = market_cache_->getRecentCandles(asset2, 100);
       
       if (candles1.size() < 50 || candles2.size() < 50) {
           return std::nullopt;  // Insufficient data
       }
       
       // Calculate returns
       std::vector<double> returns1 = calculateReturns(candles1);
       std::vector<double> returns2 = calculateReturns(candles2);
       
       // Compute Pearson correlation with lag optimization
       auto [correlation, optimal_lag] = computePearsonWithLag(returns1, returns2);
       
       // Calculate stability over rolling windows
       double stability = calculateCorrelationStability(candles1, candles2);
       
       return CrossAssetCorrelation{
           .strength = correlation,
           .optimal_lag = std::chrono::milliseconds(optimal_lag),
           .stability = stability
       };
   }
   ```

2. **Remove Hardcoded Quantum Identifiers**:
   ```cpp
   // DELETE THIS BLOCK in generateFusedSignal():
   sep::trading::QuantumIdentifiers quantum_identifiers{
       .confidence = 0.7f,  // REMOVE
       .coherence = 0.4f,   // REMOVE
       .stability = 0.5f,   // REMOVE
       .converged = true
   };
   
   // REPLACE WITH:
   auto quantum_identifiers = quantum_processor_->processAsset(asset);
   ```

3. **Implement calculateCrossAssetBoost**:
   ```cpp
   double calculateCrossAssetBoost(
       const QuantumIdentifiers& signal,
       const CrossAssetCorrelation& correlation) {
       
       double base_boost = correlation.strength * correlation.stability;
       double coherence_factor = signal.coherence / 0.3;  // Normalized to threshold
       double confidence_factor = signal.confidence / 0.65;  // Normalized
       
       return base_boost * coherence_factor * confidence_factor * 0.2;  // Max 20% boost
   }
   ```

### Task 2.2: Add Integration Tests
**Location**: `tests/signal_fusion/`

Create comprehensive tests:
```cpp
TEST(MultiAssetFusion, PositiveCorrelation) {
    // Feed known correlated sequences
    // Verify correlation > 0.7
}

TEST(MultiAssetFusion, NegativeCorrelation) {
    // Feed inverse sequences
    // Verify correlation < -0.7
}

TEST(MultiAssetFusion, ZeroCorrelation) {
    // Feed random uncorrelated data
    // Verify correlation near 0.0
}
```

---

## Priority 3: Remove Testbed and Backtesting Divergence

### Task 3.1: Delete Testbed Directory
```bash
rm -rf _sep/testbed/
git rm -rf _sep/testbed/
```

### Task 3.2: Remove SEP_BACKTESTING Macro
**Files**: All source files

1. **Search and Remove All Occurrences**:
   ```bash
   # Find all files with the macro
   grep -r "SEP_BACKTESTING" src/
   
   # Remove all #ifdef SEP_BACKTESTING blocks
   # Keep only the production code path
   ```

2. **Update CMakeLists.txt**:
   ```cmake
   # REMOVE:
   add_definitions(-DSEP_BACKTESTING)
   
   # REMOVE filter excluding oanda files:
   file(GLOB_RECURSE ALL_SOURCES 
       # Remove any exclusion patterns for oanda_*
   )
   
   # ADD for tests only:
   if(BUILD_TESTS)
       target_compile_definitions(test_target PRIVATE USE_MOCKS=1)
   endif()
   ```

3. **Update src/core/quantum_pair_trainer.cpp**:
   ```cpp
   // REMOVE:
   #ifdef SEP_BACKTESTING
       return sep::testbed::fetchMarketData(*oanda_connector_, pair_symbol, hours_back);
   #else
       // production code
   #endif
   
   // KEEP ONLY:
   return oanda_connector_->fetchHistoricalData(pair_symbol, hours_back);
   ```

### Task 3.3: Add CI Lint Check
**File**: `.github/workflows/lint.yml` or `scripts/lint.sh`

```bash
#!/bin/bash
# Fail if stub code is reintroduced
FORBIDDEN_PATTERNS=(
    "SEP_BACKTESTING"
    "demo_candles"
    "Generated demo"
    "testbed::"
    "random_device.*market_model_cache"
)

for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    if grep -r "$pattern" src/; then
        echo "ERROR: Forbidden pattern found: $pattern"
        exit 1
    fi
done
```

---

## Priority 4: Wire CLI Commands to Real Implementation

### Task 4.1: Implement CLICommands Methods
**File**: `src/core/cli_commands.cpp`

Replace printf stubs with actual implementation:

```cpp
bool CLICommands::trainPair(const std::string& pair) {
    try {
        // Initialize training session
        sep::trading::QuantumTrainingConfig config;
        config.pair_symbol = pair;
        config.training_hours = 168;  // 1 week
        
        sep::trading::QuantumPairTrainer trainer(config);
        auto session_id = training_manager_->createSession(pair);
        
        // Run training
        auto result = trainer.train(pair);
        
        // Persist results
        training_manager_->saveResults(session_id, result);
        
        spdlog::info("Training completed for {}: accuracy={:.2f}%", 
                     pair, result.high_confidence_accuracy * 100);
        return result.training_successful;
        
    } catch (const std::exception& e) {
        spdlog::error("Training failed for {}: {}", pair, e.what());
        return false;
    }
}

bool CLICommands::cleanupCache() {
    try {
        // Delete old cache files
        std::filesystem::path cache_dir("./cache/");
        auto cutoff = std::chrono::system_clock::now() - std::chrono::hours(24*7);
        
        for (const auto& entry : std::filesystem::directory_iterator(cache_dir)) {
            auto ftime = std::filesystem::last_write_time(entry);
            auto sctp = decltype(cutoff)::clock::to_time_t(cutoff);
            auto ftp = decltype(ftime)::clock::to_time_t(ftime);
            
            if (ftp < sctp) {
                std::filesystem::remove(entry);
                spdlog::debug("Removed old cache file: {}", entry.path().string());
            }
        }
        
        // Clear in-memory caches
        MarketModelCache::getInstance().invalidateCache();
        
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Cache cleanup failed: {}", e.what());
        return false;
    }
}

bool CLICommands::runBenchmark() {
    // Fixed benchmark scenario
    const std::string BENCHMARK_PAIR = "EUR_USD";
    const size_t BENCHMARK_ITERATIONS = 1000;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        // Generate signal
        auto signal = quantum_processor_->analyzeMarketData(
            sample_data, sample_history, forward_windows);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double throughput = BENCHMARK_ITERATIONS / (duration.count() / 1000.0);
    double latency = duration.count() / static_cast<double>(BENCHMARK_ITERATIONS);
    
    spdlog::info("Benchmark Results:");
    spdlog::info("  Throughput: {:.2f} signals/sec", throughput);
    spdlog::info("  Avg Latency: {:.2f} ms", latency);
    
    return true;
}
```

### Task 4.2: Wire CLI to Main
**File**: `src/app/cli_main.cpp`

```cpp
// Add proper command routing
if (command == "train") {
    if (args.size() < 2) {
        std::cerr << "Usage: sep_cli train <pair>" << std::endl;
        return 1;
    }
    return cli_commands.trainPair(args[1]) ? 0 : 1;
    
} else if (command == "train-all") {
    bool quick = (args.size() > 1 && args[1] == "--quick");
    return cli_commands.trainAllPairs(quick) ? 0 : 1;
    
} else if (command == "cleanup-cache") {
    return cli_commands.cleanupCache() ? 0 : 1;
    
} else if (command == "benchmark") {
    return cli_commands.runBenchmark() ? 0 : 1;
}
```

---

## Priority 5: Centralize Configuration Management

### Task 5.1: Create Environment Loader
**New Files**: `src/app/env_loader.h`, `src/app/env_loader.cpp`

```cpp
// env_loader.h
#pragma once
#include <string>
#include <optional>

namespace sep::config {

struct OandaEnv {
    std::string api_key;
    std::string account_id;
    std::string base_url;
};

struct AppConfig {
    OandaEnv oanda;
    std::string cache_dir;
    std::string log_level;
    bool offline_mode;
};

class EnvLoader {
public:
    static OandaEnv loadOandaEnv();  // Throws on missing vars
    static AppConfig loadFullConfig();
    static std::string getProjectRoot();
    
private:
    static void validateOandaCredentials(const OandaEnv& env);
};

} // namespace sep::config
```

```cpp
// env_loader.cpp
#include "env_loader.h"
#include <cstdlib>
#include <stdexcept>
#include <filesystem>

namespace sep::config {

OandaEnv EnvLoader::loadOandaEnv() {
    OandaEnv env;
    
    const char* api_key = std::getenv("OANDA_API_KEY");
    const char* account_id = std::getenv("OANDA_ACCOUNT_ID");
    const char* base_url = std::getenv("OANDA_BASE_URL");
    
    if (!api_key || !account_id) {
        throw std::runtime_error(
            "Missing OANDA credentials. Set OANDA_API_KEY and OANDA_ACCOUNT_ID");
    }
    
    env.api_key = api_key;
    env.account_id = account_id;
    env.base_url = base_url ? base_url : "https://api-fxpractice.oanda.com";
    
    validateOandaCredentials(env);
    return env;
}

void EnvLoader::validateOandaCredentials(const OandaEnv& env) {
    // Quick API validation
    sep::connectors::OandaConnector test_conn(env.api_key, env.account_id);
    if (!test_conn.testConnection()) {
        throw std::runtime_error("Invalid OANDA credentials or connection failed");
    }
}

std::string EnvLoader::getProjectRoot() {
    // Use CMake-defined PROJECT_ROOT
    #ifdef PROJECT_ROOT
        return std::string(PROJECT_ROOT);
    #else
        // Fallback to current directory
        return std::filesystem::current_path().string();
    #endif
}

} // namespace sep::config
```

### Task 5.2: Update All OANDA Initialization
**Files**: All files using OANDA credentials

Replace all instances of:
```cpp
// OLD:
const char* api_key = std::getenv("OANDA_API_KEY");
const char* account_id = std::getenv("OANDA_ACCOUNT_ID");

// NEW:
auto oanda_env = sep::config::EnvLoader::loadOandaEnv();
oanda_connector_ = std::make_unique<OandaConnector>(
    oanda_env.api_key, oanda_env.account_id);
```

### Task 5.3: Fix Path Resolution
**File**: `CMakeLists.txt`

```cmake
# Add PROJECT_ROOT definition
add_compile_definitions(PROJECT_ROOT="${CMAKE_SOURCE_DIR}")

# Update all path references
set(CACHE_DIR "${PROJECT_ROOT}/cache/")
set(CONFIG_DIR "${PROJECT_ROOT}/config/")
set(LOG_DIR "${PROJECT_ROOT}/logs/")
```

---

## Execution Order & Parallelization Strategy

### Phase 1: Foundation (Sequential - 1-2 days)
1. **Task 5.1-5.3**: Centralize configuration (prerequisite for all)
2. **Task 3.1-3.2**: Remove testbed/macros (cleans codebase)

### Phase 2: Core De-stubbing (Parallel - 2-3 days)
**Team A**:
- **Task 1.1**: Remove demo candle generation
- Integration tests for cache validation

**Team B**:
- **Task 2.1-2.2**: Implement real correlation
- Unit tests for correlation math

**Team C**:
- **Task 4.1-4.2**: Wire CLI commands
- End-to-end CLI testing

### Phase 3: Validation (Sequential - 1 day)
1. **Task 3.3**: Add CI lint checks
2. Full integration test suite
3. Performance benchmarking
4. Production validation on test account

### Phase 4: Deployment (1 day)
1. Deploy to staging environment
2. Validate with real OANDA data
3. Monitor for 24 hours
4. Production rollout

## Success Metrics

### Functional Validation
- [ ] Zero "demo candles" in logs
- [ ] All CLI commands execute real operations
- [ ] Correlation values match expected ranges
- [ ] No SEP_BACKTESTING references in code
- [ ] Single configuration source

### Performance Targets
- [ ] Cache hit ratio > 80%
- [ ] Correlation calculation < 100ms
- [ ] CLI response time < 500ms
- [ ] Memory usage stable over 24h

### Quality Gates
- [ ] All unit tests passing
- [ ] Integration tests with real OANDA data
- [ ] Zero compilation warnings
- [ ] Static analysis clean
- [ ] Documentation updated

## Risk Mitigation

1. **Data Loss Risk**: Backup existing cache before changes
2. **API Rate Limits**: Implement exponential backoff
3. **Correlation Accuracy**: Validate against known correlations
4. **Production Impact**: Deploy during low-volume periods
5. **Rollback Plan**: Tag release before deployment

## Notes for Codex Execution

When implementing these tasks:
1. Preserve all existing functionality
2. Add comprehensive error handling
3. Include detailed logging at each step
4. Write tests before implementation
5. Document any API changes
6. Update README.md with changes

## Dependencies Between Tasks

```
Task 5 (Config) → All other tasks
Task 3 (Remove testbed) → Task 1 (Remove demo data)
Task 1 (Remove demo) → Task 2 (Correlation)
Task 4 (CLI) → Task 1,2 (Needs real implementations)
```

## Verification Commands

```bash
# Verify no stubs remain
./scripts/lint.sh

# Test real OANDA connection
./build/src/trader_cli test-connection

# Validate correlation calculations
./build/tests/correlation_test

# Benchmark performance
./build/src/trader_cli benchmark

# Full integration test
./scripts/integration_test.sh
```