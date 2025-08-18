# General Development Guide

This document provides a technical overview of the SEP Engine's implementation, build system, testing framework, and deployment procedures after extensive refactoring and bug fixes.

## 1. Architecture & Core Components

The engine is a modular, CUDA-accelerated C++17 system with C++20 features support, utilizing modern C++ patterns and robust error handling.

### 1.1. Core Type System

**Result<T> Error Handling:**
```cpp
// Location: src/core/result.h
template<typename T>
class Result {
private:
    std::variant<T, Error> data_;
public:
    bool isSuccess() const;
    bool isError() const;
    const T& value() const;
    const Error& error() const;
    
    // Factory methods
    static Result<T> success(T value);
    static Result<T> error(Error error);
};

// Void specialization using std::monostate
template<>
class Result<void> { /* ... */ };
```

**Quantum Type System:**
- **Location:** `src/core/quantum_types.h` (consolidated from multiple locations)
- **Key Types:**
  - `Pattern`: Financial patterns with `uint32_t` IDs and `std::vector<double> attributes`
  - `QuantumState`: Market state with coherence, stability, phase
  - `PatternRelationship`: Pattern connections with strength metrics

### 1.2. Quantum Signal Bridge
- **Location:** `src/app/quantum_signal_bridge.hpp/.cpp`
- **Purpose:** Core signal generation using QFH/QBSA algorithms
- **Key Logic:** Multi-timeframe (M1, M5, M15) confirmation and trajectory damping based on entropy and coherence
- **Error Handling:** Uses `Result<T>` pattern for robust error propagation

### 1.3. Multi-Asset Signal Fusion
- **Location:** `src/app/multi_asset_signal_fusion.hpp/.cpp`
- **Purpose:** Cross-asset correlation analysis to enhance signal confidence
- **Key Logic:** Dynamic Pearson correlation, weighted voting, and confidence boosting
- **Pattern Integration:** Works with unified `Pattern` type from `quantum_types.h`

### 1.4. Market Regime Adaptive Intelligence
- **Location:** `src/app/market_regime_adaptive.hpp/.cpp`
- **Purpose:** Dynamically adapts trading thresholds based on market volatility, trend, and liquidity
- **Configuration:** Uses type-safe `QuantumThresholdConfig` from `core/types.h`

### 1.5. Market Model Cache
- **Location:** `src/app/enhanced_market_model_cache.hpp/.cpp`
- **Purpose:** An intelligent, correlation-aware cache for market data to improve performance
- **Memory Safety:** Utilizes RAII patterns for resource management

## 2. Build System

### 2.1. Build Scripts
- **Primary Script:** `build.sh` (for Linux) and `build.bat` (for Windows)
- **Containerization:** Both scripts use Docker for hermetic, dependency-free builds
- **Error Capture:**
  - Linux: Uses `tee` command for build output logging
  - Windows: Uses PowerShell `Tee-Object` to capture output to `output/errors.txt`

### 2.2. CMake Configuration
- **Root CMakeLists.txt:** Main project configuration with global compiler flags
- **CUDA Integration:** Global CUDA flags configured:
  ```cmake
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-gnu-line-directive")
  ```
- **External Libraries:** `nlohmann_json` marked as `SYSTEM` to suppress warnings
- **Generator:** Uses Ninja for fast, parallel builds

### 2.3. Precompiled Headers
- **Canonical PCH:** `src/core/sep_precompiled.h`
- **Consolidation:** Previously duplicate PCH files merged for consistency
- **Performance:** Significantly reduces compilation time

### 2.4. Docker Integration
- **Build Image:** `sep-trader-build` with GCC-11, CUDA 12.9+, CMake, Ninja
- **Consistent Environment:** Eliminates platform-specific build issues
- **Multi-platform:** Works on both Linux and Windows hosts

## 3. CUDA Development

### 3.1. CUDA Architecture
- **Stream Management:** `StreamRAII` for automatic stream cleanup
- **Memory Management:** `DeviceBufferRAII` and `DeviceMemory` templates
- **Kernel Integration:** Proper `extern "C"` linkage for C++ interoperability

### 3.2. Kernel Development
```cpp
// Location: src/core/kernel_implementations.cu
extern "C" {
    void launchQBSAKernel(const QBSAParameters& params);
    void launchQSHKernel(const QSHParameters& params);
}
```

### 3.3. Error Handling
- **CudaException:** Proper exception handling for CUDA errors
- **CUDA_CHECK macros:** Automated error checking with proper exception throwing
- **Resource Management:** RAII patterns ensure GPU resources are properly cleaned up

## 4. Testing Framework

The project has a comprehensive testing suite with modern error handling.

### 4.1. Mathematical & Core Logic Tests
- **Pattern Classification:** `test_forward_window_metrics`
- **CUDA/CPU Parity:** `trajectory_metrics_test`
- **Core Algorithms:** `pattern_metrics_test`
- **Signal Generation Pipeline:** `quantum_signal_bridge_test`
- **Type System Tests:** Validation of `Result<T>` error handling and quantum types

### 4.2. Integration & Performance Tests
- **Headless System Validation:** `quantum_tracker --test`
- **Backtesting:** `pme_testbed_phase2`
- **Memory Safety Tests:** RAII pattern validation and memory leak detection

### 4.3. Error Handling Testing
All tests now use the `Result<T>` pattern for robust error reporting:
```cpp
auto result = testFunction();
if (result.isError()) {
    // Handle error appropriately
    return result.error();
}
// Use result.value() safely
```

## 5. Deployment

### 5.1. Production Trading
- **Executable:** `quantum_tracker` in `build/src/app/oanda_trader/`
- **Configuration:** Type-safe configuration using `SystemConfig`, `CudaConfig`, and `QuantumThresholdConfig`
- **Credentials:** Set OANDA credentials in an `OANDA.env` file
- **Features:** Dynamic data bootstrapping, live trade execution via OANDA API, risk management, and market schedule awareness

### 5.2. Optimal Configuration
```cpp
// Location: src/core/types.h
QuantumThresholdConfig config {
    .stability_threshold = 0.40f,
    .ltm_coherence_threshold = 0.65f,
    .mtm_coherence_threshold = 0.30f
};
```

### 5.3. Error Monitoring
- **Result<T> Integration:** All trading operations return `Result<T>` for proper error handling
- **Logging:** Comprehensive error logging with context preservation
- **Health Checks:** API endpoints for system health monitoring

## 6. Development Workflow

### 6.1. Standard Workflow
1. **Code:** Make changes to source files in the `src/` directory
2. **Build:** Run `./build.sh` or `build.bat` with automatic error capture
3. **Test:** Run the relevant test suites to validate changes
4. **Deploy:** If all tests pass, deploy the updated `quantum_tracker` executable

### 6.2. Error Debugging Workflow
1. **Build Errors:** Check `output/errors.txt` (Windows) or console output (Linux)
2. **Common Issues:** Reference the troubleshooting section in build documentation
3. **Type Errors:** Ensure proper `Result<T>` usage and quantum type consistency
4. **CUDA Errors:** Check kernel launch parameters and memory management

### 6.3. Code Quality Standards
- **Error Handling:** Always use `Result<T>` for fallible operations
- **Memory Safety:** Use RAII patterns and smart pointers
- **Type Safety:** Leverage strong typing system and avoid `void*`
- **CUDA Safety:** Always use RAII wrappers for GPU resources

## 7. Recent Architectural Improvements

### 7.1. Type System Consolidation
- **Unified Types:** All quantum types consolidated in `quantum_types.h`
- **ID Consistency:** Standardized `uint32_t` internal, `string` external ID usage
- **Namespace Organization:** Clear separation of concerns across namespaces

### 7.2. Error Handling Modernization
- **Result<T> Pattern:** Replaced error codes with modern `std::variant`-based system
- **Exception Safety:** Strong exception safety guarantees throughout
- **Error Propagation:** Proper error context preservation across call stacks

### 7.3. Build System Improvements
- **Cross-Platform:** Unified Docker-based builds for Linux and Windows
- **Warning Management:** Proper suppression of external library warnings
- **Performance:** Precompiled header consolidation and parallel builds

### 7.4. CUDA Integration
- **Resource Management:** Proper RAII patterns for all GPU resources
- **Kernel Safety:** Type-safe kernel parameter passing
- **Error Handling:** Comprehensive CUDA error checking and reporting

## 8. Known Issues and Limitations

### 8.1. Current Status
- **Compilation:** All major compilation errors have been resolved
- **Type System:** Quantum types are now properly unified and consistent
- **Build System:** Both Linux and Windows builds are fully functional
- **CUDA Integration:** All CUDA-related compilation issues resolved

### 8.2. Outstanding Items
- **ImGui Integration:** Known issue with ImGui dependency (non-critical)
- **Performance Optimization:** Further CUDA kernel optimization opportunities
- **Testing Coverage:** Additional test coverage for edge cases in error handling

### 8.3. Future Improvements
- **Static Analysis:** Integration with additional static analysis tools
- **Documentation:** Continued documentation updates as features evolve
- **Performance Metrics:** Enhanced performance monitoring and profiling