# Build and Compilation Guide

This document provides detailed instructions for building the SEP Professional Trader-Bot from source. For a general setup guide, please see the [Quick Start Guide](../00_OVERVIEW/00_QUICKSTART.md).

## System Requirements

A C++17 compliant compiler is required with C++20 features support. The following are officially supported and tested:
- **GCC:** 11.4+
- **Clang:** 14.0+
- **MSVC:** Visual Studio 2022 (v143)+

### Dependencies

The `install` script will attempt to install all required dependencies using system package managers (`apt`, `dnf`) or `vcpkg`.

**Core Dependencies:**
- CMake 3.25+
- Boost 1.71+
- OpenSSL 3.0+
- PostgreSQL (client libraries)
- Redis (client libraries)
- nlohmann-json 3.10+
- GLM (OpenGL Mathematics Library)

**CUDA Dependencies (for local training machine):**
- NVIDIA Driver 550+
- CUDA Toolkit 12.9+
- cuDNN 8.9+

---

## Building the Project

The project uses separate build scripts for Linux and Windows, both supporting Docker containerization for consistent builds.

### Linux Build

The `build.sh` script is the primary interface for building on Linux.

**Standard Build (with Docker for remote deployment simulation):**
```bash
./build.sh
```

**Local Build (No Docker, for CUDA-enabled training):**
This is the standard for a local development machine with a GPU.
```bash
./build.sh --no-docker
```

**Clean Build:**
To force a complete rebuild, use the `--clean` flag.
```bash
./build.sh --clean
```

### Windows Build

The `build.bat` script handles the build process on Windows using Docker containerization.

**Standard Build:**
*Ensure you have Docker Desktop installed and running.*
```bat
.\build.bat
```

**Build Process Details:**
- Uses `sep-trader-build` Docker image for consistent environment
- Integrates PowerShell `Tee-Object` for build output capture
- Automatically captures build errors to `output/errors.txt`
- Uses multi-line command structure with `^` continuation for readability

---

## Build System Architecture

### CMake Configuration
- **Root CMakeLists.txt**: Main project configuration with global compiler flags
- **CUDA Integration**: Global CUDA flags set to suppress GCC extension warnings:
  ```cmake
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-gnu-line-directive")
  ```
- **External Dependencies**: `nlohmann_json` marked as `SYSTEM` to suppress warnings
- **Ninja Generator**: Used for fast, parallel builds

### Precompiled Headers (PCH)
- **Canonical PCH**: `src/core/sep_precompiled.h`
- **Consolidated**: Previously duplicate PCH files have been merged
- **Performance**: Significantly reduces compilation time for large codebase

### Docker Integration
- **Build Image**: `sep-trader-build` with GCC-11, CUDA 12.9+, CMake, Ninja
- **Consistent Environment**: Eliminates "works on my machine" issues
- **Automated**: Both Linux and Windows builds use containerization

---

## Core Type System

The project uses a sophisticated type system with the following key components:

### Result<T> Template
```cpp
// Modern error handling using std::variant
template<typename T>
class Result {
private:
    std::variant<T, Error> data_;
public:
    // ... implementation
};

// Specialized for void operations
template<>
class Result<void> {
private:
    std::variant<std::monostate, Error> data_;
    // ... implementation
};
```

### Quantum Types
- **Location**: `src/core/quantum_types.h` (consolidated)
- **Key Types**: `Pattern`, `QuantumState`, `PatternRelationship`
- **Usage**: Centralized definitions prevent duplicate type conflicts

### Configuration System
- **SystemConfig**: Global system configuration
- **CudaConfig**: CUDA-specific settings (GPU usage, memory management)
- **QuantumThresholdConfig**: QFH algorithm thresholds

---

## Troubleshooting Build Issues

### Common Compilation Errors (Recently Resolved)

**"style of line directive is a GCC extension" warnings:**
- **Fixed**: Global CUDA flags suppress these warnings
- **Root Cause**: CUDA compiler generates GCC-style line directives
- **Solution**: Added `-Xcompiler -Wno-gnu-line-directive` to CMAKE_CUDA_FLAGS

**Pattern ID Type Mismatches:**
- **Fixed**: Standardized `uint32_t` for internal IDs, `std::string` for external interfaces
- **Root Cause**: Inconsistent ID types across different modules
- **Solution**: Added conversion functions and consistent type usage

**Result<T> Template Issues:**
- **Fixed**: Implemented proper `std::variant`-based Result system
- **Root Cause**: Mixing enum-based and template-based Result types
- **Solution**: Refactored to unified template system with `std::monostate` for void

**Missing Header Includes:**
- **Fixed**: Standardized include paths using CMake configuration
- **Root Cause**: Relative paths breaking with different build configurations
- **Solution**: Used CMake's include directory structure

### "Library not found" / Linker Errors
This is common when the dynamic linker cannot find the compiled libraries from this project.

**On Linux:**
Ensure you have set the `LD_LIBRARY_PATH` environment variable in your shell session.
```bash
export LD_LIBRARY_PATH=./build/src/core:./build/src/config:./build/src/c_api
```
To make this permanent, add it to your `~/.bashrc` or `~/.zshrc` file.

**On Windows:**
Ensure that the build output directories have been added to your system's `PATH` environment variable.

### CUDA Compilation Errors
- **Verify CUDA Installation:** Ensure `nvcc --version` runs correctly and shows version 12.9+
- **Check Driver/Toolkit Compatibility:** Make sure your NVIDIA driver version is compatible with your CUDA Toolkit version
- **Compiler Compatibility:** GCC-11 is the tested and recommended compiler for CUDA builds
- **Template Issues**: Check for duplicate template declarations, especially in `.cu` files

### Build Output and Error Capture
- **Linux**: Build output automatically saved with `tee` command
- **Windows**: Build output captured to `output/errors.txt` using PowerShell `Tee-Object`
- **Error Analysis**: Filtered error logs help identify critical compilation issues

### `vcpkg` Failures
- **Outdated `vcpkg`:** Run `git -C ./vcpkg pull` and `./vcpkg/bootstrap-vcpkg.bat` (or `.sh`) to update `vcpkg`
- **Network Issues:** `vcpkg` needs to download sources. Check your network connection and firewall settings
- **Build Failures:** Check the logs in `vcpkg/buildtrees` for the specific package that failed to build

### Docker Build Issues
- **Windows**: Ensure Docker Desktop is running and configured for Linux containers
- **Memory**: CUDA builds require significant memory; ensure Docker has adequate resources allocated
- **Image Updates**: Rebuild the `sep-trader-build` image if dependency versions change

---

## Recent Architectural Improvements

### Code Quality
- **RAII Patterns**: Proper resource management for CUDA memory and streams
- **Smart Pointers**: Extensive use of `std::unique_ptr` for memory safety
- **Exception Safety**: Robust error handling with `Result<T>` pattern
- **Const Correctness**: Improved const-correctness throughout codebase

### Performance
- **CUDA Optimization**: Proper kernel launch parameter handling
- **Memory Management**: Efficient GPU memory allocation and deallocation
- **Compilation Speed**: PCH consolidation reduces build times significantly

### Maintainability
- **Type Safety**: Strong typing system prevents common errors
- **Modular Design**: Clear separation of concerns between modules
- **Documentation**: Comprehensive inline documentation and external guides
