# Build and Compilation Guide

This document provides detailed instructions for building the SEP Professional Trader-Bot from source. For a general setup guide, please see the [Quick Start Guide](../00_OVERVIEW/00_QUICKSTART.md).

## System Requirements

A C++20 compliant compiler is required. The following are officially supported and tested:
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

**CUDA Dependencies (for local training machine):**
- NVIDIA Driver 550+ 
- CUDA Toolkit 12.2+
- cuDNN 8.9+

---

## Building the Project

The project uses separate build scripts for Linux and Windows.

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

The `build.bat` script handles the build process on Windows.

**Standard Build:**
*Ensure you have Visual Studio 2022, CMake, and `vcpkg` installed and available in your PATH.*
```bat
.\build.bat
```

---

## Troubleshooting Build Issues

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
- **Verify CUDA Installation:** Ensure `nvcc --version` runs correctly and shows the expected version.
- **Check Driver/Toolkit Compatibility:** Make sure your NVIDIA driver version is compatible with your CUDA Toolkit version.
- **Compiler Compatibility:** Your C++ compiler (GCC/Clang/MSVC) must be compatible with the CUDA Toolkit version you are using. Refer to the official NVIDIA CUDA documentation for the compatibility matrix.

### `vcpkg` Failures
- **Outdated `vcpkg`:** Run `git -C ./vcpkg pull` and `./vcpkg/bootstrap-vcpkg.bat` (or `.sh`) to update `vcpkg`.
- **Network Issues:** `vcpkg` needs to download sources. Check your network connection and firewall settings.
- **Build Failures:** Check the logs in `vcpkg/buildtrees` for the specific package that failed to build. Sometimes, a manual intervention or a package-specific fix is required.
