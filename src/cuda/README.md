# SEP Engine CUDA Library

This directory contains the consolidated CUDA implementation for the SEP Engine. It provides a centralized, well-organized library for all CUDA functionality used across the codebase.

## Directory Structure

```
src/cuda/
├── common/           # Shared CUDA utilities, memory management
│   ├── memory/       # Memory buffer implementations (Device, Pinned, Unified)
│   ├── error/        # Error handling and reporting
│   ├── stream/       # Stream management utilities
│   └── types/        # Type definitions and compatibility
├── kernels/          # All CUDA kernels organized by domain
│   ├── quantum/      # Quantum processing kernels
│   ├── pattern/      # Pattern analysis kernels
│   └── trading/      # Trading computation kernels
└── api/              # Public CUDA API headers
    ├── cuda_api.h    # Main API entry point
    ├── quantum_api.h # Quantum processing API
    ├── pattern_api.h # Pattern analysis API
    └── trading_api.h # Trading computation API
```

## Core Components

### Memory Management

The memory management system provides RAII-compliant wrappers for CUDA memory operations:

- `DeviceBuffer`: GPU memory buffer
- `PinnedBuffer`: Pinned host memory
- `UnifiedBuffer`: Unified memory buffer (coming soon)

### Error Handling

Consistent error handling mechanisms for CUDA operations:

- `CUDA_CHECK(expr)`: Check CUDA errors and throw exceptions
- `CUDA_CHECK_LAST()`: Check for asynchronous errors

### Stream Management

RAII-compliant stream handling:

- `Stream`: CUDA stream wrapper with lifecycle management
- Stream synchronization and callback support

## Build System

The CUDA library is built as a standalone shared library (`libsep_cuda.so`) using CMake.

### Building

```bash
cmake -B build -S .
cmake --build build
```

### Integration

Other components should link against the CUDA library:

```cmake
target_link_libraries(your_target PRIVATE sep_cuda)
```

## API Design Principles

1. **RAII-Compliant**: All resources are properly managed
2. **Type-Safe**: Template-based API with strong typing
3. **Exception-Based**: Errors throw exceptions rather than return codes
4. **Consistent Naming**: Clear, consistent naming conventions
5. **Documented**: Comprehensive documentation for all components