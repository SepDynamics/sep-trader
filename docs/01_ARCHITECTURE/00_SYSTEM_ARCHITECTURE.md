# System Architecture

This document describes the system architecture of the SEP Professional Trader-Bot, including the high-level design, core components, and current implementation status.

## 1. High-Level Design: Hybrid Local/Remote

The system uses a hybrid architecture to optimize for both performance and cost.

- **Local Machine (CUDA):** Your local machine, equipped with a powerful NVIDIA GPU, is responsible for all computationally intensive tasks. This includes training quantum models, running backtests, and generating trading signals.
- **Remote Droplet (CPU):** A lightweight, CPU-only cloud server is used for 24/7 trade execution. It receives signals from your local machine and interacts with the broker's API. This allows for continuous operation without the high cost and maintenance of a 24/7 GPU server.

This separation is managed by a `HybridTradingSystem` class that coordinates the training, deployment, and synchronization of data between the two environments.

## 2. Core Architecture Components

### 2.1. Type System Architecture

The project implements a sophisticated type system designed for financial modeling and quantum computing concepts:

**Result<T> Error Handling System:**
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
};

// Specialized for void operations
template<>
class Result<void> {
private:
    std::variant<std::monostate, Error> data_;
    // ... implementation
};
```

**Quantum Type System:**
- **Location**: `src/core/quantum_types.h` (consolidated)
- **Key Types**:
  - `Pattern`: Financial pattern representation with `uint32_t` IDs and `std::vector<double> attributes`
  - `QuantumState`: Market state with coherence, stability, and phase information
  - `PatternRelationship`: Defines relationships between patterns with strength metrics

**Configuration Architecture:**
```cpp
// Location: src/core/types.h
namespace sep::config {
    struct SystemConfig {
        std::string log_level;
        bool enable_cuda;
        size_t max_threads;
    };
    
    struct CudaConfig {
        bool use_gpu;
        int device_id;
        size_t max_memory_mb;
        int compute_capability_major;
        int compute_capability_minor;
    };
    
    struct QuantumThresholdConfig {
        float stability_threshold;
        float ltm_coherence_threshold;
        float mtm_coherence_threshold;
    };
}
```

### 2.2. CUDA Integration Architecture

**CUDA Stream Management:**
- **StreamRAII**: Resource management for CUDA streams
- **DeviceBufferRAII**: Automated GPU memory management
- **DeviceMemory**: Template-based GPU memory allocation

**Kernel Integration:**
- **Location**: `src/core/kernel_implementations.cu`
- **Functions**: `launchQBSAKernel`, `launchQSHKernel`
- **Linkage**: Proper `extern "C"` declarations for C++ integration

**Warning Suppression:**
- Global CUDA flags configured to suppress GCC extension warnings
- External library warnings (nlohmann_json) properly isolated

### 2.3. Module Organization

**Core Modules:**
- **`src/core/`**: Core algorithms, type definitions, and CUDA integration
- **`src/app/`**: Application services and trading logic
- **`src/io/`**: I/O connectors (OANDA, Qdrant, market data)
- **`src/util/`**: Utility functions and RAII wrappers
- **`src/cuda/`**: CUDA-specific implementations

**Key Files:**
- **`src/core/sep_precompiled.h`**: Consolidated precompiled header
- **`src/core/result.h`**: Modern error handling system
- **`src/core/quantum_types.h`**: Centralized quantum type definitions
- **`src/core/qfh.h`**: Quantum Field Harmonics implementation
- **`src/core/types.h`**: Configuration and system types

## 3. Professional State Management

The system is designed for robust and reliable state management with modern C++ practices.

### 3.1. Memory Safety
- **RAII Patterns**: Automatic resource management for CUDA resources
- **Smart Pointers**: Extensive use of `std::unique_ptr` and `std::shared_ptr`
- **Exception Safety**: `Result<T>` pattern prevents undefined behavior on errors

### 3.2. Configuration Management
- **Hot-Swappable Configuration:** System and trading parameters can be updated in real-time without requiring a restart
- **Type-Safe Configuration:** Strong typing prevents configuration errors
- **Hierarchical Config:** Separate configurations for system, CUDA, and quantum parameters

### 3.3. Data Integrity
- **Persistent State:** The trading state, including open positions and historical trades, is persisted to a database with ACID properties to ensure data integrity
- **Real-Time Monitoring:** Health metrics and trading status are available via API endpoints
- **Error Propagation:** `Result<T>` system ensures errors are properly handled throughout the call stack

## 3. Optimized Cloud Architecture Example

This section describes a cost-effective and high-performance configuration using a Digital Ocean Droplet.

### 3.1. Server Specifications
- **Droplet:** 8GB RAM, 2 vCPUs, Ubuntu 24.04 LTS
- **Storage:** 25GB onboard SSD (for OS and application) + 50GB mounted volume (for persistent data).
- **Database:** Self-hosted PostgreSQL 14 with the TimescaleDB extension.

### 3.2. Storage Architecture

**Onboard SSD (25GB):**
- Ubuntu 24.04 OS
- Docker containers
- SEP application binaries
- System logs

**Volume Storage (50GB, mounted at `/mnt/sep_data`):**
- PostgreSQL database (~35GB)
- Daily database backups (~8GB)
- Raw data cache (~5GB)

### 3.3. Database Design (PostgreSQL + TimescaleDB)

This combination is ideal for financial time-series data.

- **Performance:** TimescaleDB provides automatic time-based partitioning (hypertables) and optimized queries for market data. The 8GB of RAM is optimally configured with `shared_buffers = 2GB` and `effective_cache_size = 6GB`.
- **Cost-Effective:** Self-hosting the database on the droplet is significantly cheaper than using a managed database service.
- **Data Integrity:** PostgreSQL's ACID compliance ensures financial data is never lost or corrupted.

### 3.4. Deployment & Data Flow

1.  **Initial Setup:** The `./scripts/deploy_to_droplet.sh` script automates the entire server setup, including software installation and database configuration.
2.  **Daily Operations:**
    - **Local:** Generate trading signals using your GPU.
    - **Sync:** Use `./scripts/sync_to_droplet.sh` to push signals to the remote server.
    - **Execute:** The droplet automatically executes trades based on the received signals.
3.  **Monitoring:** Use the API (`http://<your_droplet_ip>/api/status`) and direct database queries to monitor performance.

### 3.5. Security & Backups

- **Automated Backups:** A cron job should be configured on the droplet to run `pg_dump` daily.
- **Access Control:** Use SSH key-based authentication, a dedicated database user with limited permissions, and a firewall blocking all non-essential ports.