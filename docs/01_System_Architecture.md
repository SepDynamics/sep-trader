# SEP Trader-Bot: System Architecture

**Last Updated:** August 20, 2025

## 1. High-Level Design: Hybrid Local/Remote Architecture

The system utilizes a hybrid architecture to optimize for both performance and cost:

*   **Local Machine (CUDA):** Your local machine, equipped with a powerful NVIDIA GPU, is responsible for all computationally intensive tasks. This includes training quantum models, running backtests, and generating trading signals.
*   **Remote Droplet (CPU):** A lightweight, CPU-only cloud server (e.g., a Digital Ocean Droplet) is used for 24/7 trade execution. It receives signals from your local machine and interacts with the broker's API.

This separation allows for continuous operation without the high cost and maintenance of a 24/7 GPU server.

## 2. Core Technology Components

*   **Quantum Field Harmonics (QFH) Engine:** The core of the trading bot, responsible for analyzing market data and generating trading signals. It's located in `src/core/` and `src/cuda/`.
*   **Data Processing Pipeline:** A robust pipeline for ingesting, processing, and storing market data. It's located in `src/io/` and `src/util/`.
*   **Professional State Management:** A system for managing the state of the trading bot, including configuration, trading pairs, and performance metrics. It's located in `src/core/trading_state_manager.cpp`.

## 3. Data Processing and Enterprise Data Layer

The system processes both real-time and historical market data. The core of the signal processing logic involves treating market data as a complex signal in a bitspace representation.

### Bitspace Metrics Pipeline

1.  **Bitstream Conversion:** Raw market data is converted into a bitstream.
2.  **CUDA Kernel Processing:** The bitstream is processed in parallel on the GPU using QBSA (Quantum Bit State Analysis) and QFH (Quantum Field Harmonics) kernels.
3.  **Damping and Stabilization:** Metrics like coherence, stability, and entropy are repeatedly calculated until they converge to a stable "damped" value.
4.  **Confidence Scoring:** The trajectory of the metrics during the damping process is compared against a database of historical paths to generate a confidence score.

### Enterprise Data Layer

The system uses a robust, enterprise-grade data layer:

*   **PostgreSQL + TimescaleDB:** The primary database for time-series market data.
*   **Valkey:** A high-speed, Redis-compatible in-memory cache for frequently accessed data.
*   **HWLOC:** Used to enable NUMA-aware processing for performance optimization.

## 4. Cloud Deployment and Architecture

An optimized cloud architecture using a Digital Ocean Droplet is recommended:

*   **Droplet Specs:** 8GB RAM, 2 vCPUs, Ubuntu 24.04 LTS, 25GB onboard SSD + 50GB volume storage.
*   **Database:** Self-hosted PostgreSQL 14 with the TimescaleDB extension.
*   **Deployment:** The `./scripts/deploy_to_droplet.sh` script automates the entire server setup.

### Data Flow

1.  **Local:** Generate trading signals using your GPU.
2.  **Sync:** Use `./scripts/sync_to_droplet.sh` to push signals to the remote server.
3.  **Execute:** The droplet automatically executes trades based on the received signals.

## 5. Service Interfaces & API

The SEP DSL provides a rich set of built-in functions for interacting with the engine. These functions are organized by category (AGI Engine, Math, Statistical, etc.) and are available for use in your trading strategies.

For a complete list of functions, please refer to the `02_Core_Technology.md` document.

## 6. Architectural Improvement Roadmap

A comprehensive architectural improvement plan is in place to address the following areas:

1.  **CUDA Implementation Consolidation**
2.  **Legacy Mock References Cleared** *(completed)*
3.  **Unified Quantum Service Architecture**
4.  **Comprehensive Testing Framework**
5.  **Memory Tier System Optimization**

This roadmap will transform the SEP Engine into a more robust, maintainable, and high-performance system.
