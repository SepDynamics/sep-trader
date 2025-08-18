# System Architecture

This document describes the system architecture of the SEP Professional Trader-Bot, including the high-level design and a recommended, optimized implementation.

## 1. High-Level Design: Hybrid Local/Remote

The system uses a hybrid architecture to optimize for both performance and cost.

- **Local Machine (CUDA):** Your local machine, equipped with a powerful NVIDIA GPU, is responsible for all computationally intensive tasks. This includes training quantum models, running backtests, and generating trading signals.
- **Remote Droplet (CPU):** A lightweight, CPU-only cloud server is used for 24/7 trade execution. It receives signals from your local machine and interacts with the broker's API. This allows for continuous operation without the high cost and maintenance of a 24/7 GPU server.

This separation is managed by a `HybridTradingSystem` class that coordinates the training, deployment, and synchronization of data between the two environments.

## 2. Professional State Management

The system is designed for robust and reliable state management.

- **Hot-Swappable Configuration:** System and trading parameters can be updated in real-time without requiring a restart.
- **Persistent State:** The trading state, including open positions and historical trades, is persisted to a database with ACID properties to ensure data integrity.
- **Real-Time Monitoring:** Health metrics and trading status are available via API endpoints.

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