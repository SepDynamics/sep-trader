# Cloud Deployment Guide

This guide provides a comprehensive overview of how to deploy the SEP Professional Trader-Bot to a cloud environment. 

## Deployment Architecture

The system uses a hybrid architecture that separates heavy computation from lightweight trade execution.

```
┌─────────────────┐    Data Sync    ┌─────────────────┐
│   Local PC      │ ───────────────► │  Cloud Server   │
│   (Compute)     │                  │   (Execution)   │
├─────────────────┤                  ├─────────────────┤
│ • CUDA Analysis │                  │ • Live Trading  │
│ • QFH Engine    │                  │ • Market Data   │
│ • Model Training│                  │ • Cache DB      │
│ • Metrics Gen   │                  │ • OANDA API     │
└─────────────────┘                  └─────────────────┘
```

- **Local Machine:** Your local, CUDA-enabled machine is used for all computationally intensive tasks like model training and signal generation.
- **Cloud Server (Droplet):** A CPU-only cloud server (like a Digital Ocean Droplet) is used for 24/7 trade execution, data storage, and API access.

---

## Server Requirements

- **OS:** Ubuntu 24.04 LTS
- **RAM:** 8GB+
- **vCPUs:** 2+
- **Storage:** 25GB OS disk + 50GB mounted volume for data.

---

## Step 1: Provision and Secure the Server

1.  **Create a Droplet:** Provision a new Digital Ocean Droplet (or any other cloud server) with the specifications listed above.

2.  **Initial Connection:** SSH to the new server to ensure your access key is working.
    ```bash
    ssh root@<your_droplet_ip>
    ```

3.  **Run Automated Deployment Script:**
    From your **local machine**, run the deployment script. This will install all necessary software on the remote server, including PostgreSQL, TimescaleDB, Docker, Nginx, and configure the firewall.
    ```bash
    # In the project root on your local machine
    ./scripts/deploy_to_droplet.sh --ip <your_droplet_ip>
    ```

---

## Step 2: Configure the Trading Environment

1.  **SSH to Server:** Connect to your newly provisioned server.
    ```bash
    ssh root@<your_droplet_ip>
    ```

2.  **Set OANDA Credentials:**
    Navigate to the configuration directory and create a file for your OANDA credentials.
    ```bash
    cd /opt/sep-trader
    nano config/OANDA.env
    ```
    Add your credentials to the file:
    ```env
    OANDA_API_KEY=your_api_key_here
    OANDA_ACCOUNT_ID=your_account_id_here
    OANDA_ENVIRONMENT=practice # or live
    ```

3.  **Initialize the Database:**
    Run the database initialization script.
    ```bash
    cd /opt/sep-trader/sep-trader
    sudo -u postgres psql sep_trading < scripts/init_database.sql
    ```

---

## Step 3: Start and Monitor Services

1.  **Start Services:**
    Use `docker-compose` to start all the trading services in the background.
    ```bash
    cd /opt/sep-trader/sep-trader
    docker-compose up -d
    ```

2.  **Verify Health:**
    Check the health and status endpoints to ensure everything is running correctly.
    ```bash
    # From the server or your local machine
    curl http://<your_droplet_ip>/health
    curl http://<your_droplet_ip>/api/status
    ```

3.  **Monitor Logs:**
    To see the live logs from the trading container:
    ```bash
    # From the /opt/sep-trader/sep-trader directory on the server
    docker-compose logs -f
    ```

---

## Operational Workflow

The daily workflow consists of generating signals locally and syncing them to the cloud server for execution.

1.  **Generate Signals Locally:**
    Use the tools on your local machine to train models and generate trading signals.

2.  **Synchronize Data:**
    Run the sync script from your local machine to push the new signals and models to the cloud server.
    ```bash
    ./scripts/sync_to_droplet.sh
    ```

3.  **Monitor Remotely:**
    Use the API endpoints and logs on the cloud server to monitor trading activity and performance.
