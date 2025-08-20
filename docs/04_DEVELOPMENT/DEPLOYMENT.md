# SEP Professional Trader-Bot Deployment Guide

This document outlines the steps to deploy the SEP Professional Trader-Bot to a remote server.

## Prerequisites

- A remote server (droplet) with Docker and docker-compose installed.
- SSH access to the remote server.
- The local project repository cloned and up-to-date.

## Deployment Steps

1.  **Configure Deployment Script:**
    - The primary deployment script is `scripts/start_droplet_services.sh`.
    - Ensure the `DROPLET_IP` and `DEPLOY_USER` variables at the top of the script are set correctly for your remote server.

2.  **Deployment Configuration:**
    - The deployment is managed by `docker-compose.yml` and configured with `nginx.conf`.
    - These files were missing and have been created with default configurations.
    - `docker-compose.yml`: Defines the `sep-trader` and `sep-nginx` services.
    - `nginx.conf`: Configures nginx as a reverse proxy to the trading service.

3.  **Run the Deployment Script:**
    - From the root of the project repository, execute the following command:
      ```bash
      ./scripts/start_droplet_services.sh
      ```
    - This script will:
        - SSH into the remote server.
        - Build the Docker images for the services.
        - Start the services using `docker-compose`.
        - Display the status and logs of the running services.

4.  **Verify the Deployment:**
    - The script will automatically test the `health` and `api/status` endpoints.
    - You can also manually check the service URLs provided at the end of the script's output.

## Troubleshooting

- **`KeyError: 'ContainerConfig'`:** This error may occur with older versions of `docker-compose`. The new `docker-compose.yml` file should mitigate this issue. If it persists, consider upgrading `docker-compose` on the remote server.
- **`HTTP 501 Not Implemented`:** This error indicates a problem with the trading service's HTTP server. The `scripts/trading_service.py` has been updated with additional logging to diagnose this issue.
- **`404 Not Found`:** This error from nginx indicates a problem with the reverse proxy configuration. The `nginx.conf` file has been created to correctly route requests.
