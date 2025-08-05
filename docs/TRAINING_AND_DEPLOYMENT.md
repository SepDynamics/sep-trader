# Training and Deployment Process

This document outlines the process for training a new currency pair, deploying it to the remote server, and activating the trading bot.

## Syncing to the Droplet

Once you have trained your model and are satisfied with the results, you can sync the data to the remote server using the `sync_to_droplet.sh` script. This script will transfer the `output/`, `config/`, and `models/` directories to the `/opt/sep-trader` directory on the droplet.

```bash
./scripts/sync_to_droplet.sh
```

## Enabling the Trading Bot

Once the data has been synced to the droplet, you can enable the trading bot by SSHing into the droplet and running the `trader-cli` executable with the `trade` command.

```bash
ssh root@165.227.109.187

cd /opt/sep-trader

./trader-cli trade --config config/demo_trading.json
```
