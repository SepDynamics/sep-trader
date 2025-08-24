# Syncing Training Results to the Trading Droplet

After running local CUDA training, push the generated metrics and models to the remote droplet:

```bash
./scripts/sync_to_droplet.sh
```

This script copies the latest outputs to `/opt/sep-trader/data` on the droplet.

## Import Metrics into Valkey

Use SSH to load the newest metric snapshot into Valkey on the droplet. Replace `<PAIR>` with the currency pair you trained:

```bash
ssh root@YOUR_DROPLET_IP "cd /opt/sep-trader/sep-trader && ./build/src/cli/trader-cli data import /opt/sep-trader/data/latest_metrics_<PAIR>.rdb"
```

## Troubleshooting

### Missing metrics file
- Ensure the sync script completed successfully.
- Verify the file exists on the droplet:
  ```bash
  ssh root@YOUR_DROPLET_IP "ls /opt/sep-trader/data"
  ```
  Confirm `latest_metrics_<PAIR>.rdb` appears with the correct pair name.

### Valkey connection errors
- The import command reports issues like `Connection refused` or `Unable to reach Valkey` when the Valkey service is offline.
- Check Valkey status and restart if needed:
  ```bash
  ssh root@YOUR_DROPLET_IP "docker-compose ps valkey"
  ssh root@YOUR_DROPLET_IP "docker-compose restart valkey"
  ```
- Run `valkey-cli ping` to confirm connectivity before re-running the import.
