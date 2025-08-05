#!/bin/bash

# SEP Droplet Volume Setup Script
# Sets up the 50GB volume and initializes the data directories

set -e

echo "🔧 Setting up SEP droplet volume and data directories..."

# Create mount point
mkdir -p /mnt/volume_nyc3_01

# Mount the volume
mount -o discard,defaults,noatime /dev/disk/by-id/scsi-0DO_Volume_volume-nyc3-01 /mnt/volume_nyc3_01

# Add to fstab for persistent mounting
echo '/dev/disk/by-id/scsi-0DO_Volume_volume-nyc3-01 /mnt/volume_nyc3_01 ext4 defaults,nofail,discard 0 0' | tee -a /etc/fstab

# Create SEP data directory structure
mkdir -p /mnt/volume_nyc3_01/sep-data/{training,market_data,models,backups,logs}
mkdir -p /mnt/volume_nyc3_01/sep-data/postgres/{data,backups}
mkdir -p /mnt/volume_nyc3_01/sep-data/redis

# Set proper permissions
chmod -R 755 /mnt/volume_nyc3_01/sep-data
chown -R postgres:postgres /mnt/volume_nyc3_01/sep-data/postgres || true

# Create symlinks for easier access
ln -sf /mnt/volume_nyc3_01/sep-data /opt/sep-data

echo "✅ Volume setup complete!"
echo "📁 Data directories created at /mnt/volume_nyc3_01/sep-data"
echo "🔗 Symlink available at /opt/sep-data"
