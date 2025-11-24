#!/bin/bash
# Run database migration on EC2 remotely
# Usage: ./run-migration-remote.sh [key-file] [ec2-user@ec2-host]

set -e

KEY_FILE="${1:-$HOME/Downloads/MS04_ID.pem}"
EC2_HOST="${2:-ubuntu@ec2-18-223-169-5.us-east-2.compute.amazonaws.com}"

if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found: $KEY_FILE"
    echo "Usage: $0 [key-file] [ec2-user@ec2-host]"
    exit 1
fi

echo "Transferring migration script to EC2..."
scp -i "$KEY_FILE" deploy/run-migration.sh "$EC2_HOST:~/run-migration.sh"

echo "Running migration on EC2..."
ssh -i "$KEY_FILE" "$EC2_HOST" "chmod +x ~/run-migration.sh && ~/run-migration.sh"

echo "✓ Migration completed on EC2"

