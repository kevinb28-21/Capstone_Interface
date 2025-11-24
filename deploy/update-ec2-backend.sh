#!/bin/bash
# Script to update backend on EC2 from local machine
# This script will SSH into EC2 and run the update script

set -e

KEY_FILE="$HOME/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-223-169-5.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"

echo "=========================================="
echo "Updating Backend on EC2"
echo "=========================================="
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found at $KEY_FILE"
    echo "Please update KEY_FILE path in this script or place MS04_ID.pem in ~/Downloads/"
    exit 1
fi

# Make key file readable only by owner
chmod 400 "$KEY_FILE"

echo "Step 1: Transferring update script to EC2..."
scp -i "$KEY_FILE" deploy/update-backend.sh "$EC2_USER@$EC2_HOST:~/"

echo ""
echo "Step 2: Running update script on EC2..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
chmod +x ~/update-backend.sh
~/update-backend.sh
ENDSSH

echo ""
echo "=========================================="
echo "Backend update complete!"
echo "=========================================="
echo ""
echo "You can check the status by SSHing into EC2:"
echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
echo "  pm2 status"
echo "  pm2 logs"

