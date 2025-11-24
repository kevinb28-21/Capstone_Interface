#!/bin/bash
# Script to sync backend files to EC2 (alternative to git pull)
# This transfers only the changed files directly

set -e

KEY_FILE="$HOME/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-223-169-5.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================="
echo "Syncing Backend Files to EC2"
echo "=========================================="
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found at $KEY_FILE"
    exit 1
fi

chmod 400 "$KEY_FILE"

echo "Step 1: Syncing server files..."
rsync -avz --progress -e "ssh -i $KEY_FILE" \
    --exclude 'node_modules' \
    --exclude '.env' \
    --exclude '*.log' \
    "$PROJECT_DIR/server/" "$EC2_USER@$EC2_HOST:~/Capstone_Interface/server/"

echo ""
echo "Step 2: Syncing Python processing files..."
rsync -avz --progress -e "ssh -i $KEY_FILE" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '*.log' \
    --exclude 'models' \
    "$PROJECT_DIR/python_processing/" "$EC2_USER@$EC2_HOST:~/Capstone_Interface/python_processing/"

echo ""
echo "Step 3: Installing dependencies and restarting services..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
cd ~/Capstone_Interface

# Update Node.js dependencies
echo "Updating Node.js dependencies..."
cd server
npm install --production
cd ..

# Update Python dependencies (if venv exists)
if [ -d "python_processing/venv" ]; then
    echo "Updating Python dependencies..."
    cd python_processing
    source venv/bin/activate
    pip install -r requirements.txt --quiet 2>/dev/null || true
    deactivate
    cd ..
fi

# Restart PM2 services
if command -v pm2 &> /dev/null; then
    echo "Restarting PM2 services..."
    pm2 restart all
    echo ""
    echo "PM2 Status:"
    pm2 status
else
    echo "PM2 not found. Please restart services manually."
fi
ENDSSH

echo ""
echo "=========================================="
echo "Backend sync complete!"
echo "=========================================="

