#!/bin/bash
# Complete EC2 backend update script
# This updates all backend components with the latest fixes

set -e

KEY_FILE="$HOME/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-223-169-5.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

echo "=========================================="
echo "Complete EC2 Backend Update"
echo "=========================================="
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    print_error "Key file not found at $KEY_FILE"
    print_warning "Please ensure MS04_ID.pem is in ~/Downloads/"
    exit 1
fi

chmod 400 "$KEY_FILE"

# Test SSH connection
print_info "Testing SSH connection..."
if ssh -i "$KEY_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" 2>/dev/null; then
    print_status "SSH connection successful"
else
    print_error "SSH connection failed!"
    echo ""
    echo "Please use EC2 Instance Connect instead:"
    echo "  1. Go to AWS Console → EC2 → Instances"
    echo "  2. Select instance: i-0ce6adb51ca9c5a4d"
    echo "  3. Click 'Connect' → 'EC2 Instance Connect'"
    echo "  4. Run these commands:"
    echo "     cd ~/Capstone_Interface"
    echo "     git pull origin main"
    echo "     ./deploy/fix-ec2-processing.sh"
    echo ""
    exit 1
fi

# Step 1: Sync all backend files
print_info "Step 1: Syncing backend files..."
rsync -avz --progress -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    --exclude 'node_modules' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '*.log' \
    --exclude 'models/*.h5' \
    --exclude 'models/*.pkl' \
    --exclude 'uploads' \
    --exclude 'processed' \
    "$PROJECT_DIR/server/" "$EC2_USER@$EC2_HOST:~/Capstone_Interface/server/" && \
    print_status "Server files synced"

rsync -avz --progress -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude '*.log' \
    --exclude 'models/*.h5' \
    --exclude 'models/*.pkl' \
    --exclude 'uploads' \
    --exclude 'processed' \
    "$PROJECT_DIR/python_processing/" "$EC2_USER@$EC2_HOST:~/Capstone_Interface/python_processing/" && \
    print_status "Python processing files synced"

# Step 2: Sync deployment scripts
print_info "Step 2: Syncing deployment scripts..."
rsync -avz --progress -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    "$PROJECT_DIR/deploy/" "$EC2_USER@$EC2_HOST:~/Capstone_Interface/deploy/" && \
    print_status "Deployment scripts synced"

# Step 3: Run update on EC2
print_info "Step 3: Running update on EC2..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << 'ENDSSH'
cd ~/Capstone_Interface

# Make scripts executable
chmod +x deploy/*.sh 2>/dev/null || true

# Run the fix script
if [ -f "deploy/fix-ec2-processing.sh" ]; then
    ./deploy/fix-ec2-processing.sh
else
    echo "Running manual update..."
    
    # Update Node.js dependencies
    if [ -d "server" ]; then
        cd server
        npm install --production
        cd ..
    fi
    
    # Update Python dependencies
    if [ -d "python_processing/venv" ]; then
        cd python_processing
        source venv/bin/activate
        pip install -r requirements.txt --quiet 2>/dev/null || true
        deactivate
        cd ..
    fi
    
    # Restart PM2 services
    if command -v pm2 &> /dev/null; then
        pm2 restart all
        pm2 save
    fi
fi
ENDSSH

print_status "EC2 update complete!"

echo ""
echo "=========================================="
print_status "Backend update complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Check PM2 status: ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'pm2 status'"
echo "  2. View logs: ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'pm2 logs --lines 50'"
echo "  3. Test API: curl http://$EC2_HOST/api/health"

