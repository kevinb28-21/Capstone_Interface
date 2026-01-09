#!/bin/bash
# Deploy CORS fixes to EC2 backend
# This script syncs the updated server files with Netlify CORS support

set -e

KEY_FILE="$HOME/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-117-90-212.us-east-2.compute.amazonaws.com"
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
echo "Deploying CORS Fixes to EC2 Backend"
echo "=========================================="
echo ""

# Check if key file exists
if [ ! -f "$KEY_FILE" ]; then
    print_error "Key file not found at $KEY_FILE"
    print_warning "Please ensure MS04_ID.pem is in ~/Downloads/"
    echo ""
    echo "Alternative: Use EC2 Instance Connect:"
    echo "  1. Go to AWS Console → EC2 → Instances"
    echo "  2. Select your instance"
    echo "  3. Click 'Connect' → 'EC2 Instance Connect'"
    echo "  4. Run: cd ~/Capstone_Interface && git pull origin main"
    exit 1
fi

chmod 400 "$KEY_FILE"

# Test SSH connection
print_info "Testing SSH connection to $EC2_HOST..."
if ssh -i "$KEY_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" 2>/dev/null; then
    print_status "SSH connection successful"
else
    print_error "SSH connection failed!"
    echo ""
    echo "Please use EC2 Instance Connect instead:"
    echo "  1. Go to AWS Console → EC2 → Instances"
    echo "  2. Select your instance"
    echo "  3. Click 'Connect' → 'EC2 Instance Connect'"
    echo "  4. Run these commands:"
    echo "     cd ~/Capstone_Interface"
    echo "     git pull origin main"
    echo "     cd server && npm install --production"
    echo "     pm2 restart all"
    echo ""
    exit 1
fi

# Step 1: Sync server files (with CORS fixes)
print_info "Step 1: Syncing updated server files..."
rsync -avz --progress -e "ssh -i $KEY_FILE -o StrictHostKeyChecking=no" \
    --exclude 'node_modules' \
    --exclude '.env' \
    --exclude '*.log' \
    --exclude 'uploads' \
    "$PROJECT_DIR/server/src/" "$EC2_USER@$EC2_HOST:~/Capstone_Interface/server/src/" && \
    print_status "Server source files synced"

# Step 2: Update dependencies and restart
print_info "Step 2: Updating dependencies and restarting services..."
ssh -i "$KEY_FILE" -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" << 'ENDSSH'
cd ~/Capstone_Interface

echo "Updating Node.js dependencies..."
cd server
npm install --production
cd ..

# Restart PM2 services
if command -v pm2 &> /dev/null; then
    echo "Restarting PM2 services..."
    pm2 restart all
    pm2 save
    
    echo ""
    echo "PM2 Status:"
    pm2 status
    
    echo ""
    echo "Recent logs:"
    pm2 logs --lines 10 --nostream
else
    echo "PM2 not found. Please restart services manually."
fi

# Test health endpoint
echo ""
echo "Testing API health endpoint..."
curl -s http://localhost:5050/api/health | head -20 || echo "Health check failed"
ENDSSH

print_status "Backend deployment complete!"
echo ""
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""
echo "To verify the deployment:"
echo "  1. Check PM2 status:"
echo "     ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'pm2 status'"
echo ""
echo "  2. View logs:"
echo "     ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'pm2 logs --lines 50'"
echo ""
echo "  3. Test API from local machine:"
echo "     curl http://$EC2_HOST/api/health"
echo ""
echo "  4. Test CORS (should allow Netlify domains):"
echo "     curl -H 'Origin: https://your-site.netlify.app' \\"
echo "          -H 'Access-Control-Request-Method: GET' \\"
echo "          -X OPTIONS \\"
echo "          http://$EC2_HOST/api/health"
echo ""

