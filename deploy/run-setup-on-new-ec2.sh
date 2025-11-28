#!/bin/bash
# Run setup on new EC2 instance from local machine
# This script transfers and runs the setup script on your new EC2 instance

set -e

# Configuration
KEY_FILE="${HOME}/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-117-90-212.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}==========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check key file
if [ ! -f "$KEY_FILE" ]; then
    print_error "Key file not found at: $KEY_FILE"
    echo ""
    echo "Please update KEY_FILE in this script or place your key at:"
    echo "  $KEY_FILE"
    exit 1
fi

chmod 400 "$KEY_FILE" 2>/dev/null || true

print_header "EC2 Setup - Drone Crop Health Platform"
echo "This will SSH into your new EC2 instance and run the complete setup."
echo ""
print_info "EC2 Host: $EC2_HOST"
print_info "Key File: $KEY_FILE"
echo ""

# Test SSH connection first
print_info "Testing SSH connection..."
if ssh -i "$KEY_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" 2>/dev/null; then
    print_success "SSH connection successful"
else
    print_error "SSH connection failed!"
    echo ""
    echo "Please verify:"
    echo "1. Your EC2 instance is running"
    echo "2. Security group allows SSH (port 22) from your IP"
    echo "3. Key file is correct: $KEY_FILE"
    exit 1
fi

echo ""
print_info "Transferring setup script to EC2..."
scp -i "$KEY_FILE" deploy/setup-new-ec2.sh "$EC2_USER@$EC2_HOST:~/setup-new-ec2.sh"
print_success "Setup script transferred"

echo ""
print_info "Making setup script executable..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" "chmod +x ~/setup-new-ec2.sh"
print_success "Script is executable"

echo ""
print_warning "Starting setup on EC2 instance..."
print_warning "This will take 5-10 minutes. Please be patient..."
echo ""

# Run the setup script on EC2
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" "bash ~/setup-new-ec2.sh"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Test backend: curl http://18.117.90.212/api/health"
    echo "2. Update Netlify redirects (already done in netlify.toml)"
    echo "3. Redeploy Netlify site to apply new redirects"
    echo "4. Test image upload from your frontend"
    echo ""
    echo "To check services on EC2:"
    echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
    echo "  pm2 list"
    echo "  pm2 logs"
else
    echo ""
    print_error "Setup encountered errors. Exit code: $EXIT_CODE"
    echo ""
    echo "Please SSH into EC2 and check:"
    echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
    echo "  pm2 list"
    echo "  pm2 logs"
    echo "  sudo systemctl status nginx"
fi

