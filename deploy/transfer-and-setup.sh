#!/bin/bash
# Transfer code to EC2 and run setup
# This avoids git clone issues with private repos

set -e

KEY_FILE="${HOME}/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-117-90-212.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check key file
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found at: $KEY_FILE"
    exit 1
fi

chmod 400 "$KEY_FILE" 2>/dev/null || true

print_info "Transferring code to EC2..."
print_warning "This may take a few minutes..."

# Create a tarball excluding unnecessary files
cd /Users/kevinbhatt/Desktop/Projects/Capstone_Interface
tar --exclude='node_modules' \
    --exclude='venv' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='dist' \
    --exclude='.env' \
    -czf /tmp/capstone-code.tar.gz .

# Transfer to EC2
scp -i "$KEY_FILE" /tmp/capstone-code.tar.gz "$EC2_USER@$EC2_HOST:~/capstone-code.tar.gz"
print_success "Code transferred"

# Transfer setup script
scp -i "$KEY_FILE" deploy/setup-new-ec2.sh "$EC2_USER@$EC2_HOST:~/setup-new-ec2.sh"
print_success "Setup script transferred"

# Run setup on EC2
print_info "Extracting code and running setup on EC2..."
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
set -e

# Extract code
if [ -d ~/Capstone_Interface ]; then
    echo "Removing existing directory..."
    rm -rf ~/Capstone_Interface
fi

mkdir -p ~/Capstone_Interface
cd ~/Capstone_Interface
tar -xzf ~/capstone-code.tar.gz
rm ~/capstone-code.tar.gz
echo "✓ Code extracted"

# Make setup script executable
chmod +x ~/setup-new-ec2.sh

# Run setup (skip git clone step)
echo "Running setup..."
bash ~/setup-new-ec2.sh

ENDSSH

print_success "Setup complete!"
echo ""
echo "Test your backend:"
echo "  curl http://18.117.90.212/api/health"

