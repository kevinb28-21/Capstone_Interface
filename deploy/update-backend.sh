#!/bin/bash
# Script to update backend on EC2 instance
# Run this script on your EC2 instance after connecting via SSH

set -e

echo "=========================================="
echo "Updating Backend on EC2"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
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

# Check if we're in the project directory
if [ ! -d "Capstone_Interface" ]; then
    print_error "Capstone_Interface directory not found!"
    print_warning "Make sure you're in the home directory or project directory"
    exit 1
fi

cd Capstone_Interface

# Step 1: Pull latest changes from GitHub
echo ""
echo "Step 1: Pulling latest changes from GitHub..."
if git pull origin main; then
    print_status "Code updated from GitHub"
else
    print_error "Failed to pull from GitHub"
    print_warning "Make sure you have git configured and the repository is set up"
    exit 1
fi

# Step 2: Update Node.js dependencies
echo ""
echo "Step 2: Updating Node.js dependencies..."
cd server
if [ -f "package.json" ]; then
    npm install
    print_status "Node.js dependencies updated"
else
    print_warning "package.json not found in server directory"
fi
cd ..

# Step 3: Update Python dependencies (if needed)
echo ""
echo "Step 3: Checking Python dependencies..."
cd python_processing
if [ -f "requirements.txt" ]; then
    if [ -d "venv" ]; then
        source venv/bin/activate
        pip install -r requirements.txt --quiet
        print_status "Python dependencies updated"
        deactivate
    else
        print_warning "Python virtual environment not found"
    fi
else
    print_warning "requirements.txt not found"
fi
cd ..

# Step 4: Restart PM2 services
echo ""
echo "Step 4: Restarting PM2 services..."
if command -v pm2 &> /dev/null; then
    pm2 restart all
    print_status "PM2 services restarted"
    
    echo ""
    echo "Current PM2 status:"
    pm2 status
    
    echo ""
    echo "Recent logs (last 20 lines):"
    pm2 logs --lines 20 --nostream
else
    print_warning "PM2 not found. Services may need to be started manually"
fi

echo ""
echo "=========================================="
print_status "Backend update complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Check PM2 status: pm2 status"
echo "2. View logs: pm2 logs"
echo "3. Test the API: curl http://localhost:5050/api/images"

