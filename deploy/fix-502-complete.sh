#!/bin/bash
# Complete fix for 502 Bad Gateway error
# This script SSH into EC2 and fixes the backend

set -e

# Configuration
KEY_FILE="${HOME}/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-223-169-5.us-east-2.compute.amazonaws.com"
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

print_header "502 Bad Gateway - Complete Fix"
echo "This will SSH into EC2 and fix the backend issue."
echo ""

# SSH into EC2 and run fix commands
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
set -e

# Colors for remote output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

echo "=========================================="
echo "Diagnosing 502 Bad Gateway Issue"
echo "=========================================="
echo ""

# Step 1: Check PM2 status
print_info "Step 1: Checking PM2 status..."
if ! command -v pm2 &> /dev/null; then
    print_error "PM2 is not installed!"
    print_info "Installing PM2..."
    sudo npm install -g pm2
    print_success "PM2 installed"
fi

PM2_LIST=$(pm2 list 2>/dev/null || echo "")
if echo "$PM2_LIST" | grep -q "drone-backend"; then
    print_success "Backend process found in PM2"
    pm2 list | grep -E "drone-backend|server"
else
    print_warning "Backend process NOT found in PM2"
fi
echo ""

# Step 2: Check if port 5050 is listening
print_info "Step 2: Checking port 5050..."
if sudo netstat -tlnp 2>/dev/null | grep -q ":5050" || sudo ss -tlnp 2>/dev/null | grep -q ":5050"; then
    print_success "Port 5050 is listening"
    sudo netstat -tlnp 2>/dev/null | grep ":5050" || sudo ss -tlnp 2>/dev/null | grep ":5050"
else
    print_error "Port 5050 is NOT listening"
fi
echo ""

# Step 3: Test backend health
print_info "Step 3: Testing backend health endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5050/api/health 2>/dev/null || echo "000")
if [ "$HEALTH_RESPONSE" = "200" ]; then
    print_success "Backend is responding (HTTP 200)"
    curl -s http://localhost:5050/api/health | head -3
else
    print_error "Backend is NOT responding (HTTP $HEALTH_RESPONSE)"
fi
echo ""

# Step 4: Check backend logs if not working
if [ "$HEALTH_RESPONSE" != "200" ]; then
    print_info "Step 4: Checking backend logs..."
    if echo "$PM2_LIST" | grep -q "drone-backend"; then
        print_info "Recent backend logs (last 20 lines):"
        pm2 logs drone-backend --lines 20 --nostream 2>/dev/null || print_warning "Could not retrieve logs"
    else
        print_warning "No backend process to check logs"
    fi
    echo ""
fi

# Step 5: Navigate to server directory
print_info "Step 5: Navigating to server directory..."
cd ~/Capstone_Interface/server || {
    print_error "Server directory not found!"
    exit 1
}
print_success "In server directory"
echo ""

# Step 6: Check .env file
print_info "Step 6: Checking .env file..."
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating template..."
    cat > .env << 'EOF'
PORT=5050
NODE_ENV=production
ORIGIN=http://localhost:5173,http://localhost:5182,https://cropview.netlify.app

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=changeme

# AWS S3 (optional)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION=us-east-2
# S3_BUCKET_NAME=
EOF
    print_warning "Created .env template. You may need to edit it with actual values."
    print_info "Edit with: nano ~/Capstone_Interface/server/.env"
else
    print_success ".env file exists"
fi
echo ""

# Step 7: Install dependencies if needed
print_info "Step 7: Checking dependencies..."
if [ ! -d node_modules ]; then
    print_warning "node_modules not found. Installing dependencies..."
    npm install --production
    print_success "Dependencies installed"
else
    print_success "Dependencies already installed"
fi
echo ""

# Step 8: Stop and restart backend
print_info "Step 8: Restarting backend..."
pm2 delete drone-backend 2>/dev/null || print_warning "No existing process to delete"

# Create logs directory
mkdir -p logs

# Start backend with PM2
if [ -f ecosystem.config.cjs ]; then
    print_info "Starting backend with ecosystem.config.cjs..."
    pm2 start ecosystem.config.cjs
elif [ -f ecosystem.config.js ]; then
    print_info "Starting backend with ecosystem.config.js..."
    pm2 start ecosystem.config.js
else
    print_error "No PM2 config found! Creating ecosystem.config.cjs..."
    cat > ecosystem.config.cjs << 'EOF'
module.exports = {
  apps: [{
    name: 'drone-backend',
    script: 'src/server.js',
    instances: 1,
    exec_mode: 'fork',
    env: {
      NODE_ENV: 'production',
      PORT: 5050
    },
    error_file: './logs/err.log',
    out_file: './logs/out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    max_memory_restart: '1G',
    watch: false,
    ignore_watch: ['node_modules', 'logs']
  }]
};
EOF
    pm2 start ecosystem.config.cjs
    print_success "Created and started backend"
fi

pm2 save
print_success "Backend started with PM2"
echo ""

# Step 9: Wait for backend to start
print_info "Step 9: Waiting for backend to start (5 seconds)..."
sleep 5
echo ""

# Step 10: Verify backend is running
print_info "Step 10: Verifying backend is running..."
PM2_STATUS=$(pm2 list 2>/dev/null | grep "drone-backend" || echo "")
if echo "$PM2_STATUS" | grep -q "online"; then
    print_success "Backend is online"
else
    print_error "Backend is not online!"
    print_info "PM2 status:"
    pm2 list | grep "drone-backend" || echo "Not found"
    print_info "Recent logs:"
    pm2 logs drone-backend --lines 30 --nostream 2>/dev/null || echo "No logs"
fi
echo ""

# Step 11: Test backend health again
print_info "Step 11: Testing backend health endpoint again..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5050/api/health 2>/dev/null || echo "000")
if [ "$HEALTH_RESPONSE" = "200" ]; then
    print_success "Backend health check passed!"
    curl -s http://localhost:5050/api/health | head -5
else
    print_error "Backend health check failed (HTTP $HEALTH_RESPONSE)"
    print_info "Checking logs for errors..."
    pm2 logs drone-backend --lines 30 --nostream 2>/dev/null || echo "No logs"
fi
echo ""

# Step 12: Check nginx
print_info "Step 12: Checking nginx..."
if ! command -v nginx &> /dev/null; then
    print_error "Nginx is not installed!"
else
    print_success "Nginx is installed"
    
    # Test nginx config
    if sudo nginx -t 2>&1 | grep -q "successful"; then
        print_success "Nginx configuration is valid"
    else
        print_error "Nginx configuration has errors:"
        sudo nginx -t
    fi
    
    # Restart nginx
    print_info "Restarting nginx..."
    sudo systemctl restart nginx
    sleep 2
    
    if sudo systemctl is-active --quiet nginx; then
        print_success "Nginx is running"
    else
        print_error "Nginx failed to start!"
        sudo systemctl status nginx --no-pager | head -10
    fi
fi
echo ""

# Step 13: Final verification
echo "=========================================="
echo "Final Verification"
echo "=========================================="
echo ""

print_info "PM2 Status:"
pm2 list | grep -E "drone-backend|server" || echo "No backend process"

echo ""
print_info "Port 5050:"
sudo netstat -tlnp 2>/dev/null | grep ":5050" || sudo ss -tlnp 2>/dev/null | grep ":5050" || print_warning "Port 5050 not listening"

echo ""
print_info "Backend Health:"
curl -s http://localhost:5050/api/health 2>/dev/null | head -3 || print_error "Backend not responding"

echo ""
print_info "Nginx Status:"
sudo systemctl is-active --quiet nginx && print_success "Nginx is running" || print_error "Nginx is not running"

echo ""
print_info "Testing through nginx:"
curl -s http://localhost/api/health 2>/dev/null | head -3 || print_warning "Not responding through nginx"

echo ""
echo "=========================================="
echo "Fix Complete!"
echo "=========================================="
echo ""
echo "If issues persist:"
echo "1. Check backend logs: pm2 logs drone-backend --lines 50"
echo "2. Check nginx logs: sudo tail -f /var/log/nginx/error.log"
echo "3. Verify .env: cat ~/Capstone_Interface/server/.env"
echo "4. Test backend: curl http://localhost:5050/api/health"
echo ""

ENDSSH

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    print_success "Fix script completed successfully!"
    echo ""
    echo "Test your backend:"
    echo "  http://18.223.169.5/api/health"
    echo ""
    echo "Then try uploading an image on your analytics page."
else
    echo ""
    print_error "Some errors occurred. Check the output above."
    echo ""
    echo "You can also SSH manually:"
    echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
fi

