#!/bin/bash
# Complete Backend Fix Script for EC2
# Run this script on your EC2 instance to fix the 502 error

set -e

echo "=========================================="
echo "Complete Backend Fix for 502 Error"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}▶${NC} $1"
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

# Step 1: Navigate to project directory
print_step "Step 1: Navigating to project directory..."
cd ~/Capstone_Interface || {
    print_error "Project directory not found!"
    echo "Please clone the repository first:"
    echo "  git clone https://github.com/your-username/Capstone_Interface.git ~/Capstone_Interface"
    exit 1
}
print_success "In project directory"

# Step 2: Pull latest changes
print_step "Step 2: Pulling latest changes from Git..."
if git pull origin main 2>&1 | grep -q "Already up to date\|Updating"; then
    print_success "Repository updated"
else
    print_warning "Git pull had issues (may not be a git repo or no changes)"
fi

# Step 3: Check/create .env file
print_step "Step 3: Checking .env file..."
if [ ! -f server/.env ]; then
    print_warning ".env file not found. Creating template..."
    mkdir -p server
    cat > server/.env << 'EOF'
PORT=5050
NODE_ENV=production
ORIGIN=http://localhost:5173,http://localhost:5182

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=changeme

# AWS S3 (optional - uncomment and fill if using S3)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION=us-east-2
# S3_BUCKET_NAME=
EOF
    print_warning "Created .env template. Please edit it with your actual values:"
    echo "  nano ~/Capstone_Interface/server/.env"
    echo ""
    read -p "Press Enter after editing .env file, or Ctrl+C to exit and edit manually..."
else
    print_success ".env file exists"
fi

# Step 4: Install/update dependencies
print_step "Step 4: Installing Node.js dependencies..."
cd server
if [ ! -d node_modules ]; then
    print_warning "node_modules not found. Installing dependencies..."
    npm install --production
else
    print_success "Dependencies already installed"
    # Still update to be safe
    npm install --production --silent 2>/dev/null || true
fi
cd ..

# Step 5: Ensure PM2 is installed
print_step "Step 5: Checking PM2 installation..."
if ! command -v pm2 &> /dev/null; then
    print_warning "PM2 not installed. Installing..."
    sudo npm install -g pm2
    print_success "PM2 installed"
else
    print_success "PM2 is installed"
fi

# Step 6: Stop existing backend process
print_step "Step 6: Stopping existing backend process..."
pm2 delete drone-backend 2>/dev/null || print_warning "No existing process to stop"
sleep 1

# Step 7: Create logs directory
print_step "Step 7: Creating logs directory..."
mkdir -p server/logs
print_success "Logs directory ready"

# Step 8: Start backend with PM2 (prefer .cjs, fallback to .js)
print_step "Step 8: Starting backend with PM2..."
cd server

# Try .cjs first (CommonJS - more reliable)
if [ -f ecosystem.config.cjs ]; then
    print_success "Using ecosystem.config.cjs (CommonJS format)"
    pm2 start ecosystem.config.cjs
elif [ -f ecosystem.config.js ]; then
    print_success "Using ecosystem.config.js"
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
    print_success "Created and started with ecosystem.config.cjs"
fi

pm2 save
cd ..
print_success "Backend started with PM2"

# Step 9: Wait for backend to start
print_step "Step 9: Waiting for backend to start..."
sleep 5

# Step 10: Check PM2 status
print_step "Step 10: Checking PM2 status..."
pm2 list
echo ""

# Step 11: Check if port 5050 is listening
print_step "Step 11: Checking if port 5050 is listening..."
if sudo netstat -tlnp 2>/dev/null | grep -q ":5050" || sudo ss -tlnp 2>/dev/null | grep -q ":5050"; then
    print_success "Port 5050 is listening"
else
    print_error "Port 5050 is NOT listening!"
    echo "Checking backend logs..."
    pm2 logs drone-backend --lines 20 --nostream
    print_warning "Backend may have crashed. Check logs above."
fi

# Step 12: Test backend health endpoint
print_step "Step 12: Testing backend health endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5050/api/health 2>/dev/null || echo "000")

if [ "$HEALTH_RESPONSE" = "200" ]; then
    print_success "Backend health check passed!"
    echo "Response:"
    curl -s http://localhost:5050/api/health | head -5
    echo ""
else
    print_error "Backend health check failed (HTTP $HEALTH_RESPONSE)"
    echo ""
    echo "Recent backend logs:"
    pm2 logs drone-backend --lines 30 --nostream
    echo ""
    print_warning "Trying to diagnose the issue..."
    
    # Check if backend can start manually
    if [ -f server/src/server.js ]; then
        echo "Testing manual startup (5 second timeout)..."
        cd server
        timeout 5 node src/server.js 2>&1 | head -20 || echo "Backend failed to start - check errors above"
        cd ..
    fi
fi

# Step 13: Check/configure nginx
print_step "Step 13: Checking nginx configuration..."
if ! command -v nginx &> /dev/null; then
    print_warning "Nginx not installed. Installing..."
    sudo apt update
    sudo apt install -y nginx
fi

# Check if nginx config exists
NGINX_CONFIG="/etc/nginx/sites-available/drone-backend"
NGINX_CONFIG_ALT="/etc/nginx/conf.d/drone-backend.conf"

if [ -f "$NGINX_CONFIG" ] || [ -f "$NGINX_CONFIG_ALT" ]; then
    print_success "Nginx configuration found"
else
    print_warning "Nginx configuration not found. Creating it..."
    if [ -f deploy/nginx.conf ]; then
        sudo cp deploy/nginx.conf /etc/nginx/sites-available/drone-backend
        sudo ln -sf /etc/nginx/sites-available/drone-backend /etc/nginx/sites-enabled/
        # Remove default nginx site if it exists
        sudo rm -f /etc/nginx/sites-enabled/default
        print_success "Nginx configuration created"
    else
        print_error "nginx.conf template not found in deploy/ directory!"
    fi
fi

# Test nginx configuration
if sudo nginx -t 2>&1 | grep -q "successful"; then
    print_success "Nginx configuration is valid"
else
    print_error "Nginx configuration has errors!"
    sudo nginx -t
fi

# Step 14: Restart nginx
print_step "Step 14: Restarting nginx..."
sudo systemctl restart nginx
sleep 2

if sudo systemctl is-active --quiet nginx; then
    print_success "Nginx is running"
else
    print_error "Nginx failed to start!"
    sudo systemctl status nginx --no-pager | head -10
fi

# Step 15: Test through nginx
print_step "Step 15: Testing backend through nginx..."
NGINX_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost/api/health 2>/dev/null || echo "000")

if [ "$NGINX_RESPONSE" = "200" ]; then
    print_success "Nginx can reach backend!"
    curl -s http://localhost/api/health | head -3
else
    print_warning "Nginx test returned HTTP $NGINX_RESPONSE"
    echo "Checking nginx error logs..."
    sudo tail -10 /var/log/nginx/error.log
fi

# Step 16: Enable PM2 startup on boot
print_step "Step 16: Ensuring PM2 starts on boot..."
if pm2 startup 2>&1 | grep -q "already setup"; then
    print_success "PM2 startup already configured"
else
    print_warning "PM2 startup not configured"
    STARTUP_CMD=$(pm2 startup systemd -u $USER --hp $HOME 2>&1 | grep "sudo")
    if [ ! -z "$STARTUP_CMD" ]; then
        echo ""
        echo "Run this command to enable PM2 startup on boot:"
        echo -e "${YELLOW}$STARTUP_CMD${NC}"
        echo ""
        read -p "Do you want to run this command now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            eval $STARTUP_CMD
            print_success "PM2 startup configured"
        else
            print_warning "Skipped PM2 startup configuration"
        fi
    fi
    pm2 save
fi

# Step 17: Final verification
echo ""
echo "=========================================="
echo "Final Verification"
echo "=========================================="
echo ""

echo "PM2 Status:"
pm2 list | grep -E "drone-backend|NAME|online|errored" || echo "No backend process found"
echo ""

echo "Port 5050 Status:"
if sudo netstat -tlnp 2>/dev/null | grep ":5050" || sudo ss -tlnp 2>/dev/null | grep ":5050"; then
    print_success "Port 5050 is listening"
else
    print_error "Port 5050 is NOT listening"
fi
echo ""

echo "Backend Health (direct):"
if curl -s http://localhost:5050/api/health > /dev/null 2>&1; then
    print_success "Backend responding directly"
    curl -s http://localhost:5050/api/health | head -3
else
    print_error "Backend not responding directly"
fi
echo ""

echo "Backend Health (through nginx):"
if curl -s http://localhost/api/health > /dev/null 2>&1; then
    print_success "Backend responding through nginx"
    curl -s http://localhost/api/health | head -3
else
    print_error "Backend not responding through nginx"
fi
echo ""

echo "Nginx Status:"
sudo systemctl status nginx --no-pager | head -3
echo ""

# Get EC2 public IP
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
if [ "$EC2_IP" != "unknown" ]; then
    echo "External Test:"
    echo "  curl http://$EC2_IP/api/health"
    echo ""
fi

echo "=========================================="
echo "Fix Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. If backend is working, test from your frontend"
echo "2. Monitor logs: pm2 logs drone-backend"
echo "3. Check nginx logs: sudo tail -f /var/log/nginx/error.log"
echo ""
echo "If issues persist, check:"
echo "  - Backend logs: pm2 logs drone-backend --lines 50"
echo "  - Database connection: psql -U drone_user -d drone_analytics -c 'SELECT 1;'"
echo "  - .env file: cat ~/Capstone_Interface/server/.env"
echo ""







