#!/bin/bash
# Complete EC2 Setup Script for Drone Crop Health Platform
# Run this script on your NEW EC2 instance
# This is a fully automated setup for your demo

set -e  # Exit on error

echo "=========================================="
echo "EC2 Complete Setup - Drone Crop Health Platform"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

# Step 1: Update system
echo ""
print_info "Step 1: Updating system packages..."
sudo apt update && sudo apt upgrade -y
print_status "System updated"

# Step 2: Install Node.js 20.x
echo ""
print_info "Step 2: Installing Node.js 20.x..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    print_status "Node.js installed: $(node --version)"
else
    print_status "Node.js already installed: $(node --version)"
fi

# Step 3: Install Python 3 and dependencies
echo ""
print_info "Step 3: Installing Python 3..."
sudo apt install -y python3 python3-pip python3-venv python3-dev
print_status "Python 3 installed: $(python3 --version)"

# Step 4: Install PostgreSQL
echo ""
print_info "Step 4: Installing PostgreSQL..."
sudo apt install -y postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
print_status "PostgreSQL installed and started"

# Step 5: Install Git
echo ""
print_info "Step 5: Installing Git..."
sudo apt install -y git
print_status "Git installed: $(git --version)"

# Step 6: Install PM2
echo ""
print_info "Step 6: Installing PM2..."
sudo npm install -g pm2
print_status "PM2 installed: $(pm2 --version)"

# Step 7: Install Nginx
echo ""
print_info "Step 7: Installing Nginx..."
sudo apt install -y nginx
print_status "Nginx installed"

# Step 8: Install build tools
echo ""
print_info "Step 8: Installing build tools..."
sudo apt install -y build-essential libpq-dev
print_status "Build tools installed"

# Step 9: Setup PostgreSQL database
echo ""
print_info "Step 9: Setting up PostgreSQL database..."
# Generate a secure password
DB_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
echo "Generated database password: $DB_PASSWORD"
echo "SAVE THIS PASSWORD: $DB_PASSWORD" > ~/db_password.txt
chmod 600 ~/db_password.txt

# Create database and user
sudo -u postgres psql <<EOF
CREATE DATABASE drone_analytics;
CREATE USER drone_user WITH PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE drone_analytics TO drone_user;
ALTER USER drone_user CREATEDB;
\q
EOF

# Configure PostgreSQL authentication
PG_HBA_FILE="/etc/postgresql/$(ls /etc/postgresql)/main/pg_hba.conf"
sudo sed -i 's/local   all             all                                     peer/local   all             all                                     md5/' "$PG_HBA_FILE" 2>/dev/null || true
sudo sed -i 's/host    all             all             127.0.0.1\/32            ident/host    all             all             127.0.0.1\/32            md5/' "$PG_HBA_FILE" 2>/dev/null || true
sudo systemctl restart postgresql
print_status "Database and user created"

# Step 10: Clone repository
echo ""
print_info "Step 10: Cloning repository..."
if [ ! -d "$HOME/Capstone_Interface" ]; then
    cd ~
    git clone https://github.com/kevinb28-21/Capstone_Interface.git || {
        print_warning "Git clone failed. If repo is private, you may need to set up SSH keys or use HTTPS with credentials."
        print_info "Please clone manually: git clone <your-repo-url>"
        exit 1
    }
    print_status "Repository cloned"
else
    cd ~/Capstone_Interface
    git pull origin main || print_warning "Git pull failed, continuing with existing code"
    print_status "Repository updated"
fi

# Step 11: Setup Node.js backend
echo ""
print_info "Step 11: Setting up Node.js backend..."
cd ~/Capstone_Interface/server
npm install --production
print_status "Node.js dependencies installed"

# Create .env file for server
if [ ! -f .env ]; then
    cat > .env <<EOF
PORT=5050
NODE_ENV=production
ORIGIN=http://localhost:5173,http://localhost:5182,https://cropview.netlify.app

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=$DB_PASSWORD

# AWS S3 (optional - leave empty if not using S3)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION=us-east-2
# S3_BUCKET_NAME=
EOF
    print_status "Created server/.env file"
else
    print_warning "server/.env already exists, updating DB_PASSWORD..."
    sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=$DB_PASSWORD/" .env || true
fi

# Step 12: Setup Python environment
echo ""
print_info "Step 12: Setting up Python environment..."
cd ~/Capstone_Interface/python_processing
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install gunicorn
deactivate
print_status "Python environment created and dependencies installed"

# Create directories
mkdir -p uploads processed models logs
print_status "Directories created"

# Create .env file for Python
if [ ! -f .env ]; then
    cat > .env <<EOF
FLASK_PORT=5001
FLASK_DEBUG=False

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=$DB_PASSWORD

# File paths
UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed

# Background worker
WORKER_POLL_INTERVAL=10
WORKER_BATCH_SIZE=5

# AWS S3 (optional - leave empty if not using S3)
# AWS_ACCESS_KEY_ID=
# AWS_SECRET_ACCESS_KEY=
# AWS_REGION=us-east-2
# S3_BUCKET_NAME=
EOF
    print_status "Created python_processing/.env file"
else
    print_warning "python_processing/.env already exists, updating DB_PASSWORD..."
    sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=$DB_PASSWORD/" .env || true
fi

# Step 13: Run database migrations
echo ""
print_info "Step 13: Running database migrations..."
export PGPASSWORD=$DB_PASSWORD
cd ~/Capstone_Interface

# Run main schema
if [ -f server/database/schema.sql ]; then
    psql -U drone_user -d drone_analytics -h localhost -f server/database/schema.sql
    print_status "Main schema applied"
else
    print_warning "schema.sql not found, skipping"
fi

# Run GNDVI migration
if [ -f python_processing/database_migration_add_gndvi.sql ]; then
    psql -U drone_user -d drone_analytics -h localhost -f python_processing/database_migration_add_gndvi.sql
    print_status "GNDVI migration applied"
else
    print_warning "GNDVI migration not found, skipping"
fi

unset PGPASSWORD
print_status "Database migrations completed"

# Step 14: Setup PM2 for Node.js backend
echo ""
print_info "Step 14: Setting up PM2 for Node.js backend..."
cd ~/Capstone_Interface/server
mkdir -p logs

# Ensure ecosystem.config.cjs exists
if [ ! -f ecosystem.config.cjs ]; then
    cat > ecosystem.config.cjs <<'EOF'
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
fi

pm2 delete drone-backend 2>/dev/null || true
pm2 start ecosystem.config.cjs
pm2 save
print_status "Node.js backend started with PM2"

# Step 15: Setup PM2 for Flask API
echo ""
print_info "Step 15: Setting up PM2 for Flask API..."
cd ~/Capstone_Interface/python_processing
mkdir -p logs

# Ensure ecosystem.config.cjs exists
if [ ! -f ecosystem.config.cjs ]; then
    cat > ecosystem.config.cjs <<'EOF'
module.exports = {
  apps: [{
    name: 'flask-api',
    script: 'venv/bin/gunicorn',
    args: '-w 4 -b 0.0.0.0:5001 flask_api_db:app',
    instances: 1,
    exec_mode: 'fork',
    cwd: '/home/ubuntu/Capstone_Interface/python_processing',
    env: {
      FLASK_PORT: 5001,
      FLASK_DEBUG: 'False'
    },
    error_file: './logs/flask-err.log',
    out_file: './logs/flask-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    max_memory_restart: '1G',
    watch: false,
    ignore_watch: ['venv', 'logs', '__pycache__']
  }]
};
EOF
fi

pm2 delete flask-api 2>/dev/null || true
pm2 start ecosystem.config.cjs
pm2 save
print_status "Flask API started with PM2"

# Step 16: Setup PM2 for background worker
echo ""
print_info "Step 16: Setting up PM2 for background worker..."
cd ~/Capstone_Interface/python_processing

# Create worker config if it doesn't exist
if [ ! -f worker.config.cjs ]; then
    cat > worker.config.cjs <<'EOF'
module.exports = {
  apps: [{
    name: 'background-worker',
    script: 'venv/bin/python',
    args: 'background_worker.py',
    instances: 1,
    exec_mode: 'fork',
    cwd: '/home/ubuntu/Capstone_Interface/python_processing',
    error_file: './logs/worker-err.log',
    out_file: './logs/worker-out.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    merge_logs: true,
    autorestart: true,
    max_memory_restart: '1G',
    watch: false,
    ignore_watch: ['venv', 'logs', '__pycache__']
  }]
};
EOF
fi

pm2 delete background-worker 2>/dev/null || true
pm2 start worker.config.cjs
pm2 save
print_status "Background worker started with PM2"

# Step 17: Setup PM2 startup on boot
echo ""
print_info "Step 17: Configuring PM2 to start on boot..."
pm2 startup systemd -u ubuntu --hp /home/ubuntu | grep "sudo" | bash || print_warning "PM2 startup may need manual setup"
pm2 save
print_status "PM2 startup configured"

# Step 18: Configure Nginx
echo ""
print_info "Step 18: Configuring Nginx..."
cd ~/Capstone_Interface

# Update nginx.conf with correct server name
sed -i 's/server_name.*/server_name ec2-18-117-90-212.us-east-2.compute.amazonaws.com 18.117.90.212 _;/' deploy/nginx.conf

# Copy nginx config
sudo cp deploy/nginx.conf /etc/nginx/sites-available/drone-backend
sudo ln -sf /etc/nginx/sites-available/drone-backend /etc/nginx/sites-enabled/

# Remove default nginx site
sudo rm -f /etc/nginx/sites-enabled/default

# Test nginx config
sudo nginx -t
print_status "Nginx configuration valid"

# Start and enable nginx
sudo systemctl restart nginx
sudo systemctl enable nginx
print_status "Nginx started and enabled"

# Step 19: Wait for services to start
echo ""
print_info "Step 19: Waiting for services to start..."
sleep 5

# Step 20: Verify everything is working
echo ""
echo "=========================================="
print_info "Verification"
echo "=========================================="

# Check PM2 status
echo ""
print_info "PM2 Status:"
pm2 list

# Check ports
echo ""
print_info "Checking ports..."
if sudo netstat -tlnp 2>/dev/null | grep -q ":5050" || sudo ss -tlnp 2>/dev/null | grep -q ":5050"; then
    print_status "Port 5050 (Node.js) is listening"
else
    print_error "Port 5050 is NOT listening"
fi

if sudo netstat -tlnp 2>/dev/null | grep -q ":5001" || sudo ss -tlnp 2>/dev/null | grep -q ":5001"; then
    print_status "Port 5001 (Flask) is listening"
else
    print_error "Port 5001 is NOT listening"
fi

# Test backend health
echo ""
print_info "Testing backend health..."
HEALTH=$(curl -s http://localhost:5050/api/health 2>/dev/null || echo "ERROR")
if echo "$HEALTH" | grep -q "status"; then
    print_status "Backend health check passed"
    echo "$HEALTH" | head -3
else
    print_error "Backend health check failed"
    print_info "Check logs: pm2 logs drone-backend"
fi

# Test nginx
echo ""
print_info "Testing nginx..."
NGINX_HEALTH=$(curl -s http://localhost/api/health 2>/dev/null || echo "ERROR")
if echo "$NGINX_HEALTH" | grep -q "status"; then
    print_status "Nginx proxy is working"
else
    print_warning "Nginx proxy test failed (may need external access)"
fi

# Final summary
echo ""
echo "=========================================="
print_status "Setup Complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: Database password saved to: ~/db_password.txt"
echo "Password: $DB_PASSWORD"
echo ""
echo "Services running:"
echo "  - Node.js Backend: http://localhost:5050"
echo "  - Flask API: http://localhost:5001"
echo "  - Nginx: http://localhost (port 80)"
echo ""
echo "External access:"
echo "  - Backend: http://18.117.90.212/api/health"
echo ""
echo "PM2 Commands:"
echo "  - Status: pm2 list"
echo "  - Logs: pm2 logs"
echo "  - Restart: pm2 restart all"
echo ""
echo "Next steps:"
echo "1. Update Netlify redirects to point to: ec2-18-117-90-212.us-east-2.compute.amazonaws.com"
echo "2. Test image upload from your frontend"
echo "3. Check logs if issues: pm2 logs"
echo ""

