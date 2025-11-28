#!/bin/bash
# Complete the setup from where it left off
# Run this on EC2

set -e

# Get database password
DB_PASSWORD=$(cat ~/db_password.txt 2>/dev/null | grep "SAVE THIS PASSWORD" | awk '{print $4}' || echo "")

if [ -z "$DB_PASSWORD" ]; then
    echo "Error: Could not find database password in ~/db_password.txt"
    exit 1
fi

echo "Database password found: $DB_PASSWORD"
echo ""

# Setup Python environment
echo "Setting up Python environment..."
cd ~/Capstone_Interface/python_processing

if [ ! -d venv ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install gunicorn
deactivate

echo "✓ Python environment ready"
echo ""

# Create directories
mkdir -p uploads processed models logs
echo "✓ Directories created"
echo ""

# Update .env file
cat > .env <<EOF
FLASK_PORT=5001
FLASK_DEBUG=False

DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=$DB_PASSWORD

UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed
WORKER_POLL_INTERVAL=10
WORKER_BATCH_SIZE=5
EOF
echo "✓ Python .env file created"
echo ""

# Run database migrations
echo "Running database migrations..."
export PGPASSWORD=$DB_PASSWORD
cd ~/Capstone_Interface

if [ -f server/database/schema.sql ]; then
    psql -U drone_user -d drone_analytics -h localhost -f server/database/schema.sql 2>/dev/null || echo "Schema may already be applied"
fi

if [ -f python_processing/database_migration_add_gndvi.sql ]; then
    psql -U drone_user -d drone_analytics -h localhost -f python_processing/database_migration_add_gndvi.sql 2>/dev/null || echo "GNDVI migration may already be applied"
fi

unset PGPASSWORD
echo "✓ Database migrations completed"
echo ""

# Start services with PM2
echo "Starting services with PM2..."

# Node.js backend
cd ~/Capstone_Interface/server
mkdir -p logs
pm2 delete drone-backend 2>/dev/null || true
pm2 start ecosystem.config.cjs
pm2 save
echo "✓ Node.js backend started"
echo ""

# Flask API
cd ~/Capstone_Interface/python_processing
mkdir -p logs
pm2 delete flask-api 2>/dev/null || true
pm2 start ecosystem.config.cjs
pm2 save
echo "✓ Flask API started"
echo ""

# Background worker
pm2 delete background-worker 2>/dev/null || true
if [ -f worker.config.cjs ]; then
    pm2 start worker.config.cjs
else
    # Create worker config
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
    pm2 start worker.config.cjs
fi
pm2 save
echo "✓ Background worker started"
echo ""

# Setup PM2 startup
echo "Configuring PM2 startup..."
pm2 startup systemd -u ubuntu --hp /home/ubuntu | grep "sudo" | bash || echo "PM2 startup may already be configured"
pm2 save
echo "✓ PM2 startup configured"
echo ""

# Configure nginx
echo "Configuring nginx..."
sudo cp ~/Capstone_Interface/deploy/nginx.conf /etc/nginx/sites-available/drone-backend
sudo ln -sf /etc/nginx/sites-available/drone-backend /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
echo "✓ Nginx configured and restarted"
echo ""

# Wait for services
echo "Waiting for services to start..."
sleep 5

# Verify
echo "=========================================="
echo "Verification"
echo "=========================================="
echo ""
echo "PM2 Status:"
pm2 list
echo ""
echo "Testing backend:"
curl -s http://localhost:5050/api/health | head -3 || echo "Backend not responding yet"
echo ""
echo "Testing through nginx:"
curl -s http://localhost/api/health | head -3 || echo "Nginx not responding yet"
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="

