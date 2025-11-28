# Complete Setup Manually on EC2

The automated setup started but failed at git clone. Here's how to complete it:

## Option 1: Transfer Code from Local Machine (Recommended)

Run this from your local machine:

```bash
cd ~/Desktop/Projects/Capstone_Interface
bash deploy/transfer-and-setup.sh
```

This will:
1. Create a tarball of your code
2. Transfer it to EC2
3. Extract it
4. Continue with the setup

## Option 2: Complete Setup Manually on EC2

### Step 1: SSH into EC2
```bash
ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-117-90-212.us-east-2.compute.amazonaws.com
```

### Step 2: Transfer Code from Local Machine

From your local machine (in a new terminal):
```bash
cd ~/Desktop/Projects/Capstone_Interface
tar --exclude='node_modules' --exclude='venv' --exclude='.git' -czf - . | \
  ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-117-90-212.us-east-2.compute.amazonaws.com \
  "cd ~ && tar -xzf -"
```

### Step 3: Continue Setup on EC2

Once code is transferred, on EC2:
```bash
cd ~/Capstone_Interface

# The setup script already installed dependencies, so continue from database setup
# Database password was saved to ~/db_password.txt
DB_PASSWORD=$(cat ~/db_password.txt | grep "SAVE THIS PASSWORD" | awk '{print $4}')

# Setup Node.js backend
cd ~/Capstone_Interface/server
npm install --production

# Create .env file
cat > .env <<EOF
PORT=5050
NODE_ENV=production
ORIGIN=http://localhost:5173,http://localhost:5182,https://cropview.netlify.app

DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=$DB_PASSWORD
EOF

# Setup Python environment
cd ~/Capstone_Interface/python_processing
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn
deactivate

# Create .env file for Python
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

mkdir -p uploads processed models logs

# Run database migrations
export PGPASSWORD=$DB_PASSWORD
psql -U drone_user -d drone_analytics -h localhost -f server/database/schema.sql
psql -U drone_user -d drone_analytics -h localhost -f python_processing/database_migration_add_gndvi.sql
unset PGPASSWORD

# Start services with PM2
cd ~/Capstone_Interface/server
mkdir -p logs
pm2 start ecosystem.config.cjs
pm2 save

cd ~/Capstone_Interface/python_processing
mkdir -p logs
pm2 start ecosystem.config.cjs
pm2 start worker.config.cjs
pm2 save

# Setup PM2 startup
pm2 startup systemd -u ubuntu --hp /home/ubuntu | grep "sudo" | bash
pm2 save

# Configure nginx
sudo cp ~/Capstone_Interface/deploy/nginx.conf /etc/nginx/sites-available/drone-backend
sudo ln -sf /etc/nginx/sites-available/drone-backend /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

# Verify
sleep 5
pm2 list
curl http://localhost:5050/api/health
curl http://localhost/api/health
```

## Quick Check

After setup, verify everything works:

```bash
# On EC2
pm2 list  # Should show 3 services running
curl http://localhost:5050/api/health  # Should return JSON
curl http://localhost/api/health  # Should return JSON through nginx

# From local machine
curl http://18.117.90.212/api/health  # Should return JSON
```

## If Services Don't Start

```bash
# Check logs
pm2 logs drone-backend
pm2 logs flask-api
pm2 logs background-worker

# Restart services
pm2 restart all

# Check database connection
cat ~/Capstone_Interface/server/.env | grep DB_PASSWORD
cat ~/db_password.txt
```

