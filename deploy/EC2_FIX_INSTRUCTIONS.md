# EC2 Backend Fix - Step by Step Instructions

## Quick Fix (Recommended)

### Option 1: Run the Automated Script on EC2

1. **SSH into your EC2 instance:**
   ```bash
   ssh -i your-key.pem ubuntu@ec2-18-223-169-5.us-east-2.compute.amazonaws.com
   ```

2. **Run the fix script:**
   ```bash
   cd ~/Capstone_Interface
   git pull origin main
   bash deploy/run-backend-fix-on-ec2.sh
   ```

   This script will:
   - ✅ Pull latest code changes
   - ✅ Check/create `.env` file
   - ✅ Install/update dependencies
   - ✅ Start backend with PM2 (using correct config format)
   - ✅ Configure and restart nginx
   - ✅ Test all connections
   - ✅ Enable PM2 startup on boot

3. **Follow the prompts** (it will ask you to edit `.env` if needed)

### Option 2: Manual Steps

If the script doesn't work, follow these manual steps:

#### Step 1: SSH into EC2
```bash
ssh -i your-key.pem ubuntu@ec2-18-223-169-5.us-east-2.compute.amazonaws.com
```

#### Step 2: Navigate to Project
```bash
cd ~/Capstone_Interface
```

#### Step 3: Pull Latest Changes
```bash
git pull origin main
```

#### Step 4: Check/Create .env File
```bash
cd server
nano .env
```

Make sure it has:
```env
PORT=5050
NODE_ENV=production
ORIGIN=http://localhost:5173,http://localhost:5182

DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=your_actual_password
```

Save and exit (Ctrl+X, then Y, then Enter)

#### Step 5: Install Dependencies
```bash
npm install --production
```

#### Step 6: Stop Old Backend
```bash
pm2 delete drone-backend
```

#### Step 7: Start Backend with PM2
```bash
# Use .cjs config (CommonJS - more reliable)
pm2 start ecosystem.config.cjs

# OR if .cjs doesn't exist, use .js
pm2 start ecosystem.config.js

# Save PM2 configuration
pm2 save
```

#### Step 8: Verify Backend is Running
```bash
# Check PM2 status
pm2 status

# Check if port 5050 is listening
sudo netstat -tlnp | grep 5050

# Test health endpoint
curl http://localhost:5050/api/health
```

#### Step 9: Restart Nginx
```bash
sudo systemctl restart nginx
sudo systemctl status nginx
```

#### Step 10: Test Through Nginx
```bash
curl http://localhost/api/health
```

#### Step 11: Enable PM2 Startup on Boot
```bash
pm2 startup
# Run the command it outputs (usually something like):
# sudo env PATH=$PATH:/usr/bin pm2 startup systemd -u ubuntu --hp /home/ubuntu
pm2 save
```

## Verification

After running the fix, verify everything works:

```bash
# 1. PM2 shows backend as "online"
pm2 status

# 2. Port 5050 is listening
sudo netstat -tlnp | grep 5050

# 3. Backend responds directly
curl http://localhost:5050/api/health

# 4. Backend responds through nginx
curl http://localhost/api/health

# 5. Nginx is running
sudo systemctl status nginx
```

## Troubleshooting

### Backend Not Starting

Check logs:
```bash
pm2 logs drone-backend --lines 50
```

Common issues:
- **Database connection error**: Check PostgreSQL is running
  ```bash
  sudo systemctl status postgresql
  sudo systemctl start postgresql
  ```
- **Missing .env**: Create it (see Step 4 above)
- **Port already in use**: Kill the process
  ```bash
  sudo lsof -i :5050
  sudo kill -9 <PID>
  ```

### Nginx Still Shows 502

1. Check nginx error logs:
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

2. Verify nginx config:
   ```bash
   sudo nginx -t
   ```

3. Check if backend is actually running:
   ```bash
   pm2 status
   curl http://localhost:5050/api/health
   ```

### PM2 Config Issues

If PM2 can't read the config:
```bash
# Use .cjs format (CommonJS)
cd ~/Capstone_Interface/server
pm2 delete drone-backend
pm2 start ecosystem.config.cjs
pm2 save
```

## After Fixing

1. **Test from your frontend** (Netlify)
2. **Monitor logs** for a few minutes:
   ```bash
   pm2 logs drone-backend
   ```
3. **Check external access**:
   ```bash
   # Get your EC2 IP
   curl http://169.254.169.254/latest/meta-data/public-ipv4
   
   # Test from external (replace with your IP)
   curl http://YOUR-EC2-IP/api/health
   ```

## Summary

The automated script (`run-backend-fix-on-ec2.sh`) handles all these steps automatically. Just:

1. SSH into EC2
2. Run: `bash deploy/run-backend-fix-on-ec2.sh`
3. Follow prompts
4. Done!

If you encounter any issues, check the logs and refer to the troubleshooting section above.




