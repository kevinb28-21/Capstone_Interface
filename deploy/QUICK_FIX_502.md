# Quick Fix for 502 Bad Gateway Error

## Problem
When uploading images on the analytics page, you get a 502 Bad Gateway error. This means:
- Netlify is successfully proxying your request to EC2
- Nginx on EC2 is receiving the request
- **But the Node.js backend on port 5050 is not running or not responding**

## Quick Fix (Run from your local machine)

### Option 1: Automated Fix Script (Recommended)

```bash
cd ~/Desktop/Projects/Capstone_Interface
bash deploy/fix-502-complete.sh
```

This script will:
1. SSH into your EC2 instance
2. Check if the backend is running
3. Restart the backend with PM2
4. Verify it's working
5. Restart nginx
6. Test everything

**Note:** Make sure your SSH key is at `~/Downloads/MS04_ID.pem`. If it's elsewhere, edit the script first.

### Option 2: Manual SSH Fix

If the script doesn't work, SSH into EC2 manually:

```bash
# SSH into EC2
ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-223-169-5.us-east-2.compute.amazonaws.com

# Once connected, run:
cd ~/Capstone_Interface/server

# Check if backend is running
pm2 list

# If not running or shows error, restart it:
pm2 delete drone-backend
pm2 start ecosystem.config.cjs
pm2 save

# Wait a few seconds
sleep 5

# Test if it's working
curl http://localhost:5050/api/health

# Restart nginx
sudo systemctl restart nginx

# Check status
pm2 status
sudo systemctl status nginx
```

## What the Script Does

1. **Checks PM2 Status** - Verifies if the backend process is running
2. **Checks Port 5050** - Verifies if the backend is listening
3. **Tests Health Endpoint** - Confirms the backend is responding
4. **Checks Logs** - If not working, shows recent error logs
5. **Verifies .env File** - Ensures configuration exists
6. **Installs Dependencies** - If needed
7. **Restarts Backend** - Stops and starts with PM2
8. **Restarts Nginx** - Ensures nginx can connect to backend
9. **Final Verification** - Tests everything end-to-end

## Common Issues

### Issue 1: Backend Crashed Due to Database Error

**Symptoms:** Backend starts then immediately crashes

**Fix:**
```bash
# Check database is running
sudo systemctl status postgresql

# Check database connection in .env
cat ~/Capstone_Interface/server/.env | grep DB_

# Restart database if needed
sudo systemctl restart postgresql
```

### Issue 2: Missing Environment Variables

**Symptoms:** Backend fails to start, logs show missing config

**Fix:**
```bash
# Edit .env file
nano ~/Capstone_Interface/server/.env

# Make sure it has at least:
# PORT=5050
# DB_HOST=localhost
# DB_NAME=drone_analytics
# DB_USER=drone_user
# DB_PASSWORD=your_password
```

### Issue 3: Port Already in Use

**Symptoms:** Backend can't bind to port 5050

**Fix:**
```bash
# Find what's using port 5050
sudo lsof -i :5050

# Kill the process if it's not PM2
sudo kill -9 <PID>
```

### Issue 4: PM2 Not Starting on Boot

**Symptoms:** Backend works after manual start but stops after reboot

**Fix:**
```bash
# Enable PM2 startup
pm2 startup systemd -u ubuntu --hp /home/ubuntu
# Run the command it outputs
pm2 save
```

## Verification

After running the fix, verify everything works:

1. **Check PM2:**
   ```bash
   pm2 list
   ```
   Should show `drone-backend` as `online`

2. **Test Backend:**
   ```bash
   curl http://localhost:5050/api/health
   ```
   Should return JSON with status "ok"

3. **Test Through Nginx:**
   ```bash
   curl http://localhost/api/health
   ```
   Should return JSON with status "ok"

4. **Test from External:**
   ```bash
   curl http://18.223.169.5/api/health
   ```
   Should return JSON with status "ok"

5. **Test from Browser:**
   - Go to: https://cropview.netlify.app
   - Open Analytics page
   - Try uploading an image
   - Should work without 502 error

## Still Having Issues?

1. **Check Backend Logs:**
   ```bash
   pm2 logs drone-backend --lines 50
   ```

2. **Check Nginx Logs:**
   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

3. **Check Nginx Config:**
   ```bash
   sudo nginx -t
   sudo cat /etc/nginx/sites-available/drone-backend | grep proxy_pass
   ```
   Should show: `proxy_pass http://localhost:5050;`

4. **Verify File Paths:**
   ```bash
   ls -la ~/Capstone_Interface/server/src/server.js
   ls -la ~/Capstone_Interface/server/ecosystem.config.cjs
   ```

## Understanding the Flow

1. **Browser** → Uploads image to `/api/images`
2. **Netlify** → Proxies `/api/*` to `http://ec2-18-223-169-5.us-east-2.compute.amazonaws.com/api/*`
3. **Nginx on EC2** → Receives request, proxies `/api` to `http://localhost:5050`
4. **Node.js Backend** → Handles `/api/images`, saves to database
5. **Background Worker** → Processes image automatically

The 502 error happens at step 3-4, meaning nginx can't connect to the Node.js backend.
