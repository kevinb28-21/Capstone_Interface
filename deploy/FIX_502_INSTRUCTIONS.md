# Fix 502 Bad Gateway Error

## Quick Fix

SSH into your EC2 instance and run:

```bash
# Make scripts executable
chmod +x ~/Capstone_Interface/deploy/fix-502-bad-gateway.sh
chmod +x ~/Capstone_Interface/deploy/diagnose-502.sh

# Run diagnostic first (optional)
bash ~/Capstone_Interface/deploy/diagnose-502.sh

# Run the fix script
bash ~/Capstone_Interface/deploy/fix-502-bad-gateway.sh
```

## Manual Fix Steps

If the script doesn't work, follow these steps:

### Step 1: Check PM2 Status

```bash
pm2 status
pm2 list
```

If `drone-backend` is not running or shows `errored`:

```bash
cd ~/Capstone_Interface/server
pm2 restart drone-backend
# OR if it doesn't exist:
pm2 start ecosystem.config.js
pm2 save
```

### Step 2: Check Backend Logs

```bash
pm2 logs drone-backend --lines 50
```

Look for errors like:
- Database connection failures
- Port already in use
- Missing environment variables
- Module import errors

### Step 3: Test Backend Directly

```bash
curl http://localhost:5050/api/health
```

If this fails, the backend isn't running or crashed.

### Step 4: Check Port 5050

```bash
sudo netstat -tlnp | grep 5050
# OR
sudo ss -tlnp | grep 5050
```

If nothing shows up, the backend isn't listening.

### Step 5: Restart Services

```bash
# Restart backend
pm2 restart drone-backend

# Restart nginx
sudo systemctl restart nginx

# Check status
pm2 status
sudo systemctl status nginx
```

### Step 6: Check Nginx Configuration

```bash
# Test nginx config
sudo nginx -t

# Check nginx error logs
sudo tail -f /var/log/nginx/error.log
```

### Step 7: Verify Environment Variables

```bash
# Check if .env file exists
cat ~/Capstone_Interface/server/.env

# Should have at least:
# PORT=5050
# DB_HOST=localhost
# DB_NAME=drone_analytics
# DB_USER=...
# DB_PASSWORD=...
```

### Step 8: Ensure PM2 Starts on Boot

```bash
pm2 startup
# Follow the command it outputs
pm2 save
```

## Common Issues

### Issue 1: Backend Crashed Due to Database Error

**Solution:**
```bash
# Check database is running
sudo systemctl status postgresql

# Check database connection in .env
cat ~/Capstone_Interface/server/.env | grep DB_

# Restart database if needed
sudo systemctl restart postgresql
```

### Issue 2: Port 5050 Already in Use

**Solution:**
```bash
# Find what's using port 5050
sudo lsof -i :5050
# OR
sudo fuser 5050/tcp

# Kill the process if it's not PM2
sudo kill -9 <PID>
```

### Issue 3: Nginx Can't Connect to Backend

**Solution:**
```bash
# Verify nginx config points to localhost:5050
sudo cat /etc/nginx/sites-available/drone-backend | grep proxy_pass

# Should show: proxy_pass http://localhost:5050;

# Restart nginx
sudo systemctl restart nginx
```

### Issue 4: Backend Not Starting on Server Reboot

**Solution:**
```bash
# Enable PM2 startup
pm2 startup systemd -u ubuntu --hp /home/ubuntu
# Run the command it outputs
pm2 save
```

## Verification

After fixing, verify everything works:

```bash
# 1. Check PM2
pm2 status

# 2. Check port
sudo netstat -tlnp | grep 5050

# 3. Test backend
curl http://localhost:5050/api/health

# 4. Test through nginx (from EC2)
curl http://localhost/api/health

# 5. Test from external (replace with your EC2 IP)
curl http://18.223.169.5/api/health
```

## Still Having Issues?

1. **Check all logs:**
   ```bash
   pm2 logs drone-backend --lines 100
   sudo tail -100 /var/log/nginx/error.log
   ```

2. **Verify file paths:**
   ```bash
   ls -la ~/Capstone_Interface/server/src/server.js
   ls -la ~/Capstone_Interface/server/ecosystem.config.js
   ```

3. **Check file permissions:**
   ```bash
   ls -la ~/Capstone_Interface/server/
   ```

4. **Reinstall dependencies:**
   ```bash
   cd ~/Capstone_Interface/server
   npm install --production
   ```

## Security Note

The 502 error is NOT related to EC2 security groups. It's an internal issue where nginx can't connect to the Node.js backend on localhost:5050.

However, you should still secure your EC2:
- Restrict SSH (port 22) to your IP only
- Use HTTPS/SSL for production
- Don't expose database ports publicly




