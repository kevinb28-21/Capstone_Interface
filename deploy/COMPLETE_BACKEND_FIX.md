# Complete Backend Fix Guide

## What Was Fixed

1. **PM2 Config Format**: Changed `ecosystem.config.js` from ES6 `export default` to CommonJS `module.exports` for better PM2 compatibility
2. **Enhanced Fix Script**: Updated `fix-502-bad-gateway.sh` to:
   - Prefer `.cjs` config files (more reliable)
   - Check for `.env` file and create template if missing
   - Better error diagnosis
   - Database connection testing
   - Manual backend startup testing

## Steps to Fix on EC2

### Step 1: Pull Latest Changes

```bash
cd ~/Capstone_Interface
git pull origin main
```

### Step 2: Run the Fix Script

```bash
bash deploy/fix-502-bad-gateway.sh
```

This will:
- ✅ Check/create `.env` file
- ✅ Start/restart backend with PM2
- ✅ Verify port 5050 is listening
- ✅ Test backend health endpoint
- ✅ Restart nginx
- ✅ Show diagnostic information

### Step 3: If Backend Still Fails

Check the logs:

```bash
# Check PM2 logs
pm2 logs drone-backend --lines 50

# Check for common errors:
# - Database connection errors
# - Missing environment variables
# - Port already in use
# - Module import errors
```

### Step 4: Manual Diagnosis

If the script doesn't work, run these manually:

```bash
# 1. Stop all PM2 processes
pm2 delete all

# 2. Check .env file exists and has correct values
cat ~/Capstone_Interface/server/.env

# 3. Try starting backend manually to see errors
cd ~/Capstone_Interface/server
node src/server.js
# (Press Ctrl+C after seeing errors)

# 4. If manual start works, use PM2 with .cjs config
pm2 start ecosystem.config.cjs
pm2 save

# 5. Verify it's running
pm2 status
curl http://localhost:5050/api/health

# 6. Restart nginx
sudo systemctl restart nginx
```

## Common Issues and Fixes

### Issue 1: Database Connection Error

**Error in logs:**
```
Error: connect ECONNREFUSED 127.0.0.1:5432
```

**Fix:**
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Start if not running
sudo systemctl start postgresql

# Verify database exists
psql -U drone_user -d drone_analytics -c "SELECT 1;"
```

### Issue 2: Missing .env File

**Fix:**
```bash
cd ~/Capstone_Interface/server
nano .env
# Add:
# PORT=5050
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=drone_analytics
# DB_USER=drone_user
# DB_PASSWORD=your_password
```

### Issue 3: Port 5050 Already in Use

**Fix:**
```bash
# Find what's using port 5050
sudo lsof -i :5050
# OR
sudo fuser 5050/tcp

# Kill the process (if it's not PM2)
sudo kill -9 <PID>

# Restart PM2
pm2 restart drone-backend
```

### Issue 4: PM2 Config Format Error

**Fix:**
```bash
# Use .cjs file (CommonJS format)
cd ~/Capstone_Interface/server
pm2 delete drone-backend
pm2 start ecosystem.config.cjs
pm2 save
```

### Issue 5: Node Modules Missing

**Fix:**
```bash
cd ~/Capstone_Interface/server
npm install --production
pm2 restart drone-backend
```

## Verification Checklist

After running the fix, verify:

- [ ] PM2 shows backend as "online": `pm2 status`
- [ ] Port 5050 is listening: `sudo netstat -tlnp | grep 5050`
- [ ] Backend health check works: `curl http://localhost:5050/api/health`
- [ ] Nginx is running: `sudo systemctl status nginx`
- [ ] Nginx can reach backend: `curl http://localhost/api/health`
- [ ] External access works: `curl http://YOUR-EC2-IP/api/health`

## After Fixing

1. **Enable PM2 Startup on Boot:**
   ```bash
   pm2 startup
   # Run the command it outputs
   pm2 save
   ```

2. **Monitor Logs:**
   ```bash
   pm2 logs drone-backend --lines 50
   ```

3. **Test from Frontend:**
   - Open your Netlify frontend
   - Try uploading an image
   - Check if API calls work

## Still Having Issues?

1. **Check all logs:**
   ```bash
   pm2 logs drone-backend --lines 100
   sudo tail -50 /var/log/nginx/error.log
   ```

2. **Verify file paths:**
   ```bash
   ls -la ~/Capstone_Interface/server/src/server.js
   ls -la ~/Capstone_Interface/server/ecosystem.config.cjs
   ```

3. **Check Node.js version:**
   ```bash
   node --version  # Should be v18 or v20
   ```

4. **Reinstall dependencies:**
   ```bash
   cd ~/Capstone_Interface/server
   rm -rf node_modules
   npm install --production
   ```

## Refresh Steps

After fixing, you may need to:

1. **Clear browser cache** (if testing from frontend)
2. **Restart nginx** (already done by script)
3. **Wait 10-30 seconds** for services to fully start
4. **Test from different locations:**
   - Direct backend: `curl http://localhost:5050/api/health`
   - Through nginx: `curl http://localhost/api/health`
   - External: `curl http://YOUR-EC2-IP/api/health`

## Summary

The main fix was:
- ✅ Changed PM2 config to CommonJS format (more reliable)
- ✅ Enhanced fix script with better error handling
- ✅ Added .env file checking and template creation
- ✅ Added database connection testing
- ✅ Better diagnostic output

Run `bash deploy/fix-502-bad-gateway.sh` on EC2 to apply all fixes automatically.








