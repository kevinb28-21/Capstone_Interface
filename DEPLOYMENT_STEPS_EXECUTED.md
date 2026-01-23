# Deployment Steps Execution Summary

## ✅ Completed Steps

### 1. Code Changes Committed
- ✅ Fixed `server/src/server.js` to listen on `0.0.0.0`
- ✅ Updated `client/src/utils/api.js` to use relative paths in production
- ✅ Enhanced error handling in frontend
- ✅ Added health check indicator to Home page
- ✅ Updated `netlify.toml` with correct EC2 URL and port
- ✅ Created deployment documentation
- ✅ Created automated deployment script

**Commit**: `eaa2dd6` - "Fix backend connection: listen on 0.0.0.0, use Netlify proxy, improve error handling"

### 2. Changes Pushed to Git
- ✅ All changes pushed to `origin/main`
- ✅ Netlify should auto-deploy the frontend (if connected to Git)

## 🔄 Next Steps (Manual Execution Required)

### Step 1: Deploy Backend Fix to EC2

You need to run the deployment script with your EC2 key file. The key file should be named `MS04_ID.pem` and located in your Downloads folder or project directory.

**Option A: Run the automated script**
```bash
cd /Users/kevinbhatt/Desktop/Projects/Capstone_Interface
./deploy/apply-backend-connection-fix.sh ~/Downloads/MS04_ID.pem
```

**Option B: Manual SSH deployment**
```bash
# 1. SSH into EC2
ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-3-144-192-19.us-east-2.compute.amazonaws.com

# 2. Navigate to server directory
cd ~/Capstone_Interface/server

# 3. Pull latest changes (or manually update server.js)
git pull origin main

# 4. Verify server.js listens on 0.0.0.0
grep "app.listen" src/server.js
# Should show: app.listen(PORT, '0.0.0.0', async () => {

# 5. Restart server
pm2 restart all
# OR
sudo systemctl restart backend
# OR manually: npm start

# 6. Verify server is listening on 0.0.0.0
netstat -tlnp | grep 5050
# Should show: 0.0.0.0:5050 (not 127.0.0.1:5050)

# 7. Test health endpoint
curl http://localhost:5050/api/health
```

### Step 2: Verify Backend is Accessible

From your local machine, test:
```bash
curl http://ec2-3-144-192-19.us-east-2.compute.amazonaws.com:5050/api/health
```

Expected response:
```json
{"status":"ok","database":"connected",...}
```

If this fails, check:
- EC2 security group allows inbound traffic on port 5050 (from 0.0.0.0/0 or Netlify IPs)
- Backend server is running
- Server is listening on 0.0.0.0 (not localhost)

### Step 3: Verify Netlify Deployment

1. **Check Netlify Dashboard**:
   - Go to your Netlify site dashboard
   - Verify latest deployment completed successfully
   - Check deploy logs for any errors

2. **Verify Environment Variables**:
   - Go to Site settings → Environment variables
   - **DO NOT** set `VITE_API_URL` (or set it to HTTPS only)
   - Leave it unset to use Netlify proxy (recommended)

3. **Test the Site**:
   - Open your Netlify site URL
   - Open browser DevTools (F12) → Console
   - Should see: `✓ Backend health check passed`
   - Home page should show: `✓ Backend Online` badge

### Step 4: Test API Endpoints

In browser console or Network tab, verify:
- `GET /api/health` returns 200 OK
- `GET /api/images` returns 200 OK with data
- `GET /api/telemetry` returns 200 OK

## 🔍 Troubleshooting

### If backend is not accessible:
1. Check EC2 security group inbound rules for port 5050
2. Verify server is running: `ps aux | grep node` on EC2
3. Verify server listens on 0.0.0.0: `netstat -tlnp | grep 5050` on EC2

### If Netlify proxy fails:
1. Check `netlify.toml` redirects are correct
2. Verify Netlify deployment completed
3. Check Netlify function logs for proxy errors

### If CORS errors appear:
1. Backend automatically allows `*.netlify.app` domains
2. Check backend logs on EC2 for CORS rejection messages
3. Verify `ORIGIN` env var on backend (should be empty or include Netlify domain)

## 📝 Summary

**What was fixed:**
- ✅ Server now listens on 0.0.0.0 (accepts external connections)
- ✅ Frontend uses relative paths (Netlify proxy handles HTTPS→HTTP)
- ✅ Enhanced error handling and diagnostics
- ✅ Health check indicator on Home page

**What you need to do:**
1. Run the EC2 deployment script (or manually update backend)
2. Verify backend is accessible from internet
3. Check Netlify deployment completed
4. Test the site in browser

**Expected result:**
- Frontend successfully connects to backend
- No CORS errors
- No mixed content errors
- Home page shows "✓ Backend Online"
- All API endpoints working
