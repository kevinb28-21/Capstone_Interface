# AWS EC2 Backend Setup Guide

This guide helps you set up your Node.js backend on AWS EC2 and connect it to your Netlify frontend.

## Overview

Your **VITE_API_URL** will be: `http://YOUR-EC2-IP:5050` or `https://your-domain.com` (if you set up a domain)

---

## Step 1: Launch EC2 Instance

1. **Go to AWS Console** → EC2 → Launch Instance

2. **Configure Instance**:
   - **Name**: `drone-crop-health-backend`
   - **AMI**: Ubuntu 22.04 LTS (free tier eligible)
   - **Instance Type**: t2.micro (free tier) or t3.small
   - **Key Pair**: Create new or use existing
   - **Network Settings**: 
     - Allow HTTP (port 80)
     - Allow HTTPS (port 443)
     - Allow Custom TCP (port 5050) - for your backend
     - Source: 0.0.0.0/0 (or restrict to your IP)

3. **Launch Instance**

4. **Note your Public IP**: You'll see it in the EC2 dashboard

---

## Step 2: Connect to EC2 Instance

### Via SSH:

```bash
ssh -i your-key.pem ubuntu@YOUR-EC2-PUBLIC-IP
```

Replace:
- `your-key.pem` with your key file path
- `YOUR-EC2-PUBLIC-IP` with your actual EC2 IP

---

## Step 3: Install Dependencies on EC2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Node.js 20.x
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install Git
sudo apt install git -y

# Install PM2 (process manager)
sudo npm install -g pm2

# Install Nginx (reverse proxy)
sudo apt install nginx -y
```

---

## Step 4: Clone and Setup Backend

```bash
# Clone your repository
git clone https://github.com/kevinb28-21/Capstone_Interface.git
cd Capstone_Interface/server

# Install dependencies
npm install

# Create .env file
nano .env
```

**Add to `.env` file:**
```env
PORT=5050
ORIGIN=https://your-netlify-app.netlify.app
NODE_ENV=production
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your-bucket
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=postgres
DB_PASSWORD=your-password
```

**Important**: Replace `your-netlify-app.netlify.app` with your actual Netlify URL!

---

## Step 5: Set Up PM2 (Process Manager)

Create `server/ecosystem.config.js`:

```js
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
    max_memory_restart: '1G'
  }]
}
```

**Start with PM2:**
```bash
cd ~/Capstone_Interface/server
pm2 start ecosystem.config.js
pm2 save
pm2 startup
# Follow the command it outputs to enable startup on boot
```

---

## Step 6: Configure Nginx (Reverse Proxy)

This allows you to use port 80/443 instead of 5050.

**Create Nginx config:**
```bash
sudo nano /etc/nginx/sites-available/drone-backend
```

**Add this configuration:**
```nginx
server {
    listen 80;
    server_name YOUR-EC2-PUBLIC-IP;  # Or your domain name

    location / {
        proxy_pass http://localhost:5050;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }
}
```

**Enable the site:**
```bash
sudo ln -s /etc/nginx/sites-available/drone-backend /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

---

## Step 7: Set Up SSL (Optional but Recommended)

**Install Certbot:**
```bash
sudo apt install certbot python3-certbot-nginx -y
```

**Get SSL certificate:**
```bash
sudo certbot --nginx -d your-domain.com
```

Or if using IP only (no domain):
- Use AWS Application Load Balancer with SSL
- Or use a service like Cloudflare Tunnel

---

## Step 8: Get Your Backend URL

### Option A: Using EC2 IP (HTTP)
```
http://YOUR-EC2-PUBLIC-IP:5050
```
**Example**: `http://54.123.45.67:5050`

### Option B: Using Nginx (Port 80)
```
http://YOUR-EC2-PUBLIC-IP
```
**Example**: `http://54.123.45.67`

### Option C: Using Domain (Recommended)
```
https://api.yourdomain.com
```
**Example**: `https://api.dronecrophealth.com`

---

## Step 9: Update Netlify Environment Variable

1. **Go to Netlify Dashboard**
2. **Your Site** → **Site settings** → **Environment variables**
3. **Add/Edit variable**:
   - **Key**: `VITE_API_URL`
   - **Value**: Your EC2 backend URL
     - `http://YOUR-EC2-IP:5050` (if using direct port)
     - `http://YOUR-EC2-IP` (if using Nginx on port 80)
     - `https://api.yourdomain.com` (if using domain with SSL)

4. **Redeploy** your Netlify site (or it will auto-redeploy)

---

## Step 10: Update CORS in Backend

Make sure your backend allows requests from Netlify:

**In `server/src/server.js`, update CORS:**
```js
const ORIGIN = process.env.ORIGIN || 'http://localhost:5173';

app.use(cors({ 
  origin: [
    ORIGIN,
    /\.netlify\.app$/  // Allow all Netlify domains
  ],
  credentials: true
}));
```

**Or in `.env` on EC2:**
```env
ORIGIN=https://your-netlify-app.netlify.app
```

---

## Step 11: Update netlify.toml

Edit `client/netlify.toml` and update the redirect:

```toml
[[redirects]]
  from = "/api/*"
  to = "http://YOUR-EC2-IP:5050/api/:splat"  # Or your domain
  status = 200
  force = true
```

Then commit and push:
```bash
git add client/netlify.toml
git commit -m "Update netlify.toml with EC2 backend URL"
git push
```

---

## Security Checklist

- [ ] EC2 Security Group allows port 5050 (or 80/443) from your IP or 0.0.0.0/0
- [ ] Backend `.env` file has correct `ORIGIN` (your Netlify URL)
- [ ] CORS is configured to allow Netlify domain
- [ ] AWS credentials are set in `.env` (not committed to git)
- [ ] PM2 is set to auto-restart on reboot
- [ ] Nginx is configured and running
- [ ] SSL certificate is installed (if using domain)

---

## Testing

1. **Test backend directly:**
   ```bash
   curl http://YOUR-EC2-IP:5050/api/health
   ```
   Should return: `{"status":"ok"}`

2. **Test from Netlify:**
   - Visit your Netlify site
   - Open browser console (F12)
   - Check Network tab for API calls
   - Should see requests to your EC2 backend

3. **Test image upload:**
   - Upload an image through Netlify frontend
   - Check EC2 logs: `pm2 logs drone-backend`
   - Verify image appears in S3 bucket

---

## Monitoring

**View logs:**
```bash
pm2 logs drone-backend
pm2 monit  # Real-time monitoring
```

**Check status:**
```bash
pm2 status
pm2 info drone-backend
```

**Restart:**
```bash
pm2 restart drone-backend
```

---

## Troubleshooting

### Backend not accessible
- Check EC2 Security Group allows port 5050
- Check if PM2 is running: `pm2 status`
- Check backend logs: `pm2 logs drone-backend`
- Test locally on EC2: `curl http://localhost:5050/api/health`

### CORS errors
- Verify `ORIGIN` in backend `.env` includes Netlify URL
- Check CORS configuration in `server.js`
- Verify Netlify URL matches exactly

### Nginx not working
- Check Nginx status: `sudo systemctl status nginx`
- Test config: `sudo nginx -t`
- Check error logs: `sudo tail -f /var/log/nginx/error.log`

---

## Your VITE_API_URL

Once your EC2 backend is running, your **VITE_API_URL** will be:

```
http://YOUR-EC2-PUBLIC-IP:5050
```

Or if using Nginx:
```
http://YOUR-EC2-PUBLIC-IP
```

Or if using a domain:
```
https://api.yourdomain.com
```

Set this in Netlify's environment variables, and your frontend will connect to your EC2 backend!

