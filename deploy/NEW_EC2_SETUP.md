# New EC2 Instance Setup - Quick Guide

## Your New EC2 Instance
- **Instance ID**: i-0e34df16386b7de35
- **Hostname**: ec2-18-117-90-212.us-east-2.compute.amazonaws.com
- **IP Address**: 18.117.90.212
- **Key File**: `~/Downloads/MS04_ID.pem`

## Quick Setup (Recommended)

Run this from your local machine:

```bash
cd ~/Desktop/Projects/Capstone_Interface
bash deploy/run-setup-on-new-ec2.sh
```

This will:
1. ✅ Test SSH connection
2. ✅ Transfer setup script to EC2
3. ✅ Run complete automated setup
4. ✅ Install all dependencies
5. ✅ Setup database
6. ✅ Configure and start all services
7. ✅ Verify everything works

**Time**: 5-10 minutes

## Manual Setup (If automated fails)

### Step 1: SSH into EC2
```bash
chmod 400 ~/Downloads/MS04_ID.pem
ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-117-90-212.us-east-2.compute.amazonaws.com
```

### Step 2: Run Setup Script
Once connected to EC2:
```bash
# Clone repository (if needed)
git clone https://github.com/kevinb28-21/Capstone_Interface.git
cd Capstone_Interface

# Make setup script executable
chmod +x deploy/setup-new-ec2.sh

# Run setup
bash deploy/setup-new-ec2.sh
```

## What Gets Installed

- ✅ Node.js 20.x
- ✅ Python 3 with venv
- ✅ PostgreSQL
- ✅ PM2 (process manager)
- ✅ Nginx (reverse proxy)
- ✅ Git
- ✅ All project dependencies

## What Gets Configured

- ✅ PostgreSQL database (`drone_analytics`)
- ✅ Database user with secure password
- ✅ Database schema and migrations
- ✅ Node.js backend (port 5050)
- ✅ Flask API (port 5001)
- ✅ Background worker
- ✅ Nginx reverse proxy
- ✅ PM2 auto-start on boot
- ✅ Environment files (.env)

## Services Running

After setup, these services will be running:

1. **Node.js Backend** - `pm2 list` shows `drone-backend`
2. **Flask API** - `pm2 list` shows `flask-api`
3. **Background Worker** - `pm2 list` shows `background-worker`
4. **Nginx** - `sudo systemctl status nginx`

## Verification

### On EC2:
```bash
# Check PM2 services
pm2 list

# Test backend
curl http://localhost:5050/api/health

# Test through nginx
curl http://localhost/api/health

# Check nginx
sudo systemctl status nginx
```

### From Your Local Machine:
```bash
# Test backend
curl http://18.117.90.212/api/health
```

Should return: `{"status":"ok","database":"connected",...}`

## Important Files

- **Database Password**: Saved to `~/db_password.txt` on EC2
- **Server .env**: `~/Capstone_Interface/server/.env`
- **Python .env**: `~/Capstone_Interface/python_processing/.env`
- **Nginx Config**: `/etc/nginx/sites-available/drone-backend`

## Troubleshooting

### If setup fails:

1. **Check SSH access:**
   ```bash
   ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-117-90-212.us-east-2.compute.amazonaws.com
   ```

2. **Check security group:**
   - Port 22 (SSH) - from your IP
   - Port 80 (HTTP) - from anywhere (0.0.0.0/0)
   - Port 5050 - from anywhere (or just your IP)
   - Port 5001 - from anywhere (or just your IP)

3. **Check logs on EC2:**
   ```bash
   pm2 logs
   sudo tail -f /var/log/nginx/error.log
   ```

### If services don't start:

```bash
# Restart all PM2 services
pm2 restart all

# Check status
pm2 status

# View logs
pm2 logs drone-backend
pm2 logs flask-api
pm2 logs background-worker
```

### If database connection fails:

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check password in .env files
cat ~/Capstone_Interface/server/.env | grep DB_PASSWORD
cat ~/db_password.txt
```

## After Setup

1. ✅ **Update Netlify** (already done in `netlify.toml`)
   - Redeploy your Netlify site to apply new redirects

2. ✅ **Test from Frontend**
   - Go to: https://cropview.netlify.app
   - Open Analytics page
   - Upload an image
   - Should work without 502 error!

3. ✅ **Monitor Services**
   ```bash
   ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-117-90-212.us-east-2.compute.amazonaws.com
   pm2 monit
   ```

## Security Notes

- Database password is saved in `~/db_password.txt` on EC2
- Keep your SSH key secure
- Consider setting up firewall rules
- For production, use HTTPS/SSL

## Quick Commands Reference

```bash
# SSH into EC2
ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-18-117-90-212.us-east-2.compute.amazonaws.com

# Check services
pm2 list
pm2 logs
sudo systemctl status nginx

# Restart services
pm2 restart all
sudo systemctl restart nginx

# View logs
pm2 logs drone-backend --lines 50
pm2 logs flask-api --lines 50
pm2 logs background-worker --lines 50
```

