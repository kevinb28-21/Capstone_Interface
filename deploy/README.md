# EC2 Deployment Guide

## Quick Fixes

### Fix Processing Issues on EC2

If you're experiencing image processing failures on EC2, use the automated fix script:

**From your local machine:**
```bash
./deploy/deploy-processing-fixes.sh
```

**Or directly on EC2:**
```bash
cd ~/Capstone_Interface
git pull origin main
./deploy/fix-ec2-processing.sh
```

This script will:
- ✅ Check/create `.env` file with proper configuration
- ✅ Fix database file paths for failed images
- ✅ Restart PM2 services
- ✅ Verify the fixes are working

### Common Processing Issues Fixed

1. **Numpy Type Errors**: Fixed by registering psycopg2 adapters for numpy types
2. **File Not Found Errors**: Fixed by correcting upload directory paths
3. **Database Connection Errors**: Fixed by ensuring `.env` file exists with correct credentials

This directory contains automated deployment scripts for setting up the Drone Crop Health Platform on AWS EC2.

## Quick Start

### Step 1: Connect to EC2 and Transfer Files

From your local machine, run:

```bash
# Make sure your key file is in the current directory or specify the path
chmod 400 id_rsa

# Transfer deployment files to EC2
scp -i id_rsa -r deploy/ ec2-user@ec2-18-224-7-31.us-east-2.compute.amazonaws.com:~/
```

### Step 2: Run Setup Script on EC2

SSH into your EC2 instance:

```bash
ssh -i id_rsa ec2-user@ec2-18-224-7-31.us-east-2.compute.amazonaws.com
```

Then run:

```bash
chmod +x ~/deploy/ec2-setup.sh
~/deploy/ec2-setup.sh
```

### Step 3: Configure Environment Variables

After the script completes, edit the environment files:

```bash
nano ~/Capstone_Interface/server/.env
nano ~/Capstone_Interface/python_processing/.env
```

### Step 4: Restart Services

```bash
pm2 restart all
pm2 status
```

## Files Included

- `ec2-setup.sh` - Main setup script
- `nginx.conf` - Nginx configuration
- `env-templates/` - Environment variable templates

## Manual Steps Required

1. Edit `.env` files with your actual credentials
2. Configure AWS Security Group (ports 5050, 5001, 80, 443)
3. Update Netlify environment variable `VITE_API_URL`

