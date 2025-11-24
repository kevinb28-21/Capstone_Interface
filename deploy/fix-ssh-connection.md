# Fixing SSH Connection Issues

If SSH is timing out, here's how to troubleshoot and fix it.

## Common Causes

1. **Public IP Changed** - EC2 instances get new IPs when stopped/started (unless using Elastic IP)
2. **Security Group** - Port 22 not open from your IP
3. **Network ACLs** - Blocking SSH traffic
4. **Instance Status** - Instance might be stopping/stopped

## Step 1: Check Instance Status

1. **AWS Console** → **EC2** → **Instances**
2. Find your instance (`i-067f72e13b8724b18`)
3. Check:
   - **State**: Should be "running" (green circle)
   - **Public IPv4 address**: Note the current IP
   - **Public IPv4 DNS**: Note the current hostname

## Step 2: Verify Security Group

1. Click on your instance → **Security** tab
2. Click on the Security Group name
3. Check **Inbound rules**:
   - Should have rule: **Type: SSH, Port: 22, Source: Your IP or 0.0.0.0/0**
4. If missing, click **Edit inbound rules** → **Add rule**:
   - Type: SSH
   - Port: 22
   - Source: `0.0.0.0/0` (or your specific IP for security)
   - Click **Save rules**

## Step 3: Get Current Connection Details

The IP might have changed. Get the current details:

```bash
# Option A: Check AWS Console (see Step 1)

# Option B: Use AWS CLI (if installed)
aws ec2 describe-instances \
  --instance-ids i-067f72e13b8724b18 \
  --query 'Reservations[0].Instances[0].[PublicIpAddress,PublicDnsName,State.Name]' \
  --output table
```

## Step 4: Test SSH Connection

Try connecting with the current IP:

```bash
# Replace <current-ip> with the IP from AWS Console
ssh -i ~/Downloads/MS04_ID.pem ubuntu@<current-ip>

# Or with the DNS name (if it's updated)
ssh -i ~/Downloads/MS04_ID.pem ubuntu@ec2-<current-ip>.us-east-2.compute.amazonaws.com
```

## Step 5: Assign Elastic IP (Recommended)

To prevent IP changes, assign an Elastic IP:

1. **EC2 Console** → **Elastic IPs** → **Allocate Elastic IP address**
2. Click **Allocate**
3. Select the Elastic IP → **Actions** → **Associate Elastic IP address**
4. Select your instance → **Associate**

Now your IP won't change!

## Step 6: Update Scripts with New IP

Once you have a working connection, update the scripts:

```bash
# Edit deploy/migrate-and-update.sh
# Change EC2_HOST to the current IP or Elastic IP

# Or use the Elastic IP DNS name
```

## Alternative: EC2 Instance Connect

If SSH still doesn't work, try EC2 Instance Connect:

1. **EC2 Console** → Select instance → **Connect**
2. Choose **EC2 Instance Connect** tab
3. Click **Connect**

This opens a browser-based terminal (no SSH key needed, but limited to 60 seconds).

## Quick Fix Commands (Once Connected)

Once you can connect (via any method), run:

```bash
cd ~/Capstone_Interface

# Run migration
psql -U drone_user -d drone_analytics -f python_processing/database_migration_add_gndvi.sql

# Update code
git pull origin main || echo "Git not configured, will sync files manually"

# Update dependencies
cd server && npm install --production && cd ..
cd python_processing && source venv/bin/activate && pip install -r requirements.txt && deactivate && cd ..

# Restart services
pm2 restart all
```

