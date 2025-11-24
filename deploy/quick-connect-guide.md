# Quick Connection Guide - Choose Your Method

## Method 1: Fix SSH (Fastest if working)

**If your instance has a stable IP or Elastic IP:**

1. Check current IP in AWS Console
2. Test connection:
   ```bash
   ssh -i ~/Downloads/MS04_ID.pem ubuntu@<current-ip>
   ```
3. If it works, run the migration script

## Method 2: EC2 Instance Connect (No setup needed)

**Works immediately, 60-second sessions:**

1. AWS Console → EC2 → Your instance → **Connect**
2. Choose **EC2 Instance Connect** tab
3. Click **Connect** (browser terminal opens)
4. Run commands (see "Quick Fix Commands" below)

**Note:** Limited to 60 seconds, but you can reconnect multiple times.

## Method 3: Session Manager (Best long-term)

**Requires 5-10 minute setup, then permanent access:**

1. Follow `setup-session-manager.md` to:
   - Install/verify SSM agent
   - Create IAM role with SSM permissions
   - Attach role to instance
2. Wait 5-10 minutes for propagation
3. Connect via: EC2 → Instance → Connect → Session Manager

## Method 4: Elastic IP + Update Scripts

**Prevent future IP changes:**

1. EC2 → Elastic IPs → Allocate
2. Associate with your instance
3. Update scripts with the Elastic IP
4. Use Method 1 going forward

---

## Quick Fix Commands (Run Once Connected)

No matter which method you use, once connected, run:

```bash
# 1. Navigate to project
cd ~/Capstone_Interface

# 2. Run database migration
export $(grep -v '^#' python_processing/.env | xargs)
psql -U "$DB_USER" -d "$DB_NAME" -h "${DB_HOST:-localhost}" \
  -f python_processing/database_migration_add_gndvi.sql

# 3. Verify migration
psql -U "$DB_USER" -d "$DB_NAME" -h "${DB_HOST:-localhost}" -c "
SELECT column_name FROM information_schema.columns 
WHERE table_name='analyses' AND column_name LIKE 'gndvi%';
"

# 4. Update code (if git is configured)
git pull origin main 2>/dev/null || echo "Git not configured"

# 5. Update dependencies
cd server && npm install --production && cd ..
cd python_processing && source venv/bin/activate && pip install -r requirements.txt --quiet && deactivate && cd ..

# 6. Restart services
pm2 restart all
pm2 status
```

**Expected result:** The GNDVI column error should be fixed, and image uploads should work!

