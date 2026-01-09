#!/bin/bash
# Script to run on EC2 via Instance Connect
# Updates backend with CORS fixes for Netlify
# Run this on EC2: bash deploy/update-backend-cors-remote.sh

set -e

echo "=========================================="
echo "Updating Backend with CORS Fixes"
echo "=========================================="
echo ""

cd ~/Capstone_Interface || { echo "Error: ~/Capstone_Interface not found"; exit 1; }

# Step 1: Pull latest changes
echo "Step 1: Pulling latest changes from GitHub..."
git pull origin main || {
    echo "Warning: git pull failed. Continuing with local files..."
}

# Step 2: Update Node.js dependencies
echo ""
echo "Step 2: Updating Node.js dependencies..."
cd server
npm install --production
cd ..

# Step 3: Restart PM2 services
echo ""
echo "Step 3: Restarting PM2 services..."
if command -v pm2 &> /dev/null; then
    pm2 restart all
    pm2 save
    
    echo ""
    echo "PM2 Status:"
    pm2 status
    
    echo ""
    echo "Recent logs (last 20 lines):"
    pm2 logs --lines 20 --nostream
else
    echo "Error: PM2 not found. Please install PM2 or restart services manually."
    exit 1
fi

# Step 4: Test health endpoint
echo ""
echo "Step 4: Testing API health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:5050/api/health || echo "FAILED")
if [ "$HEALTH_RESPONSE" != "FAILED" ]; then
    echo "✓ Health check successful"
    echo "$HEALTH_RESPONSE" | head -5
else
    echo "✗ Health check failed - backend may not be running"
fi

# Step 5: Verify CORS configuration
echo ""
echo "Step 5: Verifying CORS configuration..."
if grep -q "netlifyPattern\|netlify\.app" server/src/server-enhanced.js 2>/dev/null || \
   grep -q "netlifyPattern\|netlify\.app" server/src/server.js 2>/dev/null; then
    echo "✓ CORS configuration includes Netlify support"
else
    echo "⚠ Warning: CORS configuration may not include Netlify support"
    echo "  Check server/src/server-enhanced.js or server/src/server.js"
fi

echo ""
echo "=========================================="
echo "Update Complete!"
echo "=========================================="
echo ""
echo "The backend should now:"
echo "  ✓ Allow requests from all Netlify domains (*.netlify.app)"
echo "  ✓ Support CORS for production and preview deployments"
echo ""
echo "To verify, check PM2 logs:"
echo "  pm2 logs --lines 50"
echo ""

