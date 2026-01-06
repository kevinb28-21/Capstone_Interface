#!/bin/bash
# Quick fix script to start backend on EC2
# Run this from your local machine

KEY_FILE="$HOME/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-223-169-5.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"

echo "=========================================="
echo "Fixing Backend on EC2"
echo "=========================================="
echo ""

# Check key file
if [ ! -f "$KEY_FILE" ]; then
    echo "Error: Key file not found at $KEY_FILE"
    exit 1
fi

chmod 400 "$KEY_FILE"

echo "Connecting to EC2 and starting backend..."
echo ""

ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
set -e

echo "=== Step 1: Navigate to server directory ==="
cd ~/Capstone_Interface/server || {
    echo "Error: Project directory not found"
    exit 1
}

echo "✓ In server directory"
echo ""

echo "=== Step 2: Stop existing backend (if any) ==="
pm2 delete drone-backend 2>/dev/null || echo "No existing process"
echo ""

echo "=== Step 3: Start backend with PM2 ==="
if [ -f ecosystem.config.cjs ]; then
    echo "Using ecosystem.config.cjs"
    pm2 start ecosystem.config.cjs
elif [ -f ecosystem.config.js ]; then
    echo "Using ecosystem.config.js"
    pm2 start ecosystem.config.js
else
    echo "Error: No PM2 config found!"
    exit 1
fi

pm2 save
echo "✓ Backend started"
echo ""

echo "=== Step 4: Wait for backend to start ==="
sleep 5
echo ""

echo "=== Step 5: PM2 Status ==="
pm2 list
echo ""

echo "=== Step 6: Test Backend Health ==="
HEALTH=$(curl -s -w "\nHTTP_CODE:%{http_code}" http://localhost:5050/api/health 2>/dev/null || echo "ERROR")
if echo "$HEALTH" | grep -q "HTTP_CODE:200"; then
    echo "✓ Backend is responding!"
    echo "$HEALTH" | grep -v "HTTP_CODE"
else
    echo "⚠ Backend not responding yet"
    echo "Checking logs..."
    pm2 logs drone-backend --lines 20 --nostream
fi
echo ""

echo "=== Step 7: Restart Nginx ==="
sudo systemctl restart nginx
echo "✓ Nginx restarted"
echo ""

echo "=== Step 8: Final Verification ==="
echo "Port 5050 status:"
sudo netstat -tlnp 2>/dev/null | grep ':5050' || sudo ss -tlnp 2>/dev/null | grep ':5050' || echo "Port 5050 not listening"

echo ""
echo "Backend through nginx:"
curl -s http://localhost/api/health 2>/dev/null | head -3 || echo "Not responding through nginx"

echo ""
echo "=========================================="
echo "Fix Complete!"
echo "=========================================="
echo ""
echo "If backend is not working, check logs:"
echo "  pm2 logs drone-backend --lines 50"
echo ""

ENDSSH

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✓ Backend fix applied successfully!"
    echo ""
    echo "Test from your browser:"
    echo "  http://18.223.169.5/api/health"
else
    echo ""
    echo "⚠ Some errors occurred. Check the output above."
    echo ""
    echo "You can also SSH manually and run:"
    echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
    echo "  cd ~/Capstone_Interface/server"
    echo "  pm2 start ecosystem.config.cjs"
    echo "  pm2 save"
fi




