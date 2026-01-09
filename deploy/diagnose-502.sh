#!/bin/bash
# Quick diagnostic script for 502 Bad Gateway error

echo "=========================================="
echo "502 Bad Gateway Diagnostic"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 1. PM2 Status
echo "1. PM2 Status:"
pm2 list 2>/dev/null | grep -E "drone-backend|server|online|errored|stopped" || echo -e "${RED}✗ PM2 not running or backend not found${NC}"
echo ""

# 2. Port 5050
echo "2. Port 5050 Status:"
if sudo netstat -tlnp 2>/dev/null | grep ":5050" || sudo ss -tlnp 2>/dev/null | grep ":5050"; then
    echo -e "${GREEN}✓ Port 5050 is listening${NC}"
else
    echo -e "${RED}✗ Port 5050 is NOT listening${NC}"
fi
echo ""

# 3. Backend Health
echo "3. Backend Health Check:"
HEALTH=$(curl -s -w "\nHTTP_CODE:%{http_code}" http://localhost:5050/api/health 2>/dev/null || echo "ERROR")
if echo "$HEALTH" | grep -q "HTTP_CODE:200"; then
    echo -e "${GREEN}✓ Backend is responding${NC}"
    echo "$HEALTH" | grep -v "HTTP_CODE"
else
    echo -e "${RED}✗ Backend is NOT responding${NC}"
    echo "$HEALTH"
fi
echo ""

# 4. Nginx Status
echo "4. Nginx Status:"
if sudo systemctl is-active --quiet nginx; then
    echo -e "${GREEN}✓ Nginx is running${NC}"
else
    echo -e "${RED}✗ Nginx is NOT running${NC}"
fi
echo ""

# 5. Recent Nginx Errors
echo "5. Recent Nginx Errors (last 5 lines):"
if [ -f /var/log/nginx/error.log ]; then
    sudo tail -5 /var/log/nginx/error.log | grep -v "^$" || echo "No recent errors"
else
    echo "Error log not found"
fi
echo ""

# 6. Recent Backend Logs
echo "6. Recent Backend Logs (last 10 lines):"
pm2 logs drone-backend --lines 10 --nostream 2>/dev/null || echo "No logs available"
echo ""

# 7. Nginx Config Test
echo "7. Nginx Configuration Test:"
if sudo nginx -t 2>&1 | grep -q "successful"; then
    echo -e "${GREEN}✓ Nginx config is valid${NC}"
else
    echo -e "${RED}✗ Nginx config has errors:${NC}"
    sudo nginx -t
fi
echo ""

echo "=========================================="
echo "Diagnostic Complete"
echo "=========================================="
echo ""
echo "To fix issues, run:"
echo "  bash ~/Capstone_Interface/deploy/fix-502-bad-gateway.sh"
echo ""








