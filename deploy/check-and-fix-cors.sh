#!/bin/bash
# Check and Fix CORS Configuration on EC2
# This script verifies CORS settings and ensures Netlify domain is allowed

set -e

echo "=========================================="
echo "CORS Configuration Check & Fix"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get Netlify domain from netlify.toml or use default
NETLIFY_DOMAIN="https://cropview.netlify.app"
if [ -f "../netlify.toml" ]; then
    # Try to extract domain from netlify.toml
    DOMAIN=$(grep -oP 'https://[a-zA-Z0-9-]+\.netlify\.app' ../netlify.toml | head -1 || echo "$NETLIFY_DOMAIN")
    if [ ! -z "$DOMAIN" ]; then
        NETLIFY_DOMAIN="$DOMAIN"
    fi
fi

echo "Checking CORS configuration for Netlify domain: $NETLIFY_DOMAIN"
echo ""

# Navigate to server directory
cd ~/Capstone_Interface/server || {
    echo -e "${RED}✗ Error: Could not find server directory${NC}"
    exit 1
}

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ Warning: .env file not found${NC}"
    echo "Creating .env file from template..."
    
    if [ -f "../deploy/env-templates/server.env.template" ]; then
        cp ../deploy/env-templates/server.env.template .env
        echo -e "${GREEN}✓ Created .env from template${NC}"
    else
        echo -e "${RED}✗ Error: Could not find template file${NC}"
        exit 1
    fi
fi

# Read current ORIGIN value
CURRENT_ORIGIN=$(grep "^ORIGIN=" .env | cut -d'=' -f2- | tr -d '"' || echo "")

echo "Current ORIGIN configuration:"
if [ -z "$CURRENT_ORIGIN" ]; then
    echo -e "${YELLOW}  (not set - will use defaults)${NC}"
else
    echo "  $CURRENT_ORIGIN"
fi
echo ""

# Check if Netlify domain is already in ORIGIN
if echo "$CURRENT_ORIGIN" | grep -q "$NETLIFY_DOMAIN"; then
    echo -e "${GREEN}✓ Netlify domain ($NETLIFY_DOMAIN) is already in CORS configuration${NC}"
    echo ""
    echo "Current allowed origins:"
    echo "$CURRENT_ORIGIN" | tr ',' '\n' | sed 's/^/  - /'
    echo ""
    echo "No changes needed!"
    exit 0
fi

# Netlify domain not found - need to add it
echo -e "${YELLOW}⚠ Netlify domain ($NETLIFY_DOMAIN) not found in CORS configuration${NC}"
echo ""

# Backup .env file
cp .env .env.backup.$(date +%Y%m%d_%H%M%S)
echo -e "${GREEN}✓ Backed up .env file${NC}"

# Update ORIGIN
if [ -z "$CURRENT_ORIGIN" ]; then
    # No ORIGIN set, add it
    NEW_ORIGIN="$NETLIFY_DOMAIN,http://localhost:5173"
    echo "ORIGIN=$NEW_ORIGIN" >> .env
    echo -e "${GREEN}✓ Added ORIGIN to .env${NC}"
else
    # Append Netlify domain to existing ORIGIN
    NEW_ORIGIN="$CURRENT_ORIGIN,$NETLIFY_DOMAIN"
    # Remove any existing Netlify domain first (in case it's different)
    NEW_ORIGIN=$(echo "$NEW_ORIGIN" | sed 's|https://[a-zA-Z0-9-]*\.netlify\.app||g' | sed 's|,,|,|g' | sed 's|^,||' | sed 's|,$||')
    NEW_ORIGIN="$NEW_ORIGIN,$NETLIFY_DOMAIN"
    # Remove duplicates
    NEW_ORIGIN=$(echo "$NEW_ORIGIN" | tr ',' '\n' | sort -u | tr '\n' ',' | sed 's|,$||')
    
    # Update .env file
    if grep -q "^ORIGIN=" .env; then
        sed -i "s|^ORIGIN=.*|ORIGIN=$NEW_ORIGIN|" .env
    else
        echo "ORIGIN=$NEW_ORIGIN" >> .env
    fi
    echo -e "${GREEN}✓ Updated ORIGIN in .env${NC}"
fi

echo ""
echo "New ORIGIN configuration:"
echo "  $NEW_ORIGIN"
echo ""
echo "Allowed origins:"
echo "$NEW_ORIGIN" | tr ',' '\n' | sed 's/^/  - /'
echo ""

# Verify the update
if grep -q "$NETLIFY_DOMAIN" .env; then
    echo -e "${GREEN}✓ Netlify domain successfully added to CORS configuration${NC}"
else
    echo -e "${RED}✗ Error: Failed to add Netlify domain${NC}"
    exit 1
fi

# Check if server is running with PM2
echo ""
echo "Checking server status..."
if command -v pm2 &> /dev/null; then
    if pm2 list | grep -q "drone-backend\|backend\|server"; then
        echo -e "${YELLOW}⚠ Server is running with PM2${NC}"
        echo "Restarting server to apply CORS changes..."
        pm2 restart drone-backend || pm2 restart backend || pm2 restart server || {
            echo -e "${YELLOW}⚠ Could not restart PM2 process, trying to reload...${NC}"
            pm2 reload drone-backend || pm2 reload backend || pm2 reload server || true
        }
        echo -e "${GREEN}✓ Server restarted${NC}"
    else
        echo "No PM2 process found for backend"
    fi
else
    echo "PM2 not found - if server is running, restart it manually to apply changes"
fi

# Test CORS configuration
echo ""
echo "Testing CORS configuration..."
sleep 2  # Give server time to restart

# Get EC2 public IP
EC2_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
if [ "$EC2_IP" != "unknown" ]; then
    echo "Testing CORS from EC2..."
    CORS_TEST=$(curl -s -H "Origin: $NETLIFY_DOMAIN" \
        -H "Access-Control-Request-Method: GET" \
        -X OPTIONS \
        "http://localhost:5050/api/health" \
        -w "\nHTTP_CODE:%{http_code}" 2>/dev/null || echo "")
    
    if echo "$CORS_TEST" | grep -q "Access-Control-Allow-Origin"; then
        echo -e "${GREEN}✓ CORS test successful${NC}"
    else
        echo -e "${YELLOW}⚠ CORS test inconclusive (server may need more time to restart)${NC}"
    fi
fi

echo ""
echo "=========================================="
echo -e "${GREEN}CORS Configuration Updated Successfully!${NC}"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Netlify domain: $NETLIFY_DOMAIN"
echo "  - Added to CORS configuration"
echo "  - Server restarted (if running)"
echo ""
echo "Next steps:"
echo "  1. Test your Netlify site - API calls should work"
echo "  2. Check browser console for any CORS errors"
echo "  3. If issues persist, verify server logs:"
echo "     pm2 logs drone-backend"
echo ""






