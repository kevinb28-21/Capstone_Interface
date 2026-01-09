#!/bin/bash

# Netlify Deployment Diagnostic Script
# This script helps diagnose why Netlify isn't deploying

echo "🔍 Netlify Deployment Diagnostic"
echo "=================================="
echo ""

# Check if netlify.toml exists
echo "1. Checking netlify.toml..."
if [ -f "netlify.toml" ]; then
    echo "   ✅ netlify.toml found"
    echo "   Contents:"
    cat netlify.toml | grep -E "^(base|command|publish)" | sed 's/^/      /'
else
    echo "   ❌ netlify.toml NOT FOUND"
    echo "   This file is required for Netlify deployment"
fi
echo ""

# Check git status
echo "2. Checking Git status..."
CURRENT_BRANCH=$(git branch --show-current)
echo "   Current branch: $CURRENT_BRANCH"

if [ "$CURRENT_BRANCH" = "main" ] || [ "$CURRENT_BRANCH" = "master" ]; then
    echo "   ✅ On production branch"
else
    echo "   ⚠️  Not on main/master branch"
    echo "   Netlify typically deploys from main/master"
fi

# Check for uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "   ⚠️  You have uncommitted changes"
    echo "   These won't deploy until committed and pushed"
else
    echo "   ✅ No uncommitted changes"
fi
echo ""

# Check if client directory exists
echo "3. Checking client directory..."
if [ -d "client" ]; then
    echo "   ✅ client/ directory exists"
    
    # Check package.json
    if [ -f "client/package.json" ]; then
        echo "   ✅ client/package.json exists"
    else
        echo "   ❌ client/package.json NOT FOUND"
    fi
    
    # Check if dist can be built
    echo "   Testing build..."
    cd client
    if npm run build 2>&1 | grep -q "error\|Error\|ERROR"; then
        echo "   ❌ Build failed - check errors above"
    else
        echo "   ✅ Build successful"
        if [ -d "dist" ]; then
            echo "   ✅ dist/ directory created"
            echo "   Files in dist: $(ls -1 dist | wc -l | tr -d ' ') files"
        else
            echo "   ❌ dist/ directory NOT CREATED"
        fi
    fi
    cd ..
else
    echo "   ❌ client/ directory NOT FOUND"
fi
echo ""

# Check recent commits
echo "4. Recent commits:"
git log --oneline -5 | sed 's/^/   /'
echo ""

# Check if pushed to remote
echo "5. Checking remote status..."
if git diff --quiet origin/$CURRENT_BRANCH..HEAD 2>/dev/null; then
    echo "   ✅ Local branch is up to date with remote"
else
    echo "   ⚠️  Local branch has commits not pushed to remote"
    echo "   Run: git push origin $CURRENT_BRANCH"
fi
echo ""

echo "=================================="
echo "📋 Next Steps:"
echo ""
echo "1. Go to https://app.netlify.com"
echo "2. Check your site's deploy status"
echo "3. Review build logs for errors"
echo "4. Verify build settings match netlify.toml:"
echo "   - Base directory: client"
echo "   - Build command: npm install && npm run build"
echo "   - Publish directory: dist (relative to base)"
echo "5. Check if site is paused (resume if needed)"
echo "6. Try 'Trigger deploy' manually"
echo ""
echo "For detailed troubleshooting, see:"
echo "docs/deployment/NETLIFY_TROUBLESHOOTING.md"

