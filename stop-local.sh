#!/bin/bash
# Script to stop all local development services

echo "=========================================="
echo "Stopping Local Development Services"
echo "=========================================="
echo ""

# Kill background worker
echo "Stopping Python background worker..."
pkill -f "background_worker.py" && echo "  ✓ Stopped" || echo "  ✗ Not running"

# Kill Node.js backend
echo "Stopping Node.js backend..."
pkill -f "node.*server" && echo "  ✓ Stopped" || echo "  ✗ Not running"

# Kill frontend
echo "Stopping React frontend..."
pkill -f "vite" && echo "  ✓ Stopped" || echo "  ✗ Not running"

echo ""
echo "All services stopped!"
echo ""

