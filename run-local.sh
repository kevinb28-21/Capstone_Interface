#!/bin/bash
# Script to run the entire application locally for testing
# This starts all required services: PostgreSQL, Node.js backend, Python worker, and frontend

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[i]${NC} $1"
}

echo "=========================================="
echo "Starting Local Development Environment"
echo "=========================================="
echo ""

# Step 1: Check PostgreSQL
print_info "Step 1: Checking PostgreSQL..."
if PGPASSWORD=aum28 psql -U drone_user -d drone_analytics -c "SELECT 1;" > /dev/null 2>&1; then
    print_status "PostgreSQL is running"
else
    print_error "PostgreSQL connection failed!"
    print_warning "Make sure PostgreSQL is running: brew services start postgresql@14"
    exit 1
fi

# Step 2: Check .env files
print_info "Step 2: Checking environment files..."
if [ -f "python_processing/.env" ]; then
    print_status "python_processing/.env exists"
else
    print_warning "python_processing/.env not found - creating..."
    cat > python_processing/.env << 'EOF'
# Python Flask API Environment Variables
FLASK_PORT=5001
FLASK_DEBUG=False

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=aum28

# File Paths
UPLOAD_FOLDER=./uploads
PROCESSED_FOLDER=./processed

# Background Worker Configuration
WORKER_POLL_INTERVAL=10
WORKER_BATCH_SIZE=5

# AWS S3 Configuration (optional for local dev)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-2
S3_BUCKET_NAME=
S3_ENABLED=False

# ML Model Configuration (optional)
ONION_MODEL_PATH=./models/onion_crop_health_model.h5
EOF
    print_status "Created python_processing/.env"
fi

if [ -f "server/.env" ]; then
    print_status "server/.env exists"
else
    print_warning "server/.env not found - creating..."
    cat > server/.env << 'EOF'
# Local Development Environment Variables
PORT=5050
NODE_ENV=development

# CORS: Allow local frontend
ORIGIN=http://localhost:5173,http://localhost:5182

# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=drone_analytics
DB_USER=drone_user
DB_PASSWORD=aum28

# AWS S3 Configuration (optional for local dev)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-2
S3_BUCKET_NAME=
EOF
    print_status "Created server/.env"
fi

# Step 3: Install dependencies if needed
print_info "Step 3: Checking dependencies..."

if [ ! -d "server/node_modules" ]; then
    print_warning "Installing Node.js dependencies..."
    cd server && npm install && cd ..
    print_status "Node.js dependencies installed"
fi

if [ ! -d "python_processing/venv" ]; then
    print_warning "Creating Python virtual environment..."
    cd python_processing
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
    deactivate
    cd ..
    print_status "Python virtual environment created"
fi

if [ ! -d "client/node_modules" ]; then
    print_warning "Installing frontend dependencies..."
    cd client && npm install && cd ..
    print_status "Frontend dependencies installed"
fi

# Step 4: Start services
echo ""
print_info "Step 4: Starting services..."
echo ""

# Start background worker
print_info "Starting Python background worker..."
cd python_processing
source venv/bin/activate
nohup python3 background_worker.py > /tmp/worker.log 2>&1 &
WORKER_PID=$!
cd ..
print_status "Background worker started (PID: $WORKER_PID)"

# Start Node.js backend
print_info "Starting Node.js backend..."
cd server
nohup npm run dev > /tmp/server.log 2>&1 &
BACKEND_PID=$!
cd ..
print_status "Node.js backend started (PID: $BACKEND_PID)"

# Wait a moment for services to start
sleep 3

# Start frontend
print_info "Starting React frontend..."
cd client
nohup npm run dev > /tmp/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
print_status "React frontend started (PID: $FRONTEND_PID)"

# Step 5: Display status
echo ""
echo "=========================================="
print_status "All services started!"
echo "=========================================="
echo ""
echo "Service URLs:"
echo "  Frontend:  http://localhost:5182 (or check vite.config.js for actual port)"
echo "  Backend:   http://localhost:5050"
echo "  Flask API: http://localhost:5001"
echo ""
echo "Process IDs:"
echo "  Worker:    $WORKER_PID"
echo "  Backend:   $BACKEND_PID"
echo "  Frontend:  $FRONTEND_PID"
echo ""
echo "Log Files:"
echo "  Worker:    tail -f /tmp/worker.log"
echo "  Backend:   tail -f /tmp/server.log"
echo "  Frontend:  tail -f /tmp/frontend.log"
echo ""
echo "To stop all services, run:"
echo "  ./stop-local.sh"
echo ""
echo "Or manually:"
echo "  kill $WORKER_PID $BACKEND_PID $FRONTEND_PID"
echo ""
print_info "Testing image processing..."
echo "  1. Open http://localhost:5182 in your browser"
echo "  2. Upload an image"
echo "  3. Check processing status in the UI"
echo "  4. View logs: tail -f /tmp/worker.log"
echo ""

