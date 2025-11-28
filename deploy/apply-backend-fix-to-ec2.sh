#!/bin/bash
# Apply Backend Fix to EC2
# This script pushes changes to GitHub, then runs the fix script on EC2

set -e

KEY_FILE="$HOME/Downloads/MS04_ID.pem"
EC2_HOST="ec2-18-223-169-5.us-east-2.compute.amazonaws.com"
EC2_USER="ubuntu"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "=========================================="
echo "Apply Backend Fix to EC2"
echo "=========================================="
echo ""

# Step 1: Check if we're in a git repo
print_step "Step 1: Checking Git repository..."
cd "$PROJECT_DIR"

if [ ! -d .git ]; then
    print_error "Not a git repository!"
    exit 1
fi

# Check if there are uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have uncommitted changes!"
    echo "Current status:"
    git status --short
    echo ""
    read -p "Do you want to commit these changes first? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Committing changes..."
        git add -A
        git commit -m "Auto-commit before EC2 deployment"
        print_success "Changes committed"
    fi
fi

# Step 2: Push to GitHub
print_step "Step 2: Pushing changes to GitHub..."
if git push origin main 2>&1 | grep -q "Everything up-to-date\|remote:"; then
    print_success "Changes pushed to GitHub"
else
    print_warning "Git push had issues (may need to set upstream or authenticate)"
    echo "Continuing anyway..."
fi
echo ""

# Step 3: Check SSH key file
print_step "Step 3: Checking SSH key file..."
if [ ! -f "$KEY_FILE" ]; then
    print_error "Key file not found at: $KEY_FILE"
    echo ""
    echo "Please provide the path to your SSH key file:"
    read -p "Key file path: " KEY_FILE
    KEY_FILE="${KEY_FILE/#\~/$HOME}"
    
    if [ ! -f "$KEY_FILE" ]; then
        print_error "Key file not found at: $KEY_FILE"
        exit 1
    fi
fi

chmod 400 "$KEY_FILE"
print_success "SSH key file found: $KEY_FILE"
echo ""

# Step 4: Test SSH connection
print_step "Step 4: Testing SSH connection to EC2..."
if ssh -i "$KEY_FILE" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$EC2_USER@$EC2_HOST" "echo 'Connection successful'" 2>/dev/null; then
    print_success "SSH connection successful"
else
    print_warning "First connection - accepting host key..."
    ssh -i "$KEY_FILE" -o StrictHostKeyChecking=accept-new "$EC2_USER@$EC2_HOST" "echo 'Connection established'" 2>&1 || {
        print_error "Could not connect to EC2 instance"
        echo ""
        echo "Please verify:"
        echo "  1. EC2 instance is running"
        echo "  2. Security group allows SSH from your IP"
        echo "  3. Key file is correct: $KEY_FILE"
        exit 1
    }
fi
echo ""

# Step 5: Transfer fix script to EC2
print_step "Step 5: Transferring fix script to EC2..."
scp -i "$KEY_FILE" "$PROJECT_DIR/deploy/run-backend-fix-on-ec2.sh" "$EC2_USER@$EC2_HOST:~/run-backend-fix-on-ec2.sh" 2>/dev/null || {
    print_error "Failed to transfer fix script"
    exit 1
}
print_success "Fix script transferred"
echo ""

# Step 6: Run the fix script on EC2
print_step "Step 6: Running fix script on EC2..."
echo "This will take a few minutes..."
echo ""

ssh -i "$KEY_FILE" "$EC2_USER@$EC2_HOST" << 'ENDSSH'
    # Make script executable
    chmod +x ~/run-backend-fix-on-ec2.sh
    
    # Navigate to project directory
    cd ~/Capstone_Interface 2>/dev/null || {
        echo "Project directory not found. Cloning repository..."
        git clone https://github.com/kevinb28-21/Capstone_Interface.git ~/Capstone_Interface || {
            echo "Error: Could not clone repository"
            echo "Please clone it manually or ensure it exists at ~/Capstone_Interface"
            exit 1
        }
    }
    
    # Pull latest changes
    cd ~/Capstone_Interface
    git pull origin main || echo "Git pull had issues, continuing..."
    
    # Run the fix script
    bash ~/run-backend-fix-on-ec2.sh
ENDSSH

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    print_success "Fix Applied Successfully!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Test the backend:"
    echo "   curl http://$EC2_HOST/api/health"
    echo ""
    echo "2. Check PM2 status on EC2:"
    echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'pm2 status'"
    echo ""
    echo "3. Monitor logs:"
    echo "   ssh -i $KEY_FILE $EC2_USER@$EC2_HOST 'pm2 logs drone-backend'"
    echo ""
else
    print_error "Fix script encountered errors (exit code: $EXIT_CODE)"
    echo ""
    echo "Please check the output above for errors."
    echo "You can also SSH into EC2 and run the script manually:"
    echo "  ssh -i $KEY_FILE $EC2_USER@$EC2_HOST"
    echo "  bash ~/run-backend-fix-on-ec2.sh"
    exit 1
fi

