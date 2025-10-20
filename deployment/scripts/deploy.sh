#!/bin/bash
set -e

# Get the project directory from the first argument or use current directory
PROJECT_DIR="${1:-$(pwd)}"
VENV_DIR="$PROJECT_DIR/venv"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

log "🚀 Starting deployment process..."

# Activate virtual environment
if [ -d "$VENV_DIR" ]; then
    log "🔧 Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    log "❌ Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Install/update dependencies
log "📦 Installing/updating Python dependencies..."
pip install --upgrade pip
pip install -r "$PROJECT_DIR/requirements.txt"

# Run database migrations (if any)
if [ -f "$PROJECT_DIR/alembic.ini" ]; then
    log "🔄 Running database migrations..."
    cd "$PROJECT_DIR"
    alembic upgrade head
fi

# Collect static files (if any)
if [ -d "$PROJECT_DIR/static" ]; then
    log "📁 Collecting static files..."
    # Add your static files collection command here if needed
    # python manage.py collectstatic --noinput
fi

# Restart application
log "🔄 Restarting application..."
sudo supervisorctl restart healthai

# Check application status
sleep 5
APP_STATUS=$(sudo supervisorctl status healthai | awk '{print $2}')

if [ "$APP_STATUS" = "RUNNING" ]; then
    log "✅ Deployment successful! Application is running."
    exit 0
else
    log "❌ Deployment failed! Application is not running."
    log "📝 Application status: $(sudo supervisorctl status healthai)"
    exit 1
fi
