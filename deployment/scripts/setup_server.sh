#!/bin/bash
set -e

echo "🚀 Starting server setup..."

# Update package lists
echo "🔄 Updating package lists..."
sudo apt-get update -y

# Install required packages
echo "📦 Installing required packages..."
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    nginx \
    supervisor \
    git \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Gunicorn
pip install gunicorn

# Set up Nginx
echo "🌐 Configuring Nginx..."
sudo rm -f /etc/nginx/sites-enabled/default

# Create Nginx config
sudo bash -c 'cat > /etc/nginx/sites-available/healthai << EOL
server {
    listen 80;
    server_name _;

    location / {
        include proxy_params;
        proxy_pass http://unix:/run/healthai.sock;
    }

    location /static/ {
        alias $1/static/;
    }
}
EOL'

# Enable the site
sudo ln -sf /etc/nginx/sites-available/healthai /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl restart nginx

# Set up Supervisor
echo "👨‍💼 Configuring Supervisor..."
sudo bash -c 'cat > /etc/supervisor/conf.d/healthai.conf << EOL
[program:healthai]
directory=$1
command=$1/venv/bin/gunicorn -w 3 -k uvicorn.workers.UvicornWorker src.api.main:app --bind unix:/run/healthai.sock
user=www-data
autostart=true
autorestart=true
stderr_logfile=/var/log/healthai/err.log
stdout_logfile=/var/log/healthai/out.log
EOL'

# Create log directory
sudo mkdir -p /var/log/healthai
sudo touch /var/log/healthai/err.log
sudo touch /var/log/healthai/out.log
sudo chown -R www-data:www-data /var/log/healthai

# Reload Supervisor
echo "🔄 Reloading Supervisor..."
sudo supervisorctl reread
sudo supervisorctl update

# Set proper permissions
echo "🔒 Setting permissions..."
sudo chown -R www-data:www-data $1
sudo chmod -R 755 $1

echo "✅ Server setup complete!"
