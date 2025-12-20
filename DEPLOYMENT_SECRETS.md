# Production Environment Setup Guide

## Using GitHub Secrets in Docker Deployment

### Step 1: Add Secret to GitHub

1. Go to your GitHub repository
2. Navigate to: **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Add:
   - **Name:** `GOOGLE_API_KEY`
   - **Value:** `***` (your API key)
5. Click **"Add secret"**

### Step 2: Update Deployment Workflow

Your workflow needs to:
1. Create `.env` file on the server with the secret
2. Docker will read it from `data/.env`

**Option A: Create .env during deployment** (Recommended)

Add this step to your workflow before docker compose:

```yaml
- name: Create .env file
  run: |
    mkdir -p data
    echo "GOOGLE_API_KEY=${{ secrets.GOOGLE_API_KEY }}" > data/.env
    chmod 600 data/.env  # Secure permissions
```

**Option B: Pass as environment variable in docker-compose**

Update `docker-compose.yml` to include:
```yaml
services:
  api:
    environment:
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
```

Then in workflow:
```yaml
- name: Deploy with Docker Compose
  run: |
    export GOOGLE_API_KEY="${{ secrets.GOOGLE_API_KEY }}"
    docker compose up -d --build
```

### Step 3: Verify on Server

SSH into EC2 and check:
```bash
# Verify .env exists
cat data/.env

# Check Docker logs
docker compose logs api | grep "API Key"
```

---

## Complete Workflow Example

```yaml
name: Deploy to EC2

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      - name: Deploy to EC2
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ${{ secrets.EC2_USER }}
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /path/to/FINAL-PROJECT
            git pull origin main
            
            # Create .env file with secrets
            mkdir -p data
            echo "GOOGLE_API_KEY=${{ secrets.GOOGLE_API_KEY }}" > data/.env
            chmod 600 data/.env
            
            # Deploy
            docker compose down
            docker compose up -d --build
            
            # Verify
            docker compose logs api | tail -20
```

---

## Security Best Practices

1. **Never commit `.env` to Git**
   ```bash
   # Ensure this is in .gitignore
   echo "data/.env" >> .gitignore
   echo ".env" >> .gitignore
   ```

2. **Secure file permissions**
   ```bash
   chmod 600 data/.env  # Only owner can read/write
   ```

3. **Use different keys for dev/prod**
   - Dev: Local `.env` file
   - Prod: GitHub secrets

4. **Rotate keys regularly**
   - Update in GitHub secrets
   - Redeploy

---

## Quick Test

After deployment, test the API:
```bash
# From EC2 server
curl http://localhost:8000/ai-v2/features

# Should return JSON, not 404 or error
```

---

## Troubleshooting

**Problem:** API can't find GOOGLE_API_KEY

**Solutions:**
1. Check `.env` file exists: `ls -la data/.env`
2. Check content: `cat data/.env`
3. Check Docker logs: `docker compose logs api | grep -i "api key"`
4. Restart containers: `docker compose restart api`

**Problem:** Permission denied on .env

```bash
chmod 600 data/.env
chown $USER:$USER data/.env
```
