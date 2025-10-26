# Base stage for common setup
FROM python:3.10-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONPATH="/app"

# Development stage
FROM base as development

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Set development-specific environment variables
ENV STREAMLIT_SERVER_RUN_ON_SAVE=true

# Production stage
FROM base as production

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy application code
COPY ui /app/ui

# Set production environment variables
ENV ENVIRONMENT=production \
    STREAMLIT_SERVER_RUN_ON_SAVE=false

# Expose port and set working directory
EXPOSE 8501
WORKDIR /app/ui

# Command to run the application
CMD ["streamlit", "run", "dashboard.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]
