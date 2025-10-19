# Use the official Python image
FROM python:3.10-slim

WORKDIR /app

# Environment settings for stability across OS
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    PIP_NO_CACHE_DIR=1

# Install build tools for numpy/pandas/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip safely (handles both pip/pip3 cases)
RUN python -m ensurepip && python -m pip install --upgrade pip setuptools wheel

# Copy and install requirements with retries and long timeout
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --default-timeout=300 --retries=1 -r /app/requirements.txt --index-url=https://pypi.org/simple

# Copy source code
COPY healthai /app/healthai

EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "healthai/src/ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
