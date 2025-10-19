FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=200 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m ensurepip && python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --default-timeout=300 --retries=1 -r /app/requirements.txt --index-url=https://pypi.org/simple

COPY healthai /app/healthai

EXPOSE 8000

CMD ["uvicorn", "healthai.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
