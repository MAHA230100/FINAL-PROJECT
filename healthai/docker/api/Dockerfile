FROM python:3.10-slim

WORKDIR /app

# Copy requirements from project root
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code folder
COPY healthai /app/healthai

EXPOSE 8000
CMD ["uvicorn", "healthai.src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
