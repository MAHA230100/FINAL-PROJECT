FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY healthai /app/healthai

EXPOSE 8501
CMD ["streamlit", "run", "healthai/src/ui/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
