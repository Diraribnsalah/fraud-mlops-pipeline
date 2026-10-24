FROM python:3.11-slim

WORKDIR /app

#  install indentical packages to ensure reproducibility
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

#  copy requirements.txt to the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

#  copy all files to the container
COPY . .

ENV PYTHONPATH=/app

EXPOSE 8000 8501
