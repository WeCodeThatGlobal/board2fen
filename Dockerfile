# Dockerfile

# 1. Use official Python base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install system dependencies (for OpenCV, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirement file and install deps
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY weights/ weights/

# 6. Expose FastAPI port
EXPOSE 8000

# 7. Start FastAPI with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
