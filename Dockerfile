# ========= Base image =========
FROM python:3.10-slim

# Do not generate .pyc and always flush stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ========= Workdir inside the container =========
WORKDIR /app

# ========= Install Python deps =========
# Copy only requirements first to leverage Docker build cache
COPY requirements.txt .

# (optional but recommended) system tools your libs might need
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libglib2.0-0 \
        libgl1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

# ========= Copy project code =========
# This will copy src/, models/, config.yaml, etc.
COPY . .

# ========= Environment for the app =========
# Adjust paths if your layout is different
ENV MODEL_DIR=/app/models \
    CONFIG_FILE=/app/config.yaml \
    APP_MODULE=src.api_fastapi:app \
    PORT=8000

# Tell Docker we’ll listen on 8000
EXPOSE 8000

# ========= Default command =========
# Runs your FastAPI app with uvicorn
CMD ["uvicorn", "src.api_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
