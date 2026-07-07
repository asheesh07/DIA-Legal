FROM python:3.10

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    tesseract-ocr \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

# Python deps — cached layer unless requirements.txt changes
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel cython \
    && pip install --no-cache-dir -r requirements.txt

# npm deps — cached layer unless package-lock.json changes
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci

# Copy full source, then build the frontend
COPY --chown=user . .
RUN cd frontend && npm run build

USER user

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
