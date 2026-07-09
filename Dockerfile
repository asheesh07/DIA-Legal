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

# Pin the HuggingFace cache to a fixed path so pre-warmed models are
# found when the app later runs as 'user'. Without this, pre-warming
# caches to /root/.cache/ but the app looks in /home/user/.cache/ —
# completely different directories, so every first request re-downloads.
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

# Pre-download models into the shared cache path set above.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# data/ is gitignored so COPY never creates it.
# Pre-create it as root then hand ownership to the app user
# so runtime mkdir/write calls succeed.
RUN mkdir -p /app/data/tmp /app/data/cases /app/data/lancedb \
    && chown -R user:user /app/data /app/.cache

USER user

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
