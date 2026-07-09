FROM python:3.10

ENV PYTHONUNBUFFERED=1
# Pin HuggingFace cache here (before any RUN that uses it) so
# pre-warmed models land in a path both root and the app user can read.
ENV HF_HOME=/app/.cache/huggingface

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

# ── Python deps (cached unless requirements.txt changes) ──────────
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel cython \
    && pip install --no-cache-dir -r requirements.txt

# ── Pre-warm models (cached after first build, never re-run unless
#    requirements.txt changes — models live in the image layer) ────
RUN mkdir -p /app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# ── npm deps (cached unless package-lock.json changes) ───────────
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm ci

# ── Copy source and build frontend ───────────────────────────────
COPY --chown=user . .
RUN cd frontend && npm run build

# ── Runtime data directory owned by app user ─────────────────────
RUN mkdir -p /app/data/tmp /app/data/cases /app/data/lancedb \
    && chown -R user:user /app/data /app/.cache

USER user

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
