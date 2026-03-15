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
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel cython \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

USER user

CMD ["python", "app.py"]
