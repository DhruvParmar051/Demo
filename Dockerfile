# =============================================================================
# AegisRAG — API image
# Build:  docker build -t aegisrag:latest .
# Run:    docker run -p 8000:8000 -v $(pwd)/data:/app/data \
#                   -v $(pwd)/checkpoints:/app/checkpoints aegisrag:latest
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# System deps: cmake for llama-cpp-python, poppler for pdfplumber, build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
        libgomp1 \
        libjpeg-dev \
        zlib1g-dev \
        libpoppler-cpp-dev \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm || true

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "run.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--model", "m5"]
