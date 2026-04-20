FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for sentence-transformers, llama-cpp-python, PDF parsing, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
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

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -m spacy download en_core_web_sm || true

COPY . .

EXPOSE 8000 8501

CMD ["python", "run.py", "serve", "--model", "m5", "--port", "8000"]
