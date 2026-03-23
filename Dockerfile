FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# tesseract-ocr is needed for OCR tasks
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    gcc \
    python3-dev \
    musl-dev \
    && rm -rf /var/lib/apt/lists/*

# Pre-download heavy PyTorch dependencies with extended timeout and retries to avoid network drops
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

# Install Langchain and related splitters
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 \
    langchain langchain-text-splitters langchain-huggingface langchain-community

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 --retries=10 -r requirements.txt

COPY . .
