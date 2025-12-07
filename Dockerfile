FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# NLTK data for sentence tokenization
RUN python -c "import nltk; nltk.download('punkt')"

# Default settings: CPU inference, small batch
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# When the container starts:
# 1) (Optional) Assume medquad_raw.csv is mounted into data/raw
# 2) Build chunks + index if they don't exist
# 3) Launch FastAPI with uvicorn
CMD ["/bin/sh", "-c", "\
  if [ ! -f data/processed/medquad_chunks.csv ]; then \
    python scripts/preprocess_medquad.py; \
  fi && \
  if [ ! -f data/index/faiss_index.bin ]; then \
    python scripts/build_index.py; \
  fi && \
  uvicorn api.main:app --host 0.0.0.0 --port 8000 \
"]
