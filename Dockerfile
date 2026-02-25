# Wintermute — Containerized Inference Server
# Phase 4 stub — to be completed with full multi-stage build

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for Capstone disassembly
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcapstone-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python package
COPY pyproject.toml .
COPY src/ src/
COPY api/ api/
COPY configs/ configs/

RUN pip install --no-cache-dir ".[api]"

# TODO (Phase 4): Add model weights, Gunicorn config, health checks
# COPY malware_model.safetensors .
# COPY data/processed/vocab.json data/processed/

EXPOSE 8000

# CMD ["gunicorn", "api.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
CMD ["echo", "Phase 4: Dockerfile not yet complete"]
