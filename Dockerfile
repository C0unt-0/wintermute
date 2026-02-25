FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install system dependencies & Radare2
RUN apt-get update && apt-get install -y wget git build-essential \
    && git clone https://github.com/radareorg/radare2 \
    && radare2/sys/install.sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/
COPY api/ api/
COPY configs/ configs/

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

# API startup command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
