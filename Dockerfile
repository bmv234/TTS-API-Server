# Use latest NVIDIA CUDA development image
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    python3.12-venv \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Create and activate virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Create necessary directories
RUN mkdir -p output uploads LibriSpeech reference_voices/english

# Copy application files
COPY main-ui.py main.py prepare_references.py analyze_librispeech.py ./
COPY templates templates/
COPY src src/
# Download and prepare LibriSpeech dataset
RUN apt-get update && apt-get install -y wget unzip curl \
    && wget https://www.openslr.org/resources/12/dev-clean.tar.gz \
    && tar -xzf dev-clean.tar.gz \
    && mv LibriSpeech/dev-clean LibriSpeech/ \
    && rm -rf LibriSpeech/dev-clean.tar.gz \
    && python prepare_references.py \
    && rm -rf dev-clean.tar.gz \
    # Cleanup to reduce image size
    && apt-get remove -y wget unzip \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1



# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main-ui.py"]