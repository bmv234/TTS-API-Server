# Use latest NVIDIA CUDA runtime image
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PATH="/opt/venv/bin:$PATH"

# Update package list and install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    ffmpeg \
    sox \
    libsndfile1 \
    curl \
    openssl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Create app directory and set working directory
WORKDIR /app

# Generate SSL certificates
RUN openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout /app/key.pem -out /app/cert.pem -days 365 \
    -subj '/CN=localhost'

# Create and activate virtual environment
RUN python3 -m venv /opt/venv

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p output uploads LibriSpeech reference_voices/english

# Copy application files
COPY main-ui.py main.py prepare_references.py ./
COPY templates templates/
COPY src src/

# Copy install script
COPY install_librispeech.py .
COPY docker-entrypoint.sh /usr/local/bin/

# Make entrypoint executable
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command (will be passed to entrypoint)
CMD ["python", "main-ui.py"]
