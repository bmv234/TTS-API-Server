# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.6.3-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_NO_CACHE_DIR=1

# Install dependencies in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    ffmpeg \
    sox \
    libsndfile1 \
    curl \
    openssl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && find /var/cache -type f -delete

# Set up application directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
# Install regular requirements
COPY requirements.txt requirements_no_git.txt
RUN sed -i '/git+/d' requirements_no_git.txt && \
    pip install --no-cache-dir -r requirements_no_git.txt && \
    find /usr/local -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true

# Install F5-TTS with build approach
RUN pip install --no-cache-dir setuptools>=61.0 pip>=23.0 build && \
    cd /tmp && \
    git clone https://github.com/SWivid/F5-TTS.git && \
    cd F5-TTS && \
    python -m build && \
    pip install --no-cache-dir dist/*.whl || \
    (echo "Failed to install F5-TTS" && exit 1)
# Copy application files
COPY main-ui.py main.py prepare_references.py install_librispeech.py docker-test.py ./
COPY templates templates/
COPY docker-entrypoint.sh /usr/local/bin/
COPY src/ src/

# Set up directories and SSL cert in one layer
RUN mkdir -p output uploads LibriSpeech reference_voices/english \
    && openssl req -x509 -newkey rsa:4096 -nodes \
       -keyout /app/key.pem -out /app/cert.pem -days 365 \
       -subj '/CN=localhost'

# Set up entrypoint and clean up
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && rm -rf /root/.cache /tmp/*

# Configure container
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -k -f https://localhost:8000/health || exit 1

# Set entrypoint and default command
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["python", "main-ui.py"]
