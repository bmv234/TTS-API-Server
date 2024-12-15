#!/bin/bash

# Stop on error
set -e

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: docker is not installed"
    exit 1
fi

# Check if nvidia-docker is installed
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
    echo "Warning: nvidia-docker runtime not found. GPU support may not be available."
    echo "Install nvidia-docker2 for GPU support."
fi

echo "Building F5-TTS Server Docker image..."
echo "Using CUDA 12.6.3 with development tools"

# Build the Docker image
docker build \
  --network=host \
  --tag f5-tts-server:latest \
  --tag f5-tts-server:cuda12.6.3 \
  .

echo
echo "Build complete!"
echo
echo "You can run the server with:"
echo "docker run --gpus all -p 8000:8000 f5-tts-server:latest"
echo
echo "To verify GPU support:"
echo "docker run --gpus all --rm f5-tts-server:latest python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"