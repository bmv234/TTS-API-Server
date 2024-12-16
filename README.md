# F5-TTS API Server

A simple REST API server that provides text-to-speech functionality using F5-TTS, with support for multiple voices.

## Installation Options

You can install and run the server either locally or using Docker. Choose the method that best suits your needs.

### Local Installation

#### Requirements
- Python 3.12
- NVIDIA GPU with CUDA support (optional, will fall back to CPU)
- ~2GB disk space for models
- ~1GB disk space for LibriSpeech dataset

#### Steps

1. Clone the repository:
```bash
git clone <repository-url>
cd TTS-API-Server
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create required directories:
```bash
mkdir -p output uploads
```

5. Prepare reference voices:
```bash
python prepare_references.py
```
This will:
- Download the LibriSpeech dev-clean dataset (~1GB)
- Process the audio files
- Create the reference_voices directory with 40 different voices
- Generate metadata.json with voice information

### Docker Installation

#### Requirements
- Docker 24.0 or later
- NVIDIA GPU with CUDA 12.6 support (optional)
- NVIDIA Container Toolkit (nvidia-docker2) for GPU support
- ~3GB disk space for Docker image
- ~2GB additional space for model cache

#### Steps

1. Install NVIDIA Container Toolkit (if using GPU):
```bash
# Ubuntu/Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

2. Clone the repository:
```bash
git clone <repository-url>
cd TTS-API-Server
```

3. Make the build script executable:
```bash
chmod +x docker-build.sh
```

4. Build the Docker image:
```bash
./docker-build.sh
```
This will:
- Download and prepare the LibriSpeech dataset
- Install all dependencies
- Configure GPU support with CPU fallback
- Set up health monitoring

5. Create local directories for persistence:
```bash
mkdir -p output uploads
```

## Running the Server

### Local Running

The server runs over HTTPS using self-signed certificates for security (required for microphone access). After completing the local installation:

1. Generate SSL certificates (if not already present):
```bash
openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365 -subj '/CN=localhost'
```

2. Run the server in one of two ways:

   a. API-only version (recommended for production):
   ```bash
   # Activate virtual environment if not already active
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

   # Run API server
   python main.py
   ```

   b. Version with web interface (recommended for testing):
   ```bash
   # Activate virtual environment if not already active
   source venv/bin/activate  # Linux/Mac
   # or
   .\venv\Scripts\activate  # Windows

   # Run web interface server
   python main-ui.py
   ```

Both versions will run on https://localhost:8000. When accessing the server:
- Your browser will show a security warning because of the self-signed certificate
- Click "Advanced" and "Proceed to localhost" to access the interface
- The web interface provides a user-friendly interface for testing the TTS functionality
- The API-only version is more suitable for production deployments where only the REST API is needed

Note: HTTPS is required for security-sensitive features like microphone access in modern browsers.

### Docker Running

After completing the Docker installation, you can run the server using either method:

1. Using docker-compose (recommended):
```bash
# Start the server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the server
docker-compose down
```

2. Using docker command:
```bash
# Start the server
docker run -d --name tts-server \
  --gpus all \
  -p 8000:8000 \
  -v ./output:/app/output \
  -v ./uploads:/app/uploads \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  f5-tts-server:latest

# View logs
docker logs -f tts-server

# Stop the server
docker stop tts-server
docker rm tts-server
```

### Verifying Installation

After starting the server, verify it's working:

1. Check the health endpoint:
```bash
curl --insecure https://localhost:8000/health
```

2. Test text-to-speech:
```bash
curl --insecure -X POST "https://localhost:8000/tts" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello, this is a test."}' \
     --output test.wav
```

3. Open web interface (if using main-ui.py):
   Visit https://localhost:8000 in your browser
   - You'll see a security warning due to the self-signed certificate
   - Click "Advanced" and "Proceed to localhost" to access the interface

## Testing the API

The repository includes a test script that demonstrates using different voices:

```bash
python test_tts.py
```

This will generate three audio files:
- test_default.wav - Using the default voice
- test_voice0.wav - Using the first alternative voice
- test_voice1.wav - Using the second alternative voice

## API Endpoints

### GET /voices
Get a list of available voices.

Response:
```json
{
    "voices": [
        {
            "id": "voice_0",
            "file": "speaker_1234.wav"
        },
        ...
    ]
}
```

### POST /tts
Convert text to speech using either the default voice or a specific voice.

Request body:
```json
{
    "text": "Text to convert to speech",
    "voice_id": "voice_0",  // Optional, uses default voice if not specified
    "seed": -1  // Optional, random seed for reproducibility (-1 for random)
}
```

Parameters:
- `text` (required): The text to convert to speech (1-1000 characters)
- `voice_id` (optional): ID of the voice to use (get available IDs from /voices endpoint)
- `seed` (optional): Random seed for reproducibility (-1 for random)

Response:
- Audio file (WAV format)

### POST /clone
Clone a voice and use it for text-to-speech conversion.

Request:
- `multipart/form-data` with the following fields:
  - `reference_audio`: WAV file containing the voice to clone
  - `reference_text`: Text that is spoken in the reference audio
  - `text`: Text to convert to speech using the cloned voice

Response:
- Audio file (WAV format)

### GET /health
Health check endpoint.

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "voices_available": 40,
    "device": "cuda",  // or "cpu" when running without GPU
    "using_mixed_precision": true,  // false when running on CPU
    "memory_info": {
        "total_gpu_memory": "8.00 GB",
        "free_gpu_memory": "6.54 GB",
        "gpu_utilization": "18.2%"
    }
}
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: https://localhost:8000/docs
- ReDoc: https://localhost:8000/redoc

Note: When accessing the API documentation or making API calls:
- For development with self-signed certificates, use the `--insecure` flag with curl or disable certificate verification in your client
- For production, use properly signed SSL certificates from a trusted Certificate Authority

## Features

- Text-to-speech conversion using F5-TTS Base model
- Multiple voice options from LibriSpeech dataset
- Voice cloning capability
- Comprehensive error handling
- Detailed logging
- CORS support for web applications
- Automatic temporary file cleanup
- Health check endpoint for monitoring

## Voice Selection

The server provides multiple voices to choose from:
1. Default voice - The original F5-TTS reference voice
2. 40 additional voices from the LibriSpeech dataset
   - Each voice has its own unique characteristics
   - Mix of male and female voices
   - Voice IDs range from voice_0 to voice_39
   - Use the /voices endpoint to get the list of available voices with gender information

## Performance

- First-time startup will download required models (about 2GB)
- Audio generation typically takes a few seconds per request
- The server requires a CUDA-capable GPU for optimal performance
- Processing time is included in response headers as 'X-Process-Time'

## Error Handling

The API handles various error cases:
- 400: Invalid input (text too long/short)
- 404: Voice ID not found
- 500: Internal server error
- 503: Model not initialized

## Docker Deployment

You can run the server using Docker with GPU support:

### Using docker-compose (Recommended)

1. Build and start the server:
```bash
docker-compose up --build
```

2. Stop the server:
```bash
docker-compose down
```

### Manual Docker Commands

1. Build the image:
```bash
./docker-build.sh
```

2. Run the server:
```bash
docker run --gpus all -p 8000:8000 \
  -v ./output:/app/output \
  -v ./uploads:/app/uploads \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  f5-tts-server:latest
```

### Docker Requirements

- NVIDIA GPU with CUDA 12.6 support (optional, will fall back to CPU if not available)
- Docker 24.0 or later
- NVIDIA Container Toolkit (nvidia-docker2) for GPU support
- At least 8GB of GPU memory recommended
- ~2GB disk space for LibriSpeech dataset and models

### Docker Build Process

The Docker build will:
1. Download the LibriSpeech dev-clean dataset (~1GB)
2. Prepare reference voices automatically
3. Install all required dependencies
4. Configure GPU support with CPU fallback
5. Set up health monitoring

### Docker Notes

- Models are cached in ~/.cache/huggingface
- Output and upload directories are mounted as volumes for persistence
- Health checks are performed every 30 seconds
- Container automatically restarts unless explicitly stopped
- Uses CUDA 12.6 development environment for optimal performance
- Automatically falls back to CPU if GPU is not available

## Troubleshooting

### Common Installation Issues

1. CUDA/GPU Issues:
   - Error: "CUDA not available"
   - Solution: The server will fall back to CPU. For GPU support, ensure NVIDIA drivers and CUDA 12.6 are installed.

2. Docker GPU Access:
   - Error: "GPU unavailable in container"
   - Solution: Ensure nvidia-container-toolkit is installed and docker service was restarted

3. Memory Issues:
   - Error: "CUDA out of memory"
   - Solution: Reduce batch size or use CPU mode if GPU memory is insufficient

4. Missing Voices:
   - Error: "Voice not found" or empty /voices response
   - Solution: Run prepare_references.py again or rebuild Docker image

### Common Runtime Issues

1. Port Conflicts:
   - Error: "Address already in use"
   - Solution: Stop other services using port 8000 or change the port:
     ```bash
     # Local
     PORT=8001 python main.py
     
     # Docker
     docker run -p 8001:8000 ...
     ```

2. Permission Issues:
   - Error: "Permission denied" for output/uploads
   - Solution: Ensure directories have correct permissions:
     ```bash
     chmod 777 output uploads
     ```

3. Disk Space:
   - Error: "No space left on device"
   - Solution: Clear old output files or HuggingFace cache:
     ```bash
     rm -rf output/*
     rm -rf ~/.cache/huggingface/hub
     ```

## Notes

- The server uses the F5-TTS Base model
- Reference voices are generated during installation/build
- Voice metadata is stored in reference_voices/metadata.json
- Temporary files are automatically cleaned up after each request
- All API endpoints support CORS for web integration
- Models are cached in ~/.cache/huggingface for faster subsequent starts
