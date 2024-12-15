# F5-TTS API Server

A simple REST API server that provides text-to-speech functionality using F5-TTS, with support for multiple voices.

## Requirements

- Python 3.12
- All packages listed in requirements.txt
- NVIDIA GPU with CUDA support

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare reference voices:
```bash
python prepare_references.py
```
This will download and prepare 12 different voices from the LibriSpeech dataset.

## Running the Server

Start the server with:
```bash
python main.py
```

The server will run on http://localhost:8000

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

### GET /health
Health check endpoint.

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "voices_available": 12
}
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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
2. 12 additional voices from the LibriSpeech dataset
   - Each voice has its own unique characteristics
   - Voice IDs range from voice_0 to voice_11
   - Use the /voices endpoint to get the list of available voices

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

## Notes

- The server uses the F5-TTS Base model
- Reference voices are stored in the reference_voices/english directory
- Voice metadata is stored in reference_voices/metadata.json
- Temporary files are automatically cleaned up after each request
- All API endpoints support CORS for web integration
