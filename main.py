from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
import json
from f5_tts.api import F5TTS
import uvicorn
from pydantic import BaseModel, Field, SecretStr
from typing import Optional, List, Literal
from enum import Enum
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, Header, HTTPException, Security
import logging
import shutil
import gc
import torch
from contextlib import contextmanager

@contextmanager
def torch_gc():
    """Context manager to handle PyTorch GPU memory cleanup"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

# Set PyTorch memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load voice metadata
METADATA_FILE = "reference_voices/metadata.json"
with open(METADATA_FILE) as f:
    VOICE_METADATA = json.load(f)

# Create voice enum from available voices
VoiceId = Enum('VoiceId', {f'VOICE_{i}': f'voice_{i}' for i in range(len(VOICE_METADATA))})

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to convert to speech")
    voice_id: Optional[VoiceId] = Field(
        default=None,
        description="Voice ID to use for synthesis. If not provided, uses the default voice."
    )
    seed: int = Field(
        default=-1,
        description="Random seed for reproducibility (-1 for random)"
    )

class Voice(BaseModel):
    id: str
    file: str
    speaker_id: str
    gender: str
    total_files: int

class VoiceList(BaseModel):
    voices: List[Voice]

# OpenAI voice mapping
OPENAI_VOICE_MAP = {
    "alloy": "voice_0",    # Map to first voice
    "echo": "voice_1",     # Map to second voice
    "fable": "voice_2",    # Map to third voice
    "onyx": "voice_3",     # Map to fourth voice
    "nova": "voice_4",     # Map to fifth voice
    "shimmer": "voice_5"   # Map to sixth voice
}

# Security
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the API key from Authorization header"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return  # Allow if no API key is set
    
    if not credentials or not credentials.credentials or credentials.credentials != f"Bearer {api_key}":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials

class OpenAITTSRequest(BaseModel):
    model: Literal["tts-1", "tts-1-hd"] = "tts-1"
    input: str = Field(..., min_length=1, max_length=1000, description="The text to generate audio for")
    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    response_format: Literal["mp3", "opus", "aac", "flac", "wav"] = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)

app = FastAPI(title="F5-TTS API Server")

# Initialize F5-TTS model
tts_model = None

@app.on_event("startup")
async def startup_event():
    """Initialize the TTS model on startup"""
    global tts_model
    try:
        logger.info("Initializing F5-TTS model...")
        tts_model = F5TTS()
        
        # Enable half precision and parallel processing
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                logger.info(f"Using {torch.cuda.device_count()} GPUs")
                tts_model.ema_model = torch.nn.DataParallel(tts_model.ema_model)
            tts_model.ema_model = tts_model.ema_model.half()
        
        logger.info("F5-TTS model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize F5-TTS model: {str(e)}")
        raise RuntimeError("Failed to initialize TTS model")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup output files on shutdown"""
    try:
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
        logger.info("Output files cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up output files: {str(e)}")

@app.get("/voices", response_model=VoiceList)
async def list_voices():
    """
    Get list of available voices
    """
    voices = []
    voice_files = list(VOICE_METADATA.keys())
    
    for i, filename in enumerate(voice_files):
        voice_id = f"voice_{i}"
        metadata = VOICE_METADATA[filename]
        voices.append(Voice(
            id=voice_id,
            file=filename,
            speaker_id=metadata["speaker_id"],
            gender=metadata["gender"],
            total_files=metadata["total_files"]
        ))
    
    return VoiceList(voices=voices)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech using F5-TTS
    """
    try:
        # Create a unique filename
        filename = f"tts_{hash(request.text)}_{request.seed}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
            
        # Get reference voice file
        if request.voice_id:
            voice_index = int(request.voice_id.value.split('_')[1])
            voice_file = list(VOICE_METADATA.keys())[voice_index]
            ref_file = os.path.join("reference_voices/english", voice_file)
            ref_text = VOICE_METADATA[voice_file]["reference_text"]
        else:
            # Use default voice
            ref_file = os.path.join(os.path.dirname(__file__),
                                  "src/f5_tts/infer/examples/basic/basic_ref_en.wav")
            ref_text = "some call me nature, others call me mother nature."
        
        # Warm up the model with a dummy inference
        logger.info("Warming up model...")
        dummy_text = "Testing, one two three."
        tts_model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=dummy_text,
            seed=-1
        )
        
        # Generate speech
        logger.info(f"Generating speech with voice: {request.voice_id or 'default'}")
        wav, sr, _ = tts_model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=request.text,
            file_wave=output_path,
            seed=request.seed
        )
        
        if not os.path.exists(output_path):
            raise RuntimeError("Generated audio file not found")
        
        # Return the audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="generated_speech.wav",
            background=None
        )
    
    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        if 'output_path' in locals() and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file after error: {str(cleanup_error)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/speech")
async def openai_tts(
    request: OpenAITTSRequest,
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    OpenAI-compatible TTS endpoint
    """
    try:
        # Map OpenAI voice to our voice ID
        voice_id = OPENAI_VOICE_MAP.get(request.voice)
        if not voice_id:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{request.voice}' not supported"
            )

        # Get reference voice file
        voice_file = list(VOICE_METADATA.keys())[int(voice_id.split('_')[1])]
        ref_file = os.path.join("reference_voices/english", voice_file)
        ref_text = VOICE_METADATA[voice_file]["reference_text"]

        # Create a unique filename
        filename = f"tts_{hash(request.input)}_{voice_id}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)

        # Generate speech
        logger.info(f"Generating speech with OpenAI-compatible endpoint using voice: {voice_id}")
        wav, sr, _ = tts_model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=request.input,
            file_wave=output_path,
            seed=-1
        )

        if not os.path.exists(output_path):
            raise RuntimeError("Generated audio file not found")

        # Convert to requested format if not WAV
        if request.response_format != "wav":
            try:
                from pydub import AudioSegment
            except ImportError:
                logger.warning("pydub not installed, falling back to WAV format")
                return FileResponse(
                    output_path,
                    media_type="audio/wav",
                    filename="speech.wav",
                    background=None
                )
            
            # Load the WAV file
            audio = AudioSegment.from_wav(output_path)
            
            # Create temporary file for converted audio
            with tempfile.NamedTemporaryFile(suffix=f".{request.response_format}", delete=False) as temp_file:
                converted_path = temp_file.name
                
                # Export to requested format
                audio.export(converted_path, format=request.response_format)
                
                # Return the converted audio
                response = FileResponse(
                    converted_path,
                    media_type=f"audio/{request.response_format}",
                    filename=f"speech.{request.response_format}",
                    background=None
                )
                
                # Clean up original WAV
                try:
                    os.unlink(output_path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup WAV file: {str(cleanup_error)}")
                
                return response

        # Return WAV if no conversion needed
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="speech.wav",
            background=None
        )

    except Exception as e:
        logger.error(f"Error in OpenAI-compatible endpoint: {str(e)}")
        if 'output_path' in locals() and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup file after error: {str(cleanup_error)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        memory_info = {
            "total_gpu_memory": f"{total_memory / (1024**3):.2f} GB",
            "free_gpu_memory": f"{free_memory / (1024**3):.2f} GB",
            "gpu_utilization": f"{(1 - free_memory/total_memory) * 100:.1f}%"
        }
    else:
        memory_info = {"gpu_status": "No GPU available"}

    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "voices_available": len(VOICE_METADATA),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "using_mixed_precision": True,
        "memory_info": memory_info
    }

@app.get("/v1/models")
async def list_models(
    credentials: HTTPAuthorizationCredentials = Security(security)
):
    """
    OpenAI-compatible models endpoint
    """
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": 1699488000,  # November 2023 (when OpenAI released their TTS)
                "owned_by": "f5-tts",
                "permission": [],
                "root": "tts-1",
                "parent": None
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": 1699488000,
                "owned_by": "f5-tts",
                "permission": [],
                "root": "tts-1-hd",
                "parent": None
            }
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem"
    )
