from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os
import json
from f5_tts.api import F5TTS
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import logging
import shutil

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
class VoiceId(str, Enum):
    VOICE_0 = "voice_0"
    VOICE_1 = "voice_1"
    VOICE_2 = "voice_2"
    VOICE_3 = "voice_3"
    VOICE_4 = "voice_4"
    VOICE_5 = "voice_5"
    VOICE_6 = "voice_6"
    VOICE_7 = "voice_7"
    VOICE_8 = "voice_8"
    VOICE_9 = "voice_9"
    VOICE_10 = "voice_10"
    VOICE_11 = "voice_11"
    VOICE_12 = "voice_12"
    VOICE_13 = "voice_13"
    VOICE_14 = "voice_14"
    VOICE_15 = "voice_15"
    VOICE_16 = "voice_16"
    VOICE_17 = "voice_17"
    VOICE_18 = "voice_18"
    VOICE_19 = "voice_19"
    VOICE_20 = "voice_20"
    VOICE_21 = "voice_21"
    VOICE_22 = "voice_22"
    VOICE_23 = "voice_23"
    VOICE_24 = "voice_24"
    VOICE_25 = "voice_25"
    VOICE_26 = "voice_26"
    VOICE_27 = "voice_27"
    VOICE_28 = "voice_28"
    VOICE_29 = "voice_29"
    VOICE_30 = "voice_30"
    VOICE_31 = "voice_31"
    VOICE_32 = "voice_32"
    VOICE_33 = "voice_33"
    VOICE_34 = "voice_34"
    VOICE_35 = "voice_35"
    VOICE_36 = "voice_36"
    VOICE_37 = "voice_37"
    VOICE_38 = "voice_38"
    VOICE_39 = "voice_39"

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
                                  "F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav")
            ref_text = "some call me nature, others call me mother nature."
        
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

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "voices_available": len(VOICE_METADATA)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
