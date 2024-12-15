from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
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
import soundfile as sf
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create temp directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

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

app = FastAPI(title="F5-TTS Web Interface")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize F5-TTS model
tts_model = None

def process_audio(audio_data, sample_rate):
    """Process audio to remove initial noise and add padding"""
    # Convert to float32 if not already
    audio_data = audio_data.astype(np.float32)
    
    # Add 0.5 seconds of silence at the start
    padding_samples = int(0.5 * sample_rate)
    padded_audio = np.concatenate([np.zeros(padding_samples), audio_data])
    
    # Find the first significant audio (threshold-based)
    threshold = 0.01  # Adjust if needed
    significant_audio_start = 0
    for i in range(len(padded_audio)):
        if abs(padded_audio[i]) > threshold:
            significant_audio_start = max(0, i - int(0.1 * sample_rate))  # Start slightly before
            break
    
    # Trim any noise before the actual speech
    processed_audio = padded_audio[significant_audio_start:]
    
    # Apply fade in
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    processed_audio[:fade_samples] *= fade_in
    
    return processed_audio

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
    """Cleanup output and upload files on shutdown"""
    try:
        shutil.rmtree(OUTPUT_DIR)
        shutil.rmtree(UPLOAD_DIR)
        os.makedirs(OUTPUT_DIR)
        os.makedirs(UPLOAD_DIR)
        logger.info("Output and upload files cleaned up")
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the web interface"""
    voices = []
    for i, (filename, metadata) in enumerate(VOICE_METADATA.items()):
        voice_id = f"voice_{i}"
        voices.append({
            "id": voice_id,
            "file": filename,
            "metadata": metadata
        })
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "voices": voices}
    )

@app.get("/voices", response_model=VoiceList)
async def list_voices():
    """Get list of available voices"""
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
    """Convert text to speech using F5-TTS"""
    try:
        # Create a unique filename
        filename = f"tts_{hash(request.text)}_{request.seed}.wav"
        temp_path = os.path.join(OUTPUT_DIR, "temp_" + filename)
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
            file_wave=temp_path,
            seed=request.seed
        )
        
        # Process the audio
        audio_data, sample_rate = sf.read(temp_path)
        processed_audio = process_audio(audio_data, sample_rate)
        
        # Save processed audio
        sf.write(output_path, processed_audio, sample_rate)
        
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
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
        for path in [temp_path, output_path]:
            if 'path' in locals() and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file after error: {str(cleanup_error)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clone")
async def clone_voice(
    reference_audio: UploadFile = File(...),
    reference_text: str = Form(...),
    text: str = Form(...)
):
    """Generate speech using a custom reference voice"""
    try:
        # Save uploaded file
        ref_file = os.path.join(UPLOAD_DIR, reference_audio.filename)
        temp_path = os.path.join(OUTPUT_DIR, f"temp_cloned_{hash(text)}_{hash(reference_text)}.wav")
        output_path = os.path.join(OUTPUT_DIR, f"cloned_{hash(text)}_{hash(reference_text)}.wav")
        
        with open(ref_file, "wb") as f:
            content = await reference_audio.read()
            f.write(content)

        # Generate speech with cloned voice
        logger.info("Generating speech with cloned voice")
        wav, sr, _ = tts_model.infer(
            ref_file=ref_file,
            ref_text=reference_text,
            gen_text=text,
            file_wave=temp_path
        )

        # Process the audio
        audio_data, sample_rate = sf.read(temp_path)
        processed_audio = process_audio(audio_data, sample_rate)
        
        # Save processed audio
        sf.write(output_path, processed_audio, sample_rate)

        # Cleanup files
        for path in [ref_file, temp_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup file: {str(e)}")

        if not os.path.exists(output_path):
            raise RuntimeError("Generated audio file not found")

        # Return the audio file
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="cloned_speech.wav",
            background=None
        )

    except Exception as e:
        logger.error(f"Error generating cloned speech: {str(e)}")
        # Cleanup files
        for path in [ref_file, temp_path, output_path]:
            if 'path' in locals() and os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup file after error: {str(cleanup_error)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": tts_model is not None,
        "voices_available": len(VOICE_METADATA)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)