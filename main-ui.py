from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import json
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    infer_process,
    target_sample_rate,
    hop_length
)
import tqdm
import uvicorn
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import logging
import shutil
import soundfile as sf
import numpy as np
import re
import torch
import torchaudio
import asyncio
import gc
from torch.nn.parallel import DataParallel
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

# Create temp directory for uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load voice metadata
METADATA_FILE = "reference_voices/metadata.json"
with open(METADATA_FILE) as f:
    VOICE_METADATA = json.load(f)

# Create voice enum from available voices
VoiceId = Enum('VoiceId', {f'VOICE_{i}': f'voice_{i}' for i in range(len(VOICE_METADATA))})

# Dynamic batch size calculation
def get_optimal_batch_size(total_chunks):
    """Calculate optimal batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return min(1, total_chunks)
    
    # Start with a minimal batch size
    return min(1, total_chunks)  # Process one chunk at a time for stability

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to convert to speech")
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

def chunk_text(text, max_chars=100):  # Reduced size for better compatibility
    """Split text into chunks based on F5-TTS implementation"""
    chunks = []
    current_chunk = ""
    # Split the text into sentences based on punctuation followed by whitespace
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def process_audio(audio_data, sample_rate):
    """Minimal audio processing to preserve the start of speech"""
    return audio_data.astype(np.float32)
def process_text(text_chunks, ref_file, ref_text, model, device):
    """Process text chunks sequentially"""
    with torch_gc():
        # Load and preprocess reference audio once
        audio, sr = torchaudio.load(ref_file)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
            del resampler
    
    all_results = []
    ref_audio_len = audio.shape[-1] // hop_length
    ref_text_len = len(ref_text.encode("utf-8"))
    
    logger.info(f"Processing {len(text_chunks)} chunks sequentially")
    
    for chunk in text_chunks:
        with torch_gc():
            with torch.amp.autocast('cuda'), torch.inference_mode():
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Process single chunk
                chunk_len = len(chunk.encode("utf-8"))
                duration = ref_audio_len + int((ref_audio_len / ref_text_len) * chunk_len)
                duration = ((duration + 63) // 64) * 64  # Round up to nearest multiple of 64
                duration = min(duration, ref_audio_len * 3)
                
                # Prepare inputs
                cond_audio = audio.to(device)
                input_text = ref_text + chunk
                
                try:
                    generated, _ = model.sample(
                        cond=cond_audio,
                        text=[input_text],
                        duration=duration,
                        steps=32,
                        cfg_strength=3.0,
                        sway_sampling_coef=-1.0
                    )
                    
                    # Process generated audio
                    gen = generated[0:1]
                    gen = gen[:, ref_audio_len:duration, :]
                    gen = gen.permute(0, 2, 1)
                    
                    if hasattr(tts_model.vocoder, 'decode'):
                        wav = tts_model.vocoder.decode(gen)
                    else:
                        wav = tts_model.vocoder(gen)
                    
                    wav = wav.squeeze().cpu().numpy()
                    all_results.append(wav)
                    
                    # Clean up
                    del generated, gen, wav
                    cond_audio = cond_audio.cpu()
                    del cond_audio
                    
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    raise e
    
    return all_results, target_sample_rate

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
                tts_model.ema_model = DataParallel(tts_model.ema_model)
            tts_model.ema_model = tts_model.ema_model.half()
        
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
    """Convert text to speech using F5-TTS with batch processing"""
    temp_path = None
    output_path = None
    
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
                                "src/f5_tts/infer/examples/basic/basic_ref_en.wav")
            ref_text = "some call me nature, others call me mother nature."
        
        # Split text into chunks for batch processing
        text_chunks = chunk_text(request.text)
        
        # Process all chunks sequentially
        results, sr = process_text(text_chunks, ref_file, ref_text, tts_model.ema_model, tts_model.device)
        
        # Combine results with cross-fade
        final_wave = results[0]
        cross_fade_duration = 0.15  # seconds
        cross_fade_samples = int(cross_fade_duration * sr)
        
        for i in range(1, len(results)):
            prev_wave = final_wave
            next_wave = results[i]
            
            # Calculate cross-fade samples
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            
            if cross_fade_samples > 0:
                # Create cross-fade
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)
                
                # Apply cross-fade
                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]
                cross_faded = prev_overlap * fade_out + next_overlap * fade_in
                
                # Combine waves
                final_wave = np.concatenate([
                    prev_wave[:-cross_fade_samples],
                    cross_faded,
                    next_wave[cross_fade_samples:]
                ])
            else:
                final_wave = np.concatenate([prev_wave, next_wave])
        
        # Save the combined audio
        sf.write(output_path, final_wave, sr)
        
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        
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
            if path and os.path.exists(path):
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

        # Split text into chunks for batch processing
        text_chunks = chunk_text(text)
        
        # Process all chunks sequentially
        results, sr = process_text(text_chunks, ref_file, reference_text, tts_model.ema_model, tts_model.device)
        
        # Combine results with cross-fade
        final_wave = results[0]
        cross_fade_duration = 0.15  # seconds
        cross_fade_samples = int(cross_fade_duration * sr)
        
        for i in range(1, len(results)):
            prev_wave = final_wave
            next_wave = results[i]
            
            # Calculate cross-fade samples
            cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
            
            if cross_fade_samples > 0:
                # Create cross-fade
                fade_out = np.linspace(1, 0, cross_fade_samples)
                fade_in = np.linspace(0, 1, cross_fade_samples)
                
                # Apply cross-fade
                prev_overlap = prev_wave[-cross_fade_samples:]
                next_overlap = next_wave[:cross_fade_samples]
                cross_faded = prev_overlap * fade_out + next_overlap * fade_in
                
                # Combine waves
                final_wave = np.concatenate([
                    prev_wave[:-cross_fade_samples],
                    cross_faded,
                    next_wave[cross_fade_samples:]
                ])
            else:
                final_wave = np.concatenate([prev_wave, next_wave])
        
        # Save the combined audio
        sf.write(output_path, final_wave, sr)

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
