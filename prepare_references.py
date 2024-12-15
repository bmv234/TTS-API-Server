import os
import shutil
import json
import glob
import soundfile as sf
import numpy as np
import tempfile
from pydub import AudioSegment, silence

def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from the edges of an audio segment"""
    # Remove silence from the start
    non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
    audio = audio[non_silent_start_idx:]

    # Remove silence from the end
    non_silent_end_duration = audio.duration_seconds
    for ms in reversed(audio):
        if ms.dBFS > silence_threshold:
            break
        non_silent_end_duration -= 0.001
    trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

    return trimmed_audio

def get_first_complete_sentence_and_file(trans_file):
    """Get first complete sentence and its corresponding audio file from transcript"""
    with open(trans_file, 'r') as f:
        for line in f:
            # Skip empty lines
            if not line.strip():
                continue
            # Split into file ID and text
            parts = line.strip().split()
            if not parts:
                continue
            file_id = parts[0]
            text = ' '.join(parts[1:])
            # Convert to proper case
            text = text.capitalize()
            # Look for simple, complete sentences
            if len(text.split()) >= 8 and len(text.split()) <= 15:  # Longer sentences like default voice
                # Get corresponding flac file
                flac_file = f"{file_id}.flac"
                return text, flac_file
    return None, None

def get_speaker_genders():
    """Parse SPEAKERS.TXT to create a mapping of speaker IDs to genders"""
    speaker_genders = {}
    speakers_file = "LibriSpeech/SPEAKERS.TXT"
    
    with open(speakers_file, 'r') as f:
        for line in f:
            # Skip comments and empty lines
            if line.startswith(';') or not line.strip():
                continue
            
            # Split line by pipe and strip whitespace
            parts = [part.strip() for part in line.split('|')]
            
            # Extract speaker ID and gender
            if len(parts) >= 2:
                speaker_id = parts[0].strip()
                gender = parts[1].strip()
                speaker_genders[speaker_id] = gender
    
    return speaker_genders

def prepare_reference_voices():
    """Prepare reference voices from existing LibriSpeech files"""
    # Create output directories
    REFERENCE_DIR = "reference_voices/english"
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    
    # Get speaker gender mapping
    speaker_genders = get_speaker_genders()
    
    # Metadata for each speaker
    metadata = {}
    
    # Process all speakers in dev-clean
    dev_clean_dir = "LibriSpeech/dev-clean"
    
    # Get all speaker directories
    speaker_dirs = [d for d in os.listdir(dev_clean_dir) if os.path.isdir(os.path.join(dev_clean_dir, d))]
    
    print(f"Found {len(speaker_dirs)} speakers")
    
    # For each speaker
    for speaker_id in sorted(speaker_dirs):
        speaker_dir = os.path.join(dev_clean_dir, speaker_id)
        
        # Get chapter directories
        chapter_dirs = [d for d in os.listdir(speaker_dir) if os.path.isdir(os.path.join(speaker_dir, d))]
        if not chapter_dirs:
            continue
            
        # Find a suitable audio file and transcript
        reference_text = None
        src_file = None
        
        for chapter_dir_name in chapter_dirs:
            chapter_dir = os.path.join(speaker_dir, chapter_dir_name)
            trans_file = os.path.join(chapter_dir, f"{speaker_id}-{chapter_dir_name}.trans.txt")
            
            if os.path.exists(trans_file):
                text, flac_name = get_first_complete_sentence_and_file(trans_file)
                if text and flac_name:
                    # Find corresponding audio file
                    potential_src = os.path.join(chapter_dir, flac_name)
                    if os.path.exists(potential_src):
                        reference_text = text
                        src_file = potential_src
                        break
        
        if not src_file or not reference_text:
            print(f"Skipping speaker {speaker_id} - no suitable audio/transcript found")
            continue
            
        dst_file = os.path.join(REFERENCE_DIR, f"speaker_{speaker_id}.wav")
        
        # Get total number of files for this speaker
        total_files = sum(len([f for f in os.listdir(os.path.join(speaker_dir, d)) if f.endswith('.flac')])
                         for d in chapter_dirs)
        
        # Convert flac to wav with specific processing
        print(f"Processing speaker {speaker_id} ({total_files} files)...")
        print(f"Reference text: \"{reference_text}\"")
        print(f"Audio file: {os.path.basename(src_file)}")
        
        # Convert FLAC to WAV with exact technical specifications matching default voice
        print(f"Processing speaker {speaker_id} ({total_files} files)...")
        print(f"Reference text: \"{reference_text}\"")
        print(f"Audio file: {os.path.basename(src_file)}")
        
        # First convert to WAV with exact specifications using sox
        # Add gain -0.1 to prevent clipping during resampling
        # Convert to WAV with exact specifications matching default voice
        # Use sox's built-in normalization without compression
        os.system(f'sox {src_file} -r 24000 -c 1 -b 16 {dst_file} norm -0.1')
        
        # Get gender from SPEAKERS.TXT mapping
        gender = speaker_genders.get(speaker_id, "Unknown")
        if gender == "Unknown":
            print(f"Warning: Could not find gender for speaker {speaker_id} in SPEAKERS.TXT")
        
        # Add metadata
        metadata[f"speaker_{speaker_id}.wav"] = {
            "speaker_id": speaker_id,
            "description": f"Speaker with {total_files} recordings",
            "gender": gender,
            "total_files": total_files,
            "sample_file": os.path.basename(src_file),
            "reference_text": reference_text
        }
    
    # Save metadata
    with open("reference_voices/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Count voices by gender
    male_voices = [id for id, info in metadata.items() if info["gender"] == "M"]
    female_voices = [id for id, info in metadata.items() if info["gender"] == "F"]
    
    print(f"\nSuccessfully prepared {len(metadata)} reference voices!")
    print(f"Reference voices are stored in: {REFERENCE_DIR}")
    print(f"Metadata is stored in: reference_voices/metadata.json")
    print(f"\nMale voices ({len(male_voices)}):", ", ".join(male_voices))
    print(f"Female voices ({len(female_voices)}):", ", ".join(female_voices))
    
    # Print top 5 speakers by number of files
    print("\nTop 5 speakers by number of recordings:")
    top_speakers = sorted(metadata.items(), key=lambda x: x[1]["total_files"], reverse=True)[:5]
    for voice_file, info in top_speakers:
        print(f"- {voice_file}: {info['total_files']} files ({info['gender']})")
        print(f"  Reference text: \"{info['reference_text']}\"")
        print(f"  Audio file: {info['sample_file']}")

if __name__ == "__main__":
    prepare_reference_voices()
