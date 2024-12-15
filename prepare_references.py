import os
import shutil
import json
import glob

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
            if len(text.split()) < 8 and ',' not in text and ';' not in text:
                # Get corresponding flac file
                flac_file = f"{file_id}.flac"
                return text, flac_file
    return None, None

def prepare_reference_voices():
    """Prepare reference voices from existing LibriSpeech files"""
    # Create output directories
    REFERENCE_DIR = "reference_voices/english"
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    
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
        
        # Convert flac to wav and copy
        print(f"Processing speaker {speaker_id} ({total_files} files)...")
        print(f"Reference text: \"{reference_text}\"")
        print(f"Audio file: {os.path.basename(src_file)}")
        os.system(f"ffmpeg -i {src_file} -ac 1 -ar 24000 {dst_file} -y")
        
        # Determine gender based on speaker ID range
        # LibriSpeech typically has female speakers in 1xxx-2xxx range and male speakers in 3xxx-4xxx range
        speaker_num = int(speaker_id)
        gender = "F" if speaker_num < 3000 else "M"
        
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
