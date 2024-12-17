import os
import requests
import tarfile
from collections import defaultdict
import csv
from tqdm import tqdm
import re

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

def download_file(url, destination):
    """Download a file with progress bar"""
    if os.path.exists(destination):
        print(f"{destination} already exists, skipping download")
        return
        
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def analyze_speakers():
    """Analyze speaker demographics"""
    # Download and extract LibriSpeech dev-clean
    tar_path = "dev-clean.tar.gz"
    download_file(LIBRISPEECH_URL, tar_path)
    
    if not os.path.exists("LibriSpeech/dev-clean"):
        print("\nExtracting archive...")
        with tarfile.open(tar_path) as tar:
            tar.extractall()
    
    # Analyze available speakers in dev-clean
    dev_clean_dir = "LibriSpeech/dev-clean"
    
    print("\nAnalyzing speakers...")
    speakers = {}
    
    for speaker_dir in os.listdir(dev_clean_dir):
        speaker_path = os.path.join(dev_clean_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            files = []
            # Count audio files for this speaker
            for chapter in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter)
                if os.path.isdir(chapter_path):
                    for file in os.listdir(chapter_path):
                        if file.endswith('.flac'):
                            files.append(os.path.join(chapter_path, file))
            if files:  # Only include speakers with audio files
                speakers[speaker_dir] = {
                    'files': files,
                    'file_count': len(files)
                }
    
    # Sort speakers by number of files
    sorted_speakers = sorted(speakers.items(), key=lambda x: x[1]['file_count'], reverse=True)
    
    # Print results
    print(f"\nFound {len(speakers)} speakers in dev-clean")
    print("\nTop 10 speakers by number of audio files:")
    for speaker_id, info in sorted_speakers[:10]:
        print(f"- Speaker {speaker_id}: {info['file_count']} files")
        
    print("\nSample audio files from top speaker:")
    top_speaker = sorted_speakers[0]
    for file in top_speaker[1]['files'][:3]:
        print(f"- {os.path.basename(file)}")
    
    # Cleanup
    print("\nCleaning up...")
    if os.path.exists(tar_path):
        os.remove(tar_path)
    
    return sorted_speakers

if __name__ == "__main__":
    analyze_speakers()
