import requests
import os
import json
import time

def load_metadata():
    """Load voice metadata"""
    with open("reference_voices/metadata.json") as f:
        return json.load(f)

def test_health():
    """Test the health check endpoint"""
    response = requests.get('http://localhost:8000/health')
    print(f"\nHealth check response: {response.json()}")
    return response.status_code == 200

def test_list_voices():
    """Test the list voices endpoint"""
    response = requests.get('http://localhost:8000/voices')
    if response.status_code == 200:
        voices = response.json()['voices']
        metadata = load_metadata()
        print("\nAvailable voices:")
        for voice in voices:
            voice_file = voice['file']
            print(f"- {voice['id']}: {voice_file}")
            if voice_file in metadata:
                metadata_info = metadata[voice_file]
                print(f"  Speaker {metadata_info['speaker_id']} ({metadata_info['gender']})")
                print(f"  Reference text: \"{metadata_info['reference_text']}\"")
                print(f"  Total recordings: {metadata_info['total_files']}")
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return False

def test_tts(text, voice_id=None, output_file="test_output.wav"):
    """Test the TTS endpoint with specified voice"""
    print(f"\nTesting TTS with{' voice ' + voice_id if voice_id else ' default voice'}")
    print(f"Text: {text}")
    
    data = {
        "text": text,
        "seed": 42
    }
    
    if voice_id:
        data["voice_id"] = voice_id
        # Get voice info
        metadata = load_metadata()
        response = requests.get('http://localhost:8000/voices')
        if response.status_code == 200:
            voices = response.json()['voices']
            voice_info = next((v for v in voices if v['id'] == voice_id), None)
            if voice_info:
                voice_file = voice_info['file']
                print(f"Using voice file: {voice_file}")
                if voice_file in metadata:
                    metadata_info = metadata[voice_file]
                    print(f"Speaker {metadata_info['speaker_id']} ({metadata_info['gender']})")
                    print(f"Reference text: \"{metadata_info['reference_text']}\"")
                    print(f"Total recordings: {metadata_info['total_files']}")
    
    # Wait a bit to ensure any previous file operations are complete
    time.sleep(1)
    
    # Remove output file if it exists
    if os.path.exists(output_file):
        try:
            os.unlink(output_file)
        except Exception as e:
            print(f"Warning: Could not remove existing file: {e}")
    
    response = requests.post(
        'http://localhost:8000/tts',
        json=data,
        stream=True  # Stream the response to handle large files
    )
    
    if response.status_code == 200:
        # Save the audio file
        try:
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Success! Audio saved to {output_file}")
            print(f"Process time: {response.headers.get('X-Process-Time', 'N/A')} seconds")
            return True
        except Exception as e:
            print(f"Error saving audio file: {e}")
            return False
    else:
        print(f"Error: {response.status_code}")
        try:
            print(response.json())
        except:
            print(response.text)
        return False

def main():
    """Run all tests"""
    print("Testing F5-TTS API Server")
    print("-------------------------")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed!")
        return
        
    # Test listing voices
    if not test_list_voices():
        print("Listing voices failed!")
        return
    
    # Get available voices
    response = requests.get('http://localhost:8000/voices')
    if response.status_code != 200:
        print("Could not get available voices!")
        return
    voices = response.json()['voices']
    
    # Test default voice
    if not test_tts(
        "Welcome to the text-to-speech demonstration.",
        None,
        "test_default.wav"
    ):
        print("Default voice test failed!")
        return
    
    # Test all available voices
    for voice in voices:
        voice_id = voice['id']
        # Use a simple test text for all voices
        if not test_tts(
            "This is a test of the text-to-speech system.",
            voice_id,
            f"test_{voice_id}.wav"
        ):
            print(f"{voice_id} test failed!")
            return
    
    print("\nAll tests completed successfully!")
    print("\nGenerated audio files:")
    print("- test_default.wav (default voice)")
    for voice in voices:
        print(f"- test_{voice['id']}.wav ({voice['file']})")

if __name__ == "__main__":
    main()
