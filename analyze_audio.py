import soundfile as sf
import numpy as np

def analyze_audio(file_path):
    print(f"Analyzing: {file_path}")
    data, samplerate = sf.read(file_path)
    print(f"Sample rate: {samplerate} Hz")
    print(f"Duration: {len(data)/samplerate:.2f} seconds")
    print(f"Channels: {data.shape[1] if len(data.shape) > 1 else 1}")
    print(f"Data type: {data.dtype}")
    print(f"Max amplitude: {np.max(np.abs(data))}")
    print(f"RMS value: {np.sqrt(np.mean(data**2))}")
    print("-" * 50)

# Analyze default voice
print("\nDefault voice:")
analyze_audio("F5-TTS/src/f5_tts/infer/examples/basic/basic_ref_en.wav")

# Analyze a custom voice
print("\nCustom voice:")
analyze_audio("reference_voices/english/speaker_1272.wav")