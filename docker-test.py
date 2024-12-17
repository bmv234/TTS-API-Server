import os
import sys
import subprocess

def print_separator(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)

def search_for_f5_tts(start_path):
    print(f"\nSearching for f5_tts/f5-tts in {start_path}")
    patterns = ['f5_tts', 'f5-tts']
    try:
        for root, dirs, files in os.walk(start_path):
            # Check directory names
            for d in dirs:
                if any(pat.lower() in d.lower() for pat in patterns):
                    full_path = os.path.join(root, d)
                    print(f"Found directory: {full_path}")
                    # List contents of found directory
                    try:
                        print("  Contents:")
                        for item in os.listdir(full_path):
                            print(f"    {item}")
                    except Exception as e:
                        print(f"  Error listing contents: {e}")

            # Check file names
            for f in files:
                if any(pat.lower() in f.lower() for pat in patterns):
                    print(f"Found file: {os.path.join(root, f)}")
    except Exception as e:
        print(f"Error searching {start_path}: {e}")

# Print Python path
print_separator("PYTHON PATH")
for path in sys.path:
    print(path)

# Print pip list output
print_separator("PIP LIST OUTPUT")
try:
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"Error running pip list: {e}")

# Print directory structure
print_separator("DIRECTORY STRUCTURE")
print("Current working directory:", os.getcwd())

print("\nParent directory (/app/..) contents:")
try:
    for item in os.listdir('/app/..'):
        if os.path.isdir(os.path.join('/app/..', item)):
            print(f"DIR  {item}/")
        else:
            print(f"FILE {item}")
except Exception as e:
    print(f"Error listing parent directory: {e}")

print("\nRecursive /app directory listing:")
for root, dirs, files in os.walk('/app'):
    level = root.replace('/app', '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = '  ' * (level + 1)
    for f in files:
        print(f"{subindent}{f}")

# Try importing f5_tts
print_separator("F5-TTS IMPORT TEST")
try:
    import f5_tts
    print("Successfully imported f5_tts")
    print(f"f5_tts location: {f5_tts.__file__}")
except ImportError as e:
    print("Failed to import f5_tts")
    print(f"Error message: {str(e)}")

# Check site-packages
print_separator("SITE-PACKAGES CONTENT")
site_packages = [p for p in sys.path if 'site-packages' in p]
for site_dir in site_packages:
    print(f"\nChecking {site_dir}:")
    if os.path.exists(site_dir):
        for item in os.listdir(site_dir):
            if 'f5' in item.lower():
                print(f"Found: {item}")

# Search entire filesystem for f5_tts
print_separator("FILESYSTEM SEARCH FOR F5-TTS")
search_for_f5_tts('/')