#!/bin/bash
set -e

# Check if LibriSpeech needs to be installed
if [ ! -d "LibriSpeech/dev-clean" ]; then
    echo "First run detected. Installing LibriSpeech dataset..."
    python install_librispeech.py
    python prepare_references.py
fi

# Execute the main command
exec "$@"