#!/bin/bash

# Exit on error
set -e

echo "Starting TTS API Server installation..."

# Check if running as root for apt install
if [ "$EUID" -ne 0 ]; then 
    echo "Please run with sudo to install system dependencies"
    exit 1
fi

# Install system dependencies
echo "Installing system dependencies..."
apt update
apt install -y ffmpeg sox python3.12 python3.12-venv python3-pip libsndfile1

# Check Python version
python_version=$(python3.12 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if (( $(echo "$python_version 3.12" | awk '{print ($1 < $2)}') )); then
    echo "Error: Python 3.12 or later is required. Found version $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3.12 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create required directories
echo "Creating required directories..."
mkdir -p output uploads

# Generate SSL certificates
echo "Generating SSL certificates..."
openssl req -x509 -newkey rsa:4096 -nodes \
    -keyout key.pem -out cert.pem -days 365 \
    -subj '/CN=localhost'

# Install LibriSpeech dataset
echo "Installing LibriSpeech dataset..."
python install_librispeech.py

# Prepare reference voices
echo "Preparing reference voices..."
python prepare_references.py

echo "Installation complete!"
echo
echo "To start the server:"
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo
echo "2. Run the server:"
echo "   a. API-only version (recommended for production):"
echo "      python main.py"
echo
echo "   b. Version with web interface (recommended for testing):"
echo "      python main-ui.py"
echo
echo "The server will be available at https://localhost:8000"
echo "Note: Your browser will show a security warning due to the self-signed certificate."
echo "Click 'Advanced' and 'Proceed to localhost' to access the interface."