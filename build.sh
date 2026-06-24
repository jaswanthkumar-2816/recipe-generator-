#!/usr/bin/env bash
# exit on error
set -o errexit

echo "Installing requirements..."
pip install -r requirements.txt

echo "Pre-downloading model weights and vocabs..."
python download_models.py

echo "Build script completed successfully."
