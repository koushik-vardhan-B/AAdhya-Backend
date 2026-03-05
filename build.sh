#!/usr/bin/env bash
# Render build script — installs deps + downloads ML models
set -o errexit

echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "⬇️  Downloading Layer 1 model (DistilBERT spam detection)..."
cd detection_layer
python download_model.py

echo "⬇️  Downloading Layer 2 model (BERT phishing classifier)..."
python download_layer2_model.py
cd ..

echo "✅ Build complete!"
