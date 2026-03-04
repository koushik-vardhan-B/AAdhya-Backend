"""
Download script for Layer 2 — Fraud Type Classifier
Model: ealvaradob/bert-finetuned-phishing
Run from detection_layer/: python download_layer2_model.py
"""

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ealvaradob/bert-finetuned-phishing"
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bert_phishing")

# Create directory if it doesn't exist
os.makedirs(save_path, exist_ok=True)

print(f"📦 Model : {model_name}")
print(f"📁 Target: {save_path}")
print("-" * 50)

print("⬇️  Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)
print("✅ Tokenizer saved!")

print("⬇️  Downloading model (this may take a few minutes)...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained(save_path)
print("✅ Model saved!")

# Show what was downloaded
print("-" * 50)
print("📂 Downloaded files:")
for f in sorted(os.listdir(save_path)):
    size_mb = os.path.getsize(os.path.join(save_path, f)) / (1024 * 1024)
    print(f"   {f} ({size_mb:.1f} MB)")

print("-" * 50)
print(f"✅ Done! Layer 2 model saved to {save_path}")
