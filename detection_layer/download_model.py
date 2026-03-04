from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "AventIQ-AI/distilBERT_spam_detection"
save_path = "./models/distilbert_spam"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)

print("Downloading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.save_pretrained(save_path)

print(f"✅ Done! Model saved to {save_path}")