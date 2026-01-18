# Save as: module_3_model.py
from transformers import AutoModelForSeq2SeqLM

# 1. Load YOUR LOCAL Model
print("Loading local model architecture...")
model = AutoModelForSeq2SeqLM.from_pretrained("./pegasus_model")

# 2. Display Architecture Details (Requirement for Module 3)
print(f"\nModel Type: {model.config.model_type}")
print(f"Vocab Size: {model.config.vocab_size}")
print(f"Max Position Embeddings: {model.config.max_position_embeddings}")
print(f"Encoder Layers: {model.config.encoder_layers}")
print(f"Decoder Layers: {model.config.decoder_layers}")

print("\n--- Full Architecture (Snippet) ---")
print(model)