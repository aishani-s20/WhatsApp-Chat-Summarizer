# Save as: module_2_preprocessing.py
from transformers import AutoTokenizer
import pandas as pd

# 1. Load YOUR LOCAL Tokenizer
print("Loading local tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")

# 2. Simulate Raw WhatsApp Data
raw_chat = """
12/05/2025, 10:00 PM - John: Hey, are we meeting tomorrow?
12/05/2025, 10:01 PM - Sarah: Yes, at the cafe.
"""

# 3. Preprocess (Clean & Tokenize)
def clean_text(text):
    # Simple cleaning for demo
    return text.replace("12/05/2025, 10:00 PM - ", "").replace("12/05/2025, 10:01 PM - ", "")

cleaned_text = clean_text(raw_chat)
print(f"\nCleaned Text:\n{cleaned_text}")

# 4. Tokenization (The Core Requirement)
tokens = tokenizer(cleaned_text, truncation=True, padding="max_length", max_length=50)

print("\n--- Tokenization Output (First 20 tokens) ---")
print(f"Input IDs: {tokens['input_ids'][:20]}")
print(f"Attention Mask: {tokens['attention_mask'][:20]}")
print("\n[Success] Preprocessing module demonstrated with local tokenizer.")