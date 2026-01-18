# Save as: module_4_evaluation.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
rouge = evaluate.load("rouge")
print(f"Running evaluation on: {device}")

# 2. Load LOCAL Artifacts
tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
model = AutoModelForSeq2SeqLM.from_pretrained("./pegasus_model").to(device)

# 3. Load Test Data (Real validation data)
dataset = load_dataset("knkarthick/samsum", split="test[:10]") # Testing on 10 samples for speed
print("Dataset loaded.")

def generate_summary(batch):
    inputs = tokenizer(batch["dialogue"], return_tensors="pt", max_length=1024, truncation=True, padding=True).to(device)
    
    # Generate
    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=128, 
        num_beams=4, 
        length_penalty=0.8
    )
    
    # Decode
    batch["pred_summary"] = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return batch

# 4. Run Inference
print("Generating summaries for evaluation...")
results = dataset.map(generate_summary, batched=True, batch_size=2)

# 5. Calculate Metrics
print("Computing ROUGE scores...")
scores = rouge.compute(predictions=results["pred_summary"], references=results["summary"])

print("\n--- Evaluation Results (ROUGE) ---")
print(f"ROUGE-1: {scores['rouge1']:.4f}")
print(f"ROUGE-2: {scores['rouge2']:.4f}")
print(f"ROUGE-L: {scores['rougeL']:.4f}")   