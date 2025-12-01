import torch
from unsloth import FastLanguageModel
import pandas as pd
from tqdm import tqdm
import re

# --- CONFIGURATION ---
MODEL_PATH = "qwen7b_telelogs_finetune" # The folder where you saved your model
TEST_FILE = "phase_1_test.csv"
OUTPUT_FILE = "submission_track2.csv"
MAX_SEQ_LENGTH = 4096 

# --- 1. LOAD FINE-TUNED MODEL ---
print(f"Loading model from {MODEL_PATH}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_PATH,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# --- 2. PREPARE DATA ---
try:
    test_df = pd.read_csv(TEST_FILE)
except:
    print(f"Error: {TEST_FILE} not found!")
    exit()

# System Prompt (MUST match training)
system_prompt = (
    "You are an expert 5G Network Engineer specialized in Root Cause Analysis (RCA). "
    "You are analyzing raw telelog data and engineering parameters to diagnose network faults. "
    "Your task is to identify the precise Root Cause ID (e.g., C1, C2, etc.) responsible for the reported issue. "
    "Pay close attention to log patterns, timestamps, and error signatures."
)

# --- 3. GENERATION LOOP ---
results = []
print(f"Generating responses for {len(test_df)} questions...")

# Batch size 1 is safest for generating 4 sequences at once (effectively batch=4)
for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
    original_id = row['ID']
    question = row['question'] # Adjust column name if it's 'Logs' or 'Instruction'

    # Apply Chat Template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    # Generate 4 Responses (Pass @ 1 Metric)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 64, # Short generation (we only need \boxed{Cx})
            use_cache = True,
            # SAMPLING SETTINGS (Crucial for 4 variations)
            do_sample = True, 
            temperature = 0.6, # Adds slight randomness for diversity
            top_p = 0.9,
            num_return_sequences = 4 # Generates 4 distinct answers
        )

    # Decode all 4 outputs
    decoded_texts = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)

    # Process each of the 4 generations
    for i, text in enumerate(decoded_texts):
        # Format ID as per challenge: ID_original_1, ID_original_2, etc.
        submission_id = f"{original_id}_{i+1}"
        
        # Regex to find \boxed{C2} or just C2
        match = re.search(r"\\boxed\{(C\d+)\}", text)
        if match:
            ans = match.group(1)
        else:
            # Fallback: look for just C + number (e.g., "The cause is C2")
            fallback = re.search(r"\b(C\d+)\b", text)
            ans = fallback.group(1) if fallback else "C1" # Default to C1 if model fails completely

        results.append({
            "ID": submission_id,
            "Qwen2.5-7B-Instruct": ans
        })

# --- 4. SAVE SUBMISSION ---
submission_df = pd.DataFrame(results)

# Create final CSV with the column required for Track 2
# Note: You will likely need to merge this with the official SampleSubmission.csv 
# to fill in the other tracks with placeholders.
submission_df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved {len(submission_df)} predictions to {OUTPUT_FILE}")
