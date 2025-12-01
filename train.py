import torch
import os
from unsloth import FastLanguageModel
from datasets import Dataset
import pandas as pd
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# --- CONFIGURATION ---
# Use only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 75% memory limit on A6000
torch.cuda.set_per_process_memory_fraction(0.75, device=0)

MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 4096 
OUTPUT_DIR = "qwen7b_telelogs_finetune_advanced_system_prompt"

# --- 1. LOAD MODEL ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, 
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none", 
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- 2. PREPARE DATA ---
try:
    df = pd.read_csv("train.csv")
except:
    print("Error: train.csv not found!")
    exit()

dataset = Dataset.from_pandas(df)

# --- IMPROVED SYSTEM PROMPT HERE ---
def format_to_text(examples):
    questions = examples["question"]
    answers = examples["answer"]
    texts = []
    
    # "Contextual" System Prompt
    # This tells the model exactly HOW to look at the data.
    system_prompt = (
        "You are an expert 5G Network Engineer specialized in Root Cause Analysis (RCA). "
        "You are analyzing raw telelog data and engineering parameters to diagnose network faults. "
        "Your task is to identify the precise Root Cause ID (e.g., C1, C2, etc.) responsible for the reported issue. "
        "Pay close attention to log patterns, timestamps, and error signatures."
    )

    for question, answer in zip(questions, answers):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"\\boxed{{{answer}}}"} 
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    return {"text": texts}

dataset = dataset.map(format_to_text, batched=True)

# Tokenize with strict limits
def tokenize_batch(examples):
    model_inputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048, # Strict limit to keep training stable
        padding="max_length",
    )
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_ds = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=dataset.column_names,
)

# --- 3. CUSTOM TRAINER ---
# This class bypasses the bug in Unsloth that was causing your crash
class CustomTrainer(Trainer):
    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)

        return loss.detach()

# --- 4. TRAIN ---
trainer = CustomTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = tokenized_ds,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=False), 
    args = TrainingArguments(
        per_device_train_batch_size = 4, 
        gradient_accumulation_steps = 2, 
        warmup_steps = 10,
        max_steps = 300, 
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "none", 
    ),
)

print("Starting training with Enhanced System Prompt...")
trainer.train()

print(f"Saving model to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Done!")
