import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# --- Configuration ---
MODEL_ID = "deepseek-ai/DeepSeek-V2-Lite-Chat" # Or your specific path
DATA_PATH = "data/prompts/single_turn.jsonl"   # Your data file
OUTPUT_DIR = "deepseek-lora-checkpoints"

# 1. Load Tokenizer & Set Padding
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right" # Fixed for fp16 training stability
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. Load Model in 4-bit (QLoRA)
#    DeepSeek V2 Lite is large; 4-bit is essential for consumer GPUs.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Prepare model for k-bit training (freezes weights, casts norms to float32)
model = prepare_model_for_kbit_training(model)

# 3. LoRA Configuration
#    DeepSeek is an MoE (Mixture of Experts). It is crucial to target 
#    all linear layers or specific projections to get good results.
peft_config = LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear" # Newer PEFT versions support this to target all layers automatically
)

# 4. Format Dataset
#    We need to convert your JSONL into the chat format the model expects.
def format_instruction(sample):
    # Depending on your JSONL structure, adjust keys
    # Assuming 'prompt' key from your shared code
    messages = [
        {"role": "user", "content": sample["prompt"]},
        # Ideally, your training data should have a "completion" or "response"
        # If your JSONL only has prompts, you can't do Supervised Fine-Tuning (SFT).
        # You need the target answer. I will assume a "completion" key exists.
        {"role": "assistant", "content": sample.get("completion", "")}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 5. Training Arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=2, # Low batch size for VRAM
    gradient_accumulation_steps=4, # Increase effective batch size
    learning_rate=2e-4,
    fp16=False,
    bf16=True,       # Use BF16 if on Ampere (A100/3090/4090)
    logging_steps=10,
    save_strategy="epoch",
    optim="paged_adamw_32bit", # Saves VRAM
    report_to="none" # Change to "wandb" if you want tracking
)

# 6. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    formatting_func=format_instruction,
    args=args,
    packing=False, # Set True if you want to pack multiple short examples
)

print("Starting training...")
trainer.train()

print(f"Saving adapter to {OUTPUT_DIR}/final_adapter")
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")