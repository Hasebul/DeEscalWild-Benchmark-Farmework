base_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/"
dataset_path = f"{base_path}qwen_finetune_data.jsonl"
output_dir = f"{base_path}falcon_3_3b"

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

# --- Configuration ---
MODEL_ID = "tiiuae/Falcon3-3B-Instruct"
NEW_MODEL_NAME = "Falcon3-3B-Instruct-Finetune"

# 1. Load Tokenizer & Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# Quantization config for 4-bit loading (saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
)

# 2. Prepare Dataset with Chat Template
# We use a standard dataset as an example. Replace with your own JSONL/HuggingFace dataset.
dataset = load_dataset("json", data_files="/content/drive/MyDrive/deescalation_project/Checkpoints/qwen_finetune_data.jsonl", split="train")

def format_chat_template(row):
    # Falcon 3 Instruct expects a specific chat format.
    # The dataset loaded from qwen_finetune_data.jsonl is expected to have a 'messages' key.

    conversations = row['messages']

    # Apply the tokenizer's chat template directly to the full conversation.
    # This converts the list of message dictionaries into a single string
    # formatted for the model (e.g., "<|user|>...<|assistant|>...").
    formatted_text = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False # Set to False for training if the last turn is assistant's response
    )
    return {"text": formatted_text}

dataset = dataset.map(format_chat_template)

# 3. LoRA Configuration (Adapter)
peft_config = LoraConfig(
    r=16,        # Rank
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # Falcon 3 uses standard Llama-style layer names, unlike Falcon 1/2
)

# Apply the PEFT adapters to the base model
model = get_peft_model(model, peft_config)

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./falcon3-finetune-results",
    num_train_epochs=1,
    per_device_train_batch_size=2, # Reduced from 4
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",      # Changed from paged_adamw_32bit to 8bit
    save_steps=50,
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,                # Use fp16 for mixed precision
    bf16=False,               # Use bf16 if on Ampere (A100/A10/3090) or newer
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none"          # Change to "wandb" to track training
)

# 5. Initialize Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    # dataset_text_field="text",
    # max_seq_length=1024,      # Adjust based on your VRAM
    # tokenizer=tokenizer,
    args=training_args,
    # packing=False,
)

# 6. Train and Save
print("Starting training...")
trainer.train()

print("Saving model...")
trainer.model.save_pretrained(NEW_MODEL_NAME)
tokenizer.save_pretrained(NEW_MODEL_NAME)