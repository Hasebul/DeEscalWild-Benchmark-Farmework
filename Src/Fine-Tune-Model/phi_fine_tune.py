base_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/"
dataset_path = f"{base_path}qwen_finetune_data.jsonl"
output_dir = f"{base_path}phi4"

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template

# --- 1. CONFIGURATION ---
max_seq_length = 4096 # Phi-4 supports very long context (up to 128k), but start with 4k for training
dtype = None # Auto-detect
load_in_4bit = True

# --- 2. LOAD MODEL ---
# We use the specific 4-bit version of Phi-4-mini to save download time and fix bugs
model_name = "unsloth/Phi-4-mini-instruct-bnb-4bit"

print(f"Loading {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# --- 3. ADD LoRA ADAPTERS ---
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

# --- 4. PREPARE DATA ---
# Load your existing JSONL file
# dataset = load_dataset("json", data_files="qwen_finetune_data.jsonl", split="train")

# Setup the Phi-4 specific chat template
# Unsloth has a built-in "phi-4" template that handles the <|im_start|> tags correctly
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "phi-4",
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

# --- 4. PREPARE DATA ---
# Load your existing JSONL file
dataset = load_dataset("json", data_files=dataset_path, split="train")
from datasets import Dataset
dataset = Dataset.from_list(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True)

# --- 5. TRAINER CONFIGURATION ---
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    formatting_func = formatting_prompts_func,

    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # 1 Full pass is usually enough for simulations
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_phi4_mini",
        torch_compile = False,
    ),
)

# --- 6. TRAIN ---
print("Starting Training...")
trainer_stats = trainer.train()

# --- 7. SAVE ---
print("Saving Model...")
model.save_pretrained("outputs_phi4_mini/checkpoint-final")
tokenizer.save_pretrained("outputs_phi4_mini/checkpoint-final")