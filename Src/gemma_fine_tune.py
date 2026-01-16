base_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/"
dataset_path = f"{base_path}qwen_finetune_data.jsonl"
output_dir = f"{base_path}gemma_2"


from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import get_chat_template

# --- 1. CONFIGURATION ---
max_seq_length = 4096 # Gemma 2 supports 8k context
dtype = None # Auto-detect
load_in_4bit = True

# --- 2. LOAD MODEL ---
# Using the 4-bit quantized version of Gemma 2 2B Instruct
model_name = "unsloth/gemma-2-2b-it-bnb-4bit"

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


# Setup the Gemma 2 specific chat template
# Gemma uses <start_of_turn>user<end_of_turn> structure
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma", # <--- Specific for Gemma
    mapping = {"role": "role", "content": "content", "user": "user", "assistant": "model"}, # Note: 'model' not 'assistant' sometimes, but Unsloth maps it
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = []

    for i, convo in enumerate(convos):
        # Ensure convo is a list of dictionaries. If it's a single dict, wrap it.
        if isinstance(convo, dict):
            convo = [convo]
        elif not isinstance(convo, list):
            # Handle cases where convo might be something unexpected (e.g., a string, None)
            print(f"Warning: Unexpected conversation format at index {i}. Skipping: {convo}")
            texts.append("")
            continue

        # Ensure all items within the conversation list are dictionaries
        if not all(isinstance(item, dict) for item in convo):
            print(f"Warning: Conversation at index {i} contains non-dictionary items. Skipping: {convo}")
            texts.append("")
            continue

        try:
            text = tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False)
            texts.append(text)
        except Exception as e:
            print(f"Error applying chat template for conversation at index {i}: {convo}")
            print(f"Error details: {e}")
            texts.append("") # Append empty string to avoid stopping the batch
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
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs_gemma2_2b",
    ),
)

# --- 6. TRAIN ---
print("Starting Training...")
trainer_stats = trainer.train()

# --- 7. SAVE ---
print("Saving Model...")
model.save_pretrained("outputs_gemma2_2b/checkpoint-final")
tokenizer.save_pretrained("outputs_gemma2_2b/checkpoint-final")