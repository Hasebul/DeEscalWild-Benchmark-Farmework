base_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/"
dataset_path = f"{base_path}qwen_finetune_data.jsonl"
output_dir = f"{base_path}ministral3"


# 1. Install Unsloth and dependencies (if not already installed)
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

from unsloth import FastLanguageModel # Corrected from FastVisionModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

# 2. Load the Model
# We use the 4-bit quantized version to fit in memory (approx 5GB VRAM)
model, tokenizer = FastLanguageModel.from_pretrained( # Corrected from FastVisionModel
    model_name = "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

# 3. Add LoRA Adapters
# We target language layers only for text fine-tuning
model = FastLanguageModel.get_peft_model( # Corrected from FastVisionModel
    model,
    # finetune_vision_layers     = False, # Not applicable for language model
    finetune_language_layers   = True,  # Set to True for text
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16,           # LoRA rank
    lora_alpha = 16,  # LoRA alpha
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Prepare Data (ShareGPT / ChatML format)
# This example uses a standard conversation format.
# Ministral uses standard Mistral chat templates.

from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "mistral", # Ministral uses the standard Mistral template
    mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"},
)

def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts }

# Example: Loading a ShareGPT style dataset
# dataset = load_dataset("philschmid/guanaco-sharegpt-style", split = "train")
# dataset = standardize_sharegpt(dataset)
# dataset = dataset.map(formatting_prompts_func, batched = True)
# 5. Training Configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    dataset_num_proc = 2,
    packing = False, # Can set to True for faster training if sequences are short
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Adjust for your full run (e.g., 300-1000 steps)
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)


# 6. Train
print("Starting Training...")
trainer_stats = trainer.train()

# 7. Save the model
model.save_pretrained("ministral-3b-finetuned")
tokenizer.save_pretrained("ministral-3b-finetuned")