"""
config.py — Single source of truth for the multi-model fine-tuning pipeline.

All paths, model IDs, LoRA ranks, dataset files, and training hyper-parameters
live here.  To add a new model, insert one entry into MODEL_REGISTRY; no other
file needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# ── Base paths (Google Drive) ──────────────────────────────────────────────────
DRIVE_BASE    = "/content/drive/MyDrive/deescalation_project/Checkpoints"
DATASET_CHATML = f"{DRIVE_BASE}/qwen_finetune_data.jsonl"          # ChatML format
DATASET_PREPROCESSED = f"{DRIVE_BASE}/preprocessed_train_data.jsonl"  # Gemma / Granite

# ── Shared LoRA defaults ───────────────────────────────────────────────────────
LORA_R               = 16
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.0
LORA_TARGET_MODULES  = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Shared training defaults ───────────────────────────────────────────────────
TRAIN_EPOCHS               = 2
TRAIN_BATCH_SIZE           = 2
GRADIENT_ACCUMULATION      = 4
LEARNING_RATE              = 2e-4
LOGGING_STEPS              = 1
EVAL_STEPS                 = 10
MAX_SEQ_LENGTH             = 2048
EVAL_SPLIT_SIZE            = 0.1   # 10% for eval
RANDOM_STATE               = 3407

# ── Plot settings ──────────────────────────────────────────────────────────────
PLOT_DPI    = 300
PLOT_WIDTH  = 10
PLOT_HEIGHT = 5


# ── Per-model registry ─────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    All information needed to fine-tune one model.

    Fields
    ------
    key             : Short identifier used in --model CLI flag and output paths.
    hf_model_id     : Hugging Face model ID (or Unsloth shorthand).
    chat_template   : Template name passed to unsloth.get_chat_template.
                      Set to None for models whose tokenizer is pre-configured
                      (e.g. Granite) or for non-Unsloth models (Falcon).
    dataset_path    : Path to the JSONL training file.
    output_dir      : Directory where checkpoints + plots are saved.
    use_unsloth     : True = load via FastLanguageModel; False = load via HF transformers.
    load_in_4bit    : Enable bitsandbytes 4-bit quantisation.
    max_seq_length  : Per-model sequence length cap.
    eval_split      : Fraction of data reserved for validation.
    role_mapping    : Dict forwarded to get_chat_template (only used when
                      chat_template is not None).
    message_key     : Key inside each JSONL record that holds the conversation
                      list ("messages" for most; "conversations" for Ministral).
    extra_kwargs    : Any additional kwargs forwarded to FastLanguageModel.from_pretrained.
    """
    key:            str
    hf_model_id:    str
    chat_template:  str | None
    dataset_path:   str
    output_dir:     str
    use_unsloth:    bool   = True
    load_in_4bit:   bool   = True
    max_seq_length: int    = MAX_SEQ_LENGTH
    eval_split:     float  = EVAL_SPLIT_SIZE
    role_mapping:   dict   = field(default_factory=lambda: {
        "role": "role", "content": "content",
        "user": "user", "assistant": "assistant",
    })
    message_key:    str    = "messages"
    extra_kwargs:   dict   = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelConfig] = {

    "qwen": ModelConfig(
        key           = "qwen",
        hf_model_id   = "unsloth/Qwen2.5-3B-Instruct",
        chat_template = "chatml",
        dataset_path  = DATASET_CHATML,
        output_dir    = f"{DRIVE_BASE}/qwen_25",
    ),

    "llama": ModelConfig(
        key           = "llama",
        hf_model_id   = "unsloth/Llama-3.2-3B-Instruct",
        chat_template = "llama-3",
        dataset_path  = DATASET_CHATML,
        output_dir    = f"{DRIVE_BASE}/llama_32",
    ),

    "gemma": ModelConfig(
        key           = "gemma",
        hf_model_id   = "unsloth/gemma-2-2b-it-bnb-4bit",
        chat_template = "gemma",
        dataset_path  = DATASET_PREPROCESSED,
        output_dir    = f"{DRIVE_BASE}/gemma_2",
        eval_split    = 0.03,
        role_mapping  = {
            "role": "role", "content": "content",
            "user": "user", "assistant": "model",
        },
    ),

    "falcon": ModelConfig(
        key           = "falcon",
        hf_model_id   = "tiiuae/Falcon3-3B-Instruct",
        chat_template = None,          # uses tokenizer's native template
        dataset_path  = DATASET_CHATML,
        output_dir    = f"{DRIVE_BASE}/falcon_3_3b",
        use_unsloth   = False,         # loaded via HF transformers + PEFT
    ),

    "granite": ModelConfig(
        key           = "granite",
        hf_model_id   = "ibm-granite/granite-3.0-2b-instruct",
        chat_template = None,          # tokenizer is pre-configured
        dataset_path  = DATASET_PREPROCESSED,
        output_dir    = f"{DRIVE_BASE}/granite_3_0_2b",
        eval_split    = 0.03,
        max_seq_length = 4096,
    ),

    "ministral": ModelConfig(
        key           = "ministral",
        hf_model_id   = "unsloth/Ministral-3-3B-Instruct-2512-GGUF",
        chat_template = "mistral",
        dataset_path  = DATASET_CHATML,
        output_dir    = f"{DRIVE_BASE}/ministral3",
        message_key   = "conversations",
        role_mapping  = {
            "role": "from", "content": "value",
            "user": "human", "assistant": "gpt",
        },
        extra_kwargs  = {"use_gradient_checkpointing": "unsloth"},
    ),

    "phi4": ModelConfig(
        key           = "phi4",
        hf_model_id   = "unsloth/Phi-4-mini-instruct-bnb-4bit",
        chat_template = "phi-4",
        dataset_path  = DATASET_CHATML,
        output_dir    = f"{DRIVE_BASE}/phi4",
        max_seq_length = 4096,
        eval_split    = 0.0,           # no eval split for Phi-4
    ),
}
