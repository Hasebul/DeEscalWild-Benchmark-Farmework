"""
model_loader.py — Loads models and tokenizers for all 7 fine-tuning targets.

Two loading paths are supported:
  * Unsloth  (use_unsloth=True)  — FastLanguageModel + get_peft_model
  * HF/PEFT  (use_unsloth=False) — AutoModelForCausalLM + LoraConfig + get_peft_model
             Used only for Falcon, which is not yet in the Unsloth model zoo.

The caller receives a ``(model, tokenizer)`` tuple regardless of which
path was used.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    RANDOM_STATE,
    ModelConfig,
)
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(cfg: ModelConfig) -> tuple:
    """
    Load and prepare the model + tokenizer described by *cfg*.

    Returns
    -------
    model     : PEFT-wrapped model ready for SFTTrainer.
    tokenizer : Tokenizer with chat template applied (when relevant).
    """
    if cfg.use_unsloth:
        return _load_unsloth(cfg)
    return _load_hf_peft(cfg)


# ── Unsloth path ──────────────────────────────────────────────────────────────

def _load_unsloth(cfg: ModelConfig) -> tuple:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    log.info("[%s] Loading via Unsloth: %s", cfg.key, cfg.hf_model_id)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.hf_model_id,
        max_seq_length = cfg.max_seq_length,
        dtype          = None,          # auto-detect float16 / bfloat16
        load_in_4bit   = cfg.load_in_4bit,
        **cfg.extra_kwargs,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                        = LORA_R,
        target_modules           = LORA_TARGET_MODULES,
        lora_alpha               = LORA_ALPHA,
        lora_dropout             = LORA_DROPOUT,
        bias                     = "none",
        use_gradient_checkpointing = "unsloth",
        random_state             = RANDOM_STATE,
    )

    # Apply chat template only when one is specified for this model
    if cfg.chat_template:
        log.info(
            "[%s] Applying chat template: %s", cfg.key, cfg.chat_template
        )
        tokenizer = get_chat_template(
            tokenizer,
            chat_template = cfg.chat_template,
            mapping       = cfg.role_mapping,
        )

    return model, tokenizer


# ── HF / PEFT path (Falcon) ───────────────────────────────────────────────────

def _load_hf_peft(cfg: ModelConfig) -> tuple:
    from peft import LoraConfig, get_peft_model

    log.info("[%s] Loading via HF + PEFT: %s", cfg.key, cfg.hf_model_id)

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit             = cfg.load_in_4bit,
        bnb_4bit_quant_type      = "nf4",
        bnb_4bit_compute_dtype   = torch.float16,
        bnb_4bit_use_double_quant = True,
    )

    # Flash Attention 2 on Ampere+ GPUs, eager attention otherwise
    attn_impl = (
        "flash_attention_2"
        if torch.cuda.get_device_capability()[0] >= 8
        else "eager"
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model_id,
        quantization_config  = bnb_config,
        device_map           = "auto",
        attn_implementation  = attn_impl,
    )

    peft_config = LoraConfig(
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        lora_dropout   = 0.05,        # slight dropout for non-Unsloth path
        bias           = "none",
        task_type      = "CAUSAL_LM",
        target_modules = LORA_TARGET_MODULES,
    )

    model = get_peft_model(model, peft_config)
    log.info("[%s] PEFT model ready — trainable params: %s", cfg.key, _trainable_params(model))
    return model, tokenizer


# ── Helpers ───────────────────────────────────────────────────────────────────

def _trainable_params(model) -> str:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
