"""
trainer_builder.py — Constructs an SFTTrainer instance for any ModelConfig.

Two trainer backends are supported:
  * SFTTrainer + SFTConfig  (Unsloth models)
  * SFTTrainer + TrainingArguments  (Falcon / HF-PEFT models)

The correct backend is chosen automatically from ``cfg.use_unsloth``.
"""

from __future__ import annotations

import torch
from datasets import Dataset

from config import (
    EVAL_STEPS,
    GRADIENT_ACCUMULATION,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_SEQ_LENGTH,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    ModelConfig,
)
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def build_trainer(
    cfg:            ModelConfig,
    model,
    tokenizer,
    train_dataset:  Dataset,
    eval_dataset:   Dataset | None,
):
    """
    Build and return a configured SFTTrainer for *cfg*.

    Parameters
    ----------
    cfg:            ModelConfig for the target model.
    model:          PEFT-wrapped model.
    tokenizer:      Configured tokenizer.
    train_dataset:  Formatted training Dataset.
    eval_dataset:   Formatted evaluation Dataset, or None.

    Returns
    -------
    SFTTrainer — ready to call .train() on.
    """
    if cfg.use_unsloth:
        return _build_unsloth_trainer(
            cfg, model, tokenizer, train_dataset, eval_dataset
        )
    return _build_hf_trainer(
        cfg, model, tokenizer, train_dataset, eval_dataset
    )


# ── Unsloth trainer ───────────────────────────────────────────────────────────

def _build_unsloth_trainer(cfg, model, tokenizer, train_dataset, eval_dataset):
    from trl import SFTTrainer, SFTConfig

    sft_args = SFTConfig(
        output_dir                 = cfg.output_dir,
        num_train_epochs           = TRAIN_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        learning_rate              = LEARNING_RATE,
        fp16                       = not torch.cuda.is_bf16_supported(),
        bf16                       = torch.cuda.is_bf16_supported(),
        logging_steps              = LOGGING_STEPS,
        eval_strategy              = "steps" if eval_dataset else "no",
        eval_steps                 = EVAL_STEPS if eval_dataset else None,
        dataset_text_field         = "text",
        max_seq_length             = cfg.max_seq_length,
    )

    trainer_kwargs = dict(
        model          = model,
        tokenizer      = tokenizer,
        train_dataset  = train_dataset,
        args           = sft_args,
    )

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    log.info("[%s] Building SFTTrainer (Unsloth) …", cfg.key)
    return SFTTrainer(**trainer_kwargs)


# ── HF / PEFT trainer (Falcon) ────────────────────────────────────────────────

def _build_hf_trainer(cfg, model, tokenizer, train_dataset, eval_dataset):
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir                  = cfg.output_dir,
        num_train_epochs            = TRAIN_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH_SIZE,
        gradient_accumulation_steps = GRADIENT_ACCUMULATION,
        learning_rate               = LEARNING_RATE,
        fp16                        = True,
        bf16                        = False,
        optim                       = "paged_adamw_8bit",
        logging_steps               = LOGGING_STEPS,
        save_steps                  = 50,
        weight_decay                = 0.001,
        max_grad_norm               = 0.3,
        warmup_ratio                = 0.03,
        group_by_length             = True,
        lr_scheduler_type           = "constant",
        report_to                   = "none",
    )

    trainer_kwargs = dict(
        model         = model,
        train_dataset = train_dataset,
        args          = training_args,
    )

    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    log.info("[%s] Building SFTTrainer (HF/PEFT) …", cfg.key)
    return SFTTrainer(**trainer_kwargs)
