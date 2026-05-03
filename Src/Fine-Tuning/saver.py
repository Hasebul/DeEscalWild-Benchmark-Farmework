"""
saver.py — Saves the fine-tuned model and tokenizer to disk.

Writes to ``{cfg.output_dir}/checkpoint-final`` so every model's final
checkpoint is always co-located with its loss plot and training config.
"""

from __future__ import annotations

import os

from config import ModelConfig
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def save_model(model, tokenizer, cfg: ModelConfig) -> str:
    """
    Save *model* and *tokenizer* to ``{cfg.output_dir}/checkpoint-final``.

    Parameters
    ----------
    model:     Trained (PEFT-wrapped) model.
    tokenizer: Configured tokenizer.
    cfg:       ModelConfig for the model that was just trained.

    Returns
    -------
    str
        Path to the saved checkpoint directory.
    """
    checkpoint_dir = os.path.join(cfg.output_dir, "checkpoint-final")
    os.makedirs(checkpoint_dir, exist_ok=True)

    log.info("[%s] Saving model → %s", cfg.key, checkpoint_dir)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    log.info("[%s] Checkpoint saved successfully.", cfg.key)
    return checkpoint_dir
