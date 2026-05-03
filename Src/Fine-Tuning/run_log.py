"""
run_log.py — Appends a one-row summary to a CSV run log after each training job.

This provides a persistent record of every fine-tuning run so you can
compare models, training durations, and final losses in one place.

Log location: {DRIVE_BASE}/run_log.csv
"""

from __future__ import annotations

import csv
import os
import time
from datetime import datetime

from config import DRIVE_BASE, ModelConfig
from logger import get_logger

log = get_logger(__name__)

LOG_PATH = os.path.join(DRIVE_BASE, "run_log.csv")

_HEADER = [
    "timestamp",
    "model_key",
    "hf_model_id",
    "epochs",
    "train_samples",
    "eval_samples",
    "final_train_loss",
    "final_eval_loss",
    "duration_min",
    "checkpoint_dir",
    "status",
]


# ── Public API ────────────────────────────────────────────────────────────────

def append_run(
    cfg:              ModelConfig,
    trainer,
    train_samples:    int,
    eval_samples:     int,
    duration_sec:     float,
    checkpoint_dir:   str,
    status:           str = "success",
) -> None:
    """
    Append one row to the global run log CSV.

    Parameters
    ----------
    cfg:             ModelConfig of the trained model.
    trainer:         Completed SFTTrainer (used to extract final losses).
    train_samples:   Number of training examples used.
    eval_samples:    Number of evaluation examples used (0 if no eval).
    duration_sec:    Wall-clock training time in seconds.
    checkpoint_dir:  Path where the model was saved.
    status:          "success" or "failed".
    """
    final_train_loss, final_eval_loss = _extract_final_losses(trainer)

    row = {
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_key":        cfg.key,
        "hf_model_id":      cfg.hf_model_id,
        "epochs":           str(trainer.args.num_train_epochs),
        "train_samples":    train_samples,
        "eval_samples":     eval_samples,
        "final_train_loss": f"{final_train_loss:.4f}" if final_train_loss else "N/A",
        "final_eval_loss":  f"{final_eval_loss:.4f}"  if final_eval_loss  else "N/A",
        "duration_min":     f"{duration_sec / 60:.1f}",
        "checkpoint_dir":   checkpoint_dir,
        "status":           status,
    }

    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    file_is_new = not os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_HEADER)
        if file_is_new:
            writer.writeheader()
        writer.writerow(row)

    log.info("[%s] Run logged → %s", cfg.key, LOG_PATH)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_final_losses(trainer) -> tuple[float | None, float | None]:
    """Return the last recorded train loss and eval loss from trainer history."""
    import pandas as pd

    history = getattr(getattr(trainer, "state", None), "log_history", [])
    if not history:
        return None, None

    df = pd.DataFrame(history)

    train_loss = None
    eval_loss  = None

    if "loss" in df.columns:
        series = df["loss"].dropna()
        if not series.empty:
            train_loss = float(series.iloc[-1])

    if "eval_loss" in df.columns:
        series = df["eval_loss"].dropna()
        if not series.empty:
            eval_loss = float(series.iloc[-1])

    return train_loss, eval_loss
