"""
plotter.py — Saves training vs. validation loss plots for each fine-tuned model.

A PNG file is written to ``{cfg.output_dir}/training_loss.png`` after every
successful training run so results are always co-located with checkpoints.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import PLOT_DPI, PLOT_HEIGHT, PLOT_WIDTH, ModelConfig
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def save_loss_plot(trainer, cfg: ModelConfig) -> str | None:
    """
    Extract loss history from *trainer* and save a PNG plot.

    Parameters
    ----------
    trainer:
        A completed SFTTrainer instance (``trainer.state.log_history`` must exist).
    cfg:
        ModelConfig for the model that was just trained.

    Returns
    -------
    str | None
        Path to the saved PNG, or None if the history was empty.
    """
    history = getattr(getattr(trainer, "state", None), "log_history", None)
    if not history:
        log.warning("[%s] No log history found — skipping plot.", cfg.key)
        return None

    df = pd.DataFrame(history)

    train_loss = _extract_metric(df, "loss")
    eval_loss  = _extract_metric(df, "eval_loss")

    if train_loss.empty:
        log.warning("[%s] No training loss data — skipping plot.", cfg.key)
        return None

    os.makedirs(cfg.output_dir, exist_ok=True)
    save_path = os.path.join(cfg.output_dir, "training_loss.png")

    fig, ax = plt.subplots(figsize=(PLOT_WIDTH, PLOT_HEIGHT))
    ax.plot(train_loss["step"], train_loss["loss"], label="Training Loss")

    if not eval_loss.empty:
        ax.plot(
            eval_loss["step"], eval_loss["eval_loss"],
            label="Validation Loss", color="red",
        )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — {cfg.key}")
    ax.legend()

    fig.savefig(save_path, dpi=PLOT_DPI)
    plt.close(fig)

    log.info("[%s] Loss plot saved → %s", cfg.key, save_path)
    return save_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_metric(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return rows where *column* is not NaN, keeping only [step, column]."""
    if column not in df.columns:
        return pd.DataFrame()
    return df[df[column].notna()][["step", column]].reset_index(drop=True)
