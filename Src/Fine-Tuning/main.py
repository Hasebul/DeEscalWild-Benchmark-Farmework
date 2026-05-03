"""
main.py — Entry point for the multi-model fine-tuning pipeline.

Each model is trained through the same five-stage pipeline:
  1. LOAD      — model + tokenizer via Unsloth or HF/PEFT
  2. DATA      — load JSONL, apply chat template, train/eval split
  3. TRAIN     — build SFTTrainer and call .train()
  4. SAVE      — write checkpoint-final to Google Drive
  5. REPORT    — save loss plot + append row to run_log.csv

Usage
-----
    # Train a single model
    python main.py --model qwen
    python main.py --model llama
    python main.py --model gemma
    python main.py --model falcon
    python main.py --model granite
    python main.py --model ministral
    python main.py --model phi4

    # Train all 7 models sequentially (full sweep)
    python main.py --all

    # Dry run — validate configs without training
    python main.py --all --dry-run
    python main.py --model qwen --dry-run

Available model keys: qwen, llama, gemma, falcon, granite, ministral, phi4
"""

from __future__ import annotations

import argparse
import sys
import time

from config import MODEL_REGISTRY, ModelConfig
from data_utils import get_train_eval, load_and_prepare
from logger import get_logger
from model_loader import load_model_and_tokenizer
from plotter import save_loss_plot
from run_log import append_run
from saver import save_model
from trainer_builder import build_trainer

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune one or all 7 models for the de-escalation project.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        metavar="MODEL",
        help=(
            "Key of the model to train.\n"
            f"Choices: {', '.join(MODEL_REGISTRY.keys())}"
        ),
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all 7 models sequentially.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and dataset paths without training.",
    )

    return parser.parse_args()


# ── Pipeline ───────────────────────────────────────────────────────────────────

def run_pipeline(cfg: ModelConfig, dry_run: bool = False) -> bool:
    """
    Execute the full fine-tuning pipeline for one model.

    Returns True on success, False on failure.
    """
    divider = "═" * 60
    log.info("%s", divider)
    log.info("  MODEL : %s  (%s)", cfg.key.upper(), cfg.hf_model_id)
    log.info("%s", divider)

    if dry_run:
        log.info("[%s] DRY RUN — skipping training.", cfg.key)
        _validate_config(cfg)
        return True

    start_time = time.time()

    try:
        # ── Stage 1: Load model & tokenizer ───────────────────────────────
        log.info("[%s] Stage 1/5 — Loading model …", cfg.key)
        model, tokenizer = load_model_and_tokenizer(cfg)

        # ── Stage 2: Prepare data ─────────────────────────────────────────
        log.info("[%s] Stage 2/5 — Preparing dataset …", cfg.key)
        dataset = load_and_prepare(cfg, tokenizer)
        train_dataset, eval_dataset = get_train_eval(dataset)

        # ── Stage 3: Train ────────────────────────────────────────────────
        log.info("[%s] Stage 3/5 — Training …", cfg.key)
        trainer = build_trainer(
            cfg, model, tokenizer, train_dataset, eval_dataset
        )
        trainer.train()

        duration_sec = time.time() - start_time
        log.info(
            "[%s] Training complete in %.1f min.",
            cfg.key, duration_sec / 60,
        )

        # ── Stage 4: Save checkpoint ──────────────────────────────────────
        log.info("[%s] Stage 4/5 — Saving checkpoint …", cfg.key)
        checkpoint_dir = save_model(model, tokenizer, cfg)

        # ── Stage 5: Report ───────────────────────────────────────────────
        log.info("[%s] Stage 5/5 — Saving plot and run log …", cfg.key)
        save_loss_plot(trainer, cfg)
        append_run(
            cfg           = cfg,
            trainer       = trainer,
            train_samples = len(train_dataset),
            eval_samples  = len(eval_dataset) if eval_dataset else 0,
            duration_sec  = duration_sec,
            checkpoint_dir = checkpoint_dir,
            status        = "success",
        )

        log.info("[%s] ✅ Pipeline complete.", cfg.key)
        return True

    except KeyboardInterrupt:
        log.warning("[%s] Interrupted by user.", cfg.key)
        sys.exit(0)

    except Exception as exc:
        log.error("[%s] ❌ Pipeline failed: %s", cfg.key, exc, exc_info=True)
        return False


# ── Validation (dry-run) ───────────────────────────────────────────────────────

def _validate_config(cfg: ModelConfig) -> None:
    import os
    log.info("[%s] Model ID     : %s", cfg.key, cfg.hf_model_id)
    log.info("[%s] Dataset path : %s", cfg.key, cfg.dataset_path)
    log.info("[%s] Output dir   : %s", cfg.key, cfg.output_dir)
    log.info("[%s] Chat template: %s", cfg.key, cfg.chat_template or "native")
    log.info("[%s] Message key  : %s", cfg.key, cfg.message_key)
    log.info("[%s] Eval split   : %s", cfg.key, cfg.eval_split)

    if not os.path.exists(cfg.dataset_path):
        log.warning("[%s] ⚠ Dataset not found: %s", cfg.key, cfg.dataset_path)
    else:
        log.info("[%s] ✅ Dataset exists.", cfg.key)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    models_to_run: list[ModelConfig] = (
        list(MODEL_REGISTRY.values())
        if args.all
        else [MODEL_REGISTRY[args.model]]
    )

    log.info(
        "Pipeline starting — %d model(s) to train: %s",
        len(models_to_run),
        ", ".join(c.key for c in models_to_run),
    )

    results: dict[str, bool] = {}
    for cfg in models_to_run:
        results[cfg.key] = run_pipeline(cfg, dry_run=args.dry_run)

    # ── Final summary ──────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("  FINAL SUMMARY")
    log.info("═" * 60)
    for key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        log.info("  %-12s  %s", key, status)
    log.info("═" * 60)

    failed = [k for k, ok in results.items() if not ok]
    if failed:
        log.error("%d model(s) failed: %s", len(failed), ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
