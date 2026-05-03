"""
data_utils.py — Dataset loading, formatting, and train/eval splitting.

Each model may use a different chat template and conversation key.
All logic is driven by ``ModelConfig`` so no model-specific branches
are needed in the calling code.
"""

from __future__ import annotations

from datasets import Dataset, DatasetDict, load_dataset

from config import ModelConfig
from logger import get_logger

log = get_logger(__name__)


# ── Public API ────────────────────────────────────────────────────────────────

def load_and_prepare(cfg: ModelConfig, tokenizer) -> DatasetDict | Dataset:
    """
    Load the JSONL file, apply the chat template, and split into
    train / eval sets.

    Parameters
    ----------
    cfg:
        The ``ModelConfig`` for the model being trained.
    tokenizer:
        The (already configured) tokenizer for this model.

    Returns
    -------
    DatasetDict with "train" and optionally "test" keys,
    or a plain Dataset when ``cfg.eval_split == 0``.
    """
    log.info("[%s] Loading dataset from %s", cfg.key, cfg.dataset_path)
    raw: Dataset = load_dataset("json", data_files=cfg.dataset_path, split="train")

    log.info("[%s] Applying chat template (key='%s') …", cfg.key, cfg.message_key)
    formatted = raw.map(
        _make_formatting_func(tokenizer, cfg.message_key),
        batched=True,
        desc=f"Formatting [{cfg.key}]",
    )

    if cfg.eval_split > 0:
        split = formatted.train_test_split(test_size=cfg.eval_split)
        log.info(
            "[%s] Split → train=%d  eval=%d",
            cfg.key, len(split["train"]), len(split["test"]),
        )
        return split

    log.info("[%s] No eval split — using full dataset for training.", cfg.key)
    return formatted


def get_train_eval(
    dataset: DatasetDict | Dataset,
) -> tuple[Dataset, Dataset | None]:
    """
    Unpack a DatasetDict or plain Dataset into (train, eval) tuple.
    eval is None when no split was created.
    """
    if isinstance(dataset, DatasetDict):
        return dataset["train"], dataset["test"]
    return dataset, None


# ── Private helpers ───────────────────────────────────────────────────────────

def _make_formatting_func(tokenizer, message_key: str):
    """
    Return a batched mapping function that applies the tokenizer's chat template
    to each conversation in the dataset.

    Includes per-item validation so a malformed conversation never stops
    an entire batch.
    """
    def formatting_func(examples: dict) -> dict:
        conversations = examples[message_key]
        texts: list[str] = []

        for i, convo in enumerate(conversations):
            convo = _coerce_to_list(convo, i)
            if convo is None:
                texts.append("")
                continue

            try:
                text = tokenizer.apply_chat_template(
                    convo,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
            except Exception as exc:
                log.warning(
                    "Chat template failed for item %d: %s — skipping.", i, exc
                )
                texts.append("")

        return {"text": texts}

    return formatting_func


def _coerce_to_list(convo, index: int) -> list | None:
    """
    Ensure *convo* is a list of dicts.  Returns None and logs a warning
    if the structure is unrecoverable.
    """
    if isinstance(convo, dict):
        return [convo]
    if isinstance(convo, list):
        if all(isinstance(item, dict) for item in convo):
            return convo
        log.warning("Item %d contains non-dict elements — skipping.", index)
        return None
    log.warning("Item %d has unexpected type %s — skipping.", index, type(convo))
    return None
