# Multi-Model Fine-Tuning Pipeline

A clean, modular pipeline for fine-tuning 7 language models on the de-escalation dataset using a single, unified codebase.  Every model runs through the same five-stage pipeline; adding a new model requires only one entry in `config.py`.

---

## Project Structure

```
finetune_pipeline/
│
├── config.py           # All model IDs, paths, LoRA ranks, hyper-params — edit here only
├── logger.py           # Shared timestamped logger
├── data_utils.py       # Dataset loading, chat-template formatting, train/eval split
├── model_loader.py     # Loads model + tokenizer via Unsloth or HF/PEFT
├── trainer_builder.py  # Builds SFTTrainer (SFTConfig or TrainingArguments)
├── plotter.py          # Saves training vs. validation loss PNG per model
├── saver.py            # Saves checkpoint-final to output_dir
├── run_log.py          # Appends one row per run to run_log.csv on Google Drive
├── main.py             # CLI entry point — orchestrates all 7 models
└── README.md
```

---

## Supported Models

| Key         | Model ID                                   | Backend       | Dataset            |
|-------------|--------------------------------------------|---------------|--------------------|
| `qwen`      | `unsloth/Qwen2.5-3B-Instruct`              | Unsloth       | qwen_finetune_data |
| `llama`     | `unsloth/Llama-3.2-3B-Instruct`            | Unsloth       | qwen_finetune_data |
| `gemma`     | `unsloth/gemma-2-2b-it-bnb-4bit`           | Unsloth       | preprocessed_train |
| `falcon`    | `tiiuae/Falcon3-3B-Instruct`               | HF + PEFT     | qwen_finetune_data |
| `granite`   | `ibm-granite/granite-3.0-2b-instruct`      | Unsloth       | preprocessed_train |
| `ministral` | `unsloth/Ministral-3-3B-Instruct-2512-GGUF`| Unsloth       | qwen_finetune_data |
| `phi4`      | `unsloth/Phi-4-mini-instruct-bnb-4bit`     | Unsloth       | qwen_finetune_data |

---

## Installation

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
pip install datasets transformers pandas matplotlib
```

---

## Usage

### Train a single model
```bash
python main.py --model qwen
python main.py --model llama
python main.py --model gemma
python main.py --model falcon
python main.py --model granite
python main.py --model ministral
python main.py --model phi4
```

### Train all 7 models sequentially
```bash
python main.py --all
```

### Dry run — validate configs and dataset paths without training
```bash
python main.py --model qwen --dry-run
python main.py --all --dry-run
```

---

## Pipeline Stages (per model)

| Stage | Module            | What it does                                              |
|-------|-------------------|-----------------------------------------------------------|
| 1     | `model_loader.py` | Load base model + tokenizer; apply LoRA adapters          |
| 2     | `data_utils.py`   | Load JSONL, apply chat template, split train/eval         |
| 3     | `trainer_builder.py` | Build SFTTrainer and run `.train()`                    |
| 4     | `saver.py`        | Save `checkpoint-final` to `cfg.output_dir`               |
| 5     | `plotter.py` + `run_log.py` | Save loss PNG + append row to `run_log.csv`   |

---

## Configuration

Everything is in `config.py`. To tune training, change the shared constants:

| Constant                 | Default  | Description                               |
|--------------------------|----------|-------------------------------------------|
| `TRAIN_EPOCHS`           | `2`      | Number of full training passes            |
| `TRAIN_BATCH_SIZE`       | `2`      | Per-device batch size                     |
| `GRADIENT_ACCUMULATION`  | `4`      | Effective batch = batch_size × accum      |
| `LEARNING_RATE`          | `2e-4`   | AdamW learning rate                       |
| `LORA_R`                 | `16`     | LoRA rank                                 |
| `LORA_ALPHA`             | `16`     | LoRA alpha                                |
| `MAX_SEQ_LENGTH`         | `2048`   | Sequence length cap (per-model overrides) |
| `EVAL_SPLIT_SIZE`        | `0.1`    | Fraction of data reserved for eval        |

To add a new model, insert one `ModelConfig` entry into `MODEL_REGISTRY` in `config.py`. No other file needs to change.

---

## Outputs (per model)

| File                                         | Description                              |
|----------------------------------------------|------------------------------------------|
| `{output_dir}/checkpoint-final/`             | Final model weights + tokenizer          |
| `{output_dir}/training_loss.png`             | Training vs. validation loss plot        |
| `{DRIVE_BASE}/run_log.csv`                   | Global CSV log of all training runs      |

### `run_log.csv` columns

```
timestamp, model_key, hf_model_id, epochs, train_samples, eval_samples,
final_train_loss, final_eval_loss, duration_min, checkpoint_dir, status
```

---

## Error Handling

- A failure in one model does **not** stop the rest when using `--all`. Each model is wrapped independently — errors are logged and the sweep continues.
- A final summary table is printed after all runs showing ✅ SUCCESS or ❌ FAILED per model.
- Exit code is non-zero if any model failed, enabling CI/scripted use.
- Keyboard interrupt exits cleanly without corrupting saved checkpoints.
