# Evaluation

This folder contains evaluation scripts for comparing base and fine-tuned language models on police de-escalation dialogue generation.

The evaluation includes two approaches:

1. **LLM-as-Judge Evaluation**
   - Evaluates realism of generated civilian/suspect responses.
   - Evaluates de-escalation quality of the full conversation.
   - Uses Gemini 2.5 Flash as the judge model by default.

2. **Metric-Based Evaluation**
   - Compares generated responses with real reference responses.
   - Computes ROUGE-L, BLEU-4, METEOR, and BERTScore F1.
   - Supports base and fine-tuned models.

---

## Folder Structure

```text
Evaluation/
├── evaluate_llm_as_judge.py
├── metric_based_eval_all_models.py
├── README.md
└── requirements.txt
```

---

## Supported Models

The scripts are designed to evaluate the following models:

| Model Family | Base Model | Fine-Tuned Model |
|---|---|---|
| Qwen | qwen_base | qwen_finetuned |
| Llama | llama_base | llama_finetuned |
| Phi | phi_base | phi_finetuned |
| Mistral | mistral_base | mistral_finetuned |
| Granite | granite_base | granite_finetuned |
| Gemma | gemma_base | gemma_finetuned |
| Falcon | falcon_base | falcon_finetuned |
| Gemini | gemini_2_5_flash | N/A |

Before running all local models, update the model paths inside:

```text
metric_based_eval_all_models.py
```

Replace all `CHANGE_ME` paths with your actual Hugging Face model names or fine-tuned checkpoint paths.

---

## Installation

Create and activate a Python environment:

```bash
python -m venv eval_env
source eval_env/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

For Google Colab, run:

```bash
pip install -r requirements.txt
```

---

## Gemini API Key

The LLM-as-judge script uses Gemini 2.5 Flash by default.

Set your Gemini API key before running:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

In Google Colab:

```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```

---

# 1. LLM-as-Judge Evaluation

## Script

```text
evaluate_llm_as_judge.py
```

## Purpose

This script evaluates model outputs using an LLM judge.

It produces:

- Realism score
- `look_Real` binary label
- De-escalation score
- De-escalation outcome category
- Judge reasoning
- Penalty reasoning

---

## Input Format

The input should be a JSON file.

Example:

```json
[
  {
    "case_id": "case_001",
    "scenario_context": "Scenario context text here.",
    "profile": "Character profile text here.",
    "ground_truth": "Real-world reference conversation here.",
    "model_outputs": {
      "qwen_base": {
        "target_conversation": "Generated suspect response.",
        "conversation_transcript": "Full officer-suspect transcript."
      },
      "qwen_finetuned": {
        "target_conversation": "Generated suspect response.",
        "conversation_transcript": "Full officer-suspect transcript."
      },
      "gemini_2_5_flash": {
        "target_conversation": "Generated suspect response.",
        "conversation_transcript": "Full officer-suspect transcript."
      }
    }
  }
]
```

---

## Run LLM-as-Judge Evaluation

```bash
python evaluate_llm_as_judge.py \
  --input evaluation_cases.json \
  --output_dir results \
  --judge_model gemini-2.5-flash
```

---

## LLM-as-Judge Outputs

The script saves:

```text
results/
├── llm_as_judge_results.jsonl
├── llm_as_judge_scores.csv
└── llm_as_judge_model_summary.csv
```

### Output Files

- `llm_as_judge_results.jsonl`: full detailed judge outputs.
- `llm_as_judge_scores.csv`: flat score table.
- `llm_as_judge_model_summary.csv`: average scores by model.

---

# 2. Metric-Based Evaluation

## Script

```text
metric_based_eval_all_models.py
```

## Purpose

This script generates civilian/suspect responses from each model and compares them with real responses using automatic text-generation metrics.

Metrics used:

- ROUGE-L
- BLEU-4
- METEOR
- BERTScore F1

---

## Input CSV Format

Each input CSV file should contain at least these columns:

```text
speaker_label, transcript
```

Example:

```csv
speaker_label,transcript
Officer,"Can you step over here and talk to me?"
Suspect,"I didn't do anything. Why are you bothering me?"
Officer,"I just want to understand what happened."
Suspect,"People keep lying about me."
```

The script evaluates model generation when an officer or authority speaker is followed by a suspect/civilian response.

---

## Authority Speaker Keywords

The script identifies officer or authority turns using keywords such as:

```text
Deputy
Officer
Dispatcher
Law enforcement
Property owner
Sergeant
Lieutenant
Trooper
Police
Sheriff
Corporal
Agent
```

All other speakers are treated as civilian/suspect responses.

---

## Update Model Paths

Before running all models, update the `MODEL_SPECS` section inside:

```text
metric_based_eval_all_models.py
```

Example:

```python
ModelSpec(
    model_name="llama_finetuned",
    family="llama",
    model_type="fine_tuned",
    backend="local_unsloth",
    model_path="/content/drive/MyDrive/deescalation_project/Checkpoints/llama/checkpoint-120",
)
```

---

## Run Only Qwen First

This is recommended for testing:

```bash
python metric_based_eval_all_models.py \
  --input_folder /content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs \
  --output_folder /content/drive/MyDrive/deescalation_project/Checkpoints/metric_results \
  --load_in_4bit \
  --only_models qwen_base qwen_finetuned
```

---

## Run All Models

After replacing all `CHANGE_ME` paths:

```bash
python metric_based_eval_all_models.py \
  --input_folder /content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs \
  --output_folder /content/drive/MyDrive/deescalation_project/Checkpoints/metric_results \
  --load_in_4bit
```

---

## Metric-Based Outputs

The script saves one CSV per model and summary files.

Example:

```text
metric_results/
├── qwen_base_metric_results.csv
├── qwen_finetuned_metric_results.csv
├── llama_base_metric_results.csv
├── llama_finetuned_metric_results.csv
├── all_metric_results.csv
├── model_metric_summary.csv
└── file_level_metric_summary.csv
```

### Output Files

- `MODEL_NAME_metric_results.csv`: turn-level predictions and metric scores for one model.
- `all_metric_results.csv`: combined results for all evaluated models.
- `model_metric_summary.csv`: average scores by model.
- `file_level_metric_summary.csv`: average scores by source file and model.

---

## Notes

- Local models are loaded one at a time to reduce GPU memory usage.
- The metric-based script uses teacher forcing.
- After each generated response, the conversation history continues with the real reference response.
- Gemini requires `GEMINI_API_KEY`.
- Models with `CHANGE_ME` paths will be skipped.
- BERTScore may be slow for large datasets.
- Use GPU when possible.

---

## Recommended Workflow

1. Place both scripts in the `Evaluation/` folder.
2. Install dependencies from `requirements.txt`.
3. Update model checkpoint paths.
4. Run metric-based evaluation first.
5. Check `model_metric_summary.csv`.
6. Run LLM-as-judge evaluation for realism and de-escalation quality.
7. Compare automatic metrics with judge-based scores.

---

## Example Full Workflow

```bash
cd Evaluation

pip install -r requirements.txt

export GEMINI_API_KEY="your_api_key_here"

python metric_based_eval_all_models.py \
  --input_folder /content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs \
  --output_folder /content/drive/MyDrive/deescalation_project/Checkpoints/metric_results \
  --load_in_4bit \
  --only_models qwen_base qwen_finetuned

python evaluate_llm_as_judge.py \
  --input evaluation_cases.json \
  --output_dir judge_results \
  --judge_model gemini-2.5-flash
```

---

## Requirements

See:

```text
requirements.txt
```

for the full Python package list.