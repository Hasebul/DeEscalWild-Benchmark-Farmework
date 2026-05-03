# Generation

This folder contains the script for generating full police de-escalation conversations using multiple base models, fine-tuned models, and Gemini 2.5 Flash.

The generated outputs from this folder can later be used for:

- LLM-as-judge evaluation
- Realism evaluation
- De-escalation evaluation
- Metric-based evaluation such as ROUGE, BLEU, METEOR, and BERTScore

---

## Folder Structure

```text
Generation/
├── generate_all_model_conversations.py
├── scenarios.json
├── responses/
└── README.md
```

---

## Purpose

The main script:

```text
generate_all_model_conversations.py
```

loads scenario information from a JSON file and generates full civilian/suspect responses for each police dialogue turn.

The script supports:

- Qwen base
- Qwen fine-tuned
- Llama base
- Llama fine-tuned
- Phi base
- Phi fine-tuned
- Mistral base
- Mistral fine-tuned
- Granite base
- Granite fine-tuned
- Gemma base
- Gemma fine-tuned
- Falcon base
- Falcon fine-tuned
- Gemini 2.5 Flash

Each model output is saved in a separate folder inside:

```text
responses/
```

---

## Input JSON Format

The input file should be named:

```text
scenarios.json
```

Example format:

```json
[
  {
    "case_id": "case_001",
    "scenario_context": "A person is upset outside a store after being accused of shoplifting. Police are trying to calm them down.",
    "character_profile": {
      "name": "Marcus",
      "age": 31,
      "emotional_state": "angry, defensive, embarrassed",
      "speaking_style": "short, frustrated, suspicious of police",
      "background": "He believes the store owner racially profiled him and called police unfairly."
    },
    "police_dialogue": [
      "Hey, I just want to talk with you for a minute. Can you step over here?",
      "I understand you're upset. Tell me what happened from your side.",
      "Nobody is saying you're under arrest right now. I just need you to lower your voice so we can figure this out.",
      "What would help calm this down right now?",
      "Okay, let's take this one step at a time."
    ]
  }
]
```

---

## Required Fields

Each case should include:

| Field | Description |
|---|---|
| `case_id` | Unique ID for the scenario |
| `scenario_context` | Background information about the situation |
| `character_profile` | Profile of the civilian/suspect |
| `police_dialogue` | List of police/officer turns |

---

## Installation

Install the required packages from the main project or evaluation environment:

```bash
pip install pandas tqdm torch transformers accelerate bitsandbytes unsloth google-genai
```

If using Google Colab and Unsloth gives installation issues, use:

```bash
pip install --upgrade pip
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

---

## Gemini API Key

Gemini 2.5 Flash requires an API key.

Set the key before running:

```bash
export GEMINI_API_KEY="your_api_key_here"
```

In Google Colab:

```python
import os
os.environ["GEMINI_API_KEY"] = "your_api_key_here"
```

---

## Update Model Paths

Before running all models, open:

```text
generate_all_model_conversations.py
```

Then update the `MODEL_SPECS` section.

Replace all `CHANGE_ME` values with the correct Hugging Face model names or fine-tuned checkpoint paths.

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

Models with `CHANGE_ME` paths will be skipped.

---

## Run Only Qwen Models

Recommended for testing first:

```bash
python generate_all_model_conversations.py \
  --input_json scenarios.json \
  --output_dir responses \
  --load_in_4bit \
  --only_models qwen_base qwen_finetuned
```

---

## Run All Models

After replacing all model paths:

```bash
python generate_all_model_conversations.py \
  --input_json scenarios.json \
  --output_dir responses \
  --load_in_4bit
```

---

## Run a Specific Model

Example:

```bash
python generate_all_model_conversations.py \
  --input_json scenarios.json \
  --output_dir responses \
  --load_in_4bit \
  --only_models llama_finetuned
```

---

## Output Structure

The script creates a separate response folder for each model.

Example:

```text
responses/
├── qwen_base/
│   ├── case_001.json
│   └── case_001.txt
├── qwen_finetuned/
│   ├── case_001.json
│   └── case_001.txt
├── llama_base/
├── llama_finetuned/
├── phi_base/
├── phi_finetuned/
├── mistral_base/
├── mistral_finetuned/
├── granite_base/
├── granite_finetuned/
├── gemma_base/
├── gemma_finetuned/
├── falcon_base/
├── falcon_finetuned/
└── gemini_2_5_flash/
```

---

## Output Files

For each case, the script saves two files:

```text
case_001.json
case_001.txt
```

### JSON Output

The `.json` file contains structured information:

```json
{
  "case_id": "case_001",
  "model_name": "qwen_base",
  "model_family": "qwen",
  "model_type": "base",
  "scenario_context": "...",
  "character_profile": "...",
  "generated_conversation": [
    {
      "turn_index": 0,
      "police": "Hey, I just want to talk with you for a minute.",
      "civilian_generated": "Why? I didn't do anything."
    }
  ],
  "conversation_transcript": "Officer: ...\nCivilian: ..."
}
```

### TXT Output

The `.txt` file contains a readable transcript:

```text
Officer: Hey, I just want to talk with you for a minute.
Civilian: Why? I didn't do anything.

Officer: I understand you're upset. Tell me what happened from your side.
Civilian: They keep accusing me like I stole something, but I didn't.
```

---

## Recommended Workflow

1. Add all scenarios to `scenarios.json`.
2. Update model checkpoint paths in `generate_all_model_conversations.py`.
3. Test with only one or two models.
4. Run all models after confirming the output format.
5. Use the generated `.json` files for evaluation.

---

## Notes

- Local models are loaded one at a time to reduce GPU memory usage.
- Gemini requires `GEMINI_API_KEY`.
- The script generates only the civilian/suspect response.
- The police dialogue comes directly from the input JSON.
- Each model gets its own response folder.
- JSON outputs are recommended for evaluation.
- TXT outputs are useful for manual reading.