# DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with SLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Status-Under%20Review-red)](https://icml.cc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

**DeEscalWild** is the first large-scale benchmark dataset derived from "in-the-wild" police-civilian interactions. It contains **1,500 high-fidelity scenarios** and **285,887 dialogue turns** designed to train Small Language Models (SLMs) for low-latency, privacy-preserving de-escalation training on edge devices (e.g., VR headsets).

This repository contains the official implementation for the entire data pipeline:
1.  **Acquisition:** Mining raw interaction videos from YouTube, TikTok, and Facebook.
2.  **Hybrid Filtering:** Using "LLM-as-a-Judge" to identify high-intensity conflict scenarios.
3.  **Diarization:** converting raw audio into structured multi-speaker scripts using Gemini 2.5 Flash.
4.  **Training:** Fine-tuning SLMs (Qwen 2.5, Llama 3.2) via QLORA.

---

## 📋 Table of Contents
- [Prerequisites & Installation](#prerequisites--installation)
- [Repository Structure](#repository-structure)
- [Data Pipeline](#data-pipeline)
  - [Step 1: Social Media Mining](#step-1-social-media-mining)
  - [Step 2: Transcription & Hybrid Filtering](#step-2-transcription--hybrid-filtering)
  - [Step 3: Diarization with Gemini](#step-3-diarization-with-gemini)
- [SLM Training & Evaluation](#slm-training--evaluation)
- [Results](#results)
- [Ethical Use](#ethical-use)
- [Citation](#citation)

---

## 🛠️ Prerequisites & Installation

### System Requirements
* **OS:** Linux (Ubuntu 20.04+ recommended)
* **Python:** 3.10+
* **GPU:** NVIDIA GPU with at least 24GB VRAM (tested on RTX 3090) for training.
* **API Keys:** Google Gemini API Key (for diarization pipeline).

### Installation
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/DeEscalWild.git](https://github.com/your-username/DeEscalWild.git)
    cd DeEscalWild
    ```

2.  **Create a virtual environment:**
    ```bash
    conda create -n deescalwild python=3.10
    conda activate deescalwild
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 📂 Repository Structure

```text
DeEscalWild/
├── src/
│   ├── data_acquisition/
│   │   ├── relevant_data_filtering/
│   │   ├── video_metadata_scraper_pipeline/
│   │   └── video_speaker_diarization/
│   │
│   ├── evaluation/
│   │   ├── evaluate_llm_as_judge.py
│   │   ├── metric_based_eval_all_models.py
│   │   ├── requirements.txt
│   │   └── README.md
│   │
│   ├── fine_tuning/
│   │   └── README.md
│   │
│   └── generate_model_response/
│       ├── generate_all_model_conversations.py
│       ├── scenarios.json
│       ├── responses/
│       └── README.md
│
├── .gitignore
├── Deescal_wild.jpg
├── index.html
├── LICENSE
├── requirements.txt
└── README.md
```

## 🚀 Data Pipeline

Each major folder contains its own `README.md` file with detailed instructions for that step.  
Follow the folder-level README files when running the full pipeline.

### Step 1: Social Media Mining

Raw video metadata is collected from identified public sources, including YouTube channels, TikTok accounts, and Facebook pages.

This step is handled inside:

```text
src/data_acquisition/video_metadata_scraper_pipeline/
```

> **Note:** The scraper should follow each platform’s terms of service, API limits, and robots policies. Use rate limiting, retry logic, and exponential backoff to avoid excessive requests.

```bash
python src/data_acquisition/video_metadata_scraper_pipeline/scrape_sources.py \
    --platforms youtube tiktok facebook \
    --output_dir ./data/raw_metadata \
    --max_videos 5000
```

For more details, see:

```text
src/data_acquisition/video_metadata_scraper_pipeline/README.md
```

---

### Step 2: Transcription and Relevant Data Filtering

Raw videos are processed to identify law-enforcement-related interactions.  
The filtering process uses transcripts and a taxonomy of binary features, such as police presence, escalation signals, and de-escalation signals.

This step is handled inside:

```text
src/data_acquisition/relevant_data_filtering/
```

The filtering logic follows:

```math
C_{valid} = (Police \ge 2) \land (Escalation \ge 3 \lor DeEscalation \ge 3)
```

Pipeline steps:

1. Transcribe raw audio using Whisper.
2. Extract relevant interaction features.
3. Filter videos using the feature-based selection rule.
4. Save filtered examples for diarization and benchmark construction.

```bash
python src/data_acquisition/relevant_data_filtering/filter_pipeline.py \
    --input_dir ./data/raw_metadata \
    --output_dir ./data/filtered_audio \
    --whisper_model large-v3 \
    --threshold_police 2 \
    --threshold_escalation 3 \
    --threshold_deescalation 3
```

For more details, see:

```text
src/data_acquisition/relevant_data_filtering/README.md
```

---

### Step 3: Speaker Diarization

Filtered audio is processed to identify speaker roles such as Officer, Subject, Bystander, or Dispatcher.  
The diarization pipeline produces structured JSON transcripts with speaker labels, timestamps, and context tags.

This step is handled inside:

```text
src/data_acquisition/video_speaker_diarization/
```

```bash
python src/data_acquisition/video_speaker_diarization/generate_scripts.py \
    --model gemini-2.5-flash \
    --input_dir ./data/filtered_audio \
    --output_dir ./data/final_benchmark
```

Set your Gemini API key before running:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_API_KEY"
```

For Google Colab:

```python
import os
os.environ["GEMINI_API_KEY"] = "YOUR_GOOGLE_API_KEY"
```

For more details, see:

```text
src/data_acquisition/video_speaker_diarization/README.md
```

---

## 🤖 Model Generation, Training, and Evaluation

The project evaluates base and fine-tuned small language models on police de-escalation dialogue generation.

The supported model families include:

- Qwen
- Llama
- Phi
- Mistral
- Granite
- Gemma
- Falcon
- Gemini 2.5 Flash

---

### Step 4: Fine-Tuning

Small Language Models are fine-tuned using QLoRA for efficient training.

This step is handled inside:

```text
src/fine_tuning/
```

Example configuration:

- Base model: Qwen 2.5 3B Instruct
- LoRA rank: `r = 16`
- LoRA alpha: `alpha = 32`
- Quantization: 4-bit

```bash
python src/fine_tuning/finetune.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --data_path ./data/final_benchmark/train.json \
    --output_dir ./checkpoints/qwen_deescalwild \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --epochs 3
```

For more details, see:

```text
src/fine_tuning/README.md
```

---

### Step 5: Generate Model Responses

After fine-tuning, generate full model responses for each benchmark scenario.

This step is handled inside:

```text
src/generate_model_response/
```

The generation script loads:

- Scenario context
- Character profile
- Police dialogue turns

It then generates the civilian/suspect side of the conversation for each model.

```bash
python src/generate_model_response/generate_all_model_conversations.py \
    --input_json ./src/generate_model_response/scenarios.json \
    --output_dir ./src/generate_model_response/responses \
    --load_in_4bit
```

To test only Qwen models first:

```bash
python src/generate_model_response/generate_all_model_conversations.py \
    --input_json ./src/generate_model_response/scenarios.json \
    --output_dir ./src/generate_model_response/responses \
    --load_in_4bit \
    --only_models qwen_base qwen_finetuned
```

For more details, see:

```text
src/generate_model_response/README.md
```

---

### Step 6: Metric-Based Evaluation

Generated model responses are compared against reference responses using automatic text-generation metrics.

This step is handled inside:

```text
src/evaluation/
```

Metrics include:

- ROUGE-L
- BLEU-4
- METEOR
- BERTScore F1

```bash
python src/evaluation/metric_based_eval_all_models.py \
    --input_folder ./data/final_benchmark/test_data_csvs \
    --output_folder ./results/metric_results \
    --load_in_4bit
```

To evaluate only Qwen models:

```bash
python src/evaluation/metric_based_eval_all_models.py \
    --input_folder ./data/final_benchmark/test_data_csvs \
    --output_folder ./results/metric_results \
    --load_in_4bit \
    --only_models qwen_base qwen_finetuned
```

For more details, see:

```text
src/evaluation/README.md
```

---

### Step 7: LLM-as-Judge Evaluation

The LLM-as-judge evaluation measures two higher-level qualities:

1. Realism of the generated civilian/suspect response.
2. De-escalation effectiveness of the full interaction.

This step is handled inside:

```text
src/evaluation/
```

```bash
python src/evaluation/evaluate_llm_as_judge.py \
    --input ./data/final_benchmark/evaluation_cases.json \
    --output_dir ./results/judge_results \
    --judge_model gemini-2.5-flash
```

Set your Gemini API key before running:

```bash
export GEMINI_API_KEY="YOUR_GOOGLE_API_KEY"
```

For more details, see:

```text
src/evaluation/README.md
```

---

## Recommended Full Workflow

Follow the README file in each folder for detailed instructions.

```text
1. src/data_acquisition/video_metadata_scraper_pipeline/README.md
2. src/data_acquisition/relevant_data_filtering/README.md
3. src/data_acquisition/video_speaker_diarization/README.md
4. src/fine_tuning/README.md
5. src/generate_model_response/README.md
6. src/evaluation/README.md
```

Recommended order:

```text
Metadata Scraping
        ↓
Transcription and Filtering
        ↓
Speaker Diarization
        ↓
Benchmark Construction
        ↓
Fine-Tuning
        ↓
Model Response Generation
        ↓
Metric-Based Evaluation
        ↓
LLM-as-Judge Evaluation
```

### 📊 Results

Our fine-tuned Qwen 2.5 (3B) outperforms the general-purpose Gemini 2.5 Flash (prompt-engineered) on domain-specific metrics while requiring a fraction of the compute.

| Model | ROUGE-L | BLEU-4 | METEOR | BERTScore (F1) | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini 2.5 Flash** | 12.4 | 1.1 | 13.2 | 0.85 | 3.50 |
| **Qwen 2.5 (3B) FT** | **15.8** | **3.8** | **19.5** | **0.88** | **0.38** |
| **Llama 3.2 (3B) FT** | 14.8 | 2.3 | 17.8 | 0.87 | 0.44 |


## ⚖️ Ethical Use & Disclaimer

* **Warning:** This dataset contains real-world conflicts which may include aggressive language, slurs, and emotional volatility.
* **Intended Use:** Strictly for educational simulation and training human officers in de-escalation strategies.
* **Prohibited Use:** This dataset must **NOT** be used for predictive policing, automated profiling, or autonomous decision-making systems.
* **Privacy:** The pipeline is designed for local processing to preserve privacy.



## 📜 Citation

If you use **DeEscalWild** in your research, please cite our paper:

```bibtex
@inproceedings{hasan2026DeEscalWild,
  title     = {DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with {SLMs}},
  author    = {Hasan, Md Hasebul and Charu, Krity Haque and Sridhar, Eshwara Prasad and Deb, Shuchisnigdha and Islam, Mohammad A.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```