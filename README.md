# DeEscalWild: A Real-World Benchmark for Automated De-Escalation Training with SLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Status-Under%20Review-red)](https://icml.cc/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

**DeEscalWild** is the first large-scale benchmark dataset derived from "in-the-wild" police-civilian interactions. It contains **1,500 high-fidelity scenarios** and **285,887 dialogue turns** designed to train Small Language Models (SLMs) for low-latency, privacy-preserving de-escalation training on edge devices (e.g., VR headsets).

This repository contains the official implementation for the entire data pipeline:
1.  **Acquisition:** Mining raw interaction videos from YouTube, TikTok, and Facebook.
2.  **Hybrid Filtering:** Using "LLM-as-a-Judge" to identify high-intensity conflict scenarios[cite: 279].
3.  **Diarization:** converting raw audio into structured multi-speaker scripts using Gemini 2.5 Flash[cite: 362].
4.  **Training:** Fine-tuning SLMs (Qwen 2.5, Llama 3.2) via QLORA[cite: 468].

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
├── data/
│   ├── raw_metadata/       # Output from Step 1
│   ├── transcripts/        # Output from Step 2 (Whisper)
│   └── final_benchmark/    # Output from Step 3 (Final JSONs)
├── pipelines/
│   ├── acquisition/        # Scrapers for YT, TikTok, FB
│   ├── filtering/          # LLM-as-a-Judge logic
│   └── diarization/        # Gemini 2.5 Flash integration
├── training/
│   ├── finetune.py         # QLORA training script
│   └── evaluate.py         # ROUGE/BERTScore testing
├── requirements.txt
└── README.md
```

## 🚀 Data Pipeline

### Step 1: Social Media Mining
We collect raw video metadata from identified channels on YouTube (15 channels), TikTok (5 channels), and Facebook (3 pages).

> **Note:** The scraper implements a rotating proxy network to bypass platform rate-limiting, which typically blocks requests after ~50 items.

```bash
# Run the multi-platform scraper
python pipelines/acquisition/scrape_sources.py \
    --platforms youtube tiktok facebook \
    --output_dir ./data/raw_metadata \
    --max_videos 5000 \
    --use_proxies True
```

### Step 2: Transcription & Hybrid Filtering
Raw videos are processed to ensure they contain relevant law enforcement interactions using a taxonomy of 30 binary features (e.g., police presence, escalation signals).

1. **Transcribe** raw audio using OpenAI Whisper (Large-v3).
2. **Filter** using an LLM to enforce the logic:
   $$C_{valid} = (Police \ge 2) \land (Escalation \ge 3 \lor DeEscalation \ge 3)$$

```bash
python pipelines/filtering/filter_pipeline.py \
    --input_dir ./data/raw_metadata \
    --whisper_model large-v3 \
    --threshold_police 2 \
    --threshold_escalation 3
```
### Step 3: Diarization with Gemini
We utilize **Gemini 2.5 Flash** for high-fidelity speaker diarization. This model processes audio directly to handle multi-party confusion and "cross-talk" better than traditional pyannote pipelines.

* **Input:** Filtered Audio
* **Output:** JSON scripts with speaker roles (Officer, Subject, Bystander) and context tags.

```bash
python pipelines/diarization/generate_scripts.py \
    --model "gemini-2.5-flash" \
    --api_key "YOUR_GOOGLE_API_KEY" \
    --input_dir ./data/filtered_audio \
    --output_dir ./data/final_benchmark
```

## 🤖 SLM Training & Evaluation

We fine-tune Small Language Models (SLMs) using 4-bit Quantized Low-Rank Adaptation (QLORA) to run efficiently on edge hardware.

### Fine-Tuning
The default configuration uses **Qwen 2.5 (3B-Instruct)** with Rank $r=16$ and Alpha $\alpha=32$.

```bash
python training/finetune.py \
    --model_name "Qwen/Qwen2.5-3B-Instruct" \
    --data_path ./data/final_benchmark/train.json \
    --output_dir ./checkpoints/qwen_deescalwild \
    --lora_r 16 \
    --lora_alpha 32 \
    --batch_size 4 \
    --epochs 3
```

### Evaluation
Evaluate the model against the held-out test set (5% of scenarios) using ROUGE-L, BLEU-4, METEOR, and BERTScore.
```bash
    python training/evaluate.py \
    --base_model "Qwen/Qwen2.5-3B-Instruct" \
    --adapter_path ./checkpoints/qwen_deescalwild \
    --test_data ./data/final_benchmark/test.json
```

### 📊 Results

Our fine-tuned Qwen 2.5 (3B) outperforms the general-purpose Gemini 2.5 Flash (prompt-engineered) on domain-specific metrics while requiring a fraction of the compute.

| Model | ROUGE-L | BLEU-4 | METEOR | BERTScore (F1) | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini 2.5 Flash** | 12.4 | 1.1 | 13.2 | 0.85 | 3.50 |
| **Qwen 2.5 (3B) FT** | **15.8** | **3.8** | **19.5** | **0.88** | **0.38** |
| **Llama 3.2 (3B) FT** | 14.8 | 2.3 | 17.8 | 0.87 | 0.44 |


## ⚖️ Ethical Use & Disclaimer

* **Warning:** This dataset contains real-world conflicts which may include aggressive language, slurs, and emotional volatility. [cite: 803]
* **Intended Use:** Strictly for educational simulation and training human officers in de-escalation strategies. [cite: 810]
* **Prohibited Use:** This dataset must **NOT** be used for predictive policing, automated profiling, or autonomous decision-making systems. [cite: 810]
* **Privacy:** The pipeline is designed for local processing to preserve privacy. [cite: 808]



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