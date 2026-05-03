"""
metric_based_eval_all_models.py

Metric-based evaluation for de-escalation / suspect-response generation.

Evaluates:
- Qwen base + fine-tuned
- Llama base + fine-tuned
- Phi base + fine-tuned
- Mistral base + fine-tuned
- Granite base + fine-tuned
- Gemma base + fine-tuned
- Falcon base + fine-tuned
- Gemini 2.5 Flash

Metrics:
- ROUGE-L
- BLEU-4
- METEOR
- BERTScore F1

Input CSV format:
Each CSV should contain at least:
    speaker_label, transcript

Install:
    pip install pandas numpy tqdm evaluate bert-score rouge-score nltk google-genai
    pip install unsloth

For Gemini:
    export GEMINI_API_KEY="your_key"

Run:
    python metric_based_eval_all_models.py \
        --input_folder /content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs \
        --output_folder /content/drive/MyDrive/deescalation_project/Checkpoints/metric_results
"""

import argparse
import gc
import glob
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import evaluate


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelSpec:
    model_name: str
    family: str
    model_type: str   # "base", "fine_tuned", or "api"
    backend: str      # "local_unsloth" or "gemini"
    model_path: str


"""
IMPORTANT:
Replace CHANGE_ME paths with your actual fine-tuned checkpoint paths.

For base models, use the Hugging Face / Unsloth model names you actually used.
The Qwen paths below use your uploaded script as the starting example.
"""

MODEL_SPECS: List[ModelSpec] = [
    # ---------------- QWEN ----------------
    ModelSpec(
        model_name="qwen_base",
        family="qwen",
        model_type="base",
        backend="local_unsloth",
        model_path="unsloth/Qwen2.5-3B-Instruct",
    ),
    ModelSpec(
        model_name="qwen_finetuned",
        family="qwen",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="/content/drive/MyDrive/deescalation_project/Checkpoints/qwen_25/checkpoint-118",
    ),

    # ---------------- LLAMA ----------------
    ModelSpec(
        model_name="llama_base",
        family="llama",
        model_type="base",
        backend="local_unsloth",
        model_path="CHANGE_ME/llama_base_model",
    ),
    ModelSpec(
        model_name="llama_finetuned",
        family="llama",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="CHANGE_ME/llama_finetuned_checkpoint",
    ),

    # ---------------- PHI ----------------
    ModelSpec(
        model_name="phi_base",
        family="phi",
        model_type="base",
        backend="local_unsloth",
        model_path="CHANGE_ME/phi_base_model",
    ),
    ModelSpec(
        model_name="phi_finetuned",
        family="phi",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="CHANGE_ME/phi_finetuned_checkpoint",
    ),

    # ---------------- MISTRAL ----------------
    ModelSpec(
        model_name="mistral_base",
        family="mistral",
        model_type="base",
        backend="local_unsloth",
        model_path="CHANGE_ME/mistral_base_model",
    ),
    ModelSpec(
        model_name="mistral_finetuned",
        family="mistral",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="CHANGE_ME/mistral_finetuned_checkpoint",
    ),

    # ---------------- GRANITE ----------------
    ModelSpec(
        model_name="granite_base",
        family="granite",
        model_type="base",
        backend="local_unsloth",
        model_path="CHANGE_ME/granite_base_model",
    ),
    ModelSpec(
        model_name="granite_finetuned",
        family="granite",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="CHANGE_ME/granite_finetuned_checkpoint",
    ),

    # ---------------- GEMMA ----------------
    ModelSpec(
        model_name="gemma_base",
        family="gemma",
        model_type="base",
        backend="local_unsloth",
        model_path="CHANGE_ME/gemma_base_model",
    ),
    ModelSpec(
        model_name="gemma_finetuned",
        family="gemma",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="CHANGE_ME/gemma_finetuned_checkpoint",
    ),

    # ---------------- FALCON ----------------
    ModelSpec(
        model_name="falcon_base",
        family="falcon",
        model_type="base",
        backend="local_unsloth",
        model_path="CHANGE_ME/falcon_base_model",
    ),
    ModelSpec(
        model_name="falcon_finetuned",
        family="falcon",
        model_type="fine_tuned",
        backend="local_unsloth",
        model_path="CHANGE_ME/falcon_finetuned_checkpoint",
    ),

    # ---------------- GEMINI ----------------
    ModelSpec(
        model_name="gemini_2_5_flash",
        family="gemini",
        model_type="api",
        backend="gemini",
        model_path="gemini-2.5-flash",
    ),
]


# ============================================================
# 2. GENERAL CONFIGURATION
# ============================================================

AUTHORITY_KEYWORDS = [
    "Deputy",
    "Officer",
    "Dispatcher",
    "Law enforcement",
    "Property owner",
    "Sergeant",
    "Lieutenant",
    "Trooper",
    "Police",
    "Sheriff",
    "Corporal",
    "Agent",
]

SYSTEM_PROMPT = (
    "You are a civilian/suspect in a police interaction. "
    "Respond naturally as the civilian only. "
    "You may be agitated, emotional, hesitant, suspicious, or non-compliant depending on the conversation context. "
    "Do not write the officer's next line. "
    "Return only the civilian's next response."
)


# ============================================================
# 3. METRIC EVALUATOR
# ============================================================

class MetricsEvaluator:
    def __init__(self, bertscore_device: Optional[str] = None):
        print("Loading metric evaluators: ROUGE, BLEU, METEOR, BERTScore...")

        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.bertscore = evaluate.load("bertscore")

        self.bertscore_device = bertscore_device

        print("Metrics loaded.")

    @staticmethod
    def clean_text(x: Any) -> str:
        if x is None or pd.isna(x):
            return " "
        x = str(x).strip()
        return x if x else " "

    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, List[float]]:
        predictions = [self.clean_text(x) for x in predictions]
        references = [self.clean_text(x) for x in references]

        rouge_l_scores = []
        bleu_4_scores = []
        meteor_scores = []

        print("Computing BERTScore...")
        bert_kwargs = {
            "predictions": predictions,
            "references": references,
            "lang": "en",
            "verbose": False,
            "batch_size": 16,
        }

        if self.bertscore_device is not None:
            bert_kwargs["device"] = self.bertscore_device

        bert_results = self.bertscore.compute(**bert_kwargs)
        bert_f1_scores = bert_results["f1"]

        print("Computing ROUGE-L, BLEU-4, and METEOR...")
        for pred, ref in tqdm(
            list(zip(predictions, references)),
            desc="Metric rows",
            leave=False,
        ):
            rouge_result = self.rouge.compute(
                predictions=[pred],
                references=[ref],
                rouge_types=["rougeL"],
            )
            rouge_l_scores.append(float(rouge_result["rougeL"]))

            bleu_result = self.bleu.compute(
                predictions=[pred],
                references=[ref],
                max_order=4,
                smooth=True,
            )
            bleu_4_scores.append(float(bleu_result["bleu"]))

            meteor_result = self.meteor.compute(
                predictions=[pred],
                references=[ref],
            )
            meteor_scores.append(float(meteor_result["meteor"]))

        return {
            "ROUGE_L": rouge_l_scores,
            "BLEU_4": bleu_4_scores,
            "METEOR": meteor_scores,
            "BERTScore_F1": bert_f1_scores,
        }


# ============================================================
# 4. ROLE AND PROMPT HELPERS
# ============================================================

def get_role(speaker_label: Any) -> str:
    """
    Maps speaker labels to chat roles.

    Officer / authority -> user
    Civilian / suspect -> assistant
    """
    if pd.isna(speaker_label):
        return "assistant"

    label = str(speaker_label).lower()

    for keyword in AUTHORITY_KEYWORDS:
        if keyword.lower() in label:
            return "user"

    return "assistant"


def build_plain_prompt(history_messages: List[Dict[str, str]]) -> str:
    """
    Fallback prompt format for models without a chat template
    and for Gemini API generation.
    """
    lines = []

    for msg in history_messages:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"Officer: {content}")
        elif role == "assistant":
            lines.append(f"Civilian: {content}")

    lines.append("Civilian:")

    return "\n".join(lines)


def clean_generated_reply(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    stop_markers = [
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
        "Officer:",
        "User:",
        "System:",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0].strip()

    prefixes = [
        "Civilian:",
        "Suspect:",
        "Subject:",
        "Assistant:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text.strip()


# ============================================================
# 5. GENERATORS
# ============================================================

class LocalUnslothGenerator:
    def __init__(
        self,
        spec: ModelSpec,
        max_seq_length: int = 2048,
        dtype=None,
        load_in_4bit: bool = True,
    ):
        import torch
        from unsloth import FastLanguageModel

        self.torch = torch
        self.FastLanguageModel = FastLanguageModel

        self.spec = spec
        self.max_seq_length = max_seq_length
        self.dtype = dtype
        self.load_in_4bit = load_in_4bit

        print(f"\nLoading local model: {spec.model_name}")
        print(f"Path: {spec.model_path}")

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=spec.model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
        )

        FastLanguageModel.for_inference(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(
        self,
        history_messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        tokenizer = self.tokenizer
        model = self.model

        try:
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                encoded = tokenizer.apply_chat_template(
                    history_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                input_ids = encoded.to(self.device)
                attention_mask = None

            else:
                prompt = build_plain_prompt(history_messages)
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_length,
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

            with self.torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.05,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][input_ids.shape[-1]:]
            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)

            return clean_generated_reply(decoded)

        except Exception as exc:
            return f"[GENERATION_ERROR: {exc}]"

    def unload(self):
        try:
            del self.model
            del self.tokenizer
        except Exception:
            pass

        gc.collect()

        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class GeminiGenerator:
    def __init__(self, spec: ModelSpec):
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GEMINI_API_KEY environment variable.")

        self.genai = genai
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.spec = spec

    def generate(
        self,
        history_messages: List[Dict[str, str]],
        max_new_tokens: int = 128,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        prompt = build_plain_prompt(history_messages)

        try:
            response = self.client.models.generate_content(
                model=self.spec.model_path,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
            )

            return clean_generated_reply(response.text)

        except Exception as exc:
            return f"[GEMINI_ERROR: {exc}]"

    def unload(self):
        pass


def create_generator(
    spec: ModelSpec,
    max_seq_length: int,
    load_in_4bit: bool,
):
    if spec.backend == "local_unsloth":
        return LocalUnslothGenerator(
            spec=spec,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

    if spec.backend == "gemini":
        return GeminiGenerator(spec)

    raise ValueError(f"Unsupported backend: {spec.backend}")


# ============================================================
# 6. FILE-LEVEL EVALUATION
# ============================================================

def generate_predictions_for_file(
    generator,
    spec: ModelSpec,
    file_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> pd.DataFrame:
    filename = os.path.basename(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"[SKIP] Could not read {filename}: {exc}")
        return pd.DataFrame()

    required_cols = {"speaker_label", "transcript"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        print(f"[SKIP] {filename} missing columns: {missing_cols}")
        return pd.DataFrame()

    if len(df) < 2:
        print(f"[SKIP] {filename} is too short.")
        return pd.DataFrame()

    history = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    rows = []

    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i + 1]

        current_role = get_role(current_row["speaker_label"])
        next_role = get_role(next_row["speaker_label"])

        current_text = str(current_row["transcript"])
        next_text = str(next_row["transcript"])

        history.append(
            {
                "role": current_role,
                "content": current_text,
            }
        )

        # Evaluation point:
        # Officer speaks now, real civilian/suspect responds next.
        if current_role == "user" and next_role == "assistant":
            predicted_reply = generator.generate(
                history_messages=history,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

            rows.append(
                {
                    "source_file": filename,
                    "turn_index": i,
                    "model_name": spec.model_name,
                    "model_family": spec.family,
                    "model_type": spec.model_type,
                    "officer_input": current_text,
                    "reference_response": next_text,
                    "predicted_response": predicted_reply,
                }
            )

            # Teacher forcing:
            # Continue the conversation with the real response,
            # not the generated response.
            history.append(
                {
                    "role": "assistant",
                    "content": next_text,
                }
            )

        elif current_role == "assistant":
            # If current row is already a civilian/suspect response,
            # it has already been added above.
            continue

    return pd.DataFrame(rows)


def evaluate_one_model(
    spec: ModelSpec,
    csv_files: List[str],
    output_folder: Path,
    evaluator: MetricsEvaluator,
    max_seq_length: int,
    load_in_4bit: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> pd.DataFrame:
    if spec.backend == "local_unsloth" and spec.model_path.startswith("CHANGE_ME"):
        print(f"[SKIP] {spec.model_name}: model_path is still CHANGE_ME.")
        return pd.DataFrame()

    generator = create_generator(
        spec=spec,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model_rows = []

    for file_path in tqdm(csv_files, desc=f"Generating: {spec.model_name}"):
        file_df = generate_predictions_for_file(
            generator=generator,
            spec=spec,
            file_path=file_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if not file_df.empty:
            model_rows.append(file_df)

    generator.unload()

    if not model_rows:
        print(f"[WARN] No rows generated for {spec.model_name}.")
        return pd.DataFrame()

    result_df = pd.concat(model_rows, ignore_index=True)

    print(f"\nComputing metrics for {spec.model_name}...")

    metric_values = evaluator.compute_metrics(
        predictions=result_df["predicted_response"].tolist(),
        references=result_df["reference_response"].tolist(),
    )

    for metric_name, values in metric_values.items():
        result_df[metric_name] = values

    model_output_path = output_folder / f"{spec.model_name}_metric_results.csv"
    result_df.to_csv(model_output_path, index=False)

    print(f"Saved model results: {model_output_path}")

    return result_df


# ============================================================
# 7. SUMMARY
# ============================================================

def save_summary(all_results: pd.DataFrame, output_folder: Path):
    if all_results.empty:
        print("[WARN] No results to summarize.")
        return

    all_results_path = output_folder / "all_metric_results.csv"
    all_results.to_csv(all_results_path, index=False)

    summary = (
        all_results
        .groupby(["model_name", "model_family", "model_type"], dropna=False)
        .agg(
            n_turns=("turn_index", "count"),
            avg_ROUGE_L=("ROUGE_L", "mean"),
            avg_BLEU_4=("BLEU_4", "mean"),
            avg_METEOR=("METEOR", "mean"),
            avg_BERTScore_F1=("BERTScore_F1", "mean"),
            std_ROUGE_L=("ROUGE_L", "std"),
            std_BLEU_4=("BLEU_4", "std"),
            std_METEOR=("METEOR", "std"),
            std_BERTScore_F1=("BERTScore_F1", "std"),
        )
        .reset_index()
        .sort_values(
            by=["avg_BERTScore_F1", "avg_METEOR", "avg_ROUGE_L"],
            ascending=False,
        )
    )

    summary_path = output_folder / "model_metric_summary.csv"
    summary.to_csv(summary_path, index=False)

    file_summary = (
        all_results
        .groupby(["source_file", "model_name", "model_family", "model_type"], dropna=False)
        .agg(
            n_turns=("turn_index", "count"),
            avg_ROUGE_L=("ROUGE_L", "mean"),
            avg_BLEU_4=("BLEU_4", "mean"),
            avg_METEOR=("METEOR", "mean"),
            avg_BERTScore_F1=("BERTScore_F1", "mean"),
        )
        .reset_index()
    )

    file_summary_path = output_folder / "file_level_metric_summary.csv"
    file_summary.to_csv(file_summary_path, index=False)

    print("\nSaved:")
    print(f"1. All results: {all_results_path}")
    print(f"2. Model summary: {summary_path}")
    print(f"3. File-level summary: {file_summary_path}")

    print("\nModel summary:")
    print(summary.to_string(index=False))


# ============================================================
# 8. MAIN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_folder",
        required=True,
        help="Folder containing test CSV files.",
    )

    parser.add_argument(
        "--output_folder",
        required=True,
        help="Folder to save evaluated results.",
    )

    parser.add_argument(
        "--only_models",
        nargs="*",
        default=None,
        help="Optional list of model_name values to evaluate.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit loading for local Unsloth models.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--bertscore_device",
        default=None,
        help="Optional BERTScore device, e.g. cuda or cpu.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(str(input_folder / "*.csv")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_folder}")

    print(f"Found {len(csv_files)} CSV files.")

    if args.only_models:
        selected_specs = [
            spec for spec in MODEL_SPECS
            if spec.model_name in args.only_models
        ]
    else:
        selected_specs = MODEL_SPECS

    if not selected_specs:
        raise ValueError("No models selected. Check --only_models names.")

    print("\nSelected models:")
    for spec in selected_specs:
        print(f"- {spec.model_name}: {spec.model_path}")

    evaluator = MetricsEvaluator(
        bertscore_device=args.bertscore_device,
    )

    all_model_results = []

    for spec in selected_specs:
        print("\n" + "=" * 80)
        print(f"Evaluating model: {spec.model_name}")
        print("=" * 80)

        try:
            model_df = evaluate_one_model(
                spec=spec,
                csv_files=csv_files,
                output_folder=output_folder,
                evaluator=evaluator,
                max_seq_length=args.max_seq_length,
                load_in_4bit=args.load_in_4bit,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            if not model_df.empty:
                all_model_results.append(model_df)

        except Exception as exc:
            print(f"[ERROR] Failed model {spec.model_name}: {exc}")

        gc.collect()

    if all_model_results:
        all_results = pd.concat(all_model_results, ignore_index=True)
    else:
        all_results = pd.DataFrame()

    save_summary(all_results, output_folder)

    print("\nMetric-based evaluation complete.")


if __name__ == "__main__":
    main()