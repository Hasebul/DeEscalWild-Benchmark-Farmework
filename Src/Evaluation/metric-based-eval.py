import pandas as pd
import glob
import os
import torch
import numpy as np
import evaluate
from unsloth import FastLanguageModel
from tqdm import tqdm

# --- CONFIGURATION ---
input_folder = "/content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs"
output_folder = "/content/drive/MyDrive/deescalation_project/Checkpoints/evaluated_results"
checkpoint_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/qwen_25/checkpoint-118"
base_model_name = "unsloth/Qwen2.5-3B-Instruct"

max_seq_length = 2048
dtype = None
load_in_4bit = True

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Keywords to identify the Officer (User)
AUTHORITY_KEYWORDS = [
    "Deputy", "Officer", "Dispatcher", "Law enforcement",
    "Property owner", "Sergeant", "Lieutenant", "Trooper", "Police"
]

# ==========================================
# 1. METRICS EVALUATOR CLASS
# ==========================================
class MetricsEvaluator:
    def __init__(self):
        print("Loading Evaluation Metrics (ROUGE, BLEU, METEOR, BERTScore)...")
        # Load all metrics
        self.rouge = evaluate.load("rouge")
        self.bleu = evaluate.load("bleu")
        self.meteor = evaluate.load("meteor")
        self.bertscore = evaluate.load("bertscore")
        print("Metrics loaded successfully.")

    def compute_row_level_metrics(self, predictions, references):
        """
        Computes metrics for a list of predictions vs references.
        Returns a dictionary of lists (one score per row) to add to the DataFrame.
        """
        # Storage for scores
        rouge_l_scores = []
        bleu_4_scores = []
        meteor_scores = []
        
        # 1. BERTScore (Efficient Batch Processing)
        # We run this first as it handles batches natively on GPU if available
        print("Computing BERTScore...")
        bert_results = self.bertscore.compute(
            predictions=predictions, 
            references=references, 
            lang="en",
            verbose=False
        )
        bert_f1_scores = bert_results['f1'] # List of F1 scores

        # 2. ROUGE, BLEU, METEOR (Loop for Row-Level Granularity)
        # We loop because standard library calls often return a single aggregated corpus score.
        print("Computing ROUGE, BLEU, and METEOR...")
        for pred, ref in zip(predictions, references):
            # Handle empty strings to prevent errors
            if not pred.strip(): pred = " "
            if not ref.strip(): ref = " "

            # ROUGE-L
            r_res = self.rouge.compute(predictions=[pred], references=[ref], rouge_types=['rougeL'])
            rouge_l_scores.append(r_res['rougeL'])

            # BLEU-4 (Standard implementation uses max_order=4)
            b_res = self.bleu.compute(predictions=[pred], references=[ref], max_order=4)
            bleu_4_scores.append(b_res['bleu'])

            # METEOR
            m_res = self.meteor.compute(predictions=[pred], references=[ref])
            meteor_scores.append(m_res['meteor'])

        return {
            "ROUGE-L": rouge_l_scores,
            "BLEU-4": bleu_4_scores,
            "METEOR": meteor_scores,
            "BERTScore_F1": bert_f1_scores
        }

# Initialize the Evaluator
evaluator = MetricsEvaluator()

# ==========================================
# 2. LOAD MODELS (SEPARATELY)
# ==========================================

print("Loading FINE-TUNED Model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

print("Loading BASE Model...")
base_model, _ = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(base_model)

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_role(label):
    if pd.isna(label): return "user"
    label_str = str(label).lower()
    for kw in AUTHORITY_KEYWORDS:
        if kw.lower() in label_str:
            return "user"
    return "assistant"

def generate_text(target_model, history_messages):
    inputs = tokenizer.apply_chat_template(
        history_messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = target_model.generate(
        input_ids = inputs,
        max_new_tokens = 128,
        use_cache = True,
        temperature = 0.6,
    )
    
    response = tokenizer.batch_decode(outputs)[0]
    
    try:
        return response.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    except:
        return response

# ==========================================
# 4. MAIN EVALUATION LOOP
# ==========================================

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"\nFound {len(csv_files)} files. Starting evaluation...\n")

for file_path in tqdm(csv_files):
    filename = os.path.basename(file_path)
    file_results = [] 

    try:
        df = pd.read_csv(file_path)
        if len(df) < 2:
            print(f"Skipping {filename} (too short)")
            continue
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    # Initialize History
    conversation_history = [{
        "role": "system",
        "content": "You are a suspect in a police interaction. You are agitated, non-compliant, and refuse to follow orders."
    }]

    # Iterate rows
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i+1]

        current_role = get_role(current_row['speaker_label'])
        next_role = get_role(next_row['speaker_label'])

        current_text = str(current_row['transcript'])
        next_text = str(next_row['transcript'])

        conversation_history.append({"role": current_role, "content": current_text})

        # Test Point: Officer -> Suspect
        if current_role == "user" and next_role == "assistant":
            
            # Generate Responses
            base_reply = generate_text(base_model, conversation_history)
            ft_reply = generate_text(model, conversation_history)

            file_results.append({
                "Turn_Index": i,
                "Officer_Input": current_text,
                "Real_Suspect_Response": next_text,
                "Base_Model_Response": base_reply,
                "Fine_Tuned_Response": ft_reply
            })

            # Update history with REAL response
            conversation_history.append({"role": "assistant", "content": next_text})

        elif current_role == "assistant":
             conversation_history.append({"role": "assistant", "content": current_text})

    # --- CALCULATE METRICS AND SAVE ---
    if file_results:
        result_df = pd.DataFrame(file_results)

        # Prepare lists for metric calculation
        refs = result_df["Real_Suspect_Response"].tolist()
        base_preds = result_df["Base_Model_Response"].tolist()
        ft_preds = result_df["Fine_Tuned_Response"].tolist()

        # Calculate metrics for Base Model
        print(f"Calculating metrics for Base Model ({filename})...")
        base_metrics = evaluator.compute_row_level_metrics(base_preds, refs)
        for key, values in base_metrics.items():
            result_df[f"Base_{key}"] = values

        # Calculate metrics for Fine-Tuned Model
        print(f"Calculating metrics for Fine-Tuned Model ({filename})...")
        ft_metrics = evaluator.compute_row_level_metrics(ft_preds, refs)
        for key, values in ft_metrics.items():
            result_df[f"FT_{key}"] = values

        # Save
        save_name = f"{os.path.splitext(filename)[0]}_evaluated.csv"
        save_path = os.path.join(output_folder, save_name)
        result_df.to_csv(save_path, index=False)

print(f"\nEvaluation Complete! Results saved in '{output_folder}'.")