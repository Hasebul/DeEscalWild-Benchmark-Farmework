checkpoint_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/qwen_25/checkpoint-118"
dataset_path = f"{base_path}qwen_finetune_data.jsonl"
output_dir = f"{base_path}qwen_25"

from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
checkpoint_path = "/content/drive/MyDrive/deescalation_project/Checkpoints/qwen_25/checkpoint-118" # Your trained LoRA adapters
base_model_name = "unsloth/Qwen2.5-3B-Instruct"
max_seq_length = 2048
dtype = None
load_in_4bit = True

# 1. Test Prompts
test_prompts = [
    "Officer: Step out of the car, please.",
    "Officer: You are currently trespassing on private property.",
    "Officer: Can I see your ID?",
    "Officer: Stop shouting, or I will have to detain you.",
    "Officer: Do you know why I pulled you over?",
]

# 2. Load the FINE-TUNED Model (Adapters included)
# We load this first. Unsloth will merge the base + checkpoint automatically.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

# 3. Load the BASE Model (Clean version for comparison)
# We name this specifically to avoid overwriting 'model'
base_model, _ = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(base_model)

# 4. Corrected Inference Function
# We removed .enable_adapters() to prevent the ValueError.
# Since 'model' has adapters and 'base_model' doesn't, we just pass the model object we want to use.
def generate_text(model_obj, prompt):
    messages = [
        {"role": "system", "content": "You are a suspect in a police interaction. You are agitated and non-compliant."},
        {"role": "user", "content": prompt}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    outputs = model_obj.generate(
        input_ids = inputs,
        max_new_tokens = 128,
        use_cache = True,
        temperature = 0.6,
    )

    response = tokenizer.batch_decode(outputs)[0]

    # Clean up the output to get only the assistant's reply
    try:
        return response.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    except:
        return response

# 5. Run Evaluation
results = []
print("Starting Evaluation...")

for prompt in tqdm(test_prompts):
    # Get response from the Base Model
    base_response = generate_text(base_model, prompt)

    # Get response from the Fine-Tuned Model
    ft_response = generate_text(model, prompt)

    results.append({
        "Input (Officer)": prompt,
        "Base Model Response": base_response,
        "Fine-Tuned Response": ft_response
    })

# 6. Save Results
df = pd.DataFrame(results)
df.to_csv("eval_comparison.csv", index=False)
print("\nEvaluation complete! Check 'eval_comparison.csv'")


# test all the scripts:

import pandas as pd
import glob
import os
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch

# --- CONFIGURATION ---
# Folder where your raw CSVs are (Upload your 20 files here)
input_folder = "/content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs"
# Folder where results will be saved
output_folder = "/content/drive/MyDrive/deescalation_project/Checkpoints/evaluated_results"

# Model Paths
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
# 1. LOAD MODELS (SEPARATELY)
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
# We load this into a separate variable 'base_model'
base_model, _ = FastLanguageModel.from_pretrained(
    model_name = base_model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(base_model)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_role(label):
    """Determines if speaker is User (Officer) or Assistant (Suspect)"""
    if pd.isna(label): return "user"
    label_str = str(label).lower()
    for kw in AUTHORITY_KEYWORDS:
        if kw.lower() in label_str:
            return "user"
    return "assistant" # The Suspect

def generate_text(target_model, history_messages):
    """Generates a response using the specific model object passed to it."""

    # Format the prompt using the chat template
    inputs = tokenizer.apply_chat_template(
        history_messages,
        tokenize = True,
        add_generation_prompt = True,
        return_tensors = "pt",
    ).to("cuda")

    # Generate
    outputs = target_model.generate(
        input_ids = inputs,
        max_new_tokens = 128,
        use_cache = True,
        temperature = 0.6,
    )

    # Decode
    response = tokenizer.batch_decode(outputs)[0]

    # Clean up to get only the assistant's reply
    try:
        return response.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
    except:
        return response


input_folder = "/content/drive/MyDrive/deescalation_project/Checkpoints/test_data_cvs"


#  ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================

csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
print(f"\nFound {len(csv_files)} files. Starting evaluation...\n")

for file_path in tqdm(csv_files):
    filename = os.path.basename(file_path)
    file_results = [] # Store results for THIS file only

    try:
        df = pd.read_csv(file_path)
        # Skip files that are too short to have a conversation
        if len(df) < 2:
            print(f"Skipping {filename} (too short)")
            continue
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        continue

    # Initialize History for this file
    conversation_history = [{
        "role": "system",
        "content": "You are a suspect in a police interaction. You are agitated, non-compliant, and refuse to follow orders."
    }]

    # Iterate through the transcript rows
    for i in range(len(df) - 1):
        current_row = df.iloc[i]
        next_row = df.iloc[i+1]

        current_role = get_role(current_row['speaker_label'])
        next_role = get_role(next_row['speaker_label'])

        current_text = str(current_row['transcript'])
        next_text = str(next_row['transcript'])

        # Add current line to history
        conversation_history.append({"role": current_role, "content": current_text})

        # TEST POINT: If Current is OFFICER and Next is SUSPECT
        if current_role == "user" and next_role == "assistant":

            # 1. Run Base Model
            base_reply = generate_text(base_model, conversation_history)

            # 2. Run Fine-Tuned Model
            ft_reply = generate_text(model, conversation_history)

            # Store Result
            file_results.append({
                "Turn_Index": i,
                "Officer_Input": current_text,
                "Real_Suspect_Response": next_text,
                "Base_Model_Response": base_reply,
                "Fine_Tuned_Response": ft_reply
            })

            # IMPORTANT: Append REAL response to history so context stays correct
            conversation_history.append({"role": "assistant", "content": next_text})

        # If it wasn't a test point (e.g. Officer -> Officer), just update history
        elif current_role == "assistant":
             conversation_history.append({"role": "assistant", "content": current_text})

    # SAVE RESULTS FOR THIS INDIVIDUAL FILE
    if file_results:
        result_df = pd.DataFrame(file_results)

        # Save as "188_evaluated.csv" inside the output folder
        save_name = f"{os.path.splitext(filename)[0]}_evaluated.csv"
        save_path = os.path.join(output_folder, save_name)

        result_df.to_csv(save_path, index=False)

print(f"\nEvaluation Complete! All individual files are in the '{output_folder}' folder.")

