"""
generate_all_model_conversations.py

Generate full conversations for:
- 7 base models
- 7 fine-tuned models
- Gemini 2.5 Flash

Input:
    JSON file containing scenario_context, character_profile, and police_dialogue.

Output:
    responses/
    ├── qwen_base/
    ├── qwen_finetuned/
    ├── llama_base/
    ├── llama_finetuned/
    ├── ...
    └── gemini_2_5_flash/

Install:
    pip install pandas tqdm torch transformers accelerate bitsandbytes unsloth google-genai

For Gemini:
    export GEMINI_API_KEY="your_api_key_here"

Run:
    python generate_all_model_conversations.py \
        --input_json scenarios.json \
        --output_dir responses \
        --load_in_4bit
"""

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm


# ============================================================
# 1. MODEL CONFIGURATION
# ============================================================

@dataclass
class ModelSpec:
    model_name: str
    family: str
    model_type: str      # base, fine_tuned, or api
    backend: str         # local_unsloth or gemini
    model_path: str


"""
IMPORTANT:
Replace CHANGE_ME paths with your actual Hugging Face model names
or fine-tuned checkpoint paths.
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
# 2. PROMPT SETTINGS
# ============================================================

SYSTEM_PROMPT_TEMPLATE = """
You are roleplaying as the civilian/suspect in a police de-escalation scenario.

Your job:
- Respond only as the civilian/suspect.
- Do not write the officer's dialogue.
- Stay consistent with the character profile.
- Stay consistent with the scenario context.
- Respond naturally, emotionally, and realistically.
- Do not over-explain.
- Do not sound like an AI assistant.
- Use short, human-like responses when appropriate.
- If the character is angry, afraid, confused, defensive, or hesitant, show that naturally.

Scenario Context:
{scenario_context}

Character Profile:
{character_profile}
""".strip()


# ============================================================
# 3. JSON HELPERS
# ============================================================

def load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "cases" in data:
            return data["cases"]
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError("Input JSON must be either a list of cases or a dict.")


def get_case_id(case: Dict[str, Any], index: int) -> str:
    case_id = case.get("case_id") or case.get("id") or f"case_{index + 1:04d}"
    return str(case_id)


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, indent=2, ensure_ascii=False)


def extract_scenario_context(case: Dict[str, Any]) -> str:
    return to_text(
        case.get("scenario_context")
        or case.get("scenario")
        or case.get("context")
        or ""
    )


def extract_character_profile(case: Dict[str, Any]) -> str:
    return to_text(
        case.get("character_profile")
        or case.get("profile")
        or case.get("civilian_profile")
        or case.get("suspect_profile")
        or ""
    )


def extract_police_dialogue(case: Dict[str, Any]) -> List[str]:
    """
    Supports these possible keys:
    - police_dialogue
    - police_dialog
    - officer_dialogue
    - officer_turns

    The value can be:
    - list[str]
    - list[dict], where each dict has text/transcript/content/dialogue
    """
    raw_dialogue = (
        case.get("police_dialogue")
        or case.get("police_dialog")
        or case.get("officer_dialogue")
        or case.get("officer_turns")
        or []
    )

    if not isinstance(raw_dialogue, list):
        raise ValueError("police_dialogue must be a list.")

    police_turns = []

    for item in raw_dialogue:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("transcript")
                or item.get("content")
                or item.get("dialogue")
                or item.get("utterance")
                or ""
            )
            text = str(text).strip()
        else:
            text = str(item).strip()

        if text:
            police_turns.append(text)

    return police_turns


def make_system_prompt(
    scenario_context: str,
    character_profile: str,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        scenario_context=scenario_context,
        character_profile=character_profile,
    )


# ============================================================
# 4. TEXT CLEANING
# ============================================================

def clean_generated_reply(text: str) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    text = re.sub(r"^```.*?\n", "", text)
    text = text.replace("```", "").strip()

    stop_markers = [
        "<|im_end|>",
        "<|endoftext|>",
        "</s>",
        "Officer:",
        "Police:",
        "Deputy:",
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
        "Character:",
        "Response:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    return text.strip()


def build_plain_prompt(history_messages: List[Dict[str, str]]) -> str:
    """
    Used for Gemini and for local models without chat templates.
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


# ============================================================
# 5. GENERATOR CLASSES
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
        max_new_tokens: int = 160,
        temperature: float = 0.7,
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
                    repetition_penalty=1.08,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][input_ids.shape[-1]:]
            decoded = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

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
            raise EnvironmentError(
                "Missing GEMINI_API_KEY. Set it before running Gemini generation."
            )

        self.client = genai.Client(api_key=api_key)
        self.types = types
        self.spec = spec

    def generate(
        self,
        history_messages: List[Dict[str, str]],
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        try:
            system_instruction = ""

            if history_messages and history_messages[0]["role"] == "system":
                system_instruction = history_messages[0]["content"]

            prompt_messages = [
                msg for msg in history_messages
                if msg["role"] != "system"
            ]

            prompt = build_plain_prompt(prompt_messages)

            response = self.client.models.generate_content(
                model=self.spec.model_path,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    system_instruction=system_instruction,
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
# 6. CONVERSATION GENERATION
# ============================================================

def generate_conversation_for_case(
    generator,
    spec: ModelSpec,
    case: Dict[str, Any],
    case_index: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    case_id = get_case_id(case, case_index)

    scenario_context = extract_scenario_context(case)
    character_profile = extract_character_profile(case)
    police_turns = extract_police_dialogue(case)

    if not police_turns:
        raise ValueError(f"No police dialogue found for case_id={case_id}")

    system_prompt = make_system_prompt(
        scenario_context=scenario_context,
        character_profile=character_profile,
    )

    history_messages = [
        {
            "role": "system",
            "content": system_prompt,
        }
    ]

    generated_conversation = []

    for turn_index, police_text in enumerate(police_turns):
        history_messages.append(
            {
                "role": "user",
                "content": police_text,
            }
        )

        civilian_response = generator.generate(
            history_messages=history_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        history_messages.append(
            {
                "role": "assistant",
                "content": civilian_response,
            }
        )

        generated_conversation.append(
            {
                "turn_index": turn_index,
                "police": police_text,
                "civilian_generated": civilian_response,
            }
        )

    return {
        "case_id": case_id,
        "model_name": spec.model_name,
        "model_family": spec.family,
        "model_type": spec.model_type,
        "backend": spec.backend,
        "model_path": spec.model_path,
        "scenario_context": scenario_context,
        "character_profile": character_profile,
        "generated_conversation": generated_conversation,
        "conversation_transcript": build_transcript(generated_conversation),
    }


def build_transcript(conversation_turns: List[Dict[str, Any]]) -> str:
    lines = []

    for turn in conversation_turns:
        lines.append(f"Officer: {turn['police']}")
        lines.append(f"Civilian: {turn['civilian_generated']}")

    return "\n".join(lines)


def save_case_output(
    result: Dict[str, Any],
    output_dir: Path,
    model_name: str,
):
    model_dir = output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    case_id = result["case_id"]
    json_path = model_dir / f"{case_id}.json"
    txt_path = model_dir / f"{case_id}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write(result["conversation_transcript"])

    return json_path, txt_path


# ============================================================
# 7. MAIN MODEL LOOP
# ============================================================

def run_generation(
    input_json: Path,
    output_dir: Path,
    selected_models: Optional[List[str]],
    max_seq_length: int,
    load_in_4bit: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_json(input_json)

    if selected_models:
        specs = [
            spec for spec in MODEL_SPECS
            if spec.model_name in selected_models
        ]
    else:
        specs = MODEL_SPECS

    if not specs:
        raise ValueError("No valid models selected.")

    print(f"Loaded {len(cases)} cases.")
    print(f"Selected {len(specs)} models.")

    for spec in specs:
        if spec.backend == "local_unsloth" and spec.model_path.startswith("CHANGE_ME"):
            print(f"[SKIP] {spec.model_name}: model_path is still CHANGE_ME.")
            continue

        print("\n" + "=" * 80)
        print(f"Generating responses for model: {spec.model_name}")
        print("=" * 80)

        generator = None

        try:
            generator = create_generator(
                spec=spec,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
            )

            for case_index, case in enumerate(tqdm(cases, desc=spec.model_name)):
                try:
                    result = generate_conversation_for_case(
                        generator=generator,
                        spec=spec,
                        case=case,
                        case_index=case_index,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    json_path, txt_path = save_case_output(
                        result=result,
                        output_dir=output_dir,
                        model_name=spec.model_name,
                    )

                except Exception as exc:
                    case_id = get_case_id(case, case_index)

                    error_result = {
                        "case_id": case_id,
                        "model_name": spec.model_name,
                        "error": str(exc),
                    }

                    model_dir = output_dir / spec.model_name
                    model_dir.mkdir(parents=True, exist_ok=True)

                    error_path = model_dir / f"{case_id}_ERROR.json"

                    with error_path.open("w", encoding="utf-8") as f:
                        json.dump(error_result, f, indent=2, ensure_ascii=False)

                    print(f"[ERROR] case_id={case_id}, model={spec.model_name}: {exc}")

        except Exception as exc:
            print(f"[MODEL ERROR] {spec.model_name}: {exc}")

        finally:
            if generator is not None:
                generator.unload()

            gc.collect()

    print("\nGeneration complete.")
    print(f"Outputs saved to: {output_dir}")


# ============================================================
# 8. ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        required=True,
        help="Path to input JSON file containing scenarios.",
    )

    parser.add_argument(
        "--output_dir",
        default="responses",
        help="Folder where model response folders will be saved.",
    )

    parser.add_argument(
        "--only_models",
        nargs="*",
        default=None,
        help="Optional list of model names to run.",
    )

    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load local models in 4-bit mode.",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=160,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_generation(
        input_json=Path(args.input_json),
        output_dir=Path(args.output_dir),
        selected_models=args.only_models,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )


if __name__ == "__main__":
    main()