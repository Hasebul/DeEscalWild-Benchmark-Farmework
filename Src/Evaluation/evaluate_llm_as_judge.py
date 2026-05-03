"""
evaluate_llm_as_judge.py

LLM-as-judge evaluation for:
1. Realism
2. De-escalation

Candidate models:
- qwen_base, qwen_finetuned
- llama_base, llama_finetuned
- phi_base, phi_finetuned
- mistral_base, mistral_finetuned
- granite_base, granite_finetuned
- gemma_base, gemma_finetuned
- falcon_base, falcon_finetuned
- gemini_2_5_flash

Install:
    pip install google-genai pandas tqdm

Run:
    export GEMINI_API_KEY="your_api_key"
    python evaluate_llm_as_judge.py \
        --input evaluation_cases.json \
        --output_dir results \
        --judge_model gemini-2.5-flash
"""

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

from google import genai
from google.genai import types


MODEL_NAMES = [
    "qwen_base",
    "qwen_finetuned",
    "llama_base",
    "llama_finetuned",
    "phi_base",
    "phi_finetuned",
    "mistral_base",
    "mistral_finetuned",
    "granite_base",
    "granite_finetuned",
    "gemma_base",
    "gemma_finetuned",
    "falcon_base",
    "falcon_finetuned",
    "gemini_2_5_flash",
]


PROMPT_FOR_REALISM_EVALUATION = """
[SYSTEM ROLE]
You are an expert Forensic Linguist and Police Communications Evaluator. Your task is to
measure the Realism Gap between a simulated civilian response and a real-world reference in
a de-escalation context. All judgments must be grounded in direct textual evidence from the
provided transcripts.

[EVALUATION RUBRIC]
Begin at 100. Apply the following deductions based on observed evidence in the Target
transcript only.

Criterion Deduction Trigger
Emotional Flow −10 to −30 Unrealistic compliance or binary mood swings
without a verbal trigger.
Linguistic AI-isms −15 Overly formal grammar or structured explanations implausible under stress.
Profile Adherence −20 Vocabulary or stance deviates from the Character Profile.
Contextual Logic −20 Ignores physical cues or prior utterances in the Scenario Context.

[SCORING TIERS]
- [85-100] High Realism: Indistinguishable from baseline.
- [51-84] Marginal: Plausible but slightly stiff or predictable.
- [0-50] Artificial: Robotic, over-compliant, or emotionally inconsistent.

[INPUT DATA]
Scenario Context: {scenario_context}
Character Profile: {profile}
Ground Truth (Real-World Reference): {ground_truth}
Target (Response to Evaluate): {target_conversation}

[TASK REQUIREMENTS]
1. Compare the Target against the Ground Truth on each of the four penalty dimensions.
2. Record a specific textual observation for each dimension.
3. Sum justified deductions and subtract from 100 to produce the Pinpoint Score. Score must
not fall below 0.
4. Assign look_Real = 1 if Score >= 85, else 0.
5. Do not apply deductions not directly evidenced in the Target transcript.

[OUTPUT RULES]
1. Return ONLY a valid raw JSON object.
2. No preamble, filler text, or markdown formatting.
3. Do not wrap output in code blocks or backticks.
4. Use EXACTLY the following key structure:
{
  "analysis": {
    "emotional_consistency": "Observation...",
    "linguistic_style": "Observation...",
    "profile_adherence": "Observation...",
    "contextual_logic": "Observation...",
    "deductions_applied": "Itemized list"
  },
  "look_Real": 0,
  "Score": 0
}
"""


PROMPT_FOR_DEESCALATING_EVALUATION = """
[SYSTEM ROLE]
You are an expert crisis intervention analyst and behavioral evaluator. Your task is to analyze
the provided interaction transcript and assign a single De-escalation Score on a scale of
0 to 100. Rather than aggregating individual skill scores, this evaluation asks one central
question: what was the final measurable outcome of the officer’s intervention? Read the
transcript carefully, determine the base score using the Outcome Rubric below, then apply
any warranted penalties for defective conversational behavior.

[OUTCOME RUBRIC — BASE SCORE RANGES]
Outcome Range Indicators
1. Escalation 0 – 25 Agitation, anger, or anxiety visibly increased; interaction ended in
physical aggression or restraint; new threats emerged; complete loss of control requiring
emergency intervention.
2. Containment 26 – 50 Crisis paused but not resolved; emotional state unchanged; officer
disengages but the problem will likely resurface; situation left in unresolved tension.
3. De-escalation 51 – 75 Emotional intensity measurably lowered; verbal threats ceased and
speech pace normalized; immediate risk of harm eliminated; subject able to engage rationally.
4. Resolution 76 – 100 All De-escalation indicators met; voluntary agreement reached; trust
and rapport established; core need addressed with a constructive path forward.

[PENALTIES FOR DEFECTIVE BEHAVIOR]
After assigning the base score, evaluate the officer’s conversational behavior for the following
failure modes. Apply deductions where directly evidenced. The minimum final score is 0.

Failure Mode Deduction Trigger Condition
Repetitive Looping −15 Officer repeats the same phrases, questions, or instructions across
multiple turns without adapting to the subject’s responses or advancing the interaction.
Unrealistic Dialogue −20 Officer produces logically incoherent, contextually inappropriate,
or clearly robotic responses that would be implausible in a real-world intervention.

[SCORING INSTRUCTIONS]
1. Identify the highest outcome level achieved and assign a Base Score within the corresponding range.
2. Apply any warranted penalty deductions. Each deduction must be justified by direct transcript evidence.
3. Compute the Final Score as Base Score minus total penalties, with a minimum value of 0.
4. Do not apply penalties that are not directly evidenced in the transcript.

[INPUT]
Transcript: {conversation_transcript}

[OUTPUT RULES]
1. Return ONLY a valid raw JSON object.
2. No preamble, filler text, or markdown formatting.
3. Do not wrap output in code blocks or backticks.
4. Use EXACTLY the following key structure:
{
  "reasoning": "2-3 sentence justification of the base outcome category.",
  "outcome_category": "Escalation | Containment | De-escalation | Resolution",
  "base_score": 0,
  "penalty_applied": "None | Repetitive Looping | Unrealistic Dialogue | Both",
  "penalty_reasoning": "Justification, or N/A if no penalty.",
  "final_score": 0
}
"""


def to_text(value: Any) -> str:
    """
    Converts dict/list/string data into readable text for the judge prompt.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)


def render_prompt(template: str, values: Dict[str, Any]) -> str:
    """
    Safer than str.format because the prompt itself contains JSON braces.
    """
    prompt = template
    for key, value in values.items():
        prompt = prompt.replace("{" + key + "}", to_text(value))
    return prompt


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """
    Parses JSON even if the model accidentally wraps it in text or code fences.
    """
    if raw_text is None:
        raise ValueError("Empty judge response.")

    text = raw_text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Could not find JSON object in response:\n{text}")

        candidate = text[start : end + 1]
        return json.loads(candidate)


class GeminiJudge:
    def __init__(
        self,
        model_name: str = "gemini-3.1-pro",
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing GEMINI_API_KEY environment variable.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def judge(self, prompt: str) -> Dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
            ),
        )

        return parse_json_response(response.text)


def judge_with_retries(
    judge: GeminiJudge,
    prompt: str,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> Dict[str, Any]:
    last_error = None

    for attempt in range(retries):
        try:
            return judge.judge(prompt)
        except Exception as exc:
            last_error = exc
            time.sleep(sleep_seconds * (attempt + 1))

    raise RuntimeError(f"Judge failed after {retries} attempts: {last_error}")


def evaluate_one_model_output(
    judge: GeminiJudge,
    case: Dict[str, Any],
    model_name: str,
    model_output: Dict[str, Any],
) -> Dict[str, Any]:
    scenario_context = case.get("scenario_context", "")
    profile = case.get("profile", "")
    ground_truth = case.get("ground_truth", "")

    target_conversation = model_output.get("target_conversation", "")
    conversation_transcript = model_output.get(
        "conversation_transcript",
        target_conversation,
    )

    realism_prompt = render_prompt(
        PROMPT_FOR_REALISM_EVALUATION,
        {
            "scenario_context": scenario_context,
            "profile": profile,
            "ground_truth": ground_truth,
            "target_conversation": target_conversation,
        },
    )

    deescalation_prompt = render_prompt(
        PROMPT_FOR_DEESCALATING_EVALUATION,
        {
            "conversation_transcript": conversation_transcript,
        },
    )

    realism_result = judge_with_retries(judge, realism_prompt)
    deescalation_result = judge_with_retries(judge, deescalation_prompt)

    return {
        "case_id": case.get("case_id", ""),
        "model_name": model_name,

        "realism_score": realism_result.get("Score"),
        "look_real": realism_result.get("look_Real"),
        "realism_result": realism_result,

        "deescalation_final_score": deescalation_result.get("final_score"),
        "deescalation_base_score": deescalation_result.get("base_score"),
        "deescalation_outcome_category": deescalation_result.get("outcome_category"),
        "deescalation_penalty_applied": deescalation_result.get("penalty_applied"),
        "deescalation_result": deescalation_result,
    }


def evaluate_dataset(
    input_path: Path,
    output_dir: Path,
    judge_model: str,
    selected_models: List[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        cases = json.load(f)

    judge = GeminiJudge(model_name=judge_model)

    jsonl_path = output_dir / "llm_as_judge_results.jsonl"
    csv_path = output_dir / "llm_as_judge_scores.csv"

    all_rows = []

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for case in tqdm(cases, desc="Evaluating cases"):
            case_id = case.get("case_id", "")

            model_outputs = case.get("model_outputs", {})

            for model_name in selected_models:
                if model_name not in model_outputs:
                    print(f"[SKIP] case_id={case_id}, missing model={model_name}")
                    continue

                try:
                    row = evaluate_one_model_output(
                        judge=judge,
                        case=case,
                        model_name=model_name,
                        model_output=model_outputs[model_name],
                    )

                    all_rows.append(row)

                    jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

                except Exception as exc:
                    error_row = {
                        "case_id": case_id,
                        "model_name": model_name,
                        "error": str(exc),
                    }

                    all_rows.append(error_row)
                    jsonl_file.write(json.dumps(error_row, ensure_ascii=False) + "\n")
                    jsonl_file.flush()

    flat_rows = []

    for row in all_rows:
        flat_rows.append(
            {
                "case_id": row.get("case_id"),
                "model_name": row.get("model_name"),

                "realism_score": row.get("realism_score"),
                "look_real": row.get("look_real"),

                "deescalation_final_score": row.get("deescalation_final_score"),
                "deescalation_base_score": row.get("deescalation_base_score"),
                "deescalation_outcome_category": row.get("deescalation_outcome_category"),
                "deescalation_penalty_applied": row.get("deescalation_penalty_applied"),

                "error": row.get("error"),
            }
        )

    df = pd.DataFrame(flat_rows)
    df.to_csv(csv_path, index=False)

    print(f"\nSaved detailed JSONL results to: {jsonl_path}")
    print(f"Saved score CSV to: {csv_path}")

    if not df.empty:
        print("\nAverage scores by model:")
        summary = (
            df.groupby("model_name", dropna=False)
            .agg(
                avg_realism_score=("realism_score", "mean"),
                avg_look_real=("look_real", "mean"),
                avg_deescalation_score=("deescalation_final_score", "mean"),
                n_cases=("case_id", "count"),
            )
            .reset_index()
            .sort_values(
                by=["avg_realism_score", "avg_deescalation_score"],
                ascending=False,
            )
        )

        print(summary.to_string(index=False))

        summary_path = output_dir / "llm_as_judge_model_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nSaved model summary to: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to evaluation_cases.json",
    )

    parser.add_argument(
        "--output_dir",
        default="results",
        help="Directory where evaluation outputs will be saved.",
    )

    parser.add_argument(
        "--judge_model",
        default="gemini-2.5-flash",
        help="Judge model name. Default: gemini-2.5-flash",
    )

    parser.add_argument(
        "--models",
        nargs="*",
        default=MODEL_NAMES,
        help="Optional subset of model names to evaluate.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    evaluate_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        judge_model=args.judge_model,
        selected_models=args.models,
    )