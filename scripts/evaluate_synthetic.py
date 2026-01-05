#!/usr/bin/env python3
"""Run evaluation over a synthetic JSONL dataset.

By default this script runs in simulation mode (no API calls). To call the LLM, set
`OPENAI_API_KEY` in the environment and pass `--use-llm`.

Usage:
    python scripts/evaluate_synthetic.py --input data/processed/synthetic_eval.jsonl
    python scripts/evaluate_synthetic.py --input data/processed/synthetic_eval.jsonl --use-llm --model o3-mini
"""
from pathlib import Path
import sys
import json
import argparse
import random
from collections import Counter

# Make src importable when running from repo root
repo_root = Path(__file__).resolve().parents[1]
# Add the project root so the `src` package can be imported as `src` from other modules
sys.path.insert(0, str(repo_root))

from src.prompter import SYSTEM_MESSAGE
from src.llm_client import generate_structured_response


def build_user_prompt(transcript: str, questions: list) -> str:
    question_blocks = ""
    for i, q in enumerate(questions, start=1):
        question_blocks += f"Question {i}: {q['question']}\n"
        question_blocks += f"Evaluation Approach {i}: {q.get('instructions','')}\n\n"

    return f"""Transcript:\n{transcript}\n\nEvaluation Questions and Approaches:\n{question_blocks}\nProvide the response strictly in the JSON format described in the system prompt."""


def simulate_response(example: dict) -> dict:
    results = []
    transcript = example.get("transcript", "")

    def pick_time():
        # Try to extract a timestamp pattern from transcript, otherwise fallback
        for line in transcript.splitlines():
            if line.startswith("00:"):
                return line.split()[0]
        return "00:00:10"

    for q in example.get("questions", []):
        judgement = random.choice(["Yes", "No", "Partially", "Unclear"])
        just = f"Auto-simulated judgement: {judgement}. Based on transcript excerpts."
        evidence = [
            {"timestamp": pick_time(), "quote": "Simulated quote from transcript."},
            {"timestamp": pick_time(), "quote": "Another simulated quote."},
        ] if judgement != "Unclear" else []

        results.append({
            "question": q.get("question", ""),
            "response": {
                "judgement_type": judgement,
                "justification": just,
                "evidence": evidence,
            },
        })

    return {"results": results}


def evaluate(input_path: Path, output_path: Path, use_llm: bool, model_name: str | None):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    with input_path.open("r", encoding="utf-8") as fh_in, output_path.open("w", encoding="utf-8") as fh_out:
        for line in fh_in:
            example = json.loads(line)
            session_id = example.get("session_id", "<no-id>")
            user_prompt = build_user_prompt(example.get("transcript", ""), example.get("questions", []))

            if use_llm:
                # Import EvaluationOutput only when using the LLM to avoid heavy imports at module load time
                try:
                    from src.run_mlflow_experiment import EvaluationOutput
                except Exception as e:
                    result = {"error": f"Failed to import EvaluationOutput: {e}. Ensure src is importable and dependencies are installed."}
                else:
                    try:
                        output_obj = generate_structured_response(
                            system_prompt=SYSTEM_MESSAGE,
                            user_prompt=user_prompt,
                            response_model=EvaluationOutput,
                            model_name=model_name,
                        )
                        result = output_obj.model_dump()
                    except Exception as e:
                        result = {"error": str(e)}
            else:
                result = simulate_response(example)

            # Tally simple counts
            for r in result.get("results", []):
                counts[r["response"]["judgement_type"]] += 1

            out_record = {"session_id": session_id, "result": result}
            fh_out.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"Wrote results to {output_path}")
    print("Summary counts:", dict(counts))


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on synthetic dataset")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/synthetic_eval_results.jsonl"),
    )
    parser.add_argument("--use-llm", action="store_true", help="Call LLM instead of simulating")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    evaluate(args.input, args.output, args.use_llm, args.model)


if __name__ == "__main__":
    main()
