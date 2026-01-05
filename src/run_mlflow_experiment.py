"""
Execution script for transcript evaluation with MLflow tracking.

Usage:
    python src/run_mlflow_experiment.py <session_id> [--model <model_name>]

Example:
    python src/run_mlflow_experiment.py 825870f7-b890-4a35-91a4-4cb016af792a
"""
import logging
import json
import argparse
from datetime import datetime
from pathlib import Path

import mlflow
from pydantic import Field, ValidationError, field_validator

from src.config import RESULTS_DIR, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_NAME
from src.data_loader import load_transcript
from src.prompter import QUESTIONS, SYSTEM_MESSAGE
from src.evaluator import EvaluationOutput, build_user_prompt, evaluate_transcript

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a teaching session transcript")
    parser.add_argument("session_id", help="Session ID (filename without extension)")
    parser.add_argument("--model", help="Override model name from config", default=None)
    args = parser.parse_args()

    try:
        transcript = load_transcript(args.session_id)
        logger.info(f"Loaded transcript for session {args.session_id}")
    except FileNotFoundError as e:
        logger.error(f"Transcript not found: {e}")
        return

    questions = QUESTIONS.copy()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.session_id}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("task", "transcript_evaluation")
        mlflow.log_param("session_id", args.session_id)
        
        effective_model = args.model or MODEL_NAME
        mlflow.log_param("model_name", effective_model)
        
        # Log transcript length for reference
        mlflow.log_metric("transcript_length_chars", len(transcript))

        # Log prompts for reproducibility and audit trail
        user_prompt = build_user_prompt(transcript, QUESTIONS)
        mlflow.log_text(SYSTEM_MESSAGE, "system_prompt.txt")
        mlflow.log_text(user_prompt, "user_prompt.txt")

        # TODO later : register prompt

        # Log questions as structured artifact
        questions_dict = {f"question_{i}": q for i, q in enumerate(questions, start=1)}
        mlflow.log_dict(questions_dict, "questions.json")

        try:
            logger.info("Calling LLM for structured evaluation...")
            output = evaluate_transcript(
                transcript=transcript,
                model_name=args.model,
            )

            result_dict = output.model_dump()
            logger.info("\n=== Evaluation Results ===\n")
            logger.info(json.dumps(result_dict, indent=2, ensure_ascii=False))

            # Log results to MLflow
            mlflow.log_dict(result_dict, "evaluation_results.json")

            # Save results to file
            Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
            output_path = Path(RESULTS_DIR) / f"results_{args.session_id}_{timestamp}.json"
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to {output_path}")
            
            # Log artifact to MLflow
            mlflow.log_artifact(str(output_path))

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            mlflow.log_param("error_type", type(e).__name__)
            mlflow.log_text(str(e), "error.txt")
            mlflow.set_tag("status", "failed")


if __name__ == "__main__":
    main()