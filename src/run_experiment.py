"""
Execution script for transcript evaluation.

Usage:
    python run_experiment.py <session_id> [--model <model_name>]

Example:
    python run_experiment.py 12345
"""

import argparse
from datetime import datetime
import json
import logging

from src.data_loader import load_transcript
from src.evaluator import   evaluate_transcript
from src.prompter import QUESTIONS
from src.config import RESULTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a teaching session transcript")
    parser.add_argument("session_id", help="Session ID (filename without extension)")
    parser.add_argument("--model", help="Override model name from config", default=None)
    args = parser.parse_args()

    try:
        transcript = load_transcript(args.session_id)
        logger.info(f"Loaded transcript for session {args.session_id}")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Use centralized questions (easy to extend in prompter.py)
    questions = QUESTIONS.copy()  # Copy to allow local modifications if needed

    try:
        logger.info("Calling LLM for structured evaluation...")
        output = evaluate_transcript(
            transcript=transcript,
            model_name=args.model,
        )
        
        result_dict = output.model_dump()
        logger.info("\n=== Evaluation Results ===\n")
        logger.info(json.dumps(result_dict, indent=2, ensure_ascii=False))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = RESULTS_DIR / f"results_{args.session_id}_{timestamp}.json"
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

    except Exception as e:
        logger.error(f"LLM evaluation failed: {e}")


if __name__ == "__main__":
    main()