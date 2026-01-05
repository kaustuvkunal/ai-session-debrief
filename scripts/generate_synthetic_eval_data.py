#!/usr/bin/env python3
"""Generate a small synthetic JSONL dataset for evaluation.

Writes to `data/processed/synthetic_eval.jsonl` by default.

Usage:
    python scripts/generate_synthetic_eval_data.py --count 30
"""
from pathlib import Path
import json
import random
import argparse
import uuid


TRANSCRIPTS = [
    # Mentor encourages questions
    (
        "00:00:05 Mentor: Welcome everyone. Feel free to interrupt me with questions.\n"
        "00:02:10 Learner1: Could you explain that again?\n"
        "00:02:20 Mentor: Sure, let me clarify..."
    ),
    # Mentor uncertain
    (
        "00:00:10 Mentor: Today we'll cover X.\n"
        "00:15:30 Learner2: How does X integrate with Y?\n"
        "00:15:35 Mentor: I'm not sure off the top of my head, let me check.\n"
    ),
    # Dominant learner
    (
        "00:00:05 Mentor: Let's start.\n"
        "00:05:12 Learner3: Question A?\n"
        "00:05:30 Learner3: Question B?\n"
        "00:06:00 Learner3: Also Question C?\n"
        "00:10:00 Mentor: OK, moving on...\n"
    ),
    # Short, unclear
    (
        "00:00:01 Mentor: Hi.\n"
        "00:00:30 Learner1: Hello.\n"
        "00:01:00 Mentor: Let's finish.\n"
    ),
]


def build_questions_template():
    # Minimal question template compatible with prompter. We keep instructions short.
    return [
        {
            "question": "Did mentor encourage learners to ask questions/doubts in the session?",
            "instructions": (
                "Look for mentor statements actively encouraging questions (e.g., 'Any doubts?', 'Feel free to ask')."
            ),
        },
        {
            "question": "Was there any learner who hijacked the session by asking too many questions?",
            "instructions": (
                "Identify if any learner dominated the Q&A by asking many questions or consecutive questions."
            ),
        },
    ]


def generate_example(i: int) -> dict:
    transcript = random.choice(TRANSCRIPTS)
    return {
        "session_id": str(uuid.uuid4()),
        "transcript": transcript,
        "questions": build_questions_template(),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation dataset (JSONL)")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthetic_eval.jsonl"),
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for i in range(args.count):
            ex = generate_example(i)
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {args.count} examples to {args.output}")


if __name__ == "__main__":
    main()
