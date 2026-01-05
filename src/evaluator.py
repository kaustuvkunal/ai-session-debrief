"""
Shared evaluation logic: Pydantic models, prompt building, and transcript evaluation function.
Used by both single-run scripts and batch evaluation.
"""

from typing import List, Dict, Optional, Tuple, Any

from pydantic import BaseModel, Field, field_validator

from src.prompter import SYSTEM_MESSAGE, QUESTIONS
from src.llm_client import generate_structured_response


class EvidenceSnippet(BaseModel):
    timestamp: str = Field(..., description="Timestamp in HH:MM:SS format")
    quote: str = Field(..., description="Exact or near-exact quote")

    @field_validator("timestamp")
    @classmethod
    def check_format(cls, v: str) -> str:
        parts = v.split(":")
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            raise ValueError("Timestamp must be in HH:MM:SS format")
        return v


class QuestionResponse(BaseModel):
    judgement_type: str = Field(..., pattern="^(Yes|No|Partially|Unclear)$")
    justification: str = Field(..., description="1–2 sentence justification")
    evidence: List[EvidenceSnippet] = Field(
        ..., description="2–4 snippets for Yes/No/Partially; 0–4 for Unclear"
    )


class SingleResult(BaseModel):
    question: str
    response: QuestionResponse


class EvaluationOutput(BaseModel):
    results: List[SingleResult]


def build_user_prompt(transcript: str, questions: Optional[List[Dict[str, str]]] = None) -> str:
    """Build the user prompt by injecting the transcript and formatted questions/instructions."""
    if questions is None:
        questions = QUESTIONS
    question_blocks = ""
    for i, q in enumerate(questions, start=1):
        question_blocks += f"Question {i}: {q['question']}\n"
        question_blocks += f"Evaluation Approach {i}: {q['instructions']}\n\n"

    return f"""Transcript:
{transcript}

Evaluation Questions and Approaches:
{question_blocks}

Analyze the transcript for each question using only its specific evaluation approach.
Provide the response strictly in the JSON format described in the system prompt.
"""


def evaluate_transcript(
    transcript: str,
    questions: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None
) -> EvaluationOutput:
    """Run the LLM evaluator on a transcript.

    Args:
        transcript: The session transcript to evaluate.
        questions: Optional list of evaluation questions. Defaults to QUESTIONS.
        model_name: Optional override for model name from config.

    Returns:
        EvaluationOutput: Parsed evaluation results.
    """
    if questions is None:
        questions = QUESTIONS
    
    user_prompt = build_user_prompt(transcript, questions)
    return generate_structured_response(
        system_prompt=SYSTEM_MESSAGE,
        user_prompt=user_prompt,
        response_model=EvaluationOutput,
        model_name=model_name,
    )