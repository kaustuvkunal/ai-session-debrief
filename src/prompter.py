from typing import List, Dict

SYSTEM_MESSAGE= """You are an expert teaching observer.
You will receive:
    - A timestamped transcript of an online teaching session (mentor–learner interaction).
    - One or more evaluation questions.
    - An evaluation approach for each question.

Your task:
    - Read the entire transcript carefully.
    - Analyze the transcript according to the evaluation approach.
    - For each question, provide:
        - judgement_type: "Yes", "No", or "Unclear" (use Unclear if evidence is insufficient for a definitive answer).
        - justification: 1–2 sentences explaining the judgment with concrete references to the transcript.
        - evidence: 2–4 short evidence snippets for Yes/No answers; 0–4 relevant snippets for Unclear (or none if truly absent).
- Evidence snippets must include exact or near-exact quotes with timestamps.
        
Important rules:
- Use only information present in the transcript. Do not infer facts that are not explicitly stated or clearly implied.
- Output only valid JSON matching this structure, with no additional text or commentary::
{
    "results": [
        {
            "question": "string",
            "response": {
                "judgement_type": "Yes | No | Unclear",
                "justification": "string (1–2 sentences)",
                "evidence": [
                    {
                        "timestamp": "HH:MM:SS",
                        "quote": "string"
                    },
                    {
                        "timestamp": "HH:MM:SS",
                        "quote": "string"
                    }
                ]
            }
        }
    ]
}
Do not include any additional commentary or text outside the JSON.
"""

USER_MESSAGE= """Transcript:
{transcript}
Evaluation Questions:
{questions}
Evaluation Approach:
{instructions}

"""


QUESTIONS: List[Dict[str, str]] = [
    {
        "question": "Did mentor encourage learners to ask questions/doubts in the session?",
        "instructions": (
            "Look for mentor statements actively encouraging questions (e.g., 'Any doubts?', 'Feel free to ask', "
            "'Please interrupt me'). List specific instances with timestamps. "
            "If found → Yes; if none and session has interaction → No; otherwise Unclear."
        ),
    },
    {
    "question": "Was there any learner who hijacked the session by asking too many questions?",
    "instructions": (
        "Analyze questions asked by each learner. "
        "Identify whether any learner dominated the Q&A by: "
        "- Asking a disproportionate share (>40% of all questions), or "
        "- Asking multiple consecutive questions that caused major detours or disrupted the flow. "
        "Provide timestamps and quotes for such instances. "
        "If such a learner is found → Yes; if not → No; if evidence is insufficient → Unclear."
        ),
    },
    {
    "question": "Were there instances where the mentor struggled to find information or answer questions due to a lack of preparation?",
    "instructions": (
        "Review the transcript for signs that the mentor struggled to answer, such as: "
        "'let me check', 'I'm not sure', 'I don't have that information', "
        "'let me get back to you on that', long pauses before responding, or incomplete answers. "
        "Identify and list such moments with specific quotes and timestamps. "
        "Answer Yes if such instances are clearly present, No if the mentor consistently answers confidently, "
        "otherwise Unclear."
        ),
    },
    {
    "question": "Did the mentor dedicate enough time for Q&A?",
    "instructions": (
        "Use timestamps to identify any Q&A segments and estimate their total duration "
        "relative to the overall session length. "
        "Assess whether Q&A felt sufficient (e.g -\"Now let's open for questions\") or rushed by looking for signals such as: "
        "mentions of time constraints, moving on quickly to the next topic, or stating there is no time for more questions. "
        "Provide specific quotes and timestamps illustrating both adequate and rushed Q&A behavior, if present. "
        "Classify the outcome as Yes, No, or Partially, and justify with estimated Q&A duration and clear reasoning. "
        "If the transcript does not provide enough evidence to judge, answer Unclear."
       ),
    },
    {
    "question": "Was the mentor able to answer learners' questions confidently and comprehensively?",
    "instructions": (
        "Focus only on learner questions and the mentor's responses in the transcript. "
        "For each learner question, list: the timestamp, speaker, the mentor's response, and a brief assessment of whether the answer was confident, complete and comprehensive "
        "When assessing confidence and completeness, look for: clear and structured explanations, use of examples or analogies, and absence of hesitation or uncertainty in the mentor's replies. "
        "After reviewing all Q&A exchanges, provide:"
        "- An overall label: Yes / No / Partially / Unclear, and "
        "- A short justification summarizing key evidence and representative timestamps."
      ),
    },

    

    # Add more questions here as needed
]
