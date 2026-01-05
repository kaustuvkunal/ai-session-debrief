import os
import pandas as pd
from src.config import DATA_TRANSCRIPTS_DIR as DATA_TRANSCRIPTS
import logging

logging.basicConfig(level=logging.INFO)

def load_transcript(session_id: str) -> str:
    logging.info(f"Loading transcript for session_id: {session_id}")
    file_path = os.path.join(DATA_TRANSCRIPTS, f"{session_id}.transcript")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Transcript not found: {file_path}")
    with open(file_path, "r") as f:
        return f.read()