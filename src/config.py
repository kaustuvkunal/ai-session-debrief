import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_TRANSCRIPTS_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH_DIR = PROJECT_ROOT / "data" / "ground_truths"
GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)


# LLM 
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
MODEL_NAME = OPENAI_MODEL
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "medium")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1024"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#mlflow specific
MLFLOW_TRACKING_URI="http://127.0.0.1:5001/"
MLFLOW_EXPERIMENT_NAME="sesion_video_debriefing"
MLFLOW_ARTIFACT_ROOT="./mlruns/artifacts/"