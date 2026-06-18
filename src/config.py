from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolo11s.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.5

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_INPUT_IMAGE = INPUT_DIR / "match.jpg"

DEVICE = "0" if torch.cuda.is_available() else "cpu"
