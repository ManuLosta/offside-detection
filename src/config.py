from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_NAME = "yolo11s.pt"
PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.3

YOLO_POSE_MODEL = "yolo11m-pose.pt"
POSE_IMAGE_SIZE = 640
POSE_CONFIDENCE_THRESHOLD = 0.10
POSE_CROP_PADDING = 24

INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DETECTIONS_OUTPUT_DIR = OUTPUT_DIR / "detections"
POSES_OUTPUT_DIR = OUTPUT_DIR / "poses"
DEFAULT_INPUT_IMAGE = INPUT_DIR / "match.jpg"

DEVICE = "0" if torch.cuda.is_available() else "cpu"
