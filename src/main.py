import argparse
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_INPUT_IMAGE, OUTPUT_DIR
from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run football analytics stages on a match image: players, teams, "
            "poses and camera vanishing point."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=DEFAULT_INPUT_IMAGE,
        help="Path to the input football match image.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Root directory where detection and pose outputs are saved.",
    )
    parser.add_argument(
        "--detect",
        action="store_true",
        help="Run player detection only.",
    )
    parser.add_argument(
        "--teams",
        action="store_true",
        help="Run team classification only (requires existing detection output).",
    )
    parser.add_argument(
        "--poses",
        action="store_true",
        help="Run pose estimation only on classified players (requires existing team output).",
    )
    parser.add_argument(
        "--vanishing-point",
        action="store_true",
        help="Run vanishing point estimation only.",
    )
    return parser.parse_args()


def _resolve_stages(args: argparse.Namespace) -> tuple[bool, bool, bool, bool]:
    if args.detect:
        return (True, False, False, False)
    if args.teams:
        return (False, True, False, False)
    if args.poses:
        return (False, False, True, False)
    if args.vanishing_point:
        return (False, False, False, True)
    return (True, True, True, True)


def main() -> int:
    args = parse_args()
    stages = _resolve_stages(args)
    return run_pipeline(args.input, args.output_dir, stages)


if __name__ == "__main__":
    sys.exit(main())
