"""
Backward-compatibility wrapper for the ablation study pipeline.
Prefer running: python -m src.pipelines.ablation_study --mode train
"""

import sys

from src.pipelines.ablation_study import main as ablation_main


if __name__ == "__main__":
    # Default to train mode if not explicitly provided
    args = sys.argv[1:]
    if "--mode" not in args and "-m" not in args:
        sys.argv.extend(["--mode", "train"])
    ablation_main()
