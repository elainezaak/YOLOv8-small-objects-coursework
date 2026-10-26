"""Run the environment and VisDrone checks with one command."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
# 默认使用仓库内部的数据目录。
DEFAULT_DATASET_DIR = REPO_ROOT / "datasets" / "VisDrone"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Also require a working NVIDIA CUDA device.",
    )
    return parser.parse_args()


def run(script_name: str, *arguments: str) -> None:
    command = [sys.executable, str(REPO_ROOT / "coursework" / script_name), *arguments]
    print(f"\n==> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    environment_arguments = ["--require-cuda"] if args.require_cuda else []
    run("check_environment.py", *environment_arguments)
    run(
        "check_visdrone.py",
        "--dataset-dir",
        str(args.dataset_dir),
        "--samples",
        str(args.samples),
    )
    print("\nAll coursework preflight checks passed.")


if __name__ == "__main__":
    main()
