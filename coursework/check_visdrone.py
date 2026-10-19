"""Validate converted VisDrone labels and generate a small local report."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from pathlib import Path
import random

from PIL import Image, ImageDraw
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT.parent / "datasets" / "VisDrone"
CACHE_ROOT = REPO_ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))

SPLITS = {
    "train": ("VisDrone2019-DET-train", 6471),
    "val": ("VisDrone2019-DET-val", 548),
    "test-dev": ("VisDrone2019-DET-test-dev", 1610),
}
CLASS_NAMES = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
)
COLORS = (
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Report directory (default: DATASET_DIR/reports).",
    )
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_label(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        fields = raw_line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        values = tuple(map(float, fields[1:]))
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"{label_path}:{line_number}: invalid class {class_id}")
        if any(value < 0 or value > 1 for value in values):
            raise ValueError(f"{label_path}:{line_number}: values outside [0, 1]")
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(f"{label_path}:{line_number}: non-positive box size")
        boxes.append((class_id, *values))
    return boxes


def inspect_dataset(
    dataset_dir: Path,
) -> tuple[dict[str, Counter[int]], dict[str, int], list[Path]]:
    distributions: dict[str, Counter[int]] = {}
    image_counts: dict[str, int] = {}
    validation_images: list[Path] = []

    for split, (directory_name, expected_images) in SPLITS.items():
        split_dir = dataset_dir / directory_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing images/ or labels/ under {split_dir}")

        image_paths = sorted(images_dir.glob("*.jpg"))
        label_paths = sorted(labels_dir.glob("*.txt"))
        if len(image_paths) != expected_images:
            raise RuntimeError(
                f"{split}: expected {expected_images} images, found {len(image_paths)}"
            )
        if len(label_paths) != len(image_paths):
            raise RuntimeError(
                f"{split}: found {len(image_paths)} images and {len(label_paths)} labels"
            )

        distribution: Counter[int] = Counter()
        for image_path in tqdm(image_paths, desc=f"Checking {split}"):
            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path.name}")
            boxes = read_label(label_path)
            distribution.update(box[0] for box in boxes)

        distributions[split] = distribution
        image_counts[split] = len(image_paths)
        if split == "val":
            validation_images = image_paths

    return distributions, image_counts, validation_images


def write_csv(output_dir: Path, distributions: dict[str, Counter[int]]) -> Path:
    output_path = output_dir / "class_distribution.csv"
    total_distribution = sum(distributions.values(), Counter())
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["class_id", "class_name", *SPLITS, "total", "percentage"])
        grand_total = sum(total_distribution.values())
        for class_id, class_name in enumerate(CLASS_NAMES):
            total = total_distribution[class_id]
            writer.writerow(
                [
                    class_id,
                    class_name,
                    *(distributions[split][class_id] for split in SPLITS),
                    total,
                    f"{total / grand_total * 100:.4f}",
                ]
            )
    return output_path


def write_markdown(
    output_dir: Path,
    distributions: dict[str, Counter[int]],
    image_counts: dict[str, int],
) -> Path:
    output_path = output_dir / "dataset_report.md"
    total_distribution = sum(distributions.values(), Counter())
    grand_total = sum(total_distribution.values())
    lines = [
        "# VisDrone dataset check",
        "",
        "## Split summary",
        "",
        "| Split | Images | Objects |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        lines.append(
            f"| {split} | {image_counts[split]:,} | {sum(distributions[split].values()):,} |"
        )
    lines.extend(
        [
            f"| Total | {sum(image_counts.values()):,} | {grand_total:,} |",
            "",
            "## Class distribution",
            "",
            "| ID | Class | Train | Val | Test-dev | Total | Percentage |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for class_id, class_name in enumerate(CLASS_NAMES):
        total = total_distribution[class_id]
        lines.append(
            f"| {class_id} | {class_name} | "
            f"{distributions['train'][class_id]:,} | "
            f"{distributions['val'][class_id]:,} | "
            f"{distributions['test-dev'][class_id]:,} | "
            f"{total:,} | {total / grand_total * 100:.2f}% |"
        )
    lines.extend(
        [
            "",
            "All image/label pairs, class IDs, normalized coordinates, and box sizes passed validation.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def draw_distribution(
    output_dir: Path, distributions: dict[str, Counter[int]]
) -> Path:
    import matplotlib.pyplot as plt

    total_distribution = sum(distributions.values(), Counter())
    counts = [total_distribution[index] for index in range(len(CLASS_NAMES))]
    figure, axis = plt.subplots(figsize=(12, 6))
    bars = axis.bar(CLASS_NAMES, counts, color=COLORS)
    axis.set_title("VisDrone object class distribution")
    axis.set_ylabel("Number of objects")
    axis.tick_params(axis="x", rotation=30)
    axis.bar_label(bars, labels=[f"{count:,}" for count in counts], padding=3, fontsize=8)
    figure.tight_layout()
    output_path = output_dir / "class_distribution.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def draw_samples(
    output_dir: Path, image_paths: list[Path], sample_count: int, seed: int
) -> Path | None:
    if sample_count <= 0:
        return None
    selected = random.Random(seed).sample(image_paths, min(sample_count, len(image_paths)))
    tile_width, tile_height, title_height = 640, 400, 26
    columns = 2
    rows = (len(selected) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_width, rows * (tile_height + title_height)), "white")
    sheet_draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(selected):
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(image)
        label_path = image_path.parents[1] / "labels" / f"{image_path.stem}.txt"
        for class_id, center_x, center_y, width, height in read_label(label_path):
            left = (center_x - width / 2) * image.width
            top = (center_y - height / 2) * image.height
            right = (center_x + width / 2) * image.width
            bottom = (center_y + height / 2) * image.height
            draw.rectangle((left, top, right, bottom), outline=COLORS[class_id], width=2)

        column = index % columns
        row = index // columns
        x = column * tile_width + (tile_width - image.width) // 2
        y = row * (tile_height + title_height) + title_height
        sheet.paste(image, (x, y))
        sheet_draw.text(
            (column * tile_width + 8, row * (tile_height + title_height) + 5),
            image_path.name,
            fill="black",
        )

    output_path = output_dir / "label_samples.jpg"
    sheet.save(output_path, quality=92)
    return output_path


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else dataset_dir / "reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    distributions, image_counts, validation_images = inspect_dataset(dataset_dir)
    csv_path = write_csv(output_dir, distributions)
    markdown_path = write_markdown(output_dir, distributions, image_counts)
    chart_path = draw_distribution(output_dir, distributions)
    samples_path = draw_samples(output_dir, validation_images, args.samples, args.seed)

    total_distribution = sum(distributions.values(), Counter())
    print(
        f"Dataset check passed: {sum(image_counts.values())} images, "
        f"{sum(total_distribution.values())} objects, {len(CLASS_NAMES)} classes"
    )
    print(f"Markdown report: {markdown_path}")
    print(f"CSV report: {csv_path}")
    print(f"Distribution chart: {chart_path}")
    if samples_path:
        print(f"Label samples: {samples_path}")


if __name__ == "__main__":
    main()
