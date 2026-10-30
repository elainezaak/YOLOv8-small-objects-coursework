"""Collect repository experiment artifacts into one comparable CSV table.

The table preserves evaluation mode and source paths so standard full-image,
tiled, and tiled-plus-full-image experiments are not compared as if they used
the same inference protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

FIELDS = [
    "record_id",
    "source_group",
    "source_type",
    "experiment_path",
    "source_files",
    "status",
    "model",
    "split",
    "evaluation_mode",
    "epochs",
    "training_minutes",
    "imgsz",
    "val_batch",
    "tile_size",
    "overlap",
    "full_image_pass",
    "full_imgsz",
    "conf",
    "predict_iou",
    "merge_iou",
    "ios_threshold",
    "boundary_aware",
    "small_filter",
    "small_area",
    "small_conf",
    "max_det",
    "parameters",
    "parameters_m",
    "precision",
    "recall",
    "f1",
    "mAP50",
    "mAP50_95",
    "images",
    "ground_truth_boxes",
    "predictions",
    "mean_preprocess_ms",
    "mean_inference_ms",
    "mean_postprocess_ms",
    "mean_total_ms",
    "p95_total_ms",
    "mean_tiles",
    "mean_boxes_per_image",
    "weights",
    "notes",
]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def blank(**values) -> dict:
    row = {field: "" for field in FIELDS}
    row.update(values)
    return row


def infer_model(text: str) -> str:
    value = text.lower().replace("_", "-")
    if "p2awf" in value or "p2-awf" in value:
        return "final-p2awf"
    if "p2-nop5" in value or "p2-nop5" in value:
        return "p2-nop5"
    if "baseline" in value:
        return "baseline"
    if "fusion" in value:
        return "fusion"
    if re.search(r"(?:^|[-/])p2(?:[-/]|$)", value):
        return "p2"
    if "final" in value:
        return "final"
    return "unknown"


def infer_epochs(text: str) -> str:
    match = re.search(r"(?:^|[^0-9])(30|100)ep(?:[^0-9]|$)", text.lower())
    return match.group(1) if match else ""


def percentile_95(timing_csv: Path) -> float | str:
    if not timing_csv.is_file():
        return ""
    with timing_csv.open(encoding="utf-8", newline="") as file:
        values = sorted(float(row["total_ms"]) for row in csv.DictReader(file) if row.get("total_ms"))
    if not values:
        return ""
    position = 0.95 * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def record_from_standard_summary(path: Path) -> dict:
    data = read_json(path)
    stage = path.parents[1].name
    model = data.get("model", path.parent.name)
    speed = data.get("validation_speed_ms", {})
    return blank(
        source_group="training_validation",
        source_type="metrics_summary_json",
        experiment_path=rel(path.parent),
        source_files=rel(path),
        status="valid",
        model=model,
        split="val",
        evaluation_mode="standard_full_image",
        epochs=infer_epochs(stage),
        training_minutes=data.get("training_minutes", ""),
        imgsz=640,
        parameters=data.get("parameters", ""),
        parameters_m=(data["parameters"] / 1e6) if data.get("parameters") else "",
        precision=data.get("precision", ""),
        recall=data.get("recall", ""),
        mAP50=data.get("mAP50", ""),
        mAP50_95=data.get("mAP50-95", ""),
        mean_preprocess_ms=speed.get("preprocess", ""),
        mean_inference_ms=speed.get("inference", ""),
        mean_postprocess_ms=speed.get("postprocess", ""),
        mean_total_ms=(
            speed.get("preprocess", 0) + speed.get("inference", 0) + speed.get("postprocess", 0)
            if speed else ""
        ),
        weights=data.get("weights", ""),
        notes="Standard Ultralytics validation record.",
    )


def record_from_full_image_csv(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as file:
        for item in csv.DictReader(file):
            rows.append(
                blank(
                    source_group="full_image_evaluation",
                    source_type="evaluate_coursework_csv",
                    experiment_path=rel(path.parent),
                    source_files=rel(path),
                    status="valid",
                    model=item.get("model", "unknown"),
                    split=item.get("split", ""),
                    evaluation_mode="standard_full_image",
                    epochs=infer_epochs(path.name),
                    imgsz=item.get("imgsz", ""),
                    val_batch=item.get("val_batch", ""),
                    conf=0.001,
                    predict_iou=item.get("iou", 0.7),
                    max_det=300,
                    parameters=item.get("params", ""),
                    parameters_m=item.get("params_M", ""),
                    precision=item.get("Precision", ""),
                    recall=item.get("Recall", ""),
                    mAP50=item.get("mAP50", ""),
                    mAP50_95=item.get("mAP50-95", ""),
                    mean_preprocess_ms=item.get("preprocess_ms_per_image", ""),
                    mean_inference_ms=item.get("inference_ms_per_image", ""),
                    mean_postprocess_ms=item.get("postprocess_ms_per_image", ""),
                    mean_total_ms=item.get("total_ms_per_image", ""),
                    weights=item.get("weights", ""),
                    notes="Independent full-image evaluation; batch=1 when recorded.",
                )
            )
    return rows


def find_artifact_parent(metrics_path: Path, root: Path) -> Path:
    """Find the nearest parent containing a config or summary artifact."""
    current = metrics_path.parent.parent
    while current != root.parent:
        if (current / "config.json").is_file() or (current / "summary.json").is_file():
            return current
        if current == root:
            break
        current = current.parent
    return metrics_path.parent.parent


def record_from_sliced_metrics(metrics_path: Path, root: Path) -> dict:
    experiment_dir = find_artifact_parent(metrics_path, root)
    metrics = read_json(metrics_path)
    config_path = experiment_dir / "config.json"
    summary_path = experiment_dir / "summary.json"
    timing_path = experiment_dir / "timing.csv"
    config = read_json(config_path) if config_path.is_file() else {}
    summary = read_json(summary_path) if summary_path.is_file() else {}
    split = "test-dev" if "test_dev" in str(metrics_path).replace("-", "_").lower() else "val"
    weights = config.get("weights", "")
    predictions = metrics.get("predictions", "")
    status = "invalid_zero_predictions" if predictions == 0 else "valid"
    full_image = config.get("full_image_pass", False)
    tile_size = config.get("tile_size", "")
    if full_image:
        mode = "tiled_plus_full_image"
    elif isinstance(tile_size, (int, float)) and tile_size >= 10000:
        mode = "single_tile_full_image"
    else:
        mode = "tiled"
    parameters = summary.get("parameters", config.get("parameters", metrics.get("parameters", "")))
    images = summary.get("images", metrics.get("images", ""))
    total_detections = summary.get("total_detections", "")
    mean_boxes = total_detections / images if isinstance(images, int) and images and isinstance(total_detections, int) else ""
    source_paths = [metrics_path]
    for candidate in (config_path, summary_path, timing_path):
        if candidate.is_file():
            source_paths.append(candidate)
    return blank(
        source_group="sliced_inference",
        source_type="sliced_config_summary_metrics",
        experiment_path=rel(experiment_dir),
        source_files=";".join(rel(item) for item in source_paths),
        status=status,
        model=infer_model(weights or rel(experiment_dir)),
        split=split,
        evaluation_mode=mode,
        epochs=infer_epochs(str(experiment_dir)),
        imgsz=config.get("imgsz", ""),
        tile_size=tile_size,
        overlap=config.get("overlap", ""),
        full_image_pass=config.get("full_image_pass", False),
        full_imgsz=config.get("full_imgsz", ""),
        conf=config.get("conf", ""),
        predict_iou=config.get("predict_iou", ""),
        merge_iou=config.get("merge_iou", ""),
        ios_threshold=config.get("ios_threshold", ""),
        boundary_aware=config.get("boundary_aware", ""),
        small_filter=config.get("small_filter_enabled", ""),
        small_area=config.get("small_area", ""),
        small_conf=config.get("small_conf", ""),
        max_det=config.get("max_det", ""),
        parameters=parameters,
        parameters_m=summary.get("parameters_m", config.get("parameters_m", metrics.get("parameters_m", ""))),
        precision=metrics.get("precision", ""),
        recall=metrics.get("recall", ""),
        f1=metrics.get("f1", ""),
        mAP50=metrics.get("mAP50", ""),
        mAP50_95=metrics.get("mAP50-95", ""),
        images=images,
        ground_truth_boxes=metrics.get("ground_truth_boxes", ""),
        predictions=predictions,
        mean_preprocess_ms=summary.get("mean_preprocess_ms", ""),
        mean_inference_ms=summary.get("mean_inference_ms", ""),
        mean_postprocess_ms=summary.get("mean_postprocess_ms", ""),
        mean_total_ms=summary.get("mean_total_ms", ""),
        p95_total_ms=percentile_95(timing_path),
        mean_tiles=summary.get("mean_tiles", ""),
        mean_boxes_per_image=mean_boxes,
        weights=weights,
        notes=("Zero predictions were recorded; keep for audit, exclude from performance ranking." if status != "valid" else ""),
    )


def record_history_only_rows(path: Path, covered: set[tuple[str, str]]) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as file:
        for item in csv.DictReader(file):
            model = item.get("model", "unknown")
            stage = "100" if "100ep" in path.name else "30" if "ablation" in path.name else ""
            if (stage, model) in covered:
                continue
            rows.append(
                blank(
                    source_group="historical_report",
                    source_type="report_csv_only",
                    experiment_path=rel(path.parent),
                    source_files=rel(path),
                    status="historical_summary_only",
                    model=model,
                    split="test-dev" if "+cut" in model else "val",
                    evaluation_mode="unspecified_reported",
                    epochs=stage,
                    parameters=item.get("parameters", ""),
                    precision=item.get("precision", ""),
                    recall=item.get("recall", ""),
                    mAP50=item.get("mAP50", ""),
                    mAP50_95=item.get("mAP50-95", ""),
                    training_minutes=item.get("training_minutes", ""),
                    weights=item.get("weights", ""),
                    notes="Historical summary row without matching structured run artifact.",
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "all_experiment_results.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    covered: set[tuple[str, str]] = set()
    for summary_path in sorted((ROOT / "reports" / "evaluation").glob("**/metrics_summary.json")):
        row = record_from_standard_summary(summary_path)
        rows.append(row)
        covered.add((row["epochs"], row["model"]))

    for history_path in (ROOT / "reports" / "ablation_comparison.csv", ROOT / "reports" / "formal_100ep.csv"):
        if history_path.is_file():
            rows.extend(record_history_only_rows(history_path, covered))

    for comparison_path in sorted((ROOT / "runs" / "05_reports").glob("comparison_*.csv")):
        rows.extend(record_from_full_image_csv(comparison_path))

    for sliced_root in (ROOT / "runs" / "cloud2", ROOT / "runs" / "cloud"):
        if not sliced_root.is_dir():
            continue
        for metrics_path in sorted(sliced_root.glob("**/evaluation/metrics.json")):
            if ".ipynb_checkpoints" in metrics_path.parts:
                continue
            rows.append(record_from_sliced_metrics(metrics_path, sliced_root))

    # Notebook checkpoints and copied cloud directories can contain byte-for-byte
    # duplicate experiment artifacts. Merge those replicas into one experiment row.
    deduplicated: list[dict] = []
    seen_sliced: dict[tuple, dict] = {}
    ignored = {"record_id", "experiment_path", "source_files", "notes"}
    for row in rows:
        if row["source_group"] != "sliced_inference":
            deduplicated.append(row)
            continue
        signature = tuple((key, row[key]) for key in FIELDS if key not in ignored)
        original = seen_sliced.get(signature)
        if original is None:
            seen_sliced[signature] = row
            deduplicated.append(row)
            continue
        original["experiment_path"] += ";" + row["experiment_path"]
        original["source_files"] += ";" + row["source_files"]
        original["notes"] = "Duplicate artifact locations merged into this single experiment record."
    rows = deduplicated

    for index, row in enumerate(rows, 1):
        row["record_id"] = f"EXP-{index:03d}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} experiment records to {args.output}")


if __name__ == "__main__":
    main()
