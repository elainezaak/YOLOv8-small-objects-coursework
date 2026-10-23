"""Unified VisDrone validation, inference benchmarking and sample predictions.

Copy this file into coursework/evaluate_coursework.py in the repository and run
from the repository root with the repo's yolov8_paper conda environment.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import random
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'
for k, d in {
    'MPLCONFIGDIR': CACHE / 'matplotlib',
    'XDG_CACHE_HOME': CACHE / 'xdg',
    'YOLO_CONFIG_DIR': CACHE / 'ultralytics',
}.items():
    os.environ.setdefault(k, str(d))
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', action='append', required=True,
                    help='Name=checkpoint.pt (repeat this option to compare models)')
    ap.add_argument('--data', default=str(ROOT.parent / 'datasets/VisDrone/VisDrone-local.yaml'))
    ap.add_argument('--split', choices=['val', 'test'], default='val')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=1,
                    help='Validation batch; use 1 to benchmark per-image latency')
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--sample-count', type=int, default=6)
    ap.add_argument('--experiment', choices=['ablation30', 'formal100'], default='ablation30')
    ap.add_argument('--half', action='store_true', help='Use FP16 for validation on CUDA')
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'runs/04_evaluation')
    args = ap.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        ap.error(f'Dataset YAML does not exist: {data_path}')
    output = args.output_dir.expanduser().resolve() / args.experiment
    output.mkdir(parents=True, exist_ok=True)
    data_cfg = yaml.safe_load(data_path.read_text(encoding='utf-8'))
    ds_root = Path(data_cfg['path']).expanduser()
    if not ds_root.is_absolute():
        ds_root = (data_path.parent / ds_root).resolve()
    sample_folder = ds_root / data_cfg[args.split]
    samples = sorted(sample_folder.glob('*.jpg'))
    if not samples:
        raise FileNotFoundError(f'No images in {sample_folder}')
    sample_paths = sorted(random.Random(args.seed).sample(samples, min(args.sample_count, len(samples))))
    rows = []
    for item in args.run:
        if '=' not in item:
            ap.error(f'Expected --run NAME=CHECKPOINT, got {item}')
        name, weights_str = item.split('=', 1)
        if not name.strip() or '/' in name or '\\' in name:
            ap.error(f'Invalid model label {name!r}')
        weights = Path(weights_str).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {weights}')
        print(f'\n===== Evaluating {name} on {args.split}: {weights} =====', flush=True)
        model = YOLO(str(weights))
        params = sum(p.numel() for p in model.model.parameters())
        metrics = model.val(
            data=str(data_path), split=args.split, imgsz=args.imgsz,
            batch=args.batch, device=args.device, workers=args.workers,
            seed=args.seed, deterministic=True,
            half=args.half, conf=0.001, iou=0.7, max_det=300,
            plots=True, project=str(output / 'validation'),
            name=f'{name}_{args.split}', exist_ok=True,
        )
        speed = dict(metrics.speed)
        inference_ms = float(speed['inference'])
        preprocess_ms = float(speed['preprocess'])
        postprocess_ms = float(speed['postprocess'])
        total_ms = inference_ms + preprocess_ms + postprocess_ms
        row = {
            'model': name, 'split': args.split, 'weights': str(weights),
            'params': params, 'params_M': round(params / 1e6, 4),
            'Precision': float(metrics.box.mp),
            'Recall': float(metrics.box.mr),
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'preprocess_ms_per_image': preprocess_ms,
            'inference_ms_per_image': inference_ms,
            'postprocess_ms_per_image': postprocess_ms,
            'total_ms_per_image': total_ms,
            'approx_fps': (1000 / total_ms) if total_ms > 0 else 0.0,
            'imgsz': args.imgsz, 'val_batch': args.batch,
            'fp16': args.half, 'device': args.device,
        }
        rows.append(row)
        if sample_paths:
            model.predict(
                source=[str(p) for p in sample_paths],
                imgsz=args.imgsz, device=args.device,
                conf=0.25, iou=0.7, max_det=300,
                save=True, project=str(output / 'predictions'),
                name=f'{name}_{args.split}', exist_ok=True, verbose=False,
            )
        print(f'{name}: P={row["Precision"]:.4f}, R={row["Recall"]:.4f}, '
              f'mAP50={row["mAP50"]:.4f}, mAP50-95={row["mAP50-95"]:.4f}, '
              f'params={row["params_M"]:.3f}M, inference={inference_ms:.2f}ms/image',
              flush=True)

    reports = ROOT / 'runs/05_reports'
    reports.mkdir(parents=True, exist_ok=True)
    csv_path = reports / f'comparison_{args.experiment}_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved summary to {csv_path}')
    print(f'Same prediction examples for every model: {len(sample_paths)} images')
    print('Timing is measured during PyTorch validation on this device; it is not Jetson Nano speed.')


if __name__ == '__main__':
    main()
