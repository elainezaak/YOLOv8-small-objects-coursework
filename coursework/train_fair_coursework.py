"""Unified VisDrone training, with stage/model/run classification. Run from repo root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'
for k, v in {
    'MPLCONFIGDIR': CACHE / 'matplotlib',
    'XDG_CACHE_HOME': CACHE / 'xdg',
    'YOLO_CONFIG_DIR': CACHE / 'ultralytics',
}.items():
    os.environ.setdefault(k, str(v))
    Path(os.environ[k]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402

CONFIGS = {
    'p2': ROOT / 'model_configs/yolov8-p2.yaml',
    'p2-nop5': ROOT / 'model_configs/yolov8-p2_woP5.yaml',
    'fusion': ROOT / 'model_configs/yolov8-p2_woP5_sandwich.yaml',
    'final': ROOT / 'model_configs/yolov8-p2_woP5_sandwich_CBAM.yaml',
}
STAGES = {
    'ablation': ('02_ablation_30ep', 30),
    'formal': ('03_formal_100ep', 100),
}


def sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    digest = hashlib.sha256()
    with p.open('rb') as file:
        for piece in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=['baseline', *CONFIGS], required=True)
    ap.add_argument('--stage', choices=STAGES, default='ablation')
    ap.add_argument('--init', choices=['pretrained', 'scratch'], default='pretrained')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='0')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--data', type=Path, default=ROOT.parent / 'datasets/VisDrone/VisDrone-local.yaml')
    args = ap.parse_args()

    stage_dir, default_epochs = STAGES[args.stage]
    epochs = args.epochs or default_epochs
    if args.batch <= 0 or epochs <= 0 or args.imgsz <= 0:
        ap.error('epochs, batch and imgsz must be positive')
    data = args.data.expanduser().resolve()
    if not data.is_file():
        ap.error(f'Dataset YAML missing: {data}')
    repo_local = (ROOT / 'ultralytics/__init__.py').resolve()
    if Path(ultralytics.__file__).resolve() != repo_local:
        ap.error('Not using repository-local Ultralytics. Check your environment.')

    model_config = 'yolov8n.pt' if args.init == 'pretrained' else 'yolov8n.yaml'
    if args.model == 'baseline':
        model = YOLO(model_config)
    else:
        model_config = CONFIGS[args.model]
        model = YOLO(str(model_config))
        if args.init == 'pretrained':
            model.load('yolov8n.pt')  # Log the *first* Transferred X/Y for COCO transfer.

    model_folder = args.model.replace('-', '_')
    project = ROOT / 'runs' / stage_dir / model_folder
    # Every invocation has its own run subdirectory. Keep completed runs intact.
    name = f'seed{args.seed}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    started_at = datetime.now().astimezone().isoformat()
    start = time.perf_counter()
    model.train(
        data=str(data), epochs=epochs, batch=args.batch, imgsz=args.imgsz,
        device=args.device, workers=args.workers, seed=args.seed,
        deterministic=True, optimizer='SGD',
        patience=epochs + 10, save=True, plots=True, val=True, amp=True,
        project=str(project), name=name, exist_ok=False,
    )
    elapsed_s = time.perf_counter() - start
    run_dir = Path(model.trainer.save_dir).resolve()
    record = {
        'stage': args.stage, 'model': args.model,
        'init_strategy': args.init, 'pretrained_source': 'yolov8n.pt' if args.init == 'pretrained' else None,
        'note': 'Custom architectures transfer only compatible named tensors. See full training log.',
        'epochs_requested': epochs, 'epochs_completed': len((run_dir / 'results.csv').read_text().splitlines()) - 1 if (run_dir / 'results.csv').is_file() else None,
        'batch': args.batch, 'imgsz': args.imgsz, 'seed': args.seed,
        'optimizer': 'SGD', 'deterministic': True, 'amp': True,
        'device': args.device, 'workers': args.workers, 'dataset_yaml': str(data),
        'dataset_yaml_sha256': sha256(data),
        'model_config': str(model_config),
        'model_config_sha256': sha256(model_config),
        'start_time': started_at, 'end_time': datetime.now().astimezone().isoformat(),
        'wall_time_seconds': round(elapsed_s, 2),
        'wall_time_minutes': round(elapsed_s / 60, 2),
        'parameters': int(sum(p.numel() for p in model.model.parameters())),
        'python': platform.python_version(), 'torch': torch.__version__,
        'torch_cuda': torch.version.cuda, 'ultralytics': ultralytics.__version__,
        'gpu': torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else args.device,
        'best_weights': str(run_dir / 'weights' / 'best.pt'),
        'last_weights': str(run_dir / 'weights' / 'last.pt'),
        'results_csv': str(run_dir / 'results.csv'),
    }
    (run_dir / 'run_summary.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\nTraining finished; wall time: {elapsed_s / 60:.2f} min', flush=True)
    print(f'Files: {run_dir}', flush=True)
    print(f'Summary: {run_dir / "run_summary.json"}', flush=True)


if __name__ == '__main__':
    main()
