# Ultralytics YOLO 🚀, AGPL-3.0 license
# YOLO 检测任务专用的训练器、预测器和验证器
from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator

__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator"
