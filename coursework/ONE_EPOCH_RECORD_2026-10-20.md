# 1 epoch 运行记录（2026-10-20）

> 本次运行用于验证环境、数据、模型和训练流程能够正常工作。只训练 1 个 epoch，结果不能代表模型最终精度。训练权重和 `runs/` 未上传 GitHub。

## 机器与环境

- 日期：2026-10-20
- 操作系统：Linux（云端容器）
- GPU 型号：NVIDIA vGPU-48GB（48 GB 显存）
- NVIDIA 驱动版本：580.76.05
- `nvidia-smi` 显示的 CUDA Version：13.0
- Python：3.10.22
- PyTorch：2.2.2+cu121
- Torchvision：0.17.2+cu121
- PyTorch CUDA build（`torch.version.cuda`）：12.1
- cuDNN：8.9.2
- Ultralytics：8.1.20（本仓库源码）

## 训练配置

- Git commit：`f39b01e`
- 模型：`final`（`yolov8-p2_woP5_sandwich_CBAM.yaml`）
- 数据集：VisDrone2019-DET
- 训练集：6471 张图片
- 验证集：548 张图片
- epochs：1
- imgsz：640
- batch：16
- device：0
- workers：0
- seed：42
- 是否加载预训练权重：是（`yolov8n.pt`）

运行命令：

```bash
python coursework/train_coursework.py \
  --data ../datasets/VisDrone/VisDrone-local.yaml \
  --model final \
  --epochs 1 \
  --batch 16 \
  --device 0 \
  --workers 0 \
  --pretrained
```

## 结果

- 是否完成 1 epoch：是，训练及最终验证均正常完成
- 总耗时：0.061 小时（约 3 分 40 秒）
- 单个 epoch 耗时：约 3 分 40 秒（包含验证）
- GPU 显存峰值：约 28.9 GB
- train box loss：3.4105
- train cls loss：3.6501
- train dfl loss：3.0517
- val box loss：2.5218
- val cls loss：2.7479
- val dfl loss：1.9103
- val precision：0.57517
- val recall：0.05546
- val mAP50：0.05904
- val mAP50-95：0.02483
- 结果目录：`runs/coursework/final2/`
- 生成文件：`results.csv`、`weights/best.pt`、`weights/last.pt` 以及训练和验证图表

## 异常与调整

第一次使用 `workers=8` 时训练阶段完成，但验证停在 12/18。将 `workers` 改为 0 后重新运行，训练、验证和 `best.pt` 最终验证全部正常完成。因此，在容器或共享服务器环境中遇到数据加载进程卡住时，可以先使用 `--workers 0` 验证完整流程，再根据机器情况逐步增加 workers。

## 说明

自动使用 `best.pt` 进行最终验证时，终端显示的四舍五入结果为：Precision 0.576、Recall 0.0554、mAP50 0.0594、mAP50-95 0.0249。由于本次只训练 1 个 epoch，指标较低属于预期现象；本记录主要证明项目可以在 GPU 环境中完整运行。
