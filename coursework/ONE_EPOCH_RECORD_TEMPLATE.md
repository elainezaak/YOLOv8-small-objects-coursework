# 1 epoch 运行记录

> GPU 可用后复制本模板并填写。训练产生的权重和 `runs/` 不上传 GitHub；可提交不含隐私或绝对路径的文字记录。

## 机器与环境

- 日期：
- 操作系统：
- GPU 型号：
- NVIDIA 驱动版本：
- `nvidia-smi` 显示的 CUDA Version：
- Python：3.10.x
- PyTorch：2.2.2
- Torchvision：0.17.2
- PyTorch CUDA build（`torch.version.cuda`）：
- Ultralytics：8.1.20（本仓库源码）

## 训练配置

- Git commit：
- 模型：`final`
- 数据集：VisDrone2019-DET
- epochs：1
- imgsz：640
- batch：
- device：0
- workers：
- seed：42
- 是否加载预训练权重：是 / 否

运行命令：

```bash
python coursework/train_coursework.py \
  --model final \
  --epochs 1 \
  --batch 8 \
  --device 0 \
  --pretrained
```

## 结果

- 是否完成 1 epoch：
- 总耗时：
- 单个 epoch 耗时：
- GPU 显存峰值：
- train box loss：
- train cls loss：
- train dfl loss：
- val mAP50：
- val mAP50-95：
- 结果目录：`runs/coursework/final/`
- 异常或调整说明：
