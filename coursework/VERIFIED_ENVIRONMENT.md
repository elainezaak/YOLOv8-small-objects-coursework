# 已验证运行环境

验证日期：2026-10-19

这是一套已经通过模型构建检查的参考环境，不表示作业必须使用相同型号的显卡。

## 软件环境

| 项目 | 已验证版本 |
|---|---|
| 操作系统 | Linux 6.8，x86_64，glibc 2.35 |
| Python | 3.10.22 |
| PyTorch | 2.2.2+cu121 |
| Torchvision | 0.17.2+cu121 |
| PyTorch CUDA build | 12.1 |
| cuDNN | 8.9.2（8902） |
| Ultralytics | 8.1.20，本仓库源码 |
| NVIDIA 驱动 | 580.178.04 |

## 验证硬件

- 2 张 NVIDIA RTX A6000，每张约 48 GB 显存。
- 该机器仅作为环境与训练流程验证设备，不是小组作业的硬件要求。
- 其他支持 CUDA 12.1 PyTorch 构建的 NVIDIA GPU 也可以使用同一套依赖；batch 应根据实际显存调整。

## Linux/NVIDIA 安装命令

```bash
conda create -n yolo_small python=3.10 pip -y
conda activate yolo_small

python -m pip install --upgrade pip
python -m pip install -r requirements-cuda121.txt
python -m pip install -r requirements-coursework.txt
```

共享服务器的 `/home` 空间有限时，应把环境创建到管理员允许的数据盘目录：

```bash
conda create --prefix /数据盘/个人目录/envs/yolo_small python=3.10 pip -y
conda activate /数据盘/个人目录/envs/yolo_small
```

不需要为了本项目单独安装系统级 CUDA 12.1 Toolkit。PyTorch 官方 `cu121` 安装包包含所需的 CUDA 运行时；机器的 NVIDIA 驱动需要支持该运行时。

`nvidia-smi` 顶部显示的 `CUDA Version` 是驱动最高支持的 CUDA 版本，可能高于 `torch.version.cuda`，两者不要求完全相同。

## 已通过检查

运行：

```bash
python coursework/check_environment.py --require-cuda
```

检查结果：

- 成功导入本仓库的 Ultralytics 源码，而不是其他位置安装的同名包。
- CUDA 可用，并识别两张 NVIDIA RTX A6000。
- `yolov8-p2.yaml` 构建成功。
- `yolov8-p2_woP5.yaml` 构建成功。
- `yolov8-p2_woP5_sandwich.yaml` 构建成功。
- `yolov8-p2_woP5_sandwich_CBAM.yaml` 构建成功。

数据集检查和 1 epoch 训练记录分别见 `VISDRONE_DATA_REPORT.md` 和 `ONE_EPOCH_RECORD_TEMPLATE.md`。
