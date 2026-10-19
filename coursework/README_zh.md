# 小组作业使用说明

这个目录是对原仓库的补充，不修改论文模型的核心实现。建议先把代码跑通，再安排正式训练和改进实验。

## 目录说明

- `check_environment.py`：检查依赖、导入路径和 4 个模型配置。
- `train_coursework.py`：用统一命令训练 baseline 或改进模型。
- `../model_configs/`：论文方法对应的模型 YAML 配置。
- `../ultralytics/`：仓库自带的、已经接入自定义模块的 Ultralytics 源码。

## 第一步：获取代码

```bash
git clone https://github.com/elainezaak/YOLOv8-small-objects-coursework.git
cd YOLOv8-small-objects-coursework
```

不要直接修改 `main` 分支。开始任务前先创建自己的分支：

```bash
git switch -c 你的名字-任务名称
```

## 第二步：检查环境

请使用 Python 3.10，并安装 `requirements-coursework.txt` 中的依赖。进入仓库根目录后运行：

```bash
python coursework/check_environment.py
```

看到 4 行 `OK` 和 `Environment check passed.`，说明仓库的自定义模型可以正常加载。

这个检查不会下载 VisDrone，也不会开始训练。

如果输出的 `Ultralytics path` 没有指向本仓库里的 `ultralytics/` 目录，请不要继续训练，先检查当前目录和 Python 环境。

## 第三步：训练前先做小规模测试

正式训练前，先使用 5 个 epochs 验证数据路径、标签转换、显卡和训练流程：

```bash
python coursework/train_coursework.py \
  --model final \
  --epochs 5 \
  --batch 16 \
  --device 0 \
  --pretrained
```

其中 `--device 0` 表示使用第 1 张可用 GPU；如果当前设备没有 GPU，可以改为 `--device cpu`，但完整训练会很慢。

确认测试成功后，再根据实际算力调整 epochs、batch 和 workers。所有对比实验应尽量使用相同的数据划分、输入尺寸、epochs、batch 和随机种子。

## 模型名称

- `baseline`：官方 YOLOv8n。
- `p2`：增加 P2 检测头。
- `p2-nop5`：增加 P2 并移除 P5 检测头。
- `fusion`：再加入双向特征融合。
- `final`：再加入 CBAM，即论文最终模型。

例如训练 baseline：

```bash
python coursework/train_coursework.py \
  --model baseline \
  --epochs 100 \
  --batch 16 \
  --device 0
```

训练结果默认保存在 `runs/coursework/`，该目录已被 Git 忽略。

## 协作规则

完成修改后提交自己的分支：

```bash
git add .
git commit -m "说明你的修改"
git push -u origin 你的分支名
```

然后在 GitHub 创建 Pull Request，不要把多个成员的实验结果直接混在同一个分支里。

## 不要上传的文件

以下内容不要提交到 GitHub：

```text
VisDrone 数据集
runs/
*.pt
*.onnx
*.engine
训练缓存和临时日志
```

