# 小组作业使用说明

这个目录是对原仓库的补充，不修改论文模型的核心实现。

## 本机检查

在仓库根目录运行：

```bash
conda activate /path/to/yolo-small-dev
python coursework/check_environment.py
```

看到 4 行 `OK` 和 `Environment check passed.`，说明代码和模型配置可以正常加载。

本机检查不会下载 VisDrone，也不会开始训练。

## 模型名称

- `baseline`: 官方 YOLOv8n
- `p2`: 增加 P2 检测头
- `p2-nop5`: 增加 P2 并移除 P5 检测头
- `fusion`: 再加入双向特征融合
- `final`: 再加入 CBAM，即论文最终模型

