# `eval_test.py`

## 文件作用

这个脚本用于加载训练好的模型，在 `.mbin`、`.bin` 或目录数据上做 rollout 评估，并把路径可视化输出到指定目录。

它依赖：

- [models/unet.py](/home/yimin/research/RAILGUN/models/unet.py)
- [MAPF_dataset.py](/home/yimin/research/RAILGUN/MAPF_dataset.py)
- [tools/path_formation.py](/home/yimin/research/RAILGUN/tools/path_formation.py)
- [tools/visualize_path.py](/home/yimin/research/RAILGUN/tools/visualize_path.py)

## 主要函数

### `main()`

完整评估入口，负责：

1. 解析命令行参数
2. 固定随机种子
3. 初始化 `UNet`
4. 加载模型权重
5. 根据输入后缀自动选择 `.mbin` 或 `.bin` 数据集实现
6. 构造 `MAPFDataset(..., first_step=True)` 或 `MAPFDatasetMbin(..., first_step=True)`
7. 逐个样本调用 `path_formation()`
8. 调用 `visualize_path()` 生成视频或弹出窗口

## 输入输出与接口情况

### CLI 参数

核心参数如下：

- `--model_path`: 模型权重路径，必填
- `--dataset_paths`: `.bin` 文件或目录，必填
- `--feature_dim`
- `--feature_type`
- `--steps`
- `--action_dim`
- `--bilinear`
- `--first_layer_channels`
- `--blocks_per_stage`
- `--output_dir`
- `--show`: 为真时直接显示交互窗口
- `--lifelong`: 是否在到达目标后继续重新分配目标

### 数据接口

- 当 `dataset_paths[0]` 以 `.mbin` 结尾时，使用 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py)
- 当 `dataset_paths[0]` 以 `.bin` 结尾时，使用 [MAPF_dataset.py](/home/yimin/research/RAILGUN/MAPF_dataset.py)
- 否则会把它当作目录，递归匹配 `**/*-0.bin`

### 模型结构参数

为了能正确加载不同规模的 checkpoint，这个脚本现在也支持显式指定 UNet 结构：

- `--first_layer_channels`
- `--blocks_per_stage`
- `--bilinear`

这些参数必须和训练该 checkpoint 时使用的模型结构保持一致；scaling law 产生的 checkpoint 尤其要注意这一点。

### 输出

- 文本日志：`<output_dir>/<dataset_paths[0].split('/')[2]>_log.txt`
- 视频或交互式可视化

说明：

- 当前实现会循环评估所有样本，但循环结束后只把最后一次 `path_formation()` 的结果传给 `visualize_path()`。

## 用法

```bash
python eval_test.py \
  --model_path runs/20260101-000000/model_checkpoint_epoch_4.pth \
  --dataset_paths data/input_data/example/example-0.mbin \
  --first_layer_channels 64 \
  --blocks_per_stage 1 \
  --show
```
