# `train_args.py`

## 文件作用

这个文件集中管理训练脚本的命令行参数，供 [train.py](/home/yimin/research/RAILGUN/train.py) 调用。

## 主要函数

### `get_args()`

解析训练参数并返回 `argparse.Namespace`。

参数分为五类：

- 日志与随机种子：
  - `--seed`
  - `--log_dir`
- 数据集：
  - `--dataset_path`
  - `--val_dataset_path`
  - `--dataset_mode`
  - `--train_map_path`
  - `--online_total_steps`
  - `--online_eval_interval_steps`
  - `--online_save_interval_steps`
  - `--online_inference_test_interval_steps`
  - `--online_time_limit_sec`
  - `--online_retry_limit`
  - `--sample_data_path`
  - `--inference_num_cases`
  - `--inference_test_interval`
  - `--inference_action_choice`
  - `--num_workers`
- 训练超参数：
  - `--epochs`
  - `--batch_size`
  - `--learning_rate`
  - `--weight_decay`
  - `--eval_interval`
  - `--save_interval`
- 模型与运行方式：
  - `--feature_dim`
  - `--feature_type`
  - `--steps`
  - `--action_dim`
  - `--model`
  - `--bilinear`
  - `--first_layer_channels`
  - `--distributed`
  - `--model_path`

## 接口情况

- 返回值直接传给 `train.py` 的主流程
- 支持两种训练模式：
  - `offline`: 训练和验证都走离线 `.mbin`
  - `online`: 训练在线生成，验证仍使用离线 `.mbin`

## 在线训练相关参数

### `--dataset_mode`

训练数据模式：

- `offline`
- `online`

### `--train_map_path`

在线训练时扫描地图的根目录。

### `--val_dataset_path`

固定离线验证集根目录；未提供时回退到 `--dataset_path`。

### `--online_total_steps`

在线训练总优化步数。

### `--online_eval_interval_steps`

在线训练每隔多少个优化步做一次验证。

### `--online_save_interval_steps`

在线训练每隔多少个优化步保存一次 checkpoint。

### `--online_inference_test_interval_steps`

在线训练每隔多少个优化步运行一次 inference test。

### `--online_time_limit_sec`

每次在线生成一个 LACAM 场景时允许使用的时间上限。

### `--online_retry_limit`

在线生成失败时单个 worker 的最大重试次数。

### `--sample_data_path`

训练中 inference test 使用的固定离线 `.mbin` 样本。

### `--inference_num_cases`

每次 inference test 要跑的固定样本数。

### `--inference_test_interval`

离线模式下每隔多少个 epoch 运行一次 inference test。

- `0` 表示复用 `--eval_interval`

### `--inference_action_choice`

inference rollout 时的动作选择策略：

- `sample`
- `max`

### `--steps`

`path_formation()` / inference test 的最大 rollout 步数。

## 用法

```python
from train_args import get_args

args = get_args()
print(args.batch_size)
```

命令行通常不单独调用，而是通过训练入口使用：

```bash
python train.py --epochs 50 --batch_size 32 --model unet

# 在线训练
python train.py \
  --dataset_mode online \
  --train_map_path data/map_files \
  --val_dataset_path data/input_data \
  --online_total_steps 200000 \
  --online_eval_interval_steps 4000 \
  --online_save_interval_steps 4000 \
  --online_inference_test_interval_steps 4000 \
  --sample_data_path data/input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin \
  --inference_num_cases 1 \
  --inference_action_choice max \
  --steps 100
```
