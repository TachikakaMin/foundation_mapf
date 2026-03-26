# `train_args.py`

## 文件作用

这个文件集中管理训练脚本的命令行参数，供 [train.py](/home/yimin/research/RAILGUN/train.py) 调用。

## 主要函数

### `get_args()`

解析训练参数并返回 `argparse.Namespace`。

参数大致分为几类：

- 日志与随机种子：
  - `--config`
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
  - `--online_buffer_size`
  - `--online_buffer_timeout_sec`
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
  - `--bilinear` / `--no-bilinear`
  - `--first_layer_channels`
  - `--blocks_per_stage`
  - `--distributed` / `--no-distributed`
  - `--model_path`

## 配置文件加载逻辑

### `--config`

训练参数可以先从 YAML 文件加载，再由 CLI 覆盖。

当前行为：

- 配置文件必须是顶层 key-value 映射
- key 名必须和 argparse 里的参数名一致
- 若配置里同时出现 `learning_rate` 和 `lr`，会直接报错
- 若 `sample_data_path` 在 YAML 里写成单个字符串，会自动规范化成列表
- 如果 YAML 中出现未知字段，也会直接报错
- 最终 `args.config_source` 会标记当前是“纯配置”还是“配置 + CLI 覆盖”

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

在线训练总 optimizer step 数。

### `--online_eval_interval_steps`

在线训练每隔多少个 optimizer step 做一次验证。

### `--online_save_interval_steps`

在线训练每隔多少个 optimizer step 保存一次 checkpoint。

### `--online_inference_test_interval_steps`

在线训练每隔多少个 optimizer step 跑一次 inference test。

### `--online_time_limit_sec`

每次在线生成一个 LACAM 场景时允许使用的时间上限。

### `--online_retry_limit`

在线生成失败时的最大重试次数。

### `--online_buffer_size`

在线 step buffer 的容量。

当前语义是：

- 单位是 `step`
- 按地图尺寸分组分别生效
- 不是 trajectory 数

### `--online_buffer_timeout_sec`

在线训练消费者从 step buffer 取数据时的超时时间。

### `--num_workers`

这是当前统一后的 worker 概念：

- `offline`：表示离线 `DataLoader` 的 worker 数
- `online`：表示在线 scenario producer 进程数
- 同时也复用于离线验证集和 inference sample 的 `DataLoader` worker 数

也就是说，当前已经没有单独的 `online_buffer_workers`。

### `--sample_data_path`

训练中 inference test 使用的固定离线 `.mbin` 样本列表。

### `--inference_num_cases`

每次 inference test 要跑的固定样本数。

### `--inference_test_interval`

只用于离线模式：每隔多少个 epoch 运行一次 inference test。

- `0` 表示复用 `--eval_interval`

### `--inference_action_choice`

inference rollout 时的动作选择策略：

- `sample`
- `max`

### `--steps`

`path_formation()` / inference test 的最大 rollout 步数。

## 模型结构相关参数

### `--bilinear`

控制 UNet 上采样阶段是否使用双线性插值。

- `true`：使用 `Upsample`
- `false`：使用 `ConvTranspose2d`

### `--first_layer_channels`

UNet 第一层的基础通道数。

这是当前最直接的模型规模控制参数之一；值越大，整网通道数会按 stage 成比例放大。

### `--blocks_per_stage`

每个 UNet stage 里的 `ResBlock` 数量。

- `1` 及以上：使用新的 `ResStage` 结构
- `0`：退回兼容旧版的 `DoubleConv`

这个参数和 `first_layer_channels` 一起控制模型大小，也是 `scaling_law.py` 默认扫描的模型轴。

## 用法

```python
from train_args import get_args

args = get_args()
print(args.batch_size)
```

命令行通常通过训练入口使用：

```bash
python train.py --config config.offline.yaml
python train.py --config config.online.yaml

# 配置文件 + 临时覆盖
python train.py --config config.online.yaml --batch_size 32 --online_total_steps 50000

python train.py --epochs 50 --batch_size 32 --model unet

# 在线训练
python train.py \
  --dataset_mode online \
  --train_map_path data/map_files \
  --val_dataset_path data/online_eval_input_data \
  --online_total_steps 200000 \
  --online_eval_interval_steps 4000 \
  --online_save_interval_steps 4000 \
  --online_inference_test_interval_steps 4000 \
  --online_time_limit_sec 1 \
  --online_retry_limit 2 \
  --online_buffer_size 10240 \
  --online_buffer_timeout_sec 0.1 \
  --sample_data_path data/online_eval_input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin \
  --inference_num_cases 1 \
  --inference_action_choice max \
  --steps 100 \
  --num_workers 20

```
