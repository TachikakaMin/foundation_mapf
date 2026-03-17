# `train.py`

## 文件作用

这是项目的主训练入口。它负责：

- 解析训练参数
- 初始化单卡或分布式训练环境
- 构建 `UNet` 或 `CNN`
- 根据配置选择离线或在线训练数据
- 为验证集构建固定离线 `.mbin` 加载器
- 执行训练、验证、TensorBoard 记录和模型保存

## 主要函数

### `get_map_dims(file_path)`

从文件名中解析地图高宽，用于把训练和验证数据按尺寸分组。

### `group_files_by_dims(file_paths, min_map_size=32)`

按 `(height, width)` 对文件分组，并过滤过小地图。

### `create_offline_validation_loaders(args)`

构建固定离线验证集 `DataLoader`。

- 默认从 `args.dataset_path/**/*.mbin` 中切出每组前 `10%` 文件做验证
- 若显式传了 `--val_dataset_path`，则直接把该路径下的全部 `.mbin` 作为验证集

### `create_offline_train_loaders(args)`

保留旧训练模式：从 `args.dataset_path/**/*.mbin` 中按地图尺寸分组，并取每组后 `90%` 作为训练集。

### `create_online_train_loaders(args)`

在线训练入口：

- 从 `args.train_map_path/**/*.map` 扫描地图
- 按尺寸分组
- 为每一组创建 [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py) 的 `MAPFOnlineDataset`
- 返回训练 loader 以及用于步数分配的组权重

### `get_online_schedule(args)`

解析在线训练的 step-based 调度参数，包括：

- `online_total_steps`
- `online_eval_interval_steps`
- `online_save_interval_steps`
- `online_inference_test_interval_steps`

### `get_model_stats(model)`

统计可训练参数量和模型大小（按 `float32` 估算 MB）。

### `format_kv_table(rows)`

把配置项格式化成易读的两列表格字符串。

### `summarize_loader_groups(loaders, weights=None)`

把 train / val loader 的地图尺寸分组整理成一行摘要文本。

### `print_runtime_summary(args, model, device, train_loaders, train_loader_weights, val_loaders)`

在训练开始前打印一份运行参数表，并写入 TensorBoard。

表中包含：

- 数据模式和数据路径
- 训练 / 验证尺寸分组
- 模型类型、参数量和模型大小
- offline 下的 epoch 周期，或 online 下的 step 周期
- inference test 配置

### `evaluate_valid_loss(args, model, val_loader, loss_fn, device)`

在验证集上计算平均损失。

- 输入：
  - `model`: 需要返回 `(logits, prob)` 的模型
  - `val_loader`: 数据加载器
  - `loss_fn`: 逐像素损失函数，当前训练流程使用 `CrossEntropyLoss(reduction="none")`
  - `device`: `cpu` 或 `cuda`
- 输出：
  - 按智能体数量归一化后的验证损失

接口特点：

- 只统计 `mask == 1` 的位置
- 进度条显示依赖 `args.local_rank`

### `run_inference_test(args, model, sample_loader, device, epoch)`

训练中的显式 inference test。

它会：

- 从固定 `sample_loader` 中读取一个或多个首步样本
- 调用 [tools/path_formation.py](/home/yimin/research/RAILGUN/tools/path_formation.py) 做 rollout
- 记录并打印 `total_cost`、`ep_length`、`makespan`、`isr`、`csr`、`final_distance` 等指标
- 同时把单样本指标和平均指标写到 TensorBoard

### `train_offline(args, model, train_loaders, val_loaders, sample_loader, optimizer, loss_fn, device)`

离线训练主循环。

- 按 epoch 遍历固定 `.mbin` 数据
- `eval_interval` / `save_interval` / `inference_test_interval` 都按 epoch 生效

### `train_online(args, model, train_loaders, train_loader_weights, val_loaders, sample_loader, optimizer, loss_fn, device)`

在线训练主循环。

- 不再使用 epoch 作为训练进度单位
- 总进度由 `online_total_steps` 定义
- 每一步按组权重从不同地图尺寸 loader 中采样一个 batch
- 验证、保存和 inference test 都按 step 触发

### `train(args, model, train_loaders, train_loader_weights, val_loaders, sample_loader, optimizer, loss_fn, device)`

按 `dataset_mode` 在 `train_offline()` 和 `train_online()` 之间分发。

## 主程序入口

`if __name__ == "__main__":` 下的流程如下：

1. 从 [train_args.py](/home/yimin/research/RAILGUN/train_args.py) 读取参数
2. 根据 `--distributed` 决定是否初始化 `torch.distributed`
3. 创建 TensorBoard `SummaryWriter`
4. 固定随机种子
5. 根据 `--model` 初始化 [models/unet.py](/home/yimin/research/RAILGUN/models/unet.py) 或 [models/CNN.py](/home/yimin/research/RAILGUN/models/CNN.py)
6. 始终构建离线固定验证集
7. 根据 `--dataset_mode` 选择：
   - 离线 `.mbin` 训练
   - 在线 `MAPFOnlineDataset` 训练
8. 为 inference test 构建固定离线 `sample_loader`
9. 打印运行参数表
10. 调用 `train(...)`

## 输入输出与接口情况

### 训练数据接口

离线模式依赖 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py)。

在线模式依赖 [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py)。

两者都返回同样的样本字典：

- `feature`
- `action`
- `mask`
- `file_name`

### 模型接口

模型的 `forward(feature)` 必须返回：

- `logits`: `[B, action_dim, H, W]`
- `prob`: softmax 后概率图

### 输出

- TensorBoard 日志：`args.log_dir/<timestamp>/`
- 模型权重：
  - offline: `model_checkpoint_epoch_<N>.pth`
  - online: `model_checkpoint_step_<N>.pth`
- 启动时的 `Runtime Config` 参数表
- 训练中的 inference test 指标

## 用法

```bash
# 离线训练，单卡
python train.py --batch_size 64

# 离线训练，多卡
torchrun --nproc_per_node=8 train.py --batch_size 8 --distributed

# 在线训练，固定离线验证
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
  --steps 100 \
  --batch_size 64 \
  --num_workers 2
```
