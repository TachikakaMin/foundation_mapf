# `train.py`

## 文件作用

这是项目的主训练入口。它负责：

- 解析训练参数
- 初始化单卡或分布式训练环境
- 构建 `UNet` 或 `CNN`
- 根据配置选择离线或在线训练数据
- 为验证集构建固定离线 `.mbin` 加载器
- 执行训练、验证、TensorBoard 记录和模型保存
- 在在线模式结束时显式清理自定义 online loader

## 主要函数

### `get_map_dims(file_path)`

从文件名中解析地图高宽，用于把训练和验证数据按尺寸分组。

### `group_files_by_dims(file_paths, min_map_size=32)`

按 `(height, width)` 分组，并过滤过小地图。

### `create_offline_validation_loaders(args)`

构建固定离线验证集 `DataLoader`。

- 默认从 `args.dataset_path/**/*.mbin` 中切出每组前 `10%` 文件做验证
- 若显式传了 `--val_dataset_path`，则直接把该路径下的全部 `.mbin` 作为验证集

### `create_offline_train_loaders(args)`

离线训练入口：

- 从 `args.dataset_path/**/*.mbin` 扫描训练数据
- 按地图尺寸分组
- 对每组保留后 `90%` 文件作为训练集
- 返回各组 `DataLoader` 与分组权重

### `create_online_train_loaders(args)`

在线训练入口：

- 从 `args.train_map_path/**/*.map` 扫描地图
- 按尺寸分组
- 为每一组创建一个 [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py) 的 `MAPFOnlineBufferLoader`
- 把 `args.num_workers` 传给 `buffer_workers`
- 把 `args.online_buffer_size` / `args.online_buffer_timeout_sec` 传给在线 step buffer
- 启动每个 loader 的后台生产进程
- 返回训练 loader 以及用于 online 采样的组权重

这里的在线 loader 不是标准 `DataLoader`，而是一个自定义 batch 迭代器。

### `get_online_schedule(args)`

解析在线训练的 step-based 调度参数，包括：

- `online_total_steps`
- `online_eval_interval_steps`
- `online_save_interval_steps`
- `online_inference_test_interval_steps`

### `make_dataloader_kwargs(batch_size, num_workers, *, sampler=None, shuffle=False)`

统一生成离线 `DataLoader` 参数。

当前行为：

- `num_workers > 0` 时启用 `persistent_workers=True`
- `num_workers > 0` 时设置 `prefetch_factor=2`
- CUDA 可用时启用 `pin_memory=True`

### `close_loader_collection(loaders)`

对 loader 字典做统一清理。

如果某个 loader 提供 `close()`，这里会主动调用。在线模式的 `MAPFOnlineBufferLoader` 就依赖这个清理钩子。

### `print_runtime_summary(...)`

在训练开始前打印一份运行参数表，并写入 TensorBoard。

在线模式下，额外会显示：

- `bilinear`
- `first_layer_channels`
- `blocks_per_stage`
- `online_total_steps`
- `online_eval_interval_steps`
- `online_save_interval_steps`
- `online_inference_interval_steps`
- `online_time_limit_sec`
- `online_retry_limit`
- `online_buffer_size`
- `online_buffer_timeout_sec`
- `online_buffer_unit=steps_per_dim_group`

### `train_online(...)`

在线训练主循环。

- 不再使用 epoch 作为训练进度单位
- 总进度由 `online_total_steps` 定义
- 每一步按组权重从不同地图尺寸 loader 中采样一个 batch
- 验证、保存和 inference test 都按 optimizer step 触发

## TensorBoard 指标

训练阶段同时记录 step 级指标和窗口/epoch 聚合指标，offline 和 online 共用同一套核心命名。

- `Optimization/LR`：当前学习率
- `Loss/TrainStep`：当前训练 step 的损失
- `Loss/TrainStep_<H>x<W>`：当前训练 step 所属尺寸分组的损失
- `Loss/Train`：online 下为每个训练 step 的损失；offline 下仍是每个 epoch 的平均训练损失
- `Loss/Train_window_avg`：online 训练窗口平均损失
- `Loss/Train_<H>x<W>`：窗口平均或 epoch 平均的尺寸分组训练损失
- `Entropy/Train`：当前训练 step 的平均 entropy
- `Entropy/Train_window_avg` / `Entropy/Train_epoch_avg`：聚合后的 entropy
- `Time/DataFetch_s`：取 batch 的耗时
- `Time/Step_s`：完整训练 step 的耗时
- `Time/DataFetch_s_window_avg` / `Time/DataFetch_s_epoch_avg`：聚合后的取 batch 耗时
- `Time/Step_s_window_avg` / `Time/Step_s_epoch_avg`：聚合后的 step 耗时
- `GPU/memory_allocated_mb`：当前已分配显存
- `GPU/memory_reserved_mb`：当前已保留显存
- `GPU/max_memory_allocated_mb`：当前 step 期间峰值已分配显存
- `GPU/max_memory_reserved_mb`：当前 step 期间峰值已保留显存
- `Loss/Val` 和 `Loss/Val_<H>x<W>`：验证损失
- `Inference/...` 和 `InferenceSummary/...`：inference test rollout 指标
- `InferenceVideo/case_0`：第一个固定 inference 样本的 rollout 轨迹视频

## 主程序入口

`if __name__ == "__main__":` 下的流程如下：

1. 从 [train_args.py](/home/yimin/research/RAILGUN/train_args.py) 读取参数
2. 根据 `--distributed` 决定是否初始化 `torch.distributed`
3. 创建带时间戳的运行目录，并在总训练步数 `>= 1000` 时启用 TensorBoard `SummaryWriter`
4. 固定随机种子
5. 根据 `--model` 初始化 [models/unet.py](/home/yimin/research/RAILGUN/models/unet.py) 或 [models/CNN.py](/home/yimin/research/RAILGUN/models/CNN.py)
6. 始终构建离线固定验证集
7. 根据 `--dataset_mode` 选择离线或在线训练 loader
8. 为 inference test 构建固定离线 `sample_loader`
9. 打印 `Runtime Config`
10. 调用 `train(...)`
11. 在 `finally` 中关闭在线 loader、flush/close writer
12. 在线模式下调用 `os._exit(0)`，避免 `mp.Queue` 内部线程在解释器关闭阶段死锁

## 输入输出与接口情况

### 训练数据接口

离线模式依赖 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py)。

在线模式依赖 [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py) 的 `MAPFOnlineBufferLoader`。

两者最终都返回同样的 batch 字典：

- `feature`
- `action`
- `mask`
- `file_name`

### 模型接口

模型的 `forward(feature)` 必须返回：

- `logits`: `[B, action_dim, H, W]`
- `prob`: softmax 后概率图

### 输出

- 运行目录：`args.log_dir/<timestamp>/`
- TensorBoard 日志：仅当总训练步数 `>= 1000` 时写入到 `args.log_dir/<timestamp>/`
- 模型权重：
  - offline: `model_checkpoint_epoch_<N>.pth`
  - online: `model_checkpoint_step_<N>.pth`
- 启动时的 `Runtime Config` 参数表
- 训练中的 inference test 指标
- TensorBoard 中的 `Args` 和 `RuntimeConfig` 文本快照

## 用法

```bash
# 推荐：直接加载配置
python train.py --config config.offline.yaml
python train.py --config config.online.yaml

# 配置 + CLI 覆盖
python train.py --config config.online.yaml --batch_size 32 --online_total_steps 50000

# 离线训练，单卡
python train.py --batch_size 64

# 离线训练，多卡
torchrun --nproc_per_node=8 train.py --batch_size 8 --distributed

# 在线训练，固定离线验证
python train.py \
  --config config.online.yaml \
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
  --batch_size 64 \
  --num_workers 20
```
