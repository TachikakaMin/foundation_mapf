# Offline Step-based 训练快速启动指南

## 概述

你已经生成了 100,000 个 scenarios (4.4 GB, 5,301 个 .mbin 文件)，现在可以使用新的 step-based 模式进行训练。

## 核心特性

✅ **Step-based 训练**: 每个样本只用一次，不重复遍历
✅ **预生成数据**: 从 .mbin 文件读取，无需在线生成
✅ **按地图尺寸分组**: 相同尺寸的地图在同一个 batch
✅ **随机采样**: 两层随机确保充分打乱
  - 第一层: dims 组之间随机切换
  - 第二层: 每个 dims 组内部数据也随机

## 数据统计

```
总大小: 4.4 GB
总文件: 5,301 个 .mbin

地图尺寸分布:
  32×32: 2,356 files (44.4%)
  64×64: 2,945 files (55.6%)

Agent 数量分布:
  16 agents: 645 directories
  32 agents: 726 directories
  64 agents: 741 directories
  96 agents: 741 directories
  128 agents: 741 directories
  256 agents: 741 directories
  512 agents: 615 directories
  1024 agents: 351 directories
```

## 训练步数估算

假设:
- 100,000 scenarios
- 每个 scenario 平均 50 steps
- batch_size = 64

计算:
```
总样本数 = 100,000 × 50 = 5,000,000 samples
可训练步数 = 5,000,000 / 64 ≈ 78,125 steps
```

**建议**: `offline_total_steps = 70000` (留一些余量)

## 快速启动

### 方式 1: 修改配置文件（推荐）

编辑 `config.offline.yaml`:

```yaml
# 启用 step-based 模式
offline_total_steps: 70000  # 改为 > 0
offline_eval_interval_steps: 2000
offline_save_interval_steps: 2000
offline_inference_test_interval_steps: 2000

# 其他参数
batch_size: 64
num_workers: 20
learning_rate: 1.0e-5
```

然后运行:

```bash
python train.py --config config.offline.yaml
```

### 方式 2: 命令行覆盖

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 70000 \
  --offline_eval_interval_steps 2000 \
  --offline_save_interval_steps 2000 \
  --offline_inference_test_interval_steps 2000 \
  --batch_size 64 \
  --num_workers 20
```

### 方式 3: 小规模测试（推荐先运行）

在正式训练前，先跑一个小测试验证配置:

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 100 \
  --offline_eval_interval_steps 50 \
  --offline_save_interval_steps 50 \
  --offline_inference_test_interval_steps 50 \
  --batch_size 64 \
  --num_workers 4
```

这会在 1-2 分钟内完成，验证:
- 数据加载是否正常
- dims 分组是否正确
- 随机采样是否工作
- eval/save/inference 是否触发

## 预期输出

### 启动时

```
Offline train group (32, 32): 2120 files
Offline train group (64, 64): 2650 files

Runtime Config:
  dataset_mode: offline
  offline_total_steps: 70000
  offline_eval_interval_steps: 2000
  batch_size: 64
  ...
```

### 训练中

```
Offline Train (step-based): 100%|████████████| 70000/70000 [XX:XX:XX<00:00, XX.XXit/s]

Step 100/70000, Training mean Loss: 1.2345
Step 200/70000, Training mean Loss: 1.1234
...
Step 2000/70000, Training mean Loss: 0.9876
[Running validation...]
[Running inference test...]
[Saving checkpoint...]
```

### 数据耗尽时（如果提前耗尽）

```
Dims (32, 32) 数据耗尽，剩余可用 dims: 1
...
所有数据已耗尽，在 step 65432/70000 提前结束训练
```

## 监控训练

### TensorBoard

```bash
tensorboard --logdir runs
```

查看指标:
- `Loss/Train`: 每 100 steps 的训练 loss
- `Loss/Train_window_avg`: 窗口平均 loss
- `Loss/Train_32x32`: 32×32 地图的 loss
- `Loss/Train_64x64`: 64×64 地图的 loss
- `Loss/Val`: 验证 loss
- `Time/DataFetch_s_window_avg`: 数据加载时间
- `Time/Step_s_window_avg`: 训练步时间
- `Inference/*`: inference test 指标

### 日志文件

训练日志保存在 `runs/<timestamp>/`

## 常见问题

### Q1: 如何确认 step-based 模式已启用？

A: 查看启动日志中的 `Runtime Config`，确认 `offline_total_steps > 0`。训练进度条会显示 `Offline Train (step-based)`。

### Q2: 数据会重复使用吗？

A: 不会。step-based 模式下，每个样本只用一次。当某个 dims 的数据耗尽时，会自动标记并跳过。

### Q3: 如何调整 dims 采样权重？

A: 当前权重 = 该 dims 的文件数量。32×32 有 2356 files，64×64 有 2945 files，所以 64×64 被采样的概率约为 55.6%。这是自动计算的，无需手动调整。

### Q4: 如果数据提前耗尽怎么办？

A: 训练会自动停止并保存最后一个 checkpoint。你可以:
1. 生成更多数据
2. 减小 `offline_total_steps`
3. 增大 `batch_size`（减少总步数）

### Q5: 如何切换回 epoch-based 模式？

A: 设置 `offline_total_steps = 0`，然后使用 `--epochs`, `--eval_interval`, `--save_interval` 参数。

### Q6: 分布式训练如何启动？

A: 使用 `torchrun`:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config.offline.yaml \
  --offline_total_steps 70000 \
  --batch_size 8 \
  --distributed
```

注意: 分布式训练时，每个 GPU 的 batch_size 应该相应减小。

## 性能优化建议

### 数据加载

- `num_workers = 20`: 适合 20+ 核 CPU
- `num_workers = 4-8`: 适合 8-16 核 CPU
- 如果 CPU 利用率低，可以增加 `num_workers`
- 如果内存不足，可以减小 `num_workers`

### Batch Size

- 单 GPU (24GB): `batch_size = 64-128`
- 单 GPU (48GB): `batch_size = 128-256`
- 多 GPU: 总 batch_size = `batch_size × num_gpus`

### 训练速度

预期速度（单 GPU A100）:
- 数据加载: ~0.01-0.05 秒/batch
- 训练步: ~0.1-0.2 秒/batch
- 总速度: ~5-10 steps/秒

70,000 steps 预计耗时: **2-4 小时**

## 下一步

1. **小规模测试**: 运行 100 steps 验证配置
2. **中等规模测试**: 运行 1000 steps 检查 loss 下降
3. **完整训练**: 运行 70000 steps
4. **评估模型**: 使用 `eval_test.py` 评估 checkpoint

## 参考

- 完整文档: `README.md`
- 参数说明: `doc/train_args.py.md`
- 数据生成: `doc/tools/generate_offline_data.py.md`
- 在线训练对比: `config.online.yaml`
