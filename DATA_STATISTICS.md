# 数据统计报告

生成时间: 2026-03-22
数据目录: `data/input_data/`

## 📊 数据总量

| 指标 | 数值 |
|------|------|
| 总文件数 | 5,301 个 .mbin |
| 总 Scenarios | 100,000 个 |
| 总 Steps (样本数) | **8,799,621 个** |
| 平均每个 Scenario | 88.0 steps |
| 数据大小 | 4.4 GB |

## 🗺️ 地图尺寸分布

| 地图尺寸 | 文件数 | Scenarios | Steps (样本) | 占比 | 平均 Steps/Scenario |
|---------|--------|-----------|-------------|------|-------------------|
| 32×32 | 2,356 | 46,957 (46.96%) | 3,184,263 (36.18%) | 36.2% | 67.8 |
| 64×64 | 2,945 | 53,043 (53.04%) | 5,615,358 (63.82%) | 63.8% | 105.9 |
| **总计** | **5,301** | **100,000** | **8,799,621** | **100%** | **88.0** |

**观察**:
- 64×64 地图的 scenario 更长（平均 105.9 steps），因为地图更大，路径更长
- 32×32 地图的 scenario 更短（平均 67.8 steps）
- 64×64 地图贡献了 63.8% 的训练样本

## 📂 训练/验证分割

假设 10% 用于验证（由 `create_offline_train_loaders` 自动分割）:

| 数据集 | Steps (样本数) | 占比 |
|--------|---------------|------|
| 训练集 | 7,919,658 | 90% |
| 验证集 | 879,963 | 10% |

## ⚙️ 推荐训练配置

### 方案 1: 标准配置 (推荐)

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 120000 \
  --offline_eval_interval_steps 2000 \
  --offline_save_interval_steps 2000 \
  --offline_inference_test_interval_steps 2000 \
  --batch_size 64 \
  --num_workers 20
```

- **可训练步数**: 最多 123,744 steps
- **推荐步数**: 120,000 steps (留余量)
- **预计时间**: 3-6 小时 (单 GPU A100)
- **适用场景**: 标准训练，平衡速度和效果

### 方案 2: 大 Batch Size

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 60000 \
  --offline_eval_interval_steps 2000 \
  --offline_save_interval_steps 2000 \
  --offline_inference_test_interval_steps 2000 \
  --batch_size 128 \
  --num_workers 20
```

- **可训练步数**: 最多 61,872 steps
- **推荐步数**: 60,000 steps
- **预计时间**: 2-4 小时 (单 GPU A100)
- **适用场景**: 快速训练，需要更大 GPU 内存

### 方案 3: 快速测试

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 10000 \
  --offline_eval_interval_steps 1000 \
  --offline_save_interval_steps 2000 \
  --batch_size 64 \
  --num_workers 20
```

- **预计时间**: 20-40 分钟
- **适用场景**: 验证配置、调试、快速迭代

### 方案 4: 小规模测试

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 100 \
  --offline_eval_interval_steps 50 \
  --batch_size 64 \
  --num_workers 4
```

- **预计时间**: 1-2 分钟
- **适用场景**: 验证数据加载、检查配置

## 📈 不同 Batch Size 的可训练步数

| Batch Size | 最多可训练步数 | 推荐步数 | 预计时间 (单 GPU A100) |
|-----------|--------------|---------|---------------------|
| 32 | 247,489 | 240,000 | 6-12 小时 |
| 64 | 123,744 | 120,000 | 3-6 小时 |
| 128 | 61,872 | 60,000 | 2-4 小时 |
| 256 | 30,936 | 30,000 | 1-2 小时 |

## ✅ 数据充足性分析

### 对比之前的估算

| 项目 | 估算值 | 实际值 | 差异 |
|------|--------|--------|------|
| 平均 Steps/Scenario | 50 | 88.0 | +76% |
| 总样本数 | 5,000,000 | 8,799,621 | +76% |
| 可训练步数 (bs=64) | 78,125 | 123,744 | +58% |

### 结论

✅ **数据量非常充足！**

- 实际数据比估算多 **76%**
- 可以训练 **120,000 steps** 而不会耗尽数据
- 可以使用更大的 batch_size 加速训练
- 数据分布合理（32×32 和 64×64 都有充足样本）

## 🎯 Agent 数量分布

虽然文件按 agent 数量分组存储，但训练时会按地图尺寸分组，所以 agent 数量的分布不影响训练。

数据包含以下 agent 数量:
- 16, 32, 64, 96, 128, 256, 512, 1024

这确保了模型能够学习处理不同密度的场景。

## 📝 注意事项

1. **训练/验证分割**:
   - 自动按文件数量的 10% 分割
   - 每个地图尺寸组独立分割
   - 32×32: ~2,120 训练文件, ~236 验证文件
   - 64×64: ~2,650 训练文件, ~295 验证文件

2. **数据不会重复使用**:
   - Step-based 模式下，每个样本只用一次
   - 当某个 dims 的数据耗尽时，会自动跳过
   - 所有 dims 耗尽时，训练自动停止

3. **随机采样**:
   - 每个 step 按权重随机选择 dims (32×32 或 64×64)
   - 权重基于文件数量: 32×32 (44.4%), 64×64 (55.6%)
   - DataLoader 内部也启用 shuffle

4. **性能优化**:
   - `num_workers=20`: 适合 20+ 核 CPU
   - `num_workers=4-8`: 适合 8-16 核 CPU
   - 如果 CPU 利用率低，可以增加 `num_workers`
   - 如果内存不足，可以减小 `num_workers`

## 🚀 快速开始

### 第一步: 小规模测试 (1-2 分钟)

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 100 \
  --offline_eval_interval_steps 50 \
  --batch_size 64 \
  --num_workers 4
```

验证:
- ✓ 数据加载正常
- ✓ Dims 随机采样工作
- ✓ Eval/save/inference 触发

### 第二步: 中等规模测试 (20-40 分钟)

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 10000 \
  --offline_eval_interval_steps 1000 \
  --offline_save_interval_steps 2000 \
  --batch_size 64 \
  --num_workers 20
```

验证:
- ✓ Loss 正常下降
- ✓ Checkpoint 保存正常
- ✓ TensorBoard 日志正常

### 第三步: 完整训练 (3-6 小时)

```bash
python train.py \
  --config config.offline.yaml \
  --offline_total_steps 120000 \
  --offline_eval_interval_steps 2000 \
  --offline_save_interval_steps 2000 \
  --offline_inference_test_interval_steps 2000 \
  --batch_size 64 \
  --num_workers 20
```

## 📊 监控训练

### TensorBoard

```bash
tensorboard --logdir runs
```

关键指标:
- `Loss/Train`: 训练 loss
- `Loss/Train_32x32`: 32×32 地图 loss
- `Loss/Train_64x64`: 64×64 地图 loss
- `Loss/Val`: 验证 loss
- `Inference/*`: Inference test 指标
- `Time/DataFetch_s_window_avg`: 数据加载时间
- `Time/Step_s_window_avg`: 训练步时间

### 预期输出

启动时:
```
Offline train group (32, 32): 2120 files
Offline train group (64, 64): 2650 files
```

训练中:
```
Offline Train (step-based): 100%|████| 120000/120000
Step 2000/120000, Training mean Loss: 0.1234
```

## 📚 相关文档

- [QUICKSTART_OFFLINE_STEP_BASED.md](QUICKSTART_OFFLINE_STEP_BASED.md) - 快速启动指南
- [README.md](README.md) - 完整文档
- [config.offline.yaml](config.offline.yaml) - 配置文件模板
- [test_offline_step_based.py](test_offline_step_based.py) - 自动化测试脚本
