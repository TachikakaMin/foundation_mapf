# Scaling Law 实验文档

## 概述

`scaling_law.py` 是统一的 scaling law 实验脚本，支持 **online** 和 **offline** 两种模式。通过传入不同的配置文件自动切换模式。

## 实验设计

### N 轴（模型大小）

5 个模型配置，通过 `first_layer_channels` × `blocks_per_stage` 控制：

| Label | first_layer_channels | blocks_per_stage | 参数量 |
|-------|---------------------|------------------|--------|
| XS    | 64                  | 0                | 31.0M  |
| S     | 64                  | 1                | 46.7M  |
| M     | 64                  | 2                | 78.2M  |
| L     | 64                  | 3                | 109.6M |
| XL    | 64                  | 4                | 141.0M |

### D 轴（数据量）

20 个 milestone steps：
```
[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000,
 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000, 100000]
```

每个模型训练一次到 100,000 steps，在每个 milestone 记录指标。

### 测试样本

12 个固定测试样本：
- 32×32 地图, 128 agents: 4 个场景
- 64×64 地图, 256 agents: 4 个场景
- 64×64 地图, 512 agents: 2 个场景（高密度）
- 64×64 地图, 1024 agents: 2 个场景（OOD 极端密度）

## 使用方法

### 模式切换

脚本通过配置文件名自动检测模式：
- `config.online.yaml` → online 模式
- `config.offline.yaml` → offline 模式

### 1. Online 模式

```bash
# Dry run（查看命令）
python scaling_law.py --config config.online.yaml --dry_run

# 运行所有模型
python scaling_law.py --config config.online.yaml

# 只运行部分模型
python scaling_law.py --config config.online.yaml --models XS S

# 自定义 milestone steps
python scaling_law.py --config config.online.yaml --steps 10000 20000 40000 70000 100000

# 指定 Python 解释器
python scaling_law.py --config config.online.yaml --python /path/to/python

# 自定义输出路径
python scaling_law.py --config config.online.yaml \
  --log_dir runs/my_scaling_law \
  --output_csv my_results.csv
```

### 2. Offline 模式

```bash
# Dry run（查看命令）
python scaling_law.py --config config.offline.yaml --dry_run

# 运行所有模型
python scaling_law.py --config config.offline.yaml

# 只运行部分模型
python scaling_law.py --config config.offline.yaml --models XS S

# 自定义 milestone steps
python scaling_law.py --config config.offline.yaml --steps 10000 20000 40000 70000 100000
```

**预计时间：** 15-30 小时（取决于硬件）

## 并行运行

如果有多个 GPU，可以并行运行不同的模型：

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python scaling_law.py \
  --config config.offline.yaml \
  --models XS S \
  --log_dir runs/scaling_law_gpu0 \
  --output_csv results_gpu0.csv

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python scaling_law.py \
  --config config.offline.yaml \
  --models M L \
  --log_dir runs/scaling_law_gpu1 \
  --output_csv results_gpu1.csv

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python scaling_law.py \
  --config config.offline.yaml \
  --models XL \
  --log_dir runs/scaling_law_gpu2 \
  --output_csv results_gpu2.csv
```

然后合并结果：

```python
import pandas as pd

df1 = pd.read_csv("results_gpu0.csv")
df2 = pd.read_csv("results_gpu1.csv")
df3 = pd.read_csv("results_gpu2.csv")

df_combined = pd.concat([df1, df2, df3], ignore_index=True)
df_combined.to_csv("scaling_law_results.csv", index=False)
```

## 输出格式

### CSV 文件

输出 CSV 文件包含以下列：

**基本信息：**
- `run_name`: 运行名称（如 `XS_flc64_bps0_steps5000` 或 `XS_flc64_bps0_steps5000_offline`）
- `source_run_name`: 源运行名称（如 `XS_flc64_bps0_steps100000` 或 `XS_flc64_bps0_steps100000_offline`）
- `label`: 模型标签（XS, S, M, L, XL）
- `first_layer_channels`: 首层通道数
- `blocks_per_stage`: 每个 stage 的 block 数
- `n_params`: 模型参数量
- `total_steps`: 当前 milestone step
- `max_total_steps`: 最大训练 step
- `data_tokens`: 数据量（steps × batch_size）
- `mode`: 训练模式（online 或 offline）
- `exit_code`: 退出码（0 表示成功）
- `wall_time_s`: 训练时间（秒）

**训练指标：**
- `train_loss`: 训练 loss
- `reported_step`: 报告的 step
- `reported_total_steps`: 报告的总 step

**验证指标：**
- `val_loss`: 验证 loss（所有地图的平均）
- `val_loss_32x32`: 32×32 地图的验证 loss
- `val_loss_64x64`: 64×64 地图的验证 loss

**Inference 指标：**
- `infer_isr`: Individual Success Rate
- `infer_csr`: Collective Success Rate
- `infer_makespan`: 完成时间
- `infer_total_cost`: 总成本
- `infer_final_distance`: 最终距离
- `infer_case_0_isr`, `infer_case_0_csr`, ...: 每个测试样本的详细指标

### 日志目录

```
runs/scaling_law/
└── online_20260322-123456/  # 或 offline_20260322-123456
    ├── XS_flc64_bps0_steps100000/  # 或 XS_flc64_bps0_steps100000_offline
    │   └── 20260322-123456-xxxxx/
    │       ├── events.out.tfevents.xxx (TensorBoard)
    │       ├── model_checkpoint_step_5000.pth
    │       ├── model_checkpoint_step_10000.pth
    │       └── ...
    ├── S_flc64_bps1_steps100000/
    │   └── ...
    └── ...
```

## 对比 Online 和 Offline

如果同时运行了 online 和 offline 版本，可以对比结果：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据（假设已经合并了 online 和 offline 的结果）
df = pd.read_csv("scaling_law_results.csv")

# 对比每个模型
for label in ["XS", "S", "M", "L", "XL"]:
    df_model = df[df["label"] == label]

    df_online = df_model[df_model["mode"] == "online"]
    df_offline = df_model[df_model["mode"] == "offline"]

    plt.figure(figsize=(10, 6))
    plt.plot(df_online["total_steps"], df_online["val_loss"],
             label="Online", marker="o", linewidth=2)
    plt.plot(df_offline["total_steps"], df_offline["val_loss"],
             label="Offline", marker="s", linewidth=2)
    plt.xlabel("Training Steps")
    plt.ylabel("Validation Loss")
    plt.title(f"Model {label} - Online vs Offline")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"comparison_{label}.png", dpi=300, bbox_inches="tight")
    plt.close()
```

## 注意事项

### 1. Online 模式

- 数据实时生成，训练速度较慢
- 适合验证模型在真实场景下的表现
- 需要确保 LACAM solver 正常工作

### 2. Offline 模式

- 数据从预生成的 .mbin 文件读取
- 训练速度较快
- 需要确保数据充足：
  - 你的数据：8,799,621 个训练样本
  - 最大实验：100,000 steps × 64 batch_size = 6,400,000 样本
  - ✓ 数据充足，不会提前耗尽

### 3. 磁盘空间

每个模型的 checkpoint：
- 每 5000 steps 保存一次
- 100,000 steps → 20 个 checkpoint
- 每个 checkpoint ~500 MB（141M params）
- 总共：20 × 500 MB × 5 models = 50 GB

**建议：** 至少预留 100 GB 磁盘空间

### 4. 时间估算

单个模型（100,000 steps）：
- XS (31M): ~3 小时
- S (47M): ~4 小时
- M (78M): ~5 小时
- L (110M): ~6 小时
- XL (141M): ~7 小时

**总时间：** ~25 小时（串行运行）

### 5. 内存要求

- GPU 内存：至少 24 GB（XL 模型）
- 系统内存：至少 64 GB（20 个 DataLoader workers）

## 参考

- 训练脚本：[train.py](../train.py)
- Online 配置：[config.online.yaml](../config.online.yaml)
- Offline 配置：[config.offline.yaml](../config.offline.yaml)
- 数据统计：[DATA_STATISTICS.md](../DATA_STATISTICS.md)
- Offline 快速启动：[QUICKSTART_OFFLINE_STEP_BASED.md](../QUICKSTART_OFFLINE_STEP_BASED.md)
