# RAILGUN: A Unified Convolutional Policy for Multi-Agent Path Finding Across Different Environments and Tasks

[Yimin Tang*](https://sites.google.com/view/yimintang), [Xiao Xiong*](https://openreview.net/profile?id=~Xiao_Xiong2), [Jingyi Xi](https://openreview.net/profile?id=~Jingyi_Xi1), [Jiaoyang Li](https://jiaoyangli.me/), [Erdem Bıyık](https://ebiyik.github.io/), [Sven Koenig](https://idm-lab.org/)

### TL;DR: We present the first centralized learning-based method for MAPF, called RAILGUN, which generates actions based on maps rather than individual agents.

Our insight is that in a valid MAPF solution, there will be no collision, which means there can be at most one agent in each map grid cell in each timestep. At any timestep, each agent chooses one of the five edges of its grid cell as its action. Therefore, if we remove all edges that the agents do not use at each timestep, we find that a valid MAPF solution can be viewed as a series of specialized graphs.

## 🚀 Performance Optimizations

This implementation includes high-performance C++ tools for faster data processing:

- **🔄 Multi-threaded Path Converter**: C++17 implementation with **5-10x speed improvement** over Python
- **⚡ Optimized Extensions**: Native C++ extensions for feature construction
- **💾 Memory Efficient**: Smart memory management for large-scale datasets
- **🎯 Production Ready**: Robust error handling and comprehensive logging



<p align="center">
<img width="1021" height="447" alt="image" src="https://github.com/user-attachments/assets/572f1f5e-9bc4-49af-9cf7-a761a4b11c87" />
</p>

<p align="center">
  <img width="504" height="501" alt="image" src="https://github.com/user-attachments/assets/d55687b9-3cdb-4517-ab2d-9aa6712a9fc5" />
</p>

## Abstract

Multi-Agent Path Finding (MAPF), which focuses on finding collision-free paths for multiple robots, is crucial for applications ranging from aerial swarms to warehouse automation. Solving MAPF is NP-hard so learning-based approaches for MAPF have gained attention, particularly those leveraging deep neural networks. Nonetheless, despite the community's continued efforts, all learning-based MAPF planners still rely on decentralized planning due to variability in the number of agents and map sizes. We have developed the first centralized learning-based policy for MAPF problem called RAILGUN. RAILGUN is not an agent-based policy but a map-based policy. By leveraging a CNN-based architecture, RAILGUN can generalize across different maps and handle any number of agents. We collect trajectories from rule-based methods to train our model in a supervised way. In experiments, RAILGUN outperforms most baseline methods and demonstrates great zero-shot generalization capabilities on various tasks, maps and agent numbers that were not seen in the training dataset.

## Model

[Link](https://drive.google.com/file/d/1r7tkbSBhterp3JK-yg6zsbDayfJketXe/view?usp=sharing)

## Installation
```bash
pip install torch numpy tqdm tensorboard psutil matplotlib pyyaml
apt update -y
apt install parallel -y
brew install parallel
cd data_generation_LACAM
git clone --recursive https://github.com/Kei18/lacam3.git && cd lacam3
cmake -B build && cmake --build build -j 16
cd ../../
```

## C++ Extensions and Tools Compilation

This project includes C++ extensions and high-performance tools. To compile them:

```bash
# Make sure you are in the correct conda environment
conda activate py310  # or your preferred Python environment

# Compile all C++ extensions and tools
cd tools
bash build.sh build
cd ..

# Or compile only specific components:
cd tools
bash build.sh converter    # 仅编译路径转换工具
bash build.sh rebuild      # 清理并重新构建所有组件
cd ..
```

**Available Tools:**
- 🔄 **路径转换工具**: 高性能多线程.path到.mbin转换
- ⚡ **C++扩展**: 高性能特征构建和数据处理
- 🧠 **在线LACAM扩展**: 训练时直接在内存中生成轨迹场景

**Important Notes:**
- The build script automatically detects and uses the currently activated Python environment
- Make sure to activate the correct conda environment before building
- The compiled extensions will be compatible with your current Python version
- C++ tools provide 5-10x performance improvement over Python versions

## Documentation

Project source documentation is under `doc/`, and the directory structure mirrors the source tree.

- Source file `models/unet.py` maps to `doc/models/unet.py.md`
- Source file `tools/build.sh` maps to `doc/tools/build.sh.md`
- Source file `data_generation_LACAM/maze_generator.py` maps to `doc/data_generation_LACAM/maze_generator.py.md`
- Source file `MAPF_online_dataset.py` maps to `doc/MAPF_online_dataset.py.md`
- Source file `tools/extensions/lacam_online_native.cpp` maps to `doc/tools/extensions/lacam_online_native.cpp.md`
- Source file `tools/profile_online_data.py` maps to `doc/tools/profile_online_data.py.md`
- Source file `scaling_law.py` maps to `doc/scaling_law.py.md`

The documentation currently covers the repository's project-owned source files and scripts, including Python modules, shell scripts, C++ sources, and CMake files. The index is available at `doc/README.md`.

## Training Workflows

This repo now has two distinct training workflows:

- `offline`: pre-generate trajectories, convert them to `.mbin`, then train
- `online`: generate training trajectories on the fly during training, but still use a fixed offline validation set

Current note:

- `train.py` now supports direct YAML loading with `--config`
- CLI arguments still override config values when you need a one-off change
- [config.offline.yaml](/home/yimin/research/RAILGUN/config.offline.yaml) and [config.online.yaml](/home/yimin/research/RAILGUN/config.online.yaml) are the recommended training entry points
- each run uses a timestamped run directory under `runs/`
- TensorBoard writes `Args` and `RuntimeConfig` for each run, but auto-disables when total training steps are below `1000`

### Offline: From Data Prep to Training

#### Step 1: Build C++ tools and extensions

```bash
cd tools
bash build.sh build
cd ..
```

#### Step 2: Generate maps

```bash
bash gen_mapfile.sh
```

Expected output:

- maps under `data/map_files/...`

Notes:

- `gen_mapfile.sh` now runs parameter groups in parallel with a progress bar
- by default it reserves `2` CPU cores and does not try to use the full machine
- you can override concurrency with `PARALLEL_JOBS=<N>` or `RESERVED_CORES=<N>`

#### Step 3: Precompute distance maps

```bash
python -m tools.precompute_distance_maps data/map_files
```

Expected output:

- distance maps under `data/distance_maps/...`

#### Step 4: Generate offline training data

**Option A: Parallel generation (Recommended)**

Use the new parallel data generation script for fast, efficient data creation:

```bash
python tools/generate_offline_data.py \
    --output_dir data/input_data \
    --num_scenarios 100000 \
    --workers 20 \
    --time_limit 5 \
    --retry_limit 5 \
    --agent_counts 16 32 64 96 128 256 512 1024 \
    --map_pattern "data/map_files/maze-*/*.map" \
    --scenarios_per_file 500
```

This will:
- Generate 100k scenarios in parallel using 20 workers
- Support all agent counts including 512 and 1024 (for high-density scenarios)
- Use both 32x32 and 64x64 maps
- Take approximately 7-15 minutes
- Output ~5-10 GB of .mbin files

**Option B: Traditional path-based generation**

Generate path files with LACAM, then convert:

```bash
# Generate path files
bash gen_pathfile.sh

# Convert to .mbin (C++ converter recommended)
cd tools
bash build.sh converter
cd ..
./tools/convert_path_to_mbin data/path_files

# Or use Python fallback
python -m tools.convert_lacam_path_to_bin data/path_files
```

Expected output:

- training / validation `.mbin` under `data/input_data/...`

**Notes:**
- Option A is faster and more flexible for large-scale data generation
- Option B is useful if you already have .path files or need specific scenarios
- Both produce compatible .mbin files for training

#### Step 5: Check required offline directories

Before training, make sure these exist:

- `data/map_files`
- `data/distance_maps`
- `data/input_data`

#### Step 6: Launch offline training

Recommended: load the offline config directly.

```bash
python train.py --config config.offline.yaml
```

Override a small number of values from CLI when needed:

```bash
python train.py --config config.offline.yaml --batch_size 32 --epochs 50
```

Single GPU:

```bash
python train.py \
  --dataset_mode offline \
  --dataset_path data/input_data \
  --batch_size 64 \
  --epochs 100 \
  --eval_interval 2 \
  --save_interval 2
```

Multi-GPU:

```bash
torchrun --nproc_per_node=8 train.py \
  --config config.offline.yaml \
  --dataset_mode offline \
  --dataset_path data/input_data \
  --batch_size 8 \
  --distributed
```

### Online: From Data Prep to Training

Online mode does not require pre-generated training `.mbin`, but it still requires maps, distance maps, and a small fixed offline validation / test set.

#### Step 1: Build C++ tools and extensions

```bash
cd tools
bash build.sh build
cd ..
```

#### Step 2: Generate maps

Use the same map generation step as offline mode:

```bash
bash gen_mapfile.sh
```

The result must be written to `data/map_files/...`.

#### Step 3: Precompute distance maps for those maps

```bash
python -m tools.precompute_distance_maps data/map_files
```

Expected output:

- distance maps under `data/distance_maps/...`

#### Step 4: Generate a small fixed validation / test set

Online training still validates on offline `.mbin`, but you do not need the full offline dataset. Use the dedicated script below to build a much smaller fixed set from a subset of existing maps:

```bash
bash gen_online_testset.sh
```

By default this script will:

- select a small subset of maps from `data/map_files`
- precompute distance maps only for those selected maps if needed
- generate a small number of `.path` files
- convert them into `.mbin`
- write the result to `data/online_eval_input_data`

Useful overrides:

```bash
MAX_MAPS_TOTAL=8 SEEDS_PER_AGENT=1 bash gen_online_testset.sh
MAX_MAPS_TOTAL=16 AGENT_COUNTS="64 32 16" SEEDS_PER_AGENT=2 bash gen_online_testset.sh
```

Expected output:

- validation `.mbin` under `data/online_eval_input_data/...` by default

#### Step 5: Check required online directories

Before training, make sure these exist:

- `data/map_files`
- `data/distance_maps`
- `data/online_eval_input_data` or your custom validation root

#### Step 6: Launch online training

Recommended: load the online config directly.

```bash
python train.py --config config.online.yaml
```

Override a small number of values from CLI when needed:

```bash
python train.py --config config.online.yaml --batch_size 32 --online_total_steps 50000
```

```bash
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
  --batch_size 64 \
  --num_workers 20
```

Online mode key idea:

- training progress is controlled by optimizer steps, not epochs
- validation / save / inference test are also step-based
- each map-size group owns one online step buffer
- `online_buffer_size` is measured in step samples, not trajectories
- online `num_workers` means scenario producer processes; it is now the only worker-count knob

### Where To Change Parameters

训练参数统一定义在 `train_args.py`，训练入口在 `train.py`。推荐做法是直接修改对应的 YAML，然后用 `--config` 启动；命令行只用于临时覆盖。

仓库根目录现在提供了两套默认模板：

- [config.offline.yaml](/home/yimin/research/RAILGUN/config.offline.yaml): 离线训练 / 离线验证
- [config.online.yaml](/home/yimin/research/RAILGUN/config.online.yaml): 在线训练 / 离线验证

[config.yaml](/home/yimin/research/RAILGUN/config.yaml) 目前保留为默认离线模板的兼容入口。

直接启动方式：

```bash
python train.py --config config.offline.yaml
python train.py --config config.online.yaml
```

配置加载规则：

- `--config` 先提供默认值
- CLI 再覆盖同名字段
- 训练开始时 `Runtime Config` 会打印最终生效值和 `config_source`

训练启动后，`train.py` 会打印一份 `Runtime Config` 表，里面会直接显示当前模型大小、数据模式、训练周期、验证周期和 inference test 配置。

### Training Parameter Table

| Category | Args | Meaning |
| --- | --- | --- |
| Data | `--dataset_mode` | `offline` 读 `.mbin`；`online` 训练时在线生成 |
| Data | `--dataset_path` | 离线训练/验证的 `.mbin` 根目录 |
| Data | `--val_dataset_path` | 固定离线验证集根目录，不填则回退到 `--dataset_path` |
| Data | `--train_map_path` | 在线训练时扫描 `.map` 的根目录 |
| Data | `--num_workers` | 统一 worker 数：offline 为 DataLoader workers，online 为 scenario producer 进程数，同时复用于验证/样例 DataLoader |
| Online train | `--online_total_steps` | 在线训练总优化步数 |
| Online train | `--online_eval_interval_steps` | 在线训练按多少步做一次验证 |
| Online train | `--online_save_interval_steps` | 在线训练按多少步保存一次 checkpoint |
| Online train | `--online_inference_test_interval_steps` | 在线训练按多少步做一次 inference test |
| Online train | `--online_time_limit_sec` | 单次 LACAM 在线生成的时间上限 |
| Online train | `--online_retry_limit` | 在线生成失败时的重试次数 |
| Online train | `--online_buffer_size` | 在线 step buffer 容量，单位是 step，按地图尺寸分组分别生效 |
| Online train | `--online_buffer_timeout_sec` | 在线训练消费者等待 step buffer 的超时时间 |
| Optimization | `--epochs` | 只用于离线模式 |
| Optimization | `--batch_size` | batch size |
| Optimization | `--learning_rate` | 学习率 |
| Optimization | `--weight_decay` | AdamW 的权重衰减 |
| Training loop | `--eval_interval` | 离线模式下每隔多少个 epoch 做一次验证 loss |
| Training loop | `--save_interval` | 离线模式下每隔多少个 epoch 保存一次 checkpoint |
| Model | `--model` | `unet` 或 `cnn` |
| Model | `--first_layer_channels` | `UNet` 首层基础通道数，是主要模型规模控制旋钮之一 |
| Model | `--blocks_per_stage` | 每个 UNet stage 的 ResBlock 数；`0` 表示兼容旧版 DoubleConv，也是模型规模控制旋钮之一 |
| Model | `--feature_dim` | 输入特征维度 |
| Model | `--feature_type` | 特征构造方式，例如 `gradient` |
| Model | `--action_dim` | 动作类别数，默认 5 |
| Inference test | `--sample_data_path` | 训练中固定用于 inference test 的离线 `.mbin` 样本 |
| Inference test | `--inference_num_cases` | 每次 inference test 跑多少个固定样本 |
| Inference test | `--inference_test_interval` | 离线模式下每隔多少个 epoch 跑一次 inference test；`0` 表示复用 `--eval_interval` |
| Inference test | `--inference_action_choice` | inference rollout 用 `sample` 还是 `max` 选动作 |
| Inference test | `--steps` | inference rollout 的最大步数 |

### Online Data Path

在线训练当前不是直接把 `MAPFOnlineDataset` 塞进 PyTorch `DataLoader`。主路径是一个自定义的 `MAPFOnlineBufferLoader`：

- 多个 producer 进程持续调用 LACAM 生成 raw scenario
- 主进程 builder 线程把 scenario 展开成逐 step 样本
- 训练循环持续从 step buffer 取 batch

这几个参数的含义最容易混：

- `online_buffer_size`：step buffer 容量，单位是 `step`，不是 trajectory
- `num_workers`：在线模式下表示 producer 进程数
- `online_buffer_timeout_sec`：消费者等待 step buffer 的超时

### Profile Online Data

如果想单独看数据端吞吐，不必直接起完整训练，可以先跑：

```bash
python tools/profile_online_data.py --config config.online.yaml

python tools/profile_online_data.py \
  --config config.online.yaml \
  --num_workers 20 \
  --batch_size 64 \
  --batches 50 \
  --warmup 5
```

这个脚本会输出 batch fetch latency 的均值和分位数；加上 `--scenario_samples` 还能直接看 raw scenario 生成时间。

### Scaling Law Sweep

仓库里现在还提供了一个小型 scaling law 扫描脚本：`scaling_law.py`。

它会基于一份 online 配置，批量扫两条轴：

- 模型大小轴 `N`：通过 `first_layer_channels × blocks_per_stage` 控制
- 数据量轴 `D`：通过一次长训练里的 milestone `step` 控制

默认行为：

- 每个模型只训练一次到最大 milestone（默认 100k steps）
- 自动选择一个能覆盖这些 milestone 的 step interval 来驱动 eval / save / inference
- 每个 milestone 记录参数量、训练 loss、验证 loss 和 inference summary
- 每跑完一组就把结果追加写到 CSV，方便中断后保留结果

**更新**：现在支持更密集的测试：
- Milestone 采样点：每 5k 一次（5k, 10k, 15k, ..., 100k），共 20 个点
- Inference test cases：12 个固定场景，覆盖多种地图大小和 agent 密度
  - 32x32 × 128 agents：4 个场景
  - 64x64 × 256 agents：4 个场景
  - 64x64 × 512 agents：2 个场景
  - 64x64 × 1024 agents：2 个场景（OOD 测试）

常见用法：

```bash
python scaling_law.py --config config.online.yaml --dry_run

python scaling_law.py \
  --config config.online.yaml \
  --python /home/yimintan/anaconda3/envs/py38/bin/python \
  --models XS S M \
  --output_csv scaling_law_results.csv
```

### Inference Test During Training

训练中现在有一个显式的 inference test，用固定离线样本做 rollout，检查模型是否真的会“走路”，而不只是看 supervised loss。

- 样本来源：`--sample_data_path`
- rollout 步数：`--steps`
- 触发周期：
  - `offline` 用 `--inference_test_interval`
  - `online` 用 `--online_inference_test_interval_steps`
- 输出指标：`total_cost`、`ep_length`、`makespan`、`isr`、`csr`、`final_distance`、`avg_density`、`total_time`、`throughput`

```bash
python train.py \
  --config config.online.yaml \
  --dataset_mode online \
  --train_map_path data/map_files \
  --val_dataset_path data/online_eval_input_data \
  --online_total_steps 200000 \
  --online_inference_test_interval_steps 4000 \
  --sample_data_path data/online_eval_input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin \
  --inference_num_cases 1 \
  --inference_action_choice max \
  --steps 100
```

### Online vs Offline Config

- `offline`：核心单位还是 `epochs`，因为训练集是固定离线 `.mbin`
- `online`：核心单位是总训练步数 `online_total_steps`
- `online` 的验证、保存和 inference test 也全部按 step 配置，不再复用 epoch 语义

## Evaluation Test

如果你评估的是非默认宽度/深度的 UNet checkpoint，例如 scaling law 扫描出来的模型，记得把训练时一致的结构参数也传给 `eval_test.py`，尤其是：

- `--first_layer_channels`
- `--blocks_per_stage`
- `--bilinear`

```bash
python eval_test.py \
  --model_path model_checkpoint_epoch_4.pth \
  --dataset_paths data/input_data/maze-32-32-60-1-75/maze-32-32-60-1-75-0-16/maze-32-32-60-1-75-0-16-1.bin \
  --first_layer_channels 64 \
  --blocks_per_stage 1 \
  --show
```
