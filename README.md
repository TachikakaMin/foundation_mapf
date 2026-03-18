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


## Installation
```bash
pip install torch numpy tqdm tensorboard psutil matplotlib
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

The documentation currently covers the repository's project-owned source files and scripts, including Python modules, shell scripts, C++ sources, and CMake files. The index is available at `doc/README.md`.

## Training Workflows

This repo now has two distinct training workflows:

- `offline`: pre-generate trajectories, convert them to `.mbin`, then train
- `online`: generate training trajectories on the fly during training, but still use a fixed offline validation set

Current note:

- `train.py` still starts from CLI arguments
- [config.offline.yaml](/home/yimin/research/RAILGUN/config.offline.yaml) and [config.online.yaml](/home/yimin/research/RAILGUN/config.online.yaml) are readable templates; use them as the source of truth when assembling your training command

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
- you can override concurrency with `PARALLEL_JOBS=<N>`

#### Step 3: Generate path files with LACAM

```bash
bash gen_pathfile.sh
```

Expected output:

- path files under `data/path_files/...`

#### Step 4: Convert `.path` to `.mbin` and precompute distance maps

Recommended C++ path converter:

```bash
cd tools
bash build.sh converter
cd ..

./tools/convert_path_to_mbin data/path_files
python -m tools.precompute_distance_maps data/map_files
```

Python fallback:

```bash
python -m tools.convert_lacam_path_to_bin data/path_files
python -m tools.precompute_distance_maps data/map_files
```

Expected output:

- training / validation `.mbin` under `data/input_data/...`
- distance maps under `data/distance_maps/...`

#### Step 5: Check required offline directories

Before training, make sure these exist:

- `data/map_files`
- `data/distance_maps`
- `data/input_data`

#### Step 6: Launch offline training

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

```bash
python train.py \
  --dataset_mode online \
  --train_map_path data/map_files \
  --val_dataset_path data/online_eval_input_data \
  --online_total_steps 200000 \
  --online_eval_interval_steps 4000 \
  --online_save_interval_steps 4000 \
  --online_inference_test_interval_steps 4000 \
  --online_time_limit_sec 5 \
  --online_retry_limit 20 \
  --sample_data_path data/online_eval_input_data/maze-32-32-10-1-75/maze-32-32-10-1-75-0-16.mbin \
  --batch_size 64 \
  --num_workers 2
```

Online mode key idea:

- training progress is controlled by optimizer steps, not epochs
- validation / save / inference test are also step-based

### Where To Change Parameters

训练参数统一定义在 `train_args.py`，训练入口在 `train.py`。推荐做法是直接在命令行传参；如果想改默认值，再改 `train_args.py`。

仓库根目录现在提供了两套默认模板：

- [config.offline.yaml](/home/yimin/research/RAILGUN/config.offline.yaml): 离线训练 / 离线验证
- [config.online.yaml](/home/yimin/research/RAILGUN/config.online.yaml): 在线训练 / 离线验证

[config.yaml](/home/yimin/research/RAILGUN/config.yaml) 目前保留为默认离线模板的兼容入口。

训练启动后，`train.py` 会打印一份 `Runtime Config` 表，里面会直接显示当前模型大小、数据模式、训练周期、验证周期和 inference test 配置。

### Training Parameter Table

| Category | Args | Meaning |
| --- | --- | --- |
| Data | `--dataset_mode` | `offline` 读 `.mbin`；`online` 训练时在线生成 |
| Data | `--dataset_path` | 离线训练/验证的 `.mbin` 根目录 |
| Data | `--val_dataset_path` | 固定离线验证集根目录，不填则回退到 `--dataset_path` |
| Data | `--train_map_path` | 在线训练时扫描 `.map` 的根目录 |
| Data | `--num_workers` | PyTorch DataLoader worker 数 |
| Online train | `--online_total_steps` | 在线训练总优化步数 |
| Online train | `--online_eval_interval_steps` | 在线训练按多少步做一次验证 |
| Online train | `--online_save_interval_steps` | 在线训练按多少步保存一次 checkpoint |
| Online train | `--online_inference_test_interval_steps` | 在线训练按多少步做一次 inference test |
| Online train | `--online_time_limit_sec` | 单次 LACAM 在线生成的时间上限 |
| Online train | `--online_retry_limit` | 单个 worker 生成失败时的重试次数 |
| Optimization | `--epochs` | 只用于离线模式 |
| Optimization | `--batch_size` | batch size |
| Optimization | `--learning_rate` | 学习率 |
| Optimization | `--weight_decay` | AdamW 的权重衰减 |
| Training loop | `--eval_interval` | 离线模式下每隔多少个 epoch 做一次验证 loss |
| Training loop | `--save_interval` | 离线模式下每隔多少个 epoch 保存一次 checkpoint |
| Model | `--model` | `unet` 或 `cnn` |
| Model | `--first_layer_channels` | `UNet` 首层通道数，直接影响模型大小 |
| Model | `--feature_dim` | 输入特征维度 |
| Model | `--feature_type` | 特征构造方式，例如 `gradient` |
| Model | `--action_dim` | 动作类别数，默认 5 |
| Inference test | `--sample_data_path` | 训练中固定用于 inference test 的离线 `.mbin` 样本 |
| Inference test | `--inference_num_cases` | 每次 inference test 跑多少个固定样本 |
| Inference test | `--inference_test_interval` | 离线模式下每隔多少个 epoch 跑一次 inference test；`0` 表示复用 `--eval_interval` |
| Inference test | `--inference_action_choice` | inference rollout 用 `sample` 还是 `max` 选动作 |
| Inference test | `--steps` | inference rollout 的最大步数 |

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
```bash
python eval_test.py --model_path model_checkpoint_epoch_4.pth --dataset_paths data/input_data/maze-32-32-60-1-75/maze-32-32-60-1-75-0-16/maze-32-32-60-1-75-0-16-1.bin --show
```
