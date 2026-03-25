# `tools/profile_online_data.py`

## 文件作用

这个脚本用于快速 profile 在线数据链路，主要看两类问题：

- 取 batch 到底慢在哪里
- 原始 scenario 生成本身要花多久

它适合在不启动完整训练的情况下，单独测 online data throughput。

## 主要流程

脚本会：

1. 读取 `--config` 指定的 YAML
2. 扫描 `train_map_path/**/*.map`
3. 构造 [MAPF_online_dataset.py](/home/yimin/research/RAILGUN/MAPF_online_dataset.py) 的 `MAPFOnlineBufferLoader`
4. 先跑若干 warmup batch
5. 统计多个 batch 的 fetch 时间
6. 可选地直接调用底层 dataset 的 scenario 生成逻辑，统计原始 scenario latency

## 主要参数

- `--config`: 在线配置文件，默认 `config.online.yaml`
- `--batches`: 统计多少个 batch
- `--warmup`: 预热多少个 batch
- `--num_workers`: 临时覆盖在线 producer 数
- `--batch_size`: 临时覆盖 batch size
- `--scenario_samples`: 额外直接测多少次 raw scenario 生成；`0` 表示不测

## 输出内容

### batch fetch 部分

输出每个 batch 的获取时间，并汇总：

- `mean`
- `p50`
- `p90`
- `p99`
- `max`

### scenario 生成部分

当 `--scenario_samples > 0` 时，还会输出：

- 每次生成是否成功
- 单次 scenario latency
- 轨迹 step 数统计

## 适用场景

适合排查这类问题：

- 训练端是否在等数据
- `num_workers` 提高后吞吐是否改善
- `online_buffer_size` 是否足够大
- 瓶颈主要在 LACAM 求解还是在 feature construction

## 用法

```bash
python tools/profile_online_data.py --config config.online.yaml

python tools/profile_online_data.py \
  --config config.online.yaml \
  --num_workers 20 \
  --batch_size 64 \
  --batches 50 \
  --warmup 5

python tools/profile_online_data.py \
  --config config.online.yaml \
  --scenario_samples 20
```
