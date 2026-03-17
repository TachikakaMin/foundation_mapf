# `train_args.py`

## 文件作用

这个文件集中管理训练脚本的命令行参数，供 [train.py](/home/yimin/research/RAILGUN/train.py) 调用。

## 主要函数

### `get_args()`

解析训练参数并返回 `argparse.Namespace`。

参数分为四类：

- 日志与随机种子：
  - `--seed`
  - `--log_dir`
- 数据集：
  - `--dataset_path`
  - `--sample_data_path`
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
- 当前没有额外的参数校验逻辑，默认值就是脚本给出的训练配置

## 用法

```python
from train_args import get_args

args = get_args()
print(args.batch_size)
```

命令行通常不单独调用，而是通过训练入口使用：

```bash
python train.py --epochs 50 --batch_size 32 --model unet
```
