# `train.py`

## 文件作用

这是项目的主训练入口。它负责：

- 解析训练参数
- 初始化单卡或分布式训练环境
- 构建 `UNet` 或 `CNN`
- 扫描 `.mbin` 数据并按地图尺寸分组
- 执行训练、验证、TensorBoard 记录和模型保存

## 主要函数

### `evaluate_valid_loss(model, val_loader, loss_fn, device)`

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
- 进度条显示依赖全局 `args.local_rank`

### `train(args, model, train_loaders, val_loaders, sample_loader, optimizer, loss_fn, device)`

完整训练循环。

主要逻辑：

1. 遍历所有按地图尺寸分组的 `train_loader`
2. 前向计算并用 `mask` 过滤损失
3. 反向传播并更新优化器
4. 按 `eval_interval` 调用验证
5. 按 `save_interval` 保存模型
6. 主进程上调用 `tools.path_formation.path_formation()` 做抽样 rollout 并记录指标

## 主程序入口

`if __name__ == "__main__":` 下的流程如下：

1. 从 [train_args.py](/home/yimin/research/RAILGUN/train_args.py) 读取参数
2. 根据 `--distributed` 决定是否初始化 `torch.distributed`
3. 创建 TensorBoard `SummaryWriter`
4. 固定随机种子
5. 根据 `--model` 初始化 [models/unet.py](/home/yimin/research/RAILGUN/models/unet.py) 或 [models/CNN.py](/home/yimin/research/RAILGUN/models/CNN.py)
6. 扫描 `args.dataset_path` 下全部 `.mbin` 文件
7. 以 `(height, width)` 分组，并跳过小于 `32x32` 的地图
8. 为每组建立训练和验证 `DataLoader`
9. 构建 `sample_loader`
10. 调用 `train(...)`

## 输入输出与接口情况

### 训练数据接口

依赖 [MAPF_dataset_mbin.py](/home/yimin/research/RAILGUN/MAPF_dataset_mbin.py) 返回的样本字典：

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
- 模型权重：`model_checkpoint_epoch_<N>.pth`

## 用法

```bash
# 单卡
python train.py --batch_size 64

# 多卡
torchrun --nproc_per_node=8 train.py --batch_size 8 --distributed
```
