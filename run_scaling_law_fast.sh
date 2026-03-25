#!/bin/bash
# Scaling Law 实验脚本（快速版本 - 使用预加载）

set -e

echo "========================================="
echo "Scaling Law 实验 - 快速版本"
echo "========================================="
echo ""

# 1. 预加载数据到内存缓存
echo "[1/3] 预加载数据到内存缓存..."
echo "这需要约 1.5 分钟..."
time find data/input_data -name "*.mbin" -exec cat {} > /dev/null \;
echo "✓ 预加载完成！"
echo ""

# 2. 检查缓存大小
echo "[2/3] 检查内存缓存..."
free -h | grep -E "Mem|cache"
echo ""

# 3. 运行 scaling law 实验（使用 num_workers=2）
echo "[3/3] 开始训练..."
echo "使用配置: num_workers=2 (预加载模式)"
echo ""

# 临时修改 num_workers
sed -i.bak 's/num_workers: 0/num_workers: 2/' config.offline.yaml

# 运行实验
python scaling_law.py \
  --config config.offline.yaml \
  --python /home/yimintan/anaconda3/envs/py312/bin/python \
  "$@"

# 恢复原配置
mv config.offline.yaml.bak config.offline.yaml

echo ""
echo "========================================="
echo "实验完成！"
echo "========================================="
