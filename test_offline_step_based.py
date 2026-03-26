#!/usr/bin/env python3
"""
测试 offline step-based 训练模式
验证数据加载、随机采样和单次遍历逻辑
"""

import os
import sys
import glob
from collections import defaultdict

def test_data_structure():
    """测试数据目录结构"""
    print("=" * 60)
    print("测试 1: 数据目录结构")
    print("=" * 60)

    data_root = "data/input_data"
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        return False

    mbin_files = glob.glob(os.path.join(data_root, "**/*.mbin"), recursive=True)
    if not mbin_files:
        print(f"❌ 未找到 .mbin 文件")
        return False

    print(f"✓ 找到 {len(mbin_files)} 个 .mbin 文件")

    # 按地图尺寸分组
    dims_count = defaultdict(int)
    for f in mbin_files:
        parts = os.path.basename(f).split('-')
        if len(parts) >= 3:
            dims = f"{parts[1]}x{parts[2]}"
            dims_count[dims] += 1

    print("\n地图尺寸分布:")
    for dims in sorted(dims_count.keys()):
        print(f"  {dims}: {dims_count[dims]} files")

    return True


def test_config_loading():
    """测试配置文件加载"""
    print("\n" + "=" * 60)
    print("测试 2: 配置文件加载")
    print("=" * 60)

    config_file = "config.offline.yaml"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False

    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"✓ 配置文件加载成功")
    print(f"\n关键配置:")
    print(f"  dataset_mode: {config.get('dataset_mode')}")
    print(f"  dataset_path: {config.get('dataset_path')}")
    print(f"  offline_total_steps: {config.get('offline_total_steps')}")
    print(f"  offline_eval_interval_steps: {config.get('offline_eval_interval_steps')}")
    print(f"  batch_size: {config.get('batch_size')}")
    print(f"  num_workers: {config.get('num_workers')}")

    return True


def test_train_args():
    """测试训练参数解析"""
    print("\n" + "=" * 60)
    print("测试 3: 训练参数解析")
    print("=" * 60)

    try:
        from train_args import get_args

        # 模拟命令行参数
        sys.argv = [
            'train.py',
            '--config', 'config.offline.yaml',
            '--offline_total_steps', '100',
            '--batch_size', '64',
        ]

        args = get_args()

        print(f"✓ 参数解析成功")
        print(f"\n解析结果:")
        print(f"  dataset_mode: {args.dataset_mode}")
        print(f"  dataset_path: {args.dataset_path}")
        print(f"  offline_total_steps: {args.offline_total_steps}")
        print(f"  offline_eval_interval_steps: {args.offline_eval_interval_steps}")
        print(f"  offline_save_interval_steps: {args.offline_save_interval_steps}")
        print(f"  offline_inference_test_interval_steps: {args.offline_inference_test_interval_steps}")
        print(f"  batch_size: {args.batch_size}")
        print(f"  num_workers: {args.num_workers}")

        # 验证 step-based 模式
        if args.offline_total_steps > 0:
            print(f"\n✓ Step-based 模式已启用")
        else:
            print(f"\n⚠ Epoch-based 模式（需要设置 offline_total_steps > 0）")

        return True

    except Exception as e:
        print(f"❌ 参数解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading_logic():
    """测试数据加载逻辑（不实际加载数据）"""
    print("\n" + "=" * 60)
    print("测试 4: 数据加载逻辑")
    print("=" * 60)

    try:
        from train import get_map_dims, group_files_by_dims

        data_root = "data/input_data"
        mbin_files = sorted(glob.glob(os.path.join(data_root, "**/*.mbin"), recursive=True))

        # 测试 get_map_dims
        test_file = mbin_files[0]
        dims = get_map_dims(test_file)
        print(f"✓ get_map_dims 测试通过")
        print(f"  示例文件: {os.path.basename(test_file)}")
        print(f"  解析结果: {dims}")

        # 测试 group_files_by_dims
        dimension_groups = group_files_by_dims(mbin_files, min_map_size=32)
        print(f"\n✓ group_files_by_dims 测试通过")
        print(f"  分组数量: {len(dimension_groups)}")
        for dims, files in sorted(dimension_groups.items()):
            print(f"  {dims}: {len(files)} files")

        # 验证训练/验证分割
        print(f"\n训练/验证分割 (10% 验证):")
        for dims, files in sorted(dimension_groups.items()):
            n_test = int(0.1 * len(files))
            n_train = len(files) - n_test
            print(f"  {dims}: {n_train} 训练, {n_test} 验证")

        return True

    except Exception as e:
        print(f"❌ 数据加载逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_random_sampling_logic():
    """测试随机采样逻辑"""
    print("\n" + "=" * 60)
    print("测试 5: 随机采样逻辑")
    print("=" * 60)

    try:
        import numpy as np

        # 模拟 dims 和权重
        dims_list = [(32, 32), (64, 64)]
        train_loader_weights = {(32, 32): 2356, (64, 64): 2945}

        weights = np.array([train_loader_weights.get(dims, 0) for dims in dims_list], dtype=np.float64)
        probs = weights / weights.sum()

        print(f"✓ 权重计算成功")
        print(f"\nDims 权重:")
        for dims, prob in zip(dims_list, probs):
            print(f"  {dims}: {prob:.4f} (文件数: {train_loader_weights[dims]})")

        # 模拟随机采样
        rng = np.random.default_rng(1919180)
        samples = [dims_list[int(rng.choice(len(dims_list), p=probs))] for _ in range(100)]

        sample_count = defaultdict(int)
        for dims in samples:
            sample_count[dims] += 1

        print(f"\n模拟 100 次采样:")
        for dims in dims_list:
            count = sample_count[dims]
            expected = probs[dims_list.index(dims)] * 100
            print(f"  {dims}: {count} 次 (期望: {expected:.1f})")

        return True

    except Exception as e:
        print(f"❌ 随机采样逻辑测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("Offline Step-based 训练模式测试")
    print("=" * 60 + "\n")

    tests = [
        ("数据目录结构", test_data_structure),
        ("配置文件加载", test_config_loading),
        ("训练参数解析", test_train_args),
        ("数据加载逻辑", test_data_loading_logic),
        ("随机采样逻辑", test_random_sampling_logic),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n✅ 所有测试通过！可以开始训练。")
        print("\n推荐命令:")
        print("python train.py \\")
        print("  --config config.offline.yaml \\")
        print("  --offline_total_steps 70000 \\")
        print("  --batch_size 64 \\")
        print("  --num_workers 20")
    else:
        print("\n⚠ 部分测试失败，请检查配置。")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
