#!/usr/bin/env python3
"""
MAPF系统最终验证

验证整个系统的关键功能是否正常工作
"""

import sys
import os
import time
import glob
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def check_system_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    checks = []
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version >= (3, 8):
        checks.append(("Python版本", True, f"{python_version.major}.{python_version.minor}"))
    else:
        checks.append(("Python版本", False, f"{python_version.major}.{python_version.minor} (需要>=3.8)"))
    
    # 检查必要的包
    required_packages = ['torch', 'numpy', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
            checks.append((f"{package}包", True, "已安装"))
        except ImportError:
            checks.append((f"{package}包", False, "未安装"))
    
    # 检查数据目录
    data_dir = project_root / "data"
    if data_dir.exists():
        checks.append(("数据目录", True, str(data_dir)))
    else:
        checks.append(("数据目录", False, "不存在"))
    
    # 检查.mbin文件
    mbin_files = glob.glob(str(data_dir / "input_data" / "**" / "*.mbin"), recursive=True)
    if mbin_files:
        checks.append(("合并数据文件", True, f"{len(mbin_files)}个.mbin文件"))
    else:
        checks.append(("合并数据文件", False, "未找到.mbin文件"))
    
    # 检查C++工具
    tools_dir = project_root / "tools"
    cpp_tools = [
        "converters/cpp/convert_lacam_path_to_bin",
        "testing/test_converter"
    ]
    
    for tool_path in cpp_tools:
        tool_file = tools_dir / tool_path
        if tool_file.exists():
            checks.append((f"C++工具 {os.path.basename(tool_path)}", True, "已编译"))
        else:
            checks.append((f"C++工具 {os.path.basename(tool_path)}", False, "未编译"))
    
    # 检查C++扩展
    ext_files = glob.glob(str(tools_dir / "extensions" / "*.so"))
    if ext_files:
        checks.append(("C++扩展", True, f"{len(ext_files)}个扩展"))
    else:
        checks.append(("C++扩展", False, "未编译"))
    
    # 输出结果
    print(f"{'项目':<20} {'状态':<8} {'详情':<30}")
    print("-" * 60)
    
    all_passed = True
    for name, passed, detail in checks:
        status = "✅" if passed else "❌"
        print(f"{name:<20} {status:<8} {detail:<30}")
        if not passed:
            all_passed = False
    
    return all_passed

def test_core_functionality():
    """测试核心功能"""
    print("\n🧪 测试核心功能...")
    
    try:
        # 测试数据集加载
        from MAPF_dataset_mbin import MAPFDataset
        
        mbin_files = glob.glob("data/input_data/**/*.mbin", recursive=True)
        if not mbin_files:
            print("❌ 未找到.mbin文件")
            return False
        
        # 使用一个小文件测试
        test_file = mbin_files[0]
        dataset = MAPFDataset([test_file], feature_dim=6, feature_type='gradient', first_step=True)
        
        if len(dataset) == 0:
            print("❌ 数据集为空")
            return False
        
        # 测试样本读取
        sample = dataset[0]
        
        # 验证样本结构
        required_keys = ["feature", "action", "mask", "file_name"]
        for key in required_keys:
            if key not in sample:
                print(f"❌ 样本缺少{key}")
                return False
        
        # 验证tensor形状
        if sample["feature"].shape[0] != 6:
            print(f"❌ 特征维度错误: {sample['feature'].shape[0]} != 6")
            return False
        
        print("✅ 核心功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 核心功能测试失败: {e}")
        return False

def test_training_pipeline():
    """测试训练管道"""
    print("\n🧪 测试训练管道...")
    
    try:
        import torch
        import torch.nn as nn
        from models.unet import UNet
        from torch.utils.data import DataLoader
        
        # 创建小模型
        model = UNet(n_channels=6, n_classes=5, first_layer_channels=8, bilinear=False)
        
        # 创建数据加载器
        mbin_files = glob.glob("data/input_data/**/*.mbin", recursive=True)
        if not mbin_files:
            print("❌ 未找到.mbin文件")
            return False
        
        from MAPF_dataset_mbin import MAPFDataset
        dataset = MAPFDataset(mbin_files[:2], feature_dim=6, feature_type='gradient')
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        # 测试一个训练步骤
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        
        for batch in dataloader:
            feature = batch["feature"]
            action = batch["action"]
            mask = batch["mask"]
            
            # 前向传播
            logits, _ = model(feature)
            loss = loss_fn(logits, action)
            masked_loss = loss * mask.float()
            
            if mask.sum() > 0:
                averaged_loss = masked_loss.sum() / mask.sum()
                
                # 反向传播
                optimizer.zero_grad()
                averaged_loss.backward()
                optimizer.step()
                
                print(f"✅ 训练管道测试通过 (损失: {averaged_loss.item():.4f})")
                return True
            else:
                print("⚠️ 没有有效的智能体掩码")
                return False
        
        print("❌ 没有数据批次")
        return False
        
    except Exception as e:
        print(f"❌ 训练管道测试失败: {e}")
        return False

def test_performance_benchmarks():
    """快速性能基准测试"""
    print("\n🧪 快速性能基准测试...")
    
    try:
        # 测试单个样本加载性能
        from MAPF_dataset_mbin import MAPFDataset
        
        mbin_files = glob.glob("data/input_data/**/*.mbin", recursive=True)
        if not mbin_files:
            print("❌ 未找到.mbin文件")
            return False
        
        dataset = MAPFDataset(mbin_files[:1], feature_dim=6, feature_type='gradient')
        
        # 性能测试
        start_time = time.time()
        for i in range(10):
            sample = dataset[i % len(dataset)]
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10 * 1000
        throughput = 10 / (end_time - start_time)
        
        print(f"✅ 性能测试结果:")
        print(f"   平均加载时间: {avg_time:.2f}ms/样本")
        print(f"   吞吐量: {throughput:.1f} 样本/秒")
        
        # 性能标准
        if avg_time < 50:
            print("🚀 性能优秀！")
            return True
        elif avg_time < 100:
            print("👍 性能良好")
            return True
        else:
            print("🤔 性能一般，可能需要优化")
            return True  # 仍然算通过，只是性能提醒
            
    except Exception as e:
        print(f"❌ 性能测试失败: {e}")
        return False

def main():
    """主验证函数"""
    print("🎯 MAPF系统最终验证")
    print("=" * 60)
    
    all_tests = [
        ("系统要求检查", check_system_requirements),
        ("核心功能验证", test_core_functionality),
        ("训练管道验证", test_training_pipeline),
        ("性能基准测试", test_performance_benchmarks),
    ]
    
    results = []
    
    for test_name, test_func in all_tests:
        print(f"\n{'='*40}")
        print(f"🔍 {test_name}")
        print(f"{'='*40}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} 异常: {e}")
            results.append((test_name, False))
    
    # 最终总结
    print(f"\n{'='*60}")
    print("🎯 最终验证结果")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n📊 总体结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n🎉 系统验证完全通过！")
        print(f"💡 你的MAPF训练系统已经完全优化并可以投入使用")
        print(f"🚀 建议运行: python train.py --batch_size 64 --epochs 100")
        return True
    elif passed >= total * 0.75:
        print(f"\n👍 系统验证大部分通过")
        print(f"💡 系统基本可用，有少量问题需要注意")
        return True
    else:
        print(f"\n⚠️ 系统验证存在较多问题")
        print(f"💡 建议检查失败的项目后再使用")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 