#!/usr/bin/env python3
"""
MAPF工具集综合测试脚本

运行所有测试来验证系统的正确性和性能
"""

import sys
import os
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def run_test_suite(test_name, test_function):
    """运行测试套件并记录结果"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        success = test_function()
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            print(f"✅ {test_name} 通过 (耗时: {duration:.2f}秒)")
            return True, duration
        else:
            print(f"❌ {test_name} 失败 (耗时: {duration:.2f}秒)")
            return False, duration
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"💥 {test_name} 异常: {e} (耗时: {duration:.2f}秒)")
        return False, duration

def test_dataloader_correctness():
    """运行数据加载器正确性测试"""
    try:
        from test_dataloader_correctness import run_dataloader_tests
        return run_dataloader_tests()
    except ImportError as e:
        print(f"❌ 无法导入数据加载器测试: {e}")
        return False

def test_training_correctness():
    """运行训练正确性测试"""
    try:
        from test_training_correctness import run_training_tests
        return run_training_tests()
    except ImportError as e:
        print(f"❌ 无法导入训练测试: {e}")
        return False

def test_cpp_extensions():
    """测试C++扩展"""
    try:
        from cpp_feature_benchmark import run_comprehensive_benchmark
        print("运行C++扩展性能测试...")
        run_comprehensive_benchmark()
        return True
    except ImportError as e:
        print(f"❌ 无法导入C++扩展测试: {e}")
        return False
    except Exception as e:
        print(f"❌ C++扩展测试失败: {e}")
        return False

def test_data_conversion():
    """测试数据转换工具"""
    try:
        from test_integration import main as run_integration_tests
        return run_integration_tests()
    except ImportError as e:
        print(f"❌ 无法导入集成测试: {e}")
        return False

def test_quick_performance():
    """快速性能测试"""
    try:
        from quick_training_test import main as run_quick_test
        return run_quick_test()
    except ImportError as e:
        print(f"❌ 无法导入快速测试: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 MAPF工具集综合测试")
    print("=" * 80)
    print("这将运行所有测试来验证系统的正确性和性能")
    
    # 测试列表
    test_suites = [
        ("数据加载器正确性测试", test_dataloader_correctness),
        ("训练函数正确性测试", test_training_correctness),
        ("C++扩展性能测试", test_cpp_extensions),
        ("数据转换集成测试", test_data_conversion),
        ("快速性能测试", test_quick_performance),
    ]
    
    results = []
    total_time = 0
    
    # 运行所有测试
    for test_name, test_func in test_suites:
        success, duration = run_test_suite(test_name, test_func)
        results.append((test_name, success, duration))
        total_time += duration
    
    # 输出总结
    print(f"\n{'='*80}")
    print("🎯 测试总结")
    print(f"{'='*80}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"总测试套件: {total}")
    print(f"通过: {passed}")
    print(f"失败: {total - passed}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"通过率: {passed/total*100:.1f}%")
    
    print(f"\n📊 详细结果:")
    for test_name, success, duration in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name:<30} {duration:>8.2f}s")
    
    # 最终评估
    if passed == total:
        print(f"\n🎉 所有测试通过！系统工作正常，可以安全使用")
        return True
    elif passed >= total * 0.8:
        print(f"\n👍 大部分测试通过 ({passed}/{total})，系统基本可用")
        return True
    else:
        print(f"\n⚠️ 多个测试失败 ({passed}/{total})，需要检查和修复")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 测试运行异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 